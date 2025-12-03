# This code was originally in simul_whisper/transcriber/simul_whisper.py . It is adapted a lot for SimulStreaming.

import os
import logging

import torch
import torch.nn.functional as F

from .whisper import load_model, DecodingOptions, tokenizer
from .config import AlignAttConfig
from .whisper.audio import log_mel_spectrogram, TOKENS_PER_SECOND, pad_or_trim, N_SAMPLES, N_FRAMES
from .whisper.timing import median_filter
from .whisper.decoding import GreedyDecoder, BeamSearchDecoder, SuppressTokens, detect_language
from .beam import BeamPyTorchInference
from .eow_detection import fire_at_boundary, load_cif
import os

from token_buffer import TokenBuffer

import numpy as np
from .generation_progress import *

# Optional HF assistant model wrapper (for distil-whisper etc.)
try:
    from transformers import AutoModelForSpeechSeq2Seq
    from transformers.modeling_outputs import BaseModelOutput
except ImportError:  # keep runtime flexible when HF is not installed
    AutoModelForSpeechSeq2Seq = None
    BaseModelOutput = None

DEC_PAD = 50257
logger = logging.getLogger(__name__)

import sys
import wave

# New features added to the original version of Simul-Whisper:
# - large-v3 model support
# - translation support
# - beam search
# - prompt -- static vs. non-static
# - context

# Rule-based sentence breaking: punctuation and conjunctions
SENTENCE_END_PUNCTUATION = {'.', '!', '?', '。', '!', '?'}  # Period, exclamation, question marks (EN/KO)
SENTENCE_BREAK_PUNCTUATION = {',', ';', ':', '、', ',', ';', ':'}  # Comma, semicolon, colon (EN/KO)
# Conjunctions that signal a good breaking point (break BEFORE the conjunction)
CONJUNCTIONS_EN = {'but', 'so', 'yet', 'nor', 'however', 'therefore', 'moreover', 'furthermore', 'meanwhile', 'otherwise', 'nevertheless'}
CONJUNCTIONS_KO = {'그리고', '하지만', '그러나', '그래서', '따라서', '그러므로', '또한', '게다가', '그런데', '한편', '그렇지만', '그럼에도'}
ALL_CONJUNCTIONS = CONJUNCTIONS_EN | CONJUNCTIONS_KO

class PaddedAlignAttWhisper:
    def __init__(self, cfg: AlignAttConfig) -> None:
        self.logdir_i = 0
        self.log_segments = 0
        if cfg.logdir is not None and not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        model_name = os.path.basename(cfg.model_path).replace(".pt", "")
        model_path = os.path.dirname(os.path.abspath(cfg.model_path))

        # Use full path if file exists to avoid downloading
        if os.path.isfile(cfg.model_path):
            self.model = load_model(name=cfg.model_path)
        else:
            self.model = load_model(name=model_name, download_root=model_path)

        logger.info(f"Model dimensions: {self.model.dims}")

        # Load assistant model for speculative decoding if enabled
        self.assistant_model = None
        if cfg.use_speculative_decoding and cfg.assistant_model_path:
            self.assistant_model = self._load_assistant_model(cfg.assistant_model_path)

        self.decode_options = DecodingOptions(
            language = cfg.language, 
            without_timestamps = True,
            task=cfg.task
        )
        self.tokenizer_is_multilingual = not model_name.endswith(".en")
        self.create_tokenizer(cfg.language if cfg.language != "auto" else None)
        self.detected_language = cfg.language if cfg.language != "auto" else None
        self.detected_language_probs = None

        self.max_text_len = self.model.dims.n_text_ctx
        self.num_decoder_layers = len(self.model.decoder.blocks)
        self.cfg = cfg

        # model to detect end-of-word boundary at the end of the segment
        self.CIFLinear, self.always_fire, self.never_fire = load_cif(cfg,
                                                                     n_audio_state=self.model.dims.n_audio_state,
                                                                     device=self.model.device)

        # install hooks to access encoder-decoder attention
        self.dec_attns = []
        def layer_hook(module, net_input, net_output):
            # net_output[1]: B*num_head*token_len*audio_len
            t = F.softmax(net_output[1], dim=-1)
            self.dec_attns.append(t.squeeze(0))
        for b in self.model.decoder.blocks:
            b.cross_attn.register_forward_hook(layer_hook)
        
        self.kv_cache = {}
        def kv_hook(module: torch.nn.Linear, _, net_output: torch.Tensor):
            if module.cache_id not in self.kv_cache or net_output.shape[1] > self.max_text_len:
                # save as-is, for the first token or cross attention
                self.kv_cache[module.cache_id] = net_output
            else:
                x = self.kv_cache[module.cache_id]
                self.kv_cache[module.cache_id] = torch.cat([x, net_output], dim=1).detach()
            return self.kv_cache[module.cache_id] 

        for i,b in enumerate(self.model.decoder.blocks):
            b.attn.key.register_forward_hook(kv_hook)
            b.attn.value.register_forward_hook(kv_hook)
            b.cross_attn.key.register_forward_hook(kv_hook)
            b.cross_attn.value.register_forward_hook(kv_hook)

        self.align_source = {}
        self.num_align_heads = 0
        for layer_rank, head_id in self.model.alignment_heads.indices().T:
            layer_rank = layer_rank.item()
            heads = self.align_source.get(layer_rank, [])
            heads.append((self.num_align_heads, head_id.item()))
            self.align_source[layer_rank] = heads
            self.num_align_heads += 1


        # tokens to be suppressed from decoding, to prevent hallucinations
        suppress_tokens = [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
                # self.tokenizer.eot 
                self.tokenizer.no_timestamps,  # added by DM
            ] + list(self.tokenizer.all_language_tokens)  # added by DM
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        suppress_tokens =  tuple(sorted(set(suppress_tokens)))
        logger.debug(f"Suppress tokens: {suppress_tokens}")
        sup_tokens = SuppressTokens(suppress_tokens)
        self.suppress_tokens = lambda logits: sup_tokens.apply(logits, None)
        # blank tokens are suppresed for new segments near the line 334

        # it's going to be regenerated after lang id
        self.segments = []
        self.init_tokens()
        
        self.last_attend_frame = -self.cfg.rewind_threshold

        if self.cfg.max_context_tokens is None:
            self.max_context_tokens = self.max_text_len
        else:
            self.max_context_tokens = self.cfg.max_context_tokens
        self.init_context()

        # decoder type: greedy or beam
        if cfg.decoder_type == "greedy":
            # Use speculative decoder if assistant model is available
            if self.assistant_model is not None and cfg.use_speculative_decoding:
                from .speculative_decoder import SpeculativeGreedyDecoder
                logger.info(f"Using speculative greedy decoder with {cfg.num_assistant_tokens} lookahead tokens")
                self.token_decoder = SpeculativeGreedyDecoder(
                    temperature=0.0,
                    eot=self.tokenizer.eot,
                    assistant_model=self.assistant_model,
                    num_assistant_tokens=cfg.num_assistant_tokens,
                    use_speculative=True
                )
            else:
                logger.info("Using standard greedy decoder")
                self.token_decoder = GreedyDecoder(0.0, self.tokenizer.eot)
            self.decoder_type = "greedy"

        elif cfg.decoder_type == "beam":
            self.decoder_type = "beam"
            if cfg.use_speculative_decoding:
                logger.warning("Speculative decoding is not supported with beam search. Using standard beam search.")
            self.inference = BeamPyTorchInference(self.model, self.initial_token_length)
            self.inference.kv_cache = self.kv_cache

            self.token_decoder = BeamSearchDecoder(inference=self.inference, eot=self.tokenizer.eot, beam_size=cfg.beam_size)

    def _load_assistant_model(self, path: str):
        """
        Load assistant model.
        - If path points to an existing .pt, load with local load_model (OpenAI checkpoint format).
        - Otherwise, try Hugging Face distil-whisper (HF Hub ID or local directory).
        """
        # 1) Local .pt (OpenAI-format) path
        if os.path.isfile(path):
            try:
                logger.info(f"Loading assistant model (.pt) for speculative decoding: {path}")
                assistant_name = os.path.basename(path).replace(".pt", "")
                assistant_path = os.path.dirname(os.path.abspath(path))
                model = load_model(name=assistant_name, download_root=assistant_path)
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Failed to load assistant .pt model '{path}': {e}")
                return None

        # 2) Hugging Face model id/directory (e.g., distil-whisper/distil-large-v2)
        if AutoModelForSpeechSeq2Seq is None or BaseModelOutput is None:
            logger.warning("transformers is not installed, cannot load HF assistant model.")
            return None

        try:
            hf_id = path
            logger.info(f"Loading assistant model from Hugging Face: {hf_id}")

            # Match dtype/device to main model
            dtype = next(self.model.parameters()).dtype
            device = next(self.model.parameters()).device

            hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            hf_model.eval()

            class HFWhisperAssistantWrapper(torch.nn.Module):
                """Adapter to expose HF Whisper logits(tokens, encoder_features) API."""

                def __init__(self, model_ref):
                    super().__init__()
                    self.model_ref = model_ref

                def logits(self, tokens: torch.Tensor, encoder_features: torch.Tensor) -> torch.Tensor:
                    # Ensure correct dtype/device
                    if encoder_features.dtype != self.model_ref.dtype:
                        encoder_features = encoder_features.to(dtype=self.model_ref.dtype)
                    if encoder_features.device != self.model_ref.device:
                        encoder_features = encoder_features.to(self.model_ref.device)
                    if tokens.device != self.model_ref.device:
                        tokens = tokens.to(self.model_ref.device)

                    outputs = self.model_ref(
                        encoder_outputs=BaseModelOutput(last_hidden_state=encoder_features),
                        decoder_input_ids=tokens,
                        use_cache=False,
                    )
                    return outputs.logits

            logger.info("Assistant model loaded successfully from Hugging Face for speculative decoding")
            return HFWhisperAssistantWrapper(hf_model)

        except Exception as e:
            logger.warning(f"Failed to load assistant model from HF ({path}): {e}")
            return None

    def create_tokenizer(self, language=None):
        self.tokenizer = tokenizer.get_tokenizer(
            multilingual=self.tokenizer_is_multilingual,  
            language=language,
            num_languages=self.model.num_languages,
            task=self.decode_options.task
        )

    def init_context(self):
        kw = {'tokenizer': self.tokenizer, 
              'device': self.model.device, 
              'prefix_token_ids': [self.tokenizer.sot_prev]}
        self.context = TokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.context = TokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.context.text += self.cfg.init_prompt

    def init_tokens(self):
        logger.debug(f"init tokens, {len(self.segments)}")
        # init tokens (mandatory prompt)
        self.initial_tokens = torch.tensor(
            self.tokenizer.sot_sequence_including_notimestamps, 
            dtype=torch.long, 
            device=self.model.device).unsqueeze(0)
        self.initial_token_length = self.initial_tokens.shape[1]
        self.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)
#        self.segments = []
        logger.debug(f"init tokens after, {len(self.segments)}")
        self.tokens = [self.initial_tokens]

    def trim_context(self):
        logger.info("Trimming context")
        c = len(self.context.as_token_ids()) - len(self.context.prefix_token_ids)
#        logger.debug(f"c= {len(self.context.as_token_ids())}, {len(self.context.prefix_token_ids)}")
        logger.info(f"Context text: {self.context.as_text()}")
#        logger.debug(f"Context tensor: {self.context.as_tensor()}")
        l = sum(t.shape[1] for t in self.tokens) + c
#        logger.debug(f"len {l}, c {c}, max_context_tokens {self.max_context_tokens}")
        if self.cfg.static_init_prompt is None:
            after = 0
        else:
            after = len(self.cfg.static_init_prompt)
#        logger.debug(f"len {l}, c {c}, max_context_tokens {self.max_context_tokens}")
        while c > self.max_context_tokens or l > self.max_text_len - 20:
            t = self.context.trim_words(after=after)
            l -= t
            c -= t
            logger.debug(f"len {l}, c {c}, max_context_tokens {self.max_context_tokens}")
            if t == 0:
                break
#        logger.debug(f"len {l}, c {c}, max_context_tokens {self.max_context_tokens}")
        logger.info(f"Context after trim: {self.context.text} (len: {l})")


    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        if self.cfg.decoder_type == "greedy":
            logit = self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)
        else:
            logger.debug(f"Logits shape: {tokens.shape}")
            logit = self.inference.logits(tokens, audio_features)
        return logit
    

    def refresh_segment(self, complete=False):

        logger.debug("Refreshing segment:")
        self.init_tokens()
        self.last_attend_frame = -self.cfg.rewind_threshold
        # Only reset detected_language if in auto mode
        # When language is forced (not "auto"), keep the detected_language to prevent re-detection
        if self.cfg.language == "auto":
            self.detected_language = None
            self.detected_language_probs = None
        self.init_context()
        logger.debug(f"Context: {self.context}")
        if not complete and len(self.segments) > 2:
            logger.debug("keeping last two segments because they are and it is not complete.")
            self.segments = self.segments[-2:]
        else:
            logger.debug("removing all segments.")
            self.segments = []
        self.log_segments += 1

        # Clear KV cache to prevent tensor size mismatch on language change
        self._clean_cache()


    def fire_at_boundary(self, chunked_encoder_feature: torch.Tensor):
        if self.always_fire: return True
        if self.never_fire: return False
        return fire_at_boundary(chunked_encoder_feature, self.CIFLinear)

    def should_break_at_punctuation(self, tokens_list):
        """
        Check if we should break the sentence based on punctuation or conjunctions.
        Returns (should_break, break_position)
        - should_break: True if we should stop decoding
        - break_position: number of tokens to keep (from the end, negative index)
        """
        if len(tokens_list) == 0:
            return False, 0

        # Decode tokens to text
        text = self.tokenizer.decode(tokens_list)
        logger.debug(f"[Rule-based] Checking text: '{text}'")

        # Check for sentence-ending punctuation
        for punct in SENTENCE_END_PUNCTUATION:
            if punct in text:
                logger.info(f"[Rule-based break] Found sentence-ending punctuation: '{punct}' in text: '{text}'")
                return True, 0  # Keep all tokens including punctuation

        # Check for sentence-break punctuation (commas, semicolons)
        for punct in SENTENCE_BREAK_PUNCTUATION:
            if text.endswith(punct) or text.endswith(punct + ' '):
                logger.info(f"[Rule-based break] Found sentence-break punctuation: '{punct}'")
                return True, 0  # Keep all tokens including punctuation

        # Check for conjunctions - break BEFORE the conjunction
        words = text.strip().split()
        if len(words) > 0:
            # Check if the last word (or last few words) is a conjunction
            last_word = words[-1].strip('.,;:!?').lower()
            if last_word in ALL_CONJUNCTIONS:
                logger.info(f"[Rule-based break] Found conjunction at end: '{last_word}' - breaking BEFORE it")
                # We need to find how many tokens to exclude (the conjunction tokens)
                # Decode all tokens except the last few to find the split point
                for i in range(1, min(5, len(tokens_list)) + 1):  # Check last 5 tokens max
                    text_without_last = self.tokenizer.decode(tokens_list[:-i])
                    if last_word not in text_without_last.lower():
                        # Found the split point - exclude last i tokens
                        logger.info(f"[Rule-based break] Excluding last {i} tokens (conjunction)")
                        return True, -i

        return False, 0


    def _current_tokens(self):

        toks = self.tokens
        # very first infer: duplicate start of seq to beam_size
        if toks[0].shape[0] == 1:
            toks[0] = toks[0].repeat_interleave(self.cfg.beam_size,dim=0)

        if not self.context.is_empty():
            context_toks = self.context.as_tensor_beam(self.cfg.beam_size, device=self.model.device)
            toks = [context_toks] + toks

        # make it one tensor
        if len(toks) > 1:
            current_tokens = torch.cat(toks, dim=1)
        else:
            current_tokens = toks[0]
        logger.debug("debug print current_tokens:")
        self.debug_print_tokens(current_tokens)
        return current_tokens


    def debug_print_tokens(self, tokens):
        for i in range(self.cfg.beam_size):
            logger.debug(self.tokenizer.decode_with_timestamps(tokens[i].tolist()))

    ### audio buffer 

    def segments_len(self):
        segments_len = sum(s.shape[0] for s in self.segments) / 16000
        return segments_len

    def _apply_minseglen(self):
        segments_len = self.segments_len()
        # wait for long enough audio to start
        if segments_len < self.cfg.audio_min_len: 
            logger.debug("waiting for next segment")
            return False
        return True

    def insert_audio(self, segment=None):
        if segment is not None:
            self.segments.append(segment)

        removed_len = 0
        # len of audio is bigger than buffer_len. Going to remove the first segment
        segments_len = self.segments_len()
        while len(self.segments) > 1 and segments_len > self.cfg.audio_max_len:
            removed_len = self.segments[0].shape[0] / 16000
            segments_len -= removed_len
            self.last_attend_frame -= int(TOKENS_PER_SECOND*removed_len)
            self.segments = self.segments[1:]
            logger.debug(f"remove segments: {len(self.segments)} {len(self.tokens)}")
            if len(self.tokens) > 1:
                self.context.append_token_ids(self.tokens[1][0,:])
                self.tokens = [self.initial_tokens] + self.tokens[2:]
        return removed_len

    def _clean_cache(self):
        '''clean the cache that stores the attention matrices and kv_cache.
        It must be called every time after generation with the model.'''
        # cleaning cache
        self.dec_attns = []
        self.kv_cache = {}
        if self.decoder_type == "beam":
            self.inference.kv_cache = self.kv_cache
            self.token_decoder.reset()

    @torch.no_grad()
    def lang_id(self, encoder_features):
        """Language detection from encoder features.
        This code is trimmed and copy-pasted from whisper.decoding.detect_language .
        """
    
        # forward pass using a single token, startoftranscript
        n_audio = encoder_features.shape[0]
        x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(self.model.device)  # [n_audio, 1]
        logits = self.model.logits(x, encoder_features)[:, 0]

        # collect detected languages; suppress all non-language tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(self.tokenizer.all_language_tokens)] = False
        logits[:, mask] = -np.inf
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()
        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(self.tokenizer.all_language_tokens, self.tokenizer.all_language_codes)
            }
            for i in range(n_audio)
        ]

        single = encoder_features.ndim == 2
        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]

        self._clean_cache()
        return language_tokens, language_probs

    ### transcription / translation

    @torch.no_grad()
    def infer(self, is_last=False):
        new_segment = True
        if len(self.segments) == 0:
            logger.debug("No segments, nothing to do")
            self.logdir_save([], [], {})
            return [], {}
        if not self._apply_minseglen():
            logger.debug(f"applied minseglen {self.cfg.audio_min_len} > {self.segments_len()}.")
            input_segments = torch.cat(self.segments, dim=0)
            self.logdir_save(input_segments, [], {})
            return [], {}

        # input_segments is concatenation of audio, it's one array
        if len(self.segments) > 1:
            input_segments = torch.cat(self.segments, dim=0)
        else:
            input_segments = self.segments[0]


        
        # mel + padding to 30s
        mel_padded = log_mel_spectrogram(input_segments, n_mels=self.model.dims.n_mels, padding=N_SAMPLES, 
                                            device=self.model.device).unsqueeze(0)
        # trim to 3000
        mel = pad_or_trim(mel_padded, N_FRAMES)

        # the len of actual audio
        content_mel_len = int((mel_padded.shape[2] - mel.shape[2])/2)

        # encode
        encoder_feature = self.model.encoder(mel)

#        logger.debug(f"Encoder feature shape: {encoder_feature.shape}")
#        if mel.shape[-2:] != (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
#            logger.debug("mel ")
        if self.cfg.language == "auto" and self.detected_language is None:
            language_tokens, language_probs = self.lang_id(encoder_feature)
            logger.debug(f"Language tokens: {language_tokens}, probs: {language_probs}")
            top_lan, p = max(language_probs[0].items(), key=lambda x: x[1])
            logger.info(f"Detected language: {top_lan} with p={p:.4f}")

            # Only EN and KO are supported. Map other languages to EN/KO based on probabilities
            if top_lan not in ['en', 'ko']:
                logger.warning(f"Detected unsupported language '{top_lan}'. System only supports EN/KO. Selecting based on EN/KO probabilities.")

                # Compare only EN and KO probabilities
                lang_probs_dict = language_probs[0] if isinstance(language_probs, list) else language_probs
                en_prob = lang_probs_dict.get('en', 0.0)
                ko_prob = lang_probs_dict.get('ko', 0.0)

                logger.info(f"EN probability: {en_prob:.6f}, KO probability: {ko_prob:.6f}")

                # Select the higher probability between EN and KO
                if ko_prob > en_prob:
                    top_lan = 'ko'
                    logger.info(f"Selected Korean ({ko_prob:.6f}) > English ({en_prob:.6f})")
                else:
                    top_lan = 'en'
                    logger.info(f"Selected English ({en_prob:.6f}) >= Korean ({ko_prob:.6f})")

            self.create_tokenizer(top_lan)
            self.detected_language = top_lan
            self.detected_language_probs = language_probs[0] if isinstance(language_probs, list) else language_probs
            self.init_tokens()
            logger.info(f"Tokenizer language: {self.tokenizer.language}, {self.tokenizer.sot_sequence_including_notimestamps}")

        self.trim_context()
        current_tokens = self._current_tokens()
#        
        fire_detected = self.fire_at_boundary(encoder_feature[:, :content_mel_len, :])


        ####################### Decoding loop
        logger.info("Decoding loop starts\n")

        sum_logprobs = torch.zeros(self.cfg.beam_size, device=mel.device)
        completed = False

        attn_of_alignment_heads = None
        most_attended_frame = None

        token_len_before_decoding = current_tokens.shape[1]

        # Hallucination detection variables
        recent_tokens = []
        consecutive_token_count = 1
        last_token = None

        generation_progress = []
        generation = {
            "starting_tokens": BeamTokens(current_tokens[0,:].clone(), self.cfg.beam_size),
            "token_len_before_decoding": token_len_before_decoding,
            "fire_detected": fire_detected,  # CIF model boundary detection
            "frames_len": content_mel_len,
            "frames_threshold": 4 if is_last else self.cfg.frame_threshold,

            # to be filled later
            "logits_starting": None,

            # to be filled later
            "no_speech_prob": None,
            "no_speech": False,

            # language detection info
            "language": self.detected_language,
            "language_probs": self.detected_language_probs if self.detected_language_probs is not None else {},

            # to be filled in the loop
            "progress": generation_progress,

            # Rule-based breaking flag
            "rule_based_break": False,
        }
        while not completed and current_tokens.shape[1] < self.max_text_len: # bos is 3 tokens
            generation_progress_loop = []

            if new_segment:
                tokens_for_logits = current_tokens
            else:
                # only need to use the last token except in the first forward pass
                tokens_for_logits = current_tokens[:,-1:]

            logits = self.logits(tokens_for_logits, encoder_feature) # B, len(tokens), token dict size
            if new_segment:
                generation["logits_starting"] = Logits(logits[:,:,:])

            if new_segment and self.tokenizer.no_speech is not None:
                probs_at_sot = logits[:, self.sot_index, :].float().softmax(dim=-1)
                no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
                generation["no_speech_prob"] = no_speech_probs[0]
                if no_speech_probs[0] > self.cfg.nonspeech_prob:
                    generation["no_speech"] = True
                    logger.info("no speech, stop")
                    break

            logits = logits[:, -1, :] # logits for the last token
            generation_progress_loop.append(("logits_before_suppress",Logits(logits)))

            # supress blank tokens only at the beginning of the segment
            if new_segment:
                logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            new_segment = False
            self.suppress_tokens(logits)
            #generation_progress_loop.append(("logits_after_suppres",BeamLogits(logits[0,:].clone(), self.cfg.beam_size)))
            generation_progress_loop.append(("logits_after_suppress",Logits(logits)))

            # Pass mel and model for speculative decoding
            if hasattr(self.token_decoder, 'use_speculative') and self.token_decoder.use_speculative:
                current_tokens, completed = self.token_decoder.update(
                    current_tokens, logits, sum_logprobs, mel=encoder_feature, main_model=self.model
                )
            else:
                current_tokens, completed = self.token_decoder.update(current_tokens, logits, sum_logprobs)
            generation_progress_loop.append(("beam_tokens",Tokens(current_tokens[:,-1].clone())))
            generation_progress_loop.append(("sum_logprobs",sum_logprobs.tolist()))
            generation_progress_loop.append(("completed",completed))

            logger.debug(f"Decoding completed: {completed}, sum_logprobs: {sum_logprobs.tolist()}, tokens: ")
            self.debug_print_tokens(current_tokens)

            # Rule-based sentence breaking (punctuation and conjunctions)
            # Check tokens generated so far (excluding initial prompt tokens)
            if not is_last:  # Only apply rule-based breaking during streaming, not at the end
                new_tokens_so_far = current_tokens[0, token_len_before_decoding:].tolist()
                should_break, break_offset = self.should_break_at_punctuation(new_tokens_so_far)
                if should_break:
                    if break_offset < 0:
                        # Break before conjunction - trim the conjunction tokens
                        logger.info(f"[Rule-based break] Breaking before conjunction, trimming {-break_offset} tokens")
                        current_tokens = current_tokens[:, :break_offset]
                    else:
                        # Break after punctuation - keep all tokens
                        logger.info(f"[Rule-based break] Breaking after punctuation")
                    completed = True
                    generation["rule_based_break"] = True
                    break

            # Hallucination detection
            current_token = current_tokens[0, -1].item()

            # Check for consecutive identical tokens (5 or more)
            if current_token == last_token:
                consecutive_token_count += 1
                if consecutive_token_count >= 5:
                    logger.info(f"Hallucination detected: 5+ consecutive identical tokens ({current_token}). Stopping.")
                    completed = True
                    current_tokens = current_tokens[:, :-1]
                    break
            else:
                consecutive_token_count = 1
                last_token = current_token

            # Track recent tokens for unique word check
            recent_tokens.append(current_token)
            if len(recent_tokens) > 10:
                recent_tokens.pop(0)

            # Check if recent 10 tokens have less than 2 unique words
            if len(recent_tokens) >= 10:
                # Decode recent tokens to text and split by words
                recent_text = self.tokenizer.decode(recent_tokens)
                words = recent_text.strip().split()
                unique_words = set(words)
                if len(unique_words) < 2:
                    logger.info(f"Hallucination detected: Less than 2 unique words in recent 10 tokens. Flushing.")
                    completed = True
                    current_tokens = current_tokens[:, :-len(recent_tokens)]
                    break

            # Check total token count limit (200 tokens ≈ 30 seconds)
            total_tokens = current_tokens.shape[1] - token_len_before_decoding
            if total_tokens >= 200:
                logger.info(f"Token limit reached: {total_tokens} tokens. Stopping to prevent runaway generation.")
                completed = True
                break

            # Early exit if completed (e.g., by rule-based breaking or <|endoftext|>)
            # This must happen BEFORE attention processing to preserve punctuation
            if completed:
                # Strip the <|endoftext|> token if present
                if current_tokens.shape[1] > 0 and current_tokens[0, -1].item() == 50257:
                    current_tokens = current_tokens[:, :-1]
                break

            # if self.decoder_type == "beam":
            #     logger.debug(f"Finished sequences: {self.token_decoder.finished_sequences}")

            #     logprobs = F.log_softmax(logits.float(), dim=-1)
            #     idx = 0
            #     logger.debug(f"Beam search topk: {logprobs[idx].topk(self.cfg.beam_size + 1)}")
            #     logger.debug(f"Greedy search argmax: {logits.argmax(dim=-1)}")
            # if completed:
            #     self.debug_print_tokens(current_tokens)

            #     logger.debug("decode stopped because decoder completed")

            attn_of_alignment_heads = [[] for _ in range(self.num_align_heads)]
            for i, attn_mat in enumerate(self.dec_attns):
                layer_rank = int(i % len(self.model.decoder.blocks))
                align_heads_in_layer = self.align_source.get(layer_rank, [])
                if len(align_heads_in_layer) == 0:
                    continue
                for align_head_rank, head_id in align_heads_in_layer:
                    if self.cfg.beam_size == 1:
                        a = attn_mat[head_id, :, :]
                        a = a.unsqueeze(0)
                    else:
                        a = attn_mat[:, head_id, :, :]
                    attn_of_alignment_heads[align_head_rank].append(a)
            tmp = []
            for mat in attn_of_alignment_heads:
                t = torch.cat(mat, dim=1)
                tmp.append(t) 
            attn_of_alignment_heads = torch.stack(tmp, dim=1)
#            logger.debug(str(attn_of_alignment_heads.shape) + " tttady")
            std, mean = torch.std_mean(attn_of_alignment_heads, dim=-2, keepdim=True, unbiased=False)
            attn_of_alignment_heads = (attn_of_alignment_heads - mean) / std
            attn_of_alignment_heads = median_filter(attn_of_alignment_heads, 7) # from whisper.timing
            attn_of_alignment_heads = attn_of_alignment_heads.mean(dim=1)
#            logger.debug(str(attn_of_alignment_heads.shape) + " po mean")
            attn_of_alignment_heads = attn_of_alignment_heads[:,:, :content_mel_len]
#            logger.debug(str(attn_of_alignment_heads.shape) + " pak ")

            # for each beam, the most attended frame is:
            most_attended_frames = torch.argmax(attn_of_alignment_heads[:,-1,:], dim=-1)
            generation_progress_loop.append(("most_attended_frames",most_attended_frames.clone().tolist()))
            logger.debug(str(most_attended_frames.tolist()) + " most att frames")

            most_attended_frame = most_attended_frames[0].item()


            generation_progress.append(dict(generation_progress_loop))
            logger.debug("current tokens" + str(current_tokens.shape))
            if completed:
            #    # stripping the last token, the eot
                current_tokens = current_tokens[:, :-1]
                break
            
            # for some rare cases where the attention fails
            if not is_last and self.last_attend_frame - most_attended_frame > self.cfg.rewind_threshold:
                # TODO: check this
                if current_tokens.shape[1] > 1 and current_tokens[0, -2] >= DEC_PAD:
                    logger.debug("ommit rewinding from special tokens")
                    self.last_attend_frame = most_attended_frame
                else:
                    logger.debug(
                        f"[rewind detected] current attention pos: {most_attended_frame}, "
                        f"last attention pos: {self.last_attend_frame}; omit this segment")
                    self.last_attend_frame = -self.cfg.rewind_threshold
                    current_tokens = torch.cat(self.tokens, dim=1) if len(self.tokens) > 0 else self.tokens[0]
                    break
            else:
                self.last_attend_frame = most_attended_frame

            if content_mel_len - most_attended_frame <= (4 if is_last else self.cfg.frame_threshold):
                logger.debug(f"attention reaches the end: {most_attended_frame}/{content_mel_len}")
                # Check if the last token contains sentence-ending punctuation
                last_token_text = self.tokenizer.decode([current_tokens[0, -1].item()])
                has_sentence_end = any(p in last_token_text for p in SENTENCE_END_PUNCTUATION)

                if has_sentence_end:
                    # Keep the last token if it contains sentence-ending punctuation
                    logger.info(f"[Attention] Keeping last token '{last_token_text}' (contains punctuation)")
                    # Don't strip, just break
                    break
                else:
                    # stripping the last token, the one that is attended too close to the end
                    current_tokens = current_tokens[:, :-1]
                    break
        
            # debug print
            for i in range(self.cfg.beam_size):
                logger.debug("attn: {}, current pos: {}, current token: {}({})".format(
                    attn_of_alignment_heads.shape if attn_of_alignment_heads is not None else None,
                    most_attended_frames[i], 
                    current_tokens[i, -1].item(),
                    self.tokenizer.decode([current_tokens[i, -1].item()])
                ))

#        for k,v in generation.items():
#            print(k,v,file=sys.stderr)
#        for x in generation_progress:
#            for y in x.items():
#                print("\t\t",*y,file=sys.stderr)
#            print("\t","----", file=sys.stderr)
#        print("\t", "end of generation_progress_loop", file=sys.stderr)
        #    sys.exit(1)
        ####################### End of decoding loop

        logger.info("End of decoding loop")

        # if attn_of_alignment_heads is not None:
        #     seg_len = int(segment.shape[0] / 16000 * TOKENS_PER_SECOND)

        #     # Lets' now consider only the top hypothesis in the beam search
        #     top_beam_attn_of_alignment_heads = attn_of_alignment_heads[0]

        #     # debug print: how is the new token attended?
        #     new_token_attn = top_beam_attn_of_alignment_heads[token_len_before_decoding:, -seg_len:]
        #     logger.debug(f"New token attention shape: {new_token_attn.shape}")
        #     if new_token_attn.shape[0] == 0:  # it's not attended in the current audio segment
        #         logger.debug("no token generated")
        #     else:  # it is, and the max attention is:
        #         new_token_max_attn, _ = new_token_attn.max(dim=-1)
        #         logger.debug(f"segment max attention: {new_token_max_attn.mean().item()/len(self.segments)}")


        # let's now operate only with the top beam hypothesis
        tokens_to_split = current_tokens[0, token_len_before_decoding:]
        if fire_detected or is_last or completed:
            new_hypothesis = tokens_to_split.flatten().tolist()
        else:
            # going to truncate the tokens after the last space
            split_words, split_tokens = self.tokenizer.split_to_word_tokens(tokens_to_split.tolist())
            generation["result"] = {"split_words": split_words[:-1], "split_tokens": split_tokens[:-1]}
            generation["result_truncated"] = {"split_words": split_words[-1:], "split_tokens": split_tokens[-1:]}

#            text_to_split = self.tokenizer.decode(tokens_to_split)
#            logger.debug(f"text_to_split: {text_to_split}")
#            logger.debug("text at current step: {}".format(text_to_split.replace(" ", "<space>")))
#            text_before_space = " ".join(text_to_split.split(" ")[:-1])
#            logger.debug("before the last space: {}".format(text_before_space.replace(" ", "<space>")))
            if len(split_words) > 1:
                new_hypothesis = [i for sublist in split_tokens[:-1] for i in sublist]  
            else:
                new_hypothesis = []


        ### new hypothesis
        logger.debug(f"new_hypothesis: {new_hypothesis}")
        new_tokens = torch.tensor([new_hypothesis], dtype=torch.long).repeat_interleave(self.cfg.beam_size, dim=0).to(
            device=self.model.device,
        )
        self.tokens.append(new_tokens)
        # TODO: test if this is redundant or not
#        ret = ret[ret<DEC_PAD]

        logger.info(f"Output: {self.tokenizer.decode(new_hypothesis)}")
        
        self._clean_cache()

        self.logdir_save(input_segments, new_hypothesis, generation)
        return new_hypothesis, generation

    def logdir_save(self, input_segments, new_hypothesis, generation):
        """The audio and result from each iteration is saved to the logdir for debugging purposes"""

        # only when the logdir arg is set
        if self.cfg.logdir is None:
            return

        self.logdir_i += 1

        # every VAD segment is in a separate directory
        dir = os.path.join(self.cfg.logdir, f"seg_{self.log_segments:05d}")
        if not os.path.exists(dir):
            os.makedirs(dir)

        logger.debug(f"Saving to {dir}, iteration {self.logdir_i:05d}")

        # saving wav:
        wav_path = os.path.join(dir, f"iter_{self.logdir_i:05d}_audio.wav")
        audio_np = np.array(input_segments)
        # Ensure audio is float32 in range [-1, 1], convert to int16 for wav
        if audio_np.dtype != np.int16:
            audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_int16 = audio_np

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        # saving readable text: context + hypothesis
        text = self.tokenizer.decode(new_hypothesis)
        with open(os.path.join(dir, f"iter_{self.logdir_i:05d}_hypothesis.txt"), "w") as f:
            if generation:
                context = generation["starting_tokens"].as_text(self.tokenizer)
            else:
                context = ""
            print("CONTEXT+FORCED:",context,sep="\t",file=f)
            print("HYPOTHESIS:", text, sep="\t", file=f)

        # TODO: generation progress can be also saved in a readable format
        #logger.debug(f"generation progress: {generation}")
