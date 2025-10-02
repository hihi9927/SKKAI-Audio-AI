import io
import os
from typing import List, Optional, Union

import torch

from .audio import load_audio, log_mel_spectrogram, pad_or_trim
from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import ModelDimensions, Whisper
from .transcribe import transcribe
from .version import __version__

# Model weights (for offline use)
_MODELS = {
    "tiny.en": "weights/tiny.en.pt",
    "tiny": "weights/tiny.pt", 
    "base.en": "weights/base.en.pt",
    "base": "weights/base.pt",
    "small.en": "weights/small.en.pt",
    "small": "weights/small.pt",
    "medium.en": "weights/medium.en.pt",
    "medium": "weights/medium.pt",
    "large-v1": "weights/large-v1.pt",
    "large-v2": "weights/large-v2.pt",
    "large-v3": "weights/large-v3.pt",
    "large-v3-turbo": "weights/large-v3-turbo.pt"
}

# base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
# highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
_ALIGNMENT_HEADS = {
    "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
    "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
    "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
    "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
    "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
    "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
    "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
    "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
    "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
    "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
    "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
    "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`"
}

def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    model_dir: str = None,
    in_memory: bool = False,
) -> Whisper:
    """
    Load a Whisper ASR model from local files

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    model_dir: str
        path to the directory containing model files; by default, uses current directory
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_dir is None:
        model_dir = "."

    if name in _MODELS:
        # 로컬 파일 경로 구성
        model_path = os.path.join(model_dir, _MODELS[name])
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        checkpoint_file = open(model_path, "rb").read() if in_memory else model_path
        alignment_heads = _ALIGNMENT_HEADS[name]
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        kwargs = {"weights_only": True} if torch.__version__ >= "1.13" else {}
        checkpoint = torch.load(fp, map_location=device, **kwargs)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)
