# KsponSpeech 100h SFT 데이터셋 빌드 산출물

llama3.3:70b SEG 라벨 기반, 원본+split(교차 흡수) 구성. 최종 학습 jsonl은
`Qwen3-ASR/finetuning/data/KSponSpeech/{train,val,eval_clean,eval_other}.jsonl`.

## 구성
- **train** ~110k = 원본 ~60k (전용 idx 1~10000+67501~110000 + 교차) + split ~50k (소스 idx 10001~67500)
- **val** 6k = 원본 3k (idx 110001~113000) + split 3k (idx 113001~116000)
- **eval** = eval_clean(원본 E01501~E03000 1.5k + split E00001~E01500 1.5k) / eval_other(원본 E04501~E06000 1.5k + split E03001~E04500 1.5k)
- 교차 = split이 너무 짧아 스킵된 소스를 원본으로 흡수(무손실)

## 폴더
- `inputs/` — SEG 소스 슬라이스(전사). `train3/train_existing`=원문, `*_dedicated`=원본전용, `*_split_input`=split소스, `*_orig`=eval원본절반
- `seg/` — gpt_seg_ko.py(llama70b) SEG 라벨 결과 (원문훼손 0 검증)
- `split/` — generate_split_data.py(forced-align) 결과. `*_data`=split클립, `*_skipped`=교차대상(원본흡수)
- `finish_pipeline.sh` — train split 후 val/eval split + 조립 자동화 스크립트

## 재현 파이프라인
1. 전사: `utils/extract_trn_to_json.py` (train.trn → inputs/)
2. SEG: `core/meaning_segmentator/utils/gpt_seg_ko.py --provider ollama --model llama3.3:70b --workers 8`
3. WAV: `Qwen3-ASR/finetuning/utils/pcm_to_wav.py`
4. split: `utils/generate_split_data.py --skipped-output ... [--flat-pcm]`
5. 조립: `Qwen3-ASR/finetuning/utils/assemble_dataset.py` → 최종 jsonl

오디오: 원본 `finetuning/data/KSponSpeech/audio/`, split `split_audio/`, eval원본 `eval_audio/`, eval split `eval_split_audio/`.
