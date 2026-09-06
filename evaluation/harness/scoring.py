"""채점. 영어는 WER, 한국어는 CER 을 쓴다.

둘 다 `jiwer` 대신 자체 편집거리를 쓰는 이유는 정규화 규칙을 데이터셋 사이에서 똑같이
맞춰야 하기 때문이다 — 소문자화하고 구두점을 공백으로 바꾼 뒤 공백을 정리한다.
"""
import logging
import re
from statistics import mean

logger = logging.getLogger(__name__)


def compute_wer_for_rows(rows):
    try:
        import jiwer
    except ImportError:
        return None

    refs = [r['reference'] for r in rows if r.get('reference') and r.get('hypothesis')]
    hyps = [r['hypothesis'] for r in rows if r.get('reference') and r.get('hypothesis')]
    if not refs:
        return None

    import re

    def normalize(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    refs = [normalize(x) for x in refs]
    hyps = [normalize(x) for x in hyps]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip() and h.strip()]
    if not pairs:
        return None

    refs2, hyps2 = zip(*pairs)
    return jiwer.wer(list(refs2), list(hyps2))

def calculate_wer(results, policy=None, emit_summary=True):
    try:
        import jiwer  # noqa: F401
    except ImportError:
        logger.warning('jiwer not installed. Install with: pip install jiwer')
        return None, {}
    overall_wer = compute_wer_for_rows(results)
    if overall_wer is None:
        logger.warning('No valid rows for WER.')
        return None, {}
    wer = overall_wer
    folder_wers = {}
    by_speaker = {}
    for r in results:
        by_speaker.setdefault(r['speaker_id'], []).append(r)

    for speaker_id, rows in by_speaker.items():
        folder_wers[speaker_id] = compute_wer_for_rows(rows)

    if emit_summary:
        first_token_latencies = [r['first_token_latency'] for r in results if r['first_token_latency'] is not None]
        avg_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0.0
        model_runtimes = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
        avg_model_runtime = sum(model_runtimes) / len(model_runtimes) if model_runtimes else 0.0

        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY%s', policy_label)
        logger.info('%s', '=' * 70)
        logger.info('Total files processed: %s', len(results))
        logger.info('Overall WER: %.2f%%', wer * 100)
        logger.info('Average First Token Latency: %.3fs', avg_first_token_latency)
        logger.info('Average Model Runtime: %.3fs', avg_model_runtime)
        logger.info('Number of speakers: %s', len(by_speaker))
        logger.info('%s\n', '=' * 70)

    return wer, folder_wers


# ── CER (한국어) ─────────────────────────────────────────────────────────────
def _strip_for_cer(text):
    """공백과 구두점을 걷어낸다. 한국어는 띄어쓰기가 표기 흔들림이 커서 문자만 본다."""
    return re.sub(r'[\s\.,\?!]', '', text or '')


def _levenshtein(ref, hyp):
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def compute_cer(reference, hypothesis):
    ref = _strip_for_cer(reference)
    hyp = _strip_for_cer(hypothesis)
    if not ref:
        return None
    return _levenshtein(ref, hyp) / len(ref)


def compute_corpus_cer(rows):
    """말뭉치 단위 CER — 발화별 CER 의 평균이 아니라 오류 문자 합 / 참조 문자 합."""
    total_err = total_len = 0
    for r in rows:
        ref = _strip_for_cer(r.get('reference'))
        hyp = _strip_for_cer(r.get('hypothesis'))
        if not ref:
            continue
        total_err += _levenshtein(ref, hyp)
        total_len += len(ref)
    if total_len == 0:
        return None
    return total_err / total_len


def calculate_cer(results, policy=None, emit_summary=True):
    """`calculate_wer` 과 같은 모양의 CER 판. 화자별 dict 도 같이 돌려준다."""
    overall = compute_corpus_cer(results)
    if overall is None:
        logger.warning('No valid rows for CER.')
        return None, {}

    by_speaker = {}
    for r in results:
        by_speaker.setdefault(r.get('speaker_id', 'all'), []).append(r)
    folder_cers = {sid: compute_corpus_cer(rows) for sid, rows in by_speaker.items()}

    if emit_summary:
        lats = [r['first_token_latency'] for r in results if r.get('first_token_latency') is not None]
        runtimes = [r['model_runtime'] for r in results if r.get('model_runtime') is not None]
        policy_label = f' - BACKEND POLICY {policy}' if policy is not None else ''
        logger.info('\n%s', '=' * 70)
        logger.info('RESULTS SUMMARY%s', policy_label)
        logger.info('%s', '=' * 70)
        logger.info('Total files processed: %s', len(results))
        logger.info('Corpus CER: %.2f%%', overall * 100)
        logger.info('Average First Token Latency: %.3fs', mean(lats) if lats else 0.0)
        logger.info('Average Model Runtime: %.3fs', mean(runtimes) if runtimes else 0.0)
        logger.info('%s\n', '=' * 70)

    return overall, folder_cers
