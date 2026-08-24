"""Table 1a 비교군 — 제안 루프가 아닌 **외부 라벨 출처**들.

세 정책 모두 소스 조각 리스트만 내놓는다. 번역·BLEU·부트스트랩은 `bleu_eval` 이
그대로 쓴다 (조건 = 문장별 `{seg_text, pieces}`).

    punct         문장부호 휴리스틱 — 외부 의존 없음
    causal_align  TransLLaMa 식 인과정렬 (SimAlign) — **참조 번역 필요**
    mu_prefix     Zhang 2020 접두사 매칭 MU — **NMT 필요**

뒤의 둘은 타깃 언어마다 산출이 **다르다** (정렬 대상 / NMT 가 타깃별이다).
이것이 Table 1a "외부 의존" 열이 말하려는 비용 그 자체다 — 제안 루프는 프롬프트 하나로
모든 타깃을 덮지만 이 둘은 타깃마다 자원을 새로 확보해야 한다.
"""



def coarsen(pieces: list[str], target_chunk: int, spaced: bool = True) -> list[str]:
    """정책이 낸 경계 중 일부만 남겨 목표 조각 크기 `T` 에 맞춘다.

    조각 수 예산은 절단기(`pipeline.chunk_budget`)와 **같은 규칙**이다:
    `k = max(2, round(단위수 / T))`. 그래야 같은 T 에서 정책 간 k 가 맞아 지연 격자가
    비교 가능해진다.

    비교군에는 순위가 없어 어느 경계를 버릴지 고를 근거가 없으므로, 등간격 이상 위치에
    **가장 가까운** 경계를 고른다 — 결정론적이고 정책의 경계 *위치*만 쓴다(새 경계를
    만들지 않는다). 정책이 가진 경계가 예산보다 적으면 가진 것을 그대로 낸다.

    **제안 곡선과 비대칭이다.** `auto_T*` 는 LLM 순위로 *어느* 경계를 남길지 고르지만
    비교군은 못 고른다. 이 차이를 분리하려고 같은 규칙을 제안 마킹에도 적용한
    `auto_greedy_T*` 를 함께 낸다 — 비교군과 맞붙일 대조는 그쪽이다.
    """
    if len(pieces) <= 1:
        return list(pieces)
    unit = (lambda x: len(x.split())) if spaced else (lambda x: len("".join(x.split())))
    counts = [unit(x) for x in pieces]
    total = sum(counts)
    if total <= 0:
        return list(pieces)

    k = max(2, round(total / target_chunk))
    if len(pieces) <= k:
        return list(pieces)

    # 조각 i 뒤의 누적 위치가 후보 절단점이다 (마지막은 문장 끝이라 제외).
    cum, acc = [], 0
    for c in counts[:-1]:
        acc += c
        cum.append(acc)

    chosen: list[int] = []
    for i in range(1, k):
        ideal = total * i / k
        best = min((j for j in range(len(cum)) if j not in chosen),
                   key=lambda j: (abs(cum[j] - ideal), j))
        chosen.append(best)
    chosen.sort()

    out, start = [], 0
    joiner = " " if spaced else ""
    for j in chosen:
        out.append(joiner.join(pieces[start:j + 1]))
        start = j + 1
    out.append(joiner.join(pieces[start:]))
    return [x for x in out if x.strip()] or list(pieces)
