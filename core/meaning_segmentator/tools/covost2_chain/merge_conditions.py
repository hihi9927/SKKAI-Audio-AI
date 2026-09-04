"""bleu_eval 이 덮어쓴 `bleu/<tgt>.json` 에 백업의 조건을 도로 합친다.

`bleu_eval` 은 이번 실행에서 계산한 조건만 담아 파일을 **통째로 덮어쓴다.** 조건 몇 개만
추가하려고 `--conditions` 로 좁혀 돌리면 나머지 70개 조건과 거기 붙어 있던 COMET 값이
같이 날아간다. 그래서 미리 떠 둔 백업을 기준으로 두고, 새로 계산된 조건만 얹는다.

겹치는 조건은 **백업 쪽을 남긴다** (COMET 값이 붙어 있는 쪽). 다만 BLEU 가 다르면
번역이 재현되지 않았다는 뜻이므로 경고를 찍는다.

    python core/meaning_segmentator/tools/covost2_chain/merge_conditions.py <백업 디렉토리> <대상 디렉토리> de ja zh
"""
import json
import sys
from pathlib import Path


def main() -> int:
    backup_dir, live_dir = Path(sys.argv[1]), Path(sys.argv[2])
    for tgt in sys.argv[3:]:
        bpath, lpath = backup_dir / f"{tgt}.json", live_dir / f"{tgt}.json"
        base = json.loads(bpath.read_text(encoding="utf-8"))
        new = json.loads(lpath.read_text(encoding="utf-8"))
        added, kept = [], []
        for name, cell in new["conditions"].items():
            if name in base["conditions"]:
                old_bleu = base["conditions"][name].get("bleu")
                if old_bleu is not None and cell.get("bleu") is not None \
                        and abs(old_bleu - cell["bleu"]) > 0.01:
                    print(f"  !! [{tgt}] {name}: BLEU 재현 안 됨 "
                          f"{old_bleu} -> {cell['bleu']} (백업 값을 남긴다)")
                kept.append(name)
            else:
                base["conditions"][name] = cell
                added.append(name)
        bpath.with_suffix(".json").parent.mkdir(parents=True, exist_ok=True)
        lpath.write_text(json.dumps(base, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        print(f"[{tgt}] 조건 {len(base['conditions'])}개 "
              f"(추가 {len(added)}: {', '.join(added) or '-'} / 유지 {len(kept)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
