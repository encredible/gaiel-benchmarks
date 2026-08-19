#!/usr/bin/env python3
"""results/ 의 lm-eval 원본 출력을 모아 summary.csv 와 README 표를 만든다.
튜닝 모델은 베이스와의 차이(Δ)를 함께 낸다 — 회귀가 한눈에 보이게 하기 위해서다."""
import json, csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, INP = ROOT / "results", ROOT / "inputs"

KOREAN = ["kobest_boolq","kobest_copa","kobest_hellaswag","kobest_sentineg","kobest_wic","haerae"]
CODING = ["humaneval_instruct","mbpp_plus_instruct"]
# tag -> 비교 기준 베이스 tag
PAIRS = {"gaiel-1.5b":"base-1.5b", "gaiel-7b":"base-7b", "gaiel-32b":"base-32b",
         "gaiel-coding-1.5b":"base-coder-1.5b", "gaiel-korean-1.5b":"base-coder-1.5b"}

def primary(d):
    """acc / pass@1 / exact_match 중 첫 번째 실측 지표를 뽑는다."""
    for k, v in d.items():
        if "stderr" in k or not isinstance(v, (int, float)): continue
        if k.split(",")[0] in ("acc","pass@1","exact_match","acc_norm"):
            return k.split(",")[0], float(v)
    return None, None

def load():
    out = {}                      # tag -> {task: (metric, value, n)}
    for tagdir in sorted(p for p in RES.iterdir() if p.is_dir()):
        scores = {}
        for f in tagdir.glob("eval_*"):
            try: doc = json.loads(f.read_text())
            except Exception: continue
            for task, d in doc.items():
                m, v = primary(d)
                if v is not None: scores[task] = (m, v, d.get("sample_len"))
        if scores: out[tagdir.name] = scores
    return out

def avg(scores, tasks):
    vals = [scores[t][1] for t in tasks if t in scores]
    return sum(vals)/len(vals) if vals else None

def fmt(v): return "—" if v is None else f"{v*100:.1f}"

def table(data, tasks, title):
    have = [t for t in data if any(k in data[t] for k in tasks)]
    if not have: return f"### {title}\n\n_아직 측정된 결과가 없습니다._\n"
    hdr = "| 모델 | " + " | ".join(t.replace("kobest_","").replace("_instruct","") for t in tasks) + " | **평균** | Δ vs 베이스 |"
    sep = "|" + "---|" * (len(tasks) + 3)
    lines = [f"### {title}", "", hdr, sep]
    for tag in sorted(have):
        s = data[tag]
        cells = [fmt(s[t][1]) if t in s else "—" for t in tasks]
        a = avg(s, tasks)
        base = PAIRS.get(tag)
        delta = "—"
        if base and base in data:
            b = avg(data[base], tasks)
            if a is not None and b is not None:
                d = (a-b)*100
                delta = f"**{d:+.1f}**" + (" ⚠️회귀" if d < -1 else (" ✅" if d > 1 else ""))
        lines.append(f"| `{tag}` | " + " | ".join(cells) + f" | **{fmt(a)}** | {delta} |")
    return "\n".join(lines) + "\n"

def main():
    data = load()
    RES.mkdir(exist_ok=True)
    with (RES/"summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag","task","metric","value","n"])
        for tag, s in sorted(data.items()):
            for task,(m,v,n) in sorted(s.items()): w.writerow([tag,task,m,f"{v:.4f}",n])

    body = (ROOT/"README.template.md").read_text()
    body = body.replace("{{KOREAN_TABLE}}", table(data, KOREAN, "한국어"))
    body = body.replace("{{CODING_TABLE}}", table(data, CODING, "코딩"))
    body = body.replace("{{N_MODELS}}", str(len(data)))
    (ROOT/"README.md").write_text(body)
    print(f"수집 완료: 모델 {len(data)}개 → results/summary.csv, README.md")
    for tag,s in sorted(data.items()): print(f"  {tag}: {len(s)}개 태스크")

if __name__ == "__main__":
    main()
