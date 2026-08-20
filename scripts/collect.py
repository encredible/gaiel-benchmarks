#!/usr/bin/env python3
"""results/ 의 lm-eval 원본 출력을 모아 summary.csv 와 README 표를 만든다.
튜닝 모델은 베이스와의 차이(Δ)를 함께 낸다 — 회귀가 한눈에 보이게 하기 위해서다."""
import json, csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES, INP = ROOT / "results", ROOT / "inputs"

KOREAN = ["kobest_boolq","kobest_copa","kobest_hellaswag","kobest_sentineg","kobest_wic","haerae"]
CODING = ["humaneval_instruct","humaneval_plus","mbpp_plus"]
# extract_code 필터 결함으로 전 모델 0.0 이 나오는 태스크. 원본 JSON 에는 남기되
# summary.csv 에서는 뺀다 — 평균 집계에 섞이면 회귀 판정이 오염된다.
EXCLUDE = {"mbpp_plus_instruct"}
# 우연 수준(무작위 선택 기대값). 이 선을 못 넘으면 해당 과제를 푸는 능력이 없다는 뜻이다.
CHANCE = {"kobest_boolq":.50, "kobest_copa":.50, "kobest_hellaswag":.25,
          "kobest_sentineg":.50, "kobest_wic":.50, "haerae":.20,
          "humaneval_instruct":.0, "humaneval_plus":.0, "mbpp_plus":.0}
# tag -> 비교 기준 베이스 tag
PAIRS = {"gaiel-1.5b":"base-1.5b", "gaiel-7b":"base-7b", "gaiel-32b":"base-32b",
         "gaiel-8b":"base-8b", "gaiel-72b":"base-72b",
         "gaiel-coding-1.5b":"base-coder-1.5b", "gaiel-korean-1.5b":"base-coder-1.5b"}

def primary(d):
    """acc / pass@1 / exact_match 중 첫 실측 지표와 그 표준오차를 뽑는다."""
    for k, v in d.items():
        if "stderr" in k or not isinstance(v, (int, float)): continue
        m = k.split(",")[0]
        if m in ("acc","pass@1","pass_at_1","exact_match","acc_norm"):
            se = d.get(k.replace(m, m+"_stderr", 1))
            return m, float(v), (float(se) if isinstance(se,(int,float)) else 0.0)
    return None, None, 0.0

def load():
    out = {}                      # tag -> {task: (metric, value, n)}
    for tagdir in sorted(p for p in RES.iterdir() if p.is_dir()):
        scores = {}
        for f in list(tagdir.glob("eval_*")) + list(tagdir.glob("lmeval_*.json")):
            try: doc = json.loads(f.read_text())
            except Exception: continue
            doc = doc.get("results", doc)      # lm-eval CLI 는 results 키 아래에 담는다
            for task, d in doc.items():
                if not isinstance(d, dict): continue
                m, v, se = primary(d)
                if v is not None: scores[task] = (m, v, d.get("sample_len"), se)
        if scores: out[tagdir.name] = scores
    return out

def avg(scores, tasks):
    vals = [scores[t][1] for t in tasks if t in scores]
    return sum(vals)/len(vals) if vals else None

def fmt(v): return "—" if v is None else f"{v*100:.1f}"

def at_chance(task, v, se):
    """표준오차 1배 안에서 우연 수준과 구분되지 않으면 '못 푼다'로 본다."""
    c = CHANCE.get(task)
    return bool(c) and v <= c + max(se, 0.005)

def cell(task, rec):
    if rec is None: return "—"
    _, v, _, se = rec
    return f"{v*100:.1f}" + (" ˣ" if at_chance(task, v, se) else "")

def table(data, tasks, title):
    have = [t for t in data if any(k in data[t] for k in tasks)]
    if not have: return f"### {title}\n\n_아직 측정된 결과가 없습니다._\n"
    names = [t.replace("kobest_","").replace("_instruct","").replace("mbpp_plus","mbpp+") for t in tasks]
    lines = [f"### {title}", "",
             "| 모델 | " + " | ".join(names) + " | **평균** | Δ vs 베이스 | 판정 |",
             "|" + "---|" * (len(tasks) + 4)]
    # 우연 수준 기준선을 첫 행에 둔다
    ch = [f"{CHANCE[t]*100:.0f}" if CHANCE.get(t) else "0" for t in tasks]
    lines.append("| _우연 수준_ | " + " | ".join(f"_{c}_" for c in ch) + " | _—_ | _—_ | _기준선_ |")
    for tag in sorted(have):
        sc = data[tag]
        cells = [cell(t, sc.get(t)) for t in tasks]
        a = avg(sc, tasks)
        base = PAIRS.get(tag)
        delta, verdict = "—", ""
        # 우연 수준을 못 넘은 과제 수
        nchance = sum(1 for t in tasks if t in sc and at_chance(t, sc[t][1], sc[t][3]))
        if base and base in data:
            # Δ 는 양쪽이 모두 가진 과제에서만 계산한다.
            # 과제 집합이 다르면 평균끼리 빼는 것은 무의미하고, 회귀를 개선으로 뒤집는다.
            common = [t for t in tasks if t in sc and t in data[base]]
            a_c, b_c = avg(sc, common), avg(data[base], common)
            if a_c is not None and b_c is not None:
                d = (a_c-b_c)*100
                partial = "" if len(common) == len([t for t in tasks if t in data[base]]) else f" ({len(common)}개 공통)"
                delta = f"**{d:+.1f}**{partial}"
                verdict = "⚠️ 회귀" if d < -1 else ("✅ 개선" if d > 1 else "≈ 동등")
        if nchance >= len(tasks)//2 and nchance > 0:
            verdict = (verdict + " · " if verdict else "") + f"ˣ{nchance}개 우연이하"
        # 평균의 분모(실제 측정된 과제 수)를 함께 적는다. 행마다 분모가 다른데
        # 숫자만 나란히 두면 3과제 평균과 1과제 평균을 같은 것으로 읽게 된다.
        nhave = sum(1 for t in tasks if t in sc)
        amark = f"**{fmt(a)}**" + (f" _({nhave}/{len(tasks)})_" if nhave < len(tasks) else "")
        lines.append(f"| `{tag}` | " + " | ".join(cells) + f" | {amark} | {delta} | {verdict or '—'} |")
    lines.append("")
    lines.append("`ˣ` = 표준오차 범위 안에서 우연 수준과 구분되지 않음 (그 과제를 실제로 풀지 못한다는 뜻)")
    lines.append("")
    lines.append("**평균 옆의 _(n/m)_ 은 m개 과제 중 n개만 측정됐다는 뜻이다.** 분모가 다른 평균끼리는 "
                 "비교할 수 없다 — 옆줄과 견주려면 Δ 열을 보라. Δ 는 양쪽이 모두 가진 과제에서만 계산한다.")
    return "\n".join(lines) + "\n"

def main():
    data = load()
    RES.mkdir(exist_ok=True)
    with (RES/"summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag","task","metric","value","stderr","n","at_chance"])
        for tag, s in sorted(data.items()):
            for task,(m,v,n,se) in sorted(s.items()):
                if task in EXCLUDE: continue
                w.writerow([tag,task,m,f"{v:.4f}",f"{se:.4f}",n,
                            "yes" if at_chance(task,v,se) else "no"])

    body = (ROOT/"README.template.md").read_text()
    body = body.replace("{{KOREAN_TABLE}}", table(data, KOREAN, "한국어"))
    body = body.replace("{{CODING_TABLE}}", table(data, CODING, "코딩"))
    body = body.replace("{{N_MODELS}}", str(len(data)))
    (ROOT/"README.md").write_text(body)
    print(f"수집 완료: 모델 {len(data)}개 → results/summary.csv, README.md")
    for tag,s in sorted(data.items()): print(f"  {tag}: {len(s)}개 태스크")

if __name__ == "__main__":
    main()
