"""LogicKor 생성 + 채점. 공식 instructkr/LogicKor 의 프롬프트/채점 방식을 그대로 따른다."""
import json, os, re, sys, time, urllib.request
from pathlib import Path

# 작업 디렉터리. 문항·채점 템플릿·생성 결과가 모두 여기 쌓인다.
# LOGICKOR_WORK 로 바꿀 수 있다. 준비물은 scripts/logickor_setup.sh 가 만든다.
WORK = Path(os.environ.get("LOGICKOR_WORK", Path.home() / "omni-work"))
sys.path.insert(0, str(WORK))
for _f in ("logickor_templates.py", "logickor_questions.jsonl"):
    if not (WORK / _f).exists():
        sys.exit(f"{WORK/_f} 없음 — scripts/logickor_setup.sh 를 먼저 실행하라")
# 채점 템플릿은 공식 저장소에서 받아 쓴다 (재배포하지 않음):
#   curl -sL https://raw.githubusercontent.com/instructkr/LogicKor/main/templates.py \
#        -o logickor_templates.py
# 문항도 마찬가지:
#   curl -sL https://huggingface.co/datasets/maywell/LogicKor/resolve/main/questions.jsonl \
#        -o logickor_questions.jsonl
from logickor_templates import JUDGE_TEMPLATE  # noqa

MODEL = sys.argv[1] if len(sys.argv) > 1 else "encredible/Gaiel-32B-Korean-Tuned-MLX-3bit"
TAG = sys.argv[2] if len(sys.argv) > 2 else "32b-3bit"
# exo 엔드포인트. `.local` mDNS 는 간헐적으로 해석에 실패하므로(같은 머신에서도
# nodename nor servname provided 로 죽는다) 응답하는 주소를 골라 쓴다.
# EXO_HOST 로 명시 지정할 수 있다 — 예) EXO_HOST=http://jg-macbookair.local:52415
def _pick_exo():
    cands = [os.environ["EXO_HOST"]] if os.environ.get("EXO_HOST") else [
        "http://127.0.0.1:52415",                      # 같은 머신이면 가장 안정적
        "http://jaegwanui-macbookpro.local:52415",     # 다른 머신에서 돌릴 때
    ]
    for base in cands:
        try:
            urllib.request.urlopen(base + "/state", timeout=5).read(1)
            return base
        except Exception:
            continue
    return cands[-1]  # 전부 실패하면 마지막 후보로 시도하고 chat() 의 재시도에 맡긴다

EXO = _pick_exo() + "/v1/chat/completions"
GEN_OUT = WORK / f"logickor_gen_{TAG}.jsonl"
JUDGE_OUT = WORK / f"logickor_judge_{TAG}.jsonl"


def chat(messages, max_tokens=1024, timeout=2400, retries=20):
    """연결이 끊겨도 죽지 않는다 — 마스터 재선출/노드 절전으로 잠깐 끊기는 일이 잦다."""
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.7}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(EXO, body, {"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            print(f"    재시도 {attempt+1}/{retries}: {str(e)[:70]}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"연결 실패 (재시도 {retries}회): {last}")


def generate():
    rows = [json.loads(l) for l in open(WORK / "logickor_questions.jsonl") if l.strip()]
    done = {}
    if GEN_OUT.exists():  # 중단되면 이어서
        for l in open(GEN_OUT):
            r = json.loads(l); done[r["id"]] = r
    print(f"생성 시작: {len(rows)}문항 (이미 완료 {len(done)}개)", flush=True)
    t0 = time.time()
    with open(GEN_OUT, "a") as f:
        for n, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            qs = row["questions"]
            msgs, outs = [], []
            for q in qs:  # 멀티턴: 이전 대화를 누적한다
                msgs.append({"role": "user", "content": q})
                a = chat(msgs)
                outs.append(a)
                msgs.append({"role": "assistant", "content": a})
            rec = {"id": row["id"], "category": row["category"],
                   "questions": qs, "outputs": outs,
                   "references": row.get("references") or [None] * len(qs)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            el = time.time() - t0
            print(f"  [{n}/{len(rows)}] {row['category']} 완료 (경과 {el/60:.0f}분)", flush=True)
    print(f"생성 완료: {(time.time()-t0)/60:.0f}분", flush=True)


def judge():
    key = None
    envf = Path("/Users/K/omni-universe/omni-universe-ai-backend/.env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY 없음 — 생성 결과만 저장하고 채점은 보류한다.", flush=True)
        return

    def ask(system, user, retries=8):
        """429(rate limit) 는 지수 백오프로 기다린다."""
        body = json.dumps({"model": "gpt-4o", "temperature": 0.0, "n": 1,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        for attempt in range(retries):
            try:
                req = urllib.request.Request("https://api.openai.com/v1/chat/completions", body,
                                             {"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
                d = json.loads(urllib.request.urlopen(req, timeout=300).read())
                time.sleep(2)  # 다음 요청까지 간격을 둔다
                return d["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = min(120, 15 * (2 ** attempt))
                    print(f"    429 — {wait}초 대기 후 재시도 ({attempt+1}/{retries})", flush=True)
                    time.sleep(wait); continue
                raise
            except Exception as e:
                print(f"    오류 재시도 ({attempt+1}/{retries}): {str(e)[:60]}", flush=True)
                time.sleep(20)
        raise RuntimeError("채점 요청 실패 — rate limit 지속")

    rows = [json.loads(l) for l in open(GEN_OUT) if l.strip()]
    print(f"\n채점 시작: {len(rows)}문항 × 2턴", flush=True)
    results = []
    with open(JUDGE_OUT, "w") as f:
        for n, r in enumerate(rows, 1):
            scores = {}
            for multi in (False, True):
                p = ("아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. "
                     "특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
                     f"**Question**\n{r['questions'][0]}")
                if r["references"][0]:
                    p += f"\n\n**Additional Reference**\n{r['references'][0]}"
                p += f"\n\n**Model's Response**\n{r['outputs'][0]}"
                if multi:
                    p += f"\n\n**Follow-up Question.**\n{r['questions'][1]}"
                    if len(r["references"]) > 1 and r["references"][1]:
                        p += f"\n\n**Additional Reference**\n{r['references'][1]}"
                    p += f"\n\n**Model's Response**\n{r['outputs'][1]}"
                p += "\n\n[[대화 종료. 평가 시작.]]"
                txt = ask(JUDGE_TEMPLATE["multi_turn" if multi else "single_turn"], p)
                m = re.findall(r"점수\s*[:：]\s*(\d+(?:\.\d+)?)", txt)
                scores["multi" if multi else "single"] = float(m[-1]) if m else None
            rec = {"id": r["id"], "category": r["category"], **scores}
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"  [{n}/{len(rows)}] {r['category']}: single={scores['single']} multi={scores['multi']}", flush=True)

    # 집계
    from collections import defaultdict
    cat = defaultdict(lambda: {"single": [], "multi": []})
    for r in results:
        for k in ("single", "multi"):
            if r[k] is not None:
                cat[r["category"]][k].append(r[k])
    print(f"\n=== LogicKor 결과: {MODEL} ===")
    print(f"{'카테고리':22}{'Single':>8}{'Multi':>8}")
    alls, allm = [], []
    for c, v in cat.items():
        s = sum(v["single"]) / len(v["single"]) if v["single"] else 0
        m = sum(v["multi"]) / len(v["multi"]) if v["multi"] else 0
        alls += v["single"]; allm += v["multi"]
        print(f"{c:22}{s:>8.2f}{m:>8.2f}")
    S = sum(alls) / len(alls); M = sum(allm) / len(allm)
    print(f"{'전체':22}{S:>8.2f}{M:>8.2f}")
    print(f"\n종합 점수: {(S+M)/2:.2f} / 10")
    print(f"비교 — Gaiel-32B-Korean-Tuned-MLX (4-bit 원본): 8.04 (single 8.67 / multi 7.41)")


if __name__ == "__main__":
    generate()
    judge()
