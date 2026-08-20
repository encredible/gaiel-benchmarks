#!/bin/bash
# LogicKor 실행 준비물을 받는다. 문항과 채점 템플릿은 공식 저장소 것을 쓰며
# 재배포하지 않으므로 이 스크립트로 내려받는다.
set -euo pipefail
WORK="${LOGICKOR_WORK:-$HOME/omni-work}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WORK"

# 문항: 이 저장소에 사본이 있으면 그걸 쓰고, 없으면 원본에서 받는다
if [ -f "$ROOT/inputs/logickor/questions.jsonl" ]; then
  cp "$ROOT/inputs/logickor/questions.jsonl" "$WORK/logickor_questions.jsonl"
else
  curl -sfL https://huggingface.co/datasets/maywell/LogicKor/resolve/main/questions.jsonl \
       -o "$WORK/logickor_questions.jsonl"
fi

# 채점 템플릿: 공식 저장소에서만 받는다 (재배포 금지)
curl -sfL https://raw.githubusercontent.com/instructkr/LogicKor/main/templates.py \
     -o "$WORK/logickor_templates.py"

python3 - "$WORK" <<'PY'
import sys, json
from pathlib import Path
w = Path(sys.argv[1]); sys.path.insert(0, str(w))
from logickor_templates import JUDGE_TEMPLATE
rows = [json.loads(l) for l in open(w / "logickor_questions.jsonl") if l.strip()]
print(f"준비 완료: {w}")
print(f"  문항 {len(rows)}개 · 템플릿 {sorted(JUDGE_TEMPLATE)}")
PY

echo
echo "채점에는 OPENAI_API_KEY 가 필요하다 (judge: gpt-4o)."
echo "실행:  OPENAI_API_KEY=... python3 tools/logickor_run.py <model_id> <tag>"
