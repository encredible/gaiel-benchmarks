#!/bin/bash
# 사용법: run_eval.sh <suite: korean|coding> <tag> <model>
# 결과는 results/<tag>/ 에 lm-eval 원본 JSON 그대로 저장한다 (가공 없음).
set -u
PY="${MLX_PY:-/Users/K/omni-universe/omni-llm-engine/venv/bin}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUITE=$1; TAG=$2; MODEL=$3
OUT="$ROOT/results/$TAG"; mkdir -p "$OUT"
[ -f "$OUT/DONE.$SUITE" ] && { echo "[$TAG/$SUITE] 완료됨 — 건너뜀"; exit 0; }

case "$SUITE" in
  korean) TASKS="kobest_boolq kobest_copa kobest_hellaswag kobest_sentineg kobest_wic haerae"; EXTRA="" ;;
  coding) TASKS="humaneval_instruct mbpp_plus_instruct"; EXTRA="--apply-chat-template --confirm-run-unsafe-code"
          export HF_ALLOW_CODE_EVAL=1 ;;
  *) echo "알 수 없는 스위트: $SUITE"; exit 2 ;;
esac

echo ">>> [$TAG/$SUITE] $MODEL  $(date '+%F %T')"
"$PY/mlx_lm.evaluate" --model "$MODEL" --tasks $TASKS $EXTRA \
    --output-dir "$OUT" > "$OUT/$SUITE.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  touch "$OUT/DONE.$SUITE"; echo "<<< [$TAG/$SUITE] 완료"
else
  echo "<<< [$TAG/$SUITE] 실패 rc=$rc"; tail -5 "$OUT/$SUITE.log"
fi
exit $rc
