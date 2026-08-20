#!/bin/bash
# 사용법: run_eval.sh <suite: korean|coding> <tag> <model>
# 결과는 results/<tag>/ 에 lm-eval 원본 JSON 그대로 저장한다 (가공 없음).
set -u
# mlx_lm 실행 파일이 있는 디렉터리. 가상환경을 쓰면 MLX_PY 로 지정한다.
#   예) MLX_PY=~/myenv/bin ./scripts/run_eval.sh korean base-1.5b <model>
PY="${MLX_PY:-$(dirname "$(command -v mlx_lm.evaluate 2>/dev/null || echo /usr/local/bin/mlx_lm.evaluate)")}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUITE=$1; TAG=$2; MODEL=$3
OUT="$ROOT/results/$TAG"; mkdir -p "$OUT"
[ -f "$OUT/DONE.$SUITE" ] && { echo "[$TAG/$SUITE] 완료됨 — 건너뜀"; exit 0; }

# 챗 템플릿 적용 규칙 — 어기면 채점이 통째로 0 이 된다.
#   *_instruct 태스크        : --apply-chat-template 가 필요하다.
#   비-instruct(_plus) 태스크: 걸면 안 된다. 완성(completion) 형식으로 채점된다.
# 그래서 코딩 스위트는 한 번에 못 돌리고 두 패스로 나눈다.
# `mbpp_plus_instruct` 는 extract_code 필터 결함으로 전 모델 0.0 이 나온다 — 쓰지 않는다.
run_pass() {  # $1=패스이름  $2=태스크들  $3=추가플래그
  local name=$1 tasks=$2 extra=${3:-}
  echo ">>> [$TAG/$SUITE:$name] $MODEL  $(date '+%F %T')"
  "$PY/mlx_lm.evaluate" --model "$MODEL" --tasks $tasks $extra \
      --output-dir "$OUT" > "$OUT/$SUITE-$name.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { echo "<<< [$TAG/$SUITE:$name] 실패 rc=$rc"; tail -5 "$OUT/$SUITE-$name.log"; }
  return $rc
}

case "$SUITE" in
  korean)
    run_pass all "kobest_boolq kobest_copa kobest_hellaswag kobest_sentineg kobest_wic haerae"
    rc=$?
    ;;
  coding)
    export HF_ALLOW_CODE_EVAL=1
    run_pass instruct "humaneval_instruct" "--apply-chat-template --confirm-run-unsafe-code"
    rc=$?
    run_pass plus "humaneval_plus mbpp_plus" "--confirm-run-unsafe-code"
    rc=$(( rc | $? ))
    ;;
  *) echo "알 수 없는 스위트: $SUITE"; exit 2 ;;
esac

if [ $rc -eq 0 ]; then
  touch "$OUT/DONE.$SUITE"; echo "<<< [$TAG/$SUITE] 완료"
else
  echo "<<< [$TAG/$SUITE] 실패 rc=$rc"
fi
exit $rc
