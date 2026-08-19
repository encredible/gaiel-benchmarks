#!/bin/bash
# 코딩 벤치마크 확충.
# 1단계: MBPP 변형 4종을 소량으로 돌려 어느 것이 정상 채점되는지 가려낸다 (--log_samples 로 진단).
# 2단계: 정상 변형 + humaneval_plus 로 코딩 타깃 모델들을 측정한다.
set -u
cd /workspace; mkdir -p results diag
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1
C="dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096"
SMALL="Qwen/Qwen2.5-Coder-1.5B-Instruct"

echo "===== 1단계: MBPP 변형 진단 $(date '+%T') ====="
for t in mbpp_plus_instruct mbpp_plus mbpp_instruct mbpp; do
  extra="--apply_chat_template --confirm_run_unsafe_code"
  case "$t" in mbpp_plus|mbpp) extra="--confirm_run_unsafe_code" ;; esac
  echo ">>> 진단 $t"
  python3.12 -m lm_eval --model vllm --model_args "pretrained=$SMALL,$C" \
    --tasks "$t" --limit 30 --output_path "diag/$t" --log_samples $extra > "diag/$t.log" 2>&1
  python3.12 - "$t" <<'PY'
import json,sys,glob
t=sys.argv[1]
f=sorted(glob.glob(f"diag/{t}/**/results_*.json",recursive=True))
if not f: print(f"   {t}: 결과 없음"); raise SystemExit
r=json.load(open(f[-1]))["results"][t]
sc={k:v for k,v in r.items() if k.startswith("pass")and"stderr"not in k}
print(f"   {t}: {sc}")
PY
done

echo "===== 2단계: 코딩 타깃 측정 $(date '+%T') ====="
run(){ local tag=$1 model=$2 tasks=$3
  local out=results/${tag}__code; [ -f "$out/DONE" ] && { echo "[$tag] 완료됨"; return; }
  mkdir -p "$out"; echo ">>> [$tag] $tasks  $(date '+%T')"
  python3.12 -m lm_eval --model vllm --model_args "pretrained=$model,$C" \
    --tasks "$tasks" --batch_size auto --output_path "$out" \
    --apply_chat_template --confirm_run_unsafe_code > "$out/run.log" 2>&1 \
    && touch "$out/DONE" && echo "<<< [$tag] 완료 $(date '+%T')" || { echo "<<< [$tag] 실패"; tail -10 "$out/run.log"; }
}
T="humaneval_instruct,humaneval_plus"
run coder-1.5b  Qwen/Qwen2.5-Coder-1.5B-Instruct  "$T"
run coder-7b    Qwen/Qwen2.5-Coder-7B-Instruct    "$T"
run coder-14b   Qwen/Qwen2.5-Coder-14B-Instruct   "$T"
run coder-32b   Qwen/Qwen2.5-Coder-32B-Instruct   "$T"
echo "===== 종료 $(date '+%F %T') ====="
