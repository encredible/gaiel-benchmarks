#!/bin/bash
# 파드에서 실행되는 벤치마크. lm-eval + vLLM (CUDA).
# MLX 결과와 태스크·버전을 동일하게 맞춰 비교 가능하게 한다.
set -u
cd /workspace
mkdir -p results
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1

KOREAN="kobest_boolq,kobest_copa,kobest_hellaswag,kobest_sentineg,kobest_wic,haerae"
CODING="humaneval_instruct,mbpp_plus_instruct"

run(){  # run <tag> <model> <suite> <tasks> [extra]
  local tag=$1 model=$2 suite=$3 tasks=$4; shift 4
  local out=results/${tag}__${suite}
  [ -f "$out/DONE" ] && { echo "[$tag/$suite] 완료됨 — 건너뜀"; return 0; }
  mkdir -p "$out"
  echo ">>> [$tag/$suite] $model  $(date '+%T')"
  lm_eval --model vllm \
    --model_args "pretrained=$model,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,trust_remote_code=True" \
    --tasks "$tasks" --batch_size auto --output_path "$out" "$@" \
    > "$out/run.log" 2>&1 \
    && touch "$out/DONE" && echo "<<< [$tag/$suite] 완료 $(date '+%T')" \
    || { echo "<<< [$tag/$suite] 실패"; tail -20 "$out/run.log"; }
}

echo "===== 벤치마크 시작 $(date '+%F %T') ====="
run base-1.5b Qwen/Qwen2.5-1.5B-Instruct korean "$KOREAN"
run base-1.5b Qwen/Qwen2.5-1.5B-Instruct coding "$CODING" --apply_chat_template --confirm_run_unsafe_code
run base-coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct coding "$CODING" --apply_chat_template --confirm_run_unsafe_code
run base-coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct korean "$KOREAN"
run base-7b Qwen/Qwen2.5-7B-Instruct korean "$KOREAN"
run base-7b Qwen/Qwen2.5-7B-Instruct coding "$CODING" --apply_chat_template --confirm_run_unsafe_code
run base-coder-7b Qwen/Qwen2.5-Coder-7B-Instruct coding "$CODING" --apply_chat_template --confirm_run_unsafe_code
echo "===== 벤치마크 종료 $(date '+%F %T') ====="
