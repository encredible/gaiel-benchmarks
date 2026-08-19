#!/bin/bash
# humaneval_plus 는 비-instruct 태스크다. --apply_chat_template 를 걸면 채점이 0 이 된다.
set -u
cd /workspace
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1
C="dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096"
run(){ local tag=$1 model=$2
  local out=results/${tag}__hep; [ -f "$out/DONE" ] && { echo "[$tag] 완료됨"; return; }
  mkdir -p "$out"; echo ">>> [$tag/humaneval_plus] $(date '+%T')"
  python3.12 -m lm_eval --model vllm --model_args "pretrained=$model,$C" \
    --tasks humaneval_plus --batch_size auto --output_path "$out" --confirm_run_unsafe_code \
    > "$out/run.log" 2>&1 && touch "$out/DONE" && echo "<<< 완료 $(date '+%T')" \
    || { echo "<<< 실패"; tail -8 "$out/run.log"; }
}
echo "===== humaneval_plus (챗템플릿 없이) $(date '+%T') ====="
run coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct
run coder-7b   Qwen/Qwen2.5-Coder-7B-Instruct
run coder-14b  Qwen/Qwen2.5-Coder-14B-Instruct
run coder-32b  Qwen/Qwen2.5-Coder-32B-Instruct
run base-1.5b  Qwen/Qwen2.5-1.5B-Instruct
run base-7b    Qwen/Qwen2.5-7B-Instruct
run base-32b   Qwen/Qwen2.5-32B-Instruct
echo "===== 종료 $(date '+%T') ====="
