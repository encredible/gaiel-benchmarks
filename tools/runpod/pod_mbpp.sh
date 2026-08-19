#!/bin/bash
# mbpp_plus(비-instruct)가 정상 작동하는 유일한 변형으로 확인됐다.
# instruct 변형의 extract_code 필터가 챗 템플릿 출력에서 코드 추출에 실패한다.
set -u
cd /workspace
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1
C="dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096"
run(){ local tag=$1 model=$2
  local out=results/${tag}__mbpp; [ -f "$out/DONE" ] && { echo "[$tag] 완료됨"; return; }
  mkdir -p "$out"; echo ">>> [$tag/mbpp_plus] $(date '+%T')"
  python3.12 -m lm_eval --model vllm --model_args "pretrained=$model,$C" \
    --tasks mbpp_plus --batch_size auto --output_path "$out" --confirm_run_unsafe_code \
    > "$out/run.log" 2>&1 && touch "$out/DONE" && echo "<<< 완료 $(date '+%T')" \
    || { echo "<<< 실패"; tail -8 "$out/run.log"; }
}
echo "===== mbpp_plus 전체 $(date '+%T') ====="
run coder-1.5b Qwen/Qwen2.5-Coder-1.5B-Instruct
run coder-7b   Qwen/Qwen2.5-Coder-7B-Instruct
run coder-14b  Qwen/Qwen2.5-Coder-14B-Instruct
run coder-32b  Qwen/Qwen2.5-Coder-32B-Instruct
run base-1.5b  Qwen/Qwen2.5-1.5B-Instruct
run base-7b    Qwen/Qwen2.5-7B-Instruct
run base-32b   Qwen/Qwen2.5-32B-Instruct
echo "===== 종료 $(date '+%T') ====="
