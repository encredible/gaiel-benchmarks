#!/bin/bash
# 8B / 32B 회귀 검증.
# 32B 는 어댑터가 전체 레이어 대상이라 병합 없이 vLLM LoRA 로 평가한다.
# 양쪽 팔(베이스/튜닝)에 반드시 같은 베이스·같은 정밀도를 쓴다 — 그래야 Δ 가 의미를 갖는다.
set -u
cd /workspace; mkdir -p results adapters
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
KOREAN="kobest_boolq,kobest_copa,kobest_hellaswag,kobest_sentineg,kobest_wic,haerae"
CODING="humaneval_instruct"
L3="NousResearch/Meta-Llama-3.1-8B-Instruct"     # 게이트 없는 미러
Q32="Qwen/Qwen2.5-32B-Instruct"

run(){ # run <tag> <suite> <tasks> <model_args...>
  local tag=$1 suite=$2 tasks=$3; shift 3
  local out=results/${tag}__${suite}
  [ -f "$out/DONE" ] && { echo "[$tag/$suite] 완료됨"; return 0; }
  mkdir -p "$out"; echo ">>> [$tag/$suite] $(date '+%T')"
  local extra=""
  [ "$suite" = "coding" ] && extra="--apply_chat_template --confirm_run_unsafe_code"
  python3.12 -m lm_eval --model vllm --model_args "$1" \
    --tasks "$tasks" --batch_size auto --output_path "$out" $extra > "$out/run.log" 2>&1 \
    && touch "$out/DONE" && echo "<<< [$tag/$suite] 완료 $(date '+%T')" \
    || { echo "<<< [$tag/$suite] 실패"; tail -12 "$out/run.log"; }
}

echo "===== 8B/32B 검증 시작 $(date '+%F %T') ====="
COMMON="dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096"

# --- 8B: 전체 병합본이므로 모델을 직접 지정 ---
run base-8b   korean "$KOREAN" "pretrained=$L3,$COMMON"
run base-8b   coding "$CODING" "pretrained=$L3,$COMMON"
run gaiel-8b  korean "$KOREAN" "pretrained=encredible/Gaiel-8B-Korean-Tuned,$COMMON"
run gaiel-8b  coding "$CODING" "pretrained=encredible/Gaiel-8B-Korean-Tuned,$COMMON"

# --- 32B: 베이스 + LoRA ---
python3.12 -c "
from huggingface_hub import snapshot_download
p=snapshot_download('encredible/Gaiel-32B-Korean-Tuned', local_dir='/workspace/adapters/g32b')
print('어댑터 다운로드:', p)"
run base-32b  korean "$KOREAN" "pretrained=$Q32,$COMMON"
run base-32b  coding "$CODING" "pretrained=$Q32,$COMMON"
run gaiel-32b korean "$KOREAN" "pretrained=$Q32,enable_lora=True,lora_local_path=/workspace/adapters/g32b,max_lora_rank=16,$COMMON"
run gaiel-32b coding "$CODING" "pretrained=$Q32,enable_lora=True,lora_local_path=/workspace/adapters/g32b,max_lora_rank=16,$COMMON"
echo "===== 검증 종료 $(date '+%F %T') ====="
