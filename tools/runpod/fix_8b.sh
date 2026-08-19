#!/bin/bash
# Gaiel-8B 레포의 tokenizer_config.json 이 존재하지 않는 클래스(TokenizersBackend)를 지정해
# 토크나이저 로드가 실패한다. 가중치는 정상이므로 베이스 토크나이저로 대체해 평가한다.
cd /workspace
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
L3="NousResearch/Meta-Llama-3.1-8B-Instruct"
C="dtype=bfloat16,gpu_memory_utilization=0.90,max_model_len=4096,tokenizer=$L3"
run(){ local suite=$1 tasks=$2 extra=$3
  local out=results/gaiel-8b__${suite}; rm -f "$out/DONE"; mkdir -p "$out"
  echo ">>> [gaiel-8b/$suite] $(date '+%T')"
  python3.12 -m lm_eval --model vllm \
    --model_args "pretrained=encredible/Gaiel-8B-Korean-Tuned,$C" \
    --tasks "$tasks" --batch_size auto --output_path "$out" $extra > "$out/run.log" 2>&1 \
    && touch "$out/DONE" && echo "<<< 완료 $(date '+%T')" || { echo "<<< 실패"; tail -8 "$out/run.log"; }
}
run korean "kobest_boolq,kobest_copa,kobest_hellaswag,kobest_sentineg,kobest_wic,haerae" ""
run coding "humaneval_instruct" "--apply_chat_template --confirm_run_unsafe_code"
echo "===== 8B 재실행 종료 $(date '+%T') ====="
