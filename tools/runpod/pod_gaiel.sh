#!/bin/bash
# 변환된 PEFT 어댑터를 베이스에 병합한 뒤 평가한다.
# vLLM 의 LoRA 경로 대신 병합을 쓰는 이유: layers_to_transform(일부 레이어만 LoRA)을
# vLLM 이 존중한다는 보장이 없다. 병합은 PEFT 가 처리하므로 의미가 정확히 보존된다.
set -u
cd /workspace; mkdir -p results merged
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
KOREAN="kobest_boolq,kobest_copa,kobest_hellaswag,kobest_sentineg,kobest_wic,haerae"
CODING="humaneval_instruct,mbpp_plus_instruct"

merge(){ # merge <out> <base> <adapter>
  [ -d "merged/$1" ] && { echo "[$1] 병합본 존재"; return 0; }
  echo ">>> 병합 $1 = $2 + $3"
  python3.12 - "$1" "$2" "$3" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
out, base, adp = sys.argv[1], sys.argv[2], sys.argv[3]
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, adp)
n_before = sum(p.abs().sum().item() for n,p in m.named_parameters() if "lora_A" in n)
m = m.merge_and_unload()
m.save_pretrained(f"merged/{out}", safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(f"merged/{out}")
print(f"   병합 완료 (LoRA A 가중치 합 {n_before:.2f} — 0이면 어댑터가 비어있다는 뜻)")
PY
}
run(){ # run <tag> <model> <suite> <tasks> [extra]
  local tag=$1 model=$2 suite=$3 tasks=$4; shift 4
  local out=results/${tag}__${suite}
  [ -f "$out/DONE" ] && { echo "[$tag/$suite] 완료됨"; return 0; }
  mkdir -p "$out"; echo ">>> [$tag/$suite] $(date '+%T')"
  python3.12 -m lm_eval --model vllm \
    --model_args "pretrained=$model,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096" \
    --tasks "$tasks" --batch_size auto --output_path "$out" "$@" > "$out/run.log" 2>&1 \
    && touch "$out/DONE" && echo "<<< [$tag/$suite] 완료 $(date '+%T')" \
    || { echo "<<< [$tag/$suite] 실패"; tail -15 "$out/run.log"; }
}

pip install -q peft 2>&1 | tail -2
echo "===== Gaiel 평가 시작 $(date '+%F %T') ====="
merge gaiel-1.5b Qwen/Qwen2.5-1.5B-Instruct encredible/Gaiel-1.5B-Korean-Tuned-PEFT
run gaiel-1.5b /workspace/merged/gaiel-1.5b korean "$KOREAN"
run gaiel-1.5b /workspace/merged/gaiel-1.5b coding "$CODING" --apply_chat_template --confirm_run_unsafe_code
merge gaiel-7b Qwen/Qwen2.5-7B-Instruct encredible/Gaiel-7B-Korean-Tuned-PEFT
run gaiel-7b /workspace/merged/gaiel-7b korean "$KOREAN"
run gaiel-7b /workspace/merged/gaiel-7b coding "$CODING" --apply_chat_template --confirm_run_unsafe_code
echo "===== Gaiel 평가 종료 $(date '+%F %T') ====="
