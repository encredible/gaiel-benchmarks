#!/bin/bash
# 검증된 버전 조합으로 환경 구성 후 벤치마크 실행
cd /workspace
echo "=== 설치 $(date '+%T') ==="
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
pip install -q vllm==0.11.0 "transformers==4.56.2" lm-eval hf_transfer 2>&1 | tail -2
python3.12 -c "import torch,vllm,transformers;print('OK',torch.__version__,torch.version.cuda,vllm.__version__,transformers.__version__,torch.cuda.is_available())"
echo "=== 벤치마크 시작 ==="
exec ./pod_big.sh
