# RunPod 실행 스크립트

실제로 결과를 만든 스크립트 원본. 파드에 올려 SSH 로 실행했다.

| 파일 | 용도 |
|---|---|
| `setup_big.sh` / `setup_coding.sh` | 검증된 버전 조합 설치 후 벤치마크 실행 |
| `pod_bench.sh` | 1.5B/7B 베이스 매트릭스 (한국어+코딩) |
| `pod_gaiel.sh` | PEFT 어댑터 병합 후 Gaiel 1.5B/7B 평가 |
| `pod_big.sh` | 8B/32B 검증 (32B 는 vLLM LoRA) |
| `fix_8b.sh` | Gaiel-8B 토크나이저 우회 재실행 |
| `pod_coding.sh` | MBPP 변형 진단 + Coder 계열 측정 |
| `pod_mbpp.sh` / `pod_hep.sh` | mbpp_plus / humaneval_plus (챗 템플릿 없이) |

운영 주의사항은 [../../docs/RUNPOD.md](../../docs/RUNPOD.md) 참조.
