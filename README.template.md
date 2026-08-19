# Gaiel Benchmarks

[Gaiel](https://huggingface.co/encredible) 모델군의 벤치마크 결과를 **원본 출력 그대로** 공개합니다.
입력·스크립트·결과가 모두 이 저장소에 있어 누구나 재현하고 검증할 수 있습니다.

> **측정된 모델: {{N_MODELS}}개.** 아직 채워지지 않은 칸은 측정 전이며, 측정되지 않은 항목에 대해
> 어떤 주장도 하지 않습니다.

## 핵심 원칙

1. **튜닝 모델은 항상 베이스와 나란히 측정합니다.** 튜닝이 성능을 올렸는지 내렸는지는
   베이스 점수 없이는 판별할 수 없습니다. Δ 열이 그 답입니다.
2. **동일 양자화 조건에서 비교합니다.** 모두 MLX 4-bit입니다. 양자화가 다르면 비교가 무의미합니다.
3. **측정 환경을 고정합니다.** 모든 수치는 RunPod A40 에서 `vllm 0.11.0 / torch 2.8.0+cu128 /
   transformers 4.56.2 / lm-eval` 로 bfloat16 측정한 값입니다. 백엔드나 정밀도가 다르면
   같은 표에 올리지 않습니다.
4. **원본 출력을 가공하지 않습니다.** `results/` 안의 JSON은 lm-eval이 뱉은 그대로입니다.
5. **불리한 결과도 그대로 싣습니다.** 회귀는 ⚠️ 로 표시됩니다.

## 결과

{{KOREAN_TABLE}}

{{CODING_TABLE}}

단위는 % 입니다. 한국어는 정확도(logprob), 코딩은 pass@1(생성 코드 실제 실행)입니다.

## 폐기 기록

삭제·비공개 처리한 모델의 학습 설정, 측정 결과, 폐기 사유는 **[docs/ARCHIVE.md](docs/ARCHIVE.md)** 에
남겨두었습니다. 모델을 지우면 근거도 사라지므로, 아티팩트보다 기록을 오래 남깁니다.

## 재현 방법

```bash
git clone https://github.com/encredible/gaiel-benchmarks
cd gaiel-benchmarks
pip install "lm-eval>=0.4.5" mlx-lm
./scripts/run_all.sh          # 전체 매트릭스 (순차, 재개 가능)
python3 scripts/collect.py    # 결과 → 표 갱신
```

개별 실행:

```bash
./scripts/run_eval.sh korean gaiel-7b encredible/Gaiel-7B-Korean-Tuned-MLX
./scripts/run_eval.sh coding base-coder-1.5b mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
```

Apple Silicon + MLX 기준입니다. 한국어 스위트는 로그확률 비교라 judge 모델이 필요 없어 **비용이 0원**입니다.
코딩 스위트는 생성된 코드를 실제로 실행하므로 격리된 환경에서 돌리기를 권합니다.

## 저장소 구조

```
inputs/
  models.yaml           평가 대상 모델과 베이스 짝
  tasks/README.md       태스크 정의·문항수·채점 방식
  logickor/questions.jsonl   LogicKor 42문항 (멀티턴)
scripts/
  run_eval.sh           단일 모델·스위트 실행
  run_all.sh            전체 매트릭스
  collect.py            결과 수집 → summary.csv + 이 README
results/
  <tag>/eval_*          lm-eval 원본 출력 (무가공)
  <tag>/*.log           실행 로그
  summary.csv           전체 점수 평면화
```

## 알려진 한계

- **`mbpp_plus_instruct` 는 표에서 제외했습니다.** 모든 모델이 pass@1 ≈ 0.0 을 기록했는데,
  같은 모델이 HumanEval 에서 57~88% 를 받습니다. 이는 모델 능력이 아니라 `extract_code` 필터가
  챗 템플릿 출력 형식과 맞지 않아 코드 추출에 실패한 **하네스 결함**입니다. 원본 JSON 에는
  0.0 이 그대로 남아 있으니 `results/*/lmeval_coding.json` 에서 직접 확인할 수 있습니다.
  수정 후 다시 채울 예정입니다.
- **LogicKor는 기본 스위트에서 제외**했습니다. GPT-4급 judge 모델이 필요해 비용이 발생하고,
  judge 모델 버전에 따라 점수가 흔들려 재현성이 떨어집니다. 문항은 `inputs/logickor/` 에 두었습니다.
- 32B/72B는 로컬 램 제약으로 측정이 지연되고 있습니다.
- `sample_len` 이 전체 문항수보다 작으면 부분 실행입니다. `results/summary.csv` 의 `n` 열로 확인하세요.

## 라이선스

결과 데이터는 CC BY 4.0, 스크립트는 MIT.
