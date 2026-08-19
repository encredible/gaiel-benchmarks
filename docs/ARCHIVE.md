# 폐기 기록 (Archive)

삭제하거나 비공개로 돌린 모델의 **학습 설정·측정 결과·폐기 사유**를 남긴다.
모델을 지우면 근거도 사라져 같은 실수를 반복하게 되므로, 아티팩트보다 기록을 오래 남긴다.

원본 측정 데이터는 `results/` 에, 요약은 `results/summary.csv` 에 그대로 있다.

---

## 1. 한국어 튜닝 회귀 — 네 체급 전부

측정한 **모든 체급에서 튜닝이 베이스보다 나빴다.** 목표였던 한국어조차 떨어졌고,
건드리지 않은 코딩 능력이 함께 파괴됐다.

| 모델 | 베이스 | 한국어 Δ | 코딩(HumanEval) Δ | 상태 |
|---|---|---|---|---|
| `Gaiel-1.5B-Korean-Tuned` | Qwen2.5-1.5B-Instruct | **−2.7** (49.6→47.0) | **−39.0** (57.3→18.3) | 삭제 |
| `Gaiel-7B-Korean-Tuned` | Qwen2.5-7B-Instruct | **−6.9** (65.3→58.4) | **−6.7** (86.0→79.3) | 삭제 |
| `Gaiel-8B-Korean-Tuned` | Llama-3.1-8B-Instruct | **−5.5** (61.8→56.4) | **−50.0** (70.7→20.7) | 회귀 확정 |
| `Gaiel-32B-Korean-Tuned` | Qwen2.5-32B-Instruct | **−6.8** (78.0→71.2) | **−9.1** (89.0→79.9) | 회귀 확정 |
| `Gaiel-72B-Korean-Tuned` | unsloth/qwen2.5-72b-bnb-4bit | 미측정 | 미측정 | 32B 와 동일 레시피 |

### 복원한 학습 설정

MLX 어댑터의 `adapter_config.json` 에서 추출했다 (1.5B/7B), 나머지는 PEFT 설정.

| | rank | alpha | alpha/r | LoRA 레이어 | 본 샘플 수 |
|---|---|---|---|---|---|
| 1.5B | 8 | 160 (scale 20.0) | **20** | 12~27 (28개 중 16) | 1,500 (batch 1 × 1500 iter) |
| 7B | 8 | 160 (scale 20.0) | **20** | 20~27 (**8개뿐**) | 1,200 (batch 1 × 1200 iter) |
| 32B | 16 | 32 | **2** (정상) | 전체, q/k/v/o | 미상 |
| 72B | 16 | 32 | **2** (정상) | 전체, 7개 모듈 | 미상 |

lr 은 1.5B/7B 모두 1e-4.

### 무엇을 배웠나

**1. 하이퍼파라미터는 원인이 아니었다.**
처음엔 `scale 20.0`(통상값의 80배)이 원인이라고 봤다. 그런데 **alpha/r 이 정상값 2 인 32B 도
−6.8/−9.1 로 회귀했다.** 스케일은 1.5B 코딩의 −39pt 같은 극단값은 설명하지만, 회귀 자체는
설명하지 못한다. 남은 공통 원인은 **학습 데이터**다.

**2. 튜닝은 그 모델이 특별히 잘하던 능력을 집중적으로 파괴한다.**
- 32B `wic` 84.0 → 64.9 (**−19.1**) — 32B 는 이 과제를 푸는 유일한 체급이었다
- 7B `sentineg` 92.9 → 67.0 (**−25.9**)
- 8B `boolq` 63.9 → 50.4 — **우연 수준으로 추락**
- 8B `haerae` 57.1 → 39.7 (**−17.4**)

**3. 코딩 붕괴는 catastrophic forgetting 의 서명이다.**
한국어 데이터만 먹이고 replay 를 넣지 않아, 건드리지 않은 능력이 지워졌다.
8B −50.0pt, 1.5B −39.0pt 는 오차가 아니라 파괴다.

**4. 베이스 점수를 먼저 찍지 않은 것이 가장 큰 실수였다.**
회귀는 처음부터 있었지만 비교 대상이 없어 몇 달간 발견되지 않았다.

### 다음 학습에 반영할 것

- `alpha/r ≤ 2`, rank 8~16
- **replay 20~30% 필수** — 코딩·일반 데이터를 섞지 않으면 재발한다
- 학습 **전에** 베이스 점수를 찍고, 학습 **중에** HumanEval 하락 ≤ 2pt 를 조기 종료 조건으로 건다
- 전체 레이어에 얕게 걸되(7B 는 마지막 8개 레이어에만 걸려 있었다), 델타는 작게

---

## 2. 빈 껍데기 — 가중치 0바이트

`README.md` 와 `.gitattributes` 만 있고 safetensors 가 하나도 없는 채로 공개돼 있었다.
"코딩 튜닝 모델" 로 표시돼 있었으나 실체가 없었다.

```
Gaiel-1.5B-Coding-Tuned      Gaiel-7B-Coding-Tuned    Gaiel-7B-Coding-Tuned-MLX
Gaiel-32B-Coding-Tuned       Gaiel-32B-Coding-Tuned-MLX
Gaiel-72B-Coding-Tuned       Gaiel-72B-Coding-Tuned-MLX
```

## 3. 중복 업로드

`Gaiel-110B-Coding-Tuned-MLX` — safetensors 12개의 LFS 해시가 `Gaiel-110B-Korean-Tuned-MLX` 와
**전부 동일**했다. 코딩 LoRA 가 fuse 되지 않은 채 한국어판이 코딩 모델로 게시된 상태였다.
한국어 원본이 남아 있어 데이터 손실은 없다.

**보존한 것**: `Gaiel-110B-Coding-Tuned` 는 1.14GB PEFT 어댑터로, 한국어 어댑터와 해시가 다른
**별개의 실제 학습 결과**다. 코딩 학습의 유일한 사본이라 삭제하지 않았다.
베이스는 Qwen1.5-110B-Chat (구세대).

---

## 4. 레포 자체의 결함

측정 과정에서 드러난, 성능과 무관한 구조적 문제들.

- **`Gaiel-8B-Korean-Tuned`**: `tokenizer_config.json` 이 `TokenizersBackend` 라는 **존재하지 않는
  클래스**를 지정한다. 표준 transformers/vLLM 으로 로드가 불가능하다. 측정 시 베이스
  토크나이저로 대체했다.
- **`Gaiel-1.5B/7B-Korean-Tuned`**: PEFT 어댑터가 아니라 MLX 학습 체크포인트였다.
  `adapter_config.json` 이 PEFT 설정이 아닌 MLX 학습 파라미터를 담고 있고,
  100 스텝마다 찍힌 중간 체크포인트 15개가 그대로 공개돼 있었다. CUDA 에서 로드 불가.
  평가를 위해 PEFT 로 변환했고(ΔW 최대오차 0.000e+00 로 등가 확인), 변환본도 함께 삭제했다.
- **`Gaiel-72B-Korean-Tuned`**: 어댑터는 `unsloth/qwen2.5-72b-instruct-bnb-4bit`(4bit)에서
  학습됐는데, 공개된 `-MLX` 병합본이 **어떤 베이스에 fuse 됐는지 기록이 없다.**
  fp16 에 얹었다면 학습·병합 조건 불일치 자체가 성능 저하 요인이다.

---

## 측정 환경

모든 수치는 RunPod 에서 아래 조합으로 bfloat16 측정했다.

```
GPU        : A40 48GB (1.5B/7B) · A100-SXM 80GB (8B/32B)
스택       : vllm 0.11.0 / torch 2.8.0+cu128 / transformers 4.56.2 / python 3.12
하네스     : lm-evaluation-harness (kobest 5종 + haerae, humaneval_instruct)
```

`mbpp_plus_instruct` 는 전 모델 pass@1 ≈ 0.0 으로 하네스 결함이 확인돼 표에서 제외했다
(같은 모델이 HumanEval 에서 57~89% 를 받는다). 원본 0.0 은 `results/*/lmeval_coding.json` 에 남아 있다.
