# 공개 수치 참고표

**이 표의 숫자는 우리가 측정한 것이 아닙니다.** 벤더·논문이 발표한 값이며, 하네스·프롬프트·샘플링이
우리와 다릅니다. `README.md` 의 측정 표와 **같은 열에 섞어서는 안 됩니다.**

용도는 하나 — **타깃 수치 파악**입니다. 우리 모델과의 Δ 비교에는 쓰지 않습니다.

## Qwen2.5-Coder-Instruct (공식)

출처: [Qwen2.5-Coder Technical Report](https://arxiv.org/html/2409.12186v3), Table 16 (Section 7.1)

| 모델 | HumanEval | HumanEval+ | MBPP | MBPP+ | BigCodeBench | LiveCodeBench |
|---|---|---|---|---|---|---|
| 0.5B-Instruct | 61.6 | 57.3 | 52.4 | 43.7 | 11.1 | 2.0 |
| 1.5B-Instruct | 70.7 | 66.5 | 69.2 | 59.4 | 32.5 | 6.1 |
| 3B-Instruct | 84.1 | 80.5 | 73.6 | 62.4 | 35.8 | 10.8 |
| 7B-Instruct | 88.4 | 84.1 | 83.5 | 71.7 | 41.0 | 18.2 |
| 14B-Instruct | 89.6 | 87.2 | 86.2 | 72.8 | 48.4 | 23.4 |
| **32B-Instruct** | **92.7** | **87.2** | **90.2** | **75.1** | **49.6** | **31.4** |

추가 공개값 (32B-Instruct): Aider 73.7, McEval 65.9, MdEval 75.2
— 출처: [Qwen2.5-Coder 블로그](https://qwenlm.github.io/blog/qwen2.5-coder-family/)

## Qwen3-Coder-30B-A3B-Instruct

출처: [모델 카드](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct),
[SGLang 문서](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3-Coder)

| 항목 | 값 |
|---|---|
| 총 파라미터 / 활성 | 30.5B / **3.3B** (128 experts, 8 활성) |
| 컨텍스트 | 262,144 (Yarn 확장 시 1M) |
| LiveCodeBench v6 | **66.0** (베이스 Qwen3-30B-A3B 는 57.4) |
| HumanEval / MBPP | 모델 카드에 수치 미공개 |

활성 3.3B 로 LiveCodeBench 66.0 은 Qwen2.5-Coder-32B(31.4)의 두 배가 넘는다.
다만 **LiveCodeBench 버전이 달라 직접 비교는 성립하지 않는다** (v6 vs 미표기).

---

# 우리 측정과의 대조 — 하네스 보정

같은 모델을 우리가 lm-eval 로 잰 값과 공식값의 차이. **하네스가 얼마나 결과를 바꾸는지**를 보여준다.

| 모델 | HumanEval 공식→측정 | HumanEval+ 공식→측정 | MBPP+ 공식→측정 |
|---|---|---|---|
| Coder-1.5B | 70.7 → 66.5 (**−4.2**) | 66.5 → 53.0 (**−13.5**) | 59.4 → 66.4 (**+7.0**) |
| Coder-7B | 88.4 → 88.4 (**+0.0**) | 84.1 → 76.2 (**−7.9**) | 71.7 → 82.3 (**+10.6**) |
| Coder-14B | 89.6 → 90.9 (**+1.3**) | 87.2 → 79.3 (**−7.9**) | 72.8 → 87.6 (**+14.8**) |
| Coder-32B | 92.7 → 91.5 (**−1.2**) | 87.2 → 74.4 (**−12.8**) | 75.1 → 80.7 (**+5.6**) |

## 읽는 법

**1. `humaneval_instruct` 는 신뢰할 수 있다.** 편차 ±4.2pt, 7B 는 88.4 로 정확히 일치한다.
챗 템플릿을 적용하는 instruct 태스크 설정이 공식 평가와 사실상 같다는 뜻이다.

**2. `humaneval_plus` 는 체계적으로 낮게 나온다 (−8~−13pt).**
비-instruct 태스크라 챗 템플릿 없이 **완성(completion) 형식**으로 돌렸는데, 대상이 instruct 모델이다.
지시 튜닝된 모델을 완성 모드로 쓰면 손해를 본다. 공식은 챗 형식 + 코드 추출로 평가한다.

**3. `mbpp_plus` 는 체계적으로 높게 나온다 (+5.6~+14.8pt).**
lm-eval 의 few-shot 프롬프트가 점수를 올리는 것으로 보인다. 공식과 프롬프트 설정이 다르다.

## 이 대조가 뒤집은 결론

우리 측정만 보면 **Coder-14B(85.9)가 Coder-32B(82.2)를 이긴다**는 결론이 나왔고,
실제로 그렇게 보고했다. **이는 틀렸다.**

14B 우위는 전적으로 `humaneval_plus` 와 `mbpp_plus` 에서 나왔는데, 위에서 보듯 그 둘이
공식값과 가장 크게 어긋나는 지표다. 공식값 기준으로는 32B 가 모든 지표에서 14B 이상이다
(HumanEval+ 87.2 = 87.2, MBPP+ 75.1 > 72.8, LiveCodeBench 31.4 > 23.4).

**교훈: 신뢰도가 검증되지 않은 지표로 모델 선택 결론을 내리면 안 된다.**
새 지표를 추가할 때는 공개값이 있는 모델로 먼저 보정(calibration)한 뒤 써야 한다.

## Gaiel 회귀 결론에는 영향 없다

Gaiel 모델의 Δ 는 전부 `humaneval_instruct` 하나로 계산했고, 그 지표는 위에서 신뢰성이
확인된 유일한 코딩 지표다. 한국어 스위트(KoBEST/HAE-RAE)는 logprob 기반이라 프롬프트
형식 문제에서 자유롭다. **네 체급의 회귀 결론은 그대로 유효하다.**
