# 평가 태스크 정의

모두 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.12 의
표준 태스크를 **수정 없이** 사용한다. 태스크 정의와 문항은 harness 및 원 데이터셋 저장소에 있다.

## 한국어

| 태스크 | 문항수 | 형식 | 채점 |
|---|---|---|---|
| `kobest_boolq`     | 1,404 | 예/아니오 | logprob 정확도 |
| `kobest_copa`      | 1,000 | 2지선다 | logprob 정확도 |
| `kobest_hellaswag` |   500 | 4지선다 | logprob 정확도 |
| `kobest_sentineg`  |   397 | 감성 이진 | logprob 정확도 |
| `kobest_wic`       | 1,260 | 문맥 내 의미 동일성 | logprob 정확도 |
| `haerae` (5종)     | 1,091 | 4지선다 | logprob 정확도 |

총 5,646 요청. **생성이 아니라 로그확률 비교**라 judge 모델이 필요 없고 비용이 0이다.

## 코딩

| 태스크 | 문항수 | 채점 |
|---|---|---|
| `humaneval_instruct`  | 164 | 생성 코드를 실행해 테스트 통과 여부 (pass@1) |
| `mbpp_plus_instruct`  | 378 | 동일 |

코드를 실제로 실행하므로 `--confirm-run-unsafe-code` 가 필요하다.

## LogicKor (별도)

`inputs/logickor/questions.jsonl` — 42문항 멀티턴. GPT-4급 judge 모델이 필요해
비용이 발생하므로 기본 스위트에서 분리했다.
