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

| 태스크 | 문항수 | 챗 템플릿 | 채점 |
|---|---|---|---|
| `humaneval_instruct` | 164 | **적용** | 생성 코드를 실행해 테스트 통과 여부 (pass@1) |
| `humaneval_plus`     | 164 | 미적용 | 동일 |
| `mbpp_plus`          | 378 | 미적용 | 동일 |

코드를 실제로 실행하므로 `--confirm-run-unsafe-code` 가 필요하다.

**챗 템플릿 규칙을 어기면 채점이 통째로 0 이 된다.** `*_instruct` 는 `--apply-chat-template` 가
필요하고, 비-instruct(`_plus`) 에 걸면 0 이 된다. 그래서 코딩 스위트는 두 패스로 나눠 돌린다
(`scripts/run_eval.sh` 참조).

**`mbpp_plus_instruct` 는 쓰지 않는다.** `extract_code` 필터 결함으로 전 모델 0.0 이 나온다.
과거 측정에서 5건이 이 상태로 남아 있으며, `results/` 의 원본 JSON 에는 기록이 보존돼 있으나
`summary.csv` 와 README 표에서는 제외한다.

**Δ 판정은 `humaneval_instruct` 기준이다.** 공개값 대조에서 신뢰성이 확인된 유일한 코딩 지표다
(`docs/REFERENCE.md`). `humaneval_plus`(−8~−13pt)·`mbpp_plus`(+5.6~+14.8pt) 는 공식값과
체계적으로 어긋나므로 보정 없이 결론에 쓰지 않는다.

## LogicKor (별도)

`inputs/logickor/questions.jsonl` — 42문항 멀티턴. GPT-4급 judge 모델이 필요해
비용이 발생하므로 기본 스위트에서 분리했다.
