# RunPod 운영 기록

이 저장소의 모든 수치는 RunPod GPU 에서 나왔다. 환경을 세우며 막힌 지점과 해결책을 남긴다.
같은 함정을 다시 밟지 않기 위한 문서다.

## 검증된 버전 조합

이 조합 외에는 전부 실패했다.

```
이미지   : runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204
python   : python3.12   ← python3 는 3.10 을 가리킨다. 반드시 python3.12 로 호출
torch    : 2.8.0+cu128
vllm     : 0.11.0
transformers : 4.56.2
lm-eval  : 0.4.12
```

설치 순서 (순서를 바꾸면 깨진다):

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.11.0 "transformers==4.56.2" lm-eval hf_transfer
python3.12 -c "import torch,vllm,transformers;print(torch.__version__,vllm.__version__,transformers.__version__,torch.cuda.is_available())"
```

## 실패 사례와 원인

### 1. `dockerArgs` 로 시작 명령을 덮어쓰면 파드가 뜨지 않는다
`podFindAndDeployOnDemand` 에 `dockerArgs` 를 주면 이미지의 기본 CMD 가 대체되고,
RunPod 자체 init 이 죽어 **`runtime` 이 영원히 `null`** 이 된다. 상태는 `RUNNING` 으로 보이지만
SSH 포트가 열리지 않는다. 37분간 과금만 됐다.

**해결**: `dockerArgs` 를 주지 않는다. `PUBLIC_KEY` 환경변수와 `ports: "22/tcp"` 로 파드를 정상
기동시킨 뒤 SSH 로 스크립트를 올려 실행한다.

### 2. 컨테이너 디스크 400GB 는 프로비저닝이 끝나지 않는다
같은 `runtime: null` 증상. 13분 대기 후 폐기했다. **200GB 이하로 잡으면 1~2분에 뜬다.**

> **규칙**: 파드가 `RUNNING` 인데 5분 안에 `portMappings` 가 안 나오면 기다리지 말고 폐기한다.

### 3. 최신 vLLM 은 드라이버보다 앞선 CUDA 를 요구한다
`pip install vllm` 은 0.27.x → torch 2.13(CUDA 13) 을 끌어오는데, 파드 드라이버는 CUDA 12.8 이다.
`RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`.
**버전을 반드시 고정한다.**

### 4. `--force-reinstall` 은 torch 를 통째로 날린다
복구에 시간이 더 걸린다. 쓰지 않는다.

### 5. `pip` 과 `python3` 가 다른 인터프리터를 가리킨다
`pip list` 에는 torch 가 보이는데 `python3 -c "import torch"` 는 ModuleNotFound 가 난다.
pip 은 python3.12 에, `python3` 는 3.10 에 연결돼 있다. **`python3.12 -m lm_eval` 로 호출한다.**

### 6. SSH 로 보낸 `pkill -f <패턴>` 이 자기 자신을 죽인다
`ssh host 'pkill -f lm_eval'` 은 그 명령 문자열에 `lm_eval` 이 들어 있어 **ssh 가 띄운 셸까지
매칭해 스스로 종료**한다. 출력 없이 조용히 실패한다. `pgrep -f` 를 쓰는 대기 루프도 같은 이유로
무한 대기에 빠진다.

**해결**: 명령을 파일로 만들어 `scp` 로 올린 뒤 실행한다.

### 7. `urllib` 은 Cloudflare 에 차단된다 (error 1010)
RunPod GraphQL API 를 파이썬 `urllib` 로 호출하면 403 이 난다. **`curl` 을 쓴다.**

## 비용 관리

- **파드 종료를 코드 경로마다 보장한다.** 스크립트 안에 `trap EXIT` 로 self-terminate 를 걸고,
  그것이 실패할 경우를 대비해 로컬에 하드 데드라인 워치독을 별도로 둔다.
- 유휴 파드 1대가 하루에 이 프로젝트 전체 예산보다 큰 돈을 태울 수 있다.
- 잔액·가동 파드 확인:
  ```bash
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"query":"query { myself { clientBalance currentSpendPerHr pods { id desiredStatus } } }"}'
  ```

## 실제 사용 GPU 와 비용

| 작업 | GPU | 시급 | 비고 |
|---|---|---|---|
| 1.5B/7B 베이스+Gaiel 전체 | A40 48GB | $0.44 | 약 $1.3 |
| 8B/32B 검증 | A100-SXM 80GB | $1.59 | 약 $1.6 |
| 코딩 스위트 확충 | A100-SXM 80GB | $1.59 | 약 $4.1 |
| 실패한 파드 2대 | A40 / A100 | — | 약 $0.6 낭비 |

A40 COMMUNITY 는 재고가 자주 없다. SECURE 로 폴백하면 시급이 $0.35 → $0.44 로 오른다.

## lm-eval 태스크별 챗 템플릿 규칙

**이 규칙을 어기면 점수가 0 이 나온다.** 30문항 진단으로 확인했다 (동일 모델).

| 태스크 | `--apply_chat_template` | 결과 |
|---|---|---|
| `humaneval_instruct` | **필요** | 정상 (공식값과 ±4pt) |
| `humaneval_plus` | **걸면 안 됨** | 걸었을 때 0.0 |
| `mbpp_plus` | **걸면 안 됨** | 76.7 (정상) |
| `mbpp_plus_instruct` | 걸어도 실패 | 0.0 — `extract_code` 필터 결함, 사용 금지 |

코드 실행 채점에는 `--confirm_run_unsafe_code` 와 `HF_ALLOW_CODE_EVAL=1` 이 함께 필요하다.
