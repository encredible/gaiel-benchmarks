#!/bin/bash
# 전체 매트릭스를 순차 실행한다. 16GB 램에서 동시 실행은 스왑 스래싱을 유발하므로 반드시 순차.
# DONE 마커로 중단 후 재개가 된다.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
E="$ROOT/scripts/run_eval.sh"
echo "=== 전체 평가 시작 $(date '+%F %T') ==="
"$E" korean gaiel-1.5b          encredible/Gaiel-1.5B-Korean-Tuned-MLX
"$E" korean base-1.5b           mlx-community/Qwen2.5-1.5B-Instruct-4bit
"$E" coding base-coder-1.5b     mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
"$E" coding base-general-1.5b   mlx-community/Qwen2.5-1.5B-Instruct-4bit
"$E" coding gaiel-coding-1.5b   encredible/Gaiel-1.5B-Coding-Tuned-MLX
"$E" coding gaiel-korean-1.5b   encredible/Gaiel-1.5B-Korean-Tuned-MLX
"$E" korean gaiel-7b            encredible/Gaiel-7B-Korean-Tuned-MLX
"$E" korean base-7b             mlx-community/Qwen2.5-7B-Instruct-4bit
python3 "$ROOT/scripts/collect.py"
echo "=== 전체 평가 종료 $(date '+%F %T') ==="
