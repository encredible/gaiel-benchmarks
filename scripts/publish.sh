#!/bin/bash
# 결과를 수집해 README 표를 갱신하고, 변경이 있을 때만 커밋·푸시한다.
# launchd 가 주기적으로 호출한다 — 평가가 끝나는 대로 저장소가 최신으로 유지된다.
cd /Users/K/omni-work/gaiel-benchmarks || exit 1
python3 scripts/collect.py || exit 1
git add -A
if git diff --cached --quiet; then echo "$(date '+%T') 변경 없음"; exit 0; fi
N=$(ls -d results/*/ 2>/dev/null | wc -l | tr -d ' ')
git -c user.name="encredible" -c user.email="descartes131@gmail.com" \
    commit -q -m "벤치마크 결과 갱신 ($(date '+%F %H:%M'), 모델 ${N}개)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push -q origin main && echo "$(date '+%T') 푸시 완료 (모델 ${N}개)"
