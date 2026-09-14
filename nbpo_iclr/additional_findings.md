# 표 슬롯이 없거나 미측정으로 남는 결과 — 2026-09-15

이 파일은 `templates/results.json`의 81개 슬롯에 들어가지 않는 결과, 그리고 측정하지
못해 `\pending`으로 남는 칸의 사유를 기록한다. 원고 문장은 수정하지 않는다.

## 1. PROSPER 적응 arm은 Safe 패널에서 미측정으로 남는다

`tab:stress_results`의 `prosperS/N/Wone/Wtwo/Min/Diff` 6칸은 `\pending`이다.

**사유.** exact target solver가 이 패널의 8Y+8Z tensor에서 인증에 실패한다. 6회 시도의
마지막 오류는 다음과 같다.

```
finite-pool certificate failed: independent stationarity exceeds tolerance;
probability floor is active; inner optimizer lacks a certified solution/refinement
```

즉 내부 최적화의 해가 **확률 하한에 활성인 상태**로 정상성 조건을 선언된 허용치 안에서
만족하지 못한다. 같은 solver가 NBPO / fixed-reference / utilitarian / maxmin /
BT-projected 다섯 표현에서는 모두 인증을 통과했으므로, 이는 데이터 로딩이나 배선 문제가
아니라 PROSPER 적응의 내부 문제에 국한된다.

**시도한 것.** 선언된 예산 항목인 `--max-dual-calls`를 기본값에서 4000으로 올려 재시도했고
같은 실패가 재현됐다. **허용치(`dual_tol`, `probability_floor`, 정상성 임계)는 바꾸지
않았다.** 허용치를 완화하면 값은 나오지만 그 값은 인증되지 않은 해이며, 이 캠페인의 규칙은
그런 숫자를 표에 넣지 않는 것이다.

**해석의 한계.** 이것은 "PROSPER가 이 데이터에서 성능이 낮다"는 결과가 **아니다.**
우리 구현의 적응 solver가 이 tensor에서 인증 가능한 해를 내지 못했다는 실행 사실이다.
원 논문 구현이나 다른 초기화에서는 다를 수 있다.

**자료.** `/work/uf4_20260910/jobs/runs/sub_solve_prosper/stdout.log` (6회 시도 전체),
큐 spec은 `jobs/queue/73_sub_solve_prosper.json.terminal_unmeasured`로 보존했다.

## 2. 학습된 arm의 응답 길이가 판정 상대보다 길다 — 승률 해석의 교란

`tab:stress_results`에는 길이 열이 없으므로 여기에 기록한다. test 패널(1,000 prompt)의
응답 길이:

| 응답 집합 | 중위 토큰 | 평균 토큰 | p10 | p90 |
|---|---|---|---|---|
| reference bank (판정 상대, base 4응답) | 34 | 205.6 | 8 | 600 |
| base fresh draw | 37 | 206.7 | 8 | 592 |
| fixed-reference Nash (seed 42) | 77 | 250.6 | 26 | 623 |

fixed-reference Nash의 승률(도움됨 0.6786, 무해성 0.6334)은 base fresh draw(0.5011,
0.4896)보다 크게 높지만, 같은 arm의 응답이 판정 상대보다 **길다.** 이런 pairwise judge는
일반적으로 긴 응답을 선호하므로, 격차의 일부는 길이 변화로 설명될 수 있다. 각 arm의 길이
분포를 이 표에 계속 추가하며, 길이를 통제한 재판정은 이번 캠페인의 선언된 범위가 아니다.

## 3. Safe 패널 구조가 기전 구별에 필요한 재료를 거의 주지 않는다

`tab:dataset_readiness`에 값으로 들어간 내용의 해석이다. 200 prompt · 8응답 ·
44,800 direct verdict에서 반복 가능한 목표 내 순환은 6,102 삼각형 중 6건(0.098%)이고,
단일 panel 순환 124건 중 118건이 두 번째 반복에서 사라진다. weak Condorcet 승자가 없는
graph는 400개 중 3.25%뿐이다. 통제 2×2에서 ΔM이 0을 배제한 것은 순환이 있는 cell뿐이었으므로,
이 데이터에서 adaptive 대 fixed 표현의 차이가 크게 나오기를 기대할 구조적 근거는 약하다.
이는 데이터 이름을 바꿔 해결되는 문제가 아니다.
