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

## 4. Table 37 세 행의 D는 같은 이름이지만 같은 양이 아니다

`tab:dataset_readiness`의 D 열은 세 행에서 이렇게 정의된다.

| 행 | D의 단위 | 정의 |
|---|---|---|
| UF / fresh fixed rubrics | 선언된 4개 목표 | 6개 목표쌍(C(4,2))에 대해, 두 목표가 모두 strict 방향을 낸 (prompt, 응답쌍) cell 중 반대로 정렬한 비율의 **평균** |
| SafeRLHF / help–safe | 선언된 2개 목표 | 목표쌍이 1개뿐이므로 위 정의와 정확히 같은 수 (K=2에서 bit 단위로 일치함을 확인) |
| WildChecklists / native items | **prompt별 checklist 항목** | 각 prompt 안에서 그 prompt 자신의 항목쌍에 대해, 두 항목이 모두 strict 방향을 낸 (항목쌍, 응답쌍) cell 중 반대로 정렬한 비율 |

앞의 두 행은 **고정된 전역 목표**에 대한 값이고, Wild 행은 **prompt 내부의 항목 불일치**다.
Wild 행에서 prompt P의 항목 k와 prompt Q의 항목 k는 서로 다른 기준이므로 같은 party로
묶지 않는다(template도 "report prompt-level variation rather than pooling checklist
positions as global parties"라고 요구한다). 따라서 Wild 행의 D를 위 두 행의 D와 **칸 대
칸으로 비교하면 안 된다.** 같은 이유로 Wild 행의 γ*와 Target TV는 `n/a`이다 — 고정된 목표
정체성이 없으면 aggregate finite game을 정의할 대상 자체가 없다.

Wild 행의 D는 두 가지 가중으로 계산해 artifact에 모두 기록한다. 표에 들어가는 값은
cell을 pooling한 값(앞 두 행과 같은 "strict하게 풀린 cell의 비율" 형태)이고, prompt마다
항목 수가 다르므로(2–12개) prompt별 비율의 평균과 표준편차도 함께 기록한다. 항목이 많은
prompt가 pooled 값을 지배할 수 있으므로 두 수가 크게 다르면 그 사실을 보고한다.

**구현 근거.** `analyze_wild.py`는 edge 판정(p_hat, δ=0.05 tie band), oriented 3-cycle
열거, Condorcet 판정을 `analyze_screen.py`에서 **import**한다. 재구현하지 않았으므로 세 행의
정의가 서로 어긋날 수 없다. 다른 점은 데이터와 기준의 단위뿐이다. `analyze_wild.py`는
손으로 계산한 합성 패널(`code/wild_selftest.py`)로 검증했다: C=1/336, D=29/84,
no-weak-Condorcet=1/6을 정확히 재현한다. K=2 일반화가 기존 Safe 행을 바꾸지 않는지도
확인했다 — D=0.12688927943760983로 기록된 값과 bit 단위로 동일하다.
