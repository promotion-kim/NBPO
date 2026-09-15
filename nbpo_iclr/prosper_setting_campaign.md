# PROSPER 세팅 캠페인 결과 — 2026-09-16

PROSPER(arXiv 2602.19041)의 실험 설정을 최대한 맞추고 집계 규칙만 바꿔
NBPO / NBPO-PW / Fixed-reference Nash-PW / PROSPER를 Qwen2.5-7B-Instruct에서
1 seed로 학습하고 Arena-Hard·AlpacaEval에서 비교했다. OpenAI API 호출 0건.
원고는 수정하지 않았다.

## 1. 최종 결과

**Arena-Hard** (gpt-4-0314 상대, 로컬 Qwen3-14B 판정, 양쪽 순서, 95% prompt bootstrap)

| arm | 집계 규칙 | 가중치 범위 | 승률 | CI95 | n | 길이(중위) |
|---|---|---|---|---|---|---|
| base Qwen2.5-7B-Instruct | — | — | 0.6108 | [0.5835, 0.6382] | 494 | 715.5 |
| PROSPER (MaxEntBW) | absolute maxmin | prompt별 | 0.6265 | [0.5977, 0.6559] | 494 | 734.0 |
| NBPO-PW | Nash | prompt별 | 0.6318 | [0.6045, 0.6596] | 495 | 718.0 |
| Fixed-reference Nash-PW | Nash (고정 참조) | prompt별 | 0.6334 | [0.6055, 0.6602] | 493 | 724.0 |
| NBPO | Nash | **전역** | **미측정** | 인증 실패 | — | — |

**AlpacaEval 2.0** (gpt4_1106_preview 상대, 동일 판정자)

| arm | 승률 | CI95 | n | 길이(중위) |
|---|---|---|---|---|
| base | 0.3979 | [0.3798, 0.4160] | 801 | 438.0 |
| PROSPER | 0.4134 | [0.3954, 0.4315] | 803 | 462.0 |
| NBPO-PW | 0.4080 | [0.3902, 0.4272] | 804 | 461.5 |
| Fixed-reference Nash-PW | 0.4055 | [0.3874, 0.4240] | 799 | 450.0 |
| NBPO (전역) | **미측정** | 인증 실패 | — | — |

## 2. 결론: 세 집계 규칙은 이 규모에서 구별되지 않는다

세 arm 모두 base를 넘는다 — Arena-Hard +1.6~2.3%p, AlpacaEval +0.8~1.6%p.
학습은 작동했다(PROSPER arm: 손실 48.1 → 0.93, Eq.(26) 잔차 4.92 → 0.86,
drift 0 → 4.22, rewards/margins 0 → 5.17).

그러나 **arm 간 순위는 의미가 없다.** 세 arm의 CI가 거의 완전히 겹치고 두
벤치마크에서 순위가 역전된다(Arena-Hard는 Fixed-ref-PW 1위, AlpacaEval은
PROSPER 1위). 291 prompt·8,148 pair 규모에서는 구별되지 않는다 — 이는
"차이가 없다"가 아니라 **"이 규모에서는 판정할 수 없다"**이다.

**길이 교란은 배제된다.** 세 arm의 응답 길이가 base와 거의 같다
(Arena-Hard 715 → 718~734, AlpacaEval 438 → 450~462).

## 3. 가중치 범위 축: 차이는 승률이 아니라 "해의 존재"에서 나타났다

| | 전역 NBPO | prompt별 Nash |
|---|---|---|
| λ | [18.74, 18.11, 18.04, 17.66] — 거의 균등 | prompt별, 표준편차가 평균보다 큼 |
| 유효 목표 수 (4개 중) | 붕괴 | **3.078 (train) / 3.093 (dev)** |
| 인증 | **실패** (정상성 13.224) | **전부 통과** |
| 학습 | 불가 | 완료 |

전역 arm은 `max_dual_calls` 200과 4000에서 모두 동일하게 실패했다:
`independent stationarity exceeds tolerance; probability floor is active;
inner optimizer lacks a certified solution/refinement`. dual은 1.2e-12 KKT
잔차로 수렴했으므로 예산 문제가 아니다. **허용치는 변경하지 않았고**
Safe 패널에서 PROSPER를 닫았던 절차대로 미측정으로 종결했다.

이는 5개 패널에서 전역 Nash 가중치가 균등으로 붕괴한다(effK/K ≥ 0.94)는
기존 측정과 일관되며, 이번에는 붕괴에 더해 인증조차 되지 않는다.

## 4. PROSPER 우위의 조건부성 — multiplier norm 의존

초기 probe에서 MaxEntBW가 697/697(100%) 인증한 것은 **L1=1.0에서**였다.
Nash arm에 맞춘 실제 scale에서는 다르다.

| weight L1 | MaxEntBW 인증 | 최소 질량 중위값 |
|---|---|---|
| 1.0 | 384/384 (100%) | 0.1179 |
| 10.0 | 384/384 (100%) | 0.0710 |
| 30.0 | 384/384 (100%) | 0.0234 |
| 100.0 | 377/384 (98.2%) | 3.29e-4 |
| **195.98** (matched) | **328/384 (85.4%)** | **2.55e-6** |

L1이 커지면 질량이 확률 하한으로 밀려 하한이 활성이 되고 정상성이 깨진다 —
전역 Nash 실패와 같은 기전이다. 또한 10자리 표현 가능성 단계에서 제외된
37 prompt 중 **29개가 PROSPER**에서 발생했다(Nash arm은 6~7개).
MaxEntBW가 Nash보다 넓게 정의되는 것은 맞지만 **그 우위는 norm에 의존하며,
같은 목표 scale에서는 Nash-PW가 100%로 앞선다.**

## 5. PROSPER 원문과 일치시킨 것 / 이탈

**일치**: WildChecklists, Qwen2.5-7B-Instruct base(rev a09a3545), 생성
T=0.8/top_p=0.9/2048 토큰 K=8(Table 5), 항목별 독립 판정(PSC, Figure 4의
5점 척도 JSON 프롬프트 원문 전사), 4096 토큰 초과 제외, 순서 대칭 후 역전
평균, Table 7 전부(batch 128 = 4×8×4GPU, LR 3e-7, wd 1e-6, AdamW ε 1e-8,
warmup 0.1, grad clip 1.0, max input 1024, max seq 2048, seed 555134),
Arena-Hard v0.1·AlpacaEval 2.0 공식 정적 문항과 baseline 답변.

**이탈 5건** (GPU 예산 및 API 비용 0 제약)

| 이탈 | PROSPER | 이번 캠페인 | 이유 |
|---|---|---|---|
| pair당 판정 수 | 10회 (순서별 5) | 2회 (순서별 1) | 채점에 16×H100 epoch당 50시간 |
| 학습 epoch | 2 (매 epoch 현재 정책 재생성) | 1 (base에서 off-policy) | 파이프라인 전체가 두 배 |
| 학습 prompt | 규모로 추정 2만+ | 291 | 위와 같음 |
| 상위 15% 점수격차 필터 | 적용 | 미적용 | 8,148 pair에 적용하면 16 step |
| 평가 판정자 | GPT-4 계열 | 로컬 Qwen3-14B | OpenAI 비용 0 |
| lr scheduler | 원문 미명시 | cosine (base recipe) | 원문에 값이 없음 |

따라서 **arm 간 비교는 유효하지만 절대 점수를 공개 리더보드와 비교하거나
인용할 수 없다.**

## 6. 규모 감소 경로

| 단계 | prompt | 원인 |
|---|---|---|
| 판정 대상 | 1,000 | 600 + 확장 400 |
| 채점 생존 | 697 | 4096 토큰 초과 pair 제외 (verdict 13% → prompt 30%) |
| train 분할 | 627 | 90/10 |
| 공통 인증 | 328 | MaxEntBW가 matched norm에서 85.4% |
| 표현 가능 | **291** | 10자리에서 0이 되는 질량을 가진 prompt 37개 |

pair 단위 제거는 불가했다 — 검증기가 prompt당 28 pair 전체를 요구한다.
제외는 세 arm의 **합집합**으로 적용해 세 arm의 prompt·pair 집합이 정확히
동일함을 검증했다(train 291·8,148, dev 31·868).

## 7. 자료

- 판정: `pool_judgments/pros2_psc/shard0..7`, 640,254 verdict, parse_rate 0.99987
- 점수: `scores/pros_scores_all/shard0..7` (단일 GPM teacher 기록)
- 목표: `targets/pros4_pw_nbpo_c`, `pros4_pw_fixedref_c`, `pros4_prosper`
- 인증 probe: `prosper/feasibility.json`, sweep: `prosper/maxmin_l1_sweep.json`
- 공통 집합: `prosper/common_certified.json`, 필터: `prosper/pair_filter.json`
- 평가: `eval_pairwise/{base,pros_pw_nbpo,pros_pw_fixedref,pros_prosper}_{arenahard,alpacaeval}`
- 전역 arm 미인증 기록: `targets/pros4_nbpo/train/solver_unresolved.json`
