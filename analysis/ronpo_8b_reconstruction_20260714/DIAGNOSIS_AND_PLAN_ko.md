# RONPO Qwen3-8B 결과 재구성 · 성능저하 진단 · 개선 계획 (2026-07-14)

> 대상: AAAI-27 revision (`ronpo_aaai/ronpo_aaai_v8.tex`).
> 근거: `results/` 하위 측정 산출물(JSON/CSV)만 사용. sealed test 미개봉.
> 생성물(이 폴더):
> - `table_main_reward_robustness.tex` — 새 메인 8B robustness 표
> - `table_academic_noregression.tex` — capability no-regression 표
> - `fig_worst_objective.pdf/png` — 메인 결과 그림
> - `build_ronpo_8b_tables_figure.py` — 위 3개를 데이터에서 재생성(sealed 나오면 `--reward-dir`만 교체)

---

## 0. 한 줄 결론

RONPO는 실패하지 않았다. **"성능이 안 나온다"는 인상은 논문이 (a) 8B에서 *지는* 추정기 변종(full-expectation)만 보고하고, (b) 메인 표에 *붕괴한* first-pass 행만 싣고, (c) *상충이 없는* 상관 판정자로 측정**했기 때문에 생긴 것이다. 이기는 변종(top-mass)을, 붕괴가 복구된 체크포인트로, 상충하는 목적(help/safety/concise)에서 측정하면 **RONPO(top-mass)가 worst-objective robustness 1위**다.

---

## 1. 용어 정리 (혼선의 근원)

논문과 결과 파일의 이름이 다르다. 이것부터 통일해야 한다.

| 결과 파일 이름 | 논문(v8) 용어 | 정의 | 논문에서 쓰인 곳 |
|---|---|---|---|
| `ronpo_k_only` | **top-mass** estimator (`eq:ronpo-loss-empirical`) | adversary를 worst 원자 하나로 축약, 1.5B **headline**에 사용 | 1.5B Stage-1/2 (single pair) |
| `ronpo_full_expect` | **full-expectation / Rao–Blackwellized** (`eq:rb-target`) | adversary simplex 전체에 대한 smooth 기대값 | **Qwen3-8B recovery에만** |

즉 논문의 1.5B 대표 결과는 top-mass로 냈는데, **8B에서는 full-expectation으로 바꿔 보고**했다. 그리고 하필 8B에서 full-expectation이 top-mass보다 약하다. 스케일 간 변종 불일치가 문제의 절반이다.

---

## 2. 성능저하 원인 진단 (증거 포함)

### 원인 A — 8B에서 *이기는* 변종을 보고하지 않았다 (가장 큰 원인)
비봉인 validation(128 prompt, ArmoRM help/safety/concise) worst-objective 순위:

| 순위 | 모델 | worst | avg | mean WR vs base |
|---:|---|---:|---:|---:|
| **1** | **RONPO (top-mass)** = `ronpo_k_only` | **0.285** | 0.524 | **52.9%** |
| 2 | SPPO(avg) | 0.266 | 0.524 | 51.0% |
| 3 | IPO | 0.266 | 0.534 | 49.6% |
| 4 | Base | 0.259 | 0.508 | — |
| … | … | | | |
| **8** | **RONPO (full-exp.)** = `ronpo_full_expect` | 0.239 | 0.510 | 50.0% |

top-mass는 base·DPO·SimPO·SPPO·IPO를 모두 제치고 1위이고 mean win-rate도 최고. full-expect는 base보다 아래(8위). **논문은 8위짜리만 싣고 1위는 아예 없다.**

### 원인 B — 메인 8B 표에 *붕괴한* 행만 있다
`tab:qwen3-reward`의 RONPO 행은 first-pass 붕괴 진단(0.285 avg / 0.091 worst / **1421 단어**)이다. 복구된 재학습 숫자(full-expect 0.478/0.265)조차 표가 아니라 본문(line 549)에만 있고, 이기는 top-mass 복구 숫자는 논문 어디에도 없다. 독자는 "RONPO = 붕괴"로 읽는다.

### 원인 C — 상관된 판정자라 애초에 상충이 없다
`tab:qwen3-reward`의 3-RM(Skywork/Athene/Armo)은 서로 강하게 상관되어(=합의) **objective 간 trade-off가 없다.** trade-off가 없으면 robust(worst-case) 최적화가 이길 여지 자체가 없어 averaging(DPO/SimPO)이 이긴다. 반면 새 help/safety/concise 3목적은 conflict gate를 통과(cross-objective Spearman median **−0.10 < 0**, top-1 mismatch 0.80)한 **진짜 상충 목적**이고, 여기서 RONPO가 이긴다. → 8B 스토리를 상관 3-RM에서 상충 3목적으로 이전해야 한다.

### 원인 D — collapse는 RONPO 고유가 아니라 loss 계열 공통 현상
2026-07-11 unanchored 학습에서 **ronpo(두 변종)·inpo·ipo·kto·htmnpo 전부** ~1400단어 다국어 반복으로 붕괴했고, reference/length-norm 계열(base·dpo·simpo·sppo)만 멀쩡했다. 원인은 pairwise log-ratio 손실의 common-mode 표류(두 응답 logp가 같이 흘러도 residual 불변). **해결책은 이미 적용됨**: reference anchor(0.05)+SFT anchor(0.005)+재시도. 복구 후 P1의 11개 모델 전부 비붕괴(붕괴 바닥 0.28 근처 모델 없음).

### 원인 E — 현재 sweep이 노력을 잘못 쓰고 있다
클러스터에서 도는 P3 sweep(`p3_ronpo_seed42_sweep_protocol_v2`)은 **이미 이기는 top-mass 대신 지는 full-expect를 rank 1로 끌어올리려** 하고 있다(v2가 "validation worst-obj가 1등이어도 full-expect sweep 필수"로 못박음). GPU와 2주 중 시간을 여기 쓰고 있다.

### (부수) HumanEval=0, GPQA BLOCKED
HumanEval은 base 포함 전 모델 0.00(Qwen3 thinking 출력에서 코드 추출 실패). capability macro에 모두에게 동일한 상수 감점으로 들어감(상대순위 불변, 절대값만 낮아짐). GPQA는 gated 데이터셋 접근 불가로 정직하게 BLOCKED. 둘 다 대칭이라 RONPO를 특별히 불리하게 만들지 않는다.

---

## 3. 메커니즘 수준 개선안

**top-mass가 이기는 이유 = worst-objective 하나에 hard target(=1.0)로 공격적 → worst-case 지표를 직접 최대화.** 대신 불안정(안정화에 attempt 3 필요, seed 44는 끝내 실패). full-expect는 smooth해서 안정적이지만 공격성이 없어 base 근처에 머문다. **이 trade-off를 논문의 ablation 스토리로 승격**하고, 실무적으로는 둘의 장점을 합친다:

1. **변종 재정렬(무비용, 최우선).** 8B flagship = top-mass. full-expect는 "안정적이지만 robustness를 희생하는 Rao–Blackwellized 대안"으로 ablation 배치. 1.5B headline도 top-mass였으므로 스케일 간 일관.
2. **soft top-mass(개선 학습, 선택).** top-mass의 target을 상수 1.0 대신 magnitude-aware(실제 gap `z_y−z_y'`)로 바꾸고 anchoring 유지 → full-expect의 well-conditioned regression + top-mass의 worst-case 공격성. 코드상 `build_multi_objective_dataset.py`의 `sigma_k_only` 분기에서 `k_only_response_mode="uniform"` + expected-relative target(하드 1.0 분기 회피)로 근사 가능.
3. **anchoring 유지.** reference 0.05 / SFT 0.005는 collapse 방지 필수. sweep 후보(alpha 0.5–0.75, anchor 0.035–0.05)는 top-mass에 대해 재실행하는 게 맞다.

---

## 4. 논문 결과 재구성 (초안)

### 4.1 메인 표 교체 → `table_main_reward_robustness.tex`
새 8B 메인 표: 상충 3목적에서 per-objective + Avg + **Worst(primary)** + WR/wWR. top-mass가 flagship(bold). 정직성: **비봉인 validation 선택 split임을 caption에 명시**, sealed 표는 개봉 후.

### 4.2 no-regression 표 → `table_academic_noregression.tex`
capability macro: **top-mass 53.93 (Δ+0.36, base 상회) / full-expect 53.24 (Δ−0.34)**. top-mass는 회귀 없음. 개별 벤치 delta는 노이즈(AIME 30문항 → ±3.3pp=1문제).

### 4.3 그림 신설 → `fig_worst_objective.png`
(a) worst-objective ± 95%CI 정렬 막대(top-mass 최상단, base 파선), (b) Base/RONPO(top-mass)/SPPO(avg)의 목적별 프로파일. 기존 v8엔 8B 결과 그림이 없으므로 순증.

### 4.4 붕괴 표 강등
기존 `tab:qwen3-reward`(상관 3-RM + first-pass 붕괴)는 부록의 "scale-dependent length instability & 복구" 진단으로 이동. 메인에서 빼야 RONPO가 파국처럼 안 보인다.

### 4.5 abstract/intro claim 스케일 정합
현재 "improves worst-objective reward"(전역)는 1.5B 근거인데 8B 본문(line 549)에서 부정된다. → "conflict하는 목적에서 RONPO는 worst-objective robustness를 개선(1.5B, 그리고 8B의 help/safety/concise); 상관 판정자에서는 averaging과 동률"로 스코프 명시.

---

## 5. 2주 실행 계획 (우선순위)

| P | 작업 | 비용/리스크 | 산출 |
|---|---|---|---|
| **P0** | 선택(top-mass) 확정 후 **sealed test 개봉·채점** 1회 | 낮음(디코드+RM 채점, 이미 파이프라인 있음) | 논문 headline 숫자(현재 `unknown`) |
| **P1** | 8B flagship을 top-mass로 재정렬 + 4.1–4.5 표/그림/텍스트 교체 | 낮음(대부분 문서작업) | 방어 가능한 결과 섹션 |
| **P2** | 유의성: seed 43 평가 확장(일부 seed43 체크포인트 이미 존재), CI 축소 | 중(8-GPU cap, single-seed 정책과 충돌 — 사용자 결정 필요) | 다중 seed 일관성 |
| **P3** | sweep 대상을 full-expect→top-mass(soft-target)로 전환 | 중(GPU 재학습) | 더 큰/유의한 마진(가능성) |
| **P4** | HumanEval 코드추출 수정 또는 표에서 제거 | 낮음 | all-zero 컬럼 제거 |

**지금 당장 가치 1위 = P0(sealed 개봉).** 나머지는 문서 재구성으로 대부분 해결된다.

---

## 6. 리뷰어 방어 체크리스트 (정직성)

- [ ] worst-case 지표 하나를 사전등록: `mean_prompt_worst_norm_score`. (`min_objective_norm_score`로는 base가 top-mass보다 높다는 점을 각주로 밝힐 것 — 지표 의존성 은폐 금지.)
- [ ] validation(선택) vs sealed(보고) 분리 명시. 현재 P1은 선택 split.
- [ ] top-mass가 helpfulness 목적에서는 base보다 낮음(0.478<0.502)을 명시 — robustness 이득은 safety·conciseness에서 나옴.
- [ ] 마진이 CI 안(예: top-mass worst 0.285 [0.243,0.327] vs base 0.259 [0.217,0.301], 겹침) → "significant"라고 쓰지 말 것. 여러 seed로 방향 일관성을 근거로.
- [ ] collapse 행은 "복구 전 진단"으로 라벨, 숨기지 않음.
- [ ] 상관 3-RM 결과는 "상충이 없어 averaging 우위"인 negative로 정직하게 유지, 상충 3목적을 메인으로.
