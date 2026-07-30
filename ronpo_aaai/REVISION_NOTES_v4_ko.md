# main_v4 리비전 노트 (2026-07-23 야간)

`main_v3.tex` → `main_v4.tex`. 26쪽, 0 undefined ref/cite, 0 overfull, em-dash 없음.
기존 내용 최대 보존, LLM Experiments 섹션만 재구성.

## 받은 리뷰 3개 반영

**리뷰1 (worst-reward를 primary story로):**
LLM Experiments를 "objective가 충돌할 때 worst objective를 개선한다"는 단일 주장으로 재구성.
SafeRLHF(충돌 세팅)를 **첫 번째 primary subsection**으로 올림. Worst-objective 표(Table 2)에 더해
**theory-aligned pairwise floor** $\E_x[\min_{k,a}P_k]$를 Table 3로 직접 보고 (kekim #417 해소).
RONPO가 floor 최고(0.0449), averaging/single-oracle baseline 대비 유의하게 우위,
IPO와는 point-estimate 우위이나 통계적 동률(정직하게 명시). Abstract·Intro·Conclusion도 이 프레이밍으로 재작성.

**리뷰2 (SimPO 등 baseline이 특정 objective를 exploit함을 정성적으로):**
Table 4 신설. SafeRLHF 유해 프롬프트 2개(seed 42 실제 생성물)에서
SimPO는 무뚝뚝한 거절로 harmlessness만 획득(help −5.8, harm 19.1),
RONPO는 거절+합법적 대안 제시로 안전하면서 helpful(help 20.4, harm 18.0).
held-out 패널에서 이 패턴이 212개 프롬프트에서 성립함을 명시.

**리뷰3 (Figure 3 분리: stage별 개선 figure + RONPO vs RMOD·baseline figure):**
- Figure 2 `saferlhf_frontier.pdf` (신규): help–harm 평면에서 RONPO vs RMOD(K=1→16) vs 모든 baseline.
  RONPO가 우측 최helpful, averaging/offline은 safe-but-unhelpful 코너, RMOD sweep은 RONPO에 미달.
- Figure 3 `saferlhf_stage_traj.pdf` (신규): RONPO가 stage 1→4로 helpfulness 단조 개선, κ 클수록 harmlessness 유지.
- figure 생성 스크립트: `results/v4_integration_20260723/figs/make_figs.py`

## ⚠️ 야간에 발견한 CRITICAL 이슈 (반드시 확인)

`main_v3`의 UF-1.5B stage-2 표 "RONPO 0.7025 best"는 **정규화 풀 아티팩트**입니다.
Raw reward(정규화 무관)로 확인: INPO-avg·MaxMin이 세 objective 전부에서 RONPO를 압도
(Skywork raw 8.9 vs RONPO 6.5 vs base 4.9). main_v3 표는 강한 averaging baseline을 풀에서 제외했기 때문에
RONPO가 최고로 보였을 뿐이며, 리뷰어가 공정 비교를 돌리면 reject 사유입니다.
근거: `results/saferlhf_stage4_joint_3seed_20260723/`, B200 `novelty_defense_20260723/per_policy_scores/raw_reward_summary.json`.
상세: 메모리 `ronpo-uf-normalization-landmine.md`.

**대응(main_v4):** UF-1.5B를 정직하게 **no-conflict 통제**로 강등.
"세 general RM은 concordant → averaging이 이미 near-optimal → RONPO는 base를 손상시키지 않지만 이점도 없음"으로 재서술.
UF 표 caption에 "pool-dependent, superiority claim 아님" 명시. INPO-avg/MaxMin이 raw에서 앞선다고 본문에 명기.
이건 오히려 **더 정직하고 강한 스토리**입니다: RONPO의 이점은 objective가 진짜 충돌할 때(SafeRLHF)만 발현되며,
이는 리뷰1(worst-reward가 핵심)과 정확히 부합. SafeRLHF에서 RONPO는 raw help 6.6(trained 최고),
INPO-avg는 2.1로 붕괴 → conflict에서 averaging은 safe-but-unhelpful로 무너짐이 실증됨.

## B200 실험 판단 (미실행, 레시피 첨부)

리뷰 3개는 **신규 GPU 학습 없이** 기존 데이터 재분석·figure·서술로 충족됨(위 참조).
"장점 극대화" 보너스 실험은 검토했으나 **야간 무인 실행하지 않음**:
SafeRLHF 학습은 P4 고정 pair + 4-stage 반복(decode→score→pair→precompute→train) 8B 파이프라인으로
fix_log에 실패가 다수 기록된 fragile 구조. 9am까지 무인으로 완료·검증·통합할 확률이 낮고,
subtly-wrong baseline은 없는 것보다 나쁨. 완결된 deliverable를 fragile run에 걸지 않는 것이 냉철한 판단.

**첨부(attended 실행 권장, 우선순위순):**
1. **MaxMin-RLHF on SafeRLHF** — conflict 세팅에 빠진 유일한 worst-objective baseline. 핵심 novelty(joint (k,a) > k-only weights) 직접 검증.
   레시피: UF의 `build_maxmin_avg_pairs.py`(w∝exp(−λ·stage1_perf)) 방식으로 help/harm 2-objective pair 재구성 후
   기존 SafeRLHF 4-stage 파이프라인(`run_qwen_online_htmnpo_ronpo.sh` 계열, INPO-loss)으로 3 seed 학습 → 28-pool floor 재채점.
2. **RONPO-OS + IPO seed 45,46 추가** — Table 3의 RONPO-vs-IPO floor 동률을 유의로 전환 가능(kekim "3 seed 부족"도 해소).
   고정 P4 pair + SEED만 변경, frozen 28-pool 대비 floor 계산이라 renormalization 불필요.

두 실험이 성공하면 main_v4의 유일한 약점(IPO 대비 floor 동률, MaxMin 부재)이 닫힘.

## 통합 산출물 위치
`results/v4_integration_20260723/` : figs/(스크립트+PDF+PNG), qual/(예시 텍스트+examples.json), novelty/(UF fair-pool 데이터)
