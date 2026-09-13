# 2026-09-14 09:00 KST 인계 — 타깃 품질 / 신경망 재현 / 새 응답 진단

작성 시각 07:15 KST. 08:45 PDF 확정 후 마지막 수치만 갱신한다.

## 1. 확인된 병목

**타깃 자체는 문제가 아니다.** 200개 고정 dev prompt, 공통 4응답 reference bank, 독립 Qwen3-14B
판정에서 exact NBPO 타깃은 자신이 유도된 경험적 소스 분포(learner 8개 위의 균일 분포)를 이긴다 —
지시이행 **+0.0271 [+0.0023, +0.0518]**, 도움됨 **+0.0505 [+0.0273, +0.0743]** (whole-prompt paired
bootstrap, 0 배제). 최소 유한-bank surplus는 **−0.0054 → +0.0054**로 부호가 바뀐다. 결측에 강건한
경계 계산에서는 더 분명하다 — 200개 전량에서 타깃의 승률 구간이 **네 목표 모두** 소스의 상한보다
완전히 위다(예: 도움됨 [0.5722, 0.5810] 대 [0.4898, 0.5001]).

**병목은 집계 규칙의 구분 가능성이다.** 같은 패널에서
- NBPO − utilitarian: +0.0010 / +0.0012 / +0.0018 / +0.0004 (네 목표)
- NBPO − fixed-reference: −0.0007 ~ +0.0003

최대 **0.0018**로, 타깃 대 소스 격차(최대 0.0505)의 1/28이다. 즉 **논문이 정책 수준에서 보고하는
귀무 결과는 학습 이전의 타깃에 이미 존재하며**, 타깃을 본 적 없는 평가자에서도 그렇다.

**왜 타깃들이 가까운가**도 측정했다. 목표들이 후보 순위에서 강하게 정렬돼 있다 — Spearman 평균
0.777~0.915, ρ<0인 prompt 최대 1.0%, top-candidate 일치 57.0~77.5%, 한쪽 argmax가 상대 목표의 풀
평균을 넘는 비율 96~100%. 가장 덜 정렬된 쌍은 진실성 대 도움됨. 타깃 분포 간 평균 TV는 Nash 대
utilitarian 0.0206, 대 fixed-reference 0.0240.

**재현(projection)은 타깃을 재현하지 않지만 품질을 잃지도 않는다.** fitted pool은 자신의 exact
target보다 지시이행·진실성·정직성에서 높고 최소 surplus도 크다(NBPO +0.0222, util +0.0163 대
+0.0054, +0.0036). 그런데 분포로는 출발점보다 타깃에서 멀어졌다 — `KL(p*‖p̃θ)` 1.457(두 arm)
대 `KL(p*‖p_t)` 1.047/1.075, TV 0.58, log-ratio RMS 3.2 이상.

**수치 차이가 오차에 묻힌다.** 같은 dev 패널·같은 28쌍 측도에서 타깃 차이 RMS는 0.179(NBPO vs
util)·0.418(vs fixed-ref)인데 적합 오차 RMS는 4.21(NBPO)·4.28~4.30(대조군)이고, **오차 차이**
RMS가 1.18·1.23이다. 학습된 두 정책 사이 차이(1.18·1.20)는 수치적으로 오차 차이와 같다. 규모
비교로만 보고했고 비율에서 검정력 계산을 끌어내지 않았다.

## 2. 미확인 / 확인할 수 없었던 것

- **fresh 단계의 절대 수준**은 신뢰하지 말 것. 미학습 base의 새 draw가 같은 계약에서 목표별
  0.4789/0.4700/0.4700/0.4706이며 **각 구간이 0.5를 포함**(대칭성 통과)하지만, 네 목표의
  최솟값은 0.4613 [0.4296, 0.4917]로 0.5를 배제한다 — 최솟값 통계의 구조적 하향 편향이다.
  따라서 fresh W_min의 무변화 기준선은 **0.461**이며 0.500이 아니다. 같은 계약을 공유하는 fresh
  행 사이의 차이는 이 오프셋에 영향받지 않는다.
- **완전-사례 선택 효과.** (prompt, objective) 칸은 76개 verdict가 모두 파싱돼야 완전해서, 목표별
  143/131/144/167, 네 목표 공통 78개만 남는다(파싱률 99.10%). 78개 완전-사례와 200개 경계 계산이
  정직성 격차의 부호에서 엇갈리는데 이는 커버리지 효과다.
- **surplus의 결측 경계는 무정보**([−0.16, +0.18]). 직접승률 경계와 달리 해석에 쓸 수 없다.
- **judge revision은 우리 기록상의 선언값**이다. `assets/Qwen3-14B` 디렉터리에 upstream snapshot
  식별자가 없어 `40c06982`를 독립적으로 검증할 수 없다.
- **유한-bank 대리량**이다. m_eval=4, β_eval=0.25(패널 실행 전 고정). 모집단 응답분포 KL의 game
  value 추정이 아니다.
- pilot의 목표 내 순환 측정(0건/11,200 삼각형)은 스칼라 값이 강한 순서를 유도하므로 거의 자동이며
  전체 쌍별 텐서에 대한 증거가 아니다.
- **PROSPER 3번째 seed, DPO 6개 가중치 중 5개, capability 잔여 칸**은 측정되지 않았다.

## 3. 실제 artifact (모두 파드 `/work/uf4_20260910/analysis/diag_20260914/`)

| 파일 | 내용 |
|---|---|
| `panel_dev200.json` / `panel_train200.json` | 동결 패널, 해시 `f899600a…` / `02a1c2ce…` |
| `bank4/verdicts/<prompt>.jsonl` (200개) | 60,800 rubric-order verdict 원본 |
| `bank4/complete_{first10,shard0of4…}.json` | 샤드별 처리량·파싱률 |
| `target_transfer.json` | 행별 승률·surplus, 2,000회 paired bootstrap, 결측 경계, 계약 검증 |
| `fresh/*.jsonl` + `complete_*.json` | 6 arm × 200 응답, 길이·cap·거부 진단 |
| `fresh_verdicts/*.jsonl` | 38,400 fresh verdict |
| `neural_pools.json`, `pool_log_ratios.json`, `neural_pool_diagnostics.json` | 재가중 분포·정책 log-ratio·pool KL |
| `projection/{dataset_build,pilot_table,pilot_fresh_ci}.json` | pilot 데이터셋·적합·fresh 페어 CI |
| `target_signal.json`, `conflict_report.json` | 타깃 차이 대 오차 차이, 목표 간 충돌 |
| `loss_identity.json`, `tensor_contract.json`, `feedback_units.json` | P0 계약 검증 3종 |
| `overnight_diagnostics_manifest.json` | 18행, 전부 DONE |

호스트: `nbpo_iclr/figures/target_transfer.pdf`, `nbpo_iclr/progress/pdfs/main_v6_*_KST.pdf`,
`nbpo_iclr/scripts/plot_target_transfer.py`, `nbpo_iclr/progress/fill_target_transfer.py`.

## 4. 후속 queue와 ETA

컨트롤러(pid 415591)는 세션과 무관하게 계속 돈다. 아래는 그 큐에 남아 있는 작업이다.

| 작업 | 상태 | ETA(연속 가동 가정) |
|---|---|---|
| `uf4_train_dpo_if_only_mse_s42` 판정 | 08:05 예상 | Table 1 행 + tradeoff 2번째 가중치 |
| DPO 가중치 5개 (truth_only, honesty_only, help_only, help_heavy, truth_heavy) | READY, p72–76 | arm당 약 145분 → 약 12시간 |
| PROSPER seed 44 | **의도적 FAILED** | 재개 방법은 아래 |
| capability 잔여 칸 (MOPO·PROSPER 안전·아레나) | 교차-arm 채점 필요 | 각 계열 seed 완료 후 약 15분 |

**PROSPER s44 재개 방법**: 05:10에 자동 디스패치됐으나 실측 단가로 계산한 판정 시각이 08:50이어서
08:45 원고 동결을 넘기고 그 사이 4장을 모두 점유하므로 중단했다. `save_steps: 1250`이라 회수 가능한
checkpoint가 없었고 손실은 약 3분이다. 사유·근거는
`jobs/runs/uf4_train_prosper_mse_s44/stopped_by_operator.json`에 있다. 재개하려면
`jobs/queue/70_train_prosper_mse_s44.json`의 아무 필드를 수정하면 된다(컨트롤러가 spec 변경을 보고
terminal 기록을 버리고 처음부터 스케줄한다).

## 5. 09:00 스냅샷에 들어간 것 / 들어가지 않은 것

들어간 것: 세 신규 표(`tab:target-transfer` 10행, `tab:projection-ablation` 2행,
`tab:target-signal` 2행), `fig:target-transfer`, `tab:diagnostic-contract`의 완료 범위, P0 검증
4건, 초록·결론의 결과 delta, 기존 exhibit 정합 스냅샷.

들어가지 않은 것(지우지 않고 `\pending`으로 유지): PROSPER 3번째 seed, DPO 6개 가중치 중 5개,
capability 10칸, DPO if-only 판정(08:15까지 도착하지 않으면 보류).
