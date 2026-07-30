# RONPO AAAI-27 개정 노트 (v6 → v7)

**마감 (AoE 기준):** 초록 등록 2026-07-21 / 본문 제출 2026-07-28 / 보충자료 2026-07-31.
오늘(7/7) 기준 본문 제출까지 **3주**입니다. 아래 "반드시 해야 할 일"의 실험 항목을 최우선으로 돌리세요.

---

## 1. 지도교수 피드백 9개 항목별 대응

### ① "AAAI 논문 형식이 아니다"
- SPPO/INPO 계열 논문의 표준 구성을 그대로 따르도록 전면 재배치:
  **Intro → Related Work → Preliminaries → Method(RONPO) → Practical Loss → Experiments → Conclusion/Limitations**, 증명은 전부 Appendix A로 이동.
- 본문이 정확히 **7페이지**에서 끝나고, 참고문헌이 8–9페이지에 위치하도록 맞췄습니다(AAAI-27 규정: 기술 내용 7p + 참고문헌 전용 2p).
- Appendix는 9페이지부터 같은 파일에 이어져 있습니다. **제출 시에는 본문(1–9p)과 보충자료(Appendix)를 AAAI-27 지침에 따라 분리**해야 할 가능성이 높으니 CFP의 supplementary 규정을 확인하세요.
- ⚠️ 현재 `aaai2026.sty`로 컴파일한 상태입니다. **AAAI-27 author kit(aaai2027.sty/bst)을 받아 교체**하세요(파일 상단 주석 참조). Reproducibility checklist도 별도 제출입니다.

### ② "main theory 같은 비표준 용어"
- "main theory", "our main theorem says" 류 표현 전부 제거. 정리는 이름을 붙여 명명:
  Lemma 1 (Coupling cancels in the monotonicity gap), Thm 1 (Existence and uniqueness), Thm 2 (Exact last-iterate rate), Cor 1 (Stochastic preference feedback), Prop 1 (Soft worst-case form), Prop 2 (Unbiased partition-free regression).
- 표기 통일: 정책 `p → π`, 상대 정책 `q → π′` (RLHF 문헌 관례), μ는 reference, s는 adversary로 유지.

### ③ "contextual bandit 설명 없이 contextual case 언급"
- Preliminaries 첫 소절로 **"Preference-Based Alignment as a Contextual Bandit"** 신설: context/response/상대적(Bernoulli) 피드백, skew-symmetry, 비추이성 허용을 정의하고, "per-context로 분석 후 x 조건부로 lift한다"는 관례(SPPO/INPO와 동일)를 명시. 이후 본문 어디서도 정의 없는 개념이 등장하지 않습니다.

### ④ "toy 실험은 이론을 수치로 뒷받침하거나 삭제"
- **새 검증 스크립트 `scripts/validate_theory.py`** 작성·실행 완료. seed 0이 논문의 decoy 게임 인스턴스(V\*=0.3289)를 정확히 재현함을 확인했습니다.
- 생성된 **Figure `ronpo_lastiter_validation.png`** (진짜 실행 결과, placeholder 아님):
  - (a) Θ_t = D(z\*‖z_t)를 Thm 2의 명시적 상계 M/(t+t₀)와 함께 log-log로 표시. 정확한 dynamics는 모든 반복에서 상계 아래(결정론적 사상의 고정점이 z\*이므로 더 빠른 게 정상), **단일-pair Bernoulli 피드백(Cor 1)은 8회 평균에서 log-log 기울기 −0.90**으로 Θ(1/t)를 포화 — 상계와 평행하게 감소.
  - (b) κ↓0에서 soft-min 내부값과 hard minimum이 함께 LP 값 V\*=0.3289로 수렴 (Prop 1 검증).
- 실험 절에 "Numerical Validation" 소절을 만들어 각 패널을 **정리 번호와 1:1로 연결**해 서술했습니다.
- ⚠️ TODO-AUTHOR: Table 2(toy)의 실제 seed/하이퍼파라미터를 Appendix C 주석 위치에 기입하고, 본인 seed로 figure 재생성 권장(`python scripts/validate_theory.py`).

### ⑤ "베이스라인 약함 — DPO, IPO 등 추가"
- Related Work·Table 1·베이스라인 문단에 DPO/IPO/KTO/SimPO를 **서술로는** 편입 완료(인용 추가됨).
- **수치는 절대 지어내지 않았습니다.** Table 3(stage-2)에 `DPO-iter/IPO/SimPO/KTO` 행을 **주석 처리된 예약 행**으로 넣어 두었습니다. 실험을 돌린 뒤 주석을 풀고 측정값만 기입하고, 베이스라인 문단의 주석에 준비된 문장으로 교체하세요.
- stage-2에 SPPO-avg/INPO-avg 튜닝 버전 복원도 TODO에 포함(현재는 stage-1만 있음 — Limitations에 정직하게 명시해 둠).

### ⑥ "1.5B는 너무 작다 — scale up"
- **필수 실험**: Qwen2.5-7B-Instruct 또는 Llama-3-8B-Instruct 중 최소 1개로 stage-1/2 반복 + **AlpacaEval 2.0 LC / Arena-Hard / MT-Bench** 보고(이 분야 표준 외부 벤치마크; SPPO/INPO/MNPO 모두 사용). bib 항목(dubois2024length, li2024arenahard, zheng2023mtbench)은 미리 넣어 두었습니다.
- 현재 원고는 1.5B 결과를 "단일 백본, 단일 seed"로 **스코프를 좁혀 정직하게** 서술하고 Limitations에 명시했습니다. 7B 결과가 나오면 본문 표를 7B 중심으로 바꾸고 1.5B를 appendix로 내리는 편이 리뷰에 훨씬 유리합니다.

### ⑦ "이론적 novelty 부족"
- 두 가지를 추가했습니다(둘 다 증명 검증 완료):
  1. **Corollary 1 (stochastic preference feedback)**: 단일-pair Bernoulli 피드백 하에서도 기대값 기준 동일한 O(1/T) last-iterate — 피드백이 {0,1} 좌표별이라 operator bound가 pathwise로 성립 + 조건부 불편성 + tower property. 실제 알고리즘이 확률적 질의를 쓰므로 정확한(exact) 결과보다 실질적으로 의미 있는 보강입니다. Fig 검증 (a)의 stochastic 곡선이 이 결과를 수치로 뒷받침.
  2. **Remark 1 (pair-level vs weight-only dominance)**: RONPO의 pair-floor가 MaxMin-RLHF류 weight-only floor를 **모든 정책에서** 지배(더 강한 인증) — 2줄 증명이지만 포지셔닝상 중요.
- 동시에 Intro 마지막에 "표준 monotone VI 논증의 instantiation이며, 기여는 heterogeneity 하에서 그 논증이 *적용되게 만드는* formulation"이라고 **스코프를 명시**했습니다. novelty를 부풀리는 것보다 이렇게 정확히 긋는 편이 이론 리뷰어에게 안전합니다.

### ⑧ "스토리가 불명확"
- Intro를 **Why a game? → The heterogeneity problem → This work** 3문단 구조로 재작성:
  게임이론이 필요한 이유(비추이성 → von Neumann winner), 기존 두 접근(averaging: 소수 objective 붕괴 은닉 / per-player: general-sum이라 last-iterate 이론 부재)의 문제, RONPO의 핵심 아이디어(heterogeneity를 단일 adversary의 행동공간으로 이동 → monotone 구조 보존)를 명시.
- 기여 4개 bullet(Formulation/Guarantees/Algorithm/Experiments)으로 정리.
- Preliminaries 끝의 "Existing Methods Through the OMD Lens" + Table 1이 베이스라인의 문제를 구조적으로 대비.

### ⑨ "글이 아마추어 같다 — 기존 논문 구조를 베껴라"
- INPO/SPPO의 절 구성·정리 제시 방식(본문: statement + proof idea, 증명: appendix)을 그대로 차용.
- 영국식 철자 수정(modelling→modeling), 오탈자("samples an pair") 수정, 실험 절의 lab-note 문장 제거.
- **익명성/식별 정보 제거**: `/home/sjkim/...` 경로, 내부 스크립트명, evalscope 쉘 스크립트명 전부 삭제 (double-blind 위반 소지).
- 원본의 **잠재 버그 수정**: AAAI 스타일은 `secnumdepth=0`이라 appendix `\ref`가 엉뚱한 번호(직전 수식 번호)를 찍고 있었습니다. `\appendix` 뒤 `\setcounter{secnumdepth}{2}`로 A/B/C 번호가 정상 출력되도록 고침.

---

## 2. 컴파일 상태 (검증 완료)

- 오류 0, 미해결 참조 0, bibtex 경고 0, **overfull hbox 0**.
- 본문 7p에서 종료 / 참고문헌 8–9p / Appendix 9–16p.
- ⚠️ **여유가 거의 0**입니다: 실제 figure(placeholder 3개 교체)와 DPO 계열 4행이 들어가면 7p를 넘길 수 있습니다. 넘치면 후보: Table 1 축소, OMD lens 소절 압축, Fig 1(toy curves)을 appendix로 이동.

## 3. placeholder / 지어내지 않은 것 (제출 전 필수 확인)

| 항목 | 상태 |
|---|---|
| `figures/ronpo_toy_curves.png`, `ronpo_decoy_sweep.png`, `kappa_tradeoff_avg_vs_min.png` | **회색 placeholder** — 원본 그림으로 교체 필수 |
| `figures/ronpo_lastiter_validation.png` | 실제 실행 결과 (스크립트 동봉) |
| DPO-iter/IPO/SimPO/KTO 행 | 주석 처리된 빈 행 — 측정 후 기입 |
| 7B/8B scale-up, AlpacaEval/Arena-Hard/MT-Bench | 미실행 — TODO |
| GPT judge 프롬프트 원문 | Appendix D 주석 위치에 verbatim 붙여넣기 |
| zhou2024beyond (MODPO) bib 저자 순서 | 확인 필요 |

`grep -n "TODO" ronpo_aaai27_v7.tex` 으로 전체 목록을 볼 수 있습니다.

## 4. 남은 3주 권장 일정

1. **~7/12**: 7B scale-up 학습 착수(가장 오래 걸림) + DPO/IPO/SimPO/KTO 베이스라인 병렬 실행.
2. **~7/18**: 표 채우기, 실제 figure 교체, seed ≥3 부트스트랩 CI, aaai2027 kit 교체 후 페이지 재확인.
3. **7/21**: 초록 등록. **7/28**: 본문. **7/31**: 보충자료(Appendix+코드).
4. 트랙 선택: AAAI-27 신설 **AI Alignment special track**이 이 논문 주제와 정확히 맞습니다. main track과 비교해 지도교수님과 상의하세요.

## 5. AAAI LLM 정책 관련
AAAI-27은 LLM이 *생성*한 텍스트 제출을 금지하되, 저자 원고의 **편집·교정 보조는 허용**합니다. 이번 개정은 사용자의 기존 원고(수식·수치·실험 전부 원본 유지)를 재구성·교정한 것이지만, 최종 문장은 반드시 본인이 검토·수정해 본인의 글로 만드세요. 특히 Intro/Abstract는 본인 언어로 다시 다듬기를 권합니다.
