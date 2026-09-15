# non-transitivity가 높은 dataset 탐색 — 2026-09-15

사용자 가설: "NBPO 성능이 좋지 않은 것은 dataset에 preference cycle이 없어서다."
결론부터: **부분적으로 사실이 아니다.** 데이터에는 PROSPER 정의로 34~43%의 순환이
있다. 사라지는 원인은 데이터가 아니라 **판정 순서 편향 보정**이다. 원고는 수정하지 않음.

## 1. PROSPER의 setting (arXiv 2602.19041 §6.1, 원문 확인)

- 데이터: **WildChecklists** held-out **100 prompt**
- 응답: **Qwen2.5-7B-Instruct base** 모델에서 **N=2~8개**
- 판정자 2종: **PJC**(체크리스트 전체 → 단일 점수), **PSC**(항목별 독립 판정)
- 지표: "**cycle of any length**"를 갖는 prompt 비율 + Condorcet 승자 부재 비율
- PSC는 "(prompt, item) 쌍마다 계산한 뒤 항목·prompt 평균"
- Figure 2, N=8: 순환 약 **60~80%**, no-Condorcet 약 **6~8%**

우리 Wild 행은 같은 데이터·같은 base 8응답·같은 항목별 판정이다. 다른 것은 정의뿐이다.

## 2. 같은 판정 데이터에 PROSPER 정의를 그대로 적용

`cycles/prosper_definition.json`. 새 판정 0건 — 기존 verdict 재계산.

| 정의 | Wild | Safe | UF |
|---|---|---|---|
| **PROSPER식**: 길이무관 cycle, 단일 pass, tie밴드 없음 | **38.4%** | **34.5%** | **42.6%** |
| 3-cycle만, 단일 pass, tie밴드 없음 | 38.3% | 34.5% | 42.1% |
| 길이무관, **양쪽 제시순서 평균** | 5.4% | 10.0% | 11.9% |
| 길이무관, 양순서 + 반복 2회 + δ=0.05 | 8.3% | 16.3% | 15.1% |
| Table 37의 C (삼각형 수준, 반복 요구) | 0.012% | 0.098% | 0.021% |

no-Condorcet(PROSPER식): Wild 10.1%, Safe 10.3%, UF 14.4% → PROSPER의 6~8%와 같은 범위.

**진단.** cycle 길이는 원인이 아니다 — 3-cycle 38.3% vs 길이무관 38.4%로 동일하다.
원인은 **제시 순서**다: 단일 순서 38.4% → 양순서 평균 5.4%로 **7배** 줄어든다.
평균 순서 격차는 Wild 0.369 · Safe 0.301 · UF 0.390 (0~1 척도)로 매우 크다.
즉 PROSPER가 보고하는 "intransitivity"의 대부분은 판정자의 **위치 편향**이며,
양쪽 순서를 평균하면 사라진다. 우리가 순서를 평균했기 때문에 C가 작게 나온 것이다.

## 3. 인간 원표(native vote) 기반 dataset 탐색 — 판정 0건, 비용 0

cycle에는 한 prompt에 **응답 3개 이상 + 세 pair 모두 비교**가 필요하다. 대부분의
preference dataset은 prompt당 pair 1개라 삼각형을 아예 만들 수 없다.

| dataset | 구조 | prompt 중 cycle | 판정자 |
|---|---|---|---|
| **lmsys/mt_bench_human_judgments** (human) | 80문항×6모델, 3.69표/pair | **14.3%** (11/77) | **인간 65명** |
| 같은 데이터 gpt4_pair | 80문항×6모델, 2표/pair | **0.0%** (0/76, 885삼각형) | GPT-4 1개 |
| **openai/summarize_from_feedback** | 14,092 post, 중위 4요약, 1.37표/pair | **3.44%** (162/4,711) | 인간 다수 |
| allenai/multipref | prompt당 비교 2개 (edge 2개) | **측정 불가** | 인간 4명/비교 |

`mt_bench`의 두 행이 핵심이다. **문항·모델·pair가 완전히 동일한데** 인간 65명은
14.3%, GPT-4 단일 판정은 정확히 0%다. 순환은 데이터가 아니라 **평가자 이질성**에서
나온다 — 우리 원고 §281의 세 집단 예시가 주장하는 바로 그 기전이다.

## 4. 결론

1. **데이터셋 교체는 지렛대가 아니다.** Wild/Safe/UF 세 출처가 raw 34~43%,
   순서보정 후 5~12%로 사실상 동일하다. 이름을 바꿔도 값이 바뀌지 않는다.
2. **지렛대는 판정자 구성이다.** 동일 데이터에서 인간 다수 14.3% vs 단일 LLM 0.0%.
   단일 판정자는 (위치 편향을 제거하면) 거의 transitive하다.
3. **규모가 필요하면** `openai/summarize_from_feedback` (4,711 post에서 3.44%,
   실제 인간 투표, 다운로드 가능). **최대 순환율이 필요하면** `mt_bench` human
   (14.3%)이지만 80문항으로 학습에는 부족하다.
4. 현재 데이터를 유지하려면 단일 Qwen3-14B 대신 **이질적 판정 panel**(서로 다른
   모델 또는 서로 다른 페르소나 다수)로 바꾸는 것이 순환을 만드는 유일한 경로다.
