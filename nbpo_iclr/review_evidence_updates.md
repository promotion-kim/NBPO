# 후속 실험 근거 — 원고 문장은 수정하지 않음

이 파일은 새로 측정된 값이 기존 표·문장과 어긋나는 지점을 기록한다. 원고의
`\reviewresponse{...}`와 본문은 고치지 않았고, 반영이 필요한 변경은
`proposed_manuscript_patch.diff`(미적용)로만 남겼다.

## 1. PROSPER 3번째 seed (2026-09-14 14:11 KST 측정 완료)

`uf4_train_prosper_mse_s44`가 1250 step을 2:47:23에 끝내고 final-eval 판정 16,000건을
1,523초에 마쳤다(parse 실패 81건, 0.51%). 이 seed를 더한 공통집합 재집계 결과:

| 항목 | 원고(2 seed) | 측정(3 seed) |
|---|---|---|
| 공통 prompt | 1,236 | **1,217** |
| PROSPER 지시이행 | 0.5109 | 0.5137 |
| PROSPER 진실성 | 0.5095 | 0.5079 |
| PROSPER 정직성 | 0.5031 | **0.4975** |
| PROSPER 도움됨 | 0.5452 | 0.5451 |
| PROSPER seed별 최소값 평균 | 0.4990 | **0.4954** |
| 지시이행 seed 산포(SD) | 0.0303 | 0.0216 |

seed별 지시이행 값은 0.5327 / 0.4901 / 0.5183이다.

**읽는 법.** 1 seed에서 PROSPER는 표 전체 최고 최소값 0.5072를 보였고 기전 대역 위 3속성이었다.
2 seed에서 최소값이 0.4990으로 내려왔고, 3 seed에서 **0.4954로 더 내려가 무변화 수준 0.5를
아래로 지난다.** 즉 단일 seed 우위는 복제되지 않았고, 이 표에서 단일-seed 순위가 복제에 실패한
사례는 4건 중 4건으로 유지된다(fixed-reference, global game-maxmin, BT-RM–Nash, PROSPER).

3 seed 공통집합(1,217 prompt)에서 최소값 순위는
`dpo_truth_only 0.5230 > dpo_uniform 0.5036 > BT-RM–Nash 0.5024 > MOPO 0.5021 >
fixed-reference 0.4994 > game-utilitarian 0.4961 > PROSPER 0.4954 > NBPO 0.4940 >
global game-maxmin 0.4932 > dpo_if_only 0.4764`. NBPO는 여전히 최소값 하위이며, 원고가
보고하는 귀무 결과의 방향은 바뀌지 않는다.

**artifact**
- `/work/uf4_20260910/analysis/final_eval_common_26arm.json`
  sha256 `94a93a2ee38ee13d197344c47438f6537be8a0724bf27e8e3796a835a7df825e`
- `/work/uf4_20260910/evaluation/final_eval/prosper_mse_s44/complete.json`
  sha256 `101019535f30e2ffd6bfaf1106a885b32ecf7d4a96603974ec1f6953b19aa86d`

**왜 적용하지 않았는가.** s44를 공통집합에 넣으면 공통 prompt가 1,236 → 1,217로 줄어 **Table 1의
모든 행이 조금씩 움직인다.** 보호 계약은 기존 표의 수치를 저자 검토 대상으로 두므로, 기존 filler가
AUTO 구역에 쓰는 전체 갱신분을 원고 사본에서 생성해 diff로만 제출했다(146줄 변경).
`python3 validate_protected.py`는 계속 통과한다.

## 2. capability 표

s44의 벤치 생성물이 나오는 대로 교차-arm 채점 4개 패널을 23-arm으로 재실행하도록 의존성 큐에
올려 두었다(`uf4_score_harmbench_23arm`, `uf4_score_xstest_22arm`,
`uf4_score_alpaca_arena_22arm`, `uf4_score_mtbench_23arm`). 현재 원고의 capability 표는
22/21-arm 패널 기준이며, 재채점 결과도 같은 방식으로 diff로만 제출한다.

## 3. 미측정으로 남는 것

- UF readiness 행(`ufN/ufC/ufD/ufGamma/ufTV`): 이번 N=8·2 repeat 계약으로 재실행하지 않으면
  미측정으로 남긴다. 기존 N=4 audit 값을 복사하지 않는다.
- DPO 7개 가중치 중 4개: 큐에 남아 있으나 신규 캠페인 뒤 잉여 용량에서 실행된다.
- rubric/reference 민감도 부록: 선언만 되어 있고 측정하지 않았다.
