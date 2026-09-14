# NBPO ICLR 2027 제출 실험: 원고를 보존하고 지정된 결과만 채우기

너는 기존 NBPO 저장소의 연구 엔지니어다. 이 프롬프트와 함께 전달된 수정 원고를 보존하면서, **새 데이터에서 NBPO의 기전이 실제로 구별되는지 검증하고 지정된 결과 템플릿을 채워라.** 목표는 좋은 결과를 골라내는 것이 아니라 현재의 설명을 반증할 수 있는 비교를 완료하는 것이다. 새 실험 실행 여부를 다시 묻거나 계획만 제시하고 멈추지 말고, 이미 승인된 자원 안에서 아래 순서대로 수행하라.

공식 마감은 abstract **2026-09-18 23:59 AoE = 2026-09-19 20:59 KST**, full paper **2026-09-25 23:59 AoE = 2026-09-26 20:59 KST**다. 시작 시 현재 UTC/KST를 실제로 확인한다. 이전 `2026-09-14 09:00 KST` 진단 마감은 이미 지났으며 이 캠페인의 마감으로 재사용하지 않는다. 공식 근거: https://iclr.cc/Conferences/2027/CallForPapers

기존 **H200 4장**, MLXP `p-aipr` project, 기존 `sjkim` / `sjkim-data` / `sjkim-workspace` storage를 사용한다. **새 유료 API 호출은 0건**이다. 기존 정상 작업과 controller를 인계하고 중복 job을 실행하지 않는다. 다른 사용자의 작업을 종료하지 않는다. 이 프롬프트는 이전 프롬프트의 '새 데이터 실험 금지'와 '기존 본문 자동 갱신' 지시를 아래의 한정된 실험 및 엄격한 원고 보호 규칙으로 대체한다.

## 1. 가장 중요한 규칙: 논문을 다시 쓰지 않는다

새 패키지의 `main_v6.tex`는 저자가 검토한 수정 원고다. 본문·이론·알고리즘·기존 표의 수치·caption·GPT 리뷰 원문·GPT 리뷰 답변·bibliography를 임의로 수정하지 않는다. 새 결과가 좋거나 나빠도 주장, abstract, conclusion, method 이름, 표의 비교군을 자동으로 바꾸지 않는다.

- 원고 내 새 결과는 **`templates/results.json` → 제공된 `render_results.py` → `templates/results_auto.tex`** 경로로만 반영한다. `templates/experiment_templates.tex`의 구조·설명·label은 유지한다. 새 정해진 그림은 `figures/stress_test.pdf`에 원본 수치에서 생성한다. 허용된 결과 셀 이외에 prose를 추가하지 않는다.
- `protected_manifest.json`과 `validate_protected.py`를 읽고, 시작 전·결과 렌더 전후·빌드 전에 검증한다. **검증을 통과시키려고 보호 파일을 삭제하거나 manifest의 기준 hash를 갱신하지 않는다.** 템플릿이나 renderer에 실제 결함이 있으면 `proposed_manuscript_patch.diff`와 사유를 작성하되 적용하지 않는다. 안전하게 쓸 수 있는 결과·코드·분석 작업은 계속한다.
- 새 실험 코드, resolved config, raw judgments, 로그, 검증 결과, `progress/` 상태 파일은 작성해도 된다. 이 제한은 실험 구현을 막는 것이 아니라 기존 논문을 보호하기 위한 것이다.
- startup에서 서버의 최신 원고·raw manifest와 전달받은 원고를 비교한다. 더 늦은 측정은 먼저 raw manifest에 병합하고 신규 결과의 대응 필드에만 반영한다. 교수님이 나중에 수정한 문장은 유지한다. 기존 표나 문장을 바꿔야 하는 사실 갱신은 근거와 최소 diff를 `startup_reconciliation.md` / `proposed_manuscript_patch.diff`에 남긴다. 과거 원고 전체를 서버 최신 원고에 덮어쓰지 않는다.
- `templates/results.json`에 없는 새로운 cell이나 metric을 결과를 본 뒤 만들지 않는다. 중요하지만 슬롯이 없는 결과는 raw artifacts와 `additional_findings.md`에 기록한다. renderer의 schema와 실제 CLI를 먼저 읽고 제공된 사용법을 따른다. 단위·seed·N·CI·status가 맞지 않는 숫자는 넣지 않는다.
- GPT 리뷰 답변은 이번 수정본에서 이미 갱신되어 있다. 후속 실험의 리뷰 대응 근거는 `review_evidence_updates.md`에 작성하고, 원고의 `\reviewresponse{...}`는 수정하지 않는다.

`NOT_STARTED / READY / RUNNING / DONE / BLOCKED / NOT_APPLICABLE`을 구분한다. DONE은 raw artifact, 집계, provenance, 검증을 모두 갖춘 경우뿐이다. 미측정 숫자는 null/미측정 상태로 유지하고 0, 기존 데이터셋 수치, 다른 seed 수치로 채우지 않는다. 결과가 없는 그림은 빈 template 상태를 유지한다.


### 결과 입력 파일의 정확한 사용법

패키지 root에서 `python3 validate_protected.py`, `python3 render_results.py`, `python3 validate_protected.py` 순서로 실행한다. `templates/result_keys.json`의 81개 key를 추가·삭제·개명하지 않는다. 각 `cells[key]`는 미측정이면 null, 측정되면 `value`(유한 numeric), `artifact`(실제 raw/aggregation 파일 경로), `sha256`(그 파일의 실제 SHA-256)을 가진 record로 채운다. 필요하면 `ci95`, seed IDs, denominator/contract metadata를 같은 record에 보존한다. renderer는 값만 4자리로 표시하고 CI는 원본 record에 남긴다. `contract_id`에는 frozen campaign contract ID를 쓴다. renderer 자체는 source bytes의 hash를 검증하지 않으므로 집계 코드에서 확인한다.

Base의 S는 training seed 수가 아니라 동결한 generation replicate 수로 기록하고 혼동하지 않는다. 각 method의 N은 고정된 모든 선언 arm/seed/rubric의 common set이다. pairwise-only N은 별도 artifact에 기록하며 본문 표의 N으로 바꾸지 않는다. Seed means와 prompt CI를 섞지 않는다. Fig의 full/stratum 결과는 별도 artifact에 저장하며 없는 JSON cell을 새로 만들지 않는다. UF readiness 행은 이번 N=8과 같은 직접 비교 계약을 실행했을 때만 채운다. 기존 N=4 audit 값을 복사하지 않으며, 재실행은 필수 GPU 작업보다 후순위다.

9쪽 확인은 `latexmk -pdf main_clean.tex`로 만든 `main_clean.pdf`에서 한다. 검토용은 `latexmk -pdf main_v6.tex`다. Main text는 현 수정본에서 9쪽 이내이며 상세 수치·기존 표는 부록에 보존되어 있다. 새 결과가 강해도 appendix template을 main으로 자동 이동하지 않고 저자 검토용 최소 diff만 제안한다.

## 2. 시작 후 30분: 상태 인계·계약·실행표 고정

1. `AGENTS.md`, 현재 repository/configs, 제공된 원고와 두 이전 프롬프트, job/controller/reporters, checkpoints, result manifests를 읽는다. 현재 KST, git revision, source hashes, GPU UUID/index·VRAM·소유 job·진행 step/sample을 기록한다. 과거 PID·teacher revision을 현재 사실처럼 복사하지 않는다.
2. 기존 UF 결과와 새 submission 실험을 다른 manifest로 관리한다. **기존 UF의 부정적/불확실한 결과는 유지**한다. 기존 SafeRLHF의 score-induced teacher에서 Nash–utilitarian target TV가 약 0.0008이었던 것도 보존한다. 그 teacher를 그대로 재사용하면서 '새 갈등 데이터'라고 부르지 않는다.
3. 새 실험의 `experiment_id, dataset/revision/license, split_hash, candidate_hash, objective_contract_hash, teacher/evaluator_revision, method, seed, target/checkpoint_hash, config_hash, job_id, status, artifact, GPU-hours, ETA` 실행표를 만든다. 선언한 행을 삭제해 완료율을 올리지 않는다.
4. 실제 로컬 모델 목록을 확인해 training judge와 evaluation judge의 역할을 고정한다. 둘이 다른 모델/학습 경로인지 확인하고 model revision 또는 실제 weight/tokenizer 파일 hash를 남긴다. 로컬 directory명만으로 upstream revision이 검증됐다고 쓰지 않는다. 같은 judge의 held-out prompts는 평가 split 독립성일 뿐 **독립 judge**가 아니다. 별도 평가 모델이 없으면 이 부분을 BLOCKED로 표시하고, 같은 judge 결과는 teacher-held-out 진단으로만 기록한다.
5. 정상 진행 중인 4-GPU 학습은 유효 checkpoint까지 이어간다. CPU 집계·데이터 검증·finite solver를 병행한다. GPU 전체를 쓰는 학습을 작은 judge job 때문에 강제로 중단하지 않는다.
6. 30분 reporter를 인계하거나 설치하고 실제 첫 상태와 PDF 빌드를 확인한다. 외부 Slack/email을 보내지 않는다. 환경이 자발적인 채팅을 지원하지 않으면 stdout 및 `progress/latest.md`에 보고하며, 보내지 않은 채팅을 보냈다고 말하지 않는다.

## 3. P0 — 먼저 답해야 할 질문

'cyclic preference가 많고 objective conflict가 크면 NBPO가 이긴다'를 전제로 삼지 않는다. 아래를 분리한다.

- **Representation:** 같은 objective·같은 response 집합에서 반복해도 남는 순환이 있는가? 이는 BT/고정 상대의 한계를 시험한다.
- **Bargaining:** 목표 간 충돌과 서로 다른 개선 여지가 존재하며, Nash·utilitarian·maxmin·PROSPER가 실제로 다른 feasible target을 만드는가? 순환만으로 Nash aggregation의 우위를 설명할 수 없다.
- **Feasibility:** 모든 objective가 reference보다 함께 개선될 수 있는가? 심한 conflict는 NBPO의 joint strict feasibility를 없앨 수 있다.
- **Transfer:** target 차이가 독립 평가에서도 의미가 있고, neural fitting 및 fresh generation으로 전달되는가?

새 실험 선택은 위 구조, 처리량, 평가 계약에 근거한다. NBPO가 가장 크게 이긴 데이터·rubric·seed만 선택하지 않는다. 구조가 없으면 '현재 데이터에서는 기전 구별이 어렵다'가 유효한 결론이다.

## 4. P1 — 9월 14–15일: 두 데이터의 소규모 직접 비교 audit

### 4.1 데이터와 objective의 의미를 고정

**A. PKU-SafeRLHF:** 공식 데이터의 helpfulness / safety 두 고정 objective를 사용한다. 원래 동일 response pair에 붙은 `better_response_id`와 `safer_response_id`는 목표 충돌을 조사하기 좋다. 다만 두 응답만 있는 레코드 자체는 preference triangle을 제공하지 않는다. 새 후보 응답을 생성하고 **직접 pairwise judge**로 두 objective를 평가한다. 원래 human label과 새 model judgment를 구분해서 보관한다. 안전성 judging과 생성은 저장소의 기존 승인된 프로토콜을 따른다.

**B. WildChecklists:** PROSPER가 사용하는 prompt별 checklist feedback에서 같은 item의 직접 비교가 얼마나 순환하는지 먼저 확인한다. 실제 dataset/config/revision, checklist 생성 provenance와 model card를 기록한다. 기본 후보는 https://huggingface.co/datasets/viswavi/wildchecklists 이며 사용 가능한 정확한 split/fields를 검사한다. PROSPER는 https://arxiv.org/html/2602.19041v2 를 기준으로 구현 계약을 확인한다. 이전 v1의 baseline 차이를 최신 v2에 대한 사실처럼 쓰지 않는다.

**주의:** WildChecklists는 prompt마다 checklist 수와 의미가 다르다. NBPO의 global multiplier `lambda_k`에 '각 prompt의 k번째 항목'을 그대로 연결하면 서로 다른 목적을 같은 bargaining party로 섞는다. 이번 기본 계획에서는 WildChecklists를 **native item cycle audit**으로만 사용한다. WildChecklists로 NBPO 학습까지 하려면 모든 prompt에서 의미가 동일한 고정 K개의 global rubric을 결과를 보기 전에 선언하고 missing-objective 처리와 joint feasibility를 검증해야 한다. 이 경우 feedback benchmark를 새로 정의한 adaptation이며 PROSPER native checklist 완전 재현이라고 주장하지 않는다. 실제 PROSPER와 같은 checklist 전체 judging은 비용도 별도로 계산한다.

HelpSteer 계열이나 PRISM 등 다른 데이터로 무한 탐색하지 않는다. 대체 데이터가 필요하면 기존 두 후보가 왜 부적합한지 기록하고 저자용 제안으로 남긴다. 새 데이터 탐색이 최종 실험을 계속 미루게 하지 않는다.

### 4.2 audit 크기와 label 예산

각 데이터의 **점수와 무관한 hash로 선택한 200 development prompts**, 같은 base에서 생성한 **8 candidate occurrences/prompt**를 고정한다. audit와 새 train/dev/test는 prompt와 source conversation 단위로 분리한다. 후보 채택을 NBPO 점수나 cycle 유무로 결정하지 않는다. 먼저 고정된 10 prompts를 실제 실행해 judge throughput과 parser failure를 측정한다. 그 10개도 200개 audit 안에 포함한다.

동일 8개 후보의 28 unordered pairs를 objective/item별로 **2 presentation orders × 각 order 2 independent decoding repeats** 평가한다. 기본 비용은 `28 × 2 × 2 × sum_x K_x = 112 sum_x K_x` verdicts/judge다. Safe K=2, N=200이면 **44,800 verdicts/judge**다. Wild의 평균 checklist item 수를 K=2로 가정하지 말고 실제 `sum_x K_x`로 계산한다. multi-item 한 call을 지원하더라도 semantic verdicts와 calls는 따로 센다.

200개 중 결과를 보지 않고 hash로 정한 **50 confirmation prompts**의 모든 pair·objective·order에 **추가 5 repeats/order**를 실행한다. 추가 비용은 `28 × 2 × 5 × sum_confirmation K_x = 280 sum_confirmation K_x`; Safe는 **28,000**, screen+confirmation 합계 **72,800 verdicts/judge**다. 이 추가 반복은 최초 두 반복과 독립인 확인 자료다. 발견된 cycle edge만 유리하게 재평가하지 않는다. 두 judge에 같은 전체 audit를 실행하면 비용은 두 배다. 실제 파일에서 model·order·sampling seed별 raw 출력을 보존한다.

먼저 training judge의 10-prompt 처리량으로 두 dataset audit와 아래 2k 학습 labeling의 총 ETA를 계산한다. 평가 judge는 같은 고정 confirmation panel에서 독립적으로 확인하는 것을 최소 요건으로 삼고, 전체 200개 확장 여부는 **첫 결과를 보기 전 예산표**에서 결정한다. 어느 judge에 몇 prompts·반복을 실행했는지 분모를 별도로 보고한다. evaluator 반복을 무단으로 줄여 '동일 계약'이라고 보고하지 않는다.

### 4.3 반드시 계산할 구조 지표

- 같은 rubric 안의 response graph에서 tie margin을 사전 고정하고 strong directed 3-cycle fraction을 계산한다. `cycle_count / eligible_distinct_response_triples`와 eligible denominator, any-cycle prompt fraction을 함께 기록한다. N=8일 때 최대 56 triples이며, raw cycle 수만으로 dataset을 비교하지 않는다. 중복 response hash가 있으면 occurrence multiplicity와 unique-response graph를 구분한다.
- 모든 edge는 **같은 objective**로 판단되어야 한다. helpfulness edge와 safety edge를 이어 만든 triangle은 within-objective cycle이 아니다. candidate order를 바꿨을 때 verdict가 달라지는 것도 cycle 자체가 아니다.
- initial repeats에서 방향을 정하고 추가 repeats에서 지속되는지 확인한다. 반복판단 모델에 적합한 BT null을 fit하고, 같은 graph·order·repeat 수를 모사하는 parametric bootstrap과 held-out-repeat predictive loss로 excess cyclicity를 평가한다. null fitting과 simulation 절차를 고정하고 finite-sample noise를 설명한다. 모델 기반 null 검정의 가정과 불확실성을 함께 기록한다.
- no-Condorcet-winner fraction과 top-response를 포함하는 cycle fraction을 기록한다. 약한 답변들끼리만 순환하는 경우와 최선 후보 선택에 영향을 주는 경우를 구분한다. direct LLM judge에서의 순환을 human preference의 순환으로 일반화하지 않는다.
- Safe에서 objective disagreement pair fraction, Kendall/rank correlation, top-1 agreement, Pareto-nondominated candidates 수를 계산한다. Wild는 native item-level 집계이며 서로 다른 item index를 global objective처럼 합치지 않는다. ties/uncertain edges와 prompt weighting의 정의를 고정한다.
- readiness의 finite-game 계산은 audit에서 이미 얻은 동일 8-response square matrix와 uniform occurrence reference를 쓰는 **8-response audit surrogate**로 명시한다. 이것을 본 학습의 독립 8Y+8Z contract와 같다고 주장하지 않는다. 본 학습용 8+8 target gate를 추가하면 필요한 cross/reference judgments의 비용을 별도로 기록한다.
- fixed-K panel에서 원고와 같은 finite temperature, disagreement 정의, shared global lambda로 max-margin joint-feasibility 문제를 풀고, `max_policy min_k s_k`의 값·solver residual·feasible/unresolved/infeasible status를 기록한다. objective 평균에 대한 global feasibility와 prompt별 margin을 섞지 않는다. 한 policy의 음수 surplus만으로 feasible set이 비어 있다고 결론내리지 않는다.
- NBPO / fixed-reference Nash / game-utilitarian / maxmin / BT-projected Nash / PROSPER target을 같은 tensors에 풀고 target TV·log-ratio RMS 차이·target policy movement를 기록한다. realized KL 또는 TV의 작은 사전 선언 grid로 movement를 맞춘 비교를 추가한다. L1 coefficient 일치만으로 policy movement가 같다고 쓰지 않는다.
- 독립 judge의 동일 candidate bank에서 target 대 empirical source와 target 간 차이를 paired 평가한다. structural gate는 기전이 구별될 여지가 있는지 판단하는 것이며 NBPO의 유의한 승리를 학습 시작 조건으로 삼지 않는다.

**9월 16일 시작 전 decision 파일을 고정한다.** 기본 학습 후보는 global objective 두 개가 명확한 Safe의 새 direct-feedback panel이다. 구조/feasibility/처리량이 부적합하면 이유를 기록하고 무리한 full run을 하지 않는다. Wild를 선택하려면 위 fixed-objective 계약을 먼저 만족해야 한다. 이번 캠페인은 **새 데이터 한 개만** neural finetuning한다. UF는 기존 결과를 유지하는 comparison/control이다.

## 5. P2 — CPU 중심: cyclicity × objective conflict의 2×2 기전 실험

기존 controlled finite-game code와 검증된 exact solver를 재사용한다. 새 2×2 조건은 low/high cyclicity × low/high objective conflict다. 기본 20개의 동일 instance seeds(0–19)와 payoff scale, reference support, temperatures, solver tolerances를 먼저 고정한다. 각 cell에서 같은 action 수·objective 수·utility scale·joint positive feasibility margin 범위를 유지한다. instance를 고르는 기준은 이 구조 조건과 수치적 유효성뿐이며 method ranking이 아니다. 모든 rejected instance 수와 이유를 보존한다.

잠재 scalar component와 antisymmetric cyclic component를 조절하는 구성부터 구현하되, 최종 실제 cycle fraction과 objective conflict가 의도한 두 축으로 변했는지 직접 측정한다. 단순 matrix norm 변화나 reference degradation으로 어려움을 바꾸지 않는다. 기존 generator가 축을 분리하지 못하면 새 generator의 계약과 결과를 exploratory로 명시한다.

NBPO / fixed-reference Nash / BT-projected Nash / game-utilitarian / global game-maxmin / PROSPER의 exact target 또는 해당 finite policy를 같은 정보로 비교한다. cyclicity 효과는 adaptive vs fixed/BT의 차이, bargaining 효과는 NBPO vs 다른 adaptive aggregation의 차이로 나눈다. 값 공간의 welfare, minimum surplus, feasible fraction, independent bank value 또는 정해진 held-out comparator score, 정책 이동량을 함께 계산한다. Nash welfare 하나를 primary로 정해 NBPO가 그 지표를 최적화한다는 동어반복을 성능 우위로 제시하지 않는다.

제공된 2×2 표를 채운다. 이 통제 실험의 추가 그림은 원고 밖 `artifacts/factorial_mechanism.pdf`로 저장한다. `figures/stress_test.pdf`는 원고 caption에 지정된 target/fitted/fresh 비교만 담는다. 새 합성 실험임을 유지하고 실제 human/natural cyclicity 증거라고 부르지 않는다. 신뢰구간은 instance seed variation으로 계산하고 LLM policy seed uncertainty와 구분한다. 승리하지 않은 cell도 같은 축·범위에 표시한다.

## 6. P3 — 9월 16–22일: 선택한 fixed-objective 새 panel 하나

### 6.1 common information / budget 계약

초기 계획은 **2,000 train / 500 dev / 1,000 test prompts**, 고정 8 learner occurrences `Y`와 독립 8 reference occurrences `Z`다. dataset source IDs와 normalized conversation hashes로 prompt leakage를 제거하고 split을 동결한다. audit prompts는 최종 test에 넣지 않는다. 기존 model family, tokenizer, full-parameter recipe, decoding length 및 sampling을 유지하며 backbone을 새로 바꾸지 않는다.

한 prompt에서 `8×8=64` Y–Z cross comparisons와 Z 내부 `C(8,2)=28` reference comparisons를 공통 feedback cache로 만든다. Safe K=2, 2 orders, 2 repeats면 **92×2×2×2=736 verdicts/prompt**다. 따라서 train **1,472,000**, dev **368,000**, 둘 합계 **1,840,000 teacher verdicts**다. 이것은 최종 evaluator, fresh generation, confirmation, repairs, tuning을 포함하지 않은 숫자다. Y–Y 28 pairs는 이 92-pair 예산에 포함되어 있지 않다. 필요하면 별도 비용을 사전 선언하며 한 방법에만 추가하지 않는다.

10-prompt audit의 실측 tokens/s·loading·parse/repair 비용으로 다시 ETA를 계산한다. 예산이 맞지 않으면 **policy 결과를 보기 전에 train을 모든 arm 공통으로 1,000 prompts로 축소**하고 train 736,000 + dev 368,000 = 1,104,000 verdicts로 계획을 다시 동결한다. final test 1,000은 유지한다. 그래도 불가능하면 target/finite-game audit 완료를 우선하며 대규모 학습을 시작한 것처럼 보고하지 않는다. repeats, rubrics, methods를 결과에 따라 선택적으로 줄이지 않는다.

### 6.2 비교군과 scalarized DPO 정의

Primary methods는 **NBPO, fixed-reference Nash, game-utilitarian, global game-maxmin, BT-projected Nash, PROSPER adaptation, uniform scalarized DPO**, 그리고 base다. native PROSPER에서 fixed two-objective feedback으로 바뀐 부분과 원 구현의 solver/update 차이를 정확히 적는다. PROSPER라는 이름의 임의 maxmin을 사용하지 않는다. 새 MOPO/HT-MNPO 구현·backbone sweep·7-weight DPO 추가 campaign은 이번 범위가 아니다.

모든 methods는 같은 candidates, pairwise raw feedback, optimizer/token budgets 및 dev tuning allowance를 사용한다. 공통 Z–Z feedback은 disagreement 및 bank 진단에 쓰고, DPO의 primary chosen/rejected pairs는 공통 Y–Z 64 pairs에서 만든다. 방법이 모든 공통 labels를 직접 loss에 소비하지 않아도 접근 가능한 정보와 비용을 명시한다.

**'scalarized DPO'와 'objective-wise DPO loss의 weighted sum'을 혼동하지 않는다.** 새 scalarized DPO는 다음처럼 정의한다.

1. 각 objective의 같은 shared comparisons에서 BT scalar scores `r_k(x,y)`를 fit한다. per-prompt 16-node graph를 jointly fitting한 것이라면 이를 **finite BT projection**이라고 부르고 globally trained neural reward model로 위장하지 않는다. gauge, regularization, score/temperature scale, optimizer와 train-only fit을 고정한다. 전역 reward model을 실제 학습했다면 별도 supervision/compute를 기록한다.
2. uniform `w=(1/2,1/2)`로 `r_w(x,y)=sum_k w_k r_k(x,y)`를 만들고, `r_w(y)>r_w(z)`인 쪽을 chosen으로 정한다. tie filter는 train data만으로 사전 고정한다.
3. `Delta_theta(y,z)=log[pi_theta(y|x)/pi_ref(y|x)]-log[pi_theta(z|x)/pi_ref(z|x)]`라 놓으면 loss는 `L_scalar=-E log sigmoid(beta_DPO Delta_theta(y_w,y_l))`이다. sampling policy, DPO reference와 beta 의미를 config에서 명시한다.
4. 반면 objective-wise method는 `L_multi=sum_k w_k E ell_DPO(Delta_theta; preference_k)`이며 일반적으로 위 scalar-label loss와 같지 않다. 이것을 추가하면 별도 named exploratory control로만 보고하고 main scalarized DPO 행을 몰래 대체하지 않는다. 직접 pairwise probability의 weighted average로 cyclic labels를 만드는 것도 scalar reward ranking과 다르다.
5. BT-projected Nash는 같은 BT fitted scores의 `P_k^BT(y,z)=sigmoid(r_k(y)-r_k(z))`를 원고와 같은 bargaining solver에 넣는 control이다. 이 arm이 원고의 기존 neural `BT-RM–Nash`와 정확히 같은 구현이라고 주장하지 않는다.

### 6.3 seeds, training exposure, 시간 제한

첫 비교는 7 methods의 **seed 42 전체를 완료**한다. 그 뒤 원래 계획은 seeds **42/43/44 모두**다. 단, 전체 21 runs와 최종 judging이 9월 22일까지 끝나지 않는다는 **pilot 처리량 근거가 test를 열기 전에** 확인되면 다음 fallback을 고정한다: NBPO / fixed-reference Nash / game-utilitarian / PROSPER / uniform DPO 5 methods를 3 seeds, maxmin / BT-projected Nash를 seed 42 exploratory로 둔다. seed 42의 승패나 p-value로 반복 여부를 선택하지 않는다. 실제 표에는 method별 seeds와 N을 표시한다.

neural projection 및 DPO의 optimizer steps, prompt exposures, nonpadding response tokens, unique comparisons와 GPU-hours를 각각 기록한다. NBPO만 더 오래 학습하거나 더 많은 hyperparameter를 주지 않는다. dev의 작은 동일 tuning allowance를 사전 고정하고, realized policy movement가 너무 다른 경우 matched-KL/TV comparison은 별도 control로 기록한다. neural train nMSE 개선을 generalization이나 fresh performance의 증거로 대신하지 않는다.

Primary는 기존 원고와 같은 **one outer stage**다. 두 번째 on-policy stage는 첫 stage의 선언된 평가·seeds가 끝나고 비용이 남을 때만 NBPO/fixed/util 세 methods에 같은 feedback·compute로 추가한다. 그 경우 별도 exploratory 결과이며 원고의 exact population T-step 정리를 neural convergence 증거로 바꾸지 않는다.

## 7. 평가·통계: 결과를 채울 때 지킬 계약

### 7.1 test를 열기 전에 고정할 것

- 같은 1,000 full-test prompt panel이 primary다. 원고의 새 표에서 objective 1/2, seed-wise minimum 평균, seed 수, common N, paired difference를 채운다. Delta는 해당 행 minus NBPO다. CI와 GPU-hours는 해당 numeric record 및 연결된 원본 artifact에 보존하며 없는 표 열을 새로 만들지 않는다. 전체 test를 '높은 conflict만'으로 사후 필터링하지 않는다.
- conflict stratum은 정책 결과와 독립적인 source metadata 또는 frozen baseline-candidate labels로 미리 정의한다. full panel의 primary 결과와 별도 secondary stratum 결과를 별도 raw artifact에 저장한다. stratum에서만 좋아진 결과를 전체 benchmark 우위라고 쓰지 않는다.
- generation seed, sampling, response limits, truncation/cap-hit/refusal 기록, objective rubrics, evaluator weights/revision, both-order parser와 repair rule, reference bank, bank beta를 고정한다. fresh outputs는 학습 candidates를 재사용하지 않는다.
- 총 evaluator 예산을 audit 다음에 별도로 계산한다. 예를 들어 method/seed마다 1 fresh response × 4 fixed reference responses × 2 objectives × 2 orders이면 **16,000 verdicts/1,000 prompts/arm**이다. repeats가 추가되면 그 배수를 적용한다. 21 trained arms이면 336,000 + base 평가이며 independent target panel과 repairs 비용은 별도다. 이 fresh WR 예시는 primary categorical target bank의 8-reference game value와 동일한 통계가 아니다.

### 7.2 최소한의 inference와 stage 진단

whole-prompt paired bootstrap 2,000회에서 한 prompt의 모든 methods/objectives/orders를 함께 resample한다. 최소 objective와 bank surplus는 **각 replicate에서 다시 계산**한다. 여러 policy seeds의 평균/SD와 prompt bootstrap CI를 구분한다. common prompt IDs, planned N, arm-valid N, pairwise-common N을 모두 저장한다. 비교군을 추가할 때 primary common set이 자동으로 달라지지 않도록 panel ID를 고정한다. 여러 기전 대조의 exploratory CI를 confirmatory 발견처럼 포장하지 않는다.

같은 dev prompts에서 exact target / fitted pool / fresh response transfer 진단을 별도 artifact로 수행한다. 원고의 `figures/stress_test.pdf`는 target/fitted의 paired dev panel과 별도의 fresh full-test/사전 정의 stratum panel을 표시한다. 이 그림의 dev와 test 단계를 서로 빼서 causal transfer로 해석하지 않는다. 각 단계에는 적용 가능한 방법만 표시하고 정의되지 않은 단계는 n/a로 둔다. Scalarized DPO의 NBPO식 exact categorical target을 발명하지 않으며 fresh panel에는 모든 선언 방법을 표시한다. 같은-dev 비교에서는 실제 같은 candidate/reference bank를 사용한다. categorical expectation은 argmax 후보 한 개로 대체하지 않는다. fitted pool은

`p_neural(i) proportional to p_t(i) exp(log pi_theta(y_i|x)-log pi_t(y_i|x))`

의 importance-reweighted occurrence distribution으로 계산한다. raw likelihood normalization이나 길이 정규화로 바꾸지 않는다. fitted support의 점수와 전체 LLM policy의 점수는 다르다. fresh response 한 번의 softmin을 stochastic policy population game value라고 쓰지 않는다.

기존 결과에서 categorical common N=78, fresh row별 N 및 figure N이 달랐던 문제를 반복하지 않는다. 서로 다른 N의 exact→fitted→fresh 숫자 차이를 인과적 단계 손실로 해석하지 않는다. stage 비교가 필요하면 stage 공통 IDs에서 재계산해 separate artifact에 저장한다.

새 base draw의 과거 minimum 0.4613은 보편적인 no-change 기준이 아니다. minimum-estimator bias는 분산·공분산·진짜 objective gaps에 따라 달라진다. 모든 정책에서 0.4613을 빼거나 reference null=0.461로 고정하지 않는다. base-control difference는 같은 prompt의 paired 비교로 계산한다.

### 7.3 missingness·보완 검증

invalid 출력을 tie/패배로 바꾸지 않는다. 결과 방향에 무관한 미리 고정된 동일 repair rule을 모든 invalid에 적용하고 original/repair outputs와 비용을 보존한다. valid이지만 원하지 않는 판정은 다시 뽑지 않는다. complete-case 수와 missing mass를 보고하며 후보를 삭제해 target을 renormalize하지 않는다. 필요한 경우 missing preference in [0,1]의 finite-panel bounds를 계산하되 이를 population confidence interval과 구분한다.

같은 응답의 order swap에서는 `P(y,z)+P(z,y)=1`, 동일 response hash에서는 `P(y,y)=0.5`를 확인한다. 독립 Y/Z occurrence pools의 8×8 cross matrix는 같은 집합이 아니므로 대각 0 또는 `A+A^T=0`을 강요하지 않는다. solver residual은 exact target의 검사이며 neural policy에 대한 KKT certificate가 아니다. 다른 method의 target RMS와 projection error는 같은 pair weights에서 계산한다.

## 8. 자원 운영과 마일스톤

| 기간(KST) | 반드시 끝낼 결과 | 다음 단계 조건 |
|---|---|---|
| 9월 14–15일 | 상태/보호 검증, 10-prompt throughput, Safe/Wild audit, CPU 2×2 준비 | 비용·rubric·구조·feasibility를 근거로 새 학습 panel 하나 선택 |
| 9월 16일 | dataset/split/objective/method/seed/budget/evaluator 동결 | test를 열지 않은 signed/hash manifest |
| 9월 16–19일 | shared labeling과 첫 seed 7 methods, finite 2×2 | ETA에 따라 사전 선언 seed plan 유지 |
| 9월 20–22일 | 선언한 repeats, fresh evaluation, paired CI, cost accounting | 새 실험 범위 확장 금지 |
| 9월 23일 | raw/aggregated data freeze, template 채우기, protected validation | 미완료 항목 명확히 남기기 |
| 9월 24–25일 | PDF·표·그림·재현 artifact 검증, 저자용 최소 수정 제안 | 새학습은 중요한 오류 복구 이외 시작하지 않기 |
| 9월 26일 | 제출 파일 재확인, 20:59 KST 이전 여유 확보 | 실제 제출은 저자의 별도 지시에 따름 |

실제 시작이 늦으면 남은 시간으로 이 일정을 압축하되 official 마감을 연장하지 않는다. 평가·컴파일 시간을 학습 ETA에서 빠뜨리지 않는다. 새 유료 자원이나 GPU를 추가하지 않는다. 4-GPU 학습을 효율적으로 쓰고 judge 전용 GPU를 무조건 상시 예약하지 않는다. model load/unload·VRAM·tokens/s·I/O를 포함해 backfill을 계획하며 높은 우선순위 4-GPU job을 긴 작은 작업 때문에 지연시키지 않는다.

30분 간격으로 실제 snapshot을 다음 형식으로 stdout와 상태 파일에 출력한다. 보고 시점마다 보호 검증 → 결과 renderer → 정상 PDF 빌드 경로를 실행하되 충돌을 lock으로 막고 마지막 정상 PDF를 유지한다. 원고가 잠금 중이거나 검증이 실패하면 상태 보고와 GPU 작업은 계속하고 PDF를 덮어쓰지 않는다.

```text
[KST] 새 panel / split hash / 보호 검증: PASS 또는 실패 이유
완료: audit a/b, finite cells c/4, methods×seeds d/e, final evaluation f/g
진행: 실제 job·steps/samples·throughput; 다음 READY job
GPU0/1: 작업·util·VRAM | GPU2/3: 작업·util·VRAM
ETA: audit / declared training / final judging+CI (실측 기반 범위)
이번 결과: 채운 template key와 raw artifact; 없으면 새 결과 없음
PDF: 마지막 성공 시각·경로·source/result hash
Blocker: 이유·영향·복구 또는 NOT_APPLICABLE 근거
```

## 9. 최종 산출물과 지금 할 일

완료 산출물은 보호된 원고로 정상 빌드한 `main_v6.pdf`, 채워진 `templates/results.json` / `templates/results_auto.tex`, 원본 수치에서 생성한 `figures/stress_test.pdf`, frozen manifests/configs, raw-to-table 집계 코드, 통계/비용/처리량 기록, dataset decision, `review_evidence_updates.md`, 필요할 때만 `proposed_manuscript_patch.diff`다. 원고 문장을 바꾸지 않은 채 결과가 주장과 충돌하면 `additional_findings.md`에 정확히 적는다. 승리/유의성/완료를 검증 전 주장하지 않는다.

지금 바로 현재 상태와 보호 규칙을 확인하고, 기존 작업을 인계하며, CPU finite-game 준비와 10-prompt throughput pilot을 실행하라. 첫 보고에는 실제 실행 중인 작업, raw/log 경로, reporter 확인, protected validation 결과, 마지막 정상 PDF와 다음 milestone ETA를 보여라. 계획의 요약만 회신하고 멈추지 않는다.
