# NBPO: 8B 일반 지시 수행을 중심으로 한 본 실험과 조기 마감 계획

2026-09-10. 근거는 최신 첨부 main_v6(1).tex/main_v6(4).pdf, 사용자가 전달한 09:13 Claude 실행 보고, 아래 원 논문 및 공식 데이터 문서다. 원격 서버의 새 raw artifact는 이 작업에서 직접 재계산하지 않았다. 아래 새로운 설정은 실행할 제안이며 완료된 실험으로 표시하지 않는다.

## 1. 논문의 중심을 바꾸는 것이 맞다

**본문의 중심은 실제 8B 정책의 다목적 성능과 일반 능력이어야 한다.** Neural projection과 finite-pool solver 진단은 구현을 설명하는 근거다. 방법의 가치 자체는 생성된 답변의 품질, 여러 목적 사이의 절충, 강한 비교군과의 차이로 판단해야 한다. 따라서 이번 수정본은 일반 instruction 4-objective 결과 표 → 4차원 결과의 두 투영 그림 → 일반 capability/safety 표 → 완료된 SafeRLHF 보완 결과 순서로 바꿨다.

현재도 8B 정책 학습을 한 것이다. 부족한 것은 모델 크기가 아니라 **다양한 일반 지시 영역에서의 효과, 독립 평가, 외부 방법 대비 이점**이다. SafeRLHF에서 좋은 target fit을 얻었다는 사실만으로 이 세 가지가 해결되지는 않는다.

Pareto 관점에서도 두 종류의 주장을 분리해야 한다. 이론은 가정하의 population bargaining optimum에 관한 것이다. 한 번의 finite-pool proximal update와 neural projection을 적용한 유한 LLM checkpoint는 자동으로 그 optimum이 되지 않는다. 실험에서는 공통 평가 조건에서 비교한 정책들의 네 목적 벡터를 측정하고, 그 집합에서 NBPO가 지배되는지, 어떤 절충을 제공하는지를 보여준다. 이는 empirical comparison-set nondominance이며 전역 Pareto 최적성 증명이 아니다.

## 2. 밤사이 결과가 말해 주는 것

| Family | nMSE | sign | worst teacher-defined surplus |
|---|---:|---:|---:|
| Nash, 3 policy seeds | .6845±.0048 | .7009±.0016 | +.1215±.0028 |
| L1-matched utilitarian, 3 policy seeds | .6737±.0242 | .7040±.0058 | +.1220±.0029 |

위 ±의 종류는 원본 집계 코드 확인 전까지 '보고된 spread'다. Both>0의 .725/.730은 집계 단위가 확인되지 않아 본문 표에서 뺐다. 이것을 정책 전체의 individual-rationality 만족률로 이름 붙이지 않는다.

이 결과는 neural fitting과 양의 teacher-defined surplus가 한 seed의 우연만은 아니라는 근거다. 그러나 **현재 패널은 Nash aggregation의 추가 이점을 보여주지 않는다.** 동일 seed·동일 prompt의 차이는 작고 방향이 일정하지 않다. 여섯 개 목적별 paired CI 모두 0을 포함한다. 차이가 검출되지 않은 것이 동등성 증명은 아니다.

두 teacher target은 mean TV .0008만 다르다. 별도로 pair-target RMS 차이 .0034는 원 target RMS1.125의 약 .3%다. TV와 RMS 비율을 같은 양으로 쓰면 안 된다. 두 target이 사실상 매우 가까운 설정에서 최종 정책도 유사한 것은 자연스럽다. Lambda 비대칭만을 키우거나 Nash가 이기는 데이터를 찾는 것이 다음 실험의 목적이어서는 안 된다.

현재 IFEval 원본 보고 값으로 계산한 평균/sample SD는 Nash .748±.013, utilitarian .745±.025, base .752다. 두 방법의 평균은 base보다 조금 낮은 점추정이다. Metric variant와 paired uncertainty 확인 없이 'base와 동등', '모든 능력 향상'이라고 쓰지 않는다. GSM8K .862–.870, Alpaca proxy .496–.516, HarmBench .266–.275는 여섯 arm 전체 범위이므로 family별 평균으로 채우지 않았다.

거절 감사도 중요하다. 동일 문자열247/1000, 짧은 답변450/1000, 거절 문두621/1000, 짧은 답변 중 거절448/450은 서로 다른 분모와 범주다. Teacher는26,790/84,000 candidate occurrences를 잘라 보았고, 잘린 경우 중앙값232 tokens를 잃는다. 이 제한 때문에 길고 일반적인 답변으로 확장할 때 teacher를 그대로 두어서는 안 된다.

## 3. 왜 지금까지 SafeRLHF였고, 선행 연구는 무엇을 썼나

SafeRLHF는 같은 답변 쌍에 helpfulness/harmlessness의 인간 비교가 있어 두 목적 GPM과 BT를 맞춰 비교하기 쉽다. 기존 teacher, split, solver cache도 있어 첫 구현 검증 비용이 낮았다. 이는 지금까지의 자료로 추론할 수 있는 실무적 이유이며 NBPO 수학이 요구하는 데이터셋은 아니다. 현재 패널에서는 거절을 더 유용하게 만드는 답변이 두 목적을 함께 올리고, Nash/utilitarian의 선택이 거의 일치한다. 이 때문에 일반 alignment와 bargaining의 구별력을 대표하기 어렵다.

| 논문 | 학습 데이터와 피드백 | 일반화/절충 평가에서 참고할 점 |
|---|---|---|
| NLHF | Reddit TL;DR 요약 비교; preference model과 policy 학습 역할 분리 | 처음부터 8B chat benchmark 논문은 아니다. Pairwise preference를 직접 최적화하는 정식화가 핵심이다. |
| SPPO | UltraFeedback 일반 instruction prompts; 생성 응답을 PairRM으로 비교 | 일반 instruction 정책 최적화와 AlpacaEval/Arena 평가를 연결한다. Released 답변을 그대로 학습하는 것과 prompt만 사용하는 것을 구분한다. |
| MOPO | HH-RLHF와 Reddit summary 등; 여러 objective의 pairwise preference와 제약 | Llama-3.1-8B 실험을 포함하며 주 목적과 다른 목적 제약을 비교한다. 논문의 모든 frontier 그림이 8B인 것은 아니다. |
| PROSPER | WildChat에서 구성한 WildChecklists와 checklist 피드백 | Qwen2.5-3B/7B 정책에 일반 chat 평가와 capability를 함께 둔다. 원래 dense generative-judge 비용을 그대로 복제하면 현재 예산과 맞지 않는다. |

근거: [NLHF](https://arxiv.org/pdf/2312.00886), [SPPO](https://arxiv.org/html/2405.00675v3), [MOPO](https://arxiv.org/html/2505.10892v2), [PROSPER](https://arxiv.org/html/2602.19041v1).

따라서 모든 논문에 공통인 단일 필수 데이터셋은 없다. 공통 요건은 **목적 정의에 맞는 학습 피드백, 독립적인 생성 정책 평가, 실제 다른 절충을 제공하는 비교군**이다. 이번에는 source와 네 속성이 명확한 UltraFeedback을 주 데이터로 고정한다. WildChecklists는 유용하지만 variable checklist/schema와 costly pairwise oracle를 새로 맞추는 추가 개발을 이번 조기 마감의 필수 경로에 올리지 않는다. [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback), [WildChecklists](https://huggingface.co/datasets/viswavi/wildchecklists).

## 4. 새 주 실험: UF-4

### 데이터와 split

`openbmb/UltraFeedback` 원본에서 `ultrachat`, `sharegpt`, `evol_instruct` sources를 사용한다. 문서상 원본 합계39,878 prompts는 필터 후 실제 크기가 아니다. 원 데이터에 TruthfulQA source도 포함되므로 source 제외와 benchmark 중복 제거가 필요하다. 모든 분할은 exact/near-duplicate prompt group 단위로 하고 revision과 hash를 고정한다. [공식 데이터 구성](https://huggingface.co/datasets/openbmb/UltraFeedback).

| 역할 | 목표 prompts | 피드백/응답 사용 |
|---|---:|---|
| Teacher train | 20,000 | 공개 completion4개와 네 속성 rating |
| Teacher dev | 2,000 | Teacher 선택 및 calibration |
| Policy train | 10,000 | Prompt만 사용, 공통 base에서 새 응답 생성 |
| Policy dev | 1,000 | Hyperparameter/모델 선택, transfer audit |
| Final objective evaluation | 2,000 | 학습/선택 동결 후 평가 |

합계35,000의 서로 겹치지 않는 prompt group이다. 필터 후 적격 unique prompts가30,000–34,999개이면 policy train만5,000으로 줄이는 fallback을 결과를 보기 전에 고정한다. 그보다 적으면 실제 부족과 scope를 보고한다. 목적은 instruction following, truthfulness, honesty, helpfulness다. Honesty를 safety라고 바꾸지 않는다.

### 새 teacher와 정책

Teacher는 native8,192 context를 지원하는 **ModernBERT-base**를 새로 학습한다. 같은 backbone/annotation/split에서 네 head antisymmetric GPM과 scalar BT를 맞춘다. GPM은 `sigmoid((f(A,B)-f(B,A))/2)`, BT는 `sigmoid(r(A)-r(B))`로 방향을 고정한다. Rating 비교 target은 승1/패0/동률.5이며 누락은 mask다. PM20k에서 unordered6pairs씩120k joint examples다. 이 rating-induced label은 transitive이므로 자연 데이터의 인간 비추이성을 입증한다고 주장하지 않는다. [ModernBERT 모델 설명](https://huggingface.co/answerdotai/ModernBERT-base).

Teacher 초기 제안은 LR2e-5, global batch32pairs, AdamW, bf16,1epoch부터 시작하고 최대2epochs 내 dev NLL로 선택한다. Native context를 넘는 입력의 대칭 budget과 truncation을 기록한다. PMdev 외에도 새 base-generated dev pairs200개에서 독립 judge와의 일치를 확인해 응답 분포 이동을 점검한다. 정책 seed42/43/44는 동일 frozen teacher seed42에 조건부다.

정책은 **Llama-3.1-8B-Instruct full fine-tuning**이다. 기존 성공한 resolved recipe를 출발점으로 확인한 뒤 peak LR scale5e-7, global32pairs, bf16, clip1, 자체cosine schedule, warmup.1을 사용한다. 새10k에는1250updates, fallback5k에는625updates로 평균 prompt당4pair exposures를 맞춘다. 이는 새 제안 예산이며 최적 horizon을 이미 찾았다는 뜻은 아니다. NBPO는 one outer stage로 비교를 시작한다. Prompt1024/response1024/total2048을 출발값으로 길이 손실을 실측한다.

새 response pool은 prompt당 learner8+reference8로 공통 생성한다. GPM의8×8 cross와 reference 내28pairs를 양 순서로 평가하면10k에서184만 joint encoder passes다. 네 head라고 네 배를 다시 곱하지 않는다. 로컬 Qwen3-14B는 이 전체 training tensor에 쓰지 않고, 최종 생성 정책의 독립 평가에 쓴다. 먼저200prompts로 전체 처리량을 측정한다.

### 비교군과 Pareto 그림

필수는 Base, NBPO, L1-matched game-utilitarian, 실제 PROSPER adaptation, 실제 MOPO adaptation, BT-scalarized DPO다. 주 네 학습 family는3policy seeds를 목표로 한다. Game-maxmin은 seed42 supporting control이다. 이미 승인된 Safe mechanism controls는 가능한 슬롯에서 완료하되 완료된 Nash/utilitarian3seeds를 다시 돌리지 않는다.

DPO는 `(IF,Truth,Honesty,Help)` 순서로 네 vertex, uniform, `(.1,.1,.1,.7)`, `(.1,.7,.1,.1)`의7weights를 모두seed42에서 실행한다. Uniform과 사전 고정 dev 규칙으로 정한 nearest competitor의 추가 seeds는 예산에 맞춰 수행한다. **본문 DPO는 uniform으로 고정**하고 좋은 weight를 metric별로 바꿔 넣지 않는다. 실제 목적/update를 구현하지 못한 variant를 PROSPER/MOPO라고 부르지 않는다.

독립 Qwen3-14B가 final2k에서 네 criterion별 A/B/tie를 평가한다. Base response cache, rubric, position swap, decoding을 공통으로 고정한다. 네 점수는 기준 base 대비 WR이고 teacher의 game surplus와 별개다. 그림은 Help–Truth, IF–Honesty의 두 투영이며 nondominance는 네 좌표 모두로 계산한다. 모든7DPOpoints를 남기고 측정하지 않은 연속 frontier를 선으로 연결하지 않는다. Prompt-cluster CI와 training-seed SD를 구별한다.

### General capability 평가

모든 주 방법에 IFEval(strict prompt/instruction), GSM8K EM, Alpaca805 local proxy, ArenaHard-v2 hard500/creative250 local proxy, HarmBench, XSTest를 적용한다. Base와 주 대표군에는 MMLU, ARC-Challenge, HellaSwag도 같은 harness 설정으로 측정한다. MT-Bench와 추가 stochastic 평가의 우선순위는 낮춘다. 안전한 prompt의 over-refusal과 유해 요청 성공률을 별도로 봐 무조건 거절을 성능 향상으로 세지 않는다.

새로운 유료 API 호출은0건이다. 공개 GPT-4 annotation 재사용은 허용하되 그 출처를 밝힌다. 로컬 judge WR은 공식 AlpacaEval2 LC 또는 공식 Arena leaderboard 점수가 아니다. 모든 benchmark를 통해 모든 능력을 포괄하거나 최고의 성능을 보장할 수는 없다.

## 5. 제출 마감보다 여유 있게 끝내는 일정

공식 abstract 마감은9/18 23:59 AoE=**9/19 20:59 KST**, full paper는9/25 23:59 AoE=**9/26 20:59 KST**다. 본문은9페이지 한도다. [ICLR 2027 공식 안내](https://iclr.cc/Conferences/2027/AuthorGuidelines).

| 내부 일정, KST | 완료할 일 |
|---|---|
| 9/10 | 데이터/splits 고정, 새 teacher 실제 학습 시작; Safe controls 병행 |
| 9/11 18:00 | GPM/BT dev 검증·동결,200-prompt 비용 실측, 공통pool/tensor |
| 9/12 | 첫 UF-4 8B checkpoint와 dev 결과 |
| 9/13–16 18:00 | 비교군·3seeds·7DPOweights, **학습과 모델 선택 동결** |
| 9/17–18 18:00 | Final objective/capability 평가, **수치 동결** |
| 9/19–20 | 본문·부록·코드·표·그림 완성; abstract 별도 조기 제출 |
| 9/21–25 | 검수·누락 복구·재현·조기 제출 buffer |

이 일정이면 실험 수치와 공식 full 마감 사이에 약8일이 남는다. GPU4장을8일간60% 실가동하면460.8GPU-hours다. 계획 범위250–450GPU-hours는 throughput 전 기획치이지 완료 보장이 아니다. Teacher scoring·generation·judge가 병목일 수 있으므로 첫200prompts의 실측으로 그날 범위를 조정한다.

초과하면 DPO 추가반복, 부가 mechanism 반복, Safe 확장, MT-Bench를 먼저 줄인다. 필수 새 데이터 주 비교를 optional로 돌리거나 final 점수가 나쁜 비교군을 빼지 않는다. Teacher/data 핵심 경로가9/11 18시까지 막히면 즉시 실패 원인과 가능한 scope를 보고한다. 더 큰 학습이 항상 더 좋은 성능을 만든다는 가정으로 horizon sweep을 먼저 시작하지 않는다.

Claude 대화 복귀에 의존하는 timer 대신 **지속 실행되는 의존성 queue**가 필요하다. READY task가 있고 GPU가 비면 자동으로 다음 job을 실제 실행한다. PID, return code, artifact hash, GPU 요구량을 연결해 RUNNING/DONE을 판정한다. 살아 있는 shell 수와 '곧 시작' 문장은 작업 증거가 아니다. 이번 prompt에는 이를 명시했다.

## 6. 실제 원고 수정 범위와 사용법

핵심 수식, 알고리즘, 정리 및 증명 구조는 유지했다. Experiments를 재배치하고 abstract/contribution/limitations에서 증거 범위를 맞췄다. 기존 controlled figure, one-seed realization, 긴 schedule/pool diagnostics는 appendix로 옮겼다. Null과 실패 증거는 삭제하지 않았다. 거절 분모와 teacher truncation을 바로잡았고 불필요하게 긴 caption을 줄였다.

본문 산출물은 다음이다.

| 위치 | 내용 | 바로 채울 입력 |
|---|---|---|
| Table1 | UF-4 네 objective 비교 | data/uf_objectives.csv |
| Figure1 | 네 목적의 두2D 투영, 실제 측정점만 | data/uf_tradeoffs.csv |
| Table2 | General capability, local chat, safety | data/new_capability_results.csv |
| Table3 | 완료된 SafeRLHF3seed/null | data/safe_three_seed_results.csv |
| Appendix | Seed별 paired차이·과거realization·solver/audit | 나머지data/*.csv |

새 표의 빈칸은 미측정이다. 그림도 DRAFT TEMPLATE이라고 명시한 빈 틀이며 가상점은 없다. 모든 새 수치가 들어온 다음 prospective 문장을 실제 실행 protocol/result로 고치고, 빈 figure/table이 남으면 최종 제출 전에 제거한다. Caption은 채택된 benchmark protocol과 seed 단위를 확인한 뒤 최종 확정한다.

패키지 내 `data/MAIN_EXPERIMENTS_SCHEMA.md`가 열 정의와 provenance 규칙을 담는다. 실행은 `python3 scripts/render_tables.py`, `python3 scripts/plot_uf_tradeoffs.py`, `latexmk -pdf -interaction=nonstopmode -halt-on-error main_v6.tex`다. 최신 서버 원고에 diff로 통합해야 하며 오래된 public checkout으로 새 실험 코드를 덮어쓰지 않는다.

Claude 실행은 제공한 `NBPO_early_main_prompt_20260910.md`를 사용한다. 코딩·수식 감사에는 `/model opus`와 지원되는 `xhigh`, 일상 구현·queue 작업에는 `high`를 권한다. 실제 model/effort 지원을 확인하고 실행 중 GPU job은 대화와 독립적으로 지속시킨다. [Claude Code 공식 설정](https://code.claude.com/docs/en/model-config).
