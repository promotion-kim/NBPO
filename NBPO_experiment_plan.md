# NBPO: 원고 수정과 제출 전 실험 판단

작성 기준: 2026-09-14. 제공된 12:17 KST TeX/PDF의 수치를 검토한 편집·실험 설계다. 원시 로그나 GPU 저장소를 실행 검증한 결과가 아니며, 이 작업에서 새 ML 실험은 실행하지 않았다.

## 1. 판단: 필요한 것은 '순환이 많은 데이터'보다 기전을 구별할 수 있는 피드백이다

**순환성과 objective trade-off는 관련 있는 조건이지만 NBPO 우위의 충분조건이 아니다.** 두 개를 서로 다른 질문으로 분리해야 한다.

- 같은 objective에서 A>B, B>C, C>A가 반복되어 유지되면 scalar BT representation과 고정 상대 비교가 놓치는 구조를 시험할 수 있다. PROSPER나 다른 game-based 방법도 이를 다룰 수 있으므로, 이것만으로 Nash aggregation을 지지하지는 않는다.
- objective 간 충돌은 compromise 선택을 시험한다. 그러나 모든 방법이 같은 feasible target을 선택한다면 충돌하는 원래 label이 많아도 정책 차이가 작다. 반대로 충돌이 너무 강하면 모든 objective가 reference보다 좋아지는 joint strict feasibility 자체가 없어질 수 있다.
- Nash는 reference-relative surplus의 균형과 효율을 선택한다. 모든 objective에서 다른 정책보다 높은 win rate, 가장 높은 minimum win rate, 모든 사용자의 선호를 동시에 보장하지 않는다. 따라서 평균·최솟값·Nash welfare 하나로 '보편적 최선'을 선언하면 안 된다.

형식적으로 확인할 연결은 다음과 같다.

`반복 가능한 preference 구조 → 공동 개선 가능성 → 서로 다른 exact targets → 독립 평가의 target 품질 → fitted/fresh policy 결과`

데이터 선택은 이 연결을 관찰할 수 있게 해야 한다. NBPO가 가장 크게 이기는 데이터·seed·judge만 남기는 선택은 피한다.

## 2. 현재 수치가 실제로 말하는 것

| 관측 | 의미 | 아직 결론낼 수 없는 것 |
|---|---|---|
| UF main: NBPO minimum .4955, uniform DPO .5044 | 현재 generated-policy 결과는 NBPO 우위를 지지하지 않는다 | solver나 population theorem이 틀렸다는 결론 |
| Exact target의 NBPO–util TV .0206, NBPO–fixed TV .0240 (동일 dev 200) | 학습 전 target 자체가 가깝다 | projection만 고치면 큰 격차가 생긴다는 보장 |
| 같은 78 prompts에서 NBPO exact–source IF +.0271, Help +.0505 | candidate target에는 독립 judge가 보상하는 신호가 있다 | Nash가 다른 aggregation보다 낫다는 증거 |
| NBPO와 각 control의 direct-win 최대 차이 .0018 | 현재 target panel은 aggregation을 잘 구별하지 못한다 | 세 target의 모든 pairwise 최대 차이가 .0018이라는 주장; fixed–util honesty 차이는 .0025 |
| Fitted pool의 독립 점수는 여러 항목에서 개선되지만 target KL은 약 1.457 | target 재현과 독립 품질은 다른 축이다 | target KL이 커지면 무조건 성능이 나빠진다는 인과 주장 |
| Projection pilot fresh minimum 차이 +.0044, CI [-.0366,+.0426] | 현재 pilot에서 개선이 통계적으로 분해되지 않는다 | 두 방법이 동등하다거나 새 recipe가 더 좋다는 결론 |
| 기존 SafeRLHF teacher target TV 약 .0008 | 원래 human conflict만으로 생성 후보의 training signal이 달라지지 않았다 | 같은 teacher로 데이터 이름만 바꾸면 개선된다는 기대 |

현재 순환 audit도 중요한 부정적 결과다. 60,000 direct judgments에서 반복-cycle excess나 order-robust witness를 확인하지 못했다. 이것은 자연 선호가 transitive라는 증명이 아니라, **현재 이 데이터·후보·rubric·judge 계약에서 representation 장점을 입증하지 못했다는 뜻**이다.

수정본은 추가로 다음을 바로잡았다. `.4613`은 새 base draw에서 보고된 minimum 통계이지 모든 정책의 공통 no-change offset이 아니다. 표시된 objective 평균들의 최솟값은 `.4700`이므로 원래 집계 정의와 분모를 확인해야 한다. Fresh DPO는 IF/Truth/Help에서는 높지만 honesty에서는 NBPO `.5221`보다 낮은 `.5171`이다. 서로 다른 prompt 집합의 exact/fitted/fresh 숫자를 단계별 인과 효과로 빼지 않는다. Averaged pair loss에서 candidate가 일곱 번 나온다는 이유만으로 gradient가 일곱 배가 되는 것도 아니다.

## 3. 데이터셋 우선순위

| 후보 | 제공되는 증거와 한계 | 이번 제출에서의 역할 |
|---|---|---|
| **PKU-SafeRLHF** | 동일 response pair에 helpfulness와 harmlessness를 분리한 human feedback. objective conflict를 직접 조사하기 좋다. 두 응답짜리 레코드는 완전한 triangle을 보장하지 않는다 | **우선 학습 후보.** 새 응답에 fixed help/safety direct pairwise feedback을 얻어 기존 scalar teacher의 약한 separation을 다시 검사 |
| **WildChecklists** | prompt-specific checklist와 직접 pairwise judge를 사용하는 PROSPER의 학습 기반. 항목 수와 의미가 prompt마다 다르다 | **우선 순환 audit.** native item-level cycle 확인. NBPO 학습은 고정 global rubric 계약을 별도로 정의했을 때만 진행 |
| **HelpSteer2 + Preference** | 여러 attribute rating과 이를 보완하는 direct preference가 있어 scalar score와 pairwise feedback의 차이를 비교하기 좋다 | 보조 후보. 전체 preference가 각 attribute의 complete triangle인 것은 아니므로 즉시 cyclic multi-objective benchmark로 쓰지 않음 |
| **HelpSteer3-Preference** | 다양한 task/language의 human preference | 새로운 다목적 cycle benchmark를 만들기에는 고정 objective별 재판정 비용이 든다. 이번 최우선 아님 |
| **PRISM** | 참여자 특성과 실제 상호작용의 평가를 연결한 pluralistic feedback | 사용자집단을 bargaining party로 삼는 장기 방향에는 적합. 지금은 party 정의·coverage·reference 설계까지 바뀌므로 범위 확대 위험 |
| **기존 UltraFeedback** | scalar attribute로 유도한 label은 objective별 transitive | 삭제하지 않고 기존 결과를 **transitive-supervision control**로 유지 |

근거: [PKU-SafeRLHF 원 논문](https://arxiv.org/abs/2406.15513), [HelpSteer2](https://arxiv.org/abs/2406.08673), [HelpSteer2-Preference](https://arxiv.org/abs/2410.01257), [HelpSteer3-Preference](https://arxiv.org/abs/2505.11475), [PRISM](https://arxiv.org/abs/2404.16019). 표의 실험 우선순위는 해당 자료에 기반한 이번 프로젝트의 설계 판단이다.

### PROSPER에서 실제로 가져올 부분

최신 v2는 WildChecklists의 prompt-specific rubrics, Qwen2.5 policy와 Qwen3-14B judge를 사용한다. 여러 반복과 양쪽 presentation order를 평균하며, held-out 100 prompts에서 any-cycle과 no-Condorcet 비율을 구분한다. 이 두 지표는 약한 응답들 사이의 순환과 최선 선택의 어려움을 구별한다. 최신 v2는 RLCF를 REBEL pipeline에서 재구현하므로 이전 버전의 baseline 차이를 그대로 인용하면 안 된다. 보고된 scoring 비용은 model/epoch당 16 H100 × 약 50시간이다. 이는 논문 recipe를 그대로 따라갈 때의 비용 근거이며 H200 처리시간으로 직접 환산할 수 없다. [PROSPER v2, §§5–6 및 Appendix C](https://arxiv.org/html/2602.19041v2)

**NBPO에 가져올 것은 데이터 이름보다 direct pairwise rubric 평가와 구조 audit다.** PROSPER의 `k번째 item`은 prompt마다 다른 의미다. 현재 NBPO의 shared global λ_k에 그 위치를 그대로 넣으면 다른 bargaining party를 같은 좌표로 합치게 된다. 본문 이론을 보존하는 이번 수정에서는 Wild native items를 순환 audit에 사용하고, finetuning은 global help/safety가 명확한 Safe panel을 기본값으로 둔다. Wild에서 고정 rubric을 새로 정의해 학습하면 이를 별도 adaptation으로 명명한다. [WildChecklists 배포처](https://huggingface.co/datasets/viswavi/wildchecklists/tree/main)

## 4. 최소 실험 묶음과 실제 비용

### P0. 기존 결과 정합성 — CPU 우선

원본 manifest에서 common N, seed 수, `.4613` minimum 집계, Table/figure denominator, evaluator 실제 weight hash를 정리한다. 기존 값을 임의로 덮어쓰지 말고 정정 근거와 최소 patch를 남긴다. 새 independent judge가 없으면 held-out teacher 결과를 independent evaluation으로 부르지 않는다.

### P1. 200-prompt screening 두 개 — 학습 전에 판단

Safe와 Wild 각각 점수와 무관하게 200 prompts, 8 responses를 고정한다. 28 unordered pairs × 2 orders × 2 repeats를 judge한다. 추가 확인은 사전 선택한 50 prompts의 모든 edges에 5 repeats/order를 더 한다. Same-objective repeated cycles, no-Condorcet, top-relevant cycles, objective conflict, missingness와 BT-noise control을 계산한다. 원래 두 응답의 human labels와 새 여덟 응답의 AI judgments를 섞지 않는다.

Safe K=2에서 screening **44,800 verdicts/judge**, confirmation 추가 **28,000**이다. Wild 비용은 실제 item 수의 합에 비례한다. 먼저 10 prompts로 loading, tokens/s, repair를 포함한 실측 ETA를 구한다. 첫 두 repetitions와 추가 confirmation은 분리해 사용한다. N=8의 possible triples는 56이므로 raw cycle count 대신 비율·분모를 보고한다.

8개 동일 후보의 square game은 저렴한 audit surrogate다. 본 학습의 독립 8Y+8Z와 동일한 문제로 부르면 안 된다. Fixed-K surrogate에서 shared global λ와 aggregate max-margin feasibility를 검사한다. 한 chosen target의 음수 surplus는 전체 infeasibility 증거가 아니다. Feasible witness는 가능성을 보이고, infeasibility 판정에는 신뢰할 만한 upper bound가 필요하다.

**진행 기준:** 루브릭 의미가 일관되고, 반복 신호·충돌이 측정 가능하며, 공동 개선이 가능하고, exact targets가 수치 오차 이상으로 구별되며, 전체 비용이 마감 안에 들어가는지 확인한다. NBPO의 유의한 승리를 dataset 채택 조건으로 두지는 않는다. 기준 미달이면 그 부정적 audit를 보고하고 대규모 학습을 중단한다. UF는 이미 수행한 N=4 audit를 N=8 템플릿에 복사하지 않는다. 새로운 동일 계약 audit를 수행하지 않으면 해당 셀은 미측정으로 남긴다.

### P2. Cyclicity × conflict 2×2 controlled test — CPU 중심

기존 finite-game solver를 이용해 두 요인을 따로 변화시킨다. Payoff scale/reference/support와 positive-feasibility 범위를 맞추고, 실제 cycle/conflict 지표로 조작을 검증한다. 각 cell에 사전 고정한 동일 20 instance seeds를 사용하되 기존 generator가 더 많은 seeds를 요구하면 실행 전 동결한다. Adaptive-vs-fixed/BT는 representation의 질문, Nash-vs-util/maxmin/PROSPER는 aggregation의 질문이다. NBPO가 정의상 최적화하는 Nash welfare만으로 우위를 주장하지 않는다. 실패·무차이 cell도 모두 보고한다.

### P3. 새 neural dataset은 하나만

기본 Safe direct-feedback panel은 **2,000 train / 500 dev / 1,000 test**, 같은 base 및 8Y+8Z, one outer stage다. Methods는 NBPO/fixed Nash/utilitarian/global maxmin/BT-projected Nash/PROSPER adaptation/uniform scalarized DPO + base. 모두 같은 label cache와 사전 고정 tuning·token budget을 사용한다. 첫 seed 42는 모든 방법을 끝내고, 원래 계획은 42/43/44다. 비용상 축소가 필요하면 test 전에 NBPO/fixed/util/PROSPER/DPO 3 seeds와 나머지 exploratory 1 seed로 동결한다.

한 prompt의 64 Y–Z + 28 Z–Z, K=2, 2 orders, 2 repeats는 **736 verdicts**다. 따라서 train+dev만 **1.84M teacher verdicts**다. 처리량이 안 맞으면 결과를 보기 전 전 방법의 train N을 1,000으로 줄여 **1.104M**으로 바꾼다. DPO가 Y–Y labels를 추가로 쓰는 비용은 여기에 없으므로 기본 DPO는 공통 Y–Z 정보에서 scalar BT projection과 uniform ranking을 사용한다. 기존 neural BT-RM baseline과 새 finite BT projection은 구별한다.

Final은 full distribution 1,000 prompts가 primary다. One fresh response × four common references × two objectives × two orders이면 **16,000 verdicts/arm**이며 21 trained arms는 **336,000 + base**다. 별도 target/fitted diagnostic·repair·tuning 비용도 더해야 한다. 독립 판단으로 reference wins, objective vector, seed-wise minimum과 paired CI를 보고하고, 비용·capability retention도 함께 본다. 사전에 정의한 conflict stratum은 secondary artifact로 유지한다.

### 무엇을 이번에 하지 않을지

새 backbone 여러 개, 전체 PROSPER recipe 재현, 새 MOPO/HT-MNPO population 구현, 대규모 human triple 수집, dataset 무한 탐색은 이번 필수 경로가 아니다. 두 번째 outer stage는 첫 stage의 선언된 seeds·평가가 끝나고 자원이 남을 때만 matched NBPO/fixed/util에 같은 비용으로 추가한다. 기존 UF DPO sweep은 이미 선언한 작업을 보존하되 새 주요 진단보다 범위를 늘리지 않는다.

## 5. 제출 일정

공식 abstract 마감은 **9월 18일 23:59 AoE = 9월 19일 20:59 KST**, full paper는 **9월 25일 23:59 AoE = 9월 26일 20:59 KST**다. 이 일정은 2026-09-14 확인 기준이다. [ICLR 2027 CFP](https://iclr.cc/Conferences/2027/CallForPapers)

| 날짜(KST) | 완료 기준 |
|---|---|
| 9/14–15 | 기존 수치 정합성, 10-prompt 처리량, Safe/Wild screening·confirmation, finite 2×2 |
| 9/16 | 데이터 하나와 모든 split/rubric/method/seed/평가 계약 동결 |
| 9/16–19 | Shared labeling, 모든 방법 첫 seed; abstract 등록은 저자 진행 |
| 9/20–22 | 선언한 seeds, independent fresh evaluation, paired CI |
| 9/23 | 데이터 동결, 새 template 수치 반영 |
| 9/24–25 | 결과 해석·페이지·재현 artifact 확인, 필요한 본문 변경은 별도 최소 diff |
| 9/26 | 제출 전 여유 확보; 새 광범위한 실험 시작 금지 |

## 6. 원고 구성과 결과에 따른 주장

원고의 정의·알고리즘·정리·증명과 기존 수치 표는 유지했다. Experiment의 반복적인 진행 상황과 해석을 줄이고, 상세 capability/realization 결과와 UF projection figure는 부록으로 옮겼다. 새 실험은 독립된 부록 template 3개 표와 그림 1개로 추가했다. Main은 현재 결과와 한계를 짧게 설명하고 prospective test는 완료된 증거와 구별한다.

공식 최초 제출 본문 제한은 **9쪽**이며 참고문헌·부록과 지정된 statement는 별도다. 제공된 clean PDF는 페이지 확인용이며 미측정 template이 있는 연구 초안이다. 결과가 채워진 뒤 핵심 새 결과가 강하면 저자가 검토할 최소 교체 diff로 main figure/table를 승격하는 편이 좋다. Claude가 자동으로 기존 주장과 리뷰 답변을 다시 쓰게 하지 않았다. [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)

- **Stable cycle + target separation + fresh improvement:** representation/compromise 기전이 어느 비교에서 작동하는지 제한적으로 주장한다. Full test와 conflict stratum을 모두 보여준다.
- **Stable cycle만 있고 Nash–util/PROSPER 차이 없음:** game representation 필요성은 지지할 수 있어도 Nash 고유 이점은 미입증이다.
- **Exact target만 좋고 fresh 차이 없음:** target construction의 가능성과 neural realization 한계를 분리한다.
- **모든 단계에서 무차이:** 데이터의 구조적 한계와 정확한 solver/theory를 중심으로 보고한다. 이 경우 현재 형태의 empirical contribution은 여전히 약하며, 데이터 이름 변경으로 해결됐다고 포장하면 안 된다.

합격 가능성을 올리는 방향은 NBPO의 승리를 미리 정하는 것이 아니라, **어떤 조건에서 어떤 비교를 이겨야 기전 주장이 성립하는지**를 분명히 하고 그 조건을 실제로 검증하는 것이다.
