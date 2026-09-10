# 교수님 GPT 리뷰 34개 해석과 main_v6 수정 대응

대상은 첨부 `main(20260910-022532).tex`의 R01–R34와 직전 수정본 `main_v6`다. 리뷰는 이전 원고를 대상으로 하므로 구판 수치를 현재 repaired 3-seed 결과와 섞지 않았다. 이번에 검증한 것은 원고 수식·주장, 확보된 공개 코드, 작은 유한 게임의 반례다. 밤사이 원격 학습의 최신 commit과 raw artifacts까지 감사한 것은 아니다.

**핵심 판단:** main_v6의 8B 일반 alignment 중심 방향은 유지하는 것이 좋다. 다만 독립 benchmark와 기전 검증을 연결해야 한다. Prompt-direct는 이전 교대 최적화의 실제 수렴 문제를 다루는 수정이다. 모든 리뷰를 해결하는 단일 장치는 아니며, 학습 teacher의 오류·후보 부족·neural generalization은 별도로 남는다.

## 1. 가장 중요한 지적과 이번 수정

| 지적 | 확인 내용 | 이번 처리 |
|---|---|---|
| R11: 회귀 손실 | v6에도 제곱이 expectation 밖에 있었음 | E[(h−target)^2]로 수정하고 target freeze와 empirical sampling law를 명시 |
| R12: 교대 갱신 | 최적 고정점 존재와 그 점으로의 수렴은 다름; 불안정 반례 검산 | Direct inner objective의 gradient/Hessian과 수치 반례를 부록에 추가 |
| R06: 개선의 의미 | Robust game surplus 양수여도 base 직접 승률은 0.5 미만일 수 있음 | Main claim을 game-value improvement로 한정, 반례 추가 |
| R14: 알고리즘 추적 | Teacher solve 뒤 neural fit 한 번은 v6에서 이미 수정; T>1 pool 위치는 불일치 | 매 outer stage에서 learner pool을 π_t로부터 다시 생성하도록 수정 |
| R25: MOPO 분류 | v6가 MOPO까지 scalar reward 방법으로 묶음 | Preference-only constrained method로 정정하고 RACO 설명 추가 |
| R02–R04: 동기 | Noisy label cycle, 서로 다른 목적, priority-free를 혼동 | 고정 rubric의 population probability 예시와 equal bargaining weight로 수정 |

R11은 원고와 코드를 분리해서 봐야 한다. 확보한 공개 commit `e90f2a2406f1c6b5c685e106d064f952403cbacf`의 `mnpo_scripts/mnpo_trainer.py`에서는 `losses=(h-target)**2`를 계산한 뒤 `core_loss=losses.mean()`을 사용한다. 그러므로 이번에 확인된 것은 **원고 수식 오류**이며, 과거 모든 학습이 mean-then-square를 사용했다고 결론 내릴 수 없다. 최신 서버 run의 실제 loss/config는 별도 대조해야 한다.

## 2. 각 리뷰의 의미와 현재 처리 상태

아래 ‘문구 반영’은 실험 완료를 뜻하지 않는다. 이전에 없던 실험 수치는 만들지 않았다.

| 리뷰 | 쉽게 말하면 | main_v6 대응 및 남은 일 |
|---|---|---|
| R01 | 좋은 Nash 공리에서 좋은 LLM으로 넘어가는 중간 연결을 보여 달라. | 8B 본문 방향 유지. Solver→neural realization→독립 정책 품질을 구분. 자연어 adaptive 대조군과 독립 비교는 미완료로 명시. |
| R02 | 한 번의 모순된 label과 population BT misspecification은 다르다. | 같은 rubric의 세 population 선호확률이 모두 .75인 수치 예시로 수정. 관측 label 하나만으로 cycle 증거라고 하지 않음. |
| R03 | A>B는 relevance, B>C는 completeness, C>A는 safety라면 같은 목적 안의 cycle을 보인 것이 아니다. | 의료 사례의 기준 전환을 제거. 하나의 고정 objective 안의 가상 확률 예시임을 명시. 실제 human cycles는 아직 미식별. |
| R04 | ‘가중치를 입력받지 않는다’가 ‘가치판단이 없다’는 뜻은 아니다. | Priority-free 삭제. 지정한 objective에 equal bargaining weight를 준다고 설명. Rubric 중복/분할 및 fallback 선택의 영향 명시. |
| R05 | Log-ratio와 regression 자체는 선행 연구에도 있으니 새 기술적 부분을 정확히 말하라. | INPO·PROSPER에 attribution. NBPO의 증분을 zero-surplus boundary와 policy-dependent games를 포함한 coupled proximal solve로 정리. |
| R06 | Game value 개선은 base를 직접 이긴다는 보장이 아니다. | 본문 주장 한정 및 finite-temperature/full-support 반례 추가. Direct WR과 game surplus 별도 보고. |
| R07 | 모든 목적 동시 개선이 가능한지 확인하고, 불가능하면 무엇을 반환할지 정하라. | Finite-pool feasibility audit와 상태 구분 추가. Certified-infeasible/numerical-unresolved/neural-rejected 구별. 별도 neural Phase0는 재도입하지 않음. |
| R08 | Affine invariance가 judge를 아무렇게나 재보정해도 같다는 뜻은 아니다. | A를 c배 하면 β도 c배 해야 같은 상대/utility scaling을 얻는다고 설명. Probability 범위를 넘는 scaling은 payoff 단위 변환으로 명시. |
| R09 | Population optimum, candidate categorical target, 실제 neural policy를 구분하라. | π*, p*, πθ 및 occurrence mapping 유지. Dual surplus가 finite target의 prompt 평균임을 명시. |
| R10 | 후보 네 개의 모든 pair를 맞춰도 후보 밖 응답 질량은 통제하지 못한다. | 식별 범위를 sampled pool로 한정. (a,a,1−2a) 예시와 fresh generation 평가 필요를 추가. |
| R11 | 제곱이 expectation 밖이면 오차가 상쇄되어 regression 증명이 성립하지 않는다. | 손실식의 제곱 위치 수정. 공개 코드 branch는 square-then-mean임을 별도 확인. |
| R12 | 고정점이면 최적이라는 증명은 교대 반복이 수렴한다는 증명이 아니다. | 불안정한 RPS map 검산. Direct concave solve의 curvature와 residual 확인을 추가. |
| R13 | λ box 경계에서 projected residual=0이어도 원래 dual KKT는 틀릴 수 있다. | 종료 조건에 unprojected stationarity, inner residual, bound activity를 반영. Box의 성공을 population/LLM 인증으로 쓰지 않음. |
| R14 | 수십만 dual step마다 LLM을 학습한 것처럼 쓰지 말고 실제 실행 구조를 보여 달라. | Finite target을 완성한 뒤 neural fit 한 번. 각 outer stage의 learner sampling과 실패 반환까지 pseudocode 수정. |
| R15 | Noisy 양수 surplus만 보고 accept하면 잘못 수락·거절할 수 있다. | Acceptance는 dev에서 하는 empirical check로 한정. 경계 근처 false accept/reject 측정은 아직 open. 단순 concentration bound를 자동 적용하지 않음. |
| R16 | O(1/T)는 정확한 proximal problem을 T번 푸는 보장이지 neural gradient step 수의 보장이 아니다. | Main headline 옆에 범위 명시. 현재 LLM T=1과 population theorem을 분리. 새 approximate theorem을 꾸며 추가하지 않음. |
| R17 | 같은 neural updates만으로 비용과 정보가 같다고 할 수 없다. | Unique semantic labels, swapped calls, generation, teacher, solver, policy, tuning 비용을 나눠 기록하도록 추가. 실제 ledger 수치는 실행 artifact로 채워야 함. |
| R18 | 독립 single-objective 학습은 interacting population인 HT-MNPO와 다르다. | 현 v6에는 HT-MNPO reproduction 주장이 없어 복원하지 않음. DPO specialist를 HT-MNPO라고 부르지 않음. 과거 구현 검증은 별도. |
| R19 | Independent judge와 다양한 opponent는 서로 다른 취약점을 검사한다. | Base WR 유지 + reference/NBPO/fixed-reference/competitor의 작은 독립 cross-play 계획 추가. 동일 finite bank·β로 game diagnostic도 비교. 미측정. |
| R20 | 기존 IFEval 하락을 평균 WR로 덮지 말고 공식 metric과 proxy를 구분하라. | 최신 metric variant 확인을 유지. 길이·거절·constraint별 오류 분석은 기술적 진단으로 명시. 구판 .762→.701을 최신 3-seed에 합치지 않음. |
| R21 | 실제 학습 가능한 다른 절충점들 사이에서 Nash가 어떤 선택인지 보여 달라. | UF 4-objective 표/7-weight DPO/empirical nondominance 계획 유지. Bootstrap 안에서 mean과 minimum을 다시 계산하도록 명시. 결과는 pending. |
| R22 | Adaptive opponent가 fixed reference보다 정말 필요한지 정면 비교하라. | Controlled evidence 유지. 완료된 Nash/utilitarian은 aggregation ablation이며 자연어 adaptive ablation은 아직 아니라고 본문에 명시. 승인된 fixed-reference control 계속 필요. |
| R23 | Target 부호가 유지되는 것은 전체 policy invariance보다 훨씬 약하다. | λ·β·bounds·step sizes의 변환 법칙 명시. 과거 sign-only 결과를 algorithm/policy invariance 증거로 승격하지 않음. |
| R24 | Prompt마다 목적을 먼저 합치는 것과 목적별 prompt 평균을 먼저 합치는 것은 다르다. | NBPO와 PROSPER 목적의 위치 차이 및 두 prompt 예시 추가. Prompt-direct에서도 λ는 전 prompt에 공유됨. |
| R25 | MOPO 분류가 틀렸고 reward-free 자체도 novelty가 아니다. | MOPO를 pairwise preference-only constrained 방식으로 정정. RACO의 DPO-style gradient conflict 처리와 구분. |
| R26 | 최신 Nash welfare 문헌과 ‘무엇을 bargain하는가’를 비교하라. | Zhong et al. v2 제목·저자·연도 갱신. Learned party reward, task-gradient improvement, robust policy surplus를 구분. BT-RM-Nash control 필요. |
| R27 | Raw maxmin의 약점을 보였다고 Nash만 유일하게 타당한 것은 아니다. | Guarantee table을 명시한 rule/모든 optimizer 범위로 한정. Pareto tie-break와 ideal-normalized 대안 언급. 기존 사용자 방향대로 KS 캠페인은 추가하지 않음. |
| R28 | 구판에서 음의 surplus나 Pareto domination이 있다면 trained policy의 IR/PO를 주장할 수 없다. | 현재의 empirical PO 미입증 및 Nash null 유지. 구판 수치는 provenance 확인 전 현재 결과에 복원/통합하지 않음. |
| R29 | Pair-level split이면 같은 prompt와 답변이 train/dev 양쪽에 들어갈 수 있다. | Source prompt를 먼저 group/split하고 모든 pairs·swaps를 함께 배치하도록 명시. 실제 split intersection/hash 감사는 남음. |
| R30 | β를 바꾸며 평가 β까지 바꾸면 학습법과 채점자가 동시에 바뀐다. | Common evaluation β/evaluator/comparator/disagreement를 고정하는 규칙 추가. 향후 temperature sweep도 이 기준 유지. |
| R31 | Regression이 h=0보다 못하면 teacher가 아무리 정확해도 policy update가 실패한 것이다. | Short recipe의 nMSE≈.68은 이 특정 실패를 개선. Long 실패는 보존. Prompt-direct와 neural schedule 개선을 서로 다른 수정으로 설명. |
| R32 | Noisy cycle subset을 골라 이겼다고 비추이성 때문에 이겼다고 할 수 없다. | Controlled cycle과 predicted cycle, human evidence를 구분. UF ordinal labels는 transitive임을 유지. 자연 데이터 cycle 이점은 아직 주장하지 않음. |
| R33 | Teacher·rubric·label source가 바뀌면 알고리즘 효과와 데이터 효과가 섞인다. | Human pairs, scalar-rating-induced labels, GPM predictions의 provenance 분리. Run별 teacher/rubric/objective-order 기록 요구. |
| R34 | Unbiased margin도 exp·normalization·log를 통과하면 unbiased하지 않을 수 있다. | 정확히 푸는 대상은 추정된 finite tensor에 조건부인 문제라고 명시. Finite-pool estimation과 neural/numerical error를 분리. |

리뷰는 전반적으로 타당하지만 모든 제안을 새 대규모 실험으로 실행할 필요는 없다. R18은 현재 baseline에 없고, R27의 KS 확대는 기존 방향과 맞지 않는다. R07도 별도 neural warm-start를 다시 만들라는 뜻으로 받아들일 필요가 없다. 다만 R11/R12/R25 같은 수식·알고리즘·사실 오류는 즉시 바로잡아야 한다.

## 3. 기존 방법에 정확히 무슨 문제가 있었나

### 3.1 원래 풀려던 문제

고정된 dual weight λ에서, prompt x의 finite response distribution p는 다음을 최대화해야 한다.

\[
f_x(p;\lambda)=\sum_k\lambda_k\widehat V_{k,x}(p)-\frac1\eta\mathrm{KL}(p\|p_{t,x}),
\qquad
\widehat V_{k,x}(p)=-\beta_k\log\sum_j\mu_{x,j}\exp\!\left(-\frac{[\widehat A_{k,x}^{\top}p]_j}{\beta_k}\right).
\]

여기서 p는8B 파라미터 벡터가 아니라 한 prompt에서 샘플한 후보에 대한 작은 categorical probability vector다. A는 learner candidate와 comparator candidate의 선호 margin 행렬이다. Reference occurrence measure μ와 현재 정책에서 뽑은 learner occurrence measure p_t를 구분한다.

상대의 최적 분포는

\[
\nu_k^*(p)=\operatorname{softmax}\left(\log\mu_x-\widehat A_{k,x}^{\top}p/\beta_k\right)
\]

이다. p가 바뀌면 ν*도 바뀐다.

### 3.2 기존 교대 방식의 논리적 빈틈

기존 방식은 현재 p로 ν*(p)를 구하고, 그 상대를 고정한 문제의 exponential update를 사용했다.

\[
p^{(r+1)}\propto p_t\exp\left(\eta\sum_k\lambda_k\widehat A_k\nu_k^*(p^{(r)})\right).
\]

이 식의 **고정점**은 원래 inner problem의 최적조건을 만족할 수 있다. 그러나 실제 반복이 그 고정점에 가까워진다는 보장은 별개다. 정책이 한쪽으로 움직이면 상대도 바뀌어 반대쪽으로 강하게 밀어낼 수 있다.

리뷰의 RPS 반례를 독립 검산했다. 기준 μ=(.34,.33,.33), β=.25, ηλ=4에서 고정점은

\[
p^*=(.33618258,.32907171,.33474571)
\]

이며, simplex의 두 독립 방향에 대한 Jacobian 고유값이 둘 다 약 **−1.33316**이다. 부호가 음수여서 오차 방향이 교대하고, 절댓값이1보다 커서 작은 오차가 커진다. 100번 반복해도 고정점과 L2 거리 약.27986이었다. 따라서 R을 키우는 것만으로 해결되지 않는다.

이것은 ‘cyclic preference라서 NBPO objective 자체가 정의되지 않는다’는 문제가 아니다. **잘 정의된 inner optimum에 도달하는 수치 절차의 문제**다. 해당 예시는 fixed multiplier inner problem이며 전체 Nash dual을 끝까지 푼 실험은 아니다.

### 3.3 Prompt-direct가 바꾸는 부분

여기서 prompt-direct는 **shared λ가 주어졌을 때 prompt별 finite-pool optimization을 직접 푸는 방식**이다. 위 ν*(p)를 softmin value에 대입한 목적 f_x를 직접 최적화한다. 상대의 변화를 빠뜨리지 않는 것이다.

그 Hessian은

\[
\nabla^2f_x(p)=
-\sum_k\frac{\lambda_k}{\beta_k}A_k
\left(\operatorname{diag}(\nu_k)-\nu_k\nu_k^\top\right)A_k^\top
-\eta^{-1}\operatorname{diag}(1/p)\prec0.
\]

첫 항은 negative semidefinite이고 KL 항은 interior에서 negative definite다. 따라서 이 작은 문제는 strictly concave이며 최적 target이 유일하다. 선언한 numerical tolerance까지 도달했는지는 inner KKT, simplex feasibility, unprojected dual residual로 확인한다. Concavity만으로 어떤 numerical routine이든 자동으로 성공하는 것은 아니다.

같은 반례에서 direct constrained optimizer는5iterations에 고정점과 거리 약1.83×10⁻¹¹, inner stationarity 약9.60×10⁻¹¹에 도달했다. 재현 코드는 패키지의 `scripts/verify_direct_solver_counterexamples.py`다. 이 수치는 실제로 계산한 유한 게임 검산이며 새 LLM 성능 결과가 아니다.

Prompt-direct만이 가능한 해법은 아니다. 적절한 수렴조건을 갖춘 damping이나 다른 saddle-point solver도 후보가 될 수 있다. 현재처럼 prompt당 후보 수가 작을 때는 목적과 residual을 직접 확인할 수 있는 방식이 설명과 검증에 유리하다.

## 4. Prompt-direct도 해결하지 못하는 것

전체 파이프라인은 다음 세 단계다.

1. **Preference tensor 추정:** 어떤 답변이 어떤 답변보다 나은지 얻는다.
2. **Finite-pool teacher solve:** tensor가 주어졌을 때 최적 후보 분포 p*를 구한다.
3. **Neural projection:** LLM의 log-probability 변화를 p*의 pairwise target에 맞춘다.

Prompt-direct는2단계를 수정한다. Teacher가 긴 답변을 잘라 보거나 잘못 평가하는 문제는1단계이고, 학습 prompt만 맞추고 held-out에서 실패하는 문제는3단계다. 또한 후보 밖의 응답 확률은 pairwise finite-pool fit만으로 결정되지 않는다.

원고의 회귀식은 다음이 맞다.

\[
\mathcal L(\theta)=\mathbb E[(h_\theta-h^*)^2].
\]

기존 표기인 `(E[h−h*])²`에서는 residual +1과−1이 상쇄되어 loss0이 된다. 올바른 MSE는1이다. 이 때문에 conditional-mean regression/identification 증명에 제곱 위치가 중요하다.

이후 short schedule에서 관측한 nMSE≈.68은3단계의 held-out fit이 h=0보다 좋아졌다는 실험적 근거다. **Prompt-direct가 문제를 정확히 풀어주는 것과 short schedule이 LLM fitting을 개선한 것은 별개의 개선**이다. 둘을 합쳐 ‘이제 알고리즘의 모든 근사·일반화 문제가 해결됐다’고 쓰면 안 된다.

## 5. 추가로 반드시 이해해야 하는 두 구분

### Game value와 직접 승률

R06의 예시에서는 robust game surplus가 +.01006인데 base 직접 승률은.4615다. 둘은 모순되지 않는다. Game value는 reference 근처에서 적응하는 상대에 대한 regularized worst-case 평가이고, 직접 WR은 base 하나와의 비교다. 따라서 general benchmark의 base-relative WR과 다양한 opponent의 독립 cross-play가 모두 필요하다.

### Prompt별 계산과 prompt별 bargaining

Prompt-direct에서도 λ는 모든 prompt에 공유되며 Nash는 **목적별 prompt 평균 surplus**에 적용된다. Prompt마다 별도의 Nash bargain을 하는 것이 아니다.

두 prompt에서 정책 P가 (.2,0)/(0,.2), Q가 (.08,.08)/(.08,.08)의 surplus를 갖는 예를 보자. P의 전체 평균은(.1,.1)이라 global Nash는 P를 선호한다. 각 prompt의 최악 목적을 먼저 계산하면 P는0, Q는.08이므로 Q를 선호한다. 이는 aggregation order의 차이이며 PROSPER의 모든 regularization을 재현한 수치 실험은 아니다.

PROSPER와 NBPO의 차이를 Nash-vs-maxmin 하나로 환원하면 안 된다. Faithful PROSPER와 NBPO의 game value 위에서 계산한 global maxmin control을 구분해야 한다.

## 6. 현재 계획에서 추가로 필요한 최소 작업

새 UF-4 일반 alignment 계획과9/18 metric freeze는 유지한다. 리뷰를 이유로 이전 neural gate sweep으로 되돌아갈 필요는 없다.

- 이미 승인된 **fixed-reference Nash, BT-RM-Nash, game-maxmin controls**를 완료한다. Nash/utilitarian3seeds는 다시 학습하지 않는다.
- 같은 teacher·prompt·pool에서 target-level와 neural-level 결과를 나란히 두어 어디서 gap이 생기는지 확인한다.
- Reference/NBPO/fixed-reference/한 경쟁 정책의 작은 independently judged cross-play를 추가한다. Comparator 선택은 dev에서 고정하고 같은 evaluation β와 bank를 사용한다.
- 기존 logs에서 비교 수·generation·teacher·solver·neural·tuning 비용을 분리한다. 새 대규모 compute sweep 대신 먼저 cost ledger를 완성한다.
- 최신 run의 loss, target freeze, stage sampling, unprojected residual, bound hits, split/rubric hash를 원고와 대조한다. 구판 수치 lineage는 따로 보관한다.

R07의 finite feasibility audit는 β>0에서 일반 LP가 아닌 convex optimization이다. 검증된 양수 feasible witness는 ‘이 finite support에서 가능’함을 보여준다. 불가능 판정에는 유효한 upper bound가 필요하고, 단순 numerical failure는 unresolved다. 별도 neural warm-start를 재도입하지 않았다.

## 7. 원고 수정 범위와 남은 증거

수정본은 core Nash objective, exact population convergence theorem, direct finite-pool 방향,8B main experiment templates를 유지한다. Intro 동기를 줄이고 정확히 했으며, 손실식·pseudocode·선행연구 분류를 고쳤다. 상세 counterexample/Hessian/feasibility/aggregation-order는 부록에 두었다. 구판 빨간 GPT 코멘트는 분석 입력으로 보존하고 clean manuscript에는 넣지 않았다.

문헌도 수정했다. MOPO는 point-wise reward 없이 pairwise preference로 primary objective와 제약을 다루며 practical algorithm에 lagged reference가 있다. RACO는 DPO-style objective gradients의 충돌을 다룬다. Zhong et al.의2026년9월7일 v2는 disagreement-adjusted welfare와 general-preference von Neumann 분석을 구분한다. 따라서 NBPO의 차별점은 ‘reward-free/Nash/normalizer 제거 최초’가 아니라 **reference-relative robust policy values의 coupled bargaining optimization**이어야 한다.

근거: [MOPO v2](https://arxiv.org/html/2505.10892v2), [PROSPER](https://arxiv.org/html/2602.19041v1), [RACO v2](https://arxiv.org/html/2602.02495v2), [Pluralistic Alignment v2](https://arxiv.org/abs/2403.05006v2), [Relative Improvement Bargaining](https://arxiv.org/html/2602.04155v1).

여전히 미완료인 것은 자연 데이터에서 adaptive game representation의 실제 이점, 새 일반 instruction의 독립 trade-off, 비용 비교, 경계 acceptance 오류 측정이다. 이를 완료된 것처럼 채우지 않았다. 현재의 SafeRLHF Nash–utilitarian null과 empirical Pareto 미입증도 그대로 남겼다.
