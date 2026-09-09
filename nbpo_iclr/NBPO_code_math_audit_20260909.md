# NBPO 수학·코드 감사와 LLM 회복 실험 권고

작성일: 2026-09-09. 대상: 제공된 main_v4.pdf, main_v6(2).pdf 및 공개 GitHub 두 revision. 이 문서는 이전 실행의 실패 원인을 확정하는 보고서가 아니라, 실제 재현된 구현 결함과 아직 검증할 원인 가설을 구분한 감사 보고서다.

## 1. 결론

**현재 구현에 이상이 없다고 판정할 수 없다. 공개 코드에서 수정해야 할 결함을 실제로 재현했다. 그러나 Nash bargaining의 핵심 수학이 LLM에서 원천적으로 작동하지 않는다는 증거도 없다.**

가장 타당한 다음 단계는 (1) 실제 실패 실행의 코드를 확보하여 해당 결함의 적용 여부를 확인하고, (2) teacher와 학습의 연결을 고치고, (3) 같은 solver target으로 기존 pairwise MSE와 weighted behavioral cloning(WBC)을 비교하는 것이다. WBC는 검증된 회복책이 아니라 문헌 근거가 있는 새로운 neural realization이다.

내일 오전까지 모든 benchmark에서 최고 성능을 약속할 수 없다. 현실적인 목표는 정확한 8B 학습 경로, 첫 matched comparison, capability 손실과 과잉 거절을 포함한 실제 생성 평가를 확보하는 것이다. 안전성과 유용성이 상충하는 문제에서 모든 개별 점수를 동시에 최대화하는 것이 Nash objective의 보장도 아니다.

## 2. 무엇을 검증했는가

| 자료 | 검증한 버전 | 범위 |
|---|---|---|
| GitHub main | `2b4c48cac19bf3509f3a9df24745885f7de90096`, 2026-09-04 17:21 KST | 기존 solver, sampled/RB target, trainer |
| `exp/iclr27-table1-v2` | `e90f2a2406f1c6b5c685e106d064f952403cbacf`, 2026-09-08 17:59 KST | direct solver, representation/aggregation, stage runner, 공개 config |
| main_v4.pdf | 24쪽, 2026-09-07 생성 | 이전 formulation 및 긍정적 성능 표 |
| main_v6(2).pdf | 22쪽, 2026-09-09 02:52 KST 생성 | 수정된 이론, direct solve, 통제 실험 및 실패 보고 |

최신 공개 branch도 canonN700/RB1200 및 새 downstream 보고보다 이전이다. 공개 builder에는 canonical target mode가 없다. 따라서 현재 서버에서 실행한 commit, tracked/untracked source 변경, resolved config, target/cache/checkpoint hash를 대조하기 전까지 **공개 코드의 결함을 해당 checkpoint의 확정 원인으로 부르면 안 된다.**

검증 결과는 기존 main의 core/dual/target CPU tests **29 passed**, 새 branch generic solver tests **23 passed**, 실제 production helper를 분리해 실행한 trainer unit functions **4 passed**다. 추가 결함 재현 assertion도 통과했다. 이는 실제 GPU, 모델 다운로드를 포함한 full Trainer integration, 원격 실패 checkpoint의 재학습을 수행했다는 뜻은 아니다. 기존 테스트가 통과해도 아래 경로 간 불일치는 놓칠 수 있다.

## 3. 알고리즘을 기초부터 나누어 보면

NBPO에는 서로 다른 세 단계가 있다.

1. **선호 모델:** 도움이 되는지, 해롭지 않은지 등 각 목적에서 두 응답의 선호를 추정한다. 이것이 틀리면 이후 최적화가 정확해도 잘못된 목표를 따라간다.
2. **게임과 bargaining solver:** 이 선호를 이용해 목적별 game value와 reference 대비 surplus를 정의하고, Nash welfare를 높이는 이상적인 정책 분포를 구한다.
3. **Neural realization:** 후보 응답 위에서 구한 목표 분포를 수십억 개 parameter의 autoregressive LLM에 옮긴다. 현재 pairwise log-ratio MSE가 이 단계다.

후보가 8개인 simplex 최적화에서 좋은 분포를 구하는 것과, 새로운 prompt에서 긴 응답을 생성하는 LLM이 좋아지는 것은 다른 명제다. 이론상 exact policy update가 존재한다는 사실만으로 finite data, approximate teacher, SGD로 그 update가 구현되지는 않는다.

### 3.1 핵심 population 수학은 유지할 근거가 있다

v6의 finite-action/full-support 설정에서 regularized opponent의 softmin, game value의 concavity, disagreement point, strictly feasible bargaining set, positive dual 및 정확한 proximal update의 논리는 대체로 일관된다. 실제 구현에서도 softmin 부호, prompt 평균 위치, 별도 reference-as-learner disagreement, 모든 prompt에 공유되는 global lambda, proximal center를 pi_t에 두는 점은 맞았다.

단, exact convergence 결과의 조건은 **정확한 inner/outer solve, full support, joint strict improvement**다. LoRA/full FT, 미니배치 SGD, 8개 후보, 고정 reward model로 자동 확장되는 정리가 아니다. v4의 반복 map이 수렴한다는 서술보다 v6의 direct concave solve가 올바른 방향이다.

### 3.2 empirical mass와 실제 LM probability를 구분해야 한다

실제 pi_t에서 IID로 후보 y_i를 추출하고 occurrence별 empirical center를 p_t(i)=1/N으로 놓았다면 neural density ratio가 만드는 분포는

\[
\widehat q_\theta(i)=\frac{p_t(i)\exp[\log\pi_\theta(y_i|x)-\log\pi_t(y_i|x)]}{\sum_jp_t(j)\exp[\log\pi_\theta(y_j|x)-\log\pi_t(y_j|x)]}.
\]

정확한 pairwise fit가 복원하는 것은 이 importance-reweighted empirical distribution이다. 일반적으로 raw LM probability를 후보 안에서 정규화한 분포와 다르다. v6 Appendix A.5는 이 정의를 명시해야 한다. **Uniform center를 raw sequence-probability softmax로 무조건 바꾸면 IID 샘플링 확률을 이중 반영할 수 있다.**

Temperature/top-p로 바꾼 sampler q에서 생성하고 raw pi_t log-probability로 학습하면 q와 pi_t가 다르다. Pairwise normalizer는 상쇄되므로 모든 label이 즉시 틀린다는 뜻은 아니지만, solver center/opponent, dual 및 회귀 measure의 population 해석이 달라진다. 새 실험은 T=1, top_p=1, top_k disabled, repetition penalty=1로 일치시키는 편이 명료하다. 유한 generation horizon과 EOS/EOT도 같은 확률 사건으로 정의해야 한다.

### 3.3 Pairwise fitting은 전체 정책을 붙잡지 못한다

\[
g_i=\log(p_i^\star/p_{t,i}),\quad \ell_i=\log\pi_\theta(y_i|x)-\log\pi_t(y_i|x).
\]

기존 loss는 ell_i−ell_j를 g_i−g_j에 맞춘다. 모든 ell_i에 같은 prompt별 상수 c(x)를 더해도 loss는 같다. 따라서 후보 간 비율이 맞더라도 후보 전체에 배정한 실제 probability mass나 pool 밖의 응답을 충분히 통제하지 못한다. Solver의 finite-pool KL이 실제 full-policy KL을 보장하지도 않는다.

이는 **가능한 실패 메커니즘**이다. 이 자유도 때문에 반드시 실패한다는 정리는 아니다. REBEL은 relative-reward squared regression을 사용해 Llama-3-8B 성과를 보고한다. 따라서 NBPO의 실패를 이 loss 계열 전체의 불가능성으로 일반화하면 안 된다. [REBEL](https://arxiv.org/html/2404.16767v4)

## 4. 실제 재현한 코드 결함

아래 경로는 별도 표기가 없으면 최신 공개 commit `e90f2a2` 기준이다.

### A. BT baseline의 teacher가 학습 데이터로 전달되지 않는다 — 확정

`nbpo_representations.py`의 BT representation은 scalar reward r을 Q로 사용한다. Solver는 q_update와 canonical target을 저장한다. 그러나 `build_nbpo_pairs.py`는 이를 읽지 않고 원래 preference tensor A에서 comparator 차이를 다시 만든다.

실제 함수 재현: r=(1,−1), eta=1, weight=1이면 solver의 pair target은 **2.0**이다. 별도의 A=0을 입력하면 builder의 RB target은 **0.0**이다. 두 값이 같아야 하는 end-to-end 계약이 깨진다. 이것이 NBPO adaptive-game arm의 붕괴를 직접 설명하지는 않지만 BT baseline과 6-method 비교의 타당성을 해친다.

**수정:** 모든 representation/aggregation이 동일한 `log(p_star/p_t)` artifact를 읽어 pair target을 만들게 한다. 이미 eta가 포함된 canonical units를 명시하고 trainer에서 eta를 다시 곱하지 않는다. 기존 sampled/RB의 eta-once 계약 자체는 맞으므로 기존 loss를 소급해서 double-eta bug라고 부르면 안 된다.

[BT representation](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/mnpo_scripts/nbpo_representations.py), [pair builder](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/scripts/nbpo/build_nbpo_pairs.py)

### B. Train과 validation이 다른 target 문제를 푼다 — 확정

Stage runner의 train은 direct `solve_finite_pool`을 사용하지만 `build_validation_pairs()`는 legacy R-step solver를 호출한다. 같은 A, mu, pi_t와 lambda=5를 사용한 재현에서 train direct target은 **−0.1046924751**, validation R=1 target은 **−0.3298213679**였다. Train residual은 약 1.5e−12였다.

**수정:** 같은 canonical backend를 사용한다. Dev에서는 train에서 구한 global lambda를 고정하고 dev prompt의 inner problem을 푼다. Dev에서 lambda를 다시 학습하는 것은 이 수정에 필요하지 않다. 실제 canonN700의 private builder가 이미 해결했는지는 별도 확인한다.

[stage runner](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/scripts/nbpo/run_nbpo_stage.py)

### C. Target identity 검증이 독립적인 solver 정확도 검증이 아니다 — 확정, 경로별 범위 주의

Direct solver는 SLSQP의 pi_hat을 그대로 반환하지 않고 Q(pi_hat)로 exponential map을 한 번 더 적용한 pi_star를 반환한다. Target check도 Q(pi_star)가 아닌 이전 Q(pi_hat)을 사용한다. 그래서 그 identity는 구성상 맞을 수 있다.

유효한 2×2 예제와 raw weight=1000, 기본 maxiter=400에서 보고 identity는 **0.0**이지만, 반환 정책에서 다시 계산한 centered Eq.27 residual은 **4.5634e−4**, extra-map residual은 **2.2686e−4**였다. Generic standalone writer는 저장했다. **최신 stage runner는 extra-map>1e−4를 검사하므로 이 예를 거부한다. 모든 경로에 검사가 없다고 하면 틀린다.**

**수정:** 실제 optimizer 정책을 반환하고 그 정책에서 nu와 Q를 재계산한다. Canonical target은 직접 log(p_star/p_t)로 만든다. SLSQP status와 실제 stationarity/feasibility를 함께 검사하고 CLI와 stage가 중앙 validator를 공유하게 한다. 단순 identity를 맞추기 위한 추가 map으로 정확도를 인증하지 않는다.

[direct solver 및 target check](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/mnpo_scripts/nbpo_generic.py), [generic writer](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/scripts/nbpo/solve_nbpo_dual.py)

### D. Stage dispatch/config가 실제 선언과 다르다 — 확정

Stage representation factory에 모든 method의 kwargs를 동일하게 전달하면 fixed-reference는 unexpected `beta`, BT는 unexpected `A_policy` TypeError를 낸다. Standalone CLI는 별도로 올바르게 분기한다. Stage는 inner_workers 등 일부 solver 설정도 전달하지 않아 CPU solve ETA가 달라질 수 있다.

또한 공개 `final_iclr2027/saferlhf_core.yaml`의 주석은 frozen GPM ensemble을 말하지만 실제 값은 `Qwen/Qwen3-32B`, target `sampled`, pool 4+4다. 실제 `validate_final_config()`는 이 조합을 validated=true로 처리했다. **현재 원격 실행이 이 config를 사용했다는 증거는 없다. 그러나 이 파일을 완성된 v6 recipe로 실행하면 안 된다.**

**수정:** 하나의 dispatch와 resolved config를 사용하고 실제 소비되는 argument를 기록한다. Judge/checkpoint, target units, pool, solver backend도 schema로 확인한다.

[공개 final config](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/training_configs/nbpo/final_iclr2027/saferlhf_core.yaml), [validator](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/mnpo_scripts/final_run_validator.py)

### E. BF16 logits에서 sequence log-probability 차이가 사라진다 — 재현 확정, 실제 활성 여부 미확정

`simpo_trainer.py`와 `precompute.py`는 input dtype으로 log_softmax와 sequence sum을 수행한다. 이후 FP32 cast로는 이미 잃은 정밀도를 복원할 수 없다.

실제 helper에 1,024-token BF16 예제를 넣었을 때 한 정답 token의 logit을 0→1로 바꿔도 기존 함수는 두 sequence log-probability를 모두 **−708.0**, 차이를 **0**으로 반환했다. 같은 BF16 logits를 FP32로 정규화·합산하면 차이는 **+0.37988**이었다.

**수정:** 공통 helper에서 log_softmax/selected token/sum을 명시적으로 FP32로 수행한다. Full vocabulary FP32 allocation은 chunking으로 줄인다. 영향을 받은 reference cache는 다시 만든다. 실제 runtime이 이미 FP32 logits를 내는 경우 이 결함은 비활성일 수 있으므로 logits부터 cache/loss까지 실제 dtype을 기록한다. FP32 loss만 선언하지 않는다.

[policy log-probability](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/scripts/simpo_trainer.py), [reference precompute](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/mnpo_scripts/precompute.py)

### F. 같은 후보가 비교 상대에 따라 다른 prompt로 평가된다 — 재현 확정, 영향 행 수 확인 필요

Pair별 긴 응답에 따라 prompt truncation이 결정된다. 실제 함수에서 prompt=abcdef, 후보 a=xy, b=z, c=uvwxy, max_length=10, max_prompt_length=4를 넣으면 pair(a,b)의 a는 abcdef에, pair(a,c)의 a는 cdef에 조건화된다.

그러면 동일 후보의 log pi(a|x)를 공유한다는 pair graph 전제가 깨진다. Precompute/trainer가 각 행에서 같은 tokenizer를 사용한다는 검사만으로 잡히지 않는다.

**수정:** prompt와 전체 후보 pool을 한 번 tokenize하고 candidate ID/token/mask를 고정한다. Pair는 그 후보를 선택만 한다. 실제 train prompt가 이미 공통 길이 제한을 만족하면 이 문제가 비활성일 수 있다. Teacher가 평가한 텍스트와 최종 학습 token event가 달라지면 해당 artifact를 재생성한다.

[pair tokenization](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/mnpo_scripts/pair_tokenization.py)

### G. `dev`를 무시하고 `test`를 evaluation으로 고르는 split trap — 확정

`run_mnpo.py`는 이름에 train이 있으면 학습, test/eval이 있으면 evaluation으로 선택한다. dev/validation은 인식하지 않는다. 새 train/dev/test 구조에서는 test를 학습 중 evaluation에 쓰게 된다.

**수정:** train_split과 eval_split을 명시하고 final test는 별도 entrypoint에서 읽는다. 과거 이름 test가 실제 dev 역할이었을 가능성이 있으므로 이름만 보고 과거 leakage를 확정하지 않는다.

[dataset selection](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/mnpo_scripts/run_mnpo.py)

## 5. 추가로 검증할 이론·데이터 조건

### 5.1 Bounded dual의 projected KKT만으로 원래 Nash 문제를 인증할 수 없다

논문은 양의 unbounded lambda를 최적화하지만 code box의 upper bound에 걸리면 projected residual이 0이어도 원래 stationarity가 아니다. 예를 들어 A=[[0,.01],[−.01,0]], reference=(.5,.5), eta=1, lambda_max=100인 2-action 문제는 boxed p=.7310586, projected residual=0이지만 lambda*s=.2311이다. 원래 optimum은 p=.9167783, lambda≈239.9357이다.

Active bound, unprojected stationarity, s>0, lambda*s≈1을 확인해야 한다. Bounds가 실제로 active라면 정당하게 확장하고 다시 푼다. Strict feasibility가 없으면 실패를 기록하며 surplus를 epsilon으로 바꿔 log에 넣지 않는다. 실제 run이 이 조건에 걸렸다는 증거는 아직 없다.

### 5.2 Teacher가 긴 답변을 제대로 관찰했는가

공개 GPM scoring script는 기본 max_length=384, longest_first truncation을 사용한다. Policy 학습 문맥은 최대 2048이다. 짧은 encoder가 학습 때부터 동일하게 제한됐다면 설정 자체를 bug라고 단정할 수 없지만, 장문 응답의 뒷부분이나 prompt가 잘릴 수 있다. Truncation 비율, refusal/length와 teacher weight의 관계, GPM과 별도 RM의 ranking 차이를 확인해야 한다. Encoder position capacity를 확인하지 않고 길이만 늘리면 안 된다.

공개 generation script는 response text를 strip하고 실제 sampled token IDs를 저장하지 않는다. 새 pipeline은 실제 token IDs와 termination reason을 보존해야 한다. 이것 역시 현재 checkpoint의 원인으로 확인된 것은 아니다.

[pool generation](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/scripts/experiments/iclr2027_table1_v2/generate_saferlhf_pools.py), [ensemble scoring](https://github.com/promotion-kim/NBPO/blob/e90f2a2406f1c6b5c685e106d064f952403cbacf/scripts/experiments/iclr2027_table1_v2/score_pools_with_ensemble.py)

### 5.3 이전 RB 해석은 어디까지 가능한가

Floor nMSE=.016이라는 수치만으로 “새 prompt의 neural network가 target의 98.4%를 예측할 수 있다”고 해석할 수 없다. Floor의 계산 방식이 finite-pool conditional noise나 pair graph residual이면 unseen-prompt generalization과 다른 양이다. Noise가 줄어도 실패했다는 결과는 noise-only 설명을 약화하지만, 모든 데이터/추정 노이즈가 무관하다고 기각하지는 않는다.

MSE의 항상 정확한 분해는 E[target²]+E[h²]−2E[target*h]다. Var/cov 표현은 평균 조건을 확인해야 한다. .854−.110이 남았다는 사실을 “정책 움직임의 87%가 noise”라는 통계적 분산 분해로 표현하면 과하다. 현재 데이터에서는 policy의 추가 제곱 오차가 회수된 cross term보다 컸다고 쓰는 것이 정확하다.

## 6. 현재 downstream 결과의 의미

| 측정 | Base | canonN700 | RB1200 |
|---|---:|---:|---:|
| IFEval strict prompt | .7098 | .6174 | .5989 |
| GSM8K zero-shot-CoT EM | .8431 | .6861 | .6899 |
| HarmBench harmful rate ↓ | .3281 | .1344 | .1281 |
| Alpaca prompt local RM WR vs base | — | .1186 | .1130 |
| Alpaca response median tokens | 495 | 229 | 230 |

수치는 사용자의 Claude Code 보고에서 가져왔으며 여기서 원본 응답과 classifier를 재평가하지 않았다. IFEval은 9.24/11.09pp, GSM8K는 15.70/15.32pp 하락했다. 이 두 학습 recipe는 broad usefulness를 심하게 훼손했다. 안전성 수치만으로 Pareto 개선이라고 부를 수 없다.

짧아짐과 harmful rate 감소는 과잉 거절/축약과 일관되지만, 실제 응답 검사 전까지 그 원인으로 확정할 수 없다. Benign prompt refusal, 조기 EOS, 잘린 답변, GSM8K parser 실패를 분리해야 한다. 현재 표는 여러 metric의 trade-off이며 reference가 모든 objective에서 우월한 엄밀한 Pareto dominance 사례도 아니다.

canonN700의 N700은 700 training prompts를 의미하는 실행명이며, RB1200은 RB target 진단의 1200 optimizer steps를 의미한다. N700의 정확한 steps와 epoch, RB1200의 prompt 수는 run manifest로 확정해야 한다. N=700, 28 pairs, batch32, 1200steps가 맞다면 약1.96 pair-row epochs다. 서로 다른 teacher/데이터/예산의 두 실행을 matched target ablation으로 취급하지 않는다.

## 7. 추천 수정: NBPO-WBC를 하나의 새 variant로 비교

\[
L_{\mathrm{WBC}}(\theta)=-\mathbb E_x\sum_{i=1}^N p^\star_{x,i}\log\pi_\theta(y_{x,i}|x).
\]

Solver가 높게 평가한 응답 자체의 likelihood를 올리는 정책 추출이다. Log probability는 **전체 vocabulary softmax의 response-token 합**이다. 8개 후보 안에서만 softmax하거나 응답 길이로 나누지 않는다. p_star를 detach하고 prompt별 mass=1을 유지한다. IID pi_t에서 샘플링했다면 p_star에 raw pi_t를 다시 곱하지 않는다.

이는 finite-pool weighted behavioral cloning이고 population forward-KL projection의 Monte Carlo 근사다. Finite normalization, 같은 pool로 추정한 Q, 신경망 일반화 오차 때문에 exact population theorem을 자동 상속하지 않는다. Finite dataset에서 uniform WBC도 sample fitting 때문에 policy를 움직일 수 있으므로 “no-change target이면 모든 gradient가 0”이라는 잘못된 테스트를 만들면 안 된다.

MOPO는 optimal importance ratio로 weighted behavioral cloning을 수행하는 policy extraction을 제시한다. SPPO는 individual likelihood에 대한 회귀의 관련 근거다. 두 방식을 동일한 loss라고 부르지 않는다. [MOPO Eq.7](https://arxiv.org/html/2505.10892v2), [SPPO](https://arxiv.org/html/2405.00675v3)

기존 all-pair dataloader를 유지하려면 N=8에서 pair(a,b)별 loss를

\[
L_{ab}=\frac N2[p_a^\star(-\log\pi_\theta(y_a))+p_b^\star(-\log\pi_\theta(y_b))]
\]

로 놓는다. 모든 28 pair를 균등 평균하면 위 prompt objective와 정확히 같다. Float64 toy 확인에서 loss 차이 0, gradient 최대 차이 1.39e−17이었다. 실제 tokenization/distributed reduction의 integration test는 별도 필요하다.

오늘의 primary comparison에 별도 benign replay, target clipping, teacher damping, KL anchor까지 동시에 추가하지 않는다. 그러면 무엇이 작동했는지 알 수 없다. 이후 teacher 자체가 과잉 거절을 선호한다는 근거가 나오면 preference model/data 또는 제약을 수정한다. WBC만으로 나쁜 teacher를 고칠 수 없다.

## 8. 밤사이 실험과 평가

별도 실행 프롬프트에 구현 지시와 세부값을 제공했다. 핵심은 아래와 같다.

| 항목 | 제안 |
|---|---|
| Base/reference | `meta-llama/Llama-3.1-8B-Instruct`, 동일 고정 revision |
| Dataset | 기존 `PKU-Alignment/PKU-SafeRLHF` subset/revision |
| Objective/teacher | helpfulness, harmlessness; 기존 검증된 3-seed GPM ensemble/calibration 고정 |
| Split | 기존 적격 train manifest에서 고정 2,000 prompts; dev500; 가능한 fresh test1000 |
| Pool | learner8 + 독립 comparator8, raw T1/top_p1, 동일 pool 공유 |
| Solver | direct, eta1, beta_help=beta_harm=.25, train global lambda |
| Arms | NBPO-MSE와 NBPO-WBC; 여유가 있으면 Game-utilitarian-WBC |
| 학습 | full FT, AdamW 5e−7, (.9,.95), eps1e−8, wd0, cosine/warmup.1, clip1 |
| Batch/steps | 32 pairs, 1750 updates, seed42, outer stage1 |
| Precision/runtime | BF16 forward + FP32 logp/loss/master/Adam state, 기존 4×H200/ZeRO2 가정 후 실측 |
| Selection | 고정 horizon checkpoint; 외부 benchmark로 선택하지 않음 |

2,000×28/32=1,750 updates는 **1 pair-row epoch**이며 각 unique response는 7번 나타난다. LR는 SPPO를 참고한 보수적 시작점이지 NBPO에서 검증된 최적값이 아니다. MSE의 reference forward/cache와 WBC 연산 차이가 있으므로 동일 update/data budget을 곧 동일 FLOPs라고 부르지 말고 tokens, GPU-hours, forward counts도 보고한다.

선택적 Game-utilitarian control은 train Nash의 L=sum(lambda)를 고정해 equal raw weights=(L/2,L/2)를 사용한다. 단순(.5,.5)와 raw Nash lambda를 같은 eta에서 비교하면 가중치 총량과 유효 KL 강도까지 달라진다. 이 control을 L1-matched라고 명시하고, 성능 차이를 곧바로 모든 utilitarian 방법에 대한 우월성으로 일반화하지 않는다.

이 계획은 이전 7,000 prompts×6 methods×3 seeds의 즉시 실행 제안을 **밤사이 범위에 한해 대체**한다. 이미 생성한 기존 pool은 삭제하지 않으며 새 sampler와 다른 artifact는 별도 이름으로 보존한다. 실제 20-step throughput과 pool/scoring 속도를 측정해 deadline 전에 두 arm과 평가가 들어가는지 판단한다.

어떤 benchmark 세트가 모든 논문에 일률적으로 필수인 것은 아니다. 이 논문의 주장에는 다음 측정이 필요하다.

| 주장/위험 | 우선 benchmark/metric | API 비용 |
|---|---|---|
| Helpful/safe bargaining | 독립 held-out prompt의 objective vector, surplus, reference delta, uncertainty | 0, frozen local models |
| 지시 이행 보존 | IFEval 541, official strict prompt/instruction; loose secondary | 0 |
| 수학 능력 보존 | GSM8K 1319, 동일 zero-shot-CoT EM, parser failure 별도 | 0 |
| Harm reduction | HarmBench text test, category별 공식 evaluation path와 harmful rate | 0 |
| 과잉 거절 | XSTest safe250/unsafe200, 적절한 refusal 판정 또는 blinded manual audit | 0 |
| General chat quality | Alpaca805/Arena hard500/creative250에 고정 local Skywork RM WR | 0, 공식 점수 아님 |
| Multi-turn | MT-Bench 80, 2-turn generation; 시간 여유 시 | 생성 API0; 공식 judging 미수행 |

HarmBench의 copyright는 공식 hash-check 경로가 따로 있다. 모든 category를 같은 classifier로 평가하면 공식 metric이라는 주장이 틀릴 수 있다. Local Skywork WR은 공식 AlpacaEval-2 LC나 Arena-Hard 점수가 아니다. 이미 본 외부 benchmark는 개발에 노출된 것으로 기록하고 새 prospective 설정의 test라고 소급 주장하지 않는다. [HarmBench evaluator](https://github.com/centerforaisafety/HarmBench/blob/main/evaluate_completions.py), [XSTest](https://github.com/paul-rottger/xstest), [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), [Arena-Hard](https://github.com/lmarena/arena-hard-auto)

Regression nMSE<.90/sign>.65의 기존 neural gate는 진단으로 남긴다. 이를 통과하지 못했다는 이유만으로 candidate 생성 평가를 막지 않는다. 수학/구현 correctness, Nash feasibility, Algorithm1의 surplus acceptance는 서로 별개다. Candidate가 acceptance를 실패하면 결과는 보고하되 accepted next-stage policy라고 부르지 않는다.

## 9. 논문은 v6를 기반으로 수정한다

v4로 되돌아가 긍정적 표를 복원하는 방향은 권하지 않는다. v4의 표는 checkpoint, judge, protocol 및 실제 response provenance가 재확인되어야 한다. Local reference-win을 공식 Alpaca LC/Arena 점수와 섞으면 안 된다.

v6에서 고칠 것은 다음과 같다.

1. A.5에 empirical importance distribution을 명시하고 실제 LM conditional distribution과 구분한다.
2. Boxed dual의 projected certificate 범위를 명시한다. Actual bound activity/feasibility를 보고한다.
3. Appendix A.6에서 h에 D를 포함한 뒤 D-weighted pairing을 다시 사용하는 표기를 정리한다. Euclidean gradient/pairing 또는 weighted gradient 중 하나로 일관되게 정의한다. Monotonicity는 t≥1을 명시한다.
4. Table13 caption의 “No arm meets either”, “every arm sits below it”는 표의 일부 양의 surplus와 모순이므로 수정한다. Surplus의 split/checkpoint/evaluator hash도 밝힌다.
5. 기존 회귀 gate를 diagnostic-only로 업데이트하고 새 downstream 실패도 투명하게 반영한다.
6. WBC가 성공하면 exact population NBPO와 practical NBPO-WBC 사이의 근사 지점을 명시하고 MSE 대조군을 남긴다.

최종 중심 주장은 “Nash가 모든 aggregation보다 좋다”가 아니라, **일반 pairwise preference game에서 정의한 bargaining objective와 검증 가능한 solver, 그리고 이를 유용성을 보존하는 LLM 정책으로 구현하는 방법**이어야 한다. 지금은 마지막 연결이 해결되지 않았다.

최종 실험에는 내부 representation/aggregation controls뿐 아니라 가까운 NLHF/SPPO, REBEL, MOPO/PROSPER 계열 비교가 필요하다. 같은 task, model, data, judge와 합리적인 각 방법의 tuning budget을 명시한다. 각각 다른 논문의 공개 leaderboard 숫자를 그대로 한 표에서 우열 비교하지 않는다. [NLHF](https://arxiv.org/pdf/2312.00886), [PROSPER](https://arxiv.org/html/2602.19041v1)

만약 수정 후 MSE/WBC 모두 teacher 목표를 따라가면서 usefulness가 떨어진다면 teacher 또는 objective/task 설정을 수정해야 한다. WBC만 회복하면 neural realization이 핵심 병목이라는 근거가 된다. Game-utilitarian-WBC도 동일하게 회복하면 neural transfer의 기여와 Nash aggregation의 추가 기여를 분리해야 한다. 모두 실패하면 LLM superiority 주장을 보류하고 문제를 더 좁혀야 한다.

## 10. 코딩 모델 선택

이번처럼 여러 파일과 수식 계약을 동시에 고치는 작업에는 **Codex `gpt-6-astra`, Extra High reasoning**을 권한다. 계정에 없다면 표시되는 지원 모델 중 `gpt-5.6-sol` Extra High를 대안으로 쓴다. Codex 공식 model 문서에서 모델과 reasoning 선택을 확인했다. [Codex models](https://learn.chatgpt.com/docs/models)

기존 Claude Code 환경을 유지한다면 **`claude-opus-5`, effort `xhigh`**를 권한다. 공식 문서상 지원되는 설치에서 `claude --model claude-opus-5 --effort xhigh`를 사용한다. 설치/계정에서 제공되는지 확인하며 표시되지 않는 model ID를 강제로 가정하지 않는다. [Claude Code model configuration](https://code.claude.com/docs/en/model-config)

한 agent가 구현하고 다른 agent가 독립 review하도록 파일 소유를 나눈다. 모든 로그 읽기에 최대 reasoning을 쓰기보다 수식/target/분산 loss 검증에 집중한다. 코딩 assistant 사용과 benchmark judge API 사용은 별개이며, 이번 실험은 paid judge API를 호출하지 않는다.
