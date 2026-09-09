# NBPO: Claude Code / Codex 실행 프롬프트

작성: 2026-09-09. 아래 구분선 이후 내용을 repository와 GPU 서버에 접근할 수 있는 coding agent에게 그대로 전달한다. 같이 제공하면 좋은 자료는 `NBPO_code_math_audit_20260909.md`, 현재 논문 v6, 실제 실패 실행의 manifest다. 이 프롬프트는 자체로 필요한 지시를 포함한다.

권장 실행: Codex `gpt-6-astra`, reasoning Extra High. Claude Code를 계속 사용한다면 지원되는 설치에서 `claude --model claude-opus-5 --effort xhigh`. 실제 account/CLI가 제공하는 model과 effort를 확인한다. 코딩 assistant 비용과 무관하게 아래 benchmark 실험에서는 유료 judge API를 호출하지 않는다.

---

당신은 NBPO repository의 구현을 수정하고 LLM 회복 실험을 실행하는 research engineer다. 분석·계획에서 멈추지 말고, 실제 코드 변경, 의미 있는 regression tests, 실행, 저장된 응답 평가까지 진행하라. 성공 수치나 ETA를 만들어내지 말라.

## A. 목표와 고정 제약

1. Deadline은 **2026-09-10 09:00 KST**다. 현재 시간을 확인하고 남은 시간, GPU 상태, 실제 throughput으로 일정을 계산하라. 이미 deadline이 지났으면 지났다고 보고하고 동일 작업의 실제 예상 완료 시각을 제시하라.
2. **Paid OpenAI/Anthropic/기타 judge API 호출 0.** 이미 설치/승인된 coding assistant 실행은 별도다. 평가 모델은 로컬로 사용한다.
3. 목표는 검증 가능한 구현과 첫 matched 8B comparison이다. SOTA, 모든 metric 최고점, ICLR acceptance를 보장하지 않는다.
4. 이전 `NBPO_API_free_downstream_protocol.md`의 즉시 7000×6 methods×3 seeds 계획은 **이번 밤사이 범위에서 아래 2000-prompt comparison으로 대체**한다. 새 pool의 sampling은 이전 T=.8/top_p=.95에서 **T=1/top_p=1**로 명시적으로 변경한다. 기존 자료는 보존한다.
5. Neural nMSE<.90/sign>.65는 **diagnostic-only**다. 이것을 새로운 training/downstream stop gate로 쓰지 않는다. Mathematical/implementation correctness와 Algorithm1 surplus acceptance는 별개다. Invalid teacher를 NBPO target이라고 학습시키지는 말라. Acceptance에 실패한 생성 후보도 평가하고 실패를 보고하되 accepted next-stage policy라고 부르지 않는다.
6. 기존 정상 작업은 활용한다. 확인된 결함이 있는 job은 checkpoint/log를 보존한 뒤 영향 범위를 좁혀 수정한다. 이전 실패 checkpoint, response, config, cache는 삭제하거나 덮어쓰지 않는다. 새 branch/worktree와 versioned artifact directory를 사용한다.
7. 불필요한 재확인 질문 없이 승인된 읽기·수정·검증·학습·로컬 평가를 수행한다. 외부 메시지 전송, 공개 업로드/논문 제출, secrets 노출은 이 지시의 범위가 아니다.

## B. 먼저 실제 실행 버전을 확정하라

Public repository: https://github.com/promotion-kim/NBPO

감사한 공개 버전은 다음 둘이다.

- main: `2b4c48cac19bf3509f3a9df24745885f7de90096` (Sep4).
- `exp/iclr27-table1-v2`: `e90f2a2406f1c6b5c685e106d064f952403cbacf` (Sep8 17:59 KST).

둘 다 canonN700/RB1200 및 최신 downstream 실행보다 이전이다. 최신 공개 branch에는 direct solver가 있지만 canonical builder는 공개되어 있지 않다. **원격 실행 파일을 읽지 않고 main으로 reset하거나 최신 코드가 잘못됐다고 가정하지 말라.**

첫 작업으로 실제 repository commit, dirty/untracked source 목록 및 필요한 patch, 실행 명령, resolved configs, environment versions, checkpoint/model/tokenizer revisions, teacher/target/reference-cache hashes를 저장하라. Secrets/env 전체를 출력하지 말라. N700/RB1200이 사용한 실제 entrypoint와 loss/optimizer를 확인한다. 이미 구현된 수정은 중복하지 말라.

감사 결과는 아래와 같다. 모든 항목에 `present / already_fixed / inactive_in_this_run / unavailable` 상태와 실제 code pointer를 남겨라. “원인이 확인됨”은 해당 run에서 재현했을 때만 사용한다.

## C. 우선 수정할 correctness 계약

### C1. 모든 method가 하나의 canonical target을 읽게 하라

공개 `build_nbpo_pairs.py`는 BT solver의 r/Q/canonical output을 무시하고 A_policy에서 sampled/RB target을 다시 만든다. 재현 예는 r=(1,−1), A=0, eta=1, weight=1에서 solved pair target=2지만 builder target=0이다.

- Candidate-level artifact `g_i = log(p_star_i/p_t_i)`를 저장한다.
- Pair target은 `g_a - g_b`로만 구성한다. Solver가 계산한 representation/aggregation을 builder가 다시 추정하지 않는다.
- 새 schema: `target_mode: canonical_logratio`, `target_column: nbpo_logratio_target`, `target_units: final_logratio_change`, `eta_already_included: true`.
- 기존 sampled/RB의 `nbpo_weighted_z`는 unscaled contract로 보존한다. 기존 eta-once loss가 원래 틀렸다고 변경하지 말라.
- Canonical MSE는 `(h - canonical_target)^2`; eta를 다시 곱하지 않는다.
- Pool/model/prompt/candidate/token/solver/target hashes와 representation/aggregation을 load 시 검증한다.
- 새 WBC에서는 같은 artifact의 p_star를 직접 읽는다. exp(g)를 다시 p_star라고 사용하거나 raw LM probability를 곱하지 않는다.

### C2. Train/dev에 같은 direct backend를 사용하라

공개 `run_nbpo_stage.py`는 train에 direct solve를 쓰지만 `build_validation_pairs()`는 legacy R-step solver를 쓴다. 동일 입력에서 train target −.1046924751, validation R1 −.3298213679가 재현됐다.

- Train/dev 모두 동일한 representation, direct inner solver, canonical builder를 사용한다.
- Dev는 train의 global lambda를 고정하고 자신의 prompt-wise inner problem만 푼다. Dev/test에서 global lambda를 다시 fit하지 않는다.
- 동일 A/mu/pi_t/lambda 입력을 두 entrypoint에 넣었을 때 pi_star와 canonical targets가 수치 오차 내에서 같아야 한다.
- Stage/CLI의 representation factory를 통일한다. Fixed-reference에 잘못된 beta, BT에 A_policy kwargs를 넘기지 않는다.
- `inner_workers`, method-specific weights/KS settings 등 선언한 supported solver options를 실제 소비되는 인자로 전달한다. 실제 worker 수를 기록한다.

### C3. Solver 검증을 독립적으로 수행하라

공개 `solve_proximal_exact()`는 SLSQP pi_hat에서 Q를 구한 뒤 extra exponential map으로 pi_star를 만들어 반환한다. Target identity check도 Q(pi_hat)을 사용하므로 구성상 성립할 수 있다. Generic CLI writer는 bad stationary result도 저장하는 반면 stage runner에는 별도 extra-map gate가 있다.

- 실제 optimizer solution을 반환하고, 바로 그 solution에서 opponent nu와 Q를 다시 계산한다.
- SLSQP success/status/message, finite/normalization/positive support, objective, genuine centered stationarity, extra-map residual을 저장한다. Status false를 조용히 성공으로 변환하지 않는다. 다른 certified fallback을 사용했다면 backend/status를 명시한다.
- 필요하면 direct solve를 더 정확하게 재실행하거나 검증 가능한 refinement를 하라. 추가 map으로 identity만 맞추어 인증하지 말라.
- Canonical target은 직접 `log(p_star/p_t)`로 계산한다. Log-domain 계산과 solver probability floor를 기록하고, floor active 여부를 검사한다. Target clipping으로 수치 오류를 숨기지 않는다.
- 하나의 central validator를 CLI, stage, writer가 공유한다.

다음 지표를 별개로 기록하라.

1. `canonical_serialization_error`: 저장 g와 log(p_star/p_t)의 차이; <1e−9. 이것은 artifact consistency이며 solver optimality 증거가 아니다.
2. `independent_stationarity_inf`: b_i=log(p_star_i/p_t_i)−eta·sum_k w_k Q_k(i; p_star), b_i에서 prompt별 p_star-weighted 평균을 뺀 값의 최대 절댓값. 이번 새 수치 계약은 **<1e−4**다. Raw dimensionless log-ratio units를 쓴다.
3. Extra-map max residual **<1e−4** 및 method-specific inner certificate.
4. Nash dual projected residual **<1e−6**, active bounds, unprojected gradient, positive train surplus s, lambda*s−1. Inactive/interior bounds에서는 unprojected residual도 확인한다.

이것은 예전 “Eq27 identity<1e−9”를 독립 stationarity로 오해한 것을 고치는 **명시적인 metric 구분**이다. 실제 기존 runtime에 더 엄격한 genuine stationarity 계약이 이미 있다면 유지한다. 결과를 본 뒤 tolerance를 조용히 완화하지 않는다.

Bounded dual projected KKT=0만으로 원래 unconstrained Nash optimum을 인증하지 않는다. Upper bound가 active이고 original stationarity가 실패하면 bound를 정당하게 확장하여 다시 푼다. Strict feasibility가 없으면 infeasible/unresolved로 기록한다. s에 epsilon을 넣거나 lambda를 simplex로 정규화해 해결한 척하지 않는다. Non-Nash methods에는 Nash complementarity를 강요하지 않는다.

### C4. 실제 BF16 path와 reference subtraction을 고쳐라

공개 `scripts/simpo_trainer.py`와 `mnpo_scripts/precompute.py`는 logits dtype으로 log_softmax/sum을 계산한다. 실제 BF16 helper 재현에서 1024-token logp의 참 변화 +.37988을 0으로 잃었다. Runtime wrapper가 FP32로 올려 주면 이 문제는 비활성일 수 있다.

- 실제 training/precompute에서 logits, log_softmax, selected logp, sequence sum, cached logp, loss, master weights, Adam state dtype을 기록한다.
- 공통 helper에서 `F.log_softmax(logits, dim=-1, dtype=torch.float32)`와 FP32 selected-token sum을 사용한다. Full-vocab FP32 메모리는 sequence chunking 또는 검증된 fused selected-logp 경로로 줄인다.
- BF16로 합친 뒤 float()하지 않는다. Reference cache가 영향을 받았으면 새 hash로 재계산한다.
- Teacher-forced response logp는 SUM이다. Prompt/padding을 mask하고 실제 EOS/EOT만 포함한다. max-token cut에 인위적 EOS를 붙이지 않는다.
- Frozen reference theta_t를 사용한다. `current_policy_logp.detach()`나 매 step 바뀌는 learner 복사는 reference가 아니다.
- 실제 collator, kernels, dtype, batching에서 같은 weights/inputs의 initial h가 반복 forward 수치 오차 수준인지 확인한다. 과거 h RMS≈.35가 남으면 mask/template/cache/dtype 차이를 좁혀 해결한다. 큰 absolute sequence logp에 대한 느슨한 relative tolerance로 h 오류를 덮지 않는다.

### C5. Candidate tokenization을 pool 단위로 고정하라

공개 pair tokenizer는 partner response 길이에 따라 prompt를 다르게 자를 수 있다. Prompt abcdef, a=xy,b=z,c=uvwxy,maxlen10,maxprompt4에서 a의 prompt가 pair(a,b)에서는 abcdef, pair(a,c)에서는 cdef가 된다.

- Prompt와 pool의 각 candidate를 한 번 tokenize한다. 공통 prompt cutoff/chat boundary를 사용하고 immutable candidate IDs, token IDs, response masks, terminal IDs를 저장한다.
- Pair construction은 후보를 선택만 한다. 같은 candidate의 token/context/mask가 모든 pair에서 같아야 한다.
- 기존 pool의 실제 affected-row 비율을 측정한다. 이미 공통 truncation이 보장된 경우 inactive라고 기록한다.
- Teacher가 점수 매긴 응답과 trainer가 보는 응답이 다르면 해당 artifact를 고치거나 재생성한다. Label을 유지한 채 response만 임의로 자르지 않는다.
- Generation text `.strip()`/decode-reencode로 실제 sampled event를 잃지 않게 raw token IDs와 termination reason을 보존한다. RM은 자신의 tokenizer를 쓰지만 같은 응답 텍스트를 받아야 한다.

### C6. Split와 resolved config를 명시하라

- `run_mnpo.py`의 substring split selection을 제거한다. `train_split=train`, `eval_split=dev`; final test는 별도 loader로 읽는다.
- 공개 final config의 주석과 달리 실제 값이 Qwen3-32B prompted judge/sampled/4+4인 문제가 있다. 이번 resolved config는 실제 frozen GPM ensemble, canonical targets,8+8을 검증한다. YAML 주석으로 판단하지 않는다.
- Existing GPM/BT 3-seed ensemble과 calibration/checkpoint hashes를 고정한다. 퇴역한 free-form Qwen3-32B judge로 되돌리지 않는다.

### C7. 최소 의미 있는 테스트 후 즉시 학습 준비를 진행하라

필수 tests는 실제 production entrypoint/helper를 대상으로 한다.

1. BT toy solved target2 → canonical builder target2.
2. 같은 입력의 train/dev direct target 일치.
3. Solver 반환 정책에서 Q를 재계산한 independent certificate; 나쁜 결과를 CLI/stage 둘 다 거부.
4. Fixed-reference/BT/adaptive representation dispatch smoke test.
5. 긴 BF16 logp 변화가 FP32 reduction에서 유지됨; actual reference initialization 비교.
6. 한 candidate의 token/context/mask가 모든 pair에서 동일.
7. Train/dev/test sentinel IDs가 정확한 loader로만 들어감.
8. eta=.5 또는2에서도 canonical target에는 eta 중복 없음.
9. 아래 WBC 직접 loss와 all-pair loss의 값 및 gradient 일치.

기존 유효한 core/dual/target/generic tests도 실행한다. Pass count와 검사 범위를 적는다. 새로 거대한 test framework나 추가 hyperparameter grid를 만들지 않는다.

## D. 새 neural realization: `nbpo_wbc`

기존 canonical MSE를 유지하고 별도 loss type을 추가한다. 기존 validator를 속여 auxiliary loss를 몰래 켜지 않는다.

같은 solver p_star에 대해:

`L_WBC = mean_over_prompts(sum_i p_star[x,i] * (-log_pi_theta(y[x,i] | x))))`

- log_pi는 **전체 vocabulary** autoregressive response logp의 합이다.
- 8개 candidate 내부 softmax로 대체하지 않는다.
- 응답 길이로 나누지 않는다. Prompt당 teacher weight 합은1이고 weights는 detach한다.
- Mini-batch에서 weight 합으로 다시 나누지 않는다. Raw pi_t(y_i)를 추가로 곱하지 않는다. exp(g)를 다시 normalization 없이 weight로 넣지 않는다.
- Finite-pool weighted BC이며 population forward-KL의 근사다. Exact theorem을 상속한다고 쓰지 않는다.

같은 pair dataloader와 policy-token exposure를 유지하기 위해 N=8의 pair(a,b)에 다음을 구현한다.

```python
# logp_a/logp_b: FP32 response log-probability SUM, shape [pairs]
# p_a/p_b: detached solver probability mass of each actual candidate
loss_per_pair = 4.0 * (p_a * (-logp_a) + p_b * (-logp_b))
loss = loss_per_pair.mean()
```

28 unordered pair가 모두 동일하게 쓰일 때 이는 prompt별 weighted NLL와 같다. Pair filtering, chosen/rejected 재정렬, 일부 candidate 누락, nonuniform sampling으로 전제를 깨지 말라. 실제 candidate index로 weights를 연결한다. Variable N은 N/2와 prompt weighting을 올바르게 일반화하기 전에는 사용하지 않는다.

WBC neutral teacher p_star=p_t도 finite sampled dataset에서는 gradient가 0일 필요가 없다. Reference(.7,.3)에서 8 IID samples가(7,1)이면 finite WBC는 empirical(.875,.125)를 향한다. **WBC에 false zero-gradient test를 넣지 않는다.** 올바른 test는 직접 계산한 empirical teacher gradient와 구현의 일치다. MSE의 zero-target initialization identity와 구분한다.

이번 primary experiment에는 별도 replay, KL anchor, target clipping, teacher damping, length penalty를 동시에 추가하지 않는다. 수정안이 실패하면 실패 결과와 원인을 보존한다. 성능을 보고 무제한 loss를 바꾸지 않는다.

## E. Dataset, pool, solver의 구체 설정

### E1. 데이터

- Base/initial policy/reference: **`meta-llama/Llama-3.1-8B-Instruct`**, 현재 프로젝트에서 사용한 동일 immutable revision/tokenizer.
- Dataset: **`PKU-Alignment/PKU-SafeRLHF`**, 기존 subset/revision을 고정한다. 현재 HF default로 조용히 바꾸지 않는다.
- Objectives K=2: helpfulness, harmlessness. 기존 검증된 3-seed GPM ensemble/calibration을 freeze한다. BT control을 추후 수행하면 대응 BT ensemble을 freeze한다.
- Tonight policy train: **2000 unique prompts**, 기존 적격7000 train manifest에서 `SHA256("20260909-repair:" + normalized_prompt)` 순으로 고정 선택한다. Dev500은 유지한다. Fresh test1000은 가능한 적격 untouched prompt groups에서 고정한다.
- Normalization: NFC, newline normalization, 양끝 whitespace 제거. Original text도 보존한다. 같은 prompt의 모든 response/comparison은 한 group이다.
- Preference-model train/calibration/test, policy splits, legacy diagnostic prompts의 교집합을 저장한다. Policy train이 preference training prompt를 쓰는지는 명시하되, 최종 generalization test는 그 학습·calibration에 노출되지 않은 group을 사용한다.
- Legacy test가 이미 선택/진단에 쓰였으면 legacy diagnostic test라고 표시한다. Untouched1000을 확보할 수 없으면 실제 적격 수를 보고하고 fresh-test claim만 보류한다. 이 이유로 다른 학습/생성 평가를 중단하지 않는다.
- Train에서 Alpaca/Arena/IFEval/GSM8K/HarmBench/MT-Bench/XSTest prompt의 exact/normalized duplicates를 제거하고 같은 train source의 hash 순으로 보충한다. 사전학습/RM의 숨은 contamination까지 없다고 주장하지 않는다.
- Train lambda는 **이 2000 prompt에서 다시 solve**한다. 7000에서 fit한 lambda를 그대로 쓰고 2000-only experiment라고 부르지 않는다.

### E2. Sampling과 teacher

- Outer stage T=1. Initial learner/reference는 같은 base다.
- Learner8와 comparator8을 독립 RNG streams로 생성한다.
- **temperature1.0, top_p1.0, top_k disabled, repetition_penalty1.0.** Native chat template를 사용한다. 이 pi_t와 reference logp의 distribution 정의를 일치시킨다.
- Prompt maximum1024, response maximum1024, train sequence maximum2048 tokens. Training source의 긴 prompt는 deterministic filter 후 같은 source의 다음 적격 prompt로 보충한다.
- Sampled EOS/EOT/token IDs, stop reason, response length, capped-horizon 여부를 보존한다. Max-length hit response를 reward를 보고 골라 제거하지 않는다.
- Learner occurrence center p_t=1/8, comparator occurrence mu=1/8, duplicates retain multiplicity. Duplicate를 임의로 dedup하여 uniform mass를 다시 주지 않는다.
- Disagreement d는 기존 별도 reference-as-learner A_ref 및 동일 game 정의를 유지한다. 재생성이 필요하면 별도 독립 reference-as-learner stream의 8개를 사용하고 actual comparator support와 어떻게 연결했는지 명시한다. 추가 generation/scoring 비용을 8+8 수에 숨기지 않는다. d를 임의로0으로 고정하지 않는다.
- 동일 learner/comparator/reference pools를 모든 arm에 공유한다. 기존 warped-sampling pool은 새 raw-policy experiment와 섞지 않고 보존한다.
- Ensemble scoring의 384-token encoder truncation 등 실제 teacher context budget을 확인한다. Fraction of truncated prompts/responses와 refusal/length별 p_star 질량을 기록한다. Encoder capacity를 넘겨 길이만 늘리지 않는다. 오늘은 teacher를 재학습하지 않는다.
- eta=1.0, beta_help=beta_harm=.25. Different old values이면 새 artifact에서 다시 solve한다.
- Adaptive-game Nash를 direct solver로 solve하고 global lambda K-vector를 저장한다. Lambda를 simplex normalize하지 않는다. C3의 independent certificate를 통과한 p_star와 g만 공유한다.
- Teacher의 per-prompt entropy, ESS=1/sum(p_star²), max mass, target RMS/quantiles, length/refusal에 따른 weights를 기록한다. 이를 보고 임계값을 튜닝하거나 target을 clip하지 않는다.

## F. 학습 비교와 runtime

Primary arms는 다음 두 개다.

1. `NBPO-MSE-fixed`: C의 correctness 수정 + canonical pairwise MSE.
2. `NBPO-WBC`: 같은 데이터/p_star/base + D의 weighted sequence NLL.

시간이 충분하면 세 번째로 `Game-utilitarian-WBC-L1matched`를 추가한다. 같은 adaptive game representation, 같은 eta/beta/pool 및 WBC로 별도 teacher를 solve한다. Train-only Nash solution에서 L=sum_k lambda_k를 한 번 고정하고 equal raw weights=(L/2,L/2)를 사용해 전체 가중치 크기를 맞춘다. Solver가 이를 다시 simplex normalize하지 않게 한다. 이는 aggregation 비교를 위한 통제 조건이며 Nash dual certificate는 N/A다. 이 control의 scale이 training-only Nash artifact에서 왔음을 명시한다. (.5,.5)를 그대로 사용하면 유효 KL 강도도 달라지므로 aggregation만의 차이라고 해석할 수 없다. 이하 Game-utilitarian-WBC는 이 L1-matched control을 뜻한다. Uniform-reference-WBC와 다른4controls/추가seeds는 후속 queue로 명시하고 오늘 두 primary를 희생하지 않는다.

공통 설정:

```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
adaptation: full_finetuning
outer_stages: 1
policy_seed: 42
train_prompts: 2000
dev_prompts: 500
fresh_test_target_prompts: 1000
candidates_per_prompt: 8
unordered_pairs_per_prompt: 28
global_batch_pairs: 32
optimizer_updates: 1750
learning_rate: 5.0e-7
optimizer: AdamW
adam_betas: [0.9, 0.95]
adam_epsilon: 1.0e-8
weight_decay: 0.0
scheduler: cosine
warmup_ratio: 0.10
max_grad_norm: 1.0
dropout: 0.0
bf16_forward: true
logp_reduction_dtype: float32
loss_dtype: float32
optimizer_master_and_state_dtype: float32
gradient_checkpointing: true
use_cache: false
deepspeed_stage: 2
max_prompt_tokens: 1024
max_sequence_tokens: 2048
microbatch_pairs_per_gpu: 1
gradient_accumulation_steps_for_4_gpus: 8
checkpoint_selection: fixed_final_horizon
```

위 YAML은 설정의 의미를 정의한다. 기존 parser가 그대로 읽을 수 있다고 가정하지 말고 실제 지원 config로 변환하여 resolved manifest를 저장하라.

- 기존 환경인 4×H200을 가정하되 실제 가용 장비를 확인한다. 4GPU×1pair×accum8=32 pairs다. 2GPU라면 accum16으로 global32를 유지하고 변경을 기록한다.
- 2000×28=56000 rows, 56000/32=1750 optimizer updates다. 이는 1 pair-row epoch이며 각 unique response가 7회 나타난다. SFT unique-response 1epoch라고 표기하지 않는다.
- Policy example order와 effective token exposure를 가능한 범위에서 맞춘다. MSE의 reference forward/cache는 WBC loss에 필요하지 않다. Exact FLOPs matched라고 부르지 말고 실측 GPU-hours와 policy/reference forward tokens를 별도로 보고한다.
- MSE reference는 candidate별 FP32 cache를 우선 사용한다. 단, 실제 동일 token IDs/masks를 사용한 frozen online reference와 비교하여 일치한 새 cache만 허용한다. 검증할 수 없으면 MSE는 frozen online reference로 실행한다. WBC 학습 loss에는 reference forward/cache를 넣지 않는다.
- 같은 LR도 서로 다른 loss의 gradient scale을 같게 만들지는 않는다. Preclip norm, clip 빈도, FP32 master update norm, logratio 통계를 기록한다. 큰 norm만 보고 clip100/no-clip으로 바꾸지 않는다.
- First20 updates에서 memory/step time을 실측한다. BF16 model에 일반 AdamW를 붙인 것만으로 FP32 master가 있다고 가정하지 말고 ZeRO state를 확인한다.
- 사전 고정된 최종 horizon의 checkpoint를 사용한다. 중간 checkpoint를 저장해도 외부 benchmark 최고점으로 선택하지 않는다.
- Dev의 nMSE/sign/Pearson/Spearman, mean logratio, sequence length, reference drift를 고정 빈도로 기록한다. 회귀 score를 stop gate로 쓰지 않는다.
- NaN/OOM/corrupt artifact 등 실행 불능은 원인을 수정해 재개한다. 낮은 성능만으로 구현 버그라고 단정하여 설정을 바꾸지 않는다.

## G. Teacher와 policy의 문제를 분리하는 진단

새 학습과 병행할 수 있는 CPU 집계를 우선하고 추가 대규모 탐색은 하지 않는다.

1. 저장된 base/N700/RB1200 responses에서 고정된 benign prompt 100개를 골라 모델명을 숨기고 refusal, 극단적 축약, early EOS, 미완성, 정상 답변을 확인한다. GSM8K의 parser failure와 수학적 오답을 구분한다. Keyword만으로 최종 원인을 단정하지 않는다.
2. 높은 p_star 후보가 base pool 평균보다 helpful하고 safe한지 살펴본다. Length/refusal 편향과 독립 local RM의 불일치를 보여준다. 학습 teacher 자체의 높은 score만으로 성공을 주장하지 않는다.
3. Finite-pool target consistency, train fitting, held-out fitting, 실제 생성 quality를 별도 열로 기록한다. Floor nMSE=.016을 unseen neural predictability98.4%라고 쓰지 않는다.
4. MSE/WBC가 모두 teacher를 따라가면서 usefulness를 떨어뜨리면 teacher/task/objective 문제의 가능성이 커진다. WBC만 회복하면 realization 문제의 증거가 된다. Game-utilitarian도 비슷하게 회복하면 Nash 고유의 이득을 아직 주장하지 않는다.

## H. API-free evaluation

평가 script/dataset/model revisions와 generation protocol을 고정한다. Base cache는 prompt/template/decoding/max tokens 등이 일치할 때만 재사용한다. Model loading/export 후 작은 일치 검사를 통해 잘못된 checkpoint, adapter merge, stop token 적용을 확인한다.

표준 generation은 기존 비교와 일치하는 user-only native chat template, greedy temperature0/top_p1, max_new_tokens2048, seed20260909를 기본으로 한다. GSM8K는 기존 zero-shot-CoT prompt/extractor를 유지한다. 이미 고정된 benchmark별 설정이 있다면 모든 arm과 base에 공통 적용하고 기록한다. Benchmark prompt를 임의로 짧게 만들지 않는다.

Primary evaluation:

| Dataset | 전체 수/범위 | 보고 항목 |
|---|---|---|
| SafeRLHF held-out | dev500 / fresh test 최대1000 | objective vector, V,d,s,delta vs reference, CI, pool과 실제 생성의 구분 |
| IFEval | official541 | strict prompt/instruction accuracy; loose secondary |
| GSM8K | official test1319 | same zero-shot-CoT EM, parse-failure rate |
| HarmBench | official text test의 실제 대상 | harmful rate, category별 n, evaluator 종류 |
| XSTest | safe250 / unsafe200 | safe over-refusal, unsafe refusal/appropriate handling, 판정법 |
| AlpacaEval prompt set | 805 | local Skywork RM WR vs base, response tokens |
| Arena-Hard-v2 | hard500 / creative250 | 같은 local proxy를 subset별 보고, length/refusal |

MT-Bench80의 2turn 생성은 secondary로 두고, 주요 평가와 두 arm을 완료한 뒤 여유가 있으면 진행한다. Official judge를 호출하지 않으므로 공식 MT score를 만들지 않는다.

구현 시 주의 사항:

- Evaluation-only local model은 동일 revision의 `Skywork/Skywork-Reward-V2-Qwen3-8B`다. 공식 model card의 sequence-classification/logits와 chat template를 사용한다. Free-form judge를 만들지 않는다.
- Proxy WR은 score(model)>score(base)를1, 동점을.5로 두고 같은 prompt에서 비교한다. Scalar scoring에는 pairwise permutation이 필요하지 않다. NaN/tie/failed 수를 보고한다. AlpacaEval LC/Arena 공식 score라고 명명하지 않는다.
- IFEval은 공식 evaluator, GSM8K는 고정 parser를 사용한다. Outputs와 failure logs를 저장한다.
- HarmBench classifier에는 공식 generation truncation(통상512tokens)을 같은 tokenizer로 적용한다. Copyright/hash-check 등 category별 공식 경로를 확인하고 모든 category를 같은 classifier에 넣지 않는다. 이전 보고가 다른 평가법을 썼다면 기존 수치를 덮어쓰지 말고 정정표를 만든다.
- XSTest는 적절한 기존 local refusal classifier 또는 모델명을 숨긴 manual audit를 사용한다. HarmBench harmfulness classifier를 refusal detector로 사용하지 않는다. 미구현이면 responses를 생성하고 미판정 건수와 방법을 밝힌다. 단순 keyword 결과는 diagnostic으로 표시한다.
- Refusal/length/EM을 함께 보고 trade-off를 판단한다. 안전성 개선만으로 Pareto 개선이라고 부르지 않는다.
- Paired prompt bootstrap CI를 고정 seed로 계산한다. 이는 training-seed uncertainty가 아니다. 1seed 결과로 3seed 안정성을 주장하지 않는다.
- 이미 본 외부 benchmark는 개발 과정에 노출된 이력을 기록한다. 이번 설정을 미리 고정해도 과거 test 노출이 사라지지는 않는다.
- Raw responses, checkpoint hash, prompt hash, gen/eval config, sample counts, failures를 모두 저장한다.

## I. Deadline과 자원 사용

- 짧은 초기 감사·수정과 실제 20-update profile을 바탕으로 ETA를 계산한다. 4GPU 학습과 별도 GPU judge를 동시에 돌릴 계획은 실제 장비 할당부터 확인한다. CPU tests/solver/report는 가능하면 병렬 실행한다.
- **두 primary arm과 평가 시간을 확보한다.** 마지막 약2시간은 checkpoint export, 생성·채점·보고용으로 먼저 배정하고 실제 평가 throughput으로 조정한다.
- 1750updates×2가 가능하면 고정 계획을 실행한다. 불가능하면 **본학습 결과를 보기 전에** 두 arm 공통 horizon을 줄이고, 예를 들어1000updates를 prospective manifest에 고정한다. 이는1750 실험과 다르다고 명시한다. 첫 arm만1750, 두 번째만 줄이고 matched라고 부르지 않는다.
- 수정한 NBPO-WBC를 먼저 실행해 생성 평가에 넘기고 같은 사전 고정 budget의 MSE control을 이어서 실행해도 된다. 결과에 따라 arm을 추가/중단하지 말고 compute가 부족하면 실제 진척을 보고한다.
- 세 번째 Game-utilitarian-WBC는 두 primary와 평가를 마칠 수 있는 실측 여유가 있을 때만 실행한다. 6methods×3seeds를 자동으로 시작하지 않는다.
- 30분마다 이용자에게 완료 항목, 새로운 증거, 다음 작업, 실측 ETA를 짧게 보고한다. API 비용0과 GPU 사용 시간은 따로 기록한다.
- Deadline에 미완료여도 실제 checkpoint, 생성 건수, metric, 남은 ETA를 정직하게 보고한다. 유리한 subset만 전체 결과처럼 제시하지 않는다.

## J. 필수 성과물

1. 실제 run source의 snapshot/hash와 새 branch의 review 가능한 diff/commit.
2. `audit_findings.json`: 각 결함의 present/fixed/inactive/unknown, repro, affected artifacts.
3. `resolved_protocol.yaml`: model/data/teacher/sampler/solver/target/loss/optimizer/actual horizon/decoding 고정값.
4. Split/pool/token/target/checkpoint manifest와 hashes, reference cache invalidation 기록.
5. Correctness test report. 기존 unit pass와 실제 GPU integration을 구분한다.
6. 두 arm의 train/dev metrics, throughput, clipping/update, teacher entropy/ESS, solver certificate.
7. Base와 후보의 responses, API-free benchmark table, length/over-refusal, CI, failed counts.
8. `morning_report.md`: 수정, 실험 조건, 결과, 남은 원인 가설, next step. nMSE gate와 acceptance를 분리한다.
9. v6 개정 메모: empirical importance measure, bounded-dual certificate, neural approximation, Table13 caption 정정, 최신 downstream 결과. WBC 성공 전에 성공했다고 본문에 쓰지 않는다.

## K. 논문과 후속 실험

v6를 기준으로 수정한다. v4의 긍정적인 표를 checkpoint/judge/protocol 검증 없이 복원하지 않는다. 새 WBC는 기존 Eq26의 단순 버그 수정이 아닌 새로운 realization이므로 MSE 대조군을 남긴다.

이번에 회복하면 3seeds, 6개 내부 controls, uniform-WBC, 가까운 외부 baselines를 다음 단계로 계획한다. 비교 후보는 SPPO/NLHF, REBEL, MOPO/PROSPER다. 각 방법의 training data/judge/compute를 맞추고 다른 setup의 논문 숫자와 직접 SOTA 비교하지 않는다.

문헌은 설계 근거로 사용한다. MOPO의 QLoRA LR를 full FT에 복사하거나 SPPO eta를 NBPO에 대입하지 않는다. PPO/GRPO로 즉시 전환하여 sum(Q_k/s_k)를 reward로 사용하지 않는다. 초기 reference에서 s=0이므로 그 단순 reward는 정의되지 않으며 epsilon 추가는 별도 objective가 된다.

참고할 1차 자료:

- REBEL: https://arxiv.org/html/2404.16767v4
- SPPO: https://arxiv.org/html/2405.00675v3
- MOPO policy extraction: https://arxiv.org/html/2505.10892v2
- NLHF: https://arxiv.org/pdf/2312.00886
- PROSPER: https://arxiv.org/html/2602.19041v1
- IFEval: https://github.com/google-research/google-research/tree/master/instruction_following_eval
- HarmBench: https://github.com/centerforaisafety/HarmBench
- XSTest: https://github.com/paul-rottger/xstest
- AlpacaEval: https://github.com/tatsu-lab/alpaca_eval
- Arena-Hard: https://github.com/lmarena/arena-hard-auto
- Skywork RM: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B

첫 응답에는 실제repository/version/GPU/time 확인 결과와 수정할 파일을 짧게 제시하고, 곧바로 작업을 수행하라.
