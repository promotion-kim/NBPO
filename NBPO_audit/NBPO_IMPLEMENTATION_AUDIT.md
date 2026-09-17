# NBPO 논문–구현 대조 보고서

## 판정 요약

**현재 prompt-wise NBPO의 핵심 수치 solver와 neural regression 계산은 작은 독립 테스트에서 논문과 일치했다. 그러나 업로드된 저장소 전체를 최신 논문 Algorithm 1의 실행 가능한 일관된 구현이라고 판정할 수는 없다.**

가장 중요한 문제는 (1) 최근 union scorer가 learner–learner / learner–reference / reference–reference 텐서를 solver의 의미와 다르게 연결하고 있다는 점, (2) 최근 canonical target 경로가 공개 패키지에 존재하지 않는 API를 호출한다는 점이다. 추가로 README의 구형 global 알고리즘 안내, 현재 training 경로의 local acceptance orchestration 부재, 인증 이후 target 양자화가 있다.

이것은 **업로드된 코드의 오류·불일치에 대한 판정**이다. 모든 과거 실험이 이 코드로 실행됐거나 모든 표의 값이 잘못됐다는 판정은 아니다. 실제 pod의 원시 텐서·판정·resolved config·checkpoint별 실행 manifest는 제공되지 않았다.

## 검증 대상과 재현성

| 항목 | 대상 |
|---|---|
| 논문 | 대화의 최신 `main.tex`: prompt-wise NBPO로 통합, 이전 global 방법은 Global Nash control |
| 코드 | 업로드한 `NBPO-main.zip` |
| ZIP comment | `92f75310188d11be8141be4da7d207c33b356162` |
| ZIP SHA256 | `acc2b49d140fccbe40e21bad0ffd84424e475d8015a97c76d2d1093bcc92df8e` |
| 논문 SHA256 | `c16f447ca05b757c7637fc9b1df8277c701870d4e1bf681f4eccfbc600cb0477` |
| 브랜치 범위 | ZIP은 `NBPO-main`이다. 이전에 요청된 `exp/iclr27-table1-v2`와 동일하다고 확인한 것은 아니다. |
| 실행 환경 | Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2 |
| 실행 범위 | CPU 수치 테스트, 실제 scorer를 synthetic JSONL에 실행, package import, 기존 test 일부. LLM 학습/생성 미실행. |
| 변경 여부 | 원본 논문·소스 코드 수정 없음. 발췌한 소스의 bytes를 ZIP 원본과 대조해 일치 확인. |

아래에서 `S/`는 `analysis/sub_20260914/code_snapshot_20260917_union/`를 뜻한다. 모든 코드 근거는 `code_evidence.md` 또는 `code_evidence.html`에 원본 파일 경로·원본 줄 번호·SHA256와 함께 있다. 실제 결과는 `audit_results.json`, 실행 가능한 재현기는 `audit_checks.py`이다.

## 1. 같은 이름 아래에 다른 구현 경로가 있다

### 1.1 README 경로: 이전 global / R-step 근사

README가 안내하는 `scripts/nbpo/run_nbpo_stage.py`는 `mnpo_scripts/nbpo_solver.py::solve_nbpo_dual`을 호출한다. 이 경로는 `(K,)` shared multiplier, prompt-averaged surplus, R-step fixed-point update, sampled/Rao–Blackwell pair target을 사용한다. README도 R=1을 명시한다.

최신 논문은 prompt별 lambda, 직접 최적화한 finite-pool target, canonical pairwise log-ratio, local safeguard를 정의한다. 따라서 이름이 NBPO라는 이유만으로 README 경로가 현재 방법을 구현한다고 볼 수 없다. 다만 구형 코드를 Global Nash/legacy control로 유지하는 것은 가능하다.

**근거:** E01, E19; `mnpo_scripts/nbpo_solver.py:111–175,179–199`.

### 1.2 최근 실험 경로: local wrapper + 새 generic core

`S/solve_pros4_pw_nbpo_uw1.py::_solve_one`은 prompt x 하나만 `A[:, x:x+1]`로 잘라 generic core에 전달한다. 내부 core가 prompt 평균을 계산해도 X=1이므로 이 호출 경로는 실제 local solve다.

호출은 `inner_solver="exact"`, `dual_solver="root"`다. 파일명에 `pw`가 남아 있는 것은 그 자체로 문제가 아니다. 또한 이 경로에서 `--weight-l1`는 **Nash에서는 사용하지 않는다**고 명시하고 `weight_l1=None`을 사용한다. 이 코드를 근거로 “NBPO가 lambda를 L1-normalize한다”고 지적하면 잘못이다.

**근거:** E06 (`63–87`, `96–116`).

### 1.3 Generic core의 기본값 주의

`mnpo_scripts/nbpo_generic.py::solve_finite_pool`을 전체 prompt batch에 기본값으로 호출하면 local root/exact 경로와 동일하지 않다. 현재 기본값은 fixed-point/subgradient이며 weight는 `(K,)`이다. 권위 있는 단일 entrypoint가 promptwise 분리와 exact/root 선택을 책임져야 한다.

**근거:** E12 (`1048–1105`).

## 2. A01 / P0 — Scorer와 solver의 텐서 의미가 다르다

### 논문 정의

현재 논문 Section 5.2는 learner bank Y~pi_t, comparator bank Z~pi_ref를 별도로 뽑는다. 이때:

- A_policy[k,x,i,j] = Delta_k(y_i,z_j | x): **learner–reference**.
- A_ref는 disagreement 계산용 **reference–reference** 행렬이다.
- 동일 reference bank Z를 양쪽에 쓰면 A_ref[k,x,j,j'] = Delta_k(z_j,z_j' | x)이다.

A_policy로 정책 value와 opponent를 구하고, A_ref로 동일 reference game의 fallback d를 계산한다. 비교은행을 추가로 독립 추출한 reference–reference 구성도 가능하지만, 그 경우도 두 bank 모두 reference 분포와 대응 관계를 명시해야 한다.

### 실제 union scorer

`S/union_score_panel.py:129–167`과 `S/union_score_uw.py:114–154`는:

```
A_policy <- learner–learner (LL)
A_ref    <- learner–reference (LR)
RR       <- mean abs margin diagnostic only
```

로 저장한다. `S/solve_pros4_targets_uw1.py:133–150`은 이 fields를 그대로 읽고, `AdaptiveGameRepresentation`은 A_ref를 `compute_disagreement_point`에 넘긴다. 후자는 입력을 reference-as-learner tensor로 해석한다. 주석만 다른 것이 아니라 실제 값이 다르게 계산된다.

**근거:** E03–E06, E09–E10. 실제 pipeline의 scorer 호출은 E18 `94–111`.

### 실제 scorer로 재현한 반례

한 prompt, 두 objectives, learner/reference 각 8개를 사용했다. Learner끼리는 모두 tie, reference끼리도 모두 tie이며, 모든 learner는 모든 reference를 0.75의 확률로 이긴다고 구성했다. 양 순서의 판정 JSONL을 실제 scorer에 넣었다.

| 값 | 논문식 LR/RR 연결 | 현재 scorer LL/LR 연결 |
|---|---:|---:|
| A_policy entries | +0.25 | 0 |
| A_ref entries | 0 | +0.25 |
| Policy value | +0.25 | 0 |
| Disagreement | 0 | +0.25 |
| 각 objective surplus | **+0.25** | **−0.25** |

따라서 같은 관측 비교 그래프에서도 이 텐서 연결은 양의 surplus를 음수로 바꾸며 finite-pool feasibility 판단까지 바꾼다. 수치 solver가 정확하게 풀더라도 논문에 명시된 finite problem이 아니다.

처음 stage에서 pi_t=pi_ref라서 두 bank가 같은 생성 모델에서 왔다는 사실만으로 LL, LR, RR의 실현된 행렬이 동일해지지는 않는다. 이 반례는 **finite-pool 정의 불일치**를 보이는 것이며, 원래 population preference가 실제로 이 숫자라는 주장이나 전체 LLM 성능 차이의 원인 규명은 아니다.

추가로 RR 판정을 모두 제거해도 scorer가 이 prompt를 `prompts_complete=1`로 남기는 것을 재현했다. `reference_disagreement.prompts_without_any=1`만 기록한다.

### 최소 수정 방향

행렬 이름만 치환하지 말고 semantic tensor를 분리한다.

```
A_LL : learner–learner labels
A_LR : learner–reference game payoffs
A_RR : reference–reference fallback payoffs

NBPO solver: A_policy=A_LR, A_ref=A_RR
Single-objective pair labels: A_LL
```

`build_panel_softlabels.py`는 현재 A_policy를 LL로 해석하므로, scorer만 바꾸고 이 consumer를 그대로 두면 baseline label도 잘못된다(E21). 저장소의 기존 `scripts/nbpo/build_preference_tensor.py`에는 올바른 policy/reference 역할 분리가 이미 있다(E22). 이 부품을 활용하되 최신 canonical/local 경로에 연결하고, tensor별 response-ID 순서와 요구되는 RR completeness를 검증하는 방식이 좋다.

이미 LR/RR raw judgments가 있다면 **재판정·재생성 없이 재스코어링부터** 시작할 수 있다. 수정한 tensor로 feasibility, target, 공통 retained prompts를 다시 계산하고, 영향을 받은 정책은 다시 학습해야 한다. 기존 target과 새 tensor를 혼용하지 않는다.

## 3. A02 / P0 — Canonical 경로가 업로드된 패키지에서 실행되지 않는다

최근 base loader에는 다음 import가 있다.

```python
from scripts.nbpo.build_nbpo_pairs import build_rows, load_canonical_artifact
from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
```

그러나 root 파일에는 `load_canonical_artifact`와 `write_generic_solution_artifact`가 없다. Root pair builder는 `TARGET_MODES=("sampled","rao_blackwell")`이고, `build_rows`는 `canonical_data`를 받지 않는다.

최근 promptwise wrapper는 `canonical_logratio`와 `canonical_data={g,p_star,p_t}`를 넘긴다. 실제로 다음 두 오류를 재현했다.

```
ImportError: cannot import name 'load_canonical_artifact'
TypeError: build_rows() got an unexpected keyword argument 'canonical_data'
```

**근거:** E05:48–49, E06:229–234, E07:46,68–93, E08. 이 import 실패는 transformers/accelerate 미설치와 별개의 저장소 내부 API 불일치다.

`canonical_data`를 무시하는 `**kwargs`를 추가해 예외만 없애는 것은 해결이 아니다. 현재 builder의 non-sampled branch는 여전히 sampled opponent에 대한 margin difference이며 canonical g의 차이가 아니다.

**조치:** 실제 실험에 사용한 canonical loader/writer/builder를 완전하게 공개하고, 하나의 깨끗한 checkout에서 scorer→target→pair JSONL→dataset까지 이어지는 CPU smoke test를 추가한다. `/work/.../code`에만 존재하는 다른 버전을 import해야 실행되는 상태를 제거한다.

이 문제는 “실험이 실행된 적이 없다”는 증거가 아니다. **pod에 있던 실행 코드와 배포 ZIP이 다를 수 있다는 문제**다. E17의 실행 cwd/PYTHONPATH가 pod 경로를 가리킨다.

## 4. 일치하는 핵심 계산과 독립 테스트

### 4.1 Local dual, adaptive inner solve

Generic core는 정확한 tensor가 주어졌을 때 다음 목적을 푼다.

```
max_p sum_k lambda_k V_k(p) - KL(p || p_t)/eta
V_k(p) = -beta_k log sum_j mu_j exp(-(A_k^T p)_j/beta_k)
```

Gradient의 opponent 항 A_k nu_k와 negative KL 항이 일치하고, Hessian은 negative covariance와 negative diagonal KL Hessian의 합이다. 최종 p에서 opponent를 다시 계산하며, root는 log lambda 좌표에서 s−1/lambda=0을 찾는다. **Root solver는 gradient descent가 아니므로 lambda를 곱하지 않은 residual을 사용하는 것이 맞다.**

Canonical check도 solver가 돌려준 p에서 Q를 재계산하며 stationarity, logratio identity, 원래 dual residual, floor activity를 검사한다. 단, 이는 **입력 tensor와 직렬화 전 수치 문제에 대한 검사**다.

**근거:** E11–E12.

서로 다른 작은 feasible game 6개에서 이 core를 별도의 primal SLSQP 구현과 대조했다.

| 최대 오차 / 결과 | 관측값 |
|---|---:|
| 모든 사례의 core certificate / 독립 primal success | 6/6 |
| Primal objective 차이 | 1.85e−13 |
| Policy coordinate 최대 차이 | 1.01e−7 |
| s−1/lambda residual | 1.62e−13 |
| Canonical target identity residual | 3.05e−11 |
| Gradient analytic/autograd 차이 | 4.44e−16 |
| Hessian analytic/autograd 차이 | 8.88e−16 |

현재 `_solve_one`의 함수 body를 수정하지 않고 고립 실행한 테스트에서는 다른 prompt 입력을 바꿔도 첫 prompt 결과의 변화는 0이었다. Prompt별 multiplier도 서로 달랐다. 이 테스트는 missing import 문제를 해결한 전체 모듈 실행과 구분한다.

별도 RPS stress test에서는 direct inner residual이 2.76e−11이고, 구형 undamped map은 100회 후 direct solution에서 L2 거리 0.339였다. Uniform reference, 비균일 proximal center를 사용한 **fixed-multiplier inner test**이지 논문 속 예제의 모든 상수 또는 full dual을 그대로 재현한 것은 아니다.

### 4.2 Neural loss, target scale, frozen reference

`mnpo_trainer.py:432–460`에서 NBPO는 residual을 example별로 제곱하고 `723–724`에서 mean한다. Canonical target에는 eta를 다시 곱하지 않는다. 현재 reference-forward path는 `no_grad` 후 detach하고, `run_mnpo.py:318–330`은 별도 reference model을 eval/frozen 상태로 둔다.

서로 +1/−1인 residual에 대해 MSE=1, squared mean=0을 비교해 **올바른 MSE**가 사용됨을 확인했다. Canonical target이 eta=2.5에서도 이중 scaling되지 않는 것도 확인했다. 원래 Trainer의 loss 함수 body만 고립 실행한 검사로, Trainer 전체를 mock해 end-to-end 검증했다고 주장하지 않는다.

`response_logps` 실제 함수는 causal shift, prompt/padding mask, full-vocabulary log-softmax, sequence-sum을 구현한다. 독립 torch 표현과 비교한 forward error는 0, backward 최대 오차는 5.96e−8이었다. 실제 tokenizer/checkpoint 데이터를 넣은 전체 경로까지 검증한 것은 아니다.

**근거:** E13–E16.

## 5. A04 / P1 — Algorithm 1의 local acceptance gate가 현재 training 경로에 연결되지 않았다

최근 `make_pros_train_jobs.py:149–162`와 `panel_stage2.py:337–349`가 발행하는 job은 `mnpo_scripts.run_mnpo`를 직접 호출한다. 이 entry는 학습 후 model을 저장한다. 이 호출 경로에서 **candidate evaluation→local gate→parent retention/accepted checkpoint promotion**을 수행하는 orchestration은 확인되지 않는다.

구형 `run_nbpo_stage.py`에는 gate가 있지만, 그 경로는 구형 global solver와 global evaluator를 사용한다. `eval_game_value.py:41–64`는 prompt를 먼저 평균한 surplus의 minimum만 반환한다.

두 prompt의 local surplus가 +0.2와 −0.1일 때 legacy evaluator의 `min_surplus=+0.05`가 되는 것을 재현했다. 이것은 global rule에서는 가능한 평가지만 최신 논문의 local-positive gate와는 다르다.

또한 legacy `apply_gate(NaN,...)`는 고립 테스트에서 accepted=true였다. `NaN <= 0`이 false이기 때문이다. 이 검사는 파일 promotion 부작용만 대체했으며, upstream에서 실제 NaN이 발생해 과거 모델이 승인됐다는 주장은 아니다.

**근거:** E16–E20.

**조치:** 단일 최신 entrypoint에서 candidate 저장과 accepted-policy 승격을 구분하고, finite values/local signs/선언된 dev 조건을 명시적으로 검사한다. 실패하면 parent를 유지한다. 실행한 적 없는 gate가 과거 run에도 적용됐다고 논문에 쓰지 않는다. 과거 실험이 ungated projection이라면 그 근사 범위를 정직하게 기술한다.

## 6. A05 / P1 — 인증한 canonical target을 이후 반올림으로 바꾼다

`S/solve_pros4_pw_nbpo_uw1.py:184–188`은 직렬화 전 target identity를 확인한다. 이후 `239`에서 `quantize_canonical_row`를 호출한다. 이 함수는 p* mass를 소수점 10자리로 반올림한 뒤 logratio를 다시 계산한다(E05:83–109).

실제 함수를 실행한 예:

```
p*_a = 1.49e-10, p*_b = 0.2, p_t,a = p_t,b = 0.125
원래 target       = -21.01763689754899
반올림 후 target  = -21.4164130175
절대 차이         = 0.3987761199510089
```

p*_a는 보고된 1e−10 threshold보다 크므로, 이 크기의 target drift가 threshold만으로 방지되지는 않는다. 이 숫자는 선택한 반례의 실제 함수 결과이며, 실제 학습 데이터 전체의 평균 오차를 측정한 값은 아니다.

이는 단순히 JSON 포맷을 바꾼 것이 아니라 **인증된 p*가 정의한 canonical target을 다른 target으로 바꾸는 추가 근사**다. 반올림된 row 내부에서 identity가 일치하는 것과 원래 optimizer KKT를 유지하는 것은 다르다.

**조치:** p*, p_t, g를 충분한 precision으로 round-trip하고 g 차이를 source of truth로 사용한다. unavoidable quantization이면 원래 target 대비 오차 분포와 post-serialization consistency/optimality를 별도로 보고한다. 실수 epsilon과 실제 로그 오차를 혼동하지 않는다.

## 7. 보조 지적: 결과 연결과 평가 지표

### 7.1 Pool hash가 데이터가 아니라 split 이름만 해시한다

`S/solve_pros4_pw_nbpo_uw1.py:267–268`의 train/dev pool SHA는 둘 다 `object_hash(sorted(outputs))`다. outputs가 train/dev dict이면 이는 `['dev','train']`만 해시하며 실제 prompt/response pool 내용이 바뀌어도 변하지 않는다. 같은 파일의 `panel: 'UF-4'`도 UW 생성 파일에 남아 있다.

전체 provenance가 전혀 없다는 뜻은 아니다. Target npz와 dataset에 별도의 hash가 있는 점은 긍정적이다. 다만 이 **pool hash라는 이름의 두 필드**는 의도한 binding을 수행하지 않는다. Prompt IDs, response IDs/token events, 배열 bytes, 해당 split manifest를 실제로 해시해야 한다(E06).

### 7.2 LC score의 원천을 확인해야 한다

`S/lc_winrate.py`는 `y=a+b*tanh(length_gap/sigma)`를 OLS로 적합하고 intercept를 반환한다. 코드 자신도 official LC GLM이 아니라고 명시한다(E23).

따라서 이 출력을 사용했다면 표에는 custom/local length-adjusted score로 구분해야 한다. **현재 paper LC cell이 반드시 이 파일에서 왔다고 확인한 것은 아니다.** Cell→scoring artifact 연결을 확인해야 한다.

### 7.3 Arena-Hard judge: 코드상 기본값과 실제 run은 다르다

최근 `S/judge_eval_batch_ahv2.py:258–264`의 기본값은 Qwen2.5-72B-Instruct, revision `495f39366efef23836d0cfae4fbe635880d2be31`이다. 더 오래된 `judge_eval_pairwise.py`에는 Qwen3-14B 경로가 있다(E24–E25).

이제 **적어도 최근 Arena-Hard v2 evaluation script의 기본 judge**는 확인했다. 그러나 CLI override가 가능하고 실제 complete.json이 없으므로, paper의 특정 score가 반드시 72B judge 결과라고 확정하지는 않는다. Training PSC judge와 benchmark judge를 동일시하지 않는다.

### 7.4 Sampling 정의

Historical `generate_pros_pool.py`에는 temperature0.8/top-p0.9와 uniform occurrence mass가 명시돼 있다(E26). 이 sampler는 unmodified raw autoregressive pi와 같은 분포가 아니다. 이 경로를 사용한 실험을 paper의 iid pi_t sampling과 연결하려면 decoder-induced distribution을 별도 설명하거나 proposal/likelihood correction을 명시해야 한다. 현재 union pool이 이 historical generator로 실제 만들어졌는지는 settings artifact 없이는 단정하지 않는다. Token별 top-p는 support도 바꾸므로 단순 global temperature 상수로 correction할 수 있다고 가정하지 않는다.

## 8. 저장소 test와 한계

실행한 저장소 테스트는 core, dual, target, provenance 파일의 일부다.

```
113 passed, 5 deselected in 56.26s
```

처음 core/target/provenance를 실행했을 때 105 passed/5 failed였다. 다섯 실패는 transformers 또는 accelerate 미설치에 따른 환경 차단이었다. 해당 다섯 개를 제외하고 dual tests를 더해 위 결과를 얻었다. 환경 차단을 코드 defect로 집계하지 않았다.

별도 재현기는 15개 검사항목을 기록한다. 이 중 7개는 정상 동작 확인이고, 나머지는 API/텐서/직렬화/구형 gate의 실패·불일치 재현이다. `REPRODUCED_FAILURE`는 테스트 프로그램 자체가 실패했다는 뜻이 아니라 **해당 결함을 관찰했다**는 상태다. 여섯 독립 primal comparison은 한 항목 안에 여섯 사례로 저장돼 있다.

이 test pass는 모든 2천여 파일의 정합성이나 모든 baseline을 보증하지 않는다. 특히 GPU mixed precision, 실제 LLM tokenizer/dataloader, 모든 distributed config, 원시 judge outputs와 모든 benchmark score를 재실행하지 않았다. Snapshot README는 raw pools/targets/config/checkpoints가 pod/HF에 있다고 명시한다. 이 ZIP에는 원시 numerical target npz가 없다.

## 9. 수정 및 재실험 우선순위

1. **실험에 사용한 정확한 source snapshot을 완성한다.** Missing canonical API와 pod 경로 의존을 해결하고, import smoke test를 통과시킨다.
2. **LL/LR/RR 역할을 분리한다.** NBPO는 LR/RR, pair-label baseline은 LL를 사용하게 하고 모든 loader/adapter를 점검한다. Scorer 반례와 missing-RR rejection을 회귀 테스트로 고정한다.
3. **직렬화한 target을 다시 검증한다.** Delta h를 원래 g와 비교하고 rounding-induced drift를 없애거나 측정한다.
4. **Local acceptance를 실제 실행 경로에 연결한다.** Candidate와 accepted checkpoint를 구분하고 NaN/Inf/누락/음수localsurplus 테스트를 포함한다.
5. **원시 판정으로 영향 범위를 산정한다.** 수정 전후 d, local feasibility, retained IDs, lambda, p*, target RMS 차이를 동일 prompt에서 보고한다. LR/RR가 있으면 먼저 재사용한다.
6. **영향받은 target의 정책을 재학습한다.** Matched panel의 prompt set이 바뀌면 baseline도 새 공통 set에서 비교한다. 과거 값과 수정 후 값을 run_id로 분리한다.
7. **논문 표현을 마지막에 갱신한다.** Core solver 수식은 유지할 수 있다. 실행 확인 없이 'Algorithm1 완전 구현', '모든 gate 적용', '현재 점수 official LC'로 바꾸지 않는다.

## 10. Claude Code용 완료 기준

- Clean checkout에서 current promptwise module import 성공; canonical API smoke test 성공.
- Scoring output에 LL/LR/RR, row/column response IDs, proposal roles 명시.
- Synthetic graph에서 expected A_policy=LR, A_ref=RR; RR missing이면 error 또는 명시적 invalid status.
- Six small exact solves가 independent primal과 objective tolerance1e−8 안에서 일치.
- 다른 prompt의 데이터 변경이 local multiplier에 영향을 주지 않음.
- Canonical pair target round-trip max error≤1e−9 또는 사전 정의·보고된 현실적 오차.
- h-target MSE, eta-once, sequence-sum masking/gradient tests 유지.
- Local-negative / all-positive / NaN / Inf / missing-monitoring 사례에 대해 gate unit/integration tests.
- Tensor→target→dataset→candidate→accepted checkpoint→evaluation cell의 전체 provenance chain.
- Tiny CPU end-to-end smoke test와 별도의 실제 모델 단일-batch/step 검증 후에만 대규모 재실행.

## 재현 명령

```bash
python audit_checks.py \
  --repo /absolute/path/to/NBPO-main \
  --out ./audit_results.json
```

NumPy, SciPy, torch가 필요하다. 현재 script는 대형 모델을 불러오지 않으며 scorer I/O는 임시 디렉터리만 쓴다. `mnpo_loss`, local worker, legacy gate의 일부 검사는 **원문 함수 AST 고립 실행**으로 명시돼 있다. 소스가 바뀌면 함수·출력 schema에 맞춰 테스트도 점검한다.

**최종 결론:** NBPO의 local constrained-proximal 수학을 다시 설계해야 한다는 근거는 이번 검증에서 발견되지 않았다. 우선 해결할 것은 그 수학에 들어가는 비교 텐서, canonical target 전달, 실제 학습 승인 경로와 배포 소스의 연결이다. 수정 전 현재 union 경로의 결과를 최신 paper Algorithm1의 일치된 구현 결과로 주장하는 것은 권하지 않는다.
