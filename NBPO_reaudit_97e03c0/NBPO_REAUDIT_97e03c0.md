# NBPO 재검증: b949919 → 97e03c0

## 결론

**이전의 P0 canonical API 누락과 여러 panel에 남아 있던 target 반올림은 수정됐다. 정상 입력에서 scorer → prompt-wise worker → canonical builder → JSON → canonical dataset validator가 실제 CPU 연결 테스트를 통과했다.** 다만 **현재 논문 Algorithm 1의 neural-candidate local acceptance가 campaign training 경로에 연결됐다는 근거는 여전히 없다.** 따라서 수치/target 생성 경로와 전체 알고리즘 구현을 구분해 판정한다.

이번에는 “함수 이름이 존재한다”에서 끝내지 않고 새 writer와 verified loader를 실행하고 실제 target 값을 비교했다. 이전 지적이 해소된 항목은 해소됐다고 판정했다. 남은 schema-alias 및 cached row-hash 검사는 정상 입력 계산의 오류가 아니라 **일관되지 않은 입력을 조기에 차단하는 추가 검증 공백**이다. 아래 인위적 반례가 실제 실행에서 발생했다는 주장은 하지 않는다.

원본 논문·저장소를 수정하지 않았다. 이 보고서는 업로드 ZIP의 특정 commit에 대한 것이며, live GitHub main이나 실제 pod의 환경을 독립적으로 fetch/검증한 보고서는 아니다.

## 1. 대상, diff, 실행 범위

| 항목 | 확인값 |
|---|---|
| 이전 ZIP commit | `b9499195ebd88d9161eda487bb7fd6167e4b8f80` |
| 새 ZIP commit | `97e03c053e08892f72b68aec4203de071537ce5a` |
| 새 ZIP SHA256 | `512259b037e9ea77a1457fb2b64350ad7150f74706783187e06008bc65f86622` |
| 변경 | 기존 35개 수정, 18개 추가, 삭제 없음 |
| 추가 18개 | 이전 재검증 보고서/재현기/로그 사본; 새로운 `tests/` 파일은 아님 |
| ZIP 내 원본 파일 | 1,993개 |
| 테스트 이후 원본 파일 변경 | 0개; 새 `.pytest_cache` 4개만 생성 |
| 주요 코드 | `analysis/sub_20260914/code_snapshot_20260917_union/` 및 root `scripts/nbpo/`, `mnpo_scripts/` |
| 실험 범위 | CPU synthetic JSONL/NPZ, 실제 source modules, independent primal comparison, 기존 pytest, file promotion fixture |
| 미실행 | 실제 LLM 학습·생성, 실제 AutoTokenizer 모델 로딩, GPU/bf16/DDP, Arrow materialization, actual benchmark/checkpoint 검증 |

이하 `S/`는 위 union snapshot directory를 뜻한다. 소스의 원래 줄 번호와 SHA256는 `code_evidence.html` / `evidence_manifest.json`, diff는 `modified_files.diff`에 있다.

## 2. 이전 지적별 상태

| 항목 | 이번 판정 | 근거 |
|---|---|---|
| LR/RR scorer 의미 | 해결 유지 | K=2/4 actual scorer 출력이 기준 행렬과 정확히 일치 |
| 누락 RR prompt 유지 | 해결 유지 | 한 RR edge의 한 order가 없으면 해당 prompt 제외 |
| canonical API 배포 | **해결 확인** | 15개 module import, real builder/writer/loader 실행 |
| serializer 전파 | **해결 확인** | base/v2/uw1/uw1c/uw3/us1/ut1 모두 float64 fixture 통과 |
| 구형 scorer 실수 사용 | **해결 확인** | explicit acknowledgement 없이 중단 |
| stale schema loader | **기존 사례 해결** | `lr_rr_v2`가 없는 shard 거부; 새 정상 shard 통과 |
| root README 경로 | **주요 구분 수정** | current prompt-wise path와 Global Nash control 구분 |
| split 이름만 해시 | **주요 문제 수정** | split별 candidate identity digest로 변경; 아래 cached-hash 검증 보완 가능 |
| legacy gate NaN 승인 | **해결 확인** | NaN/±Inf/0/negative 모두 거부 |
| local acceptance orchestration | **미해결 / full algorithm 판정 제한** | campaign jobs는 여전히 direct trainer, 기존 gate evaluator는 global |
| style의 token/word 혼용 | **수정 확인** | 동일 counter 사용, 실제 supported words-mode fixture에서 self-difference 0 |
| style 미수렴 미표시 | **수정 확인** | fit status + downstream failure check |
| style 한 order 유지 | **수정 확인** | actual paired loader가 한 order prompt 제외 |

## 3. Canonical API는 실제 값을 주고 통과했다

`build_nbpo_pairs.py`가 `canonical_data`와 `canonical_logratio`를 처리하고, `load_canonical_artifact()`가 추가됐다. `solve_nbpo_dual.py`에는 `write_generic_solution_artifact()`가 추가됐다. [E02](code_evidence.html#E02), [E03](code_evidence.html#E03)

### Import

다음 5개 panel × 3개 method의 실제 module을 깨끗한 subprocess에서 import했다.

- Panels: `uw1`, `uw1c`, `uw3`, `us1`, `ut1`.
- Methods: `pw_nbpo`, `pw_fixedref`, `prosper`.

**15/15 성공.** Missing `load_canonical_artifact` / `write_generic_solution_artifact` 문제는 더 이상 재현되지 않는다.

### 연결 테스트

K=2(SafeRLHF wrapper)와 K=4(WildChecklists wrapper)에 각각 두 prompt, prompt당 learner 8개와 reference 8개를 주었다. Pairwise observations를 JSONL로 만들고 다음 source functions를 실제로 실행했다.

```
union_score_panel.main
  → solve_pros4_targets_<panel>.load_scores
  → solve_pros4_pw_nbpo_<panel>._init / _solve_one
  → build_nbpo_pairs.build_rows(canonical_data=...)
  → quantize_canonical_row
  → JSON write/read
  → validate_canonical_pair_dataset
```

각 panel의 **56개 pair rows(2×28)**가 통과했다. Canonical `g_i-g_j`와 최종 직렬화 target 차이의 최대값은 **4.440892098500626e-16**, 다른 prompt 데이터를 바꾼 뒤 첫 prompt policy의 변화는 **0**이었다. Pair flip 이후에도 token/weight/target 정합성을 유지하고 validator를 통과했다. 모든 local worker가 certified를 반환했다. [E04](code_evidence.html#E04), [E06](code_evidence.html#E06), [E19](code_evidence.html#E19)

Generic writer → certified loader도 K=2/4의 각 prompt에서 별도로 실행했다. 저장한 target과 복원한 target의 오차는 **0**이며, NPZ target을 바꾸고 manifest를 유지하면 loader가 거부했다.

**범위:** 이것은 실제 수치 함수/파일 형식 사이의 연결 테스트다. Worker의 `main()` 전체(Tokenizer metadata lookup 및 Arrow 포함), Trainer, GPU 학습을 실행한 것은 아니다. Synthetic tokens와 synthetic manifest identifiers를 썼으며, 실제 checkpoint provenance가 검증됐다고 주장하지 않는다. Generic writer의 1-prompt artifact 검사와 campaign의 per-prompt collection export도 구분한다.

## 4. Float64 serializer 수정은 전파됐다

다음 7개 모듈의 serializer를 import한 그대로 실행했다.

```
solve_pros4_targets.py
solve_pros4_targets_v2.py
solve_pros4_targets_uw1.py
solve_pros4_targets_uw1c.py
solve_pros4_targets_uw3.py
solve_pros4_targets_us1.py
solve_pros4_targets_ut1.py
```

8개 후보의 모든 28개 pair에 `1e-12` 최소 mass를 포함해 JSON round-trip과 actual canonical validator를 검사했다. **7/7 통과, 최대 target drift 0, 최소 mass 1e-12 보존.** 이전 `0.3987761199510089` drift 반례 역시 새 구현에서는 0이다. [E05](code_evidence.html#E05)

복사된 여러 모듈을 모두 수정한 것은 확인했지만 공용 함수로 refactor한 상태는 아니다. 이후 generator가 다시 오래된 source에서 산출물을 만들지 않도록 parameterized regression을 repository tests에 유지하는 것이 좋다. 이번 commit에는 `tests/` 추가·수정이 없다.

## 5. Tensor/loader 연결의 수정과 남은 검증 범위

Actual scorer로 K=2 및 K=4, 각각 세 prompt의 서로 다른 LL/LR/RR 행렬을 구성했다. `A_LL`, `A_LR`, `A_RR`, `A_policy`, `A_ref` 각각의 기준 대비 오차는 **0**이다. Missing RR one-order fixture에서 `p0`만 제외되고 `p1,p2`가 남았다. 이전 learner .75/reference-tie 반례도 이제 surplus +.25를 반환한다. [E04](code_evidence.html#E04)

`union_score_uw.py`는 기본 실행을 중단하고 generalized scorer로 안내한다. Old scorer를 일부러 사용하려면 긴 acknowledgement flag를 명시해야 한다. 이는 재현용 legacy 보존과 current 경로의 분리로 적절하다. [E07](code_evidence.html#E07)

5개 current base loader가 정상 `lr_rr_v2` shard를 읽고, schema가 없는 동일 shard는 거부함을 실제 호출로 확인했다.

### 보완 가능 P2: schema 문자열만으로 alias 내용까지 검증되지는 않는다

`load_scores()`는 shard schema와 file hash를 검사하지만, `A_policy == A_LR`, `A_ref == A_RR` 및 `bank_ids`의 대응을 검사하지 않는다. Fixture에서 **schema는 v2로 두고 `A_policy`만 `A_LL`로 바꾼 뒤 해당 NPZ의 manifest hash를 갱신**하면 loader가 받아들인다. 이 fixture의 policy tensor 차이는 .625였다. [E05](code_evidence.html#E05)

이는 정상 scorer 출력에서 오류가 다시 발생했다는 뜻이 아니다. **부분 migration 또는 잘못 조립한 “v2” artifact에 대한 방어가 아직 약하다**는 의미다. 현재 문제가 없던 데이터에 단순 tag 추가만 하여 migration하면 안 된다. 명시적 semantic array와 alias가 함께 저장되므로 cross-check를 추가하는 비용은 작다.

## 6. 남은 핵심: local acceptance가 실제 training에 연결되지 않았다

최신 원고 Algorithm 1은 finite target 인증 후 neural candidate를 만들고 development check를 통과할 때만 다음 정책으로 승인한다. 실패하면 parent policy를 유지한다. Code-only 수치 certificate는 이 마지막 단계를 대체하지 않는다.

### 변화는 있었으나 global stage runner의 변화다

`run_nbpo_stage.py`에는 exact/generic dispatch와 canonical configuration이 추가됐다. NaN/Inf gate도 수정됐다. 하지만 `solve_stage_finite_pool()`은 여전히 전체 prompt batch를 한 번의 shared-weight core solve에 넘기고, monitoring은 `eval_game_value.evaluate_game_value()`를 사용한다. 그 함수는 prompt 평균 surplus로 scalar minimum을 만든다. [E10](code_evidence.html#E10), [E11](code_evidence.html#E11), [E12](code_evidence.html#E12)

Local surpluses가 +.2와 -.1인 두 prompt fixture에서 actual evaluator와 actual gate를 연결하면:

```
evaluator.min_surplus = +0.05
apply_gate.accepted   = True
```

파일시스템에서도 실제 synthetic candidate가 승격됐다. 이는 **Global Nash control에는 허용되는 동작**이며 global evaluator를 무조건 local로 바꾸라는 뜻은 아니다. 다만 prompt-wise NBPO용 gate로 재사용할 수는 없다.

### Current campaign path

`S/make_pros_train_jobs.py`, `S/panel_stage2.py`, `mnpo_scripts/run_mnpo.py`는 이번 commit에서도 byte-identical이다. 추적한 경로는:

```
job generator → torch.distributed.run → run_mnpo
  → trainer.train() → trainer.save_model()
```

이 사이 또는 이후에 local candidate evaluation, rejection, accepted checkpoint promotion을 연결한 코드는 해당 경로에서 확인되지 않는다. [E13](code_evidence.html#E13), [E14](code_evidence.html#E14), [E15](code_evidence.html#E15)

**완료 기준:** canonical/prompt-wise 경로를 호출하는 하나의 stage controller에서 local development array와 declared coverage/nMSE/finite checks를 평가하고, 실패 시 parent를 반환하며, 평가에 사용하는 checkpoint가 accepted artifact임을 기록한다. 별도 controller가 pod에 있다면 그 code/config/promotion log를 공개해야 한다. 실제 과거 실험이 ungated projection이었다면 논문에서 그 사실을 분리 기술해야 한다. Gate 추가만으로 과거 실행에도 gate가 있었다고 소급해서는 안 된다.

## 7. NaN gate, pool digest, naming

### NaN/Inf gate: 해결 확인

Actual `apply_gate()`를 임시 디렉터리에서 실행했다. NaN, +Inf, -Inf, zero, negative는 parent를 반환하고 promotion을 만들지 않는다. +.1은 versioned path 및 symlink로 승격한다. 고립 AST가 아닌 실제 import 함수와 filesystem test다. [E10](code_evidence.html#E10)

### Pool digest: 기존 split-name 오류 해결, cached-hash check는 남음

`split_pool_digest()`는 prompt ID, role, occurrence index, candidate ID와 response SHA를 split별로 해시한다. Train/dev digest가 다르고, 해당 split 밖의 변경은 무시하며, response text와 그 SHA를 함께 변경하면 digest가 변한다. 이전 `sorted(outputs)` 오류는 수정됐다. [E05](code_evidence.html#E05)

다만 기존 `response_sha256`이 있으면 이를 신뢰하고 text에서 다시 계산하지 않는다. **Text만 바꾸고 cached SHA를 유지한 fixture**에서는 digest가 그대로다. `load_pool()`도 각 JSONL 파일의 outer hash는 확인하지만 이 per-row text/SHA 일치까지 확인하지 않는다. 정상 producer가 SHA를 올바르게 기록했다면 문제없다. Strong byte-binding을 원하면 hash를 다시 계산하고 기존 필드와 비교하거나 token event hash를 포함해야 한다. 이는 P2 입력 일관성 보강이다.

### Panel label

Generated `uw1c/uw3/us1/ut1`은 각각의 label로 수정됐다. 하지만 source `solve_pros4_targets_uw1.py:93`의 `PANEL_LABEL`은 아직 `"UF-4"`다. `uw1` 실행을 계속 쓰면 provenance label이 틀릴 수 있다. 수치 target의 문제는 아니며, label 수정 시 generator가 이 literal을 치환하는 anchor도 함께 수정해야 한다. [E16](code_evidence.html#E16), [E17](code_evidence.html#E17)

## 8. Style-control 수정은 원래 결함을 해소했다

두 평가 스크립트가 같은 `length` closure를 arm과 baseline에 사용한다. Default는 하나의 tokenizer, 명시적 `--length-unit words`는 양쪽 모두 같은 whitespace counter다. Stored arm `n_tokens`만 별도 사용하는 경로는 제거됐다. [E08](code_evidence.html#E08), [E09](code_evidence.html#E09)

실행 결과:

| 검사 | 결과 |
|---|---|
| 동일 text, 서로 다른 stored token metadata | style difference `[0,0,0,0]` |
| 한 presentation order만 존재 | 해당 prompt 제외, dropped count 1 |
| Fit max iterations 1 | `converged=False`, gradient infinity norm .4768117 |
| Converged intercept-only case | `converged=True`, beta ≈ -1.11e-16 |
| Single-arm CLI, words mode | point .5, 성공 |
| Paired CLI, 같은 synthetic arm, 10 bootstrap | difference 0, failed replicates 0 |

Default AutoTokenizer 모델은 이 환경에 transformers/weights가 없어 로딩하지 않았다. Words-mode의 actual CLI execution과 tokenizer branch의 source inspection을 구분한다. 이 fixture는 구현 invariant 검사이지 실 데이터의 SC 점수 측정이 아니다. 코드도 해당 metric을 local implementation으로 구분하며, 수정된 code만으로 기존 표의 score가 재계산됐다고 볼 수 없다.

## 9. 테스트 수치

### 저장소 pytest

```
core + dual + targets + provenance subset:
113 passed, 5 deselected in 77.95s

legacy pipeline including CPU dry-run and six CLI --help checks:
8 passed in 23.09s
```

합계 **121 passed, 5 deselected**. 이전과 같은 heavyweight 의존성 관련 5개를 제외했다. Pipeline 8개는 legacy/Global Nash 경로 테스트이므로 이를 latest local acceptance 검증으로 계산하지 않는다.

### 기존 audit 재실행

15 check 중 **14 PASS, 1 global/local mismatch observation**. Canonical signature를 확인하려고 `None` tensor를 넘기던 구형 harness는 API가 구현된 후 본문까지 들어가므로 적절하지 않아, signature 검사와 값이 있는 별도 fixture로 바꿨다. `math`를 새로 쓰는 legacy gate AST에도 해당 module을 제공했다. 이는 audit harness 조정이며 source 수정이 아니다. 초기 stale harness의 `NoneType.shape`를 구현 결함으로 세지 않았다.

6개 작은 feasible games의 independent primal 비교 최대 오차:

| 통계 | 값 |
|---|---:|
| Objective difference | 1.8607337892717624e-13 |
| Policy coordinate difference | 1.0229770346786848e-7 |
| Unprojected dual residual | 2.5374147227807953e-13 |
| Canonical target identity | 6.0176308380732735e-12 |
| Analytic/autograd gradient | 4.440892098500626e-16 |
| Analytic/autograd Hessian | 8.881784197001252e-16 |

MSE-reduction, eta-once, response-logp forward/backward 검사도 유지됐다. Trainer loss 함수는 heavyweight Trainer 전체가 아니라 원문 AST 고립 검사라는 범위가 동일하다.

### 새 검사

`expanded_results.json`: **58 check, 54 PASS / 4 OPEN / 0 HARNESS_ERROR**.

OPEN 네 개는 서로 독립적인 네 버그가 아니다: alias/schema 검사를 K=2와 K=4로 반복한 두 사례, cached SHA 불일치 한 사례, legacy global gate를 local로 사용할 수 없다는 한 사례다. 새 정보가 없는 old-snapshot 성능 결론이나 test-set metric은 추정하지 않았다.

## 10. 권장 다음 작업

1. **우선 local acceptance controller를 actual NBPO job에 연결**하거나 해당 실행을 ungated projection으로 명시한다. Candidate/accepted checkpoint 식별과 failure handling의 통합 테스트가 필요하다.
2. Schema alias/shape/bank mapping과 pool row hash 검증을 보강하고 UW1 provenance label을 바로잡는다. 정상 tensor/target에 영향을 주지 않는 검증·기록 수정이다.
3. 현재 parameterized regressions를 `tests/`에 포함한다. 올바른 파일 몇 개만 수정되고 생성된 다른 panel이 과거 코드로 남는 회귀를 방지한다.
4. 실제 tokenizer/dataset/7B model의 한 batch/한 optimizer-step smoke test 및 resolved runtime code hash를 기록한 후 대규모 corrected run을 수행한다.
5. 잘못된 텐서로 학습한 과거 checkpoint는 코드 수정만으로 바뀌지 않는다. 새 target/checkpoint/evaluator manifest와 score cell을 연결해 논문을 갱신한다.

**최종 판단:** 이번 commit은 이전 두 차례보다 분명히 진전됐다. P0 import 문제와 serializer 전파 문제는 해결됐고, 정상 입력의 local numerical/canonical path는 이번 테스트 범위에서 통과했다. 남은 가장 중요한 차이는 이 경로 뒤의 neural acceptance를 논문 Algorithm 1대로 실제 실행하는가이다.
