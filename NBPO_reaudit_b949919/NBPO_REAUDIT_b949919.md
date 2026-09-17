# NBPO 재검증: 92f7531 → b949919

## 결론

**가장 중요한 LR/RR 텐서 연결 오류는 `union_score_panel.py`에서 실제로 수정되었다. 그러나 공개 저장소 전체가 현재 논문의 prompt-wise NBPO Algorithm 1과 일치하는 상태는 아니다.** Canonical loader/writer/builder API 누락이 그대로여서 새 `uw1c`·`uw3` 실행 파일 모두 clean import에 실패한다. Float64 직렬화 수정은 `uw1`·`uw1c`에만 적용되었고, 새로 추가한 `uw3` 및 기존 `us1`·`ut1`은 여전히 기존 반례에 실패한다. Local acceptance orchestration과 root README도 이전 상태다.

이 판정은 **첨부 ZIP의 코드와 아래 테스트**에 대한 것이다. 실제 pod에서 실행한 소스가 ZIP과 같은지, 기존/수정 후 checkpoint가 어느 target을 썼는지는 run manifest가 없어서 확정하지 않았다. 현재 점수의 상승·하락, 실제 학습 실행의 성공 여부를 추정하지 않는다. 논문과 저장소를 수정하지 않았다.

## 1. 검증 대상·변경 범위

| 항목 | 확인값 |
|---|---|
| 이전 ZIP commit comment | `92f75310188d11be8141be4da7d207c33b356162` |
| 새 ZIP commit comment | `b9499195ebd88d9161eda487bb7fd6167e4b8f80` |
| 새 ZIP SHA256 | `e7ceabe88667efa202260d069dd556d8d3adea498ea1ee7c423314a271c41537` |
| 수정된 기존 파일 | 4개: snapshot README, generalized scorer, soft-label loader, uw1 serializer |
| 추가 파일 | 23개: 이전 audit 자료 11개, 실행/평가 파일 12개 |
| 삭제 파일 | 0개 |
| root canonical API / core / trainer / stage runner | 이전 ZIP과 byte-identical |
| 저장소 tests 변경 | 없음; audit 자료 복사는 회귀 테스트의 새 실행을 뜻하지 않음 |
| 실행 후 원본 source bytes 변경 | 없음 |

라이브 GitHub main을 독립적으로 fetch해 확인한 것이 아니라 ZIP comment 및 실제 파일 bytes로 대조했다. `diff_inventory.json`, `modified_files.diff`, `unchanged_critical_files.json`, `audit_manifest.json`을 참조한다. 이하 `S/`는 `analysis/sub_20260914/code_snapshot_20260917_union/`이다. 모든 소스 줄 번호는 새 ZIP 기준이다.

## 2. 수정된 부분: 실제 회귀 테스트 통과

### 2.1 A01 — generalized scorer의 텐서 역할

현재 `S/union_score_panel.py`는 아래 세 블록을 별도로 만들고 저장한다.

```text
A_LL = learner–learner
A_LR = learner–reference
A_RR = reference–reference

A_policy = A_LR
A_ref = A_RR
```

이 연결은 논문 Section 5.2의 `A[i,j]=Delta(y_i,z_j)`와 reference–reference disagreement 정의에 맞다. LR을 억지로 antisymmetric하게 만들지 않으며, LL/RR에만 skew-symmetry와 zero diagonal을 적용한다. [E01](code_evidence.html#E01)

이전의 실제 scorer 반례를 재실행했다. 모든 LL/RR 비교가 동점이고 모든 learner가 모든 reference를 0.75로 이기는 입력에서:

| 값 | 92f7531의 결과 | b949919의 결과 | 논문상 기대 |
|---|---:|---:|---:|
| Policy value | 0 | +0.25 | +0.25 |
| Disagreement | +0.25 | 0 | 0 |
| Surplus | −0.25 | **+0.25** | +0.25 |

추가로 **K=2와 K=4, 각각 prompt 3개**의 서로 다른 LL/LR/RR 비교 fixture를 actual scorer에 입력했다. 출력의 `A_LL`, `A_LR`, `A_RR`, `A_policy`, `A_ref` 다섯 배열 모두 예상 배열과 최대 절대 오차가 **0**이다.

### 2.2 RR 완결성과 baseline labels

RR 판정 전체를 누락하면 scorer가 결과를 내지 않고 `no prompt survived scoring`으로 거부한다. RR의 한 edge에서 한 presentation order만 없앤 추가 테스트에서도 해당 prompt 전체만 제외되고 나머지 두 prompt가 보존되었다. 이는 K=2/4 모두 통과했다.

`S/build_panel_softlabels.py:32–54`는 이제 `A_LL`을 명시적으로 읽으며, `A_LL`이 없는 이전 schema를 거부한다. 실제 loader 함수 body에 corrected scorer 출력을 넣어 LL 전달 오차 **0**과 stale-schema 거부를 확인했다. 여기서는 datasets 패키지 전체가 아니라 **수정하지 않은 loader 함수 AST**를 고립 실행했다. [E02](code_evidence.html#E02)

### 2.3 A05 — uw1/uw1c의 serializer

`solve_pros4_targets_uw1.py` 및 `solve_pros4_targets_uw1c.py`는 fixed-decimal rounding을 제거했다. 실제 serializer 함수와 `json.dumps/loads`를 통해 이전 반례의 target drift가 **0**이 되었다.

추가로 최소 확률이 `1e-12`인 8-candidate/28-pair fixture를 만들고, 실제 `validate_canonical_pair_dataset()`까지 실행했다. 두 경로 모두 확률 0으로의 손실 없이 통과했다. 이는 **합성 row의 JSON round-trip + 실제 lightweight validator** 검사이며, missing API를 우회해 전체 pipeline이나 Arrow materialization을 성공시킨 검사가 아니다. [E05](code_evidence.html#E05), [E23](code_evidence.html#E23)

## 3. 아직 P0: 새 실행 파일도 canonical API 때문에 import에 실패

새 `S/solve_pros4_targets_uw1c.py:54–55`와 `..._uw3.py:54–55`가 요구하는 API는 다음과 같다.

```python
from scripts.nbpo.build_nbpo_pairs import build_rows, load_canonical_artifact
from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
```

그러나 root 패키지에는 두 이름이 없으며, `build_rows()`의 signature도 여전히 다음과 같다.

```python
build_rows(prompt_ids, objectives, A_policy, nu, lam, betas, policy,
           ref_seed_of, rng, target_mode, meta_ids, provenance)
```

즉 `canonical_data`를 받지 않고, root `TARGET_MODES`는 `sampled`, `rao_blackwell`뿐이다. 두 root 파일은 이전 commit과 동일하다. [E03](code_evidence.html#E03), [E04](code_evidence.html#E04), [E25](code_evidence.html#E25)

별도의 깨끗한 Python subprocess에서 repository root와 snapshot directory를 PYTHONPATH에 둔 뒤 아래 **6개 실제 module import**를 실행했고 모두 같은 오류로 실패했다.

```text
solve_pros4_pw_nbpo_uw1c
solve_pros4_pw_fixedref_uw1c
solve_pros4_prosper_uw1c
solve_pros4_pw_nbpo_uw3
solve_pros4_pw_fixedref_uw3
solve_pros4_prosper_uw3

ImportError: cannot import name 'load_canonical_artifact'
```

별도 writer 존재 검사와 builder keyword 검사도 여전히 실패한다.

```text
write_generic_solution_artifact: absent
TypeError: build_rows() got an unexpected keyword argument 'canonical_data'
```

이는 transformers/accelerate/datasets 미설치로 인한 오류가 아니다. 같은 저장소 내부 symbol이 없다는 문제다. Pod의 별도 `/work/.../code`가 필요한 symbol을 제공한다면 **그 실험 소스가 공개 ZIP과 다르다**는 뜻이며, 실제 source를 공개하거나 의존 구조를 정리해야 한다. 단, 그러한 pod 차이는 여기서 직접 확인하지 않았다.

**수정 완료 기준:** 실제 canonical loader/writer/builder를 동기화하거나, 사용하지 않는 global-script import를 공용 helper에서 분리하고 canonical builder를 올바르게 구현한다. `canonical_data`를 무시하는 `**kwargs` 또는 sampled target 대체는 수정이 아니다. Clean checkout에서 6개 import와 tensor→target→JSONL→dataset smoke test를 통과해야 한다.

## 4. A05는 전체 해결이 아니다: uw3/us1/ut1에 남은 반올림

| 실제 serializer | 이전 반례 target drift | 1e−12 mass 보존 | 28-pair validator |
|---|---:|---|---|
| `solve_pros4_targets_uw1.py` | 0 | 예 | 통과 |
| `solve_pros4_targets_uw1c.py` | 0 | 예 | 통과 |
| **`solve_pros4_targets_uw3.py` (새 파일)** | **0.3987761199510089** | 아니오: 0이 됨 | 실패 |
| `solve_pros4_targets_us1.py` | 0.3987761199510089 | 아니오 | 실패 |
| `solve_pros4_targets_ut1.py` | 0.3987761199510089 | 아니오 | 실패 |

새 `uw3` wrapper는 실제로 `solve_pros4_targets_uw3`를 base로 import한다. 그러므로 `uw1`에 수정이 들어갔다는 이유로 `uw3`에도 적용됐다고 볼 수 없다. [E06–E08](code_evidence.html#E06), [E13](code_evidence.html#E13)

실패하는 코드는 여전히 다음과 같다.

```python
CANONICAL_DECIMALS = 10
row[key] = round(float(row[key]), CANONICAL_DECIMALS)
```

원래 반례 `p_a=1.49e-10`, `p_b=.2`, `p_t,a=p_t,b=.125`에서:

```text
original:  -21.01763689754899
rounded:   -21.4164130175
difference: 0.3987761199510089
```

`1e-12`을 포함한 fixture는 `Canonical targets require strictly positive solver masses`로 거부된다. **수정된 serializer를 공통 모듈 하나로 옮기고 모든 active solver에서 공유**하거나 generator 산출물을 전부 갱신한 뒤 경로별 parametrized test를 두어야 한다. Historical artifact를 덮어쓰지 않는 것과 버그 있는 함수를 active 경로로 남기는 것은 다르다.

## 5. 텐서 수정의 경계: 구형 scorer와 stale loader

### 5.1 같은 current snapshot의 union_score_uw.py는 여전히 이전 연결

`S/union_score_uw.py`는 변경되지 않았다. 실제 이 스크립트를 새로운 fixture에 실행하면 `A_policy=LL`, `A_ref=LR`가 재현된다. 이 파일은 frozen 전일 snapshot에만 남은 것이 아니라 **current snapshot에도 callable 상태**다. 실제 corrected run이 generalized scorer만 사용했다면 그 run에 문제를 소급하는 것은 아니다. 다만 잘못된 경로를 재사용할 수 있다. [E09](code_evidence.html#E09)

추천: current alias는 generalized scorer에 위임하고, 재현용 old implementation은 historical 전용 경로·flag로 제한한다.

### 5.2 새 NBPO loader는 이전 schema를 거부하지 않는다

Baseline loader는 `A_LL`을 확인하지만, `uw1c`의 `load_scores()`는 여전히 `A_policy/A_ref`, file hash, shape만 확인한다. **실제 old scorer가 쓴 파일을 실제 `load_scores()` 함수에 넣었더니 잘못된 old tensor가 그대로 반환**되었다. 파일 hash가 맞다는 것은 파일이 바뀌지 않았다는 뜻이지, 텐서의 역할이 올바르다는 뜻은 아니다. [E10](code_evidence.html#E10)

이 테스트는 missing imports 때문에 **수정하지 않은 loader 함수 AST를 고립 실행**한 것이다. Full module import가 해결됐다는 의미가 아니다. Corrected output만 전달하면 LR/RR 계산은 맞으며, 지적은 잘못된 artifact 혼입에 대한 검증 공백이다.

추천: `tensor_schema_version`, role tags와 response-bank identifiers를 저장하고 load 시 검증한다. `A_policy == A_LR`, `A_ref == A_RR` 및 objective/shape/response ID를 함께 확인하고, 이전 역할 schema는 explicit historical mode 외에는 거부한다.

## 6. 나머지 기존 문제의 상태

| 항목 | 상태와 근거 |
|---|---|
| A03 root README | 이전 shared-multiplier/R=1 pipeline을 Algorithm 1 implementation으로 안내. Root README는 변경 없음. [E19](code_evidence.html#E19) |
| A04 local acceptance | `make_pros_train_jobs.py`, `panel_stage2.py`, `run_mnpo.py`는 변경 없음. 추적한 경로는 trainer 실행/저장까지이며 최신 local candidate gate 연결은 확인되지 않음. [E14–E16](code_evidence.html#E14) |
| A06 pool hashes | 새 uw1c/uw3에도 `object_hash(sorted(outputs))` 사용: split 이름 목록을 해시함. `panel: UF-4`도 복사 상태. 다른 target/artifact hash의 존재를 부정하는 지적은 아님. [E12](code_evidence.html#E12) |
| A07 legacy NaN gate | 실제 legacy `apply_gate` function에 NaN을 넣은 고립 테스트가 여전히 accepted=true. Upstream 실험에 NaN이 실제 전달됐다고 판단한 것은 아님. [E17](code_evidence.html#E17) |
| A08 LC source | 기존 custom OLS LC 파일은 변경 없음. 현재 논문 cell과 실제 metric artifact의 연결은 여전히 확인하지 못함. |

논문 Algorithm 1의 마지막 단계는 candidate를 무조건 저장하는 것과 달리, development 조건을 통과한 경우에만 다음 정책으로 승인한다. 따라서 corrected scorer와 numeric certificate만으로 full Algorithm 1 일치를 선언할 수 없다. 별도의 외부 gate가 실제로 수행됐다면 해당 source와 promotion logs를 추가하면 된다.

## 7. 이번 commit의 새 SC 평가 코드에서 추가 발견

이 항목들은 NBPO 학습 수식의 오류와 구분한다. **새 SC 열을 채우거나 해석하기 전에** 확인할 사항이다. 두 스크립트가 local implementation임을 명시한 점은 좋지만, 그 명시가 아래 구현상의 비대칭을 해결하지는 않는다.

### B01 — model token count와 whitespace word count 혼용

`ahv2_style_control.py`와 `ahv2_sc_paired.py`는 arm에 `style(response, n_tokens)`, baseline에 `style(text)`를 호출한다. `style()`의 길이 fallback은 `len(text.split())`이다. 생성기의 `n_tokens`는 model output token 수다. [E20–E22](code_evidence.html#E20)

같은 문자열을 양쪽에 놓고 arm `n_tokens=18`, baseline whitespace words=7인 fixture를 실제 paired loader에 통과시키면 길이 feature가:

```text
(18 - 7) / (18 + 7) = 0.44
```

로 나온다. 내용과 표면 형태가 동일한 두 응답에서 normalized style difference는 0이어야 하므로 이 invariant가 깨진다. 18은 fixture의 declared token count이며 이 문자열을 실제 tokenizer로 측정했다고 주장하지 않는다. 더 일반적으로 **서로 다른 길이 단위를 사용한다는 코드상의 불일치**를 보인다.

양쪽 답변을 같은 고정 tokenizer와 동일한 special-token convention으로 다시 세거나, 양쪽 모두 같은 word count를 쓰고 이름도 word count로 표시해야 한다. 저장된 답변/판정은 재사용할 수 있고, feature·fit·bootstrap만 다시 계산하면 된다.

### B02 — fit의 iteration exhaustion을 실패로 보고하지 않음

`fit()`는 convergence criterion이 만족되면 coefficient를 반환하지만, iteration budget을 모두 소진한 경우에도 같은 형태의 coefficient를 반환한다. 실제 함수에 `iters=1`을 지정한 테스트에서 gradient infinity norm 약 `0.4768117`인 결과가 반환됐다. Caller는 `None` 여부로만 실패를 판단하므로 이런 미수렴을 `failed_replicates`에 반영하지 못한다. [E20](code_evidence.html#E20)

이는 모든 실제 default-budget fit이 실패했다는 뜻이 아니다. **실패를 탐지하는 API가 불완전**하다는 뜻이다. `converged`, residual, iteration count를 반환하고 downstream bootstrap에서 검사해야 한다.

### B03 — 한 presentation order만 있는 prompt도 SC loader가 유지

실제 paired loader에 한 order만 준 fixture가 유지된다. 양 순서 완결성을 요구하는 raw 비교와 같은 관측 집합을 주장하려면 해당 조건을 맞춰야 한다. Available-order estimator를 의도했다면 그것을 명시하고 대응 raw 점수/분모도 동일하게 계산해야 한다.

## 8. 재실행한 테스트와 한계

### 저장소의 기존 테스트

```text
113 passed, 5 deselected in 67.32s
```

대상: `test_nbpo_core.py`, `test_nbpo_dual.py`, `test_nbpo_targets.py`, `test_nbpo_provenance.py`의 실행 가능한 subset. 이전 검증과 같은 5개 의존성 관련 항목은 제외했다. 이 환경에는 transformers, accelerate, datasets가 없으며 이를 code defect로 세지 않았다.

### 이전 재현기 재실행

15개 check에서 10개 PASS. 3개 canonical API failure, legacy global/local mismatch, legacy NaN acceptance가 남아 있다. `REPRODUCED_FAILURE`는 audit script 자체의 예외가 아니라 코드 결함을 관찰했다는 상태다. 이전에 `REPRODUCED_FAILURE`였던 generalized tensor role, missing RR, uw1 quantization은 이번에 PASS로 바뀌었다.

### 새로 작성한 경로별 검사

23개 check: 9 PASS, 6 import BLOCKED, 7 OPEN, 1 contract OBSERVATION. `HARNESS_ERROR`는 0. 특정 조건을 여러 module에 적용하므로 check 수를 독립적인 결함 수로 해석하지 않는다.

### 핵심 수치 계산도 다시 확인

6개 small feasible game의 nested exact/root solver와 독립 primal optimization의 최대 objective 차이 **1.86e−13**. Policy coordinate 차이 최대 **1.03e−7**. Local-worker independence, adaptive gradient/Hessian, mean-of-squared-residuals, canonical eta-once, response-logp forward/backward 검사는 유지됐다.

이 결과는 kernel correctness를 지지하지만, clean module import와 실제 end-to-end 경로를 대신하지 않는다. 특히 **LLM 학습/생성, 실제 tokenizer batch, GPU/bf16/DDP, Arrow materialization, actual checkpoints, 실제 benchmark 값은 재실행하지 않았다.** 코드에서 scalar/array 함수 body를 고립한 검사는 결과마다 scope를 명시했다.

## 9. 다음 수정과 실행 순서

1. **P0 canonical API 배포 정합성**을 먼저 해결한다. Clean checkout에서 current 6개 entry import/--help를 통과시킨다. Pod overlay에 의존하지 않게 한다.
2. Serializer를 공용화하고 **uw3/us1/ut1까지** 재생성한다. 1e−12 mass와 원래 0.398776 target-drift fixture를 regression test로 고정한다.
3. Generalized scorer만 공식 current 경로로 노출하고, score loader에 schema-role validation을 추가한다.
4. Local development gate를 actual training orchestration에 연결한다. Candidate와 accepted policy를 구분하고 local-negative/NaN/Inf/missing-monitoring rejection을 검사한다.
5. Immutable pool hashes와 panel labels, root README를 수정한다. 새 SC estimator는 length units와 convergence status를 수정하기 전 사용을 보류한다.
6. 그 후 수정 텐서와 target으로 영향을 받은 실험을 재학습·평가한다. 완결된 LL/LR/RR 판정은 재사용할 수 있으나, 새 decoder/pool을 만든 uw3에 다른 response bank의 판정을 재사용해서는 안 된다. 공통 retained prompts가 달라지면 비교 baseline도 같은 조건으로 다시 정렬한다.

**이번 상태를 한 문장으로 요약하면: 주요 텐서 수정은 올바르게 검증됐지만, 수정의 전파와 공개 실행 코드의 완결성이 아직 부족하다. 핵심 수학을 다시 설계하기보다, 공개 entry point와 artifact 연결을 먼저 완성하는 것이 우선이다.**
