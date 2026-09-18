# NBPO 재검증: 97e03c0 → f0efbb5

## 결론

**Canonical API, LR/RR 연결, float64 target 직렬화는 정상 입력의 회귀 테스트를 계속 통과한다. 이번에 standalone prompt-wise local gate도 추가됐으며 실제 CLI에서 local value 계산·일부 승인/거절·filesystem 기록은 작동한다. 그러나 제공된 논문 Algorithm 1과 Appendix H의 승인 조건을 구현했다고 판정할 수는 없다.**

핵심은 세 가지다. (1) 새 gate의 기본값은 절반의 prompt만 양수여도 승인하며, 요청한 nMSE가 없어도 통과시킨다. (2) beta·target manifest·실제 checkpoint와 연결이 검증되지 않아 다른 game을 평가하거나 존재하지 않는 checkpoint를 승인할 수 있다. (3) 새 tensor guard는 정당한 동일 수치 행렬을 잘못 거부한다. 실제 campaign training 뒤에 gate가 호출되는 연결도 아직 보이지 않는다.

이것은 첨부 ZIP과 합성 회귀 테스트에 대한 판정이다. 실제 실험이 아래 잘못된 입력을 사용했다거나, 성능이 어느 방향으로 바뀔 것이라는 주장이 아니다. 원본 논문과 코드에는 손대지 않았다.

## 1. 검증 대상과 변경 범위

| 항목 | 확인값 |
|---|---|
| 이전 ZIP commit | `97e03c053e08892f72b68aec4203de071537ce5a` |
| 현재 ZIP commit | `f0efbb5935a4f522b8335292d6927161b274b796` |
| 현재 ZIP SHA256 | `52d8b51a4691986a6e299f838b149116a034446b690ea816c38c4c278b23eab6` |
| 논문 SHA256 | `c16f447ca05b757c7637fc9b1df8277c701870d4e1bf681f4eccfbc600cb0477` |
| Diff | 기존 24개 수정 / 22개 추가 / 삭제 없음 |
| 추가 파일 | 이전 audit 사본 20개 + `nbpo_local_gate.py`, `nbpo_local_gate_selftest.py` |
| source integrity | 원본 archive 파일 2,015개 모두 byte-identical; 생성된 추가 source/cache 파일 없음 |
| 테스트 환경 | Python 3.13.5; 실제 상세 버전은 audit_manifest.json |

Live GitHub commit 조회는 cache miss였다. 커밋 표기는 ZIP comment 및 파일 bytes로 확인한 것이며, remote ancestry나 서명을 검증한 것은 아니다. 이하 `S/`는 `analysis/sub_20260914/code_snapshot_20260917_union/`이다. [diff_inventory.json](diff_inventory.json), [modified_files.diff](modified_files.diff), [source_integrity.json](source_integrity.json).

## 2. 해소된 항목과 유지된 정상 동작

| 항목 | 이번 실제 검사 |
|---|---|
| Canonical API | 15개 current module import; generic writer/verified loader/builder 값 전달 유지 |
| LR/RR scorer | K=2/4에서 LL/LR/RR 및 alias의 기준 대비 오차 0; RR 한 order 누락 제외 |
| Serializer | base/v2/uw1/uw1c/uw3/us1/ut1의 7개 실제 함수가 1e-12 mass 보존; 28-pair validator 통과 |
| Target 연결 | 실제 scorer main → loader → local worker → canonical builder → JSON → validator; K=2/4에서 각 56개 row, 최대 target 오차 4.44e-16 |
| Local independence | 다른 prompt 변경으로 첫 prompt 결과가 바뀌지 않음 |
| Pool content SHA | 5개 panel에서 text 수정 시 digest 변경, stale cached SHA는 verifier가 거부 |
| Panel label | UW1/UW1C/UW3/US1/UT1 확인; generator anchor도 UW1로 변경 |
| Style-control | 양쪽 동일 counter, 한 order prompt 제외, fit convergence 상태 및 words-mode actual CLI 유지 |

따라서 이전 P0 canonical import 오류나 target 반올림 오류가 이번에도 남았다고 말하는 것은 맞지 않다. Tensor의 semantic alias 비교와 잘못된 bank-role metadata 거부도 추가되어 해당 malformed fixture는 통과하지 못한다. 다만 Section 5의 새 false-positive 조건은 별개의 회귀다.

## 3. 새 local gate: 성공한 부분과 논문과 다른 승인 규칙

### 3.1 실제로 수행하는 계산은 local이다

`S/nbpo_local_gate.py`는 cached sequence log-probabilities를 받아 다음과 같이 후보의 occurrence distribution을 재구성한다.

```
p_candidate(i) ∝ (1/8) * exp(log pi_candidate(y_i) - log pi_parent(y_i))
```

각 prompt를 X=1인 `AdaptiveGameRepresentation`에 따로 넣어 surplus를 계산하므로 이전 Global Nash evaluator처럼 prompt를 먼저 평균하지 않는다. 이 부분은 올바르게 추가됐다. [E01–E02](code_evidence.html#E01).

실제 gate CLI subprocess를 별도로 실행했고, fake model weights를 로딩하지 않았다. `--logprobs`가 제공하는 계산된 fixture 값을 사용했다. Candidate/parent 디렉터리는 fingerprint용 sentinel 파일을 가진 합성 디렉터리이다. 실제 LLM checkpoint로 검증한 것은 아니다.

| 입력 | 실제 결과 |
|---|---|
| 4개 prompt 모두 +0.206526, 유효 fit 값 | 승인, ACCEPTED.json 및 versioned symlink 생성 |
| 4개 모두 −0.206526 | 거절, exit 3, parent 반환, promotion 없음 |
| 한 candidate logp가 NaN 또는 +Inf | 거절, nonfinite prompt 집계 |
| nMSE=1.2, NaN, +Inf | 거절 |
| K=2의 별도 panel, 양의 local surplus | 승인 |
| 캐시에서 response logp 하나 누락 | process 비정상 종료; 승격 없음. 단 decision record는 없음 |

이 테스트는 standalone helper의 동작을 검증한 것이다. 학습을 끝낸 뒤 해당 helper를 호출하는 실제 campaign orchestration까지 검증한 것은 아니다.

### 3.2 기본 coverage=.5는 현재 논문의 all-prompt gate와 다르다

제공된 원고 Appendix H의 `Selection and reporting (P)`는 **고정된 retained monitoring subset의 모든 local surplus >1e-8, dev nMSE<=1**를 요구한다. 이는 원래 exact theorem의 자동 귀결이 아니라 원고가 따로 선언한 empirical execution contract다.

새 gate의 기본값과 실제 판단식은 다음과 같다.

```python
--coverage-min 0.5
positive = isfinite(mins) & (mins > 0)
accepted = finite_ok and coverage >= coverage_min and (fit_ok is not False)
```

Actual CLI에 local surplus `[+.206526,+.206526,-.206526,-.206526]`를 주면 **coverage=.5, accepted=true, promotion 발생**이다. `--coverage-min 1.0`을 주면 동일 fixture는 거부된다. 즉 local 계산으로 바뀌었어도 기본 acceptance는 논문과 다르다. [E01–E03](code_evidence.html#E01).

또한 coverage=1로 두어도 모든 surplus가 약 `5e-9`인 fixture는 승인된다. 원고 threshold `1e-8`보다 작기 때문이다. 이 tiny-margin 입력은 threshold 검사만을 위한 합성 사례다.

**수정:** paper-matching profile은 모든 retained local value가 명시한 `surplus_tol=1e-8`보다 커야 한다. 50% coverage variant를 실험하려면 별도 명시적인 relaxed contract로 취급해야 하며, 현재 원고의 gate와 동일하다고 해서는 안 된다. Monitoring subset selection과 해당 subset 내 전수 통과는 별개다.

### 3.3 요청한 nMSE가 누락돼도 승격된다

`--nmse-from`을 주되 `log_history=[]`인 파일을 넣으면 `fit_ok=None`이다. 그런데 `(fit_ok is not False)`는 True이므로 **checks.fit=null인데도 accepted=true**가 된다. 실제 filesystem 승격까지 재현됐다.

또한 nMSE의 조건이 upper bound 하나뿐이라 `-Infinity`를 넣어도 승인된다. NaN와 +Inf는 거부되지만 이것만으로 모든 invalid metric을 검증한 것은 아니다. [E03](code_evidence.html#E03).

`--nmse-from`을 아예 생략할 수 있다는 점은 help에 의도적으로 명시되어 있다. 따라서 '생략 기능이 존재함' 자체와 '요청한 검사 결과가 없는데도 통과함'을 구분한다. 하지만 생략 profile도 논문의 세 조건을 모두 검증한 실행으로 표시하면 안 된다. 제공된 selftest는 nMSE를 전달하지 않으므로 세 조건 전체의 검증도 아니다.

**수정:** paper mode에서는 metric 존재, finite, `0<=nMSE<=nmse_max`가 모두 필요하다. Cached logprobs와 canonical g가 이미 있으므로 선언된 pair distribution에 대해 nMSE를 직접 재계산할 수도 있다. 분모가 0이면 유효 metric을 자동으로 가정하지 말고 별도 상태를 정해야 한다.

## 4. Gate의 실행 경로와 artifact binding

### 4.1 Training 뒤 연결은 아직 별개다

`S/make_pros_train_jobs.py`, `S/panel_stage2.py`, `mnpo_scripts/run_mnpo.py`는 97e03c0와 byte-identical이다. 추적한 경로는 여전히

```
job -> torch.distributed.run -> run_mnpo -> trainer.train -> trainer.save_model
```

이며, current Python call-site 검색에서 새 gate 호출은 별도 selftest에서만 확인됐다. [E08–E10](code_evidence.html#E08), [gate_callsite_search.json](gate_callsite_search.json).

수동으로 새 helper를 실행하는 것은 가능하다. 그러나 그런 실제 명령·promotion logs 또는 stage controller가 제공되지 않았으므로 full Algorithm 1 integration이 완료됐다고 할 수는 없다. Repository 밖의 controller가 실제 있었는지도 이 ZIP으로는 알 수 없다.

README가 과거 run을 ungated projection으로 명시한 것은 적절한 구분이다. 이 서술을 독립적인 과거 실행 로그 검증으로 취급하지는 않았다. Gate를 추가해도 과거 수치에 소급 적용되지 않는다. [E07](code_evidence.html#E07).

### 4.2 다른 beta를 써도 대상 solve와의 불일치를 검출하지 않는다

Gate는 `dev_per_prompt.npz`에서 prompt IDs와 recorded min-surplus만 읽고, 관련 `solution.json`의 beta와 target hash는 확인하지 않는다. Beta는 독립적인 CLI 인자이며 default=.25다.

실제 scorer와 worker로 **beta=.05에서 certified된 target**을 만들었다. 같은 target과 같은 candidate logprobs에 대해:

| Gate beta | Local surplus | 판단 |
|---|---:|---|
| Default .25 | +0.0346720 | 승인 |
| Target과 일치하는 .05 | −0.1462211 | 거절 |

문제는 서로 다른 beta의 결과가 다르다는 사실이 아니다. **원래 solve와 같은 game을 검증한다는 gate가 불일치를 거부하지 않는다는 것**이다. 실제 production beta가 잘못됐다는 주장은 하지 않는다. [E02](code_evidence.html#E02), [E06](code_evidence.html#E06).

같은 이유로 NPZ의 g를 변경하고 solution manifest hash를 그대로 둬도 gate는 승인한다. `g`가 local value 계산 자체에 필요하지 않다는 점은 맞지만, fit condition과 해당 candidate의 target provenance를 검증했다는 주장은 성립하지 않는다.

### 4.3 Cached replay에서는 없는 checkpoint도 승인된다

`--logprobs`를 사용하는 상태에서 candidate 경로를 존재하지 않는 디렉터리로 바꾸었다. Actual CLI가 `accepted=true`로 종료하며 accepted record를 만들고, fingerprint는 SHA256(empty)인 `e3b0c442...`가 된다. 파일을 순회하지 못해 빈 입력을 해시한 것이다. [E01](code_evidence.html#E01), [E03](code_evidence.html#E03).

이것은 실제 체크포인트가 삭제됐다는 뜻이 아니라, **replay 입력과 승인된 실제 artifact를 연결하는 검증이 없다는 반례**다. Candidate와 parent의 실재, 지원 weight/config 파일, cache의 checkpoint/token-event hash, fixed monitoring IDs, target/score/solver config binding을 승격 전에 확인해야 한다.

승격 symlink 아래는 model files가 아니라 `ACCEPTED.json` metadata만 있다. 이 자체는 pointer design으로 가능하지만, 이후 stage/evaluator가 JSON의 checkpoint를 resolve해야 한다. 단순 `from_pretrained(accepted_symlink)`가 되는 checkpoint directory라고 취급해서는 안 된다.

### 4.4 Clean-checkout portability

Repo root와 current snapshot만 PYTHONPATH에 둔 `nbpo_local_gate.py --help`는 `diag_neural_pools` import에 실패한다. 필요한 helper는 ZIP 내 `analysis/uf4_20260910/code_snapshot_20260917/`에 있다. 이 실제 디렉터리를 PYTHONPATH에 추가하면 import와 위 CLI 테스트가 성공한다. Missing source symbol이나 missing transformers 문제가 아니다. [E01](code_evidence.html#E01), [E11](code_evidence.html#E11).

따라서 package-relative helper 또는 완전한 launch command가 필요하다. Gate는 score shard 수도 literal 4로 읽는다. Selftest는 private /work artifact에 의존하며, 실패 리스트를 출력한 경우의 명시적 nonzero exit도 없다. Portable CI fixture로 교체/보강해야 한다. [E12](code_evidence.html#E12).

## 5. 새 loader guard가 올바른 데이터를 거부하는 회귀

올바른 검사:

```
A_policy == A_LR
A_ref == A_RR
```

는 추가되어 malformed alias와 잘못된 bank-role fixture를 거부한다. 하지만 함께 추가된 다음 조건은 잘못됐다.

```python
if np.array_equal(arrays['A_policy'], arrays['A_LL']):
    raise ValueError('pre-fix wiring ...')
```

**Bank의 역할이 다르다는 것은 수치가 반드시 다르다는 뜻이 아니다.** 같은 reference/current policy에서 independent draws가 같은 답변이나 같은 preference ordering을 얻을 수 있다. 의도적으로 같은 pool을 쓰지 않더라도 숫자상 동일한 행렬이 나오는 것을 금지할 근거는 없다.

실제 generalized scorer로 역할을 올바르게 선언한 `A_LL=A_LR=A_RR`를 생성했다. 8개 action에 scalar preference levels를 주고 `A[i,j]=level[i]-level[j]`로 구성해, 모든 entry가 valid range이고 zero diagonal/skew symmetry를 만족한다. 정확한 core solver는 이 문제의 positive local improvement를 인증한다.

| 목적 수 | Core certified min surplus | Current loader |
|---|---:|---|
| 2 | +0.1404634984 | 거부 (`us1`, `ut1`) |
| 4 | +0.1694780633 | 거부 (`uw1`, `uw1c`, `uw3`) |

이것은 임의로 schema만 바꾼 파일이 아니라 actual scorer가 만든 올바른 파일을 loader가 거부하는 regression이다. 이런 경우가 실제 현재 실험 데이터에 얼마나 있는지는 확인하지 않았다. [E04](code_evidence.html#E04).

반대로 semantic arrays 중 A_LR/A_RR를 제거하면 `if all(... in arrays.files)`가 False가 되어 alias check 전체를 건너뛴다. Schema를 v2로 남기고 A_policy를 LL로 바꾼 fixture가 통과한다. 입력을 깨뜨린 테스트이며 production artifact가 이 상태라는 뜻은 아니다.

**수정:** v2 required arrays/metadata를 필수로 검사하고, 올바른 alias equality만 유지하라. `A_policy != A_LL` 조건은 삭제해야 한다. 잘못된 역할과 우연히 같은 값을 혼동하면 안 된다. 이 수정은 solver 수식이나 정상 output 값은 바꾸지 않는다.

## 6. 실제 테스트 결과

### 저장소 pytest

선정한 core/dual/targets/provenance/pipeline 전체 실행의 raw 결과:

```
121 passed, 5 failed
```

5개 실패는 각각 **transformers 미설치 2개, accelerate 미설치 3개**다. `pytest_summary.json`에 exact test IDs와 messages가 있다. Code defect로 집계하지 않는다. 제외하고 성공했다고 기록한 것이 아니라 실제 전체 결과와 원인을 함께 보고한다. Pipeline 8개는 legacy path이며 최신 local gate integration test로 세지 않는다.

### 이전 재현기 재실행

- 15 check: 14 PASS, 1 `REPRODUCED_MISMATCH` (global evaluator는 local evaluator가 아니라는 기존 정상적 구분).
- 58 check: 57 PASS, 1 OPEN (동일 global/local observation). 이전 alias 오류와 stale cached SHA 반례는 이제 통과한다.
- 7 serializer, 15 imports, 실제 tensor→target→JSON→validator chain, style-control words-mode CLI 유지.

### 새 f0efbb5 검사

- 32 check: **17 PASS, 14 OPEN, 1 OBSERVATION, 0 HARNESS_ERROR**.
- OPEN 14개는 14개의 독립적인 결함이 아니다. 같은 false-positive guard를 5개 loader에서 반복하는 경우 등이 포함된다.
- 정상/거절 gate tests는 실제 CLI subprocess와 filesystem을 사용했다. `--logprobs`를 사용해 real model forward는 대체된 입력이며, LLM 학습/생성을 검증했다고 주장하지 않는다.
- Paper의 full training-profile gate와 비교하는 경우, 단순 API 명세가 아니라 Appendix H에 명시된 all-prompt + nMSE 규칙을 기준으로 PASS/OPEN을 정했다.

`new_checks_results.json`, `regression_results.json`, `prior_regression_results.json`, `pytest_all.log`, fixture 하위 decision.json/command.json/stdout/stderr를 참조한다.

## 7. Claude Code용 다음 작업: 수식 변경보다 계약을 연결할 것

1. **Paper-mode predicate부터 고정:** 모든 retained local surplus >1e-8, finite nMSE in [0,1], missing check는 fail-closed. Relaxed/diagnostic modes는 명시적이고 non-promoting으로 분리한다.
2. **Artifact binding:** candidate/parent/target/score/LP/fit hashes와 beta를 한 run manifest로 묶고 mismatch를 거부한다. 존재하지 않는 checkpoint는 승격하지 않는다.
3. **Single stage controller:** 실제 trainer job 뒤에 gate를 호출하고 다음 stage/evaluator는 gate가 반환한 policy만 사용한다. 전역 control과 local method를 분리한다.
4. **Loader regression 제거:** 필수 semantic arrays 검사; numerical inequality 검사 삭제; 5개 generated module과 generation tests를 함께 수정한다.
5. **Portable regressions와 실제 smoke:** pod 전용 selftest 대신 작은 재현 테스트를 tests/에 넣고, 이후 실제 tokenizer/batch/한 optimizer-step 및 다음 checkpoint selection을 별도로 실행한다.

기존 ungated projection의 연구 결과를 폐기해야 한다는 판정은 아니다. 다만 gated algorithm의 결과로 소급 설명할 수 없으며, 이 code-only audit은 실제 표의 checkpoint/evaluator mapping을 재검증하지 않았다.

**최종 판단:** 이전의 numerical/canonical path는 유지됐다. 이번에는 local helper 구현이 추가됐으므로 진전이 있지만, README의 'all four closed'를 그대로 확인해 줄 수는 없다. 해결할 부분은 local gate의 승인 조건과 실제 실행 연결, 그리고 새로 과도하게 제한한 tensor guard다.
