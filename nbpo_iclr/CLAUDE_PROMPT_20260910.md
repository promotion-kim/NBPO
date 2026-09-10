# NBPO: 일반 instruction 8B 본 실험·4-objective 비교·조기 결과 동결

권장 실행 모델: 현재 Claude Code에서 사용할 수 있는 Opus. 원식과 코드 대조·실험 설계에는 지원되는 경우 thinking effort `xhigh`, 그렇지 않으면 `high`를 사용한다. 일반 구현·작업 관리는 `high`로 충분하다. 실제 지원 이름을 확인하며 존재하지 않는 모델 또는 effort 설정을 만들지 않는다. 아래 구분선 이후를 Claude Code에 전달한다.

---

당신은 NBPO의 코드, 실제 학습, 평가, 논문 갱신을 책임진다. 사용자 목표는 **일반 instruction 데이터로 학습한 실제 Llama-3.1-8B-Instruct의 general capability와 다목적 절충을 논문 본문에서 보여주는 것**이다. SafeRLHF의 neural projection 진단을 계속 늘리는 것을 주 작업으로 삼지 않는다. 지금부터 일반 instruction 4-objective 본 실험을 필수 작업으로 실행한다. 이미 승인된 범위의 다음 job마다 다시 착수 여부를 묻지 말고 실제 프로세스와 산출물을 만들며 진행하라.

내부 마감은 **2026-09-16 18:00 KST 학습·모델 선택 동결 → 9월 18일 18:00 KST 최종 metric 동결 → 9월 20일 원고·supplement 완성**이다. 9월 21–25일은 복구·검수·제출 buffer로 남긴다. 공식 일정은 abstract 9월 18일 23:59 AoE = **9월 19일 20:59 KST**, full paper 9월 25일 23:59 AoE = **9월 26일 20:59 KST**이며 실행 시 공식 페이지의 변경 여부만 확인한다. 더 늦은 이전 내부 일정보다 이 조기 동결 계획을 우선한다. 최고의 결과나 채택을 약속하지 말고 제한된 시간에 주장에 필요한 비교를 완결한다.

**비용 조건은 새로운 유료 API 호출 0건이다.** 공개 데이터에 이미 들어 있는 GPT-4 annotation은 사용할 수 있으나 합성 annotation임을 명시한다. 이것을 새 human feedback 또는 API 비용이 원래 없었던 데이터라고 설명하지 않는다. 로컬 GPU 추론·학습 비용은 따로 보고한다.

## 1. 현재 증거를 정확히 고정하고 주 실험을 전환한다

최근 SafeRLHF 결과는 기존 full-model 8B 정책의 학습 성공을 보여준다. SafeRLHF train2000/dev500/test1000, 자체250-update schedule, policy seeds42/43/44에서 다음이 보고됐다. 원본 JSON과 정확한 metric 정의를 먼저 확인한다. 보고의 ±가 SD/SE/CI 중 무엇인지는 생성 코드를 확인한 뒤 확정한다.

| Family | nMSE, 보고된 mean±spread | sign, 보고된 mean±spread | worst surplus, 보고된 mean±spread | both>0, 정의 확인 필요 |
|---|---:|---:|---:|---:|
| NBPO-Nash | .6845±.0048 | .7009±.0016 | +.1215±.0028 | .725 |
| L1-matched utilitarian | .6737±.0242 | .7040±.0058 | +.1220±.0029 | .730 |

IFEval은 Nash .760/.750/.734로 평균 .7480, sample SD 약 .0131이고, utilitarian .721/.771/.743으로 평균 .7450, sample SD 약 .0251이다. 각 seed의 원본 precision으로 재계산한다. GSM8K .862–.870, Alpaca proxy .496–.516, HarmBench .266–.275는 **6개 arm 전체 범위**다. Family별 평균이나 SD로 배분해 만들지 않는다. 동일 seed·동일1000 prompt에서 Nash−utilitarian helpful/safe 차이 CI는 모두0을 포함하고 부호도 일관되지 않는다. 현재 결과는 이 panel에서 차이가 해상되지 않았다는 것이며 동등성 증명은 아니다. 두 objective가 함께 개선되는 현상을 Nash 고유의 서명으로 쓰지 않는다.

Target TV 평균 .0008과 pair-target RMS 차이 .0034 / target RMS1.125 ≈ .3%는 서로 다른 양이다. TV를 RMS 비율이라고 표현하지 않는다. `both>0`의 분모·집계 단위·surplus 정의를 source에서 확인하기 전에는 individual-rationality 만족률이라는 열 이름을 붙이지 않는다. 평균 surplus 양수, prompt별 양수 비율, population constraint는 구분한다.

정정된 감사 수치도 그대로 보존한다. 전체1000에서 동일 거절 문자열247개, 짧은 응답450개, 거절 문두621개, 짧은 응답 중 거절448/450이다. Teacher truncation은 candidate occurrence26,790/84,000이며 prompt truncation1/84,000이다. 잘린 occurrence에서 중앙값232 tokens를 잃고 응답의 약61%만 관측한다. 이를 전체 응답이61%만 관측된다는 뜻으로 바꾸지 않는다.

SafeRLHF는 helpful/safe의 일치·충돌을 관찰하기 좋은 **보조 safety panel**로 남긴다. 현재 시작한 game-maxmin, fixed-reference Nash, BT-RM Nash의 CPU solve와 job은 보존하고 가능한 슬롯에서 완료한다. 이미 완료된 Nash/utilitarian3-seed 학습은 반복하지 않는다. 추가 SafeRLHF horizon sweep, WBC seed 확대, 더 큰 SafeRLHF-only panel은 일반 instruction 본 실험보다 낮은 우선순위다.

## 2. 첫30분: 상태 확인과 실제 작업 실행

서버의 현재 KST, GPU 종류·메모리·가용 GPU 수, 실행 중인 PID/job, 실제 code revision·dirty changes를 확인한다. 남아 있는 shell 수나 timer 존재를 GPU 작업의 증거로 쓰지 않는다. 타인 작업을 종료하거나 사용자 변경을 reset하지 않는다. 서버 최신90개 이후 commit을 이전 public checkout으로 덮어쓰지 말고 provenance만 대조한다.

현재 Safe mechanism solve의 실제 출력과 return code를 확인해 의존성 큐에 등록한다. 동시에 UltraFeedback metadata 읽기·필터·split 구축은 CPU에서 시작하고, GPU가 비어 있으면 teacher 학습 준비를 진행한다. `VLLM_WORKER_MULTIPROC_METHOD=spawn`은 모든 launcher와 child에 전달한다. Main guard와 classifier 경로를 포함한 공통 실행 entrypoint를 수정해 누락을 반복하지 않는다. 이미 끝난 평가를 환경 확인만을 위해 전부 재실행하지 않는다.

기존 `result_index.json`과 완료 JSON의 존재·hash를 실제 파일과 대조한다. Failure marker가 남았지만 이름을 바꾼 재실행이 성공한 경우 양쪽 이력과 연결을 남긴다. 결과가 있는데 controller marker만 없는 경우 측정 실패와 구분한다. 존재하지 않는 결과·명령·job을 만들거나 완료했다고 말하지 않는다.

이후 새 campaign manifest를 작성하고 **실제 첫 child PID, log 경로, GPU allocation, 첫 출력 artifact**를 보고하라. Manifest 작성만 하고 다음 사용자 메시지를 기다리지 않는다.

## 3. 필수 주 데이터: UltraFeedback의 일반 instruction 4-objective panel

새 panel 이름은 `UF-4`이며 instruction_following, truthfulness, honesty, helpfulness를 뜻한다. SafeRLHF safety label과 혼합하거나 honesty를 safety의 대체 이름으로 쓰지 않는다.

- Dataset: `openbmb/UltraFeedback`의 원본 released annotations를 revision pin해 읽는다. `ultrachat`, `sharegpt`, `evol_instruct` source만 사용한다. 데이터 카드상 이3개 source의 raw 합계는39,878 instructions이나, 다운로드한 revision의 실제 count와 필터 후 unique count를 다시 기록한다.
- `truthful_qa`, `false_qa`, `flan` 등 다른 source는 이 panel에서 제외한다. TruthfulQA와 일반 capability 평가 항목의 직접 포함 위험, source 역할의 혼합을 줄이기 위한 **결과를 보기 전 고정한 범위**다. 이 제외로 사전학습 오염이 사라졌다고 주장하지 않는다.
- Instruction source ID, 정규화 prompt exact hash, 사전 고정 near-duplicate 규칙으로 중복군을 만든다. 모든 split은 중복군 단위다. 여러 source에 같은 prompt가 있으면 단일 group으로 처리한다. Source별 수를 유지해 기록하고 source를 바꿔 Nash가 이기는 panel을 찾지 않는다.
- IFEval, GSM8K, Alpaca805, Arena hard/creative, HarmBench, XSTest, MMLU, ARC, HellaSwag 등 사용할 평가 prompt를 확보해 exact/near overlap을 **split 전에** 제거한다. Match rule, 제거 ID, 이유, source별 count를 저장한다. 검증이 불가능한 pretraining overlap은 별도 한계다.
- Label schema를 실제 row에서 확인한다. Released completion4개를 사용하되 criteria 이름, rating의 string/int/dict/list 표현을 견고하게 읽고 valid ordinal rating1..5만 인정한다. 누락·오류 rating을0으로 채우지 않는다. Group별 valid coverage를 남긴다.

필터·중복 제거 후 적격 prompt가35,000개 이상이면 아래 split을 고정한다. 배정 전에 해당 크기의 disjoint group 구성이 가능한지 확인한다.

| 역할 | prompt 수 | 사용 범위 |
|---|---:|---|
| Preference-model train | 20,000 | Released completion과4개 objective annotation |
| Preference-model dev | 2,000 | Teacher 선택·calibration, final policy 점수와 분리 |
| Policy train | 10,000 | Prompt만 사용; 공통8B로 새 candidate 생성 |
| Policy dev | 1,000 | Policy/recipe 선택과 실행 진단 |
| Final policy evaluation | 2,000 | 모델 선택 동결 후 전체 방법의 최종4-objective 평가 |

35,000개 미만이지만30,000개 이상이면 **policy train만5,000개로 줄인 고정 fallback**을 사용한다. 이는 성능 결과를 보기 전 적용하는 데이터 가용성 규칙이다. 30,000개도 안 되면 실제 수와 필터별 손실을 보고하고, source를 몰래 넓히거나 기존 test를 train에 넣지 않는다. 그룹 단위 배정 때문에 목표를 정확히 못 맞추면 실제 n과 누락을 기록한다. 가장 적은 objective-valid set을 기준으로 main split coverage를 계획하되 completion당 missing label은 mask로 처리한다.

Split seed20260910을 고정하고 source 비율을 가능한 한 층화한다. PM train/dev와 policy train/dev/final은 모두 서로 disjoint다. 기존 SafeRLHF train/test와 중복되는 prompt가 있으면 명시하고 train 쪽에서 제거한다. Human label이라고 부를 수 있는 것과 released GPT-4 label을 metadata에서 구분한다. Dataset ID/revision/hash/filter version/split assignment hash를 저장한다. Final 결과는 policy 선택과 teacher 개발에 사용하지 않는다.

## 4. 새4-objective teacher: truncation 문제를 구조적으로 줄인다

기존 SafeRLHF RoBERTa512 teacher의 `max_length` 숫자만 올리거나 새 영역에 그대로 재사용하지 않는다. 새 UF-4 panel에는 `answerdotai/ModernBERT-base`의 **native8192 context, 약149M parameters**를 사용한다. 공식 model card와 실제 installed implementation을 확인하고 revision을 고정한다. Transformers4.48 이상 지원이 필요한지 확인하며 이 encoder에 없는 `token_type_ids`를 생성·전달하지 않는다. 별도 train/eval dependency 환경을 재사용하되 공용 환경을 무작정 업그레이드하지 않는다.

한 joint encoder에4개의 preference head를 두어4개 criteria를 동시에 학습한다. Pair context에는 instruction, responseA, responseB를 구분하는 명시적 delimiter를 넣는다. GPM의 criterion k score는

`P_k(A > B | x) = sigmoid((f_k(x,A,B) - f_k(x,B,A)) / 2)`

로 구현해 swap complement를 보장한다. 같은 응답 비교는 .5다. Released completion4개에서 unordered6pairs를 만들며 criterion별 rating A>B면 target1, A<B면0, 동점이면.5다. Missing criterion은 masked loss로 제외한다. Overall score나 completion 평균으로4개 head label을 대체하지 않는다. Loss는 유효 objective별 수와 pair 수를 정확히 정규화하고 objective별 Brier/NLL/accuracy/tie coverage를 기록한다.

Matched BT teacher도 같은 ModernBERT backbone·released annotation·prompt split·training seed에서 학습한다. `r_k(x,y)`4개 scalar head와 `sigmoid(r_k(x,A)-r_k(x,B))`를 사용한다. GPM을 먼저 학습하고 그 예측을 BT 정답으로 삼는 방식은 기본 matched annotation 비교가 아니다. 파라미터 수·token/FLOP 차이와 실제 teacher 비용은 숨기지 않는다. Policy optimization3-seed에는 **동일한 frozen PM seed42**를 사용해 teacher seed와 policy seed의 효과를 구분한다. PM seed 반복은 현재 필수 범위가 아니다.

초기 PM recipe는 **새 제안값**이다: full encoder/head training, AdamW LR2e-5, betas(.9,.999), weight_decay.01, global batch32 unordered pairs, bf16, warmup ratio.06, gradient clipping1, 자체 cosine schedule. 길이 bucket과 gradient accumulation을 사용하고 hardware에 맞춰 microbatch를 줄이되 global batch를 유지한다. 우선1epoch를 실행하고 고정 dev 주기(예:100 optimizer updates와 epoch 끝)로 criterion별 성능을 측정한다. 사전 고정 예산에서 최대2epochs까지만 허용한다. Validation criterion은4objective 평균 NLL, 같은값이면 최악 objective NLL로 고정하고 final policy 결과를 본 뒤 checkpoint를 바꾸지 않는다. BT에는 동일한 제한과 선택 규칙을 적용한다.

8192 tokens는 actual full serialized input 기준이다. 각각 A/B 단독 길이만 세지 않는다. Instruction·A·B별 길이, 양 순서의 oversize 여부, 잘린 토큰/문자 수를 기록한다. Policy candidate를 prompt≤1024, response≤1024로 생성하면 대부분 native context 안에 들어가지만 tokenizer 차이까지 실측한다. Released PM pair가8192를 넘으면 사전 선언한 symmetric segment budget으로 instruction을 보존하고 A/B에 공평한 budget을 배정하되 truncated flag·coverage를 남긴다. 그 subset의 dev metric을 별도로 보고하고 final evaluation prompt를 길이 때문에 몰래 제외하지 않는다. Truncation이 여전히 크면 더 긴 native-context 대안으로 바꾸는 것은 별도 teacher recipe 변경으로 기록한다.

작은 검증은 의미 있는 위험에만 집중한다: A/B swap complement, identical text tie, missing-label masking, valid-label denominator, response boundary/EOS, actual input length, finite scores. 코드와 수식이 똑같이 틀린 자기 재현 테스트를 대량으로 만들지 않는다.

**이 annotation은 각 응답에 대한 scalar rating에서 유도되므로 source preference는 transitive하다.** GPM이 일반 preference를 표현할 수 있다는 것과 이 데이터가 인간의 cyclic preference를 입증한다는 것은 다르다. Nontransitive claim은 기존 controlled 실험에 맡기고 UF-4에서 허구의 cycle evidence를 만들지 않는다. Teacher 품질이 낮으면 모든 후속 방법에 공통된 측정 한계로 보고한다. Released completion과 새 base-generated 응답의 분포 차이를 확인하기 위해 policy-dev에서 사전 고정한200개 새 response pairs를 독립 Qwen judge로도 평가하고4objective agreement, tie, confidence, truncation을 보고한다. 이 label을 PM train에 합치지 않는다. 이를 무한한 새 neural gate로 만들지 말고 명백한 파싱·방향 오류를 고친 후 계획한 비교를 진행하되, teacher가 일반화하지 못하면 결론의 한계와 가능한 scope를 즉시 보고한다.

## 5. Candidate pool·게임 tensor·profile

Policy train/dev의 instruction만 공통 original `meta-llama/Llama-3.1-8B-Instruct`에서 생성한다. 성공한 Safe run과 같은 base revision/tokenizer/chat template/EOS를 우선하되 실제 hash를 확인한다. Released GPT-4 completions를 policy target response로 몰래 섞지 않는다. 모든 방법이 동일 generated pool, annotations, pair construction 범위를 공유한다.

기본은 prompt당 learner8 + comparator8이다. 기존 solver의 base/reference 정의와 정확히 맞는 독립 sampling stream을 사용한다. Raw policy sampling이면 temperature1, top_p1, top_k disabled, repetition penalty1을 명시한다. 기존 recipe가 다른 distribution을 사용했다면 sampling distribution과 계산하는 reference measure를 일치시키고 이탈을 기록한다. 같은 text가 중복되었다고 임의로 resampling해 분포를 바꾸지 말고 multiplicity·weight 처리 방식과 EOS를 유지한다. 생성 실패는 유효 응답처럼 세지 않는다.

GPM tensor는 모든 criterion을 joint head로 한 번에 얻는다. 8×8 cross comparisons와 comparator8개의 unordered28 self-bank comparisons를 각각 A/B 두 순서로 계산하면 policy train10k에서 `10,000 × 2 × (64+28) = 1,840,000` joint encoder forward passes다. 이를 objective4개라며 다시4배로 계산하지 않는다. Batching된 wall-clock, tokens, padding, memory가 실제 비용을 결정한다. BT scoring은 scalar per-response 결과를 cache하고 불필요하게 모든 pair마다 재인코딩하지 않는다.

전체 queue 전에 source/길이 층화200 prompts로 generation→teacher→solver→pair build의 actual throughput과 disk size를 측정한다. Profile 표본은 train 안에서 고정하고 새 dataset 선택을 위한 점수 탐색에 쓰지 않는다. Candidate generation, GPM joint encoder, BT encoder, solve, policy training, full evaluation의 비용을 분리한다. 1.84M 숫자는 train10k만의 estimate이며 dev/final, fallback, 재시도 비용은 별도다.

새4-objective panel의 β, η, disagreement/fallback d, normalization, feasibility constraint를 actual method와 대응시킨다. 성공한 solver config는 출발점일 뿐 새 dataset에서 계산된 λ·target을 이전 Safe panel에서 복사하지 않는다. Response sampling distribution과 `p_t`, comparator base measure를 일치시킨다. Positive surplus feasible set을 찾지 못하면 infeasibility를 기록하고 numerical tolerance·dual residual·oracle를 점검한다. 목적별 surplus를 임의로 epsilon clipping해 성공 target을 만들지 않는다. Scope를 바꾸어 이기는 panel을 찾는 대신 실패 이유를 보고한다.

## 6. 공통8B policy recipe와 필수 baseline

Policy는 모두 original Llama-3.1-8B-Instruct에서 시작하고 **full-parameter optimization**을 사용한다. Safe-MSE250 checkpoint를 warm start하면 다른 실험이므로 기본값으로 쓰지 않는다. Teacher는 frozen이다. 외부 방법의 필요한 outer/inner update는 원식을 보존하고 NBPO의1250updates를 무조건 복사하지 않는다. NBPO는 **한 outer stage**로 시작하며 solver target projection과 population-optimum 이론 보장을 구분한다.

공통 nuisance 출발값은 실제 성공 run의 resolved config다. 그것과 일치하는 경우 peak LR5e-7, AdamW betas(.9,.95), weight_decay0, max_grad_norm1, bf16, global batch32 pairs를 사용한다. 실제 optimizer가 다르면 몰래 AdamW로 바꾸지 말고 성공 recipe를 우선해 diff를 기록한다. Generation prompt cap1024, total policy context2048, response cap1024를 사용하되 long prompt retention과 truncation 비율을 확인한다. Prompt만 잘라 질문의 핵심을 잃으면 그 subset을 감사하고 main final prompt를 삭제하지 않는다.

Policy10k의 기본 horizon은1250updates, fallback5k는625updates다. 이는 global32pairs에서 기존2k×250updates와 같은 **prompt당 평균4pairs 노출량**을 유지하는 출발점이지 새 데이터에서 입증된 optimal horizon이 아니다. 각 run은 자기 horizon 전체에 warmup ratio.1과 cosine annealing을 적용한다. Prompt별 sample count·pair coverage를 기록하고 “전체6pairs 또는64pairs를 몇epoch 봤다”는 표현과 혼동하지 않는다. 조기 smoke failure가 없으면 본 비교를 시작하며 Safe horizon sweep을 선행 조건으로 삼지 않는다.

Primary family는 아래와 같다. 기존 Safe mechanism3개는 별도 supporting panel에서 계속한다.

| UF-4 비교군 | 기본 반복 | 목적 |
|---|---:|---|
| Base | frozen model1개 | 동일 생성·평가의 기준 |
| NBPO-MSE | seeds42/43/44 | 제안 방법의 실제4-objective8B 결과 |
| Game-utilitarian, L1 matched | seeds42/43/44 | 같은 utility·oracle·realization에서 aggregation 비교 |
| PROSPER, 실제 알고리즘의 공통 조건 adaptation | seeds42/43/44 | 가까운 외부 게임 기반 비교 |
| MOPO, 실제 알고리즘의 공통 조건 adaptation | seeds42/43/44 | 제약 기반 다목적 비교 |
| BT-scalarized DPO | 아래7weights 모두seed42 | 다양한 절충점과 specialist |

각 adaptation은 원논문 objective, constraints, update, extraction, oracle input, 비용을 `algorithm_fidelity.md`에 대응시킨다. `/work/v2/mopo_targets.py`가 있다는 이유만으로 검증 완료라고 하지 않는다. Proposition3.1, feasibility/normalization/weight 처리, batch 정의를 실제 구현과 비교한다. MOPO의 primary objective는 helpfulness, 나머지 instruction following/truthfulness/honesty는 고정 reference 기반 dev bounds로 둔다. Bounds 선택과 reward calibration을 train/dev에서 고정하고 실제 MOPO coupling을 보존한다. 단순 WBC를 MOPO라고, 기존 Game-maxmin을 PROSPER라고 부르지 않는다. 실제 adaptation이 불가능하면 실패·이탈을 보고하고 임의의 variant를 같은 이름으로 넣지 않는다.

Game-utilitarian은 새 UF-4 Nash dual λ의 L1 norm과 일치하는 equal coefficient를 사용한다. λ mass가 prompt별인지 global인지 solver 정의를 확인해 정확히 맞춘다. 과거 Safe λ(6.325,6.186)를4objective에 재사용하지 않는다. 이 panel에서도 Nash/utilitarian target이 비슷하면 그대로 보고한다. Unequal λ를 만드는 것이 dataset 채택 기준이 아니며 objective 단위가 λ에 영향을 준다는 점을 기억한다.

DPO scalarization weight 순서는 **(instruction_following, truthfulness, honesty, helpfulness)**다. 사전 고정7점은 `(1,0,0,0)`, `(0,1,0,0)`, `(0,0,1,0)`, `(0,0,0,1)`, `(.25,.25,.25,.25)`, `(.1,.1,.1,.7)`, `(.1,.7,.1,.1)`이다. Train/dev에서 고정한 BT reward calibration 후 `r_w=sum_k w_k r_k`로 같은 pool의 preference pairs를 만든다. Tie threshold, minimum margin, valid pair count를 기록한다. 이 방식은4개 objective DPO loss의 가중합과 같다고 주장하지 않는다.

7points는 결과와 무관하게 모두 보존한다. **본문의 compact DPO 행은 결과와 무관하게 uniform(.25,.25,.25,.25)으로 고정한다.** Uniform representative는 seeds43/44를 반복하고, 사전 고정 dev4objective 거리로 선택한 NBPO의 가장 가까운 DPO competitor도43/44를 반복한다. 이미 uniform이 nearest이면 중복학습하지 않는다. Repeat selection은 final 결과를 보기 전에 고정하며 선택 규칙·selection score를 보고한다. 다른5개 point가1seed라면 frontier 전체를3seed 평균이라고 쓰지 않는다.

Family별 dev tuning은 최대3trials, 초기에 profile상 필요하면 최대2로 줄여 사전 고정한다. Trial 후보, 총tokens/updates, selection metric, 실패 처리, 정해진 예산을 결과 전에 기록한다. 동일 wall-clock만을 공정성의 유일한 기준으로 삼지 말고 data·oracle access·trainable parameters·tokens·GPU-hours를 함께 보고한다. NBPO 과거 개발 비용과 새 baseline 비용은 구분한다. Test에서 좋은 seed·checkpoint를 골라 본문에 쓰지 않는다.

새 panel의 policy seed42/43/44는 optimizer와 data shuffle에 실제로 반영한다. 같은 seed의 비교군에는 가능한 공통 초기값과 prompt/pair 순서를 사용한다. 모든 run에서 완전히 같은 shuffle과 dropout0을 유지한 채 seed명만 바꾸지 않는다. PM seed42는 고정한다. Policy3seeds의 불확실성은 이 frozen teacher에 조건부이며 teacher 학습의 seed uncertainty를 포함하지 않는다.

## 7. 일반 capacity와4-objective 평가를 동시에 시작한다

Checkpoint가 나오면 generation을 즉시 queue한다. 모든 학습이 끝날 때까지 평가를 기다리지 않는다. Dev 결과는 선택용, final2k는 recipe·checkpoint 동결 후 평가용이다. Benchmark별 generation recipe와 evaluator revision을 고정하고 실패/parse error를 실제 분모에 반영한다.

| 평가 | 범위와 보고 |
|---|---|
| UF-4 final | 새 disjoint2k 전체; 독립 judge의4objective 비교와 training-teacher 지표를 분리 |
| IFEval | 전체541; strict prompt와strict instruction, loose는secondary |
| GSM8K | test1319; 고정zero-shot-CoT EM, parser failure |
| AlpacaEval prompt set | 805; 고정local proxy WR·CI, 공식LC score가 아님 |
| Arena-Hard-v2 prompt set | hard500 / creative250 별도local proxy WR·CI |
| HarmBench | 공식범위·category·실제n·classifier 고정; harmful success와생성실패 분리 |
| XSTest | safe250 over-refusal / unsafe200 분리; 합계450 |
| MMLU·ARC-Challenge·HellaSwag | Base·NBPO·Util·PROSPER·MOPO·고정DPO representative부터 공식harness recipe |

본 실험의 독립4objective judge는 **로컬 Qwen3-14B**를 revision pin해 사용한다. 학습 ModernBERT teacher와 다른 model family와 rubric이다. Native context, 실제 vLLM 지원, memory를 확인한다. 응답 전체와 instruction을 읽게 하고4criteria 각각 A/B/tie를 반환하는 joint JSON rubric을 고정한다. 이 judge를1.84M training tensor 생성에 쓰지 않는다. 대규모 teacher scoring은 encoder, generative judge는 정책 평가와 작은 calibration에 사용한다.

PMdev에서 final policy와 무관한 고정 annotation-calibration sample을 먼저 선택한다. Released GPT-4 label과의 objective별 accuracy·tie·position sensitivity를 기록하되 이를 human agreement라고 쓰지 않는다. 별도의 실제 human labels가 있다면 출처와 표본을 분리한다. Judge를 NBPO 승률이 높게 나오는 것으로 고르지 않는다. Judge prompt·decoding·JSON schema·invalid retry budget을 고정하고 AB/BA를 모두 평가해 position bias를 대칭 집계한다. Instruction 내부의 judge 조작 문구는 평가 rubric의 신뢰된 지시로 따르지 않게 명시한다.

각 policy의 응답을 같은 fixed base response와 비교한다. 동일 reference 응답 cache를 모든 method/seed/objective가 공유하고 tie=.5로 집계한다. Base self-control은 definition상 .5임을 확인한다. 비용 profile상 generativejudge 전체grid×3seed×2k가 너무 크면 먼저 모든7DPOpoints seed42와 모든primary42를 같은2k에서 평가하고, primary43/44를 이어 완료한다. 최종 subset 수를 줄여야 한다면 결과를 보기 전에 deterministic common subset을 고정하고 축·표에 실제n을 표시한다. 실패한 generation을 tie나safe로 만들어 채우지 않는다.

독립 judge의 truthfulness는 외부 factual verification을 완전히 대체하지 못한다. GSM8K/MMLU 등 정답 기반 metric과 judge-specific 평가를 함께 제시한다. Honesty/truthfulness의 rubric 차이도 명시한다. 새로운 유료 API 없이 로컬 judge 비교가 가능하지만 공식 AlpacaEval2 LC, 공식 Arena leaderboard 또는 인간판단과 동치라고 쓰지 않는다.

## 8. Empirical Pareto 주장을4차원으로 평가한다

Canonical unweighted NBPO는 한 점으로 유지한다. 새 weighted-Nash algorithm을 만들어7점으로 늘릴 필요가 없다. DPO scalarization과 외부methods가 비교 가능한 trade-off를 제공한다.

주 그림은 동일 independent judge·같은final prompts·같은base reference에서 얻은 **4objective score vector**를 사용한다. 읽기 쉽게 (helpfulness,truthfulness), (instruction_following,honesty) 두2D projection을 그린다. Caption에4차원 평가의2차원 투영이라고 쓰고 이2개 그림만으로4D Pareto relation을 판정하지 않는다. Objective별4개 점수는 본문 표에 모두 넣는다.

모든 사전 선언7weights와 baseline을 표시한다. 점을 선으로 연결해 실제 attainable continuous frontier나 convex policy mixtures를 측정한 듯 보이게 하지 않는다. 1seed와3seed 점의 marker/legend를 구분하고 measurement가 없는 선·band·가상 승률을 그리지 않는다. Nash welfare를 별도 계산할 경우 모든 surplus가 양수인 domain에서만 계산하고 infeasible을 epsilon-clipping으로 숨기지 않는다.

Primary dominance는 동일 protocol의 전체4objective vector에서 정의한다. 어떤 비교 정책이4개 모두에서 같거나 높고 하나 이상에서 엄격히 높으면 point-estimate domination으로 표시한다. 이는 **평가한 유한 정책 집합**에 대한 결과다. “전역 Pareto optimality 입증”이나 “신경망에 이론의보장이 자동승계”라고 쓰지 않는다. Noise 때문에 dominance가 바뀌는지 paired prompt bootstrap으로 보고하고 mean±seedSD와 prompt CI를 분리한다.

모든 method와 objective에 같은 prompt resample을 사용한다. Prompt 내AB/BA, response replicate, shared reference는 같은 cluster로 묶는다. 비선형 worst surplus/softmin은 bootstrap마다 다시 계산한다. Statistical domination을 주장하려면 사전 고정 method contrasts에 simultaneous one-sided bounds 또는 명시된 multiplicity correction을 사용한다. CI에0포함은 동등성 증명이 아니고 .5포함은 parity 증명이 아니다. 3seeds만으로 optimizer population variance를 정밀추정했다고 하지 않는다.

Independent judge WR과 training-teacher game surplus는 척도가 다르므로 같은 축/열에서 혼합하지 않는다. Training-teacher 좌표는 목적함수 구현/정합성 지표이며 main독립 평가를 대신하지 않는다. 기존 controlled nontransitivity 결과는 별도 이론 보완이고 UF-4의 rating-induced label에서 발견한 인간cyclicity라고 포장하지 않는다.

시간이 남으면 사전 고정300final prompts에서 Base/NBPO/PROSPER/DPO representative의6unordered policy pairs를4objectives로AB/BA 비교하는 matrix를 부록에 넣는다. 이것은 fixed-base WR로 숨길 수 있는 관계를 보완한다. 실제 유의한cycle이 없으면 cycle을 주장하지 않는다. Stochastic policy8samples 평가도 보완 작업이며 main8B comparison의 선행gate로 두지 않는다.

## 9. 실험 일정·계산 예산·자동 실행

| KST 내부 목표 | 필수 완료 |
|---|---|
| 9/10 오늘 | UF source/filter/splits 동결, PM 실제 학습 시작, 기존 Safe CPU controls 진행 |
| 9/11 18:00 | GPM/BT dev audit·freeze, 200-prompt end-to-end profile, 공통 pool/tensor queue |
| 9/12 | UF-4 최초 NBPO/Util/외부 baseline checkpoint와 dev evaluation |
| 9/13–16 18:00 | 주4families3seeds, DPO7weights, 선택된 추가 seed, 모델 선택 동결 |
| 9/17–18 18:00 | 고정 final·capability·Pareto 평가 완료, metric freeze |
| 9/19–20 | 결과 표·그림·본문·supplement 완성, abstract 제출 확인 |
| 9/21–25 | 누락 복구·재현·익명성·PDF 검수 및 조기 제출 buffer |

4GPUs×8days×24hours×60% 실제 가동률이면460.8GPU-hours다. 본 scope의 기획 예산 약250–450GPU-hours는 **실측 견적이 아니다**. 200-prompt profile과 최초 training speed로 generation·teacher·policy·judge·disk 예산을 갱신한다. 이전2k 단일 arm2.2GPU-hours를 모든 외부 method의10k 비용으로 그대로 주장하지 않는다. GPU-hours와4GPU 동시 사용 wall-time을 혼동하지 않는다.

예산이 넘으면 DPO 추가 repeat, UF mechanism 추가 반복, Safe 확장, MT-Bench, 모든 mechanism에 MMLU 평가, stochastic 확장을 순서대로 줄인다. 주4families, 사전 선언 DPO7points, 공통 독립4objective 평가, 기본 capability를 우선 보존한다. 결과를 보고 불리한 point를 없애지 않는다. 핵심 teacher/data 경로가9/11 18시까지 실패하면 실패 원인과 실제 가능한 scope를 즉시 보고한다. 일반 instruction 결과가 없는데 Safe 진단을 새 general 실험으로 이름만 바꾸거나 동결일을 몰래 늦추지 않는다.

작업 관리는 실제 서버에서 지속 실행되는 Python/shell queue 또는 기존 controller를 사용한다. 각 job에는 `job_id`, dependencies, command/config hash, GPU requirements, timeout, log, PID, return code, artifact checks, state(PENDING/READY/RUNNING/DONE/FAILED)를 둔다. `nohup`/기존 서비스 등 현재 환경에 맞는 방식으로 대화가 끝나도 자식 작업이 이어지게 한다. Hash pin 실패를 우회하기 위해 assert를 삭제하지 말고 올바른 artifact와 manifest를 대조해 복구한다.

Controller는 DONE의 artifact와 return code를 검증하고 READY job을 즉시 schedule한다. 다른 사용자 작업과 메모리 요구를 존중하고 중복 fullFT를 한 GPU에 무리하게 배치하지 않는다. READY가 있는데 GPU가 놀면 실패 상태와 dependency를 해결하고 그 이유를 보고한다. `sleep` timer, conversation 복귀, 살아 있는 shell은 학습 실행이 아니다. 부모 프로세스가 끝났는데 child GPU 작업은 없으면 RUNNING으로 표시하지 않는다. 기존 약6시간 idle을 반복하지 않는다.

30분마다 활성 작업 현황은 실제 처리량·PID·GPU 활용·완료 artifact·다음 ETA로 짧게 기록한다. 대화가 없어도 파일에 누적한다. 측정한 ETA가 내부 동결일을 넘으면 그날 즉시 범위를 줄인다. 이미 승인된 후속 학습을 “착수할까요?”로 막지 않는다. 외부 메시지 발송·유료 API·무관한 작업 삭제는 이 승인 범위에 없다.

## 10. 원고·표·그림: 일반8B 결과가 본문 주역

새 `NBPO_v6_main_experiments_20260910` 패키지를 최신 서버 원고와 diff한 뒤 반영한다. 최신90개 이후 변경을 옛 버전으로 reset하지 않는다. 핵심 method/theorem은 최소 수정하고 experiments의 순서와 증거 범위를 수정한다.

본문 구성은 (1) UF-4의4objective8B main 결과 표, (2) 독립4objective score의2개2D projection 그림, (3) general capability/로컬 chat/safety 표, (4) 간결한 Safe2objective 보완 결과다. Controlled 실험은 방법의 비추이성 동기를 뒷받침하는 compact 결과 또는 부록으로 유지한다. 과거 neural gate,2×2horizon,full trajectory,pool-drift는 부록의 재현·진단 근거다. 실패나 null을 삭제해 성공한 이야기만 남기지 않는다.

`data/MAIN_EXPERIMENTS_SCHEMA.md`를 먼저 읽고 아래 입력에 source artifact importer를 연결한다.

| CSV 입력 | 역할 |
|---|---|
| `data/uf_objectives.csv` | 독립 judge4objective 본문 표 |
| `data/new_capability_results.csv` | 새 일반 capability·chat·safety 표 |
| `data/safe_three_seed_results.csv` | 기존 Safe3seed 보완 표 |
| `data/aggregation_policy_pairs.csv` | Safe paired Nash−utilitarian 차이와 CI |
| `data/ifeval_per_seed.csv` | 기존 seed별 IFEval과 재계산 근거 |
| `data/refusal_truncation_audit.csv` | 정정된 거절·절단 분모와 conditional 수치 |
| `data/uf_tradeoffs.csv` | 모든7DPO points와 외부 methods의4objective 실제 관측 |

최소 metadata는 method/family/dataset/objective/variant/weight/seed/base hash/split hash/teacher hash/judge hash/decoding/n/value/uncertainty/source path/status다. Four-objective 열 순서를 모든 CSV와 그림에서 일치시킨다. Per-seed 원본에서 mean과 sample SD를 계산하고 pooled range만 있는 현재 Safe 값을 임의의 family mean으로 채우지 않는다. Missing은 blank/pending/`—`이며0이나 예상값을 넣지 않는다. Historical1seed 결과에3seed label만 붙이지 않는다.

본문 표의 DPO는 **고정 uniform weights**다. 좋은 결과를 낸 weight를 metric마다 달리 선택하지 않는다. 나머지7point 전체와 mechanism별 full metric은 appendix/CSV에 보존한다. 본문 objective 표에는 독립 criterion WR4개·실제 seed 수·평가 n을 넣고 training teacher의 game value를 같은 열로 합치지 않는다. 그림은 source CSV에 실제 값이 있을 때만 점/CI를 그린다. Empty template은 draft라고 명시되어 있으며 최종판에 실측 frontier처럼 넣지 않는다.

패키지 폴더에서 실제 제공된 entrypoint를 실행한다.

```bash
python3 scripts/render_tables.py
python3 scripts/plot_uf_tradeoffs.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main_v6.tex
```

`render_tables.py`는 `generated/uf_objectives_main.tex`, `new_capability_main.tex`, `safe_three_seed.tex`, `aggregation_policy_pairs.tex`를 포함한 표를 생성한다. `plot_uf_tradeoffs.py`는 `figures/uf_tradeoffs.pdf`를 생성한다. 이 명령과 schema는 패키지 기준이며 실제 서버 adaptation 후 경로를 확인한다. Source JSON→CSV→TeX/figure를 유지하고 수치를 본문에 수기로 복사하지 않는다.

Root `.gitignore`의 `data/`가 논문 CSV를 숨기는 문제는 targeted exception 규칙으로 고친다. 단순 force-add에만 의존하지 말고 `git check-ignore`와 tracked files를 확인해 새 CSV도 추적 가능하게 한다. Training dataset나 model cache를 통째로 git에 넣지 말고 필요한 `nbpo_iclr/data/` 출력만 범위를 제한한다. 다른 사용자 변경은 stage하지 않는다. Tracked source만으로 clean export한 paper가 CSV→generated TeX→PDF로 빌드되는지 한 번 확인한다.

최종 산출물은 manifest/index, per-seed JSON/CSV, 표·그림, fidelity/data/teacher audit, queue/report, TeX diff다. Undefined/citation/overfull, table font, figure label, 본문 page 수를 검수한다. 컴파일 성공만으로 과학적 검증이 완료됐다고 하지 않는다.

첫 보고에는 UF 데이터 처리/PM job와 Safe controls의 실제 상태를 보내고, 첫 general8B checkpoint와 paired 평가를 우선 완료하라. 불리한 결과도 보존한다.

## 참고할 1차 자료

- UltraFeedback 원본data card·schema: https://huggingface.co/datasets/openbmb/UltraFeedback
- ModernBERT-base nativecontext·사용법: https://huggingface.co/answerdotai/ModernBERT-base
- MOPO: https://arxiv.org/html/2505.10892v2
- PROSPER: https://arxiv.org/html/2602.19041v1
- SPPO: https://arxiv.org/html/2405.00675v3
- Nash Learning from Human Feedback: https://arxiv.org/pdf/2312.00886
- Claude Code model/effort: https://code.claude.com/docs/en/model-config
- ICLR 2027 author guidance: https://iclr.cc/Conferences/2027/AuthorGuidelines
