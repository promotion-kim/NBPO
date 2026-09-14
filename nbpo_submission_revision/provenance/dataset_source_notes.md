> Supplemental source audit. The authoritative execution size is the 200-prompt, N=8 plan in NBPO_experiment_plan.md; any 500-prompt, N=4 suggestion below is an alternative, not an additional required run.

# NBPO dataset 조사 (2026-09-14)

## 결론
새 main LLM 실험 후보는 PKU-SafeRLHF가 가장 실용적이다. 동일 응답쌍에 도움이 되는 쪽(better_response_id)과 안전한 쪽(safer_response_id)을 독립적으로 제공하므로 objective 충돌을 원본 human preference에서 직접 측정할 수 있다. 다만 해당 dataset이 단일 objective의 population cyclicity를 풍부하게 가진다는 근거는 확인되지 않았다. 새로운 base-policy 후보에서 충돌이 유지되는지 별도로 평가해야 한다. 기존 UF-4는 negative control로 보존한다.

## 1. PKU-SafeRLHF: 우선순위 1
공식 카드: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
ACL 2025 논문: https://aclanthology.org/2025.acl-long.1544/
카드의 서술 규모는 dual-preference 83.4K entries, 현재 viewer default는 약 82.1K rows로 표시되어 있으므로 실제 revision/subset/split 수를 freeze해야 한다. 각 entry에는 prompt, 두 응답, better_response_id, safer_response_id, 안전성 메타라벨/19유형/심각도/응답 SHA256이 있다. Helpfulness는 safety와 분리하여 질문에 답하는 품질을 판단하도록 정의했다. 원 논문 전체의 166.8K preference는 dual과 single을 합친 규모라 166.8K 독립 dual-pair라 부르면 안 된다.

즉시 할 감사: prompt 기준 dedup/split 후 C=mean[better_response_id != safer_response_id]를 전체 및 safe-safe/safe-unsafe/unsafe-unsafe에서 따로 보고한다. Forced binary harmlessness label의 사실상 tie를 충돌로 과대계상하지 않도록 safe-safe 및 severity동일 pair를 분리한다. 공개 label의 충돌률을 신뢰도 확정된 population conflict라 부르지 않는다. response SHA256를 같은 prompt 내 연결해 완전한 triangle 수를 먼저 센다. 두응답 entry 자체만으로 cycle은 계산할 수 없다.

제안 규모: 원본 train의 outcome-independent 500 prompt를 audit panel로 먼저 고정; fresh Llama3.1-8B 후보 4개, helpfulness/harmlessness 두 objective, 모든 6 unordered pair를 양방향 평가하면 500*6*2*2=12,000 verdict다. 동일 rubric 내 order-balance와 일부 반복검사로 단발 judge noise를 구분한다. 통과 후 5K train/500 dev/1K test prompt의 작은 학습 실험. 원본 human label은 teacher 감독/검증에 재사용할 수 있지만, 새 후보 쌍의 label은 다시 생성해야 한다. 기존 base와 다른 모델의 응답에 대한 원본 label을 새 응답에 붙여서는 안 된다. Training과 evaluation judge는 분리한다. 최소 strict IR feasibility와 exact target separation을 확인한 후 LLM full fine-tune에 투자한다.

## 2. HelpSteer2/3: 품질 데이터이지만 이번 대체 main에는 차선
HelpSteer2 공식 카드: https://huggingface.co/datasets/nvidia/HelpSteer2
21,362 response rows, 각 prompt에 2개 응답, helpfulness/correctness/coherence/complexity/verbosity 5개 human attribute. 20,324 train rows와 1,038 validation rows; preference와 disagreements subset도 공개한다. 동일 scalar attribute의 점수차로 만든 pair labels는 strict majority cycle을 만들 수 없다. 두응답만으로 triangle 측정도 불가하다. Complexity/verbosity는 양이 높다는 의미이므로 무조건 최대화할 desirable utility로 해석하지 말고 task별 목표를 정의해야 한다. 기존 UF와 비슷한 correctness/helpfulness 상관으로 또 약한 충돌에 머무를 위험이 있다.
HelpSteer3 공식 카드: https://huggingface.co/datasets/nvidia/HelpSteer3
40,476 preference examples, 2응답, overall preference와 최대 3 annotator별 preference/reasoning. 다국어/STEM/code 확장은 유용하지만 native K-objective 동일 pair labels 또는 완전 triad dataset이 아니다. 마감 직전 이 데이터로 갈아탄다고 Nash-specific trade-off가 생긴다고 기대할 근거는 없다.

## 3. PersonalLLM/PRISM: 후속 분석용
PersonalLLM 공식: https://huggingface.co/datasets/namkoong-lab/PersonalLLM
논문: https://arxiv.org/html/2409.20296v2
10,402 prompts, 각 8응답과 10 RM scores; 9,402 train/1,000 test. 이미 있는 8후보/점수를 이용한 후보 분포 실험은 싸다. 하지만 개인 preference가 RM scalar mixture로 생성되는 simulator라 각 개인/고정 objective는 transitive하다. 이질적 사람집단의 pairwise vote를 평균하면 cycle이 생길 수 있지만 이것은 미리 정의한 synthetic population oracle이며 natural human cyclicity의 증거가 아니다. Response는 여러 외부 LLM에서 생성돼 현재 learner-policy sampled pool도 아니다. 소스에 AlpacaEval/MTBench/XSTest/RewardBench가 있어 evaluation contamination 제외 필수.
PRISM 공식: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
논문: https://arxiv.org/abs/2404.16019
1,500사람,8,011대화,21LLM. 첫turn4모델응답을1~100scalar rating하고 이후선택모델의A/B응답을평가. 사람별가치차이는 유용하나 같은prompt/응답집합에 여러사람의 complete pairwise graph와 고정objective별라벨이 있다고 볼 수 없다. 자연적 diversity의 증거가 곧 within-objective cycles의 증거는 아니다. 신규데이터매핑/개인화설계가필요해이번마감main에비추천.

## GPM cyclic dataset 인용 주의
원문 Appendix E: https://arxiv.org/html/2410.02197v3#A5
직접 페이지: https://arxiv.org/html/2410.02197v3
CyclicPreference는 UltraFeedback에서 216~363개 규모 subset 4개를 구성하고, A>B는 honesty, B>C는 helpfulness 등 edge마다 objective를 달리하여 cycle을 유도한다. 이는 controlled constructed preference이며 동일 고정objective 아래 자연적으로 관측된 human majority cycle의 증거가 아니다. NBPO의 within-objective P_k cyclicity를 입증하려면 동일 k·동일 judge population·동일3응답에 대해 세 edge를 모두 평가해야 한다.

## 실험 의사결정
충돌/cycle 데이터가 NBPO의 상대적 이점을 드러낼 가능성은 있으나 성능 보장이 아니다. 너무 강한충돌은 strict positive surplus feasible policy를 없앨수있다. Cycle수보다 coverage/신뢰도/triangle당비율/BT미스스펙과heldoutNLL차이/정확타깃들의TV·독립gamevalue차이가중요하다. Fixed-reference Nash로adaptive게임효과, game-utilitarian으로Nashaggregation효과, scalarBT-Nash로표현효과를분리한다. SafeRLHF mainpilot과 UFnegativecontrol만 먼저 하고 PersonalLLM/PRISM을 동시에추가하지않는것이마감상현실적이다.
