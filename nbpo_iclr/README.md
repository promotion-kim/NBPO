# NBPO submission revision — 2026-09-14

## 파일

- `main_v6.tex` / `main_v6.pdf`: GPT 리뷰 34개와 갱신된 답변을 표시하는 검토본. 총 53쪽.
- `main_clean.tex` / `main_clean.pdf`: 리뷰를 숨기는 wrapper와 PDF. 본문 9쪽, 참고문헌·부록 등을 포함해 총 44쪽. 새 결과가 미측정인 연구 초안이며 최종 제출 확정을 뜻하지 않는다.
- `NBPO_experiment_plan.md`: 최신 결과 해석, 문헌 근거, 데이터 우선순위, 실험 예산과 제출 일정.
- `NBPO_Claude_submission_prompt.md`: 기존 H200 4장 환경에서 실험을 이어가는 Claude Code용 프롬프트.
- `templates/experiment_templates.tex`: 새 screening / controlled 2×2 / policy comparison 표 3개와 그림 1개.
- `templates/results.json`: 81개 지정 결과 슬롯. 현재 모두 null. 새 실험 결과를 발명하지 않았다.
- `render_results.py`: 원본 artifact 경로와 hash가 있는 numeric record만 `templates/results_auto.tex`로 렌더한다.
- `protected_manifest.json` / `validate_protected.py`: 기존 본문·리뷰·수치·이론 및 template 구조의 변경 감지.
- `main_v6_changes.diff`, `review_update_notes.md`, `validation.json`: 편집 내역과 검증.
- `figures/`: 새로 그린 기존 Table 1/target-transfer 그림, 재생성 코드·입력 수치, 원본 그림 snapshot, controlled figure.
- `provenance/`: 편집 기준 원고와 추가 데이터 조사 근거. 배포용 최종 supplement에서는 내부 검토 자료의 포함 여부를 저자가 별도로 결정한다.

## 이번 수정 범위

원본 GPT 리뷰 34개, 기존 36개 tabular 본문, 모든 기존 label을 보존했다. Background부터 Method까지의 수학 본문 및 Proofs는 리뷰 답변 외에 변경하지 않았다. Bibliography/style/math 파일도 원본과 같다. Experiment의 중복 설명을 줄이고 상세 capability·target realization·supporting 결과와 UF projection figure를 부록으로 옮겨 본문 9쪽을 맞췄다. Abstract의 장황한 진단 부분과 conclusion의 과거 'planned' 표현을 정리했다.

`.4613` common-offset 주장, DPO의 fresh 전 objective 우위 주장, 다른 N 사이의 causal stage comparison, candidate 7회 등장에 대한 자동 gradient 배율 해석을 수정했다. 이미 완료된 target diagnostic과 pilot을 리뷰 답변에 반영했지만, 자연 데이터의 순환성 또는 NBPO 우위를 새로 입증했다고 쓰지 않았다.

Figure의 원본 plotting files/raw CI arrays가 첨부되지 않아 Table 1과 target-transfer 그림은 제공된 표의 수치에서 다시 그렸다. 첫 그림은 보고된 seed SD를 사용한다. Target-transfer 그림은 means만 표시하며 원본 CI나 paired differences를 추정하지 않았다. 원래 그림은 snapshot으로 보존했다. 기존 수치의 raw execution audit를 수행했다는 의미는 아니다.

## 빌드

표준 TeX Live(algorithm/algorithmic 포함) 또는 Overleaf에서 패키지 root를 연다.

```bash
python3 validate_protected.py
python3 render_results.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main_v6.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_clean.tex
python3 validate_protected.py
```

빌드 중단으로 `.aux` 또는 `.out`이 부분 기록되면 해당 생성 파일을 백업 이동하고 다시 빌드한다. 소스나 보호 manifest를 변경해 빌드 문제를 숨기지 않는다.

## 결과 입력 계약

`templates/result_keys.json`의 key는 유지한다. 미측정은 null이다. 측정 record에는 `value`(유한 숫자), `artifact`(실제 집계/원본 파일 경로), `sha256`(해당 파일의 실제 SHA-256)이 필요하다. `ci95`와 seed/denominator metadata도 보존할 수 있다. `contract_id`를 동결한 campaign ID로 설정한다. Renderer는 값을 표시하고 source 파일 자체의 진위를 보증하지 않으므로 raw-to-table 집계 단계에서 hash·단위·N을 확인해야 한다.

새 결과로 기존 주장이나 표를 수정해야 한다면 Claude는 원고를 자동 수정하지 않고 `proposed_manuscript_patch.diff`를 만든다. 제목·초록·핵심 결과의 최종 해석과 제출은 저자가 검토한다. 제출 마감은 2026-09-26 20:59 KST이며 abstract는 2026-09-19 20:59 KST다.

검증 결과: 두 PDF 빌드 성공, undefined citation/reference 및 overfull box 없음. Clean PDF 본문 마지막 페이지 9, 전 페이지 축소 렌더와 주요 표·그림·리뷰를 시각 확인했다. 새 ML 실험 실행 0건.
