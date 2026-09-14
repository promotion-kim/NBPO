# 제출 캠페인 startup 인계 — 2026-09-14

작성 13:55 KST. 공식 마감은 abstract 2026-09-19 20:59 KST, full paper 2026-09-26 20:59 KST
(현재 UTC 04:27 / KST 13:27에 확인). 이전의 `2026-09-14 09:00 KST` 진단 마감은 지났고 이
캠페인의 마감으로 쓰지 않는다.

## 1. 원고와 패키지의 대조 결과

| 항목 | 확인 |
|---|---|
| 전달된 패키지 | `nbpo_submission_revision/` (13:29에 압축 해제) |
| 패키지 `main_v6.tex` | 저자 검토본, `\input{templates/*}` 형태, 53쪽 |
| repo `nbpo_iclr/main_v6.tex` (13:23) | **같은 저자 개정본**이지만 두 template 파일이 inline으로 펼쳐진 사본 |
| 두 파일의 diff | hunk 2개뿐이고 둘 다 `\input` ↔ inline 치환 자리 |
| 패키지가 편집 기준으로 삼은 원고 | `provenance/source_main_v6.tex`, repo의 12:17 빌드에서 유래 |
| 내가 오늘 넣은 측정치의 보존 | `\tabonedpoovernbpo`, `\figtradeoffpendingcount`, `\taboneprosperseedphrase` 모두 패키지 안에 존재 |

즉 **더 늦은 측정이 유실된 곳은 없다.** 11:21의 caption 수정까지 저자 개정본에 반영되어 있다.

조치: repo의 inline 사본을 커밋(`343a502`)으로 보존한 뒤, 패키지를 `nbpo_iclr/`에 그대로
설치했다(`408f6df`). 이제 `nbpo_iclr/main_v6.tex`는 보호 대상 12개 파일 중 하나와 byte 단위로
같고 `python3 validate_protected.py`가 통과한다.

빌드 확인: `main_v6.pdf` 53쪽, `main_clean.pdf` 44쪽, 본문 마지막 페이지 9(10쪽이 AI Use
Statement로 시작), error 0, undefined reference/citation 0. `latexmk`는 이 호스트에서 perl
`Time::HiRes`가 없어 실행되지 않으므로 `pdflatex → bibtex → pdflatex ×2`로 빌드한다.

## 2. 보호 계약 때문에 바뀐 운영 방식

`validate_protected.py`는 `main_v6.tex`를 포함한 12개 파일의 **전체 파일 sha256**을 본다.
따라서 원고 본문에 한 글자라도 쓰면 검증이 깨진다. 결과는 오직

```
templates/results.json → render_results.py → templates/results_auto.tex
```

경로로만 들어간다. 이를 위해 `progress/fill_results_json.py`를 새로 만들었다. 이 스크립트는
81개 선언 key 외의 key, 비유한 값, 빈 artifact 경로, 64자가 아닌 sha256, 자기 CI 밖의 값을
모두 거부하고, 이미 측정된 cell을 덮어쓰려면 `--replace`와 사유를 요구한다.

**중단한 것**: 기존 UF reporter(`progress/reporter.py`)는 30분마다 원고의 AUTO 구역을 직접
고쳐 쓴다. 보호 계약 아래에서는 첫 tick에 검증이 깨지므로 프로세스를 종료하고, 파일 맨 앞에
`protected_manifest.json`이 있으면 시작을 거부하는 guard를 넣었다.

**그 결과 생긴 제약**: 오늘 14:10에 들어오는 PROSPER 3번째 seed처럼 **기존 표의 수치를 갱신하는
일은 이제 자동 반영 대상이 아니다.** 근거와 최소 diff를 `proposed_manuscript_patch.diff`에 적어
저자 검토로 넘긴다. 새 실험 결과만 위 results 경로로 들어간다.

## 3. 자원 재배치

| 작업 | 이전 | 지금 | 사유 |
|---|---|---|---|
| PROSPER seed 44 학습 | RUNNING | 13:38 완료(2:47:23), 판정 진행 | 유효 checkpoint까지 계속한다는 규칙대로 끝냈다 |
| s44 final-eval + capability | p62/63/67 | 진행 중 | 선언된 UF exhibit을 닫는 짧은 작업 |
| UF DPO 가중치 4개 arm | p73–76 | **p88–91로 강등** | 4장을 10시간 점유한다. 선언된 작업이므로 큐에서 지우지 않고 잉여 용량에서 돌게 했다 |
| 신규 캠페인 audit/판정 | 없음 | p64–68 | 제출 캠페인이 primary |

컨트롤러는 기존 pid 415591 하나만 쓰고 중복 프로세스를 띄우지 않았다. 다른 사용자 job은 건드리지
않았다.

## 4. judge 역할 고정

| 역할 | 모델 | 근거 |
|---|---|---|
| training judge (labelling) | 로컬 `Qwen3-14B` (`/work/uf4_20260910/assets/Qwen3-14B`) | 기존 캠페인과 동일 |
| evaluation judge 후보 | 로컬 `Llama-3.3-70B-Instruct`, `Llama-3.1-8B-Instruct`, `Skywork-Reward-V2-Qwen3-8B` | `/work/hf_cache/hub`에 존재 |

Qwen3-14B 디렉터리에는 upstream snapshot 식별자가 없어 revision `40c06982…`는 **우리 기록상의
선언값**이며 독립 검증된 식별자가 아니다. 이 문장을 settings.json마다 같이 남긴다. 같은 judge의
held-out prompt는 독립 judge가 아니라 teacher-held-out 진단으로만 쓴다. 최종 평가는 별 모델
(Llama-3.3-70B)로 확인하는 것을 최소 요건으로 하고, 예산은 audit 실측 후 별도로 계산한다.

## 5. 새 데이터 계약(동결)

- **PKU-SafeRLHF**, snapshot `9421ffafec3fa40a1f1a7d567b4d525079477ecb`, license cc-by-nc-4.0,
  train 73,907행 → 정규화 중복 제거 후 38,571개 고유 prompt.
- objective 2개(helpfulness, harmlessness), rubric sha256 `1d0aadb7…`, 후보 생성 전에 동결.
- audit panel 200개(pilot 10, confirmation 50 포함) — 점수와 무관한 namespace hash 정렬로 선택.
  `screen200.jsonl` sha `00a18458…`.
- policy panel split 2,000 / 500 / 1,000 — audit 200개를 먼저 제거한 pool에서 선택, 3-way
  disjoint 검증. `train2000.jsonl` sha `b448248c…`, `test1000.jsonl` sha `8dd31ce4…`.
- 후보는 base `Llama-3.1-8B-Instruct` rev `0e9e39f2…`에서 prompt당 8개, T=1/top_p=1/1024 토큰
  (학습 pool과 같은 분포이므로 나중에 learner pool Y로 쓸 수 있다).

**실측 기반 예산 확정**: pilot 10개에서 judge 처리량 **40.17 verdict/s/GPU**, parse 99.82%.
따라서 train+dev labelling 1,840,000 verdict = **12.7 GPU-h ≈ 4장에서 3.2시간**. 축소 조항을
쓸 필요가 없어 **선언된 2,000 train을 그대로 확정**했고, 근거는 `splits/freeze.json`의
`commitment_evidence`에 정책 학습 전에 기록했다.

## 6. WildChecklists

캐시에 없다. HF hub는 도달 가능(200)하므로 무료 다운로드로 받을 수 있다. 유료 API 호출은 0건을
유지한다. native item cycle audit 용도로만 쓰고, 고정 global rubric 계약 없이 NBPO 학습에는
쓰지 않는다.

## 7. UF readiness 행

`tab:dataset_readiness`의 UF 행은 이번 N=8·2 repeat 계약으로 다시 실행해야 채운다. 기존 N=4
audit 값을 복사하지 않는다. 필수 GPU 작업보다 후순위이며, 실행하지 않으면 해당 cell은 미측정으로
남긴다.
