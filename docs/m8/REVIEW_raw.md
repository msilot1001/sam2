# M8 독립 검토 (raw)

- 검토일: 2026-10-14. 대상 커밋: `fc5a95c` (작업 트리 기준). GPU 코드는 실행하지 않았다. 숫자는 모두 원시 CSV에서 CPU로 다시 계산했다.
- 재계산 범위: RunPod arm은 `runpod/runs/<ARM>/*.csv`와 `runpod/runs/SGGL/*.csv`를 비교했다. Ubuntu arm은 `runs/<ARM>/*.csv`와 `diag/<unit>.csv`를 비교했다. 작은 객체는 `area0 < 0.01`이고, 지표는 단위별 평균 JF 차이다. 영상 군집 bootstrap은 독립 구현으로 다시 돌렸다.

## 요약 판정

1. **숫자는 재현된다.** 헤드라인 ΔJ&F, 상승/하락 영상 수, LOVO, miami-surf 제외 값, 최악 단위, 놓친 frame 수, 거짓 재획득, 짝 비교가 모두 원시 CSV와 일치한다. CI 차이는 bootstrap seed 잡음 수준(≤ 0.002)이다.
2. **누출은 없다.** 배포형 arm에서 GT 누출이나 미래 frame 사용은 발견하지 못했다.
3. **기계 짝짓기가 지켜졌다.** 모든 판정 비교가 같은 기계 안에서 이뤄졌다.
4. **결정(KEEP_SGGL)은 타당하다.**
5. **그러나 서술에 수정이 필요하다.**
   - **VK 무효과에 대한 설명('VK에는 crop 이탈 사건이 거의 없다')은 데이터와 반대다.** VK SGGL의 'GT 있는 동시 부재' frame 비율은 DEV/REP의 2배다. 운반 arm은 이 frame을 줄이지 못했다(89 → 97).
   - 통계적 강도, '재현'이라는 단어, 사후 변형(FLOWCARRY)을 헤드라인에 쓴 점, 메모리 기록 미해결 가설의 근거는 과장되어 있다.
   - Stage 1 진출 규칙을 문자 그대로 적용하면 B_att가 통과한다. 그런데 이 재해석을 공개하지 않았다.
6. §4 결정 표에서 **CURRENT_INCUMBENT는 바뀌지 않는다.** BEST_CAUSAL_CHALLENGER, HIGHEST_VALUE_UNRESOLVED_HYPOTHESIS, NEXT_EXPERIMENT_PRIORITY는 문구 수정이 필요하다(§결정 표 영향).

### 재계산 결과 (작은 객체 57단위 / 34영상, 같은 기계)

| arm | ΔJ&F | 95% CI (재계산) | ↑/↓ | LOVO 최솟값 | miami 제외 | 최악 단위 | 놓친 frame (GT 있음, SGGL 131) | 거짓 재획득 | 문서와 일치 |
|---|---|---|---|---|---|---|---|---|---|
| FLOWVEL | +0.0130 | [+0.0010, +0.0288] | 5/2 | +0.0067 | +0.0138 | −0.026 | 63 | 93/93 | 일치 |
| FLOWCARRY | +0.0141 | [+0.0018, +0.0292] | 5/1 | +0.0079 | +0.0148 | −0.031 | 62 | 93/93 | 일치 |
| CAMCARRY | +0.0137 | [+0.0022, +0.0292] | 5/1 | +0.0075 | +0.0144 | −0.027 | 63 | 93/93 | 일치 |
| FV_PRESENT | +0.0016 | [−0.0049, +0.0104] | 2/3 | −0.0015 | +0.0040 | −0.104 | 110 | 93/93 | 일치 |
| M3vel | +0.0046 | [−0.0001, +0.0117] | 2/1 | +0.0017 | +0.0045 | −0.014 | 108 | 93/93 | 일치 |
| RESC_OCLR | −0.0001 | [−0.0037, +0.0027] | 2/1 | −0.0008 | +0.0016 | −0.058 | 130 | 99/93 | 일치 (회수 11) |
| RESC_OCLR_ADMIT | +0.0046 | [−0.0059, +0.0207] | 3/1 | −0.0019 | +0.0086 | −0.116 | 97 | 105/93 | 일치 (회수 8) |
| RESC_CPFR | −0.0070 | [−0.0252, +0.0103] | 2/3 | −0.0118 | −0.0016 | −0.330 | 108 | 118/93 | 일치 (회수 22) |
| ORACLE_POOL † | +0.0136 | [+0.0020, +0.0315] | 5/0 | +0.0068 | +0.0152 | 0.000 | 124 | 93/93 | 일치 (회수 61, Ubuntu 53 + RunPod 11, 중복 단위 0) |
| ORACLE_POOL_ADMIT † | +0.0169 | [+0.0020, +0.0419] | 5/0 | +0.0068 | +0.0188 | 0.000 | 77 | 93/93 | 일치 (회수 14) |
| K_OCLR_M3 | +0.0071 | [−0.0009, +0.0177] | 3/1 | | | −0.058 | 88 | 100 | 일치 |
| K_OCLR_FLOWVEL | +0.0121 | [−0.0008, +0.0296] | 4/3 | | | −0.058 | 62 | 99 | 일치 |
| K_ADMIT_FLOWVEL | +0.0103 | [−0.0035, +0.0280] | 4/3 | | | −0.127 | 65 | 105 | 일치 |
| FLOWVEL_U (2080, 46단위) | +0.0161 | [+0.0017, +0.0345] | 5/1 | +0.0084 | +0.0179 | −0.023 | — | 52/52 | 일치. RunPod FLOWVEL과 단위 상관 0.999998, 최대 차이 0.00076 |

**짝 비교 11개 모두 일치한다.** 예:
- FLOWCARRY − CAMCARRY: +0.0003
- FLOWVEL − FV_PRESENT: +0.0114
- K_ADMIT_FLOWVEL − FLOWVEL: −0.0027
- ORACLE_POOL_ADMIT − ORACLE_POOL: +0.0032 [−0.0001, +0.0099]. 이 비교는 문서에 CI 없이 나온다.

**영상 평균(단위가 아니라 영상을 같은 가중으로 본 평균)도 계산했다.**

| arm | 영상 평균 |
|---|---|
| FLOWVEL | +0.0102 |
| FLOWCARRY | +0.0112 |
| CAMCARRY | +0.0107 |
| ORACLE_POOL | +0.0124 |

헤드라인은 사전 등록대로 단위 평균이다. 영상 평균은 0.002–0.003 낮다.

### 확인되어 문제가 없는 것

- **GT 누출 없음.**
  - `run_arm`이 결정 전에 GT를 읽는 곳은 ORACLE 분기(`scripts/m8.py:478-479`)뿐이다.
  - `scripts/m8.py:495`의 GT는 출력과 crop이 정해진 뒤 지표 계산에만 쓴다.
  - trigger는 점수만 쓴다(`m8.py:460`).
  - `Track.observe`는 출력 box만 받는다.
  - `g0`(prompt frame)는 허용 범위다.
- **미래 frame 사용 없음.**
  - 광류는 `flow(prev=t−1, frame=t)`이고, `d.view(t)` 이후에 계산된다(`m8.py:459`).
  - t+1의 crop 중심은 t−1→t 변위로 정해진다.
  - CAMCARRY는 `d.center + trk.cam`을 쓰며, cam도 t−1→t다(`motion.py:81`). t+1 정보는 쓰지 않는다.
- **비저장 probe 검증.** RunPod `runs/SGGL`(run_arm)과 RunPod diag(diag_unit)가 11단위에서 최대 차이 0.0이다. diag의 비저장 probe가 궤적을 바꾸지 않는다.
- **SGGL 경로의 버전 간 불변.**
  - FLOWCARRY와 CAMCARRY는 출력이 한 번도 비지 않은 23영상에서 Δ가 정확히 0이다.
  - `motion.py`는 `718e618` 이후 변경되지 않았다.
- **기계 짝짓기.**
  - RunPod arm은 모두 RunPod SGGL과, Ubuntu arm은 Ubuntu diag와 짝지었다. ORACLE의 Ubuntu/RunPod 단위는 서로 겹치지 않는다.
  - `pairwise`는 `machine == runpod`로 제한한다.
  - 기계 간 SGGL 차이(RunPod − Ubuntu diag)는 53단위 중 50단위에서 0이고, 최대 0.0016이다.
  - 기계를 섞은 비교는 서술 수준의 'FLOWCARRY ≈ ORACLE_POOL' 하나뿐이다. 영향은 무시할 만하다.
- **FLOWVEL의 사전 고정 (git으로 확인).**
  - FLOWVEL은 `718e618`(18:12)의 arms.json에 있다. 첫 FLOWVEL CSV는 18:41이다.
  - FLOWVEL의 'why'에는 처음부터 '두 view가 놓친 동안 광류 box를 따라간다'가 들어 있다. 이득의 원천으로 밝혀진 요소가 사후에 추가된 것은 아니다.
  - FV_PRESENT, FLOWCARRY, CAMCARRY도 `b1ca5a0`(19:11)에 있다. 첫 출력은 20:26 이후다.
- **ORACLE 표기.** STAGE2·NON_ATTENTION 표에 †와 '(배포 불가)'가 붙어 있다. arms.json에 `diagnostic: true`가 있고, 대시보드에 배지가 있다.

## 지적 사항

### 1. [HIGH] VK 무효과 설명이 데이터와 반대다

**문서의 주장:**
- `FINAL_PORTFOLIO_ASSESSMENT.md:85`: "VK에는 그런 사건이 거의 없다"
- `STAGE2_CAUSAL_RESULTS.md:87`: "VK에는 crop 이탈 사건이 거의 없다"
- `NEXT_RESEARCH_DECISION_TREE.md:9`: "VK에서는 효과 없음(사건 부족)"

**근거 (재계산):**

| 항목 | VK (28단위, 1,736 frame) | DEV/REP (64단위, 5,082 frame) |
|---|---|---|
| SGGL 동시 부재 frame | 239 | 486 |
| └ GT 있음 (놓친 frame) | **89 (5.1%)** | 131 (2.6%) |

- VK는 frame당 '놓친 동시 부재'가 DEV/REP의 **2배**다.
- 그런데 FLOWVEL과 FLOWCARRY의 VK 놓친 frame은 **97**로, 오히려 늘었다. DEV/REP에서는 131 → 62–63으로 줄었다.
- VK run CSV에는 crop box나 `gt_in_crop`이 기록되지 않는다. 따라서 'crop 이탈 사건이 없다'는 측정된 사실이 아니라 추정이다.

**수정 요청:**
- 세 곳의 문구를 다음으로 바꾼다: "VK에도 GT 있는 동시 부재가 89 frame(5.1%) 있으나 운반 arm은 이를 줄이지 못했다(97). 원인 미규명."
- 일반화 근거가 약하다는 점을 FINAL §1/§6에 명시한다.
- NEXT_EXPERIMENT_PRIORITY에 다음을 추가한다: C나 새 데이터를 쓰기 전에, VK에서 놓친 frame을 crop 안/밖, GT 면적, 카메라 운동 크기로 나누는 진단을 먼저 한다. VK SGGL과 CAMCARRY에 crop box와 cam을 기록하는 GPU 1회 실행이면 된다.

### 2. [MEDIUM] 'CI 하한 > 0'의 증거력을 과대 제시했다

**문서:**
- `STAGE2:83`: "하한 > 0 (M7–M8에서 처음)"
- `FINAL:12`, `FINAL:45`, `TREE:8`: CI를 헤드라인 근거로 쓴다

**근거 (재계산):**

| arm | bootstrap P(Δ ≤ 0) (20,000회) | Bonferroni(배포형 11 arm) 99.55% 하한 | 영상 부호 검정 (단측) | drone + kite-surf 비중 | 두 영상 제외 Δ |
|---|---|---|---|---|---|
| FLOWVEL | 0.0095 | −0.0010 | 5↑/2↓ → p = 0.23 | 69% | +0.0044 |
| FLOWCARRY | 0.0024 | ≈ 0.000 | 5↑/1↓ → p = 0.11 | 64% | +0.0055 |
| CAMCARRY | 0.0032 | −0.0002 | 5↑/1↓ → p = 0.11 | 65% | +0.0051 |

- FLOWCARRY는 34영상 중 11영상에서만 Δ ≠ 0이다.
- 군집 34개 중 양의 군집이 5개뿐이면, percentile bootstrap의 하한은 주로 '재표본에 양의 영상이 하나라도 들어갈 확률'을 반영한다.

**수정 요청:**
- CI 옆에 다음을 함께 적는다.
  - 다중 비교 보정 시 하한 ≈ 0
  - 영상 부호 검정 p
  - 상위 2영상 제외 값
- 'M7–M8에서 처음'이라는 강조는 삭제한다. 'PROMISING'은 사전 등록 규칙의 판정이므로 유지할 수 있다.

### 3. [MEDIUM] FLOWVEL_U는 '재현'이 아니라 수치 재현성이다

**문서:**
- `FINAL:45` "2080에서 재현(+0.0161, 단위 상관 1.00)"
- `STAGE2:102-108`
- `FINAL:20` "기계 간 재현"

**근거:**
- 같은 코드, 같은 데이터, 결정론적 추론이다. 단위 상관이 0.999998이고 최대 차이가 0.00076이다.
- 새 정보는 'GPU와 torch 버전(2.11 → 2.14)이 결과를 바꾸지 않는다'는 것뿐이다. 효과의 통계적 재현을 뜻하지 않는다.
- +0.0161은 RunPod 기준선이 없는 11단위를 뺀 46단위 값이다. 57단위의 +0.0130과 나란히 놓고 비교할 수 없다.

**수정 요청:**
- 용어를 '기계 간 수치 재현성(computational reproducibility)'으로 바꾼다.
- BEST_CAUSAL_CHALLENGER의 근거 열에서 빼거나, '효과의 독립 확인이 아님'을 붙인다.
- 같은 46단위로 제한한 RunPod FLOWVEL 값을 함께 적는다.

### 4. [MEDIUM] 사후 변형 3개 중 최댓값(FLOWCARRY)이 헤드라인이다

**문서:**
- `FINAL:45`와 `TREE:8`은 FLOWCARRY(+0.0141)를 맨 앞에 놓는다.
- `TREE:18`은 PUMaVOS C 확인에 CAMCARRY를 동결하자고 한다.

**근거:**
- 출력 전에 고정된 운동 arm은 FLOWVEL(16:14)뿐이다.
- FV_PRESENT, FLOWCARRY, CAMCARRY는 FLOWVEL이 PROMISING으로 나온 뒤 19:08에 고정됐다.
- 세 변형은 짝 비교 차이 ≤ 0.0011로 사실상 같다. 이 중 최댓값을 헤드라인에 쓰면 승자의 저주가 생긴다.
- 사후성 자체는 `STAGE2` §2.1 끝과 `FINAL:84`에 공개되어 있다.
- 그러나 사전 등록의 'arm당 수정 1회' 정신에 비추면, 한 arm에서 변형 3개를 만든 것이다.
- `NON_ATTENTION_MECHANISM_COMPARISON.md:61`의 제목 "출력 전 고정"은 'FLOWVEL 결과 이후'라는 맥락을 빠뜨린다.

**수정 요청:**
- 헤드라인은 사전 고정된 FLOWVEL(+0.0130)로 한다. FLOWCARRY와 CAMCARRY는 '사후 ablation, 동등'으로 표기한다.
- C나 새 데이터로 확인할 arm 하나를 고르는 이유를 사전 등록 파일에 적는다. 예: CAMCARRY는 기제가 가장 단순하지만 사후 선택이다.
- NON_ATTENTION §4 제목에 "(FLOWVEL 결과 이후, 각 arm 출력 전)"을 붙인다.

### 5. [MEDIUM] Stage 1 진출 규칙의 재해석을 공개하지 않았다 (B_att는 문자 그대로 통과한다)

**문서:**
- `STAGE1_DIAGNOSTIC_RESULTS.md:78` "사전 등록 진출 조건을 통과한 생성기 계열도 없다"
- `STAGE2:15`
- `ATTENTION_MECHANISM_COMPARISON.md` §3: "진출 조건(…DEV FP ≤ 0.10, 4영상 이상)을 통과한 attention 생성기·순위화기가 없다"

**근거 — 사전 등록 조건별로 B_att를 대조했다:**

| 조건 | B_att 값 | 충족 |
|---|---|---|
| recall ≥ 0.25 | 0.389 | 예 |
| max(Gp, Lp) recall의 2배 이상 | 0.389 ≥ 2 × 0.069 | 예 |
| 4영상 이상 적중 | 4영상 | 예 (`stage1_report.json`의 `adv_recall=True`) |
| REP 고정 threshold로 DEV FP ≤ 0.10 | `rankers_GLOBAL.json`의 R0/R1/R3 REP→DEV `fp_rate_empty` = 0.0 | 예 |

- 즉 문자 그대로는 통과다.
- 다만 같은 threshold의 DEV 회수도 0이어서 통과가 퇴화적이다. B_att arm은 DEV에서 SGGL과 같고, DEV→REP 순이득은 −9.0이다.
- 결론(인과 arm을 실행할 가치가 없다)은 바뀌지 않는다. 그러나 '통과 없음'은 'held-out 순이득 > 0'이라는 추가 조건을 넣은 해석이다.

**수정 요청:**
- STAGE1의 '사전 등록 일탈' 목록에 다음을 추가한다: "B_att는 문자 규칙을 퇴화적으로 통과(FP 0, 회수 0). 순이득 > 0을 추가 요구해 진출시키지 않음."
- ATTENTION §3의 문장을 정정한다.

### 6. [MEDIUM] ablation 해석에 구성상의 한계가 있다

**(a) '대상 광류 vs 카메라 운동'은 깨끗한 대조가 아니다.**
- 부재 중 FLOWCARRY가 쓰는 변위는 '마지막 위치 box의 중앙 50%에서 잰 median flow'다(`motion.py:82`, `box_flow`).
- 대상이 이미 그 box를 벗어났거나 안 보이면, 이 값은 그 자리의 배경 흐름이다.
- 따라서 FLOWCARRY ≈ CAMCARRY(+0.0003)는 구성상 예상되는 결과다. '카메라 운동으로 충분하다'를 시험한 것이 아니다.

**(b) 카메라 운동 크기를 측정하지 않았다.**
- '카메라 운동 보정'이라는 기제를 직접 측정하지 않았다. `trk.cam`이 CSV에 기록되지 않는다.
- `TREE`의 "DAVIS의 카메라 패닝 영상(drone, kite-surf, skate-park)"도 측정 없이 단정한 것이다.

**(c) '끌려'라는 표현은 방향이 반대다.**
- `STAGE2:73`과 `PRIOR_ART:42`는 '부재 중 crop이 카메라 운동에 끌려'라고 쓴다.
- 실제 SGGL은 부재 중 `set_center(None)`로 crop을 영상 좌표에 **고정**한다(`sggl.py:95-98`). 장면이 움직이고 crop은 그대로 있다.

**(d) FLOWVEL − FV_PRESENT의 CI 하한이 seed에 따라 부호가 바뀐다.**
- 문서는 굵게 표시했다(`STAGE2:66`).
- 저장된 하한은 +0.0000127이고, 다른 seed로 재계산하면 −0.0002다.

**수정 요청:**
- 결론을 다음과 같이 낮춘다: "부재 중 crop을 장면 흐름으로 옮기는 것이 이득의 대부분. 대상 흐름과 배경 흐름은 이 구현에서 구별되지 않음(구성상)."
- 카메라 운동 크기와 이득 영상의 대응을 측정하기 전까지 '카메라 운동 보정'은 가설로 표기한다.
- (c)와 (d)는 문구와 숫자를 정정한다.

### 7. [MEDIUM] HIGHEST_VALUE_UNRESOLVED_HYPOTHESIS의 근거가 약하다

**문서:** `FINAL:49` "운동 보정 위에서의 '검증된 rescue만 메모리 기록' … +0.003 여지"

**근거:**
- ORACLE_POOL_ADMIT − ORACLE_POOL = +0.0032이고, CI는 [−0.0001, +0.0099]로 0을 포함한다.
- 이 차이는 **SGGL 위에서** 잰 값이다. '운동 보정 위'에서 잰 것이 아니다.
- 운동 보정과 기록을 실제로 결합한 유일한 arm인 K_ADMIT_FLOWVEL은 FLOWVEL보다 −0.0027 낮고, 최악 단위가 −0.127이다.
- ORACLE은 순위화뿐 아니라 존재 판정도 완벽하다(#8). 그런데 존재 판정은 문서 스스로 '해결되지 않은 병목'이라고 한 부분이다.

**수정 요청:**
- 근거 열에 위 CI와 K_ADMIT_FLOWVEL의 반대 증거를 적는다.
- '운동 보정 위에서'는 '미측정'으로 표기한다.

### 8. [LOW–MEDIUM] ORACLE의 정의를 잘못 썼다

**근거:**
- `STAGE2:34`는 "순위화가 완벽할 때의 인과 상한"이라고 쓴다.
- 그러나 코드는 GT가 빈 frame에서 모든 후보의 J를 0으로 두므로 절대 출력하지 않는다(`m8.py:479`). 즉 **존재 판정도 완벽하다**.
- 그래서 거짓 재획득이 93 = 93이다.
- `FINAL:13`은 "GT로 완벽히 고른 rescue의 상한"이라고 쓴다. 실제로는 이 31개 후보 풀, frame별 탐욕 선택, 동시 부재 trigger, 출력만 바꾸는 조건에서의 값이다. 일반적인 rescue 상한이 아니다. ADMIT 변형이 이 값을 넘는다(+0.0169).
- `causal_report.json`은 ORACLE 두 arm에 verdict "PROMISING"을 기록하고 diagnostic 플래그가 없다. 대시보드는 arms.json으로 보정하지만, JSON만 읽으면 오해할 수 있다.

**수정 요청:**
- 정의를 '존재 판정과 순위화가 모두 GT인 경우(이 후보 풀 기준)'로 정정한다.
- `_pilot`에서 diagnostic arm의 verdict를 "DIAGNOSTIC"으로 쓰게 한다.

### 9. [LOW–MEDIUM] FLOWVEL '출력 전 고정' 문구가 부정확하다

**문서:** `STAGE2:17` "M8 시작 시점에 고정했다(16:14 KST, 출력 전)"

**근거:**
- FLOWVEL은 `preregistration.json`(16:00:05)에 없다.
- 첫 diag CSV가 16:01에 생겼고, 16:12까지 7단위의 Stage 1 출력이 있었다. `preregistration_amend1.json`도 16:05에 "6 units"를 언급한다.
- 'causal 출력 전'이라는 점은 git으로 확인된다(18:12 < 18:41). 그러나 16:14라는 시각 자체는 git으로 검증할 수 없다. 첫 커밋이 18:12다.

**수정 요청:** "사전 등록(16:00) 이후 추가, Stage 1 diag 7단위 출력 후, 모든 인과 출력 전 고정"으로 고친다.

### 10. [LOW] 공개되지 않은 그 밖의 사전 등록 일탈

**(a) 운동 반증 시험의 비교 대상이 다르다.**
- 사전 등록은 "C/E vs production L view"다.
- 실제로는 Lp(보호 메모리)와 비교했다(STAGE1 §2 마지막 행).
- 동시 부재 frame에서는 production L의 점수가 ≤ 0이라 비교가 무의미하다. 이유는 타당하지만 공개되지 않았다.

**(b) 'coabs-subset dJF' 보고가 빠졌다.**
- 사전 등록 `also_reported`의 'coabs-subset dJF'가 문서 표에 없다.
- `causal_report.json`의 `coabs_dJF_framesum`에만 있다.

**수정 요청:** 두 항목을 일탈 목록에 추가하거나 표에 보충한다.

### 11. [LOW] AUC 수치를 재생성할 스크립트가 없다

**근거:**
- `STAGE1:84-85`의 AUC 0.60–0.71과 0.815는 `m8_analysis.py`에서 계산되지 않는다. 어떤 저장 리포트에도 없다.
- 재계산 결과:

| 판별 대상 | 문서 값 | 재계산 |
|---|---|---|
| frame 수준 present vs absent (후보 최대 score) | 0.60–0.71 | 0.658 |
| frame 수준 present vs absent (최대 cos_fg) | 0.60–0.71 | 0.710 |
| frame 안 맞는 후보 vs 틀린 후보 (cos_fg) | 0.815 | 0.861 (전체 후보 풀) / 0.899 (두 종류가 모두 있는 frame) |

- present vs absent는 문서와 일치한다. 0.815는 두 정의 어느 쪽으로도 재현되지 않는다.

**수정 요청:** AUC 계산을 `stage1()`에 넣고, 정의를 명시한다.

### 12. [LOW] 문서 사이의 불일치

1. **FLOWCARRY 고정 시각:** `arms.json`은 19:08이고, `preregistration_vk_amend1.json:3`은 "~19:25"다.
2. **비교한 arm 수:** `FINAL:83`과 `TREE:9`는 "15개", `STAGE2:89`는 "10여 개"다.
3. **영상 수:** `preregistration.json:5`과 계획 문서는 "64 units / 34 videos"라고 쓴다. 실제는 64단위 / **39영상**이다(`stage1_report.json` `n_videos=39`). 34는 작은 객체 단위의 영상 수다.
4. **FLOWCARRY·CAMCARRY의 비용:**
   - 지연은 0.416 s/frame이고 SGGL은 0.365다(+14%, 동시 부하 조건). 문서는 "0 (+광류)"만 적는다.
   - `FINAL:47`은 CAMCARRY에 대해 "대상 광류가 필요 없다"고 쓴다. 그러나 전 frame RAFT는 여전히 필요하다.

**수정 요청:** 통일하고 보충한다.

### 13. [LOW] 출처(provenance) 기록

**근거:**
- **ENV 파일 충돌:** FLOWCARRY의 VK 실행이 DEV/REP 실행과 같은 shard 태그(`0of4`)를 썼다. 그래서 DEV/REP의 ENV 파일이 덮어써졌고 `.conflict-20261014-212002`로 남았다.
- **코드 버전이 arm마다 다르다:**

| arm | 코드 버전 |
|---|---|
| SGGL DEV/REP 기준선 | `b5c5091` + 미커밋 12경로 |
| ablation | `b1ca5a0` + 미커밋 2경로 |
| K_ADMIT_FLOWVEL | `b1ca5a0` + 미커밋 26경로 |

- CODE_SYNC에는 파일별 해시가 없다.
- 실제 영향은 '확인되어 문제가 없는 것'의 근거(Δ = 0인 23영상, run_arm == diag)로 배제된다.

**수정 요청:**
- ENV 파일명에 set 태그를 넣는다.
- CODE_SYNC에 파일별 sha를 기록한다.

### 14. [LOW] 새로움과 일반성의 표현

**문서:**
- `FINAL` §7과 `PRIOR_ART:42`: "동시 부재 손실의 절반이 이 결함에서 온다"
- `PRIOR_ART:42`의 CMC 인용에는 "원문 미확인"이라고 적혀 있다.

**근거:**
- '절반'은 개발 세트에서 놓친 frame 수가 131 → 62로 줄었다는 데서 나온 것이다.
- 이득의 64%가 2영상에서 나온다.
- VK에서는 같은 결함이 보이지 않는다(#1).

**수정 요청:**
- "이 개발 세트(DAVIS 5영상 주도)에서"라는 한정을 붙인다.
- CMC 선행연구는 원문을 확인한 뒤 인용한다.

## 결정 표(§4) 영향

| 범주 | 현재 | 권고 |
|---|---|---|
| CURRENT_INCUMBENT | SGGL | **유지** (동의) |
| BEST_CAUSAL_CHALLENGER | FLOWCARRY ≈ CAMCARRY ≈ FLOWVEL, FLOWCARRY 수치를 선두로 | **FLOWVEL(사전 고정) +0.0130**을 선두로 한다. FLOWCARRY/CAMCARRY는 '사후 ablation, 동등'으로 둔다. 근거에서 '2080 재현'을 빼거나 '수치 재현성'으로 바꾼다. 'VK: GT 있는 동시 부재 89 frame을 줄이지 못함'과 '상위 2영상이 64–69%'를 추가한다 (#1–#4) |
| BEST_NON_ATTENTION_DIRECTION | CAMCARRY, 대상 광류 불필요 | '부재 중 crop 운반 (기제는 가설)'로 바꾼다. RAFT가 여전히 필요하고 지연이 +14%라는 점, 사후 선택이라는 점을 명시한다 (#6, #12) |
| HIGHEST_VALUE_UNRESOLVED_HYPOTHESIS | 운동 보정 + 검증된 기록, +0.003 여지 | CI가 0을 포함한다는 점, SGGL 위에서 측정했다는 점, K_ADMIT_FLOWVEL의 반대 증거를 명시한다. 우선순위를 낮추거나 '미측정'으로 표기한다 (#7) |
| NEXT_EXPERIMENT_PRIORITY | 새 데이터에서 CAMCARRY/FLOWCARRY → C | 먼저 VK 놓친 frame을 진단한다(crop box·cam 기록, GPU 1회). 그다음 단일 arm을 사전 등록하고 새 데이터로 확인한다. C는 그 뒤에 쓴다 (#1, #4) |
| 그 밖의 행 | | 변경 없음 |
