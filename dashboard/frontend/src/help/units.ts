import type { HelpMap } from "./types"

// keys: "units.<element>"
// Sources: scripts/m8.py (diag_unit, run_arm, generators, cand_feats), scripts/m8_analysis.py (load, per_frame, causal),
// dashboard/server.py (units, unit, frames), src/baseline/metrics.py (J/F), docs/m8/FAILURE_EPISODE_ANALYSIS.md,
// docs/m8/NON_ATTENTION_MECHANISM_COMPARISON.md, docs/m8/STAGE2_CAUSAL_RESULTS.md.
export const help: HelpMap = {
  "units.unit": {
    title: "Unit이란? (unit 선택기)",
    body: `
**unit은 한 영상 속 한 객체다.** M7/M8 지표는 모두 unit마다 계산한 뒤 영상(video) 단위로 묶어 판정한다.

### 이름 규칙
\`<데이터셋>__<영상 경로>__obj<k>\`
- CSV의 \`unit\` 열: \`davis__davis/drone__obj1\`
- 파일명·URL(stem)은 \`/\`를 \`_\`로 바꾼 것: \`davis__davis_drone__obj1\` (\`scripts/m7.py: stem()\`)
- \`obj<k>\` = 그 영상 annotation의 객체 번호. 같은 영상의 obj1과 obj2는 서로 다른 unit이고, 서로에게는 "다른 객체(방해물)"이다 (후보 표의 \`oth\` 열 참고).

| 접두어 | 데이터 | set |
|---|---|---|
| \`davis__\` | DAVIS 2017 val / train | DEV(val) 또는 REP(train) — 지표 표의 \`set\` 열로 구분 |
| \`pumavos__\` | PUMaVOS A/B (C는 봉인, 사용 안 함) | DEV |
| \`segtrack__\` | SegTrack | DEV |
| \`vkitti__\` | VKITTI2 Scene01/06 clone (합성) | VK |

### set
- **DEV**: M6 개발 crop unit 21개 (DAVIS val + PUMaVOS A/B + SegTrack). 여러 번 들여다본 개발 데이터다.
- **REP**: DAVIS 2017 train crop unit 43개 / 23영상 (M6.1 재현용, miami-surf 발견 사례 포함). 학습형 순위화기는 REP와 DEV로 교차 적합한다.
- **VK**: VKITTI2 28 unit / 2 scene. 고정된 finalist만 돌려 서술적으로 확인하는 용도다. 인과 판정(pilot rule) 통계에서 빠지므로 \`causal_per_unit.csv\`에도 VK 행이 없다(지표 표 오른쪽 열이 "—").
- DEV와 REP는 **둘 다 이미 본 데이터라 모든 결과가 탐색적**이다.

### crop unit만 있다
SGGL의 L crop(한 변 = prompt box 긴 변의 4배, 최소 min(H,W)/4)이 프레임보다 작을 때만 unit으로 넣는다 (\`vcam.crop_side() is not None\`). crop이 프레임 전체면 SGGL이 P5와 같아져 비교할 의미가 없기 때문이다.

### 크기 구분
\`area0\` = frame 0(prompt) GT 면적 / 프레임 면적.
- **small**: area0 < 1%. 인과 판정 대상이다 (DEV+REP 64 unit 중 small만 집계, 문서 기준 57 unit / 34영상)
- **tiny**: area0 < 0.1%. 따로 보고한다 (\`tiny_dJF\`)

### 선택기
- 목록은 서버가 \`outputs/milestone8/runs/<arm>/*.csv\`(Ubuntu)와 \`outputs/milestone8/runpod/runs/<arm>/*.csv\`(RunPod)를 훑어 만든 stem 목록이다 (\`server.py: units()\`).
- 괄호 안 숫자는 그 unit에 있는 \`기계/arm\` CSV 수다. **RunPod SGGL 기준선도 하나로 센다.** 그래서 (1)이면 기준선만 있고 arm 결과는 아직 없다.
- diag만 돌고 arm CSV가 하나도 없는 unit은 목록에 나오지 않는다.
- 데이터는 30초마다 다시 읽는다. 실행 중인 arm의 unit CSV는 다 쓴 뒤에 한 번에(원자적으로) 저장하므로 반쯤 찬 곡선은 보이지 않는다.
`,
  },

  "units.howto": {
    title: "실패 사례(episode)를 조사하는 방법 — 단계별 예시",
    body: `
이 페이지의 주 용도는 **"평균 ΔJ&F 뒤에 실제로 무슨 일이 있었나"**를 프레임 단위로 확인하는 것이다. 평균은 몇 개 영상에 몰린 이득이나 손실로 쉽게 움직이므로(문서 기준 FLOWVEL 이득은 drone·skate-park·kite-surf에 집중), 판정 전에 반드시 사례를 직접 본다.

### 기본 절차
1. **unit 고르기.** 출발점은 \`docs/m8/FAILURE_EPISODE_ANALYSIS.md\`의 사례 표(\`outputs/milestone8/stage0_coabs_episodes.csv\`)나 Arms 페이지의 최악·최고 unit이다.
2. **위 차트에서 실패 구간 찾기.** 아래쪽 **빨간 띠(co-absence)** 가 **회색 음영(gt_empty)** 없이 이어지는 구간은 대상이 있는데 SGGL이 빈 마스크를 낸 구간이다(기준선 J&F = 0). 빨간 띠가 회색 위에 있으면 진짜 부재를 올바르게 비운 것(J&F = 1)이라 문제없다.
3. **실패 유형 판단** (문서의 분류):
   - **위치형**: 대상이 crop 밖으로 나감. 탐색·운동이 필요하다
   - **표현형/가림·축소형**: 대상이 crop 안에 있지만 아주 작아지거나 가려짐. 새 관측으로도 어렵다
4. **arm 곡선 비교.** 범례에서 arm을 켜고 끄며 SGGL(굵은 회색)과 비교한다. 아래 Δ 차트에서 처음 갈라지는 프레임을 찾는다. 인과 rollout이라 **한 번 갈라지면 이후 전부가 바뀔 수 있다**. 원인은 처음 갈라진 지점에 있다.
5. **rescue 점 확인.** rescue arm의 굵은 점이 1 근처면 올바른 회수, 회색 위에서 0이면 진짜 부재에 대한 거짓 양성, 회색 밖에서 0이면 엉뚱한 객체를 잡은 것이다.
6. **후보 표로 원인 분석.** 차트를 클릭해 그 프레임의 stage-1 후보를 연다. 초록 행(J > 0.5)이 있으면 생성은 된 것이다. 있다면 그 행의 \`score\`, \`cos_fg − cos_bg\`, \`dist_flow\` 값이 배포 가능한 순위화기가 고를 수 있는 수준인지 본다. 같은 프레임의 다른 후보 중 \`oth\`와 \`score\`가 둘 다 높은 것이 있으면 방해물에게 질 위험이 있다.
7. **지표 표로 확인.** 오재획득 arm/SGGL이 늘었는지, 최악 unit(−0.05 미만이면 PROMISING 불가)에 걸리는지 본다.

### 예시 (2026-10-14 기준, 출처: FAILURE_EPISODE_ANALYSIS.md, NON_ATTENTION_MECHANISM_COMPARISON.md)
**① drone obj1 — 위치형, 생성으로 풀리는 유일한 대형 사례**
- M7 기록으로는 frame 18부터 54프레임 동안 동시 부재이고, 그중 39프레임에 GT가 있다. 대상은 crop 밖에 있고(crop 내부 비율 0.01) 크기는 크다.
- 차트: frame 18–23은 회색과 빨강이 겹친다(진짜 부재, 문제없음). 그 뒤로 회색 없이 빨간 띠가 길게 이어지고 SGGL은 0에 붙어 있다.
- 후보 표: 그 구간 프레임을 클릭하면 \`A_cos#*\`, \`B_att#*\`, \`D_ref\` 행이 거의 다 초록이다(문서: 39프레임 중 A_cos 38, B_att 37). \`F_oclr8_L\`도 22프레임이다. 반면 \`Lp\`, \`Gp\`, \`C_flowcrop\`, \`E_beam1\`은 0이다. **L crop 위치가 틀렸을 뿐, 전역 외형 매칭은 대상을 찾는다.**
- FLOWVEL 곡선: 대상이 다시 나타난 직후 재획득하고 그 뒤를 추적한다 (문서: unit Δ +0.379). SGGL·M3vel은 끝까지 놓친다. 같은 영상의 다른 drone은 진짜 부재 프레임의 거짓 양성 원천이므로 \`oth\`를 함께 본다.

**② kite-surf obj2 — 혼합형**
- GT 있는 프레임 23개, crop 내부 비율 0.48, 면적비 0.55. crop 안팎을 오가며 작아진다.
- 전역 매칭은 절반 정도만 잡는다(A_cos 10/23). Gp 8, Lp 6. FLOWVEL은 frame 21–28을 회수했다(Δ +0.132).
- 확인할 점: 초록 행이 없는 프레임은 생성 단계에서 이미 실패한 것이라 순위화를 고쳐도 못 살린다.

**③ skate-park obj2 — 혼합형, 빠른 재획득 문제**
- GT 있는 프레임 20개. 전역 매칭은 0이고, heatmap 위치에 L crop을 다시 놓는 \`B_att_crop\`만 5프레임을 잡는다.
- FLOWVEL과 M3vel 모두 SGGL보다 약 17프레임 일찍 재획득한다(FLOWVEL Δ +0.173). 위 차트에서 두 곡선이 SGGL보다 먼저 1 근처로 올라오는 지점이 그것이다.

**④ scooter-board obj2 — 가림·축소형, 회수 불가 + FLOWVEL 최악 unit**
- GT 있는 프레임의 GT 면적이 처음의 약 1%다 (거의 완전히 가려짐). **어떤 생성기도 J > 0.5 후보를 만들지 못한다** (후보 표에 초록 행이 없다. 단, gt_empty=True 행의 초록은 "올바르게 빈 후보"이므로 빼고 본다).
- FLOWVEL은 여기서 −0.026으로 최악 unit이다. Δ 차트에서 음수 구간이 어디서 시작되는지, crop 이동이 남은 조각을 놓치게 했는지를 본다.
- 교훈: 대상이 조각만 남은 프레임(문서상 GT 있는 co-absence의 약 1/3)은 recall 상한을 낮춘다.

### 흔한 함정
- 빨간 띠는 **기준선 SGGL 궤적**의 co-absence다. arm(특히 FLOWVEL)은 crop이 달라 co-absence 프레임 자체가 다르다 (문서: GT 있는 co-absence 131 → 63).
- 후보 표는 arm이 아니라 **stage-1 diag(SGGL 궤적)** 에서 만든 것이다.
- 한 unit의 큰 이득은 영상 단위 판정에서 한 표일 뿐이다(LOVO, miami-surf 제외 규칙). 여기서 본 사례로 가설을 세울 수는 있지만, 새 데이터로 확인하기 전까지는 탐색적 결과다.
`,
  },

  "units.chart_jf": {
    title: "프레임별 J&F 차트",
    body: `
### 무엇을 보여주나
x = 프레임 번호(\`video_idx\`), y = 그 프레임의 J&F (0–1).
- **굵은 반투명 회색 선** = SGGL 기준선 (진한 회색 = RunPod \`runpod/runs/SGGL/<stem>.csv\`, 옅은 회색 = Ubuntu \`diag/<stem>.csv\`)
- **색 선** = 각 \`기계/arm\` 실행 (\`runs/<arm>/<stem>.csv\`). 9번째 이후 arm은 색을 재사용하므로 점선으로 구분한다.
- 차트를 클릭하면 그 프레임이 아래 후보 표의 필터로 들어가고 점선 세로줄이 그려진다.

### 계산 (\`src/baseline/metrics.py\`, DAVIS-2017 공식 구현과 동일)
- **J (region)** = |P∩G| / |P∪G| (IoU). **P와 G가 둘 다 비면 J = 1.**
- **F (boundary)** = 경계 픽셀의 precision/recall 조화평균. 허용 거리 = ⌈0.008 × 이미지 대각선⌉ px.
  - 둘 다 경계 없음 → P = R = 1 → F = 1
  - 예측만 빔 → P = 1, R = 0 → F = 0
  - GT만 빔 → P = 0, R = 1 → F = 0
- **J&F** = (J + F) / 2.
- **frame 0(prompt 프레임)은 뺀다.** GT 마스크를 prompt로 주는 프레임이라 평가하지 않는다 (\`for t in range(1, n)\`). unit 점수 = 프레임 1..n−1의 J&F 평균.

| GT | 출력 | J | F | J&F |
|---|---|---|---|---|
| 있음 | 있음 | IoU | 경계 F | 평균 |
| 있음 | 빈 마스크 (놓침) | 0 | 0 | **0** |
| 빔 | 빈 마스크 (올바른 부재) | 1 | 1 | **1** |
| 빔 | 무엇이든 (1픽셀이라도) | 0 | 0 | **0** |

### 왜 거짓 양성이 비싼가
GT가 빈 프레임에서 아무것이나 출력하면 그 프레임 점수가 1에서 0이 되어 **−1**이다. 반면 GT가 있는 프레임을 회수하면 얻는 것은 그 프레임의 J&F(보통 0.5–0.8)뿐이다.
- 계산 예: 평가 프레임이 80개인 unit에서 거짓 양성 1프레임은 unit 평균 −1/80 = **−0.0125**다. 이것만으로 PROMISING 기준(+0.005)의 2배가 넘는다.
- 문서(FAILURE_EPISODE_ANALYSIS.md, M7 기록) 기준 co-absence 486프레임 중 GT 없음 355, GT 있음 131이다. **진짜 부재가 2.7배 많아서** "놓친 것을 다시 찾는" rescue는 구조적으로 손해 보기 쉽다. 모든 학습형 순위화기가 실패한 이유가 이것이다.

### 읽는 법
- 곡선이 0에 붙은 구간: 놓침(회색 음영 없음) 또는 거짓 양성(회색 음영 위).
- 곡선이 1 근처: 정확히 추적 중이거나, 진짜 부재를 올바르게 비움.
- 0과 1 사이를 오가는 톱니: 작은 객체에서 경계 F가 흔들리는 경우가 많다.
- 툴팁에 각 선의 값이 나온다. 선의 CSV 경로는 범례 칩에 마우스를 올리면 보인다.

### 주의
- 이 차트의 평균(지표 표 \`mean J&F\`)은 small/tiny 여부와 관계없이 그 unit 하나의 값이다. 판정은 small unit만 모아 영상 군집으로 한다.
- 한 unit의 곡선은 사례 이해용이다. 이득이 소수 영상에 몰려 있는지는 Arms 페이지의 영상별 표로 확인한다.
`,
  },

  "units.shading": {
    title: "음영·띠·점의 의미 (gt_empty / co-absence / rescued)",
    body: `
### 회색 음영 = \`gt_empty\`
- 그 프레임의 GT 마스크가 비어 있음 (대상이 화면 밖이거나 완전히 가려짐).
- GT 속성이라 모든 arm에 공통이다. 값은 기준 궤적 CSV의 \`gt_empty\` 열에서 읽는다.
- 여기서 올바른 출력은 **빈 마스크**(J&F = 1)다. 무언가 출력하면 J&F = 0 (거짓 재획득).
- 두 차트 모두에 표시된다.

### 빨간 띠(아래쪽 얇은 띠) = co-absence (동시 부재)
- 정의: **score_G ≤ 0 그리고 score_L ≤ 0** (G·L 두 상태의 SAM 2 object score logit이 모두 ≤ 0). 이때 SGGL 출력은 빈 마스크다. \`server.py: frames()\`에서 계산한다.
- **기준선 SGGL 궤적**(RunPod SGGL이 있으면 그것, 없으면 Ubuntu diag)의 co-absence다. **arm 자신의 co-absence가 아니다.** FLOWVEL처럼 crop을 다르게 움직이는 arm은 co-absence 프레임 자체가 달라진다.
- 카드 설명의 \`N frames\`는 이 기준 궤적의 co-absence 프레임 수다.
- 위쪽 J&F 차트에만 그린다.

### 조합별 해석

| 회색 | 빨강 | 의미 | SGGL J&F |
|---|---|---|---|
| ✓ | ✓ | 진짜 부재를 올바르게 비움 | 1 |
| ✗ | ✓ | **대상이 있는데 놓침**. rescue/운동 방법이 노리는 회수 여지 | 0 |
| ✓ | ✗ | 한 상태 이상이 "있다"고 함. 진짜 부재 프레임에서 출력이 나왔다면 거짓 재획득 | 보통 0 |
| ✗ | ✗ | 정상 추적 구간 (품질은 곡선 값으로 판단) | 0–1 |

문서 기준(2026-10-14, M7 기록 전체 64 unit): co-absence 486프레임 = GT 있음 131 + GT 없음 355. GT 있는 131프레임을 전부 완벽히 회수해도 unit 평균 +0.028이 상한이다. 그래서 rescue만으로는 STRONG_WIN(+0.03)에 닿을 수 없다.

### 굵은 점 = rescued 프레임
- arm CSV의 \`rescued == True\`인 프레임에서 그 arm 선 위에 찍힌다. y는 그 프레임의 arm J&F다.
- rescue가 있는 arm(RESC_*, ORACLE_*, K_*)에서만 생긴다. FLOWVEL·M3vel·ablation(FV_PRESENT/FLOWCARRY/CAMCARRY)은 rescue를 하지 않으므로 점이 없다.
- rescue는 **arm 자신의** co-absence(자기 score_G, score_L ≤ 0)에서 발동하므로 빨간 띠 밖에 점이 있을 수 있다.
- 점 높이로 판단한다. 1 근처는 올바른 회수, 회색 위 0은 진짜 부재에 대한 거짓 양성, 회색 밖 0은 다른 객체를 잡은 것이다(\`oth\`가 높은 후보였을 가능성).
- 어떤 생성기가 뽑혔는지는 arm CSV의 \`gen\` 열에 있다 (지표 표의 \`rescued frames\`는 개수만 보여 준다).
`,
  },

  "units.baseline": {
    title: "같은 기계 기준선 규칙 (machine-matched baseline)과 범례",
    body: `
### 규칙
**각 arm은 같은 기계에서 돈 SGGL과만 비교한다.**

| arm이 돈 기계 | 기준선 파일 |
|---|---|
| RunPod RTX 4090 (\`runpod/runs/<arm>/\`) | \`outputs/milestone8/runpod/runs/SGGL/<stem>.csv\` |
| Ubuntu RTX 2080 (\`runs/<arm>/\`) | \`outputs/milestone8/diag/<stem>.csv\` (stage-1 diag가 기록한 live SGGL 궤적) |

- \`scripts/m8_analysis.py: _base_frames()\`와 \`server.py: baseline_path()\`가 같은 규칙을 쓴다. Δ 차트와 지표 표의 Δ는 모두 이 짝을 기준으로 한다.
- Ubuntu diag는 후보 생성기를 **non-storing**(메모리를 바꾸지 않는) probe로만 돌리므로 SGGL 궤적이 그대로 남는다. 그래서 별도 SGGL 실행 없이 기준선으로 쓴다.

### 왜 필요한가
- 두 GPU는 수치가 bit 단위로 같지 않다. 인과 rollout에서는 작은 차이가 출력, 다음 crop 중심, 다음 프레임으로 이어지며 커져서 긴 영상에서는 궤적이 갈라질 수 있다.
- 예 (M7 기록, Vast 5090 vs Ubuntu 2080, fill_hole_area=0 고정 후에도): G의 J&F가 5082프레임 중 266프레임에서 달랐다 (unit 평균 차 ≤ 8e-5).
- 다른 기계 기준선과 비교하면 이런 기계 차이가 arm 효과에 섞인다. 다루는 효과 크기가 +0.005 수준이라 무시할 수 없다.

### 범례 읽기
- **SGGL runpod / SGGL ubuntu (기준선)**: 해당 기계 기준선이 있을 때만 나타난다. 둘 다 있으면 두 회색 선의 차이로 기계 간 차이를 직접 볼 수 있다. VK unit은 Ubuntu diag가 없어 RunPod 기준선만 있다.
- **arm 칩** (\`기계/arm\`): 클릭하면 숨기거나 다시 보인다 (모든 차트에 적용되고 지표 표는 그대로다). 마우스를 올리면 CSV 경로가 보인다.
- **◇ORACLE**: \`arms.json\`에서 \`diagnostic: true\`인 arm이다. co-absence 프레임에서 **GT로** 31개 후보 중 J가 가장 높고 J > 0.5인 것을 고른다. 완벽한 순위화의 인과 상한을 재는 진단용이고 **배포할 수 없다**.
- 같은 arm이 두 기계에 다 있으면 칩이 두 개 생긴다 (예: ORACLE_POOL은 Ubuntu 53 unit과 RunPod 11 unit으로 나눠 돌렸고, 각각 자기 기계 기준선과 짝짓는다).
`,
  },

  "units.chart_delta": {
    title: "Δ J&F 차트 (arm − 같은 기계 SGGL), 프레임별",
    body: `
### 계산
프레임 t마다 **Δ(t) = J&F_arm(t) − J&F_SGGL,같은기계(t)**. 기준선 규칙은 "같은 기계 기준선" 도움말을 참고한다. 같은 기계 기준선 프레임이 없으면 그 점은 비고, 그 arm의 Δ 통계는 "—"로 나온다.
- 이 곡선의 평균이 지표 표의 \`mean Δ (frames)\`이고, 정의상 \`causal_per_unit.csv\`의 \`d\`(= arm 평균 J&F − 기준선 평균 J&F, 같은 프레임 집합)와 같다.
- 회색 음영은 gt_empty다 (위 차트와 같다). 0 수평선이 기준이다.

### 읽는 법
- **+1 근처 막대/구간**: arm은 대상을 잡았는데 SGGL은 놓쳤거나(0), SGGL의 거짓 양성을 arm이 피했다.
- **−1 근처**: arm이 놓쳤거나, 회색 위라면 arm의 거짓 양성(거짓 재획득)이다.
- **작은 진동(±0.1 이하)**: 둘 다 추적 중이지만 마스크 경계가 조금 다르다. 평균에 잡음으로 들어간다.
- **긴 평탄 구간**: 궤적이 갈라져 서로 다른 상태로 진행하는 중이다.

### 해석 주의
- **인과 rollout이다.** 한 프레임에서 다른 결정을 내리면 crop 위치와 메모리가 달라져 그 뒤 모든 프레임이 바뀔 수 있다. 손실 구간의 원인은 그 구간이 아니라 **처음 Δ ≠ 0이 된 지점**에 있는 경우가 많다.
- 한 unit의 Δ는 판정 규칙의 "최악 unit ≥ −0.05" 조건에 직접 쓰인다. 큰 음수 구간이 신원 전환(다른 객체 추적)인지 확인한다. 문서 예: RESC_CPFR disc-jockey obj3 −0.330, RESC_OCLR_ADMIT miami-surf obj4 −0.116.
- 이득이 한 episode에 몰린 unit(예: drone obj1)은 영상 군집 CI와 LOVO에서 한 영상으로만 반영된다.
`,
  },

  "units.metrics": {
    title: "arm별 지표 표 (이 unit)",
    body: `
이 unit에서 각 \`기계/arm\` 실행 하나가 한 행이다.

### 두 출처가 섞여 있다
| 열 | 출처 |
|---|---|
| mean J&F, mean Δ (frames), rescued frames | 브라우저가 지금 읽은 arm CSV와 기준선 CSV에서 **바로 계산** |
| d (report), dJ, dF, 오재획득, latency, set | \`outputs/milestone8/causal_per_unit.csv\` (\`uv run python scripts/m8_analysis.py causal\`이 쓴 것) |

- \`mean Δ (frames)\`와 \`d (report)\`는 정의상 같아야 한다. 다르면 **보고서가 오래됐거나**(arm을 다시 돌리거나 추가한 뒤 analysis를 안 돌림) 기준선 파일이 바뀐 것이다.
- 오른쪽 열이 "—"이면 아직 analysis가 돌지 않은 arm이거나 VK unit이다 (VK는 \`causal_per_unit.csv\`에서 제외).

### 어떻게 판정으로 이어지나
판정(PROMISING/REJECTED/INCONCLUSIVE)은 이 표가 아니라 **small unit 전체를 영상 단위로 묶어서** 내린다 (Arms 페이지). 이 표는 한 unit의 기여와 위험(최악 unit, 거짓 재획득 증가)을 확인하는 용도다.

### arm 열
\`기계/arm\` 이름. 색 점은 차트 선 색과 같다.
- 기계: \`runpod\` = RunPod RTX 4090 worker, \`ubuntu\` = Ubuntu RTX 2080 (결과의 원본 저장소).
- arm 정의는 결과를 내기 전에 \`outputs/milestone8/arms.json\`에 고정(frozen)해 둔다. 주요 arm:
  - **FLOWVEL**: crop 중심 = 현재 대상 중심 + 0.5 × RAFT 광류 변위. 출력이 비어도 광류로 운반한 box를 따라간다.
  - **M3vel**: M7 도전자. 출력 중심 차분으로 구한 감쇠 속도(0.5).
  - **RESC_OCLR / RESC_CPFR**: co-absence에서 각각 평범한 crop, 문맥 보존 foveation으로 rescue하고, score > 0이면 채택한다. **_ADMIT**는 채택한 rescue를 메모리에 기록한다.
  - **ORACLE_POOL(_ADMIT)**: GT로 후보를 고르는 상한. 배포할 수 없다.
  - **K_\\***: 조합. **FV_PRESENT / FLOWCARRY / CAMCARRY**: FLOWVEL ablation.
`,
  },

  "units.m.jf": {
    title: "mean J&F",
    body: `
이 unit에서 이 arm의 평균 J&F. arm CSV의 프레임 1..n−1 \`JF\` 평균이다 (frame 0 prompt 제외, gt_empty 프레임 포함). 브라우저에서 계산한다.
- 절대값은 unit 난이도에 크게 좌우된다(작고 가림이 많은 객체는 낮음). arm 간 비교는 이 값보다 **Δ 열**로 한다.
- gt_empty 프레임이 많은 unit은 "올바르게 비우기"만 해도 높게 나온다 (J&F = 1 규칙).
`,
  },

  "units.m.dframes": {
    title: "mean Δ (frames)",
    body: `
Δ 차트 곡선의 평균이다. 프레임별 (arm J&F − **같은 기계** SGGL J&F)를 평균한 값으로, 브라우저에서 계산한다. 음수(−0.0001 미만)는 빨간색으로 표시한다.
- 같은 프레임 집합에서 계산하므로 \`d (report)\`와 같아야 한다. 다르면 보고서가 오래된 것.
- 같은 기계 기준선이 없으면 "—".
- 해석 기준: 판정 규칙은 이 값의 small-unit 평균 ≥ +0.005, 그리고 **−0.05 미만인 unit이 하나도 없을 것**이다. 여기서 −0.05 아래가 보이면 그 arm은 PROMISING이 될 수 없다.
`,
  },

  "units.m.rescued": {
    title: "rescued frames",
    body: `
arm CSV에서 \`rescued == True\`인 프레임 수. co-absence에서 후보를 채택해 출력한 횟수다.
- rescue arm(RESC_*, ORACLE_*, K_*)만 0이 아닐 수 있다. 운동 arm(FLOWVEL, M3vel)은 항상 0.
- **많을수록 좋은 게 아니다.** 진짜 부재 프레임에서의 rescue는 −1짜리 거짓 양성이다. 차트의 굵은 점 높이와 \`오재획득\` 열을 함께 본다.
- 문서 예(2026-10-14, 전체 집계): RESC_OCLR은 11프레임을 회수했지만 거짓 재획득이 93에서 99로 늘어 평균 −0.0001이었다.
`,
  },

  "units.m.d": {
    title: "d (report)",
    body: `
\`causal_per_unit.csv\`의 \`d\` = (arm 평균 J&F) − (같은 기계 SGGL 평균 J&F). 두 CSV를 \`video_idx\`로 합친 뒤 계산하고, 프레임 수가 다르면 analysis가 assert로 멈춘다.
- 이 값이 Arms 페이지의 영상 평균, LOVO, 최악 unit 계산에 그대로 들어간다.
- \`mean Δ (frames)\`와 다르면 보고서를 다시 만들어야 한다.
`,
  },

  "units.m.dJ": {
    title: "dJ",
    body: `
J(region IoU)만 본 차이. arm 평균 J − 같은 기계 SGGL 평균 J (\`causal_per_unit.csv\`).
- 놓침과 회수, 거짓 양성과 올바른 부재 사이의 변화는 J와 F를 함께 0과 1 사이에서 바꾸므로 dJ와 dF가 비슷하게 움직인다.
- dF만 크고 dJ가 작으면 주로 **경계 품질** 차이(작은 객체에서 흔함), dJ가 크면 **존재/위치** 차이로 볼 수 있다.
`,
  },

  "units.m.dF": {
    title: "dF",
    body: `
경계 F만 본 차이. arm 평균 F − 같은 기계 SGGL 평균 F (\`causal_per_unit.csv\`). F 허용 거리는 ⌈0.008 × 대각선⌉ px.
- 작은 객체에서는 몇 픽셀 차이로 F가 크게 흔들리므로 dF만 작은 양수라면 잡음일 수 있다.
- 문서 예(전체 집계, 2026-10-14): FLOWVEL ΔJ +0.0106 / ΔF +0.0154.
`,
  },

  "units.m.fp": {
    title: "오재획득 arm/SGGL (거짓 재획득)",
    body: `
**gt_empty 프레임 중 출력이 비어 있지 않은 프레임 수** (정확히는 \`gt_empty & J < 1\`). 왼쪽은 arm(\`fp_arm\`), 오른쪽은 같은 기계 SGGL(\`fp_base\`)이다.
- 이런 프레임의 J&F는 0이다 (빈 GT에 무엇이든 출력 → J = F = 0). 하나당 unit 평균에 −1/(n−1).
- arm이 SGGL보다 크면 그 arm이 진짜 부재 프레임에서 "있다"고 착각한 것이다. rescue arm이 실패하는 주된 원인이다.
- 문서 예(2026-10-14, 전체 합): FLOWVEL 93 / 93 (증가 없음), RESC_CPFR 118 / 93 (foveation이 방해물을 잡음).
- 방해물이 많은 영상(miami-surf 서퍼들, drone의 다른 drone, longboard, smile)에서 주로 늘어난다.
`,
  },

  "units.m.lat": {
    title: "latency arm/SGGL (s/frame)",
    body: `
arm CSV \`latency_s\`의 평균 / 같은 기계 SGGL의 평균. 한 프레임 처리 시간(G·L 추론 + 광류 + rescue 후보 + 순위화, GT 읽기 제외)이고, CUDA 동기화 뒤에 잰다.
- Ubuntu 기준선(diag)에는 \`latency_s\` 열이 없어 오른쪽이 비어 있을 수 있다.
- **RunPod에서는 3–4 worker가 GPU를 공유**했고, arm과 SGGL을 같은 시간대에 재지 않았다. 그래서 arm이 SGGL보다 빠르게 나오기도 한다(문서: FLOWVEL 0.33 vs SGGL 0.37). 비용은 절대값보다 **추가 SAM 2 호출 수**와 단독 측정(COMPUTE_AND_PERFORMANCE_AUDIT.md)으로 판단한다.
`,
  },

  "units.m.set": {
    title: "set (· tiny)",
    body: `
이 unit이 속한 데이터 분할(\`causal_per_unit.csv\`의 \`set\`).
- **DEV**: DAVIS val + PUMaVOS A/B + SegTrack. **REP**: DAVIS train. 둘 다 이미 본 데이터라 결과는 탐색적이다. 학습형 순위화기는 한쪽에서 적합하고 다른 쪽에서 평가한다(교차 적합).
- **· tiny**: prompt 면적 < 0.1%. 판정에서 \`tiny_dJF\`로 따로 보고한다 (문서 예: FLOWVEL 극소 7 unit +0.046).
- VK unit은 이 파일에 없어서 "—"다.
`,
  },

  "units.candidates": {
    title: "Stage-1 후보 표 (candidates)",
    body: `
### 무엇인가
stage-1 진단(\`scripts/m8.py diag\`)에서 이 unit의 **trigger 프레임마다 모든 rescue 생성기를 돌려 얻은 후보 마스크**의 특징과, 정답과 비교한 품질이다. 한 행이 (프레임, 생성기 변형) 하나다.
- 파일: \`outputs/milestone8/diag/<stem>.cand.csv\` (Ubuntu), 없으면 \`outputs/milestone8/runpod/diag/<stem>.cand.csv\`. 카드 설명에 실제 경로가 나온다.
- trigger 프레임 하나당 후보 31개: A_cos/B_att/H_s8 각 3 peak(9) + G_neg + D_ref + A/B_crop(2) + E_beam1/2(2) + C_flowcrop/flowbox(2) + F 계열 12 + Gp/Lp(2).
- 모든 생성기는 **non-storing probe**다. 메모리를 바꾸지 않으므로 기록된 궤적은 순수 SGGL이다. 그래서 이 표는 arm 궤적이 아니라 **SGGL 궤적 위의 후보**다.

### stage-1 지표와의 관계 (\`m8_analysis.py: per_frame\`)
- **recall**: co-absence이고 GT 있는 프레임에서, 그 생성기 계열의 후보 중 하나라도 **J > 0.5**이면 hit.
- **raw FP**: co-absence이고 GT 없는 프레임에서, 그 계열 후보 중 하나라도 "present"(\`score > 0\` 그리고 \`area > 0\`)이면 FP.
- **net_dJ_frames** = 회수한 프레임의 +J − 거짓 양성 1개당 1.
- 진출 조건: recall ≥ 0.25, Gp/Lp recall의 2배 이상, 4영상 이상 hit, REP에서 적합한 순위화기로 DEV FP ≤ 0.10. 문서 기준 이를 통과한 계열은 없다.

### 초록 행 = J > 0.5
- GT 있는 프레임에서는 이 후보가 대상을 잡았다는 뜻이다.
- **함정**: gt_empty=True 프레임에서 빈 후보는 J = 1이라 초록이 된다. 이것은 "올바르게 비었다"는 뜻이지 회수가 아니다.

### 주의 (사전 등록 수정 1·2)
- \`.cand1b.csv\`가 있는 unit(초기 shard-0의 22 unit)은 이 표에 \`F_*_pt\` 행이 없고, \`H_s8\` 행은 **버그 버전**(디코더의 conv_s1 64-d 특징을 쓴 것)이다. analysis는 이 H_s8 행을 버리고 \`.cand1b.csv\`의 재계산 행을 쓰는데, 이 페이지는 \`.cand1b.csv\`를 보여주지 않는다.
- \`J\`, \`oth\`, \`gt_empty\`는 GT를 쓰는 **사후 분석용** 열이다. 배포 가능한 순위화기(R0–R3)는 \`score, iou, cos_fg, cos_bg, cos_fg−cos_bg, log(area_ratio), dist_flow\`(+ R3는 후보 특징 벡터)만 본다.
- 최대 1000행까지만 그린다. 프레임 필터를 쓰면 한 프레임의 31개를 비교하기 쉽다.
`,
  },

  "units.cand_filter": {
    title: "후보 필터 (프레임 / 생성기)",
    body: `
- **frame (video_idx)**: 정확히 일치하는 프레임 번호만 보여 준다. 위 차트를 클릭해도 채워지고, 차트에 점선 세로줄이 생긴다. "프레임 필터 해제"로 지운다.
- **생성기 필터**: \`gen\` 이름을 대소문자 구분 없이 부분 문자열로 찾는다. 예: \`A_cos\` → A_cos#0–2와 A_cos_crop, \`_pt\` → 점 prompt F 변형들, \`F_cpfr\` → foveation 계열.
- **후보가 있는 프레임** 목록(처음 40개)은 trigger(coabs/disagree/regular)가 발생한 프레임이다. 차트의 빨간 띠 구간과 비교하면 co-absence 프레임이 어디인지 알 수 있다.
- 권장 사용법: 빨간 띠가 있고 회색이 없는 구간에서 프레임 하나를 고른다. 31개 후보 중 초록 행이 있는지 보고, 있다면 그 행이 \`score\`·\`cos_fg−cos_bg\`·\`dist_flow\`에서 다른 후보보다 앞서는지 확인한다.
`,
  },

  "units.cand.video_idx": {
    title: "video_idx",
    body: `
프레임 번호. 차트 x축과 같다. frame 0은 prompt 프레임이라 후보가 없다. 같은 \`video_idx\`의 행들은 같은 프레임에 대한 서로 다른 생성기의 후보다.
`,
  },

  "units.cand.trig": {
    title: "trig (trigger 종류)",
    body: `
그 프레임을 왜 후보 생성 대상으로 골랐는지 나타낸다. **배포 시점에 볼 수 있는 정보(G·L의 점수와 마스크)만으로** 인과적으로 정한다 (\`scripts/m8.py: trigger()\`). 우선순위 순서:

| 값 | 조건 | 용도 |
|---|---|---|
| \`coabs\` | score_G ≤ 0 **그리고** score_L ≤ 0 (SGGL 출력이 빔) | **stage-1 recall/FP 집계와 인과 rescue arm은 이것만 사용** |
| \`disagree\` | 정확히 한 상태만 score > 0, 또는 IoU(붙인 L, G) < 0.5 | 진단·순위화기 학습 보조 |
| \`regular\` | 위에 해당 없고 프레임 번호가 10의 배수 | 정상 상태 보정용 표본 |

- coabs 행에서 gt_empty=False면 회수 대상, gt_empty=True면 출력해서는 안 되는 프레임이다.
`,
  },

  "units.cand.gen": {
    title: "gen (생성기)",
    body: `
후보를 만든 생성기. \`#k\`는 heatmap의 k번째 peak (0 = 최고점, NMS 후 상위 3개).

| 생성기 | 방식 | 사용 메모리 |
|---|---|---|
| \`A_cos#k\` | frame-0 전경 prototype과 G stride-16 특징의 cosine heatmap의 peak에 점 prompt | G |
| \`B_att#k\` | frame-0 전경 8 + 배경 32 k-means 토큰에 대한 대조 attention(τ 0.1) 사후확률로 점 prompt | G |
| \`H_s8#k\` | stride-8 FPN 특징의 prototype cosine으로 점 prompt | G |
| \`G_neg\` | B top-1 양성 점 + B가 배경으로 보는 A peak에 음성 점 | G |
| \`D_ref\` | B top-1 점, **frame-0 메모리만** 사용 | G (cond only) |
| \`A_cos_crop\`, \`B_att_crop\` | A/B top-1 위치에 L crop을 다시 놓음 (prompt 없음) | L |
| \`E_beam1\`, \`E_beam2\` | 마지막 신뢰 중심 + 1×/2× 속도 × 경과 프레임 위치에 L crop | L |
| \`C_flowcrop\` | RAFT 광류로 운반한 box 중심에 L crop | L |
| \`C_flowbox\` | 광류 box를 G에 box prompt | G |
| \`F_oclr{8,4}_{G,L}\` | 광류 box 중심, 대상 긴 변이 8/4 cell이 되도록 확대한 **평범한 crop** | G/L |
| \`F_cpfr{8,4}_{G,L}\` | 같은 배율의 **문맥 보존 foveation**(주변을 압축해 남김) | G/L |
| \`…_pt\` | 위 F 변형 + 예측 중심에 점 prompt (사전 등록 수정 1) | G/L |
| \`Gp\`, \`Lp\` | M7 보호 메모리 probe (참조 기준) | G/L |

문서 기준(2026-10-14): drone obj1에서는 A_cos/B_att/D_ref가 거의 다 회수하고, skate-park obj2에서는 B_att_crop만 일부 회수하며, scooter-board obj2에서는 어떤 생성기도 0이다.
`,
  },

  "units.cand.score": {
    title: "score (SAM 2 object score logit)",
    body: `
그 probe에서 SAM 2 디코더가 낸 **object score logit** (\`object_score_logits\`). > 0이면 SAM 2가 대상이 있다고 판단한 것이고, ≤ 0이면 출력 마스크를 비운다. Gp/Lp는 해당 view의 score다.
- **R0 순위화기**는 이 값이 가장 큰 후보를 고른다. **RESC_* arm의 채택 규칙**은 최고 score > 0이다 (\`thr 0.0\`).
- stage-1 "present" 판정은 score > 0이고 area > 0인 것이다. GT가 빈 프레임에서 present면 raw FP다.
- **함정**: 비슷한 방해물(다른 서퍼, 다른 drone)에도 높은 score가 나온다. 생성기마다(전체 프레임 vs 확대 crop vs foveation) score 분포가 달라 서로 직접 비교하기 어렵다. foveation(cpfr)은 문맥이 보여서 방해물에 높은 score를 주기 쉬웠다.
`,
  },

  "units.cand.iou": {
    title: "iou (디코더 예측 IoU)",
    body: `
SAM 2 마스크 디코더가 **스스로 예측한** IoU (multimask 출력이면 그중 최댓값, 아니면 단일 값). 해당 probe의 디코더 호출에서 hook으로 기록한다 (\`sggl.py: _hook_iou\`).
- GT와 비교한 실제 IoU가 아니다. 실제 품질은 \`J\` 열에서 본다.
- 마스크 "모양"에 대한 자신감이라 엉뚱한 객체에서도 높을 수 있다. R2/R3 순위화기의 입력 특징 중 하나다.
`,
  },

  "units.cand.area": {
    title: "area",
    body: `
후보 마스크의 픽셀 수 (원본 프레임 해상도, logit > 0). crop/foveation view의 후보는 프레임 좌표로 되돌린 뒤 센다. view 밖은 관측하지 못한 곳이라 배경으로 친다.
- **0이면 빈 후보다.** 이때 \`cos_fg\`, \`cos_bg\`, \`dist_flow\`, \`oth\`는 비어 있다(NaN). GT가 빈 프레임에서 빈 후보는 J = 1이다.
- 상대적인 크기는 \`area_ratio\`로 본다.
`,
  },

  "units.cand.cos_fg": {
    title: "cos_fg (전경 prototype과의 cosine)",
    body: `
후보 마스크로 **G 상태의 현재 프레임 stride-16 특징**(64×64 grid, 256-d)을 면적 가중 평균하고 L2 정규화한 벡터와, **frame-0 전경 prototype**(prompt 마스크로 같은 방식으로 pooling한 벡터) 사이의 cosine이다 (\`m8.py: cand_feats\`, \`rescue.py: proto\`).
- 범위는 −1..1. 높을수록 frame 0의 대상과 외형이 비슷하다.
- 어떤 생성기가 만들었든 **항상 G의 전체 프레임 특징**으로 계산하므로 생성기 간 비교가 공정하다.
- **함정**: 같은 종류의 방해물(다른 drone, 다른 서퍼)도 cos_fg가 높다. 아주 작은 후보는 한 cell(약 프레임 1/64 폭)보다 작아 주변 배경이 섞인다.
- R1 순위화기는 score > 0인 후보 중 **cos_fg − cos_bg**가 가장 큰 것을 고른다.
`,
  },

  "units.cand.cos_bg": {
    title: "cos_bg (배경 prototype과의 cosine)",
    body: `
\`cos_fg\`와 같은 후보 벡터와 **frame-0 배경 prototype**의 cosine이다. 배경 prototype은 frame 0에서 prompt 마스크를 max(4, √면적) px 팽창한 영역의 바깥 전체를 pooling한 것이다.
- 높으면 후보가 frame 0의 배경(물, 하늘, 도로 등)처럼 보인다는 뜻이다. 배경 조각을 잡았을 수 있다.
- 판단은 보통 **margin = cos_fg − cos_bg**로 한다 (R1의 점수, R2/R3의 입력). margin이 크고 score > 0이며 dist_flow가 작으면 그럴듯한 후보다.
`,
  },

  "units.cand.dist_flow": {
    title: "dist_flow (광류 box 중심까지 거리)",
    body: `
후보 마스크 중심과 **광류로 운반한 box의 중심** 사이 유클리드 거리 ÷ **crop 한 변**(px).
- 광류 box: 마지막으로 SGGL 출력이 있던 프레임의 bbox에서 시작해, 매 프레임 RAFT-small 광류의 대상 영역 변위만큼 옮긴 box다. 출력이 다시 생기면 그 bbox로 다시 고정한다 (\`motion.Track\`).
- crop 한 변 = prompt box 긴 변 × 4 (최소 min(H,W)/4).
- 0은 예측 위치 바로 그 자리, 0.5는 crop 반 변, 1은 crop 한 변만큼 떨어진 것이다. 빈 후보는 NaN이다 (R2/R3는 5로 채우고 0–5로 자른다).
- **함정**: 부재가 길어지면 광류 box가 대상에서 표류한다. 그때는 올바른 후보도 멀게 나온다. FLOWVEL은 같은 광류 box를 crop 중심 제어에 쓴다.
`,
  },

  "units.cand.area_ratio": {
    title: "area_ratio (prompt 면적 대비)",
    body: `
후보 면적 ÷ **frame 0 prompt 마스크 면적**.
- ≫ 1이면 배경이나 더 큰 다른 객체로 번진 것이다. ≪ 1이면 조각만 잡았거나, 대상 자체가 작아졌거나 가려진 것이다.
- 대상의 크기는 실제로 변하므로 1이 정답인 것은 아니다. 문서(M7 기록) 기준 GT 있는 co-absence 프레임의 45%는 대상이 초기 면적의 0.5배 미만이었다.
- R2/R3는 log(area_ratio)(0.001–1000으로 자름)를 입력으로 쓴다.
`,
  },

  "units.cand.J": {
    title: "J (후보의 실제 IoU)",
    body: `
후보 마스크와 그 프레임 GT의 **실제 IoU** (\`j_measure\`). 둘 다 비면 1.
- **> 0.5 = hit** (stage-1 recall 기준, 초록 행).
- GT를 쓰므로 **배포 때 쓸 수 없는 정보**다. ORACLE arm만 이것으로 후보를 고른다 (J 최대이면서 > 0.5).
- 같은 프레임에 초록 행이 있는데 SGGL/arm이 놓쳤다면 문제는 **순위화/채택**이다. 초록 행이 하나도 없다면 문제는 **생성**이다(새 관측으로도 대상을 못 잡음, 예: scooter-board obj2).
`,
  },

  "units.cand.oth": {
    title: "oth (다른 객체 위 비율)",
    body: `
후보 픽셀 중 **같은 영상의 다른 annotation 객체** 위에 놓인 비율 = |후보 ∩ 다른 객체들의 GT 합집합| / |후보| (\`m7.other()\`). 빈 후보는 NaN.
- 사후 귀속(attribution)용 GT 정보라 순위화기는 보지 않는다.
- 높으면(예: > 0.5) **신원 전환 후보**다. 같은 영상의 다른 drone, 서퍼, 양 등을 잡은 것이다. 이런 후보가 score도 높으면 rescue/ADMIT가 방해물로 갈아탈 위험이 크다 (문서 예: RESC_OCLR_ADMIT miami-surf obj4 −0.116).
- **한계**: annotation된 객체만 센다. annotation 없는 방해물을 잡아도 0이다.
`,
  },

  "units.cand.gt_empty": {
    title: "gt_empty",
    body: `
그 프레임의 GT 마스크가 비어 있는지 (대상이 화면 밖이거나 완전히 가려짐). 차트의 회색 음영과 같은 정보다.
- True인 프레임에서 올바른 후보는 **빈 마스크 또는 score ≤ 0**이다. present 후보(score > 0, area > 0)는 raw FP이고, 채택되면 J&F가 1에서 0이 된다.
- True인 행이 초록(J = 1)이면 "후보가 비었다"는 뜻일 뿐이다.
- 문서 기준(M7 기록): co-absence 프레임 중 진짜 부재 355, GT 있음 131. 표에서 coabs 행 대부분이 gt_empty=True인 unit은 rescue가 손해 보기 쉬운 unit이다.
`,
  },
}
