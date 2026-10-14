import type { HelpMap } from "./types"

// keys: "arms.<element>"
// Sources: scripts/m8_analysis.py (_pilot, causal), scripts/m8.py (run_arm, Rank, m3_center), src/spatial/motion.py,
// outputs/milestone8/{arms.json,causal_report.json,causal_per_unit.csv,preregistration*.json}, docs/m8/*.md.

const BASE = `**같은 기계 기준선 규칙:** 모든 Δ는 'arm − 같은 unit의 SGGL'이며, SGGL은 arm과 **같은 기계**에서 돈 것만 쓴다.
- RunPod RTX 4090 arm → \`outputs/milestone8/runpod/runs/SGGL/<unit>.csv\`
- Ubuntu RTX 2080 arm → Ubuntu diag 궤적(\`outputs/milestone8/diag/*.csv\`, SGGL과 비트 동일한 rollout)

기계가 다르면 같은 SGGL도 수치가 조금 달라지므로(커널·정밀도 차이) 기계를 섞어 빼면 가짜 Δ가 생긴다. 짝이 없는 unit은 계산에서 빠진다.`

const SMALL = `**대상 unit:** 판정 지표는 **작은 객체 unit(첫 frame 면적 area0 < 1%)만** 쓴다. 한 unit = 한 영상의 한 객체. VK(VKITTI) unit은 따로 \`<arm>@VK\`로 집계되고 여기서는 빠진다.`

const PILOT = `**판정 규칙 (사전 등록 \`preregistration.json\` → \`stage2_causal_rule\`, M7 pilot 규칙과 같음):**

| 판정 | 조건 |
|---|---|
| PROMISING | ΔJ&F ≥ +0.005 **그리고** ΔJ&F > +0.01인 영상 ≥ 3개 **그리고** LOVO 최솟값 > 0 **그리고** miami-surf 제외 ΔJ&F > 0 **그리고** 최악 unit ≥ −0.05 |
| REJECTED | ΔJ&F ≤ 0, **또는** (상승 영상 ≤ 2개 **이고** LOVO 최솟값 ≤ 0) |
| INCONCLUSIVE | 위 둘 다 아님 |
| STRONG_WIN | ΔJ&F ≥ +0.03 **그리고** 영상 군집 CI 하한 > 0 (위 판정을 덮어씀) |

STRONG_WIN은 이 데이터에서 사실상 불가능하다. GT가 있는 동시 부재 frame 131개를 모두 완벽히 살려도 평균 +0.028이기 때문이다(\`preregistration.json\` \`headroom_disclosed\`).`

const STATUS_NOTE = `※ 수치는 실험 진행에 따라 바뀐다. 위 예시는 2026-10-14 기준 \`causal_report.json\` / \`docs/m8/*.md\` 값이며, 현재 값은 표에서 확인할 것.`

export const help: HelpMap = {
  // ------------------------------------------------------------------ page / main table
  "arms.table": {
    title: "모든 causal arm 지표 (표 전체)",
    body: `**무엇인가:** Stage 2/3 인과(causal) arm 각각을 같은 기계의 SGGL과 비교한 요약 한 줄씩이다.

**인과 rollout이란:** 영상 전체를 처음부터 끝까지 돌리며, arm의 출력이 다음 frame의 crop 중심(그리고 ADMIT이면 메모리)을 바꾼다. 즉 한 frame의 변화가 이후 모든 frame에 전파된다. Stage 1(진단)은 SGGL 궤적 위에서 후보를 '저장하지 않고' 찔러 본 것이라 이런 연쇄 효과가 없다. 그래서 Stage 1에서 좋아 보이던 rescue가 인과 rollout에서는 사라지거나 해로워질 수 있다.

**원자료:**
- \`outputs/milestone8/causal_report.json\` — 이 표의 모든 열 (\`scripts/m8_analysis.py\` \`_pilot()\`)
- \`outputs/milestone8/causal_per_unit.csv\` — unit별 Δ (아래 산점도, 비교 표)
- \`outputs/milestone8/arms.json\` — 각 arm의 frozen spec
- 재생성: \`uv run python scripts/m8_analysis.py causal\`

${BASE}

${SMALL}

**읽는 법:**
- 열 머리글 클릭 = 정렬(기본: ΔJ&F 내림차순). 행 클릭 = 아래 상세 카드가 그 arm으로 바뀐다.
- 빨간 ORACLE 배지 = GT를 보고 후보를 고른 진단용 arm. **배포 불가**이며 순위에서 '경쟁자'로 보면 안 된다.
- 판정(verdict)만 보지 말고 영상 수(↑/↓), LOVO, 최악 unit을 함께 볼 것. ΔJ&F 평균은 drone·kite-surf·skate-park 같은 몇 영상에 크게 몰린다.

**주의:**
- DEV + REP 모두 이전 마일스톤에서 이미 본 데이터다 → 모든 결과는 **탐색적**이다.
- 같은 영상 집합에서 arm 10여 개를 비교했다 → 다중 비교. 가장 좋은 arm의 값은 낙관적으로 편향된다.
- 일부 arm(ablation, 후기 결합)은 아직 실행 중일 수 있다. 실행 중인 arm은 끝난 unit만 집계되거나 표에 없다.`,
  },

  "arms.col.arm": {
    title: "arm (이름)",
    body: `\`outputs/milestone8/arms.json\`에 출력 전에 고정(freeze)된 arm 이름이다. 아래 상세 카드 제목 옆 "?"에 arm별 메커니즘·가설·계열·선행연구·상태 설명이 있다.

- **ORACLE** 배지: spec에 \`diagnostic: true\` — GT가 후보를 고른다. 배포 불가 상한.
- \`@VK\`가 붙은 이름(VKITTI 결과)은 이 표가 아니라 아래 VK 표에 나온다.
- SGGL 자체는 기준선이라 표에 행이 없다(모든 Δ의 0점).

이름 규칙: \`RESC_*\` = 동시 부재 rescue, \`*_ADMIT\` = rescue를 메모리에 기록, \`K_*\` = 두 계열 결합, \`ORACLE_*\` = GT 선택 상한, \`FV_PRESENT/FLOWCARRY/CAMCARRY\` = FLOWVEL ablation.`,
  },

  "arms.col.verdict": {
    title: "verdict (판정)",
    body: `\`causal_report.json\`의 \`verdict\`. \`scripts/m8_analysis.py\` \`_pilot()\`가 아래 규칙을 기계적으로 적용한 결과다.

${PILOT}

**읽는 법·주의:**
- PROMISING = "다음 단계(새 데이터 확인)로 보낼 가치가 있다"이지 "SGGL보다 낫다는 확증"이 아니다. CI가 0을 넘는지는 PROMISING 조건에 없다(따로 CI 열을 볼 것).
- ORACLE arm에도 같은 규칙이 적용돼 PROMISING이 찍힐 수 있지만, GT를 쓰므로 **판정 자체가 의미 없다**(상한 측정용).
- 규칙의 각 조건은 서로 다른 실패를 막는다: 영상 수 = 한두 영상 쏠림, LOVO = 한 영상 의존, miami-surf 제외 = 그 영상의 큰 변동, 최악 unit = 신원 전환 같은 재앙적 손실.
- 판정은 탐색적 데이터에서 나온다. 다중 비교로 인해 PROMISING 하나는 우연일 수 있다.`,
  },

  "arms.col.n_units": {
    title: "units (unit 수)",
    body: `판정에 들어간 **작은 객체 unit**(area0 < 1%) 수. \`n_units = len(S)\`, S = VK가 아닌 unit 중 \`small\`.

- DEV + REP 전체가 끝났고 모든 unit이 같은 기계 SGGL과 짝지어졌으면 2026-10-14 기준 57이다(전체 64 unit 중 작은 객체).
- 이보다 작으면 아직 실행 중이거나 짝 SGGL이 없는 unit이 있다는 뜻이다. 그런 arm의 지표는 다른 arm과 같은 unit 집합이 아니므로 직접 비교하면 안 된다.
- 작지 않은 unit(area0 ≥ 1%)은 판정에서 빠지지만 산점도와 \`all_units_dJF\`(표에는 없음)에는 들어간다.`,
  },

  "arms.col.n_videos": {
    title: "videos (영상 수)",
    body: `작은 객체 unit이 속한 서로 다른 영상 수(\`S.video.nunique()\`). 2026-10-14 기준 완료 arm은 34.

영상이 bootstrap·LOVO·'상승 영상 수'의 단위다. 한 영상에 객체가 여러 개(예: miami-surf obj2/4/5)여도 영상 단위로 묶어서 본다. unit 수보다 이 숫자가 실질적인 표본 크기에 가깝다.`,
  },

  "arms.col.dJF": {
    title: "ΔJ&F (dJF)",
    body: `**계산:** unit마다 \`d = mean(arm J&F) − mean(SGGL J&F)\` (frame 1..n−1, frame 0은 prompt라 제외). 그다음 **작은 객체 unit들의 단순 평균**: \`dJF = mean(d)\`.

- J&F = (J + F) / 2. J = 마스크 IoU, F = 경계 F-measure. GT가 빈 frame에서 출력도 비면 J = 1.
- unit 가중 평균이다. 객체가 많은 영상이 더 큰 비중을 가진다(CI와 LOVO는 영상 단위).

${BASE}

**읽는 법:**
- +0.005가 PROMISING 최소 조건. +0.01 정도면 이 데이터에서 큰 편이다. 동시 부재 frame을 모두 완벽히 살려도 +0.028이 한계(headroom)다.
- 평균은 소수 영상에 몰린다. 예: 2026-10-14 기준 FLOWVEL의 +0.0130은 drone(+0.126), kite-surf(+0.132), skate-park(+0.086) 영상이 대부분을 만든다(\`causal_report.json\` per_video). 그래서 아래 영상별 막대와 LOVO를 꼭 같이 본다.
- 판정은 이 값만으로 나지 않는다(verdict 열 설명 참고).`,
  },

  "arms.col.ci": {
    title: "CI lo / CI hi (영상 군집 95% 신뢰구간)",
    body: `\`causal_report.json\` \`CI_video\` = [하한, 상한].

**계산:** \`rfeas.risk.cluster_bootstrap(S, "video", mean(d), n=2000, seed=0)\`. 영상 34개를 **영상 단위로** 복원추출(같은 영상의 unit은 통째로 함께 뽑힘) → 매번 작은 객체 unit의 평균 d → 2000개 값의 2.5 / 97.5 백분위.

**왜 영상 군집인가:** 같은 영상의 객체들은 같은 카메라·장면 문제를 공유해 독립이 아니다. unit 단위로 뽑으면 CI가 지나치게 좁아진다.

**읽는 법:**
- 하한 > 0: 영상 표본을 바꿔도 평균이 양수로 남을 가능성이 높다. M7–M8에서 배포 가능한 arm으로는 FLOWVEL이 처음 이 조건을 만족했다(2026-10-14 기준, \`STAGE2_CAUSAL_RESULTS.md\` §3).
- 하한 ≤ 0 ≤ 상한: 효과 방향도 확정하지 못한다.
- 상한이 크고 하한이 0 근처: 몇 영상의 큰 이득이 평균을 끌어올리는 전형적 모양.

**주의:** 이 CI는 개별 arm의 것이고 **다중 비교 보정이 없다**. 10개 이상의 arm 중 하나가 하한 > 0이 되는 것은 우연으로도 일어날 수 있다. 또 같은 데이터를 이미 본 탐색 단계이므로 확증이 아니다. 두 arm의 '차이'에 대한 CI는 이 표에 없다(비교 섹션 설명 참고).`,
  },

  "arms.col.dJ": {
    title: "ΔJ (영역 IoU 차이)",
    body: `unit마다 \`mean(arm J) − mean(SGGL J)\`를 구해 작은 객체 unit 평균(\`dJ\`). J = 마스크 IoU(영역 정확도).

ΔJ&F = (ΔJ + ΔF) / 2이다. ΔJ와 ΔF를 나눠 보면 효과의 성격을 알 수 있다.
- 대상을 아예 놓치던 frame을 되찾으면(J 0 → 높음) ΔJ와 ΔF가 함께 오른다.
- 경계만 다듬는 효과는 ΔF 쪽이 크다.
- crop 제어 arm(FLOWVEL, M3vel)은 둘이 비슷하게 오르는 편이며(2026-10-14 기준 FLOWVEL ΔJ +0.0106 / ΔF +0.0154), 이는 '재획득' 형태의 이득과 맞다. 작은 객체에서는 F가 몇 픽셀 차이에 민감해 ΔF가 더 크게 나오기 쉽다.`,
  },

  "arms.col.dF": {
    title: "ΔF (경계 정확도 차이)",
    body: `unit마다 \`mean(arm F) − mean(SGGL F)\`의 작은 객체 unit 평균(\`dF\`). F = 경계 F-measure(예측 윤곽과 GT 윤곽의 정밀도·재현율 조화평균, 허용 거리 내 일치).

- 작은 객체는 윤곽 길이가 짧아 몇 픽셀 차이로 F가 크게 흔들린다. 그래서 ΔF가 ΔJ보다 크거나 부호가 다를 수 있다.
- ΔJ ≈ 0인데 ΔF만 움직이면 '같은 대상을 잡되 경계가 조금 다르다' 수준의 변화로, 연구 질문(놓친 대상 회수)과는 거리가 있다.`,
  },

  "arms.col.up": {
    title: "vid ↑0.01 (상승 영상 수)",
    body: `\`videos_up_0.01\`: 영상별 평균 d(그 영상 작은 객체 unit들의 평균)가 **+0.01보다 큰** 영상 수.

- PROMISING 조건: **≥ 3개**. 이득이 한두 영상의 우연한 사건이 아님을 요구한다.
- REJECTED 조건의 일부: ≤ 2개이면서 LOVO 최솟값 ≤ 0.
- 예(2026-10-14 기준): FLOWVEL 5개(drone, kite-surf, skate-park, sheep, lady-running), M3vel 2개(skate-park, sheep) → M3vel은 평균이 +0.0046이어도 이 조건 때문에 INCONCLUSIVE.
- 대부분 영상은 |d| < 0.01이다. SGGL이 이미 잘 따라가는 영상에서는 crop 정책이 거의 영향을 주지 않기 때문이다.`,
  },

  "arms.col.down": {
    title: "vid ↓0.01 (하락 영상 수)",
    body: `\`videos_down_0.01\`: 영상별 평균 d가 **−0.01보다 작은** 영상 수.

판정 규칙에 직접 들어가지는 않지만 해로움의 폭을 본다. 상승 5 / 하락 2와 상승 2 / 하락 3은 평균이 같아도 전혀 다른 arm이다. 하락 영상이 많으면 '어떤 영상에서는 돕고 어떤 영상에서는 해치는' 정책이며, 그 경우 평균 양수는 영상 구성에 따라 쉽게 뒤집힌다. 예(2026-10-14 기준): RESC_CPFR은 하락 3개(disc-jockey −0.33 등).`,
  },

  "arms.col.lovo_min": {
    title: "LOVO min (영상 하나씩 빼기 최솟값)",
    body: `**계산:** 34개 영상 각각에 대해 그 영상의 unit을 모두 빼고 나머지 작은 객체 unit의 평균 d를 구한다(34개 값). 그중 **최솟값**이 \`lovo_min\`.

- 보통 가장 큰 이득을 낸 영상을 뺐을 때가 최솟값이 된다. 즉 "가장 좋은 영상 하나가 없었어도 평균이 양수인가?"를 묻는다.
- PROMISING 조건: > 0. REJECTED 조건의 일부: ≤ 0 (상승 영상 ≤ 2와 함께).
- 예(2026-10-14 기준): FLOWVEL +0.0067 — drone을 빼도 양수. RESC_OCLR_ADMIT −0.0019 — 이득이 drone 한 영상에 의존.

**주의:** 하나만 뺀다. 이득이 두세 영상에 몰려 있으면 LOVO는 통과해도 '두 개 빼기'에서는 무너질 수 있다. 영상별 막대 그래프로 직접 확인할 것.`,
  },

  "arms.col.without_miami": {
    title: "w/o miami (miami-surf 제외 ΔJ&F)",
    body: `\`without_miami\`: \`davis/miami-surf\` 영상의 unit을 모두 빼고 구한 작은 객체 unit 평균 d.

**왜 이 영상만 따로 빼나:** miami-surf는 작은 서퍼 여러 명(obj2/4/5 등)이 서로 비슷하게 생긴 영상이라, M7부터 신원 전환·방해물 오인으로 큰 변동을 반복적으로 만들었다. M7 pilot 규칙에 이 조건이 고정돼 있고 M8도 그대로 쓴다(사전 등록).

- PROMISING 조건: > 0.
- 이 값이 dJF보다 크면 miami-surf에서 손해를 봤다는 뜻이다. 예(2026-10-14 기준): RESC_OCLR은 전체 −0.0001, miami 제외 +0.0016 — 손실이 miami-surf obj5(−0.058)에 있다.
- rescue 계열에서 이 차이가 크게 나는 것은 '비슷한 방해물을 대상으로 잘못 회수'하는 실패의 신호다.`,
  },

  "arms.col.tiny_dJF": {
    title: "tiny ΔJ&F (극소 객체 평균)",
    body: `\`tiny_dJF\`: 작은 객체 unit 중 **area0 < 0.1%**인 극소(tiny) unit만의 평균 d. 개수는 \`n_tiny\`(2026-10-14 기준 7개).

- 판정 규칙에는 들어가지 않는다(서술용).
- 극소 객체는 마스크 중심이 몇 픽셀만 흔들려도 crop이 크게 움직이고, 놓치면 J가 0으로 떨어진다. crop 정책의 효과가 가장 크게 나타나는 집단이다.
- 단 7개 unit의 평균이라 한 unit이 값을 좌우한다. 예: 0.0이면 대개 '아무것도 바뀌지 않음'(rescue가 극소 unit에서 한 번도 발동하지 않음)이다.
- 이 값을 근거로 "극소 객체에 특화된 방법"이라고 주장하려면 별도 사전 등록과 더 많은 극소 unit이 필요하다.`,
  },

  "arms.col.worst": {
    title: "worst unit (최악 unit Δ)",
    body: `\`worst_unit\` = [unit 이름, d]. 작은 객체 unit 중 d가 가장 작은 unit. 표에는 d만, 상세 카드·산점도에서 이름을 확인할 수 있다.

- PROMISING 조건: **≥ −0.05**. 평균이 좋아도 한 객체를 크게 망가뜨리는 방법은 배포할 수 없다는 뜻이다.
- −0.05보다 훨씬 나쁜 값(−0.1, −0.3)은 대부분 **신원 전환**이다. rescue나 메모리 기록이 비슷한 다른 객체를 대상으로 잡고, 이후 그 객체를 계속 따라간다. 예(2026-10-14 기준): RESC_CPFR disc-jockey obj3 −0.330, RESC_OCLR_ADMIT miami-surf obj4 −0.116.
- ORACLE arm은 GT가 J > 0.5일 때만 후보를 고르므로 보통 0.000(해친 unit이 없음)이다.
- 클릭해 Unit 탐색 페이지에서 frame별 궤적을 보면 원인이 바로 보인다.`,
  },

  "arms.col.best": {
    title: "best unit (최고 unit Δ)",
    body: `\`best_unit\` = [unit 이름, d]. 작은 객체 unit 중 d가 가장 큰 unit.

- 2026-10-14 기준 대부분 arm에서 **drone obj1**이다. 이 unit은 대상이 crop 밖으로 나가 G도 놓치는 '위치형' 실패 사례(\`FAILURE_EPISODE_ANALYSIS.md\`)로, 재획득에 성공하면 d가 +0.2~+0.5까지 나온다.
- best가 dJF의 수십 배면 평균이 이 한 unit에 크게 기대고 있다는 뜻이다. 57 unit 평균에서 +0.38짜리 unit 하나는 평균을 약 +0.0067 올린다.
- 같은 best unit을 여러 arm이 공유하면, 그 arm들의 차이는 결국 '그 한 사건을 얼마나 일찍·오래 회수했나'로 갈린다.`,
  },

  "arms.col.rescued": {
    title: "rescued (회수 frame 수)",
    body: `\`rescued_frames\`: rollout 중 rescue가 **실제로 출력을 바꾼** frame 수의 합(모든 unit, VK 제외. 작은 객체로 한정하지 않음).

**rescue가 일어나는 조건 (\`scripts/m8.py\` \`run_arm\`):**
1. 트리거: 그 frame에서 G와 L의 object score가 모두 ≤ 0 (동시 부재).
2. spec의 \`gens\` 생성기를 저장 없이 실행해 후보를 만든다.
3. ranker가 최고 후보를 고르고, 점수가 \`thr\`을 넘고 마스크가 비어 있지 않으면 → 출력 교체 + crop 재배치 (+ ADMIT이면 메모리 기록).

**읽는 법:**
- 운동 arm(FLOWVEL, M3vel, ablation)은 생성기가 없어 항상 0이다. 이들은 놓친 뒤 찾는 대신 놓치지 않게 만든다.
- rescued가 많다고 좋은 것이 아니다. 잘못된 대상을 회수하면 거짓 재획득과 최악 unit이 함께 나빠진다(예: RESC_CPFR 22 frame 회수, 거짓 재획득 +25, 2026-10-14 기준).
- ADMIT arm은 rescued가 오히려 적을 수 있다. 메모리에 들어간 rescue 덕에 SAM 2가 다음 frame을 스스로 잡아 트리거 자체가 줄기 때문이다(예: ORACLE_POOL 61 vs ORACLE_POOL_ADMIT 14).
- '회수 frame'은 GT를 보지 않는다. 회수가 맞았는지는 J로만 알 수 있다.`,
  },

  "arms.col.fp": {
    title: "오재획득 arm (거짓 재획득 frame, arm)",
    body: `\`false_reacq_frames\`: **GT가 비어 있는 frame에서 arm 출력이 비어 있지 않은** frame 수의 합(모든 unit, VK 제외).

정확한 정의(\`m8_analysis.py\`): \`gt_empty & (J < 1)\` — GT가 비었을 때 출력도 비면 J = 1이므로, J < 1은 '대상이 없는데 무언가를 출력했다'는 뜻이다.

- 사전 등록의 \`also_reported\` 항목: "false re-acquisition (output non-empty on GT-empty frames)".
- 이름은 '재획득'이지만 rescue로 생긴 것만이 아니라 SGGL 자체의 거짓 양성도 포함한다. 그래서 반드시 옆의 **SGGL 값과의 차이**로 읽는다.
- 대상이 진짜로 사라진 frame(동시 부재 486 중 355개가 GT 빈 frame)에서 rescue가 비슷한 방해물을 '대상'으로 내놓으면 이 값이 늘어난다.
- 예(2026-10-14 기준): FLOWVEL 93 = SGGL 93(증가 없음), RESC_OCLR 99, RESC_CPFR 118.`,
  },

  "arms.col.fpb": {
    title: "오재획득 SGGL (같은 기계 SGGL의 거짓 재획득)",
    body: `\`false_reacq_frames_SGGL\`: 같은 unit들에 대해 짝지어진 SGGL 궤적에서 \`gt_empty & (J_SGGL < 1)\`인 frame 수.

- 모든 arm이 같은 unit 집합·같은 기계 SGGL을 쓰면 이 값은 arm마다 같다(2026-10-14 기준 완료 arm은 93).
- 다르면 unit 집합이나 기계 구성이 다르다는 뜻이므로(실행 중인 arm, ORACLE의 Ubuntu+RunPod 혼합) arm끼리 거짓 재획득 절대값을 비교하지 말고 **각자의 SGGL과의 차이**만 비교한다.
- arm − SGGL > 0 = 그 arm이 대상 없는 frame에 무언가를 더 출력했다. rescue 계열의 주된 위험 지표다.`,
  },

  "arms.col.lat": {
    title: "latency s (arm 지연, s/frame)",
    body: `\`latency_s_mean\`: unit별 frame당 평균 지연(\`latency_s\`)의 unit 평균. 한 frame의 G/L 예측 + (광류) + (rescue 생성기·ranker)까지, \`torch.cuda.synchronize()\` 후 측정한다. 디스크 I/O·GT 평가는 포함하지 않는다.

**주의 (중요):**
- RunPod 4090에서 worker 3–4개가 같은 GPU를 **동시에** 쓰며 측정했다. GPU 경합 정도가 시간대마다 달라 arm 간 차이 수십 ms는 잡음일 수 있다. 예: FLOWVEL(광류 추가)이 SGGL보다 빠르게 찍힌 것은 측정 시간대 차이다(\`STAGE2_CAUSAL_RESULTS.md\` §4).
- 기계가 다른 arm(ORACLE: Ubuntu 2080 + RunPod)은 절대값을 비교할 수 없다.
- 깨끗한 비용 비교는 같은 기계 단독 측정이 필요하다(2080 단독: RAFT-small 광류 0.028–0.047 s/frame).
- 비용을 보는 더 안정적인 지표는 '추가 SAM 2 호출/frame'이다.`,
  },

  "arms.col.latb": {
    title: "SGGL s (같은 기계 SGGL 지연)",
    body: `\`latency_s_SGGL\`: 짝지어진 SGGL 궤적의 frame당 평균 지연. SGGL 파일에 \`latency_s\`가 있을 때만 값이 있다(Ubuntu diag 궤적에는 없을 수 있음 → 빈 칸).

arm 지연과의 차이가 '그 arm의 추가 비용' 추정치지만, 두 rollout이 다른 시간대에 같은 GPU를 다른 worker와 나눠 쓰며 측정됐으므로 차이는 거칠게만 읽는다. ORACLE처럼 기계가 섞인 arm은 이 값도 다른 arm과 다르다.`,
  },

  "arms.col.vram": {
    title: "VRAM GB (최대 GPU 메모리)",
    body: `\`peak_vram_gb\`: unit마다 \`torch.cuda.max_memory_allocated()\`(unit 시작 시 초기화)를 기록하고, 그 최댓값 / 1e9.

- PyTorch가 할당한 텐서 메모리의 최대치다. CUDA context·캐시 예약분은 빠져 있어 \`nvidia-smi\` 값보다 작다.
- 배포 대상 RTX 2080은 8 GB. 2026-10-14 기준 대부분 arm 1.3–1.45 GB, ORACLE(31개 후보 probe) 약 2.3 GB로 여유가 크다.
- RAFT-small 광류는 약 +0.1 GB(FLOWVEL 1.45 vs M3vel 1.34).`,
  },

  "arms.col.machines": {
    title: "machine (실행 기계별 unit 수)",
    body: `\`machines\`: 이 arm의 unit이 어느 기계에서 돌았는지 \`기계:unit 수\`. VK 제외, **작지 않은 unit도 포함**이라 units 열(작은 객체만)보다 클 수 있다(예: runpod:64 vs units 57).

- \`runpod\` = RunPod RTX 4090 worker → \`runpod/runs/SGGL\`과 짝.
- \`ubuntu\` = Ubuntu RTX 2080 → diag 궤적과 짝.
- 결과는 RunPod에서 sha256 검증 후 Ubuntu로 가져온다(\`SYNC_LEDGER\` / \`SYNC_STATUS\`). Ubuntu가 원본 저장소다.

${BASE}

ORACLE arm은 Ubuntu 53 + RunPod 11로 섞여 있다. 각 unit은 자기 기계의 SGGL과 짝지었으므로 Δ는 유효하지만, 지연·거짓 재획득 SGGL 값은 다른 arm과 직접 비교하지 않는다.`,
  },

  // ------------------------------------------------------------------ VK
  "arms.vk": {
    title: "VKITTI (@VK) — scene별",
    body: `**무엇인가:** 최종 후보(finalist)를 VKITTI2 Scene01/06 clone crop unit에서 돌린 결과. \`causal_report.json\`의 \`<arm>@VK\` 항목.

**사전 등록 (\`preregistration_vk.json\`, 2026-10-14 19:25 KST, VK 출력 전):**
- finalist = FLOWVEL(DEV+REP에서 유일한 PROMISING). 참조 = SGGL(같은 기계 기준선), M3vel(M7 도전자).
- 기계 = RunPod RTX 4090(세 arm 모두).
- 규칙 = **비회귀**: VK ΔJ&F ≥ −0.005 (M7 최종 규칙의 비회귀 조항). 이득은 요구하지 않는다.

**열:**
- verdict: VK 항목에는 pilot 판정이 없다. 대신 \`non_regression_pass\`(ΔJ&F ≥ −0.005)를 '비회귀 통과/실패'로 표시한다.
- ΔJ&F: VK unit들의 평균 d(같은 기계 SGGL 대비). 작은 객체 필터를 쓰지 않는다.
- per_scene: scene별 평균 d (\`V.groupby("scene").d.mean()\`).
- 화면에 없는 필드: \`CI_object\` — **객체 단위** bootstrap 2000회(영상이 2개뿐이라 영상 군집이 불가능), \`units_up_0.01 / units_down_0.01\`, \`worst_unit\`.

**읽는 법·주의:**
- Scene01/06은 M2–M4 개발 scene이었다 → **서술적**이며 독립 확인이 아니다.
- VK에는 crop 이탈 사건이 거의 없어 crop 제어 arm의 이득이 0 근처로 나오는 것이 자연스럽다(2026-10-14 기준 FLOWVEL +0.0002). 목적은 "다른 도메인(합성 주행 영상)에서 해치지 않는가"다.
- unit 수가 arm마다 다르면 아직 실행 중이다.`,
  },

  "arms.vk.dJF": {
    title: "VK ΔJ&F",
    body: `VK unit 전체의 평균 \`d = mean(arm J&F) − mean(SGGL J&F)\`. 같은 기계(RunPod) SGGL 대비. 작은 객체 필터 없음.

- 비회귀 기준: ≥ −0.005.
- 2 scene · 수십 객체뿐이라 신뢰구간은 객체 단위 bootstrap(\`CI_object\`)으로만 낸다. 같은 차량들이 여러 frame에 걸쳐 상관돼 있어 실제 불확실성은 그보다 크다.`,
  },

  "arms.vk.per_scene": {
    title: "per_scene (scene별 ΔJ&F)",
    body: `scene(Scene01, Scene06)별 unit 평균 d. 소수점 4자리 반올림.

두 scene 부호가 같고 크기가 비슷하면 효과(또는 무효과)가 scene에 의존하지 않는다는 약한 증거다. 한 scene만 크게 음수면 그 scene 특유의 장면(차량이 서로 겹침, 터널 등)에서 정책이 실패한다는 뜻이므로 Unit 탐색에서 확인한다.`,
  },

  // ------------------------------------------------------------------ detail card
  "arms.detail.summary": {
    title: "상세 카드 요약 줄",
    body: `선택한 arm의 ΔJ&F, 영상 군집 95% CI, 작은 객체 unit 수 / 영상 수. 표의 같은 열과 정의가 같다(dJF, CI_video, n_units, n_videos).

제목 옆 "?"는 이 arm의 메커니즘·가설·선행연구·상태 설명이다. ORACLE 배지가 있으면 GT를 쓰는 진단용 상한이다.`,
  },

  "arms.spec": {
    title: "frozen spec (arms.json)",
    body: `**무엇인가:** 이 arm을 정의한 설정. \`outputs/milestone8/arms.json\`에 **첫 출력 전에** \`scripts/m8.py freeze_arm\`으로 기록된다.

- freeze 이후 같은 이름으로 다른 spec을 넣으려 하면 거부된다("arm is frozen and differs"). 출력 폴더가 이미 있으면 freeze 자체가 거부된다.
- 사전 등록 \`no_tuning\`: arm마다 사전 등록된 개정은 최대 1회, 임계값은 인과 실행 전에 고정.
- 그래서 결과를 보고 임계값이나 생성기를 바꾼 arm은 **새 이름**이 되어야 한다. 그런 '결과를 본 뒤 고른' arm은 \`why\`에 그렇게 적혀 있다(예: K_ADMIT_FLOWVEL).

**모든 arm의 공통 골격 (\`run_arm\`):** 매 frame SGGL(G + L 두 view, 자기 gate)을 그대로 돌린다 → 동시 부재 frame이면 \`gens\` 생성기로 rescue 시도 → \`center\` 규칙으로 다음 frame의 L crop 중심을 정한다. \`gens\`가 비어 있고 \`center\`가 \`out\`이면 SGGL과 같다.

각 필드 이름 옆 "?"에 자세한 설명이 있다.`,
  },

  "arms.spec.gens": {
    title: "gens (rescue 생성기 목록)",
    body: `동시 부재 frame(G, L object score 모두 ≤ 0)에서 **저장하지 않고** 실행해 후보 마스크를 만드는 Stage 1 생성기들. 빈 목록 \`[]\` = rescue 없음(운동·기준선 arm).

| 생성기 | 하는 일 (\`preregistration.json\` stage1.generators) |
|---|---|
| \`A_cos#k\` | frame-0 prototype과 G stride-16 특징 코사인 → 상위 3 peak → G 메모리에 점 prompt (#0–2 = 1–3위 peak) |
| \`B_att#k\` | 대조 token attention(전경 8 + 배경 32 k-means token, τ 0.1) → 상위 3 → 점 prompt |
| \`H_s8#k\` | stride-8 FPN 특징으로 같은 prototype 매칭 |
| \`G_neg\` | B 1위 양성 점 + 배경으로 판정된 A peak에 음성 점 |
| \`D_ref\` | frame-0 메모리만으로 B 1위 점 prompt |
| \`A_cos_crop\` / \`B_att_crop\` | A / B 1위 peak 위치에 L 메모리 crop을 다시 놓음 |
| \`E_beam1/2\` | 마지막 확신 중심 + 1× / 2× 속도 × 경과 frame에 L crop |
| \`C_flowcrop\` / \`C_flowbox\` | RAFT-small 광류로 운반한 box → 그 중심에 L crop / G에 box prompt |
| \`F_oclr{8,4}_{G,L}\` | occupancy crop: 대상 긴 변이 stride-16 cell 8개(또는 4개)가 되도록 자른 평범한 crop, 광류 중심, G 또는 L 메모리 |
| \`F_cpfr{8,4}_{G,L}\` | 같은 배율의 문맥 보존 foveation(분리형 비균일 확대, 시야 전체 유지) |
| \`*_pt\` | 같은 view + 광류 중심에 양성 점 1개 (amend1) |
| \`Gp\` / \`Lp\` | M7 보호 메모리 probe (참조용) |

**비용:** 후보 하나당 SAM 2 호출 1회 이상(동시 부재 frame에서만). 그래서 목록이 길수록(ORACLE 31개) 추가 호출과 지연이 커진다.

**주의:** Stage 1에서 사전 등록된 진출 조건(\`advance_to_causal\`: recall ≥ 0.25 등)을 통과한 계열은 **없었다**. RESC_* arm은 그 대신 '진단적 인과 실행'으로 돌린 것이다(\`STAGE2_CAUSAL_RESULTS.md\` 사전 등록과 다른 점).`,
  },

  "arms.spec.ranker": {
    title: "ranker (kind, thr) — 후보 선택 규칙",
    body: `후보가 여러 개일 때 어느 것을 출력할지, 그리고 **아무것도 출력하지 않을지(기권)**를 정한다. \`scripts/m8.py\` \`class Rank\`. 최고 점수 후보의 점수가 \`thr\`보다 **커야** 채택된다(\`s > thr\`). 채택돼도 마스크가 비어 있으면 rescue하지 않는다.

| kind | 점수 | 학습 | 비고 |
|---|---|---|---|
| \`score\` (R0) | SAM 2 object score 그대로 | 없음 | 현재 RESC_* / K_* arm이 사용. \`thr: 0.0\` = "SAM 2가 존재한다고 말하면 채택" |
| \`r1\` (R1) | score > 0인 후보만, \`cos_fg − cos_bg\`(frame-0 전경/배경 prototype 코사인 차) | 없음 | \`src/spatial/ranker.py r1\` |
| \`lr\` (R2) | 로지스틱 회귀(score, iou, 면적 비, 코사인, 광류 거리 등 스칼라) | REP에 fit | \`outputs/milestone8/rankers/<name>.json\` 로드 |
| \`set\` (R3) | 후보 token + frame-0 token에 대한 set-attention, '기권' slot 대비 margin | 영상 fold 2개 모델 평균 | \`rankers/<name>.pt\`, KeepTrack류 |
| \`oracle\` | GT J가 최대인 후보, **J > 0.5일 때만** | — | **진단 전용, 배포 불가** |

**thr:** 채택 임계값. 사전 등록 규칙은 "fit split의 동시 부재 frame에서 GT 빈 frame 채택률 ≤ 0.10을 만족하면서 순이득(net_dJ_frames)이 최대인 값"이지만, 현재 \`score\` arm은 학습 없는 고정값 0.0을 쓴다. 학습된 파라미터가 없으므로 DEV + REP 전체에서 평가할 수 있다(학습 ranker는 fit하지 않은 split에서만 평가해야 함).

**읽는 법:** \`score > 0\`은 SAM 2 자신의 존재 판단에 기댄다. Stage 1에서 점 prompt 후보는 GT가 비어도 거의 항상 score > 0이었다(amend1). 그래서 점 prompt 생성기에 \`score\` ranker를 쓰면 거짓 재획득이 폭증한다 — RESC_* arm이 prompt 없는 \`F_oclr8_L\` / \`F_cpfr8_L\`만 쓰는 이유다.`,
  },

  "arms.spec.center": {
    title: "center (crop 중심 규칙)",
    body: `다음 frame에서 L crop(정사각, 한 변 = prompt box의 4배)을 어디에 놓을지. \`scripts/m8.py run_arm\` 끝부분. 아래에서 '출력 중심' = arm 출력 마스크의 bounding box 중심, v = RAFT-small 광류의 **대상 box 중앙 50% 영역 median 변위**(t−1 → t), cam = **대상 box 2배 바깥**의 median 광류(카메라 운동).

| 값 | 출력이 있을 때 | 출력이 비었을 때 | 쓰는 arm |
|---|---|---|---|
| \`out\` | 출력 중심 | 제자리 (crop 고정) | SGGL, RESC_*, ORACLE_* |
| \`m3vel\` | 출력 중심 + 0.5 × (이번 중심 − 직전 중심) | 제자리, 속도 이력 초기화 | M3vel, K_OCLR_M3 |
| \`flowvel\` | 출력 중심 + 0.5 × v | 광류로 운반한 box 중심 + 0.5 × v (계속 따라감) | FLOWVEL, K_OCLR_FLOWVEL, K_ADMIT_FLOWVEL |
| \`fv_present\` | 출력 중심 + 0.5 × v | 제자리 | FV_PRESENT (ablation) |
| \`flowcarry\` | 출력 중심 | 광류로 운반한 box 중심 (속도 선행 없음) | FLOWCARRY (ablation) |
| \`camcarry\` | 출력 중심 | 현재 crop 중심 + cam (카메라 운동만큼 이동) | CAMCARRY (ablation) |

- '광류로 운반한 box'(\`motion.Track.fbox\`): 출력이 있을 때마다 출력 box로 다시 고정되고, 출력이 없으면 매 frame 그 box 안의 median 광류만큼 이동한다.
- 0.5는 감쇠(damp) 계수다. M7의 M3-vel 값을 그대로 썼고 M8에서 조정하지 않았다.
- 광류가 필요한 center(\`flowvel/fv_present/flowcarry/camcarry\`)는 매 frame RAFT-small을 1회 돌린다(긴 변 640). SAM 2 호출은 늘지 않는다.

**ablation 설계:** FLOWVEL = '속도 선행'(present 때 +0.5v) + '부재 중 추적'(absent 때 따라감). FV_PRESENT는 앞의 것만, FLOWCARRY는 뒤의 것만(대상 광류), CAMCARRY는 뒤의 것을 카메라 운동으로 바꾼 것이다.`,
  },

  "arms.spec.admit": {
    title: "admit (rescue를 메모리에 기록)",
    body: `\`true\`면 rescue로 채택된 후보의 SAM 2 출력을 그 후보를 만든 상태(state)의 메모리(\`non_cond_frame_outputs[t]\`)에 써 넣는다. 사전 등록 결합표의 D2/K8 "rescue + 메모리 기록".

- 어느 상태에 쓰나: \`A_cos#\`, \`B_att#\`, \`H_s8#\`, \`G_neg\`, \`D_ref\`, \`C_flowbox\`, 이름이 \`_G\`로 끝나는 생성기 → G 상태. 나머지(\`F_*_L\`, crop 계열 등) → L 상태.
- 없거나 \`false\` = 출력만 바꾸고 메모리는 SAM 2가 원래 만든 것 그대로.

**왜 중요한가:** 출력만 바꾸면 SAM 2 메모리는 여전히 '대상 없음'이라 다음 frame에서 다시 놓치기 쉽다. 기록하면 추적이 이어진다. 대신 **잘못된 rescue도 기록**되어 이후 계속 엉뚱한 객체를 따라가는 신원 전환(장기 오염)이 생긴다.
- 예(2026-10-14 기준): RESC_OCLR_ADMIT − RESC_OCLR ≈ +0.0047이지만 최악 unit miami-surf obj4 −0.116.
- 이것은 DAM4SAM, SAMURAI, SAM2Long, SENTRY 등 '검증 후 기록' 계열이 다루는 문제와 같다(\`PRIOR_ART_AND_NOVELTY_AUDIT.md\`).`,
  },

  "arms.spec.diagnostic": {
    title: "diagnostic (진단 전용)",
    body: `\`true\` = GT를 쓰는 진단용 arm이다(현재 \`ranker.kind = oracle\`인 ORACLE_POOL, ORACLE_POOL_ADMIT). 대시보드에서 ORACLE 배지로 표시된다.

- 배포할 수 없다. 실제 추적 중에는 GT가 없다.
- 용도: "후보 생성기 풀이 이미 정답을 포함하고 있다면, 완벽한 선택기로 인과 rollout에서 얼마나 얻을 수 있나" = rescue 경로의 **상한**.
- 이 arm에 찍힌 verdict, CI, 순위는 다른 arm과 경쟁하는 값이 아니다. 배포 가능한 arm이 이 상한에 얼마나 가까운지 보는 기준으로만 쓴다.`,
  },

  "arms.spec.why": {
    title: "why (고정 시점의 근거)",
    body: `freeze 시점에 적은 이 arm의 목적·가설·비교 대상. 결과를 보기 전에 쓴 문장이므로, 나중에 결과를 해석할 때 "원래 무엇을 시험하려 했나"의 기록이다.

- '시너지 시험 vs X, Y'라고 적혀 있으면 그 arm들과 같은 기계에서 비교해야 한다(아래 arm 대 arm 비교).
- 'post-hoc' / 'AFTER seeing'이 적혀 있으면 다른 arm 결과를 본 뒤 고른 것이다 → 확증적 해석 불가, 탐색적으로만 읽는다.
- Stage 1 재생 수치(예: "23 rescued / 5 FP")는 저장 없는 진단에서의 값이며, 인과 rollout 결과와 다를 수 있다.`,
  },

  "arms.spec.frozen_kst": {
    title: "frozen_kst (고정 시각, KST)",
    body: `\`freeze_arm\`이 arms.json에 기록한 시각(한국 표준시). 이 시각 이전에는 이 arm의 출력이 없었음이 보장된다(출력 폴더가 있으면 freeze가 거부됨).

다른 arm의 결과가 나온 시각과 비교하면 '사전 고정'인지 '결과를 본 뒤'인지 알 수 있다. 예: FLOWVEL·M3vel은 16:14(M8 첫 출력 전), FLOWVEL ablation은 19:08(FLOWVEL PROMISING 확인 후, 하지만 ablation 자체의 출력 전), K_ADMIT_FLOWVEL은 19:43(RESC_OCLR_ADMIT, FLOWVEL 결과를 본 뒤).`,
  },

  "arms.detail.cost": {
    title: "지연 시간 · 추가 SAM 2 호출 · VRAM",
    body: `**latency arm (s/frame):** \`latency_s_mean\`. 아래 'SGGL'은 같은 기계 SGGL의 \`latency_s_SGGL\`, 괄호는 차이. 4090 한 장을 worker 3–4개가 동시에 쓰며 잰 값이라 수십 ms 차이는 GPU 경합 잡음일 수 있다(광류를 더 하는 FLOWVEL이 SGGL보다 빠르게 찍히는 이유).

**추가 SAM2 호출/frame:** \`extra_sam2_calls_per_frame = mean(calls) − 2\`. 매 frame G와 L 두 번이 기본(2)이고, 동시 부재 frame에서 rescue 후보 수만큼 더한다. unit별 평균의 평균.
- 운동 arm: 0 (광류는 SAM 2 호출이 아님).
- RESC_* (생성기 1개): 약 0.1 — 동시 부재 frame에서만 1회.
- ORACLE (31개 후보): 약 3.
- GPU 경합에 흔들리지 않으므로 비용 비교에서는 지연보다 이 값이 더 믿을 만하다.

**peak VRAM:** \`torch.cuda.max_memory_allocated()\`의 unit 최댓값. 배포 대상 RTX 2080(8 GB)에 대한 여유를 본다.

**연구적 의미:** "더 많이 계산해서 좋아졌다"는 설명을 배제하는 데 쓴다. FLOWVEL은 SAM 2 호출이 늘지 않으므로 이득이 추가 SAM 2 연산에서 온 것이 아니다(\`NON_ATTENTION_MECHANISM_COMPARISON.md\` §5).`,
  },

  "arms.detail.counts": {
    title: "rescue / 오재획득 (arm vs SGGL) 막대",
    body: `**막대 두 줄:**
- rescued 프레임: arm이 rescue로 출력을 바꾼 frame 수(\`rescued_frames\`). SGGL은 rescue가 없어 항상 0으로 그린다.
- 오재획득 프레임: GT가 빈 frame에서 출력이 비어 있지 않은 frame 수 — arm(\`false_reacq_frames\`) vs 같은 기계 SGGL(\`false_reacq_frames_SGGL\`).

모두 VK 제외 **전체 unit**(작은 객체 한정 아님)의 합이다.

**아래 한 줄:**
- co-absence 프레임: 이 arm 자신의 rollout에서 G, L score가 모두 ≤ 0이었던 frame 수(\`coabs_frames\`). crop 제어가 좋아지면 이 수 자체가 줄어든다. 참고로 SGGL은 같은 기계에서 약 486(그중 GT 있음 131, 2026-10-14 기준 문서 값).
- co-absence ΔJ&F 합: 그 frame들에서의 frame별 (arm J&F − SGGL J&F) 합. 단위가 'frame 수'라 unit 평균과 다르다. 대략 "동시 부재 frame에서 몇 frame어치를 얻거나 잃었나". 단, 이 집합은 arm의 동시 부재이므로 SGGL이 잘하던 frame에서 arm이 놓친 손해가 주로 잡힌다(운동 arm은 음수가 나오기 쉬움).
- tiny unit: 극소 unit 수와 그 평균 ΔJ&F.

**좋은 모양:** rescued > 0이면서 오재획득이 SGGL과 같음. **나쁜 모양:** rescued는 많은데 오재획득도 크게 늘어남 → 방해물을 대상으로 회수(신원 전환 위험). 예(2026-10-14 기준): RESC_CPFR 22 회수 / 118 vs 93.`,
  },

  "arms.detail.perVideo": {
    title: "video별 ΔJ&F (정렬 막대)",
    body: `**무엇인가:** \`causal_report.json\` \`per_video\` — 영상마다 그 영상 **작은 객체 unit들의 평균 d**(소수 4자리). 위에서부터 큰 순서. 초록 = arm 우세, 빨강 = SGGL 우세.

**읽는 법:**
- ±0.01 바깥 막대 개수가 표의 'vid ↑0.01 / ↓0.01'이다.
- 대부분 영상은 0 근처다. SGGL이 이미 잘 추적하는 영상에서는 crop 정책을 바꿔도 출력이 거의 같다.
- 맨 위 막대 한두 개가 압도적이면 평균 ΔJ&F가 그 영상에 의존한다 → LOVO 최솟값, miami-surf 제외 값과 함께 본다.
- 같은 영상이 여러 arm에서 반복해 위에 나오면(drone, kite-surf, skate-park, sheep) 그것은 '대상이 crop 밖으로 나가는' 사건이 있는 영상이고, 연구 질문(동시 부재·crop 이탈)이 실제로 걸린 곳이다(\`FAILURE_EPISODE_ANALYSIS.md\`).
- 막대 이름을 Unit 탐색에서 찾아 frame별 궤적으로 확인하면 이득이 '재획득이 빨라짐'인지 '경계만 달라짐'인지 구분된다.

**주의:** 영상 수 34개, 이득 영상은 보통 2–5개. 몇 영상에 대한 사건 기반 결과이므로 새 데이터에서 같은 사건 유형이 얼마나 자주 나오느냐에 따라 효과가 크게 달라질 수 있다.`,
  },

  "arms.detail.scatter": {
    title: "unit별 Δ (d) vs 첫 프레임 면적 area0",
    body: `**무엇인가:** \`causal_per_unit.csv\`에서 이 arm의 unit 하나 = 점 하나. x = \`area0\`(frame 0 GT 마스크 면적 / frame 면적), y = \`d\`(unit의 arm − 같은 기계 SGGL 평균 J&F). 점 클릭 = Unit 탐색 페이지로 이동. 툴팁에 dJ, dF, 기계, split(DEV/REP), rescued, 오재획득(arm/SGGL)이 있다.

**왜 x축이 로그인가:** 면적이 0.0001(0.01%)부터 0.1 이상까지 세 자릿수 넘게 퍼져 있다. 선형 축이면 작은 객체가 모두 왼쪽 끝 한 줄에 겹친다. 연구 대상은 면적이 작은 쪽이므로 로그 축으로 그 구간을 펼친다. 눈금 1e-3 = 0.1%(tiny 경계), 1e-2 = 1%(small 경계).

**색:**
- tiny unit(area0 < 0.1%): 별도 색. 판정의 '극소' 집단이다.
- 일반 unit: 나머지. **area0 ≥ 1%인 unit도 그려지지만 판정(ΔJ&F, CI, 영상 수)에는 들어가지 않는다.** 면적 0 unit은 로그 축에 그릴 수 없어 빠진다.

**읽는 법:**
- 점 대부분이 y = 0 근처: 대부분 unit에서 arm과 SGGL 출력이 같다(트리거나 crop 변화가 거의 없음).
- 왼쪽(작은 면적)에서 위로 튀는 점 = crop 정책이 효과를 낸 곳. 아래로 크게 떨어진 점 = 신원 전환 후보 → 클릭해서 확인.
- 면적이 작을수록 d의 분산이 커지는 것이 정상이다. 작은 대상은 놓치면 J가 0이 되고, 다시 잡으면 크게 회복하기 때문이다.

**주의:** 면적은 frame 0 값이다. 영상 중간에 대상이 멀어지거나 가까워지는 것은 반영하지 않는다.`,
  },

  // ------------------------------------------------------------------ compare
  "arms.compare": {
    title: "arm 대 arm 비교 (시너지 판단)",
    body: `**무엇인가:** arm 2–3개를 골라(색 = 선택 순서) 영상별 ΔJ&F 막대와 unit별 d를 나란히 본다. 기본 선택은 PROMISING arm 앞의 두 개.

- 막대: 각 arm의 \`per_video\`(작은 객체 unit 평균), 첫 번째 arm 기준으로 정렬.
- 표: \`causal_per_unit.csv\`의 unit별 d(작지 않은 unit 포함), 그리고 **'arm k − 첫 번째 arm'** 차이. 차이가 큰 unit부터 정렬. 초록 = 뒤 arm이 더 좋음, 빨강 = 첫 arm이 더 좋음. 행 클릭 = Unit 탐색.

**두 arm의 차이는 왜 SGGL을 거치지 않아도 되나:** 같은 unit에서 같은 기계 SGGL을 빼므로 \`d_k − d_1 = arm_k − arm_1\`이다. **단, 두 arm이 그 unit을 같은 기계에서 돌렸을 때만.** 기계가 다르면(예: ORACLE의 Ubuntu unit vs FLOWVEL의 RunPod unit) 두 기계 SGGL의 차이가 섞인다. 툴팁·Unit 탐색에서 기계를 확인할 것.

**시너지 판단법 (\`CROSS_FAMILY_COMBINATIONS.md\` 사전 등록):**
1. **첫 번째로 '가장 강한 단일 구성요소'를 고르고, 두 번째로 결합 arm을 고른다.** 그러면 차이 열 = 결합 − 최강 구성요소.
2. 시너지 주장 조건: **결합 − max(구성요소들) > 0 이고, 그 차이의 영상 군집 CI가 0을 넘을 때.** 결합이 SGGL보다 좋다는 것만으로는 부족하다(좋은 구성요소 하나의 효과일 수 있음).
3. 모든 비교는 같은 기계에서 돈 arm끼리 한다(현재 K_* 와 구성요소는 모두 RunPod).
4. 이 페이지는 차이의 CI를 계산하지 않는다. 평균 차이와 unit 분포로 방향을 보고, 주장하려면 \`m8_analysis.py\`로 짝지은 영상 군집 bootstrap을 따로 돌려야 한다.

**예 (2026-10-14 기준, \`causal_report.json\`):**
- K_OCLR_FLOWVEL +0.0121 vs FLOWVEL +0.0130 → 결합 − 최강 ≈ −0.0009. rescue를 더해 얻은 것이 없다.
- K_OCLR_M3 +0.0071 vs M3vel +0.0046, RESC_OCLR −0.0001 → 평균은 +0.0025 높지만 CI 미계산이고, 최악 unit(miami-surf obj5 −0.058)은 RESC_OCLR의 실패를 그대로 물려받았다.
- 결합 arm의 최악 unit이 구성요소 중 하나와 같은 unit이면, 그 구성요소의 실패 방식이 결합에서도 살아 있다는 뜻이다.

**주의:** 결합을 결과를 본 뒤 고르면(K_ADMIT_FLOWVEL) 그 비교는 탐색적이다. 또 여러 쌍을 비교할수록 우연한 '시너지'가 나올 확률이 커진다.`,
  },

  "arms.compare.diff": {
    title: "arm k − 첫 번째 arm (unit별 차이)",
    body: `unit마다 \`d(arm k) − d(첫 번째 arm)\`. 두 arm이 같은 기계에서 그 unit을 돌렸다면 = 두 arm 출력의 평균 J&F 차이 그 자체다(SGGL 항이 상쇄).

- 시너지 판단은 첫 번째 = 최강 단일 구성요소, k = 결합 arm으로 놓고 이 열을 본다. 양수 unit이 여러 영상에 퍼져 있어야 의미가 있다.
- |차이| < 0.0001은 색을 칠하지 않는다(두 arm 출력이 사실상 같음 — 트리거가 한 번도 안 걸린 unit 등).
- 한 unit의 큰 차이는 대개 '한 사건(재획득 시점, 신원 전환)의 차이'다. 클릭해서 frame별로 확인할 것.
- 빈 값(—)은 한쪽 arm에 그 unit이 아직 없다는 뜻이다(실행 중).`,
  },

  // ------------------------------------------------------------------ per-arm entries
  "arms.arm.SGGL": {
    title: "SGGL — 현행(incumbent) 기준선",
    body: `**메커니즘:** frozen SAM 2.1 Hiera-T로 두 상태를 동시에 돌린다.
- G = 전체 frame, P5 prompt 메모리.
- L = 고정 크기 정사각 crop(한 변 = prompt box의 4배). 중심은 출력 마스크 중심을 따라간다(\`center: out\`). 출력이 비면 crop은 제자리.
- 자기 gate: L의 object score > 0, 마스크 비어 있지 않음, 내부 crop 경계에 닿지 않음 → L 출력, 아니면 G 출력.

**역할:** M6에서 채택, M7 최종 결정 KEEP_SGGL. M8의 모든 arm은 매 frame SGGL을 그대로 돌리고 그 위에 rescue나 crop 중심 규칙만 바꾼다. arms.json의 SGGL 항목은 "같은 기계 기준선"을 각 worker에서 돌리기 위한 것이다(RunPod: \`runpod/runs/SGGL\`; Ubuntu에서는 diag 궤적과 동일).

**계열:** 기준선 (crop view + 전체 frame view 결합).

**선행연구 근접도:** VOTS2025 참가 15팀 중 80%가 SAM 2 기반이었지만 crop view와 전체 frame SAM 2 view를 결합한 참가자는 없었다(\`PRIOR_ART_AND_NOVELTY_AUDIT.md\`). crop 탐색창 자체는 Siamese 추적기 계열의 오래된 관행이다.

**알려진 약점:** 두 view가 동시에 대상을 놓치는 **동시 부재**(486 frame, GT 있음 131)가 가장 큰 실패 범주다. 대상이 crop 밖으로 나가면 crop은 출력이 없으니 움직이지 않고, G는 작은 대상을 놓친다.

**상태 (2026-10-14 기준):** 현행. 표에는 행이 없다 — 모든 Δ의 0점이다.`,
  },

  "arms.arm.M3vel": {
    title: "M3vel — M7 도전자 (감쇠 속도 crop 중심)",
    body: `**메커니즘:** \`center: m3vel\`. 다음 crop 중심 = 출력 중심 + 0.5 × (이번 출력 중심 − 직전 출력 중심). 출력이 비면 crop은 제자리, 속도 이력은 초기화. rescue 없음, SAM 2 추가 호출 없음, 광류 없음.

**가설:** 대상이 빠르게 움직이면 SGGL의 crop은 한 frame 늦게 따라가 대상이 crop 가장자리로 밀리고 결국 빠져나간다. 속도만큼 미리 옮겨 두면 crop 이탈이 줄고, 재등장 시 더 빨리 재획득한다.

**계열:** C/E (운동, 마스크 중심 차분 기반). M7에서 M3-vel(damp 0.5)을 그대로 가져온 **같은 기계 대조군**이다.

**선행연구 근접도:** 높음. 운동 모델로 탐색 영역을 옮기는 것은 DaSiamRPN, KeepTrack(탐색 영역), SAMURAI(Kalman)와 같은 연산이다.

**상태 (2026-10-14 기준):** ΔJ&F +0.0046, CI [−0.0000, +0.0109], 상승 영상 2개(skate-park, sheep) → **INCONCLUSIVE**. M7 Vast의 +0.0046과 RunPod에서 같은 값으로 재현됐다(기계 간 재현성은 좋음). 이득은 '재등장 시 빠른 재획득'. 약점: 마스크 중심 차분이라 극소 마스크의 중심 흔들림을 속도로 오인할 수 있고, 출력이 빈 동안에는 아무것도 하지 않는다. FLOWVEL은 바로 이 두 점을 바꾼 것이다.

${STATUS_NOTE}`,
  },

  "arms.arm.FLOWVEL": {
    title: "FLOWVEL — 광류 기반 crop 제어 (현재 유일한 배포 가능 PROMISING)",
    body: `**메커니즘:** \`center: flowvel\`. 매 frame RAFT-small 광류(t−1 → t, 긴 변 640)를 계산한다.
- v = 대상 box 중앙 50% 영역의 median 광류 변위.
- 출력이 있으면: 다음 중심 = 출력 중심 + 0.5 × v.
- 출력이 비면: 광류로 운반한 box(출력이 없는 동안 매 frame v만큼 이동)의 중심 + 0.5 × v → **대상이 안 보이는 동안에도 crop이 대상 쪽으로 계속 움직인다.**
- rescue 없음, SAM 2 추가 호출 없음(광류 1회만 추가).

**가설 (C3):** 동시 부재의 상당수는 '대상이 crop 밖으로 나가서'다. 놓친 뒤 찾기(rescue)보다, 놓치기 전과 놓친 동안 crop을 대상 위치로 옮겨 두면 SAM 2가 스스로 다시 잡는다. 마스크 중심 차분(M3vel) 대신 dense flow를 쓰면 극소 마스크의 흔들림에 덜 민감하다.

**계열:** C (운동). M8 시작 시점(16:14 KST)에 Stage 1과 무관하게, 출력 전에 고정.

**선행연구 근접도:** **높음.** MoSAM(광류로 warp한 마스크 → prompt), SAMURAI(Kalman 운동 모델), DaSiamRPN(탐색 영역 이동)과 '운동 모델로 탐색 영역을 옮긴다'는 연산이 겹친다. 남는 차이는 마스크 중심 차분 대신 대상 영역 dense flow 변위로, SAM 2의 crop view만 움직인다는 점이다. 새로움 주장 여지는 작다.

**상태 (2026-10-14 기준):**
- ΔJ&F +0.0130, CI [+0.0010, +0.0285] (M7–M8에서 처음으로 하한 > 0), 상승 5 / 하락 2, LOVO 최솟값 +0.0067, miami 제외 +0.0138, 최악 unit −0.026 → **PROMISING**.
- 거짓 재획득 증가 없음(93 = 93). 같은 기계에서 GT가 있는 동시 부재 frame을 131 → 63으로 줄였다(M3vel 108).
- 크기가 GT로 고른 rescue 상한(ORACLE_POOL +0.0136)과 같다.
- VK(개발 scene, 서술적): +0.0002 — 비회귀 통과, 이득 없음.
- **탐색적:** 같은 영상에서 arm 10여 개를 비교했다. 이득은 drone·kite-surf·skate-park에 몰린다. 새 데이터 확인이 필요하다(PUMaVOS C는 봉인).
- 어느 요소가 이득을 내는지는 ablation FV_PRESENT / FLOWCARRY / CAMCARRY로 분리 중.

${STATUS_NOTE}`,
  },

  "arms.arm.ORACLE_POOL": {
    title: "ORACLE_POOL — rescue 인과 상한 (GT 선택, 배포 불가)",
    body: `**메커니즘:** 동시 부재 frame마다 Stage 1의 31개 후보(A_cos#0–2, B_att#0–2, H_s8#0–2, crop·beam·광류·F 계열 전부, Gp/Lp)를 모두 만들고, **GT J가 최대인 후보를 그 J > 0.5일 때만** 출력 + crop 재배치. 메모리 기록 없음, \`center: out\`.

**가설 / 질문:** "현재 생성기 풀이 맞는 후보를 갖고 있고 선택이 완벽하다면, 출력 교체만으로 인과 rollout에서 얼마까지 얻는가?" 배포 가능한 rescue arm이 이 값에 얼마나 다가가는지의 기준이다.

**계열:** 진단 (rescue 상한). \`diagnostic: true\`, \`ranker.kind: oracle\`.

**선행연구 근접도:** 해당 없음(방법이 아니라 측정).

**상태 (2026-10-14 기준):** +0.0136, CI [+0.0020, +0.0297], 상승 5 / 하락 0, 최악 unit 0.000, 회수 61 frame, SAM 2 추가 약 3.1 호출/frame. 기계는 Ubuntu 53 + RunPod 11 unit(각자 같은 기계 SGGL과 짝).

**해석:** 완벽한 선택을 해도 +0.0136 — 사전 공개된 동시 부재 headroom(+0.028)의 절반 정도다. 같은 크기를 FLOWVEL이 GT 없이, SAM 2 추가 호출 없이 냈다 → "후보 생성·순위화보다 crop 제어가 이 데이터의 병목"이라는 M8의 핵심 근거다. verdict 열이 PROMISING으로 찍혀도 GT를 쓰므로 판정으로 읽지 않는다.

${STATUS_NOTE}`,
  },

  "arms.arm.ORACLE_POOL_ADMIT": {
    title: "ORACLE_POOL_ADMIT — 상한 + 메모리 기록 (배포 불가)",
    body: `**메커니즘:** ORACLE_POOL과 같고, 추가로 GT가 고른 후보를 그것을 만든 상태(G 또는 L)의 메모리에 기록한다(\`admit: true\`, 결합표 D2).

**가설 / 질문:** 회수한 frame을 메모리에 넣으면 이후 frame에서 SAM 2가 스스로 추적을 이어가므로, 출력만 바꿀 때보다 상한이 얼마나 높아지는가? 즉 "메모리 기록의 가치"의 상한이다. 잘못된 기록이 없으므로(GT 선택) 메모리 오염의 위험은 빠져 있다.

**계열:** 진단 (D 메모리 상한). \`diagnostic: true\`.

**선행연구 근접도:** 해당 없음(측정). 배포형 대응물은 DAM4SAM, SAMURAI, SAM2Long, SENTRY 같은 '검증 후 기록' 방법들이다.

**상태 (2026-10-14 기준):** +0.0169, CI [+0.0022, +0.0394], 상승 5 / 하락 0. 회수 frame은 14로 ORACLE_POOL(61)보다 적다 — 기록 덕에 SAM 2가 다음 frame을 스스로 잡아 동시 부재 트리거가 줄었기 때문이다(동시 부재 479 → 432). 배포 불가.

**해석:** 기록의 상한 이득은 ORACLE_POOL 대비 약 +0.003. 배포형(RESC_OCLR_ADMIT)에서는 잘못된 rescue까지 기록돼 신원 전환이 생겼다. 즉 기록의 가치는 '무엇을 기록해도 되는지 판별하는 능력'에 달려 있다.

${STATUS_NOTE}`,
  },

  "arms.arm.RESC_OCLR": {
    title: "RESC_OCLR — occupancy crop rescue (출력만)",
    body: `**메커니즘:** 동시 부재 frame에서 생성기 \`F_oclr8_L\` 하나만 실행: L 메모리로, 광류 운반 box 중심에, 대상 긴 변이 stride-16 cell 8개가 되도록 자른 평범한 crop을 예측한다. SAM 2 object score > 0이면(\`ranker: score, thr 0.0\`) 출력 교체 + crop 재배치. 메모리 기록 없음.

**가설:** 작은 대상은 고정 4× crop에서 특징 cell 몇 개밖에 차지하지 않아 SAM 2가 놓친다. 대상 크기에 맞춰 확대(occupancy 조절)하면 다시 '존재'로 인식된다.

**배경:** Stage 1에서 진출 조건을 통과한 계열은 없었다. 그 대신 Stage 1 재생에서 순이득이 유일하게 양수였던 조합(F_oclr8_L + score > 0: 회수 23 / 거짓 양성 5, drone 22 + lady-running 1)을 '진단적 인과 실행'으로 돌렸다. 학습 파라미터가 없어 DEV + REP 전체에서 평가 가능.

**계열:** F (척도·시야).

**선행연구 근접도:** 높음. ZoomTrack Eq.1, Siamese 추적기의 context factor, AdaSAM2 — '대상 크기로 crop 크기를 정한다'는 연산이 같다. 차이는 stride-16 cell 단위로 표현한 것뿐.

**상태 (2026-10-14 기준):** −0.0001, CI [−0.0034, +0.0029], 상승 2 / 하락 1 → **REJECTED**. 회수 11 frame, 거짓 재획득 99 vs 93, 최악 unit miami-surf obj5 −0.058. 회수해도 메모리가 그대로라 다음 frame에서 다시 놓친다 — 출력만 바꾸는 rescue의 한계.

${STATUS_NOTE}`,
  },

  "arms.arm.RESC_OCLR_ADMIT": {
    title: "RESC_OCLR_ADMIT — occupancy crop rescue + L 메모리 기록",
    body: `**메커니즘:** RESC_OCLR과 같고(\`F_oclr8_L\`, score > 0), 채택된 rescue를 L 상태 메모리에 기록한다(\`admit: true\`, 결합표 K8 "rescue + D2 메모리 기록").

**가설:** 출력만 바꾸면 다음 frame에서 다시 잃는다. 메모리에 넣으면 SAM 2가 이후 frame을 스스로 추적해 회수 효과가 길게 이어진다. 위험: 잘못된 후보가 메모리를 오염시켜 장기 신원 전환.

**계열:** F + D (척도 + 메모리).

**선행연구 근접도:** 높음. gate된 메모리 쓰기·외부 마스크 기록은 DAM4SAM, SAMURAI, SAM2Long, SENTRY, StreamDAM, Track Anything이 다룬다. 차이는 '동시 부재 rescue를 기록한다'는 트리거뿐.

**상태 (2026-10-14 기준):** +0.0046, CI [−0.0066, +0.0193], 상승 3 / 하락 1, LOVO 최솟값 −0.0019(drone 의존), 최악 unit miami-surf obj4 **−0.116**(신원 전환), 거짓 재획득 105 vs 93 → **INCONCLUSIVE**. rescue 변형 중 평균이 양수인 유일한 arm이라 K_ADMIT_FLOWVEL 결합의 근거가 됐다. 메모리 기록이 장기 효과를 만든다는 것과, 동시에 오염 위험을 키운다는 것을 함께 보여 준다.

${STATUS_NOTE}`,
  },

  "arms.arm.RESC_CPFR": {
    title: "RESC_CPFR — 문맥 보존 foveation rescue (출력만)",
    body: `**메커니즘:** 동시 부재 frame에서 \`F_cpfr8_L\` 실행: RESC_OCLR과 **같은 배율**(대상 긴 변 = 8 cell)이지만 잘라내지 않고 분리형 비균일 resize로 대상 주변을 확대하면서 시야 전체를 유지(foveation)한 영상을 L 메모리로 예측, 마스크를 역변환. score > 0이면 출력 교체 + crop 재배치. 기록 없음.

**가설 (보고서 1순위 'context-preserving foveated rescue'):** 확대하면서 주변 문맥을 보존하면, 평범한 crop보다 대상 재인식이 좋아진다. RESC_OCLR과 배율을 맞춰 '문맥 보존' 요소만 시험한다.

**계열:** F (foveation).

**선행연구 근접도:** **연산 수준에서 직접 선례 있음 — ZoomTrack (NeurIPS'23)**: 이전 box 중심의 분리형 비균일 resize, 배율 제어, 시야 보존, 역변환, 기성 추적기 test-time 적용이 모두 겹친다(FOVEA ICCV'21, LZU CVPR'23도 가까움). 남는 차이는 메모리 전파형 SAM 2 안에서 쓰고 마스크를 역변환한다는 점뿐.

**상태 (2026-10-14 기준):** −0.0070, CI [−0.0258, +0.0091], 상승 2 / 하락 3 → **REJECTED**. 회수 22 frame이지만 거짓 재획득 118 vs 93, 최악 unit disc-jockey obj3 **−0.330**(신원 전환). Stage 1에서도 같은 배율 OCLR 대비 recall 차이가 0이었다(CI가 0 포함). 문맥이 보이면서 비슷한 방해물을 '존재'로 잡는 거짓 양성이 늘었다 → 현재 데이터에서 **문맥 보존 가설은 반증 방향**.

${STATUS_NOTE}`,
  },

  "arms.arm.K_OCLR_M3": {
    title: "K_OCLR_M3 — 결합: occupancy rescue × M3-vel crop 중심",
    body: `**메커니즘:** RESC_OCLR(동시 부재 시 \`F_oclr8_L\`, score > 0, 기록 없음) + 매 frame \`center: m3vel\`(감쇠 속도 crop 중심).

**가설:** 두 구성요소가 서로 다른 사건을 겨냥한다 — rescue는 drone의 crop 이탈, M3-vel은 skate-park/sheep의 빠른 재획득. 겹치지 않으면 효과가 더해진다.

**계열:** Stage 3 결합 (운동 제어 × rescue 생성, C × F).

**선행연구 근접도:** 각 구성요소의 선행연구(M3vel, RESC_OCLR 참고)와 같다. 결합 자체도 '운동 모델 + 재탐지' 구조로 장기 추적기에서 흔하다.

**시너지 시험 (\`why\`):** vs RESC_OCLR, M3vel, SGGL — 같은 기계. 시너지 = 결합 − max(구성요소) > 0이고 그 영상 군집 CI가 0 초과.

**상태 (2026-10-14 기준):** +0.0071, CI [−0.0009, +0.0172], 상승 3 / 하락 1, 최악 unit miami-surf obj5 −0.058(RESC_OCLR의 실패를 그대로 물려받음, −0.05 미만이라 PROMISING 불가) → **INCONCLUSIVE**. 최강 구성요소 M3vel 대비 평균 +0.0025이지만 그 차이의 CI는 계산하지 않았다 — 시너지 주장 불가.

${STATUS_NOTE}`,
  },

  "arms.arm.K_OCLR_FLOWVEL": {
    title: "K_OCLR_FLOWVEL — 결합: occupancy rescue × FLOWVEL",
    body: `**메커니즘:** RESC_OCLR(동시 부재 시 \`F_oclr8_L\`, score > 0, 기록 없음) + 매 frame \`center: flowvel\`(광류 기반 crop 중심, 부재 중에도 추적).

**가설:** FLOWVEL이 줄이지 못한 나머지 동시 부재 frame을 rescue가 회수하면 효과가 더해진다.

**계열:** Stage 3 결합 (C 운동 × F 척도).

**선행연구 근접도:** FLOWVEL·RESC_OCLR 각각의 선행연구와 같다(MoSAM, SAMURAI / ZoomTrack, AdaSAM2).

**시너지 시험 (\`why\`):** vs RESC_OCLR, FLOWVEL, SGGL — 같은 기계(RunPod).

**상태 (2026-10-14 기준):** +0.0121, CI [−0.0010, +0.0295], 상승 4 / 하락 3, 최악 unit miami-surf obj5 −0.058, 회수 6 frame, 거짓 재획득 99 vs 93 → **INCONCLUSIVE**. FLOWVEL 단독(+0.0130)보다 낮다 → 결합 − 최강 구성요소 < 0, **시너지 없음**. rescue가 추가한 것은 거짓 재획득(+6)과 miami-surf 손실이었다. FLOWVEL이 이미 crop을 대상 근처에 두므로 남은 동시 부재는 대부분 진짜 부재·방해물 상황이라 rescue가 해롭게 작동한 것으로 보인다(해석, 미검증).

${STATUS_NOTE}`,
  },

  "arms.arm.FV_PRESENT": {
    title: "FV_PRESENT — FLOWVEL ablation: 속도 선행만",
    body: `**메커니즘:** \`center: fv_present\`. 출력이 있을 때만 다음 중심 = 출력 중심 + 0.5 × v(대상 영역 광류). 출력이 비면 crop 제자리(SGGL과 같음). rescue 없음.

**가설 / 질문:** FLOWVEL의 이득 중 '출력이 있을 때 광류 속도만큼 미리 옮기는' 부분이 얼마인가? M3vel과 구조가 같고 속도 추정만 마스크 중심 차분 → dense flow로 바뀐다.

**읽는 법:**
- FV_PRESENT ≈ FLOWVEL → 이득은 속도 선행(그리고 dense flow 속도 추정)에서 온다.
- FV_PRESENT ≈ M3vel 또는 SGGL → 이득의 대부분은 '부재 중 추적'(FLOWCARRY 쪽)에서 온다.
- FV_PRESENT − M3vel = 같은 '현재 있을 때만 속도' 구조에서 광류 속도 vs 마스크 중심 차분의 차이.

**계열:** C (운동) ablation. FLOWVEL PROMISING 확인 후, 이 arm의 출력 전에 고정(19:08 KST).

**선행연구 근접도:** FLOWVEL과 같음(SAMURAI 류 운동 모델).

**상태 (2026-10-14 기준):** 실행 중. 표에 없거나 unit 수가 57보다 작으면 아직 완료되지 않은 것이다. 단일 ablation이므로 결과는 기여의 방향만 알려 주며 확증이 아니다.

${STATUS_NOTE}`,
  },

  "arms.arm.FLOWCARRY": {
    title: "FLOWCARRY — FLOWVEL ablation: 부재 중 광류 추적만",
    body: `**메커니즘:** \`center: flowcarry\`. 출력이 있으면 SGGL처럼 출력 중심(속도 선행 없음). 출력이 비면 광류로 운반한 대상 box의 중심을 따라간다(매 frame 대상 영역 median 광류만큼 이동).

**가설 / 질문:** FLOWVEL의 이득이 '대상이 안 보이는 동안에도 crop이 대상을 따라가는' 부분에서 오는가? 동시 부재 frame 131 → 63 감소의 주원인이 이것이라면 FLOWCARRY가 FLOWVEL 이득 대부분을 재현해야 한다.

**읽는 법:**
- FLOWCARRY ≈ FLOWVEL → 부재 중 추적이 핵심. 속도 선행은 부차적.
- FLOWCARRY + FV_PRESENT 이득의 합과 FLOWVEL을 비교하면 두 요소의 상호작용을 대략 볼 수 있다(같은 기계, 같은 unit일 때).
- CAMCARRY와 비교하면 '대상 영역 광류'가 필요한지 '카메라 운동'만으로 충분한지 알 수 있다.

**계열:** C (운동) ablation, 19:08 KST 고정.

**선행연구 근접도:** 높음. MoSAM / SAM-PT처럼 광류·점 추적으로 대상 위치를 운반하는 것과 같다. 여기서는 그 위치를 prompt가 아니라 crop view 이동에만 쓴다.

**상태 (2026-10-14 기준):** 실행 중(RunPod). 완료 전에는 unit 수가 적어 다른 arm과 직접 비교할 수 없다.

${STATUS_NOTE}`,
  },

  "arms.arm.CAMCARRY": {
    title: "CAMCARRY — FLOWVEL ablation: 부재 중 카메라 운동만",
    body: `**메커니즘:** \`center: camcarry\`. 출력이 있으면 출력 중심. 출력이 비면 현재 crop 중심을 **카메라 운동**(대상 box 2배 바깥의 median 광류)만큼 옮긴다. 대상 영역 광류는 쓰지 않는다.

**가설 / 질문 (C3):** 대상이 안 보일 때 crop이 밀려나는 이유가 주로 **카메라 패닝**인가, 아니면 **대상 자체의 운동**인가? 카메라 운동 보상만으로 FLOWCARRY와 같은 효과가 나면, 대상 국소 광류는 필요 없고 더 싸고 안정적인 전역 운동 추정으로 충분하다.

**읽는 법:**
- CAMCARRY ≈ FLOWCARRY → 카메라 보상이 핵심, 대상 국소 광류 불필요.
- CAMCARRY ≈ SGGL < FLOWCARRY → 대상 자체의 운동을 따라가야 한다(drone처럼 대상이 카메라와 다르게 움직이는 영상).

**계열:** C3 (카메라 운동 보정) ablation, 19:08 KST 고정.

**선행연구 근접도:** 카메라 운동 보정은 추적 분야의 표준 기법(전역 운동 보상, 결합표 K15 '카메라 보정')이며 새로움 주장 여지는 없다. 이 arm은 메커니즘 분리용이다.

**상태 (2026-10-14 기준):** 대기/실행 중. 표에 없으면 아직 결과가 없다.

${STATUS_NOTE}`,
  },

  "arms.arm.K_ADMIT_FLOWVEL": {
    title: "K_ADMIT_FLOWVEL — 결합: 기록형 rescue × FLOWVEL (post-hoc 선택)",
    body: `**메커니즘:** \`center: flowvel\`(광류 crop 제어) + 동시 부재 시 \`F_oclr8_L\` rescue(score > 0) + 채택된 rescue를 L 메모리에 기록(\`admit: true\`).

**가설:** FLOWVEL이 crop을 대상 근처에 두면, 그 위치에서의 rescue는 맞을 확률이 높고, 메모리 기록으로 회수 효과가 길게 이어진다. 운동 제어 × 메모리 기록 rescue.

**계열:** Stage 3 결합 (C 운동 × F 척도 × D 메모리).

**선행연구 근접도:** FLOWVEL(MoSAM, SAMURAI) + 메모리 기록(DAM4SAM, SAM2Long, SENTRY)과 같다.

**중요한 공개 사항:** RESC_OCLR_ADMIT(+0.0046, rescue 변형 중 유일한 양수)과 FLOWVEL(+0.0130) 결과를 **본 뒤에** 고정했다(19:43 KST). 사전 계획된 시험이 아니라 **탐색적** 결합이다. 좋은 결과가 나와도 확증으로 쓰면 안 된다.

**시너지 시험 (\`why\`):** vs FLOWVEL, RESC_OCLR_ADMIT (같은 기계). 결합 − FLOWVEL > 0이고 그 CI가 0 초과여야 시너지. 동시에 최악 unit이 RESC_OCLR_ADMIT의 신원 전환(miami-surf obj4 −0.116)을 물려받는지 확인할 것. K_OCLR_FLOWVEL(기록 없음)이 FLOWVEL보다 낮았던 점을 고려하면, 기록이 그 차이를 뒤집는지가 관전 포인트다.

**상태 (2026-10-14 기준):** 대기/실행 중. 표에 없으면 아직 결과가 없다.

${STATUS_NOTE}`,
  },
  "arms.arm.FLOWVEL_U": {
    title: "FLOWVEL_U — FLOWVEL 기계 간 재현 (Ubuntu RTX 2080)",
    body: `**메커니즘:** FLOWVEL과 **spec이 완전히 같다**(\`center: flowvel\`, rescue 없음). 다른 것은 실행 기계뿐: Ubuntu RTX 2080에서 돌리고 Ubuntu diag SGGL 궤적과 짝짓는다(결과는 \`outputs/milestone8/runs/FLOWVEL_U/\`).

**질문:** RunPod 4090에서 나온 FLOWVEL의 PROMISING이 기계(GPU·커널·정밀도) 차이에 민감한 우연이 아닌가? 같은 알고리즘을 다른 기계에서, 그 기계의 SGGL과 비교해도 같은 Δ가 나오는가. M7에서 M3-vel이 Vast와 RunPod에서 같은 +0.0046을 낸 것과 같은 종류의 확인이다.

**읽는 법:**
- FLOWVEL_U ≈ FLOWVEL (ΔJ&F, 상승 영상, 최악 unit) → 결과가 기계에 견고하다.
- 크게 다르면 → 이득이 rollout의 작은 수치 차이에서 갈라지는 '분기점' 사건(재획득 시점 등)에 의존한다는 뜻이다. 이 경우 영상별 막대로 어느 영상에서 갈렸는지 확인한다.
- **같은 데이터 위의 재현**이다. 기계 견고성만 보여 줄 뿐 새 데이터 일반화의 증거는 아니다.
- FLOWVEL과 비교할 때 unit별 차이는 두 기계 SGGL의 차이도 포함한다(각자 자기 기계 SGGL을 뺐으므로). arm 대 arm 비교 표의 차이 열은 이 경우 엄밀한 같은 기계 차이가 아니다.

**계열:** C (운동), 재현 실행. FLOWVEL PROMISING 확인 후 20:50 KST 고정(이 arm의 출력 전).

**선행연구 근접도:** FLOWVEL과 같음.

**상태 (2026-10-14 기준):** 실행 중. 표에 없거나 unit 수가 57보다 작으면 아직 완료되지 않은 것이다.

${STATUS_NOTE}`,
  },
}
