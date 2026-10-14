import type { HelpMap } from "./types"

// keys: "overview.<element>"
export const help: HelpMap = {
  "overview.roadmap": {
    title: "연구 로드맵: M7에서 M8까지, 그리고 지금 내리려는 결정",
    body: `
### 출발점: M7의 결론 (\`docs/m7/FINAL_ALGORITHM_DECISION.md\`)

- **FINAL_DECISION: KEEP_SGGL.** 18개 후보 계열, prototype 33종을 실험했다. 그러나 사전 등록한 최종 규칙(DEV + REP 영상 군집 CI 하한 > 0)으로 SGGL을 이긴 방법은 없었다.
- **SGGL**은 frozen SAM 2.1 Hiera-T로 두 상태를 동시에 돌린다.
  - G: 전체 frame, P5 prompt 메모리.
  - L: 변이 prompt box의 4배인 정사각 crop. 중심은 출력을 따라간다.
  - self gate: L의 object score > 0이고, mask가 비어 있지 않고, crop 안쪽 가장자리에 닿지 않으면 L을 쓴다. 아니면 G를 쓴다.
- 가장 가까웠던 도전자는 **M3-vel**(crop 중심 속도 예측)이다. +0.0046 [−0.0001, +0.0115]였지만 이득이 2개 영상에 몰려 있었다.
- 남은 가장 큰 실패는 **동시 부재(co-absence)**다. G와 L의 object score가 모두 ≤ 0이라 SGGL 출력이 비는 frame이다.
  - 486 frame 중 131개는 대상이 실제로 있었다(놓침). 355개는 진짜로 비어 있었다(가림, 이탈).
  - 131개를 전부 완벽히 고쳐도 객체 평균 +0.028이다. 그래서 STRONG_WIN 기준(+0.03)은 rescue만으로는 도달할 수 없다(사전 공개).

### M8의 질문

> 동시 부재 실패를 줄여 SGGL을 이길 수 있는가?
> (a) 놓친 뒤 **후보를 생성하고 순위화해서 회수(rescue)** 하는 방식, (b) 놓치기 전과 놓친 동안 **crop을 대상 쪽으로 옮겨 두는 운동 및 crop 제어** 방식 중 무엇이 통하는가?

외부 보고서는 "selection보다 candidate generation"을 제안했다. M8은 이를 25개 생성기, 4개 순위화기, 인과 arm, 결합으로 넓혀 토너먼트로 검증한다(\`docs/m8/EXPERIMENT_TOURNAMENT_PLAN.md\`, 사전 등록 \`outputs/milestone8/preregistration.json\`).

### 단계

| 단계 | 내용 | 산출물 |
|---|---|---|
| 0 | M7 기록으로 실패 사례와 상한(+0.028)을 재구성한다. SGGL 재현과 성능 감사도 한다 | \`FAILURE_EPISODE_ANALYSIS.md\`, \`COMPUTE_AND_PERFORMANCE_AUDIT.md\` |
| 1 | **진단.** 살아 있는 SGGL 궤적의 trigger frame마다 25개 생성기를 *비저장* probe로 돌린다. recall, 거짓 양성, 위치 정확도, 짝 반증 시험을 잰다. 순위화기 R0–R3는 REP↔DEV 교차 적합한다 | \`stage1_report.json\`, \`STAGE1_DIAGNOSTIC_RESULTS.md\` |
| 2 | **인과 rollout.** arm을 영상 전체에 인과적으로 돌린다(출력 → 다음 crop → 다음 frame). 같은 기계의 SGGL과 비교하고 M3vel을 대조군으로 둔다 | \`runs/<arm>/\`, \`STAGE2_CAUSAL_RESULTS.md\` |
| 3 | **결합** \`K_*\`. 구성요소 근거가 있을 때만 하고 ablation을 붙인다 | \`CROSS_FAMILY_COMBINATIONS.md\` 등 |
| 4 | **결선.** VKITTI2 Scene01/06에 한 번 돌린다(서술적, 비회귀 ≥ −0.005). 이어서 최종 평가를 한다 | \`preregistration_vk.json\`, 최종 평가 문서 |

### 무엇을 결정하는가

- **M8의 결정:** SGGL을 바꿀 근거가 있는 *배포 가능한* 방법이 있는지, 있다면 무엇을 다음 확인 단계의 지정 도전자로 둘지 정한다.
- **PROMISING**(M7 pilot 규칙)은 "후보로 올릴 만하다"는 뜻이다. 교체 결정이 아니다.
  - 같은 개발 영상(DEV + REP)에서 arm 10여 개를 비교했다. 그래서 모든 결과는 **탐색적**이다.
  - 교체하려면 새 데이터나 봉인 세트로 확인해야 한다.
- 새 tiny 데이터(LVOS v2, SA-V, VOST 후보)는 다운로드에 **사용자 승인이 필요해 BLOCKED**다. PUMaVOS C는 봉인 상태이고 열지 않는다.
- 2026-10-14 기준 상태:
  - Stage 1: 진출 조건을 통과한 생성기 계열이 없다. held-out 순 이득이 양수인 순위화기도 없다.
  - Stage 2: 배포 가능한 arm 중에서는 **FLOWVEL**(RAFT 광류로 crop 중심 제어)만 PROMISING이다. 출력만 바꾸는 rescue arm은 REJECTED 또는 INCONCLUSIVE다.
  - VK: FLOWVEL과 M3vel은 비회귀를 통과했지만 이득은 없다.
  - ablation과 최종 문서는 진행 중이다. 현재 상태는 아래 "현재 작업" 패널을 보라.

### 이 페이지 읽는 순서

1. **현재 작업:** 무엇이 돌고 있는지 본다.
2. **상태 카드:** 기준 알고리즘, 최고 도전자, 진행률, 동기화 건강 상태를 본다.
3. **forest plot:** 모든 arm의 효과 크기와 불확실성을 한눈에 본다.
4. **Stage-1 요약:** rescue 후보 생성의 상한과 비용을 본다.
`,
  },

  "overview.tasks": {
    title: "현재 작업 패널",
    body: `
**무엇인가:** M8 실행 계획의 작업 목록이다. 진단, 인과 arm, 결합, VK, ablation, 문서 작업이 들어 있다. 상태별로 묶어 보여 준다.

**데이터 출처:**
- 출처는 \`outputs/milestone8/TASKS.json\`이다. 서버의 \`/api/tasks\`가 읽고, 화면은 15초마다 다시 가져온다.
- **상태(status)는 자동으로 감지되지 않는다.** 에이전트가 \`uv run python scripts/m8_tasks.py <id>=<STATUS>\`로 직접 갱신한다. 카드 설명의 "갱신 …" 시각(\`updated_kst\`)은 마지막으로 상태를 바꾼 시각이다.
  - 따라서 GPU 작업이 실제로 끝났어도, 누군가 갱신하기 전까지 RUNNING으로 남을 수 있다.
  - 실제 진척은 진행률 막대(파일 수를 실시간으로 셈)와 로그 tail로 확인하라.
- **진행률**은 서버가 매 요청마다 계산한다. \`progress.glob\`에 맞는 결과 파일 수를 \`total\`로 나눈다(자세한 내용은 진행률의 "?").

**한 줄의 구성:**
- 상태 배지, 제목, 작업 id(\`C-FLOWVEL\` 등), \`· 기계\`.
  - 기계는 TASKS.json의 \`machine\` 자유 텍스트다. 예: "RunPod 4090", "Ubuntu 2080 (53) + RunPod (11)".
  - 같은 arm이라도 기계가 다르면 기준선이 달라진다. Ubuntu arm은 Ubuntu diag 궤적과, RunPod arm은 \`runpod/runs/SGGL\`과 짝짓는다.
- 설명(무엇을 하는 작업인지).
- 진행률 막대. QUEUED 상태이고 완료 파일이 0개면 숨긴다.
- 선행 작업 배지.

**작업 id 접두사 관례:**

| 접두사 | 뜻 |
|---|---|
| \`S1-\` | Stage 1 |
| \`C-\` | 인과 arm |
| \`O-\` | ORACLE(진단용) |
| \`K-\` | 결합 |
| \`A-\` | ablation |
| \`VK\` | 결선 |
| \`DOCS\`, \`DASH\` | 문서, 대시보드 |

**오른쪽 "최신 로그"**는 최근에 수정된 로그 4개의 마지막 15줄이다(그쪽 "?" 참고).
`,
  },

  "overview.taskStatus": {
    title: "작업 상태 배지 (실행 중 / 대기 / 차단됨 / 완료)",
    body: `
| 배지 | TASKS.json 값 | 뜻 |
|---|---|---|
| **● 실행 중** (파랑) | \`RUNNING\` | GPU worker나 CPU에서 지금 돌고 있다 |
| **대기** (회색) | \`QUEUED\` | 순서를 기다린다. 보통 선행 작업이 끝나야 시작한다. 진행률 막대는 완료 파일이 생겨야 나타난다 |
| **차단됨** (빨강) | \`BLOCKED\` | 외부 조건 때문에 시작할 수 없다. 예: 데이터 다운로드에 사용자 승인 필요, worker 연결 불가. 설명에 이유를 적는다 |
| **완료** (초록) | \`DONE\` | 끝났다. 목록이 길어지지 않도록 "완료 N건" 접기 안에 모아 둔다 |

이 네 값 외의 상태는 목록 맨 아래에 그대로 표시한다.

**주의:**
- 상태는 사람(에이전트)이 \`scripts/m8_tasks.py\`로 직접 바꾸는 값이다. 실행 여부를 자동으로 감지하지 않는다.
  - 진행률이 100%인데 "실행 중"이면 아직 상태 갱신을 안 했을 수 있다.
  - 반대로 "완료"인데 진행률이 100% 미만일 수도 있다. 예: ORACLE은 Ubuntu와 RunPod에 나눠 돌았는데 glob은 한쪽만 센다.
- "완료"는 실행이 끝났다는 뜻일 뿐이다. 결과가 좋다는 뜻이 아니다. 판정은 forest plot과 Causal arms 페이지에서 본다.
`,
  },

  "overview.taskProgress": {
    title: "진행률 막대: 결과 파일 수를 실시간으로 센다",
    body: `
**계산:**
- \`done\` = 저장소 안에서 \`progress.glob\`에 맞는 파일 수. \`dashboard/server.py: safe_glob\`가 허용된 경로 안에서만 센다.
- \`percent\` = \`100 × done / total\`. \`total\`은 TASKS.json에 적힌 기대 개수다. 예: 인과 arm은 64단위(DEV 21 + REP 43), Ubuntu에서 돌린 Stage 1 진단은 53.
- 막대 색: **초록**은 상태가 DONE, **파랑**은 그 외. 막대 폭은 100%에서 자른다. 숫자는 100%를 넘을 수 있다.
- 막대에 마우스를 올리면 glob 패턴이 보인다.

**파일 하나 = 단위 하나 완료:**
- 결과 CSV는 \`atomic_csv\`로 단위가 끝날 때 한 번에 쓴다.
- 그래서 파일이 있으면 그 단위는 끝난 것이다. 반쯤 쓴 파일은 세지 않는다.

**함정:**
1. **RunPod 결과는 동기화 지연만큼 늦게 잡힌다.**
   - Ubuntu에 도착한 파일만 센다. RunPod 결과는 \`gpu_sync.py loop 300\`(5분 주기)이 가져온다.
   - worker에서 수정된 지 1분이 안 된 파일은 다음 pull로 미룬다.
   - 따라서 실제 진척보다 최대 몇 분 늦다.
2. **100%를 넘을 수 있다.** glob이 같은 폴더의 VKITTI 결과(28단위)까지 세면 그렇다. 예: \`runpod/runs/FLOWVEL\`에는 64 + 28 = 92개 파일이 있다(2026-10-14 기준).
3. **파일 수는 결과의 품질이나 유효성과 무관하다.** 실행 오류는 로그에서 확인한다.
4. **\`?\`로 표시되면** glob이 허용 경로 밖이라 서버가 세기를 거부한 것이다.
`,
  },

  "overview.taskDeps": {
    title: "선행 작업 (depends_on)",
    body: `
**무엇인가:**
- TASKS.json의 \`depends_on\` 목록이다. 이 작업이 시작되기 전에 끝나야 하는 작업의 id를 적는다.
- 배지에는 선행 작업의 *현재* 상태가 함께 붙는다(예: \`A-FLOWCARRY · 실행 중\`). 선행 작업이 완료되면 글자가 초록이 된다.

**두 종류의 의존이 섞여 있다:**
- **실행 순서:** 같은 RunPod GPU 큐에서 차례를 기다린다. 예: ablation \`A-FV_PRESENT → A-FLOWCARRY → A-CAMCARRY\`. 과학적 의존은 아니다.
- **결과 의존:** 앞 결과를 본 뒤 사양을 고정한다. 예: \`K-ADMIT-FLOWVEL\`은 설명에 "탐색적 (결과를 본 뒤 고정)"이라고 적혀 있다.
  - 이런 arm은 출력 전에 고정한 arm보다 증거가 약하다. 사후 선택이 개입하기 때문이다.

어느 쪽인지는 각 작업의 설명과 \`arms.json\`의 \`frozen_kst\`(사양을 고정한 시각)로 판단한다.
`,
  },

  "overview.logs": {
    title: "최신 로그 tail",
    body: `
**무엇인가:**
- 다음 위치의 로그 중 **가장 최근에 수정된 4개**를 골라 각각 마지막 15줄을 보여 준다.
  - \`outputs/milestone8/logs/*.log\`: Ubuntu 쪽 진단, 실행, 동기화 로그.
  - \`outputs/milestone8/runpod/logs/queue*.log\`: RunPod 큐 로그.
- 서버는 파일 끝 2 MB만 읽는다(\`server.py: latest_logs\`).
- 파일 이름 옆 시각은 파일 수정 시각(브라우저 시간대)이다. 파일 이름을 누르면 원본 파일 보기로 간다.

**자주 보는 줄:**

| 로그 | 줄 예시 | 뜻 |
|---|---|---|
| \`gpu_sync.log\` | \`pull outputs/milestone8/runpod: 24 new, 0 conflicts\` | 5분마다 RunPod에서 가져온 새 파일 수와 충돌 수 |
| \`gpu_sync.log\` | \`RETRY (changed during transfer)\` | 전송 중 파일이 바뀌었다. 저장하지 않고 다음 pull에서 다시 받는다(정상) |
| \`gpu_sync.log\` | \`FAILED\` / \`GAVE UP\` | worker 연결 실패다. worker 쪽 파일은 그대로 둔다. "동기화 상태" 카드를 보라 |
| \`queue*.log\` | \`FLOWVEL 2026-10-14T09:52:57Z units=64\` | 해당 arm이 끝난 시각(UTC)과 단위 수 |
| \`queue*.log\` | \`QUEUE1_DONE\` | 그 큐의 모든 작업이 끝났다 |
| \`diag_*.log\` | \`diag <unit> <frame 수> <trigger 수> <J&F> <초>s\` | Stage 1 진단에서 단위 하나가 끝났다 |

**주의:**
- RunPod 로그는 동기화된 사본이라 실제보다 늦다.
- 로그가 계속 자라는 동안은 동기화 ledger에 \`APPENDED\`로 여러 번 기록된다.
- 기존 내용과 앞부분이 다르게 바뀐 로그는 \`*.conflict-<ts>\`로 따로 저장된다.
`,
  },

  "overview.incumbent": {
    title: "현 기준 알고리즘 (incumbent): SGGL",
    body: `
**무엇인가:**
- 모든 M8 arm이 이겨야 하는 기준선이다.
- M7 최종 결정은 **KEEP_SGGL**이다(\`docs/m7/FINAL_ALGORITHM_DECISION.md\`).
- 이 카드는 고정 값을 보여 준다. 데이터로 계산하지 않는다.

**SGGL (Self-Gated Global-Local)**
- frozen SAM 2.1 Hiera-T로 두 상태를 매 frame 돌린다.
  - **G:** 전체 frame, P5 prompt 메모리.
  - **L:** 변이 prompt box의 4배인 정사각 crop. 중심은 출력 mask를 따라간다.
- **self gate:** L의 object score > 0이고, mask가 비어 있지 않고, crop 안쪽 가장자리에 닿지 않으면 L 출력을 쓴다. 아니면 G를 쓴다.
- 학습한 파라미터가 없다.
- M7의 VK 결과(서술적)는 P5 대비 +0.017이다. 다만 VK scene 2개는 이전 milestone의 개발 scene이었다.

**machine-matched 비교 (가장 중요한 규칙)**
- 모든 ΔJ&F는 *같은 기계에서 돌린* SGGL과의 차이다.
  - RunPod RTX 4090 arm은 \`outputs/milestone8/runpod/runs/SGGL/<unit>.csv\`와 짝짓는다.
  - Ubuntu RTX 2080 arm은 Stage 1 진단 실행이 기록한 SGGL 궤적 \`outputs/milestone8/diag/<unit>.csv\`와 짝짓는다.
- **왜 필요한가:** GPU와 커널이 다르면 SAM 2 출력이 미세하게 달라진다. 인과 rollout에서는 그 차이가 궤적 차이로 커질 수 있다. 다른 기계의 SGGL과 비교하면 기계 차이가 알고리즘 효과로 섞인다.
- 단위별 기계 배정(shard)은 모든 arm에서 같게 유지한다.

**연구 질문과의 관계:**
- M8의 어떤 arm도 아직 이 카드를 바꾸지 않았다(2026-10-14 기준).
- PROMISING은 후보 지위일 뿐이다. 교체하려면 새 데이터나 봉인 세트로 확인해야 한다.
`,
  },

  "overview.bestChallenger": {
    title: "최고 causal 도전자",
    body: `
**무엇인가:**
- \`outputs/milestone8/causal_report.json\`에서 다음을 모두 만족하는 arm 중 ΔJ&F가 가장 큰 것을 보여 준다.
  - verdict가 \`PROMISING\` 또는 \`STRONG_WIN\`이다.
  - \`arms.json\`에서 \`diagnostic: true\`가 아니다(즉 배포 가능하다).
- 큰 숫자는 ΔJ&F다. 아래 줄은 95% CI와 판정 대상 단위 수다. 누르면 해당 arm의 상세 페이지로 간다.
- 조건을 만족하는 arm이 없으면 "PROMISING arm 없음"이 나온다.

**계산** (\`scripts/m8_analysis.py: causal / _pilot\`)
- 단위마다 \`d = mean(J&F_arm) − mean(J&F_SGGL)\`을 계산한다.
  - J&F는 prompt frame을 뺀 frame 1..n-1의 평균이다.
  - SGGL은 같은 기계의 기준선이다.
- **작은 객체**(prompt 면적 < 1%)만 평균한다. 그래서 단위 수가 64가 아니라 57이다(34영상).
- **CI**는 영상 군집 bootstrap이다. 영상 단위로 2000회 재표집하고 백분위 2.5–97.5%를 쓴다. 한 영상 안의 여러 객체는 독립이 아니라서 영상 단위로 묶는다.

**읽는 법:**
- ΔJ&F +0.013은 "작은 객체 평균 J&F가 SGGL보다 0.013 높다"는 뜻이다(J&F는 0–1 척도).
- CI 하한이 0보다 크면, 영상 표본이 달라져도 부호가 유지될 가능성이 높다.
  - 이것이 M7 최종 규칙의 조건이다. M3-vel은 −0.0001로 실패했다.
- 예시(\`STAGE2_CAUSAL_RESULTS.md\`, 2026-10-14 기준): FLOWVEL은 +0.0130 [+0.0010, +0.0285], 57단위로 M7–M8에서 처음으로 CI 하한 > 0이다.

**주의:**
- **탐색적이다.** 같은 개발 영상에서 arm 10여 개를 비교했다(다중 비교). 출력 전에 고정한 arm이라도 새 데이터 확인이 필요하다.
- **이득이 소수 영상에 몰린다.**
  - FLOWVEL은 5개 영상에서 오르고 2개에서 내렸다. 최고 단위는 drone obj1 +0.379다.
  - 그래서 판정 규칙에 "영상 하나씩 빼기 최솟값 > 0"과 "miami-surf 제외 > 0"이 들어 있다.
- **PROMISING 판정은 CI 하한 > 0을 요구하지 않는다**(pilot 규칙). CI는 따로 확인해야 한다.
`,
  },

  "overview.oracleNote": {
    title: "\"진단(ORACLE) 포함 시\" 줄",
    body: `
**무엇인가:**
- ORACLE 같은 진단용 arm까지 포함했을 때 ΔJ&F가 가장 큰 PROMISING arm이 배포 가능한 최고 도전자와 다르면 이 줄이 나온다.

**ORACLE arm** (\`ORACLE_POOL\`, \`ORACLE_POOL_ADMIT\`, \`arms.json\`의 \`diagnostic: true\`)
- 동시 부재 frame에서 31개 rescue 후보 중 **정답(GT) J가 가장 높고 J > 0.5인 후보**를 고른다.
- \`_ADMIT\`은 그 결과를 SAM 2 메모리에도 기록한다.
- 실제로는 GT를 알 수 없으므로 **배포할 수 없다.** "순위화기가 완벽하다면 rescue로 얼마까지 가능한가"라는 상한을 재는 진단이다.

**왜 보여 주는가:**
- 배포 가능한 arm과 상한을 비교하면, 남은 개선 여지가 어디에 있는지 알 수 있다.
- 예시(\`STAGE2_CAUSAL_RESULTS.md\`, 2026-10-14 기준): ORACLE_POOL은 +0.0136, FLOWVEL은 +0.0130이다.
  - 순위화나 SAM 2 추가 호출 없는 crop 제어가 완벽한 순위화 rescue의 상한과 같은 크기를 냈다.
  - 그래서 연구 중심이 "놓친 뒤 찾기"에서 "놓치지 않게 crop 옮기기"로 이동했다.

**주의:**
- ORACLE이 PROMISING이어도 의미는 "상한이 기준을 넘는다"는 것뿐이다. 교체 후보가 아니다.
- ORACLE은 Ubuntu 53단위와 RunPod 11단위로 나눠 실행했다. 각 단위는 같은 기계의 SGGL과 짝지었다.
`,
  },

  "overview.progress": {
    title: "Causal 진행 상황: 기계별, arm별 완료 단위 수",
    body: `
**무엇인가:**
- 기계별로 인과 arm 결과 폴더에 있는 CSV 파일 수를 기대 단위 수 64(DEV 21 + REP 43)와 비교한다.
  - **Ubuntu 2080:** \`outputs/milestone8/runs/<arm>/\`
  - **RunPod 4090:** \`outputs/milestone8/runpod/runs/<arm>/\`
- 출처는 \`/api/overview\`의 \`runs\`이고, 30초마다 갱신한다.
- 파일 하나는 단위(한 영상의 한 객체) 하나의 frame별 결과다. 단위가 끝날 때 원자적으로 쓴다.
- 막대가 **초록**이면 64개 이상, **파랑**이면 진행 중이다.
- \`◇\`는 \`arms.json\`의 \`diagnostic: true\`(ORACLE, 배포 불가) arm이다.
- RunPod의 \`SGGL\` 행은 arm이 아니라 **같은 기계 기준선**이다. 이것이 없으면 RunPod arm의 ΔJ&F를 계산할 수 없다.

**읽는 법과 함정:**
1. **ORACLE은 두 기계에 나뉘어 있다.** Ubuntu 53 + RunPod 11 = 64다. 한 줄만 보고 미완료로 판단하지 말라.
2. **64를 넘을 수 있다.** FLOWVEL, M3vel, SGGL 폴더에는 VKITTI 결선 결과 28단위가 함께 있다. 그래서 92/64가 될 수 있다. VK 결과는 판정(\`causal_report.json\`의 본 항목)에 섞이지 않고 \`<arm>@VK\`로 따로 보고된다.
3. **RunPod 숫자는 동기화(5분 주기) 지연만큼 늦다.** worker에서 끝났어도 Ubuntu로 pull되기 전에는 세지 않는다.
4. 파일 수가 64라도 판정에 들어가는 것은 **작은 객체(< 1%) 57단위**뿐이다. forest plot 이름 옆 괄호가 그 수다.
5. forest plot과 판정은 \`scripts/m8_analysis.py causal\`을 다시 돌려야 갱신된다. 진행 막대가 늘어도 forest plot은 그대로일 수 있다.
`,
  },

  "overview.sync": {
    title: "동기화 상태: RunPod 결과가 Ubuntu에 안전하게 도착했는가",
    body: `
**왜 중요한가:**
- **Ubuntu(RTX 2080)가 모든 연구 결과의 유일한 원본(source of truth)이다.** GPU worker(RunPod RTX 4090)는 언제든 버릴 수 있는 계산 자원이다.
- worker는 결과를 밀어 보내지 않는다. Ubuntu가 \`scripts/gpu_sync.py\`로 끌어온다(pull).
- pull이 실패해도 worker 사본은 건드리지 않는다.
- 자세한 내용은 \`docs/GPU_WORKER_OPERATIONS.md\`에 있다.

**pull 절차** (디렉터리마다, \`gpu_sync.py loop 300 outputs/milestone8/runpod\`로 5분마다)
1. worker가 manifest를 출력한다. 각 파일의 경로, 크기, sha256이다. 수정된 지 1분 안 된 파일은 아직 쓰는 중으로 보고 제외한다.
2. tar 스트림 하나로 Ubuntu staging에 받는다.
3. 받은 파일마다 크기와 sha256을 manifest와 비교한다. 다르면 저장하지 않고 다음 pull에서 다시 받는다.
4. **덮어쓰지 않고 병합한다.** 결과는 ledger 상태(NEW, SAME, APPENDED, CONFLICT)로 남는다. 각 상태의 뜻은 ledger 배지의 "?"를 보라.
5. 검증된 파일을 \`outputs/SYNC_LEDGER.tsv\`에 한 줄씩 추가한다.
   - worker에는 \`.ubuntu_verified.tsv\`를 써 둔다. 여기에 적힌 파일만 worker에서 지워도 된다.
6. \`outputs/SYNC_STATUS.json\`에 디렉터리별 마지막 성공과 실패를 기록한다. 실패를 성공으로 보고하지 않는다.

**카드 구성:**
- 맨 위 배지: ledger 상태별 누적 줄 수.
- 그 아래: pull 대상 디렉터리별 상태(\`last_ok\`, \`last_fail\`, 신규, 충돌).
- 설명 줄의 시각: \`SYNC_STATUS.json\`의 \`updated\`, 즉 상태 파일을 마지막으로 쓴 시각이다.

**worker를 끄기 전 규칙:** \`gpu_sync.py audit outputs/milestone8/runpod\`가 exit 0이어야 한다. exit 0은 worker의 모든 파일이 Ubuntu에서 검증되었다는 뜻이다.
`,
  },

  "overview.ledger": {
    title: "Ledger 상태 배지: NEW / SAME / APPENDED / CONFLICT",
    body: `
**무엇인가:**
- \`outputs/SYNC_LEDGER.tsv\`의 \`status\` 열 값별 줄 수다. 처음부터 지금까지의 **누적**이다.
- ledger 한 줄은 "worker 파일 하나(경로 + sha256)가 Ubuntu에서 검증되었다"는 기록이다.
- 열: \`time, worker, remote, local, sha256, size, status\`.

| 상태 | 언제 | Ubuntu 파일에 일어난 일 |
|---|---|---|
| **NEW** | Ubuntu에 같은 경로가 없었다 | 그대로 옮겨 놓았다. 대부분의 결과 CSV가 여기에 해당한다 |
| **SAME** | 같은 경로가 있고 sha256도 같다 | 아무것도 바꾸지 않았다. 다른 경로로 이미 받았거나 수동으로 복사된 파일을 확인만 한 경우다 |
| **APPENDED** | 같은 경로가 있고, 기존 Ubuntu 내용이 새 파일의 **앞부분과 정확히 같다**(뒤에만 덧붙음) | 새 파일로 교체했다. 기존 바이트는 접두부로 보존된다. 계속 자라는 로그 파일이 이렇게 된다. 자랄 때마다 sha가 바뀌므로 같은 로그가 여러 번 기록된다 |
| **CONFLICT** | 같은 경로가 있는데 내용이 다르고, 덧붙인 것도 아니다 | **Ubuntu 사본은 그대로 둔다.** worker 버전은 \`<이름>.conflict-<ts>\`로 따로 저장한다. 원본을 절대 덮어쓰지 않는다 |

빨간 배지(destructive)는 이름에 FAIL이나 CONFLICT가 들어간 상태다.

**읽는 법:**
- NEW가 늘어나는 것은 정상이다. RunPod 결과가 도착하고 있다는 뜻이다.
- **CONFLICT는 결과 CSV에 생기면 심각하다.** 같은 단위를 두 번, 다르게 계산했다는 뜻이기 때문이다. 어느 쪽이 맞는지 확인한 뒤에 분석해야 한다.
- 로그 파일의 CONFLICT는 대개 무해하다. 로그가 다시 쓰이거나 처음부터 다시 시작된 경우다.
- 예시(2026-10-14 기준 ledger): CONFLICT 9건 = \`_synctest\` 동기화 자체 시험 1건(의도적) + RunPod 로그 파일 8건. 결과 CSV의 CONFLICT는 없었다.
`,
  },

  "overview.syncTargets": {
    title: "디렉터리별 동기화 상태: last_ok, last_fail, 신규, 충돌",
    body: `
\`outputs/SYNC_STATUS.json\`의 키는 \`<종류>:<원격 디렉터리>\` 형식이다. 이 카드는 디렉터리마다 한 줄씩 보여 준다. 예: \`runpod\`, \`runpod/runs\`, \`runpod/diag\`, \`runpod/bench\`, \`runpod/_synctest\`.

| 표시 | 키 | 뜻 |
|---|---|---|
| **OK 시각** | \`last_ok:<dir>\` | 그 디렉터리를 마지막으로 **성공적으로** pull한 시각(Ubuntu 시간). manifest, 전송, 검증, 병합이 모두 끝났을 때만 기록한다. 받을 파일이 없어도 성공이다 |
| **신규** | \`last_new:<dir>\` | *마지막 성공 pull* 한 번에서 NEW로 들어온 파일 수. 누적이 아니다. APPENDED와 SAME은 세지 않는다 |
| **충돌** | \`conflicts:<dir>\` | *마지막 성공 pull* 한 번에서 생긴 CONFLICT 수 |
| **빨간 글씨** | \`last_fail:<dir>\` | 마지막 실패 시각과 오류 메시지. 예: \`manifest failed: ssh: … Connection refused\`, \`verification failed: …\`. pull은 최대 3번 시도하고(실패 사이에 30 s, 60 s 대기) 그래도 안 되면 포기한다. worker 사본은 그대로 둔다 |

**✓ OK / ✕ 실패 배지 규칙:**
- \`last_fail\`의 시각(앞 19자)이 \`last_ok\`보다 **나중**이면 ✕ 실패다. 즉 마지막 시도가 실패했다.
- 실패 기록이 있어도 그 뒤에 성공했으면 ✓ OK다. 과거 실패는 이미 회복되었다는 뜻이다.
- 성공 기록이 전혀 없고 실패만 있어도 ✕다.

**읽는 법과 함정:**
- 현재 실제로 loop가 도는 대상은 \`outputs/milestone8/runpod\`(상위 디렉터리)다. 하위 디렉터리 줄은 과거에 따로 pull했던 기록이 남은 것일 수 있다.
  - OK 시각이 오래되었다고 문제는 아니다. 상위 디렉터리 pull이 같은 파일을 받기 때문이다.
- \`_synctest\`는 설정 직후 동기화 기능을 시험한 디렉터리다. 일부러 연결 실패와 충돌을 만든 기록이 남아 ✕로 보일 수 있다.
- **정말 봐야 할 것:** 상위 \`runpod\` 줄이 ✓이고 OK 시각이 최근(5분 주기 기준 10분 이내)인지 본다.
  - 오래되었으면 tmux \`m8:sync\` 창과 \`outputs/milestone8/logs/gpu_sync.log\`를 확인한다.
`,
  },

  "overview.forest": {
    title: "Causal arms forest plot 읽는 법",
    body: `
**무엇인가:**
- \`causal_report.json\`의 모든 인과 arm에 대해, 같은 기계의 SGGL 대비 ΔJ&F와 95% CI를 한 축에 그린다.
- VKITTI 결과(\`<arm>@VK\`)는 제외한다. 위에서부터 ΔJ&F가 큰 순서다.
- 그림을 누르면 Causal arms 페이지로 간다. 아래 칩을 누르면 해당 arm 상세로 간다.

**계산** (\`scripts/m8_analysis.py causal\`)
- 단위마다 \`d = mean(J&F_arm) − mean(J&F_SGGL)\`을 계산한다. frame 1..n-1을 쓰고, 같은 기계의 SGGL과 비교한다.
  - RunPod arm은 \`runpod/runs/SGGL\`과 짝짓는다.
  - Ubuntu arm은 \`diag/<unit>.csv\` 궤적과 짝짓는다.
- **점** = 작은 객체(prompt 면적 < 1%) 단위들의 d 평균.
- **이름 옆 괄호** = 그 단위 수. 보통 57(34영상)이다.
- **수염(error bar)** = 영상 군집 bootstrap 95% 백분위 CI.
  - 영상을 복원 추출로 2000번 다시 뽑고, 매번 평균을 계산해 2.5%와 97.5% 지점을 쓴다.
  - 한 영상 안의 객체들은 독립이 아니라서 영상 단위로 묶는다. 그래서 수염이 단위 수에 비해 넓다.
- **마우스를 올리면** 정확한 값, CI, 판정, 기계별 단위 수(예: \`runpod:64\`)가 보인다.
  - 기계별 수는 큰 객체를 포함한 전체 단위 수다.

**읽는 법:**
- **세로선(0)** = SGGL과 같음. 오른쪽은 SGGL보다 좋고, 왼쪽은 나쁘다.
- **색 = 판정(verdict).** 초록은 PROMISING(또는 STRONG_WIN), 주황은 INCONCLUSIVE, 빨강은 REJECTED, 회색은 판정 없음. 규칙은 범례의 "?"를 보라.
- **속이 빈 마름모 ◇** = \`arms.json\`의 \`diagnostic: true\`, 즉 ORACLE arm이다.
  - GT로 최선 후보를 고르므로 배포할 수 없다. "완벽한 순위화의 상한" 기준선으로만 읽는다.
  - 채워진 원(배포 가능한 arm)과 순위를 직접 비교하지 말라.
- **CI 하한 > 0(수염 전체가 0선 오른쪽)이 왜 중요한가**
  - M7의 최종 교체 규칙이 "DEV + REP 영상 CI 하한 > 0"이었다.
  - M3-vel은 하한 −0.0001로 이 선을 넘지 못했다.
  - 수염이 0을 걸치면, 영상 표본에 따라 효과의 부호가 바뀔 수 있다는 뜻이다.
  - 2026-10-14 기준, 배포 가능한 arm 중 이 조건을 만족하는 것은 FLOWVEL뿐이다(+0.0010 하한). M7–M8에서 처음이다.
- **점 위치와 색이 다를 수 있다.**
  - 판정 규칙은 평균, 상승 영상 수, 하나씩 빼기 등을 보지만 CI는 보지 않는다(STRONG_WIN 제외).
  - 그래서 CI가 0을 걸쳐도 PROMISING일 수 있다. 반대로 평균이 양수여도 INCONCLUSIVE일 수 있다(예: 상승 영상이 2개뿐).

**주의:**
- **다중 비교:** 같은 개발 영상(DEV + REP)에서 arm 10여 개를 비교했다. 그중 하나가 우연히 CI 하한 > 0일 확률은 무시할 수 없다. 모든 결과는 탐색적이다.
- **소수 영상에 몰림:** 이득이 drone, kite-surf, skate-park 같은 몇 영상에 몰리기 쉽다. Causal arms 페이지의 영상별 결과를 같이 보라.
- **STRONG_WIN(+0.03 & CI 하한 > 0)은 사실상 불가능하다.** 동시 부재 frame을 모두 완벽히 고쳐도 +0.028이기 때문이다.
- 이 그림은 판정 스크립트를 다시 돌려야 갱신된다. 진행 중인 arm은 아직 나타나지 않을 수 있다.
`,
  },

  "overview.verdicts": {
    title: "판정(verdict) 규칙과 색 범례",
    body: `
M7 pilot 규칙을 그대로 사전 등록했다(\`preregistration.json\` \`stage2_causal_rule\`, 구현은 \`m8_analysis.py: _pilot\`). 계산 대상은 작은 객체 단위다. "상승 영상"은 영상별 평균 d가 0.01을 넘는 영상이다.

| 판정 | 색 | 조건 |
|---|---|---|
| **PROMISING** ▲ | 초록 | 아래 다섯 조건을 **모두** 만족 |
| **REJECTED** ✕ | 빨강 | ΔJ&F ≤ 0, **또는** (상승 영상 ≤ 2 **이고** 하나씩 빼기 최솟값 ≤ 0) |
| **INCONCLUSIVE** ? | 주황 | 위 둘 다 아님 |
| **STRONG_WIN** | 초록 | ΔJ&F ≥ 0.03 **이고** CI 하한 > 0. 동시 부재 상한이 +0.028이라 이 데이터에서는 도달 불가로 사전 공개 |

**PROMISING의 다섯 조건:**
1. ΔJ&F ≥ +0.005
2. 상승 영상 ≥ 3
3. 영상 하나씩 빼기(LOVO) 최솟값 > 0: 어떤 영상 하나를 빼도 평균이 양수
4. \`davis/miami-surf\`를 빼도 평균 > 0: M7에서 큰 손실과 이득이 모두 나온 영상
5. 최악 단위 ≥ −0.05: 재앙적 신원 전환이 없음

**◇ 속이 빈 마름모:** ORACLE(진단용, GT 사용). 판정 색은 다른 arm과 같은 규칙으로 칠하지만, 배포할 수 없으므로 교체 후보가 아니다.

**왜 이런 규칙인가:**
- 평균 하나만으로는 한 영상의 큰 이득(예: drone obj1 +0.38)이 전체를 끌어올린 경우를 걸러내지 못한다.
- LOVO와 상승 영상 수는 **효과가 여러 영상에 퍼져 있는지** 본다.
- 최악 단위 조건은 **드물지만 큰 손실**을 막는다. 예: RESC_CPFR의 disc-jockey obj3 −0.330.

**한계:**
- PROMISING은 "다음 확인 단계로 올릴 후보"라는 뜻이다. 교체 결정이 아니다.
- 이 규칙은 CI를 보지 않는다. CI 하한 > 0은 forest plot 수염으로 따로 확인한다.
`,
  },

  "overview.stage1": {
    title: "Stage-1 진단 요약",
    body: `
**무엇인가:**
- Stage 1은 "놓친 대상을 다시 찾을 후보를 만들 수 있는가"를 *인과 실행 전에* 값싸게 잰 진단이다.
- 출처는 \`outputs/milestone8/stage1_report.json\`이다(\`scripts/m8_analysis.py stage1\`로 재생성).
- 원자료:
  - \`outputs/milestone8/diag/\`: Ubuntu 2080, 53단위
  - \`outputs/milestone8/runpod/diag/\`: RunPod 4090, 11단위

**방법:**
- 살아 있는 SGGL 궤적을 돌리면서 trigger frame마다 25개 생성기를 **비저장 probe**로 실행한다. 메모리에 쓰지 않으므로 SGGL 궤적은 바뀌지 않는다. trigger 종류는 다음과 같다.
  - **동시 부재(coabs):** G와 L의 score가 모두 ≤ 0
  - **불일치(disagree):** 한쪽만 존재하거나 IoU(L, G) < 0.5
  - **정기(regular):** 나머지 frame 중 10번째마다, 보정용
- 생성기 계열:
  - A_cos: 전역 prototype 탐색
  - B_att: 대조 token attention
  - H_s8: stride-8 특징
  - G_neg: 음성 점
  - D_ref: frame-0 메모리
  - A/B_crop: heatmap 기반 crop
  - E_beam: 속도 beam
  - C_flow: RAFT 광류
  - F_oclr / F_cpfr: occupancy crop / foveation
  - Gp / Lp: M7 보호 메모리, 참조용
- 지표:
  - recall: GT가 있는 동시 부재 frame에서 후보 중 하나라도 J > 0.5인 비율
  - 원시 거짓 양성(FP): GT가 빈 frame에서 후보가 '존재'를 출력한 비율
  - 위치 정확도와 짝 반증 시험
- 순위화기 R0–R3를 REP↔DEV로 교차 적합했다.

**인과 단계 진출 조건(사전 등록):** 다음을 모두 만족해야 한다.
- recall ≥ 0.25
- recall ≥ 2 × max(Gp, Lp) recall
- 적중 영상 ≥ 4
- REP에서 고정한 순위화기와 threshold로 DEV 진짜 부재 frame의 FP ≤ 0.10

**결과 요지** (\`STAGE1_DIAGNOSTIC_RESULTS.md\`, 2026-10-14 기준)
- recall이 높은 생성기(A_cos, B_att 등 약 0.39)는 원시 FP가 약 1.0이다. 거의 모든 빈 frame에서도 "있다"고 답한다.
- 적중도 drone과 kite-surf 두 영상에 몰렸다.
- held-out 순 이득이 양수인 순위화기가 없었다. 그래서 진출 조건을 통과한 계열은 없다.
- Stage 2의 rescue arm은 이 결론을 공개한 '진단적 인과 실행'이다.

**주의:** 생성기 약 25개와 짝 비교 약 10개를 같은 개발 영상에서 돌렸다.
 모든 Stage 1 결과는 탐색적이다. 자세한 내용은 Stage 1 페이지에 있다.
`,
  },

  "overview.s1Units": {
    title: "units / videos / frames",
    body: `
**계산:**
- \`n_units\`: Stage 1 진단 frame 표에 있는 고유 단위 수다. 단위는 한 영상의 한 객체다. DEV 21 + REP 43 = 64.
- \`n_videos\`: 고유 영상 수다.
- \`frames\`: 모든 단위의 frame 행을 합한 수다. prompt frame(0번)은 빼고 1..n-1만 센다. J&F 평가 대상 frame과 같다.

**읽는 법:**
- 이 수는 큰 객체까지 포함한 **전체** 단위다.
- 인과 판정은 이 중 **작은 객체(prompt 면적 < 1%)** 만 쓴다. 2026-10-14 기준 57단위 / 34영상이다. forest plot의 괄호 수와 다른 이유가 이것이다.
- 극소 객체(< 0.1%)는 7단위뿐이다. 극소 객체에 대한 결론은 특히 불확실하다.

**데이터:**
- DEV: DAVIS val, PUMaVOS A/B, SegTrack
- REP: DAVIS 2017 train
- 둘 다 이미 여러 번 본 개발 데이터라 탐색적이다.
- PUMaVOS C는 봉인 상태이고, VKITTI는 Stage 4에서만 쓴다.
`,
  },

  "overview.s1Coabs": {
    title: "co-absence(동시 부재) 트리거 frame 수",
    body: `
**정의:** 살아 있는 SGGL 궤적에서 **G와 L의 object score가 모두 ≤ 0**인 frame이다. SGGL 출력이 빈다. trigger는 정답을 보지 않고 실제 운영에서 볼 수 있는 값만으로 정한다(인과적).

**구성:**
- 큰 숫자: \`trigger_counts.coabs\`. 동시 부재 frame 총수다.
- **GT 있음** (\`coabs_gt_present\`): 대상이 실제로 보이는데 둘 다 놓쳤다. rescue가 **고쳐야 할** frame이다. recall의 분모가 된다.
- **GT 없음** (\`coabs_gt_empty\`): 대상이 정말 가려졌거나 화면 밖이다. rescue는 여기서 **침묵해야** 한다. 여기서 무언가를 출력하면 거짓 양성(거짓 재획득)이다.

**읽는 법:**
- 예시(2026-10-14 기준): 486 = GT 있음 131 + GT 없음 355.
- **동시 부재 frame의 다수가 진짜 부재다.** 그래서 무조건 다시 찾는 전략은 FP를 크게 늘린다. M8이 거짓 양성률을 1차 지표로 사전 등록한 이유다.
- 131개 frame을 모두 완벽히 고쳐도 작은 객체 평균은 +0.028이다(Stage 0 상한). 이것이 rescue 방식의 천장이다.
- 이 frame들은 14단위 / 11영상에 몰려 있다. 효과도 소수 영상에 집중될 수밖에 없다.

**연구 질문과의 관계:** M7의 가장 큰 실패 범주이고, M8 전체의 표적이다. crop 제어 방식(FLOWVEL)은 이런 frame이 *생기지 않게* 한다. \`STAGE2_CAUSAL_RESULTS.md\`에 따르면 GT 있는 동시 부재가 131에서 63으로 줄었다(같은 기계 기준, 2026-10-14).
`,
  },

  "overview.s1OracleRecall": {
    title: "oracle any-generator recall",
    body: `
**계산** (\`oracle_any_generator_recall\`)
- 분모: GT가 있는 동시 부재 frame(131개).
- 분자: **모든 생성기의 모든 후보 중 하나라도** J > 0.5인 frame 수.
- 단위, 영상, frame 별로 최고 J를 구한 뒤, 0.5를 넘는 비율이다.

**읽는 법:**
- "완벽한 순위화기(정답을 아는 선택)가 있다면 rescue로 회수할 수 있는 frame의 최대 비율"이다.
- 예: 0.51이면 놓친 frame의 약 절반은 25개 생성기 중 어딘가에 쓸 만한 후보가 있었다는 뜻이다. 나머지 절반은 어떤 생성기도 찾지 못했다.

**주의:**
- **배포할 수 없는 상한이다.** 실제로는 GT 없이 후보를 골라야 한다.
  - recall이 높은 생성기는 GT가 빈 frame에서도 거의 항상 후보를 낸다(원시 FP ≈ 1.0).
  - Stage 1에서는 held-out 순 이득이 양수인 순위화기가 없었다.
- 적중이 drone, kite-surf 같은 소수 영상에 몰린다. 비율 하나로 일반성을 판단하지 말라.
- frame 단위 recall일 뿐이다. 인과 rollout에서의 J&F 이득으로 바로 옮겨지지 않는다.
  - 회수한 다음 frame에서 다시 놓칠 수 있다.
  - 메모리에 기록(ADMIT)하면 잘못된 대상이 굳을 수 있다.
  - 인과 상한은 forest plot의 ORACLE_POOL을 보라.
`,
  },

  "overview.s1RefRecall": {
    title: "reference recall (max Gp/Lp)",
    body: `
**계산** (\`reference_recall_max_Gp_Lp\`)
- M7의 보호 메모리 probe인 **Gp**(G 상태)와 **Lp**(L 상태)의 recall 중 큰 값이다.
- recall의 정의는 같다. GT가 있는 동시 부재 frame에서 후보 J > 0.5인 비율이다.
- 같은 실행, 같은 frame에서 잰다.

**왜 있는가:**
- 새 생성기가 "이미 있던 단순한 방법"보다 확실히 나은지 보는 기준선이다.
- 사전 등록한 진출 조건 중 하나가 **recall ≥ 2 × 이 값**이다. 나머지 조건은 recall ≥ 0.25, 적중 영상 ≥ 4, FP ≤ 0.10이다.

**읽는 법:**
- 예시(2026-10-14 기준, \`STAGE1_DIAGNOSTIC_RESULTS.md\`): Gp 0.069, Lp 0.061이다. 그래서 기준은 약 0.14 이상이다.
- 값이 낮다는 것은 M7 방식으로는 놓친 frame을 거의 회수하지 못했다는 뜻이다.
- 새 생성기 다수가 recall에서는 이 기준을 넘었다. 그러나 FP 조건과 순위화기 조건에서 막혔다.
`,
  },

  "overview.s1Latency": {
    title: "SGGL latency / frame과 rescue 비용",
    body: `
**계산** (\`latency_s\`)
- **큰 숫자:** \`sggl_per_frame\`. Stage 1 진단 실행에서 frame마다 잰 SGGL 한 step(G + L)의 평균 시간(초)이다.
- **아래 줄:** \`rescue_all_generators_per_trigger\`. trigger frame 하나에서 **모든 생성기 후보(약 31 probe)** 를 돌리는 데 걸린 평균 시간이다.

**읽는 법과 주의:**
- 이 값들은 **진단 실행의 측정값**이다. 깨끗한 성능 벤치마크가 아니다.
  - Ubuntu RTX 2080(53단위)과 RunPod RTX 4090(11단위)이 섞여 있다.
  - 진단 기록 부담과 GPU 동시 사용 경합이 포함된다.
  - 인과 arm 비교의 지연은 같은 기계의 SGGL과 따로 보고한다(Causal arms 페이지). 예: RunPod에서 SGGL 약 0.37 s, FLOWVEL 약 0.33 s(\`STAGE2_CAUSAL_RESULTS.md\`, 측정 시간대가 달라 경합 차이 있음).
- "rescue(all)"는 모든 후보를 돌리는 **진단 비용**이다. 배포형 rescue는 동시 부재 frame에서만 SAM 2를 1번 더 호출한다(평균 +0.1 호출/frame).
- 비용 기준(사전 등록 보고 항목): 지연, 최대 VRAM, 추가 SAM 2 호출 수.
- 광류(RAFT-small) 비용 \`flow_per_frame\`은 JSON에 따로 있다. 2080 단독 측정으로 약 0.03–0.05 s다.
`,
  },

  "overview.s1Vram": {
    title: "peak VRAM",
    body: `
**계산:** Stage 1 진단 실행 전체에서 기록한 \`peak_vram_bytes\`의 최댓값을 1e9로 나눈 값(GB)이다. 모든 단위와 frame에 걸친 최댓값이다. 생성기 probe를 모두 돌린 상태의 최대치다.

**읽는 법:**
- 기준 GPU인 Ubuntu RTX 2080은 8 GB다. 그 안에 들어가면 이 프로젝트의 원본 기계에서 재현할 수 있다.
- 진단은 생성기를 모두 동시에 다루므로 배포형 arm보다 크다.
  - 인과 arm은 1.3–1.45 GB였다. ORACLE은 31 probe 때문에 약 2.3 GB였다(\`STAGE2_CAUSAL_RESULTS.md\`, 2026-10-14 기준).
- 최댓값 하나라 일시적 피크를 반영한다. 평균 사용량이 아니다.
`,
  },
}
