import type { HelpMap } from "./types"

// keys: "m9.<element>"
// Sources: outputs/milestone9/preregistration.json, outputs/milestone9/arms.json, outputs/milestone9/TASKS.json,
// outputs/milestone9/manifests/*_units.json, docs/m9/*.md, scripts/m9_analysis.py (stage1/taxonomy/causal),
// scripts/m9_manifest.py, scripts/m8.py (run_arm, M9 columns), src/spatial/motion.py (phase_shift, ScaleCtl).

const TAX_TABLE = `
| 순서 | 범주 | 조건 (사전 등록 그대로) |
|---|---|---|
| 1 | OBSERVABILITY (관측 불가) | GT 면적 < prompt 면적 × 0.25, 또는 GT 면적 < 64 px |
| 2 | SCALE (스케일) | GT 긴 변 > crop 변 × 0.6, 또는 GT 면적 > prompt 면적 × 4 |
| 3 | POSITION (위치) | GT 중 crop 안에 든 비율 < 0.5 |
| 4a | PRESENCE_ABSENT_CLAIM (부재 주장) | 위 셋이 아니고, SGGL 출력이 비어 있음 |
| 4b | PRESENCE_WRONG_TARGET (오대상) | 위 셋이 아니고, 출력이 비어 있지 않음 |

위에서부터 차례로 보고, 처음 맞는 범주 하나에만 넣는다. 그래서 작아진 데다 crop 밖에도 있는 frame은 OBSERVABILITY로 간다.`

export const help: HelpMap = {
  "m9.about": {
    title: "M9 교차 데이터셋 검증이란",
    body: `M6–M8은 같은 영상(DAVIS, PUMaVOS A/B, SegTrack)을 여러 번 보면서 방법을 골랐다. 그래서 거기서 나온 결과는 모두 탐색적이다. M9은 **한 번도 보지 않은 데이터셋**에서 같은 질문을 다시 던진다.

**질문 세 가지** (\`docs/m9/RESEARCH_PLAN_AND_PREREGISTRATION.md\`)
1. M8에서 가장 강했던 FLOWVEL(두 view가 놓친 동안 crop을 운동에 맞춰 옮김)이 새 데이터에서도 이득을 내는가?
2. 새 데이터에서 SGGL이 놓치는 frame의 주된 원인은 무엇인가? 위치, 크기, 존재·신원, 관측 불가 중에서 찾는다.
3. 그 원인에 맞춘 최소한의 메커니즘(SCALE, JOINT, PCCARRY)이 SGGL보다, 그리고 더 단순한 부모 arm보다 나은가?

**데이터:** MOSEv2 train 139단위/102영상, VOST val 74단위/44영상. LVOS v2는 받지 못했다(BLOCKED). PUMaVOS C는 봉인 상태 그대로다.

**이 페이지:** 위에서부터 실행 현황, 인과 결과, 실패 분류, Stage 1, 사전 등록 순서다. 아직 나오지 않은 결과는 NOT_RUN 또는 IN_PROGRESS로 표시한다. 파일이 생기면 30초 안에 화면에 반영된다.`,
  },
  "m9.tasks": {
    title: "M9 작업 패널",
    body: `\`outputs/milestone9/TASKS.json\`을 읽는다. 형식은 M8 작업 패널과 같다.

- **상태**는 에이전트가 직접 고친다. 자동으로 바뀌지 않으므로, 실제 진척은 진행률 막대와 로그로 확인한다.
- **진행률**은 \`progress.glob\`에 맞는 결과 파일 수를 서버가 요청마다 세어 \`total\`로 나눈 값이다. 예를 들어 MOSEv2 진단은 \`runpod/diag/mosev2/*.cand.csv\` 수 / 139다.
- **로그**는 \`outputs/milestone9/*.log\`, \`outputs/milestone9/runpod/logs/*.log\` 중 최근에 바뀐 4개의 끝부분이다.

M9-DATA-LVOS는 Google Drive 할당량 때문에 BLOCKED다. 데이터셋 표의 LVOS v2 "?"를 참고.`,
  },
  "m9.datasets": {
    title: "데이터셋 범위 표",
    body: `\`outputs/milestone9/manifests/<데이터셋>_units.json\`에서 바로 읽은 숫자다.

| 열 | 뜻 |
|---|---|
| 훑은 영상 | manifest 규칙을 적용해 본 영상 수 (\`n_videos_scanned\`) |
| 평가 단위 | 규칙을 통과한 (영상, 객체) 쌍 수 (\`n_units\`). 모두 작은 객체(prompt 면적 < 1%)다 |
| 단위가 있는 영상 | 단위가 하나 이상 나온 영상 수. 통계는 영상 단위로 묶으므로 이 숫자가 실제 표본 크기에 가깝다 |
| dev / conf | 영상 해시로 나눈 두 분할의 단위 수 |
| 극소 | prompt 면적 < 0.1%인 단위 수 |
| 총 frame | 단위별 prompt 이후 frame 수의 합 |
| 커지는 객체 | 최대 면적이 prompt 면적의 2배를 넘는 단위 수 (\`growth_max > 2\`) |
| 부재 frame 있는 단위 | GT가 빈 frame이 하나라도 있는 단위 수 |

manifest가 없는 데이터셋(LVOS v2)은 사전 등록 파일의 상태 문장을 그대로 보여 준다. PUMaVOS C는 봉인이라 표에 넣지 않는다.`,
  },
  "m9.ds.mosev2": {
    title: "MOSEv2",
    body: `**무엇:** 복잡한 장면의 영상 객체 분할 데이터셋 MOSE의 2판이다. 가림, 사라졌다 다시 나타나는 객체, 비슷한 물체가 많은 장면이 많다. 2025-08에 공개됐다.

**왜 쓰나:** SAM 2.1(2024)보다 늦게 나와서 학습에 들어갔을 수 없다. 다만 MOSEv1 영상(SAM 2 학습에 포함)과 겹치는지는 확인하지 못했다.

**어떻게 받았나** (\`docs/m9/DATASET_AUDIT.md\`)
- 출처: Hugging Face \`FudanCVL/MOSEv2\`. 라이선스 CC BY-NC-SA 4.0, 비상업 연구용.
- valid는 첫 frame 정답만 공개돼 평가할 수 없다. 그래서 frame마다 정답이 있는 **train**을 쓴다.
- train 압축 3개(약 58 GB)를 RunPod에서 한 번 스트리밍하면서, 미리 고른 180영상만 풀었다. 분할 파일 sha256은 공식값과 일치했다.

**표본:** 마스크를 보기 전에 meta(frame 수)만 보고 40–200 frame 영상 2,531개 중 180개를 seed 9로 뽑았다. 그중 139단위/102영상이 규칙을 통과했다.`,
  },
  "m9.ds.vost": {
    title: "VOST",
    body: `**무엇:** Video Object Segmentation under Transformations. 자르기, 껍질 벗기기, 반죽 나누기처럼 **모양이 바뀌거나 쪼개지는 물체**를 1인칭 시점으로 찍은 데이터셋이다. 원본 영상은 Ego4D와 EPIC-KITCHENS에서 왔다.

**정답 frame:** 5 fps로 정답이 달린 frame만 쓴다(정답 보간 없음). 그래서 frame 사이 시간 간격이 다른 데이터셋보다 길다.

**어떻게 받았나:** 공식 S3 zip에 byte-range 요청을 보내 val 정답 전부(70영상, 7,820 PNG)와 단위가 있는 44영상의 JPEG 4,935장만 받았다. 라이선스 CC BY-NC-SA 4.0.

**표본:** val 70영상을 모두 훑어 74단위/44영상을 고정했다.

**참고:** Ubuntu에서 VOST 생성기 진단을 시작했다가 첫 단위에서 118 frame 중 114 frame에 trigger가 걸려 단위당 4.6분이 걸렸다. 그래서 중단하고 SGGL 기준선과 FLOWVEL arm만 돌린다. 중단된 결과는 \`outputs/milestone9/_aborted/\`에 있다.`,
  },
  "m9.ds.lvosv2": {
    title: "LVOS v2 (BLOCKED)",
    body: `**무엇:** Long-term Video Object Segmentation 2판. 영상이 길고 객체가 늦게 등장하거나 오래 사라졌다 돌아오는 경우가 많다. manifest 규칙 1번(처음 보이는 frame부터 시작)은 이런 늦게 나오는 객체를 위해 넣었다.

**왜 막혔나:** 공식 경로가 Google Drive 링크뿐이다. 처음 받기는 디스크 위치 문제로 직접 중단했고, 다시 시도했을 때 Drive가 "Too many users have viewed or downloaded this file recently"로 거부했다. 할당량이 풀릴 때까지 받을 수 없다.

**지금 상태:** RunPod에서 20분 간격으로 최대 6번 자동 재시도한다. 끝내 실패하면 \`BLOCKED_USER_ACTION\`으로 두고, 사람이 브라우저로 받거나 Baidu 경로를 써야 한다. meta JSON 3개만 받아 둔 상태다.

LVOS 없이도 FLOWVEL 일반화 기준은 2개 데이터셋으로 판정할 수 있다.`,
  },
  "m9.frozenManifest": {
    title: "고정 manifest 규칙",
    body: `**평가 단위는 SAM 2를 그 데이터셋에서 한 번도 돌리기 전에, GT만 보고 정했다.** 결과를 본 뒤 단위를 넣거나 빼서 숫자를 좋게 만드는 일을 막기 위해서다. 파일이 이미 있으면 \`scripts/m9_manifest.py\`가 덮어쓰기를 거부한다.

**규칙** (M8 정의 그대로)
1. prompt는 대상이 **처음 보이는** 정답 frame
2. prompt 이후 정답 frame이 10개 이상
3. crop 단위: prompt box 한 변의 4배가 frame 짧은 변보다 작아야 함 (\`vcam.crop_side\`가 정의됨). crop이 frame 전체면 SGGL이 비교 대상과 같아지기 때문이다
4. 작은 객체만: prompt 면적 < 1%. 극소는 < 0.1%
5. 영상당 최대 3개 (seed 9 무작위). 한 영상이 결과를 좌우하지 못하게 한다
6. dev/conf는 영상 id 해시로 나눔
7. 정답 파일이 있는 frame만 사용

MOSEv2는 마스크를 보기 전에 meta만으로 180영상 표본을 먼저 고정했다(\`mosev2_extraction_sample.json\`).`,
  },
  "m9.split": {
    title: "dev / conf 분할",
    body: `영상 id의 해시로 단위를 **dev**(개발)와 **conf**(확인) 둘로 나눈다. 한 영상의 객체는 모두 같은 쪽에 들어가서, 같은 장면이 양쪽에 새지 않는다.

| | dev | conf |
|---|---|---|
| MOSEv2 | 62 | 77 |
| VOST | 31 | 43 |

**쓰는 법** (사전 등록 \`use_of_splits\`)
- **M8에서 고정된 arm (SGGL, FLOWVEL, FLOWCARRY):** 더 고칠 것이 없으니 dev와 conf 전부에서 평가하고, 분할별로도 따로 보고한다.
- **M9 새 arm (SCALE, JOINT, PCCARRY):** 파라미터는 출력 전에 고정했다. 메커니즘마다 근거가 있는 수정을 **한 번만** 허용한다. 수정은 dev 결과만 보고 하고, 판정은 conf로만 한다.

인과 표에서 \`mosev2:dev\`처럼 콜론이 붙은 행이 분할별 결과다.`,
  },
  "m9.histArea": {
    title: "prompt 면적 분포 (area0_frac)",
    body: `단위마다 prompt frame에서 GT가 frame 면적의 몇 %인지(\`area0_frac\`)를 로그 구간으로 센다. 구간 폭은 10의 0.25제곱(약 1.78배)이고, x축 라벨은 구간 하한이다.

- 1% 미만만 단위가 되므로 오른쪽 끝은 1% 바로 아래다.
- 0.1% 아래가 극소(tiny)다. MOSEv2는 139단위 중 51개, VOST는 74단위 중 11개가 극소다.

M8 경험상 극소 객체는 SAM 2 출력이 쉽게 비고, 중심 추정이 흔들린다. 그래서 극소 단위의 ΔJ&F는 \`tiny_dJF\`로 따로 보고한다.`,
  },
  "m9.histGrowth": {
    title: "최대 성장 분포 (growth_max)",
    body: `\`growth_max\` = 그 단위에서 가장 큰 GT 면적 / prompt 면적. x축은 2배 간격 구간의 하한이다(×1, ×2, ×4 …).

**왜 보나:** SGGL의 crop 크기는 prompt box에 맞춰 고정된다. 객체가 크게 자라면 crop 안에 다 들어가지 않아 SCALE 범주 실패가 생길 수 있다. 실패 분류에서 SCALE 조건 하나가 "GT 면적 > prompt 면적 × 4"다.

manifest 기준으로 2배 넘게 커지는 단위는 MOSEv2 139개 중 85개, VOST 74개 중 36개다(\`DATASET_MANIFEST_AND_SPLITS.md\`). SCALE arm이 이 문제를 겨냥한다.`,
  },
  "m9.runs": {
    title: "실행 현황",
    body: `결과 CSV 수를 데이터셋별로 센 값이다. 분모는 manifest의 단위 수다.

| 줄 | 파일 위치 | 뜻 |
|---|---|---|
| runpod/diag | \`runpod/diag/<데이터셋>/*.csv\` | RunPod SGGL 궤적 + Stage 1 진단. RunPod arm의 기준선 |
| ubuntu/SGGL | \`runs/SGGL/*.csv\` | Ubuntu SGGL 궤적. Ubuntu arm의 기준선 |
| runpod/<arm>, ubuntu/<arm> | \`runpod/runs/<arm>/\`, \`runs/<arm>/\` | 인과 arm 실행 결과 |

\`.cand.csv\`(후보 표)는 세지 않는다. 각 arm은 **같은 기계**의 SGGL과만 비교한다(machine-matched). GPU와 라이브러리 버전이 달라 같은 알고리즘도 기계마다 숫자가 조금 다르기 때문이다.`,
  },
  "m9.prereg": {
    title: "사전 등록",
    body: `\`outputs/milestone9/preregistration.json\`은 새 데이터에서 SAM 2 출력이 하나도 나오기 전에 고정했다(\`created_before_any_m9_sam2_output_on_new_data: true\`). 판정 기준, 분류 규칙, 통계 방법을 미리 적어 두어 결과를 본 뒤 기준을 바꾸지 못하게 한다.

이 카드는 그 파일을 그대로 보여 준다. 새 arm 세 개의 구체적인 파라미터는 \`arms.json\`에서 온다(고정 2026-10-14 23:13 KST).

규칙을 바꾸고 싶어지면 그건 새 사전 등록이 필요하다는 뜻이다. 같은 데이터에서 threshold를 여러 개 시도해 고르는 것은 금지(\`kill_rules\`의 no threshold fishing).`,
  },
  "m9.armsTable": {
    title: "M9 arm 목록",
    body: `| arm | 하는 일 | 고정 |
|---|---|---|
| SGGL | 기준 시스템. 전체 frame(G) + 고정 크기 crop(L) | M7 |
| FLOWVEL | crop 중심을 출력 중심 + 0.5 × RAFT 광류 변위로 옮김. 출력이 비면 광류 box를 따라감 | M8 |
| FLOWCARRY | 출력이 빌 때만 광류 box를 따라감 (보조 ablation) | M8 |
| PCCARRY | 출력이 빌 때만 위상 상관 전역 이동만큼 crop을 옮김. RAFT 없음 | M9 |
| SCALE | crop 크기를 최근 출력 box에 맞춰 바꿈. 중심은 SGGL과 같음 | M9 |
| JOINT | FLOWVEL 중심 + SCALE 크기 | M9 |

각 arm 이름 옆 "?"에 자세한 정의가 있다.`,
  },
  "m9.arm.SGGL": {
    title: "SGGL (기준선)",
    body: `현재 기준 알고리즘이다. 고정된 SAM 2.1 Hiera-T를 전체 frame(G)과 대상 주변 정사각 crop(L)에 동시에 돌리고, self gate로 둘 중 하나의 마스크를 고른다. crop 한 변은 prompt box 긴 변의 4배(최소 min(H,W)/4)로 고정이고, 중심은 직전 출력을 따라간다. 출력이 비면 중심을 그대로 둔다.

M9에서는 모든 arm을 같은 기계의 SGGL 궤적과 비교한다. RunPod에서는 Stage 1 진단 행(\`runpod/diag/<데이터셋>/\`), Ubuntu에서는 \`runs/SGGL/\`이 그 궤적이다.`,
  },
  "m9.arm.FLOWVEL": {
    title: "FLOWVEL (M8 고정, 변경 없음)",
    body: `- 다음 crop 중심 = 현재 출력 중심 + 0.5 × (대상 영역의 RAFT-small 광류 변위)
- 출력이 비면 광류로 운반한 대상 box를 따라간다
- SAM 2 추가 호출 없음. frame마다 RAFT-small 1회

M8에서 고정(2026-10-14 16:14)한 그대로 쓰고, 새 데이터의 dev+conf 전체 단위에서 평가한다. M8 결과는 +0.0130 [+0.0010, +0.0285]로 PROMISING이었지만, 반복해서 본 데이터라 탐색적이었다. M9가 이것을 독립 데이터로 확인하는 자리다. 판정 기준은 "FLOWVEL 일반화 기준"의 "?" 참고.`,
  },
  "m9.arm.FLOWCARRY": {
    title: "FLOWCARRY (보조)",
    body: `M8의 사후 ablation이다. 속도 항 없이, **출력이 빈 동안에만** 광류로 운반한 box를 따라 crop을 옮긴다. 출력이 있을 때는 SGGL과 같다.

FLOWVEL의 이득이 "놓친 동안 따라가기"에서 오는지 보는 비교용이다. 사전 등록에서 보조(secondary)로 분류돼 주 판정에는 쓰지 않는다.`,
  },
  "m9.arm.PCCARRY": {
    title: "PCCARRY (M9 고정, 값싼 운동)",
    body: `**정의** (\`arms.json\`, \`src/spatial/motion.py\` \`phase_shift\`)
- SGGL 출력이 빈 동안에만, frame t−1과 t 사이의 **전역 이동량**만큼 crop을 옮긴다.
- 이동량은 위상 상관(phase correlation)으로 구한다. 회색조로 바꾸고 4배 줄인 두 frame의 FFT를 비교해 한 개의 (dx, dy)를 얻고, 원래 해상도 px로 되돌린다.
- 학습된 모델이 없고 CPU에서 몇 ms면 된다. 출력이 있으면 SGGL 중심을 쓴다.

M8의 CAMCARRY(카메라 운동만큼 이동)를 RAFT 없이 흉내 낸 것이다. 실행 행의 \`pc_dx\`, \`pc_dy\`가 그 frame의 이동량이다.

**성공 기준:** 같은 단위에서 FLOWVEL ΔJ&F의 70% 이상을 내면서 지연은 FLOWVEL보다 작아야 한다.`,
  },
  "m9.arm.SCALE": {
    title: "SCALE (M9 고정, 크기만)",
    body: `**정의** (\`arms.json\` scale 파라미터)
- crop 한 변 = 4 × (최근 믿을 만한 출력 box의 긴 변) (\`k = 4\`)
- EMA로 부드럽게 따라간다 (\`alpha = 0.3\`)
- 한 frame에 바뀌는 폭은 ×0.8 ~ ×1.25로 제한 (\`lo\`, \`hi\`)
- 하한은 SGGL과 같은 min(H,W)/4 (\`max_zoom = 4\`), 상한은 min(H,W)
- 출력이 비거나 믿기 어려우면 크기를 그대로 둔다
- 중심은 SGGL과 같다

극소 객체는 하한에 걸려 SGGL과 같은 crop을 쓰게 된다. 객체가 커져 crop을 넘치는 경우(실패 분류의 SCALE 범주)를 겨냥한다.

**성공 기준:** SCALE 범주로 분류된 놓친 frame을 SGGL보다 30% 이상 줄이고, dev에서 ΔJ&F ≥ 0이며, −0.05 미만 단위가 새로 생기지 않을 것. 그다음 conf에서 확인한다.`,
  },
  "m9.arm.JOINT": {
    title: "JOINT (M9 고정, FLOWVEL + SCALE)",
    body: `중심은 FLOWVEL(광류 따라가기), 크기는 SCALE 컨트롤러(같은 파라미터: k 4, alpha 0.3, ×[0.8, 1.25])로 정한다.

**시너지 기준:** JOINT − max(SCALE, FLOWVEL)의 영상 군집 CI 하한 > 0. 둘을 합친 것이 더 나은 쪽 하나보다도 확실히 좋아야 한다. 아니면 더 단순한 부모를 쓴다.`,
  },
  "m9.stage1Gates": {
    title: "Stage 1 진출 기준",
    body: `Stage 1은 SGGL이 놓친 frame에서 후보 생성기가 대상을 찾을 수 있는지 **인과 실행 없이** 보는 진단이다. 기준을 넘은 것만 다음 단계로 간다.

| 관문 | 조건 |
|---|---|
| search | GT가 있는 동시 부재 frame에서 recall ≥ 0.25 이고, 적중 영상 ≥ 4 |
| attention | B_att가 A_cos보다 짝지은 recall/hit@1에서 나아야 함 (영상 군집 CI > 0). 아니면 attention 계열 중단 |
| ranking | 데이터셋·영상 기준 교차 적합한 held-out 순 이득 > 0. 아니면 인과 실행에 쓰지 않음 |
| R3 attention 순위화 | 후보 적중이 독립 영상 20개 이상일 때만 시도 |

생성기는 11개: A_cos#0–2, B_att#0–2, C_flowcrop, C_flowbox, F_oclr8_L, 그리고 참조용 Gp, Lp.`,
  },
  "m9.flowvelCriteria": {
    title: "FLOWVEL 일반화 기준",
    body: `아래를 **모두** 만족해야 "FLOWVEL이 새 데이터로 일반화된다"고 판정한다 (사전 등록 \`flowvel_generalizes_if\`).

1. 새 데이터 전체(pooled)의 영상 군집 bootstrap CI 하한 > 0
2. 2개 이상 데이터셋에서 평균 > 0. 데이터셋이 하나뿐이면 그 데이터셋의 Holm 보정 CI > 0
3. +0.01 넘게 오른 영상이 5개 이상이고, 데이터셋이 2개 이상이면 2개 데이터셋에 걸쳐 있을 것
4. 상위 2개 영상이 전체 이득의 50% 미만
5. −0.05 미만으로 떨어진 단위가 1개 이하

데이터셋별 FLOWVEL 검정은 Holm으로 다중 비교를 보정한다. 인과 표의 \`Holm p\` 열이 그 값이다.

이 기준은 M8에서 본 "몇 개 영상이 이득을 다 만드는" 패턴을 걸러 내려고 넣었다.`,
  },
  "m9.newArmCriteria": {
    title: "새 arm 판정과 중단 규칙",
    body: `| arm | 성공 조건 |
|---|---|
| SCALE | SCALE 범주 놓친 frame −30% 이상, dev ΔJ&F ≥ 0, 새로운 −0.05 미만 단위 없음 → conf에서 확인 |
| JOINT | JOINT − max(SCALE, FLOWVEL)의 CI 하한 > 0 |
| PCCARRY | FLOWVEL ΔJ&F의 70% 이상 + 지연이 FLOWVEL보다 작음 |

**중단(kill) 조건:** 평균이 음수, 더 단순한 부모보다 나쁨, 거짓 재획득이 눈에 띄게 늘어남, −0.05 미만 단위가 생김, 이득이 2개 영상 이하에서만 나옴, 메커니즘 근거가 없음. 하나라도 걸리면 멈춘다.

수정은 메커니즘마다 한 번만, dev에서만 한다.`,
  },
  "m9.endpoint": {
    title: "주 지표와 통계",
    body: `**주 지표:** 데이터셋별, 작은 단위의 짝지은 ΔJ&F = arm − 같은 기계 SGGL. 단위마다 frame 1..n−1의 J&F 평균 차이를 구한다(prompt frame 제외).

**같이 보는 것**
- 영상 군집 bootstrap 95% CI (2,000회). 같은 영상의 단위는 독립이 아니라 영상째로 다시 뽑는다
- 하나씩 빼기(LOVO) 최솟값
- 영상 부호 검정 (|d| > 0인 영상만)
- 상위 1·2 영상의 이득 비중, 최악 단위, −0.05 미만 단위 수
- J와 F 따로, 극소 분층, 동시 부재 frame 회복, 거짓 재획득, 지연, VRAM

**다중 비교:** 데이터셋별 FLOWVEL 검정에 Holm 보정.`,
  },
  "m9.stage1": {
    title: "Stage 1 최소 진단 (데이터셋별)",
    body: `\`outputs/milestone9/reports/stage1_<데이터셋>.json\` (\`scripts/m9_analysis.py stage1\`).

**막대:** 생성기 계열별
- **recall**: GT가 있는 동시 부재 frame 중, 그 생성기 후보 하나라도 J > 0.5인 비율
- **원시 FP**: GT가 없는 동시 부재 frame 중, 후보가 "있음"(score > 0, 면적 > 0)을 낸 비율. 순위화 전 값이다

**표의 열:** \`mean_best_J\`(frame별 최고 J의 평균), \`videos_hit\`(적중 영상 수), \`n_present\`/\`n_empty\`(GT 있음/없음 frame 수).

**위 숫자:** \`oracle_any_generator_recall\`은 어떤 생성기든 하나라도 맞춘 비율이다. 순위화가 완벽할 때의 상한이다. \`trigger_counts\`는 trigger 종류별 frame 수.

**지금 상태:** MOSEv2는 RunPod에서 진단 중이다. VOST 생성기 진단은 너무 느려 중단했다(단위당 4.6분). VOST 리포트가 생기면 후보 없이 SGGL 궤적만으로 만든 것이다.`,
  },
  "m9.pairedTests": {
    title: "짝지은 검정",
    body: `같은 frame에서 두 생성기의 적중 여부(0/1)를 빼서 평균을 낸다. CI는 영상 군집 bootstrap이다. CI가 0을 벗어나면 파랑(왼쪽이 나음)이나 빨강(오른쪽이 나음), 아니면 회색이다.

| 검정 | 보는 것 |
|---|---|
| B_att_vs_A_cos_recall_top3 | attention 후보가 cosine 후보보다 나은가 (attention 관문) |
| B_att#0_vs_A_cos#0_hit1 | 1순위 후보끼리 비교 |
| C_flowcrop_vs_Lp | 광류 crop이 L view 참조보다 나은가 |
| F_oclr8_L_vs_Lp | OCLR 8-방향 crop이 L view 참조보다 나은가 |`,
  },
  "m9.ruleReplay": {
    title: "score > 0 규칙 재생",
    body: `학습 없이 "후보 점수가 0보다 크면 채택"하는 규칙을 생성기 계열마다 기록된 후보에 다시 적용한 결과다. 인과 실행이 아니라 frame 단위 재생이고, J만 본다.

| 열 | 뜻 |
|---|---|
| accepted | 채택한 frame 수 |
| rescued | 채택했고 GT가 있으며 J > 0.5인 frame |
| fp | 채택했는데 GT가 없는 frame |
| net_dJ_frames | (GT 있는 채택 frame의 J 합) − fp. 0보다 커야 쓸모가 있다 |
| videos_rescued | rescued가 나온 영상 수 |`,
  },
  "m9.taxonomy": {
    title: "실패 분류 (H4 결정 관문)",
    body: `**목적:** 새 데이터에서 SGGL이 왜 놓치는지 정하고, 다음 실험(SCALE, FLOWVEL, PCCARRY 등)을 고른다.

**대상 frame:** 같은 기계 SGGL 궤적에서 **GT가 있는데 J < 0.5**인 frame. 손실은 frame마다 (1 − J&F)다.
${TAX_TABLE}

"crop 변"과 "crop 안 비율"은 그 frame에서 SGGL이 실제로 쓴 L crop 기준이다. GT box 크기는 \`outputs/milestone9/gtbox/<데이터셋>.csv\`에 미리 계산해 둔다.

**막대:** 데이터셋마다 범주별 손실 비중을 쌓았다. 괄호 안이 지배 범주다.

**표의 열:** \`frames\`(frame 수), \`loss\`(손실 합), \`loss_share\`, \`units\`, \`videos\`, \`coabs_frames\`(그중 동시 부재 frame), \`top_video_share\`(손실이 가장 큰 영상 하나의 비중. 높으면 한 영상 이야기다).`,
  },
  "m9.dominant": {
    title: "지배 범주 (dominant)",
    body: `한 범주가 놓친 손실(1 − J&F 합)의 **50% 이상**을 차지하고, 그 손실이 **3개 이상 영상**에 걸쳐 있으면 그 범주가 지배 범주다. 손실이 큰 범주부터 이 조건을 본다.

어느 범주도 조건을 못 채우면 **MIXED**다. MIXED면 한 메커니즘으로 큰 개선을 기대하기 어렵다는 뜻이다.

영상 3개 조건은 한 영상의 특이한 실패가 결론을 정하지 못하게 하려는 것이다.`,
  },
  "m9.falseReacq": {
    title: "거짓 재획득 frame",
    body: `GT가 비어 있는데(대상이 실제로 없음) SGGL이 무언가를 출력한 frame 수다. 분석 코드에서는 GT가 빈 frame 중 J < 1인 frame으로 센다. 둘 다 비어 있으면 J = 1이다.

비슷한 다른 물체를 대상으로 잘못 잡는 경우가 대부분이다. 새 arm이 놓친 frame을 회복하더라도 이 숫자를 늘리면 안 된다(중단 규칙).`,
  },
  "m9.cat.OBSERVABILITY": {
    title: "OBSERVABILITY (관측 불가)",
    body: `**조건:** GT 면적 < prompt 면적 × 0.25, 또는 GT 면적 < 64 px. 분류 순서 1번이라 다른 조건보다 먼저 본다.

대상이 너무 작아졌거나 대부분 가려진 경우다. SAM 2 입력(1024)에서 몇 cell도 안 되는 크기라, crop을 어디로 옮기든 회복하기 어렵다. 이 범주가 크면 crop 제어 메커니즘으로는 얻을 것이 적다.`,
  },
  "m9.cat.SCALE": {
    title: "SCALE (스케일)",
    body: `**조건:** GT 긴 변 > crop 변 × 0.6, 또는 GT 면적 > prompt 면적 × 4. (관측 불가가 아닐 때)

대상이 커져 고정 크기 crop에 다 들어가지 않거나 가장자리에 걸린 경우다. VOST처럼 물체가 펼쳐지거나 카메라가 다가가는 장면에서 생긴다. SCALE arm이 이 범주를 줄이는 것이 목표다(30% 이상 감소가 성공 기준).`,
  },
  "m9.cat.POSITION": {
    title: "POSITION (위치)",
    body: `**조건:** GT 중 crop 안에 든 비율 < 0.5. (관측 불가도 스케일도 아닐 때)

크기는 적당한데 crop이 대상을 놓쳐 다른 곳을 보고 있는 경우다. 대상이 빨리 움직이거나 카메라가 흔들릴 때 생긴다. FLOWVEL과 PCCARRY가 겨냥하는 범주다.`,
  },
  "m9.cat.PRESENCE_ABSENT_CLAIM": {
    title: "PRESENCE_ABSENT_CLAIM (부재 주장)",
    body: `**조건:** 관측 가능, 크기 적당, crop 안에 있음. 그런데 SGGL 출력이 **비어 있다**.

분석 코드는 "L이 gate에서 거절되고 G score ≤ 0"을 출력이 빈 것으로 본다. 대상이 보이는 자리에 있는데 SAM 2가 "없다"고 판단한 경우다. crop 위치나 크기를 바꿔도 해결되지 않고, 존재 판단(object score)이나 메모리 쪽 문제다.`,
  },
  "m9.cat.PRESENCE_WRONG_TARGET": {
    title: "PRESENCE_WRONG_TARGET (오대상)",
    body: `**조건:** 관측 가능, 크기 적당, crop 안에 있음. 출력은 있는데 J < 0.5.

사전 등록은 "출력이 비어 있지 않고 J < 0.1"을 오대상으로 적었다. 분석 코드는 위 세 범주와 부재 주장이 아닌 나머지 놓친 frame(J < 0.5)을 모두 이 범주에 넣는다.

비슷한 다른 물체를 잡았거나 대상의 일부만 잡은 경우다. 신원(identity) 문제라 crop 제어로는 고치기 어렵다.`,
  },
  "m9.causal": {
    title: "M9 인과 결과",
    body: `\`outputs/milestone9/reports/causal.json\` (\`scripts/m9_analysis.py causal <arm>\`).

각 arm을 **같은 기계의 SGGL 궤적**과 단위별로 비교한다. 작은 단위(area0 < 1%)만 집계한다.

**키 구조:** arm마다 \`pooled\`(새 데이터 전체), \`<데이터셋>\`, \`<데이터셋>:<dev|conf>\`.

**기준선 파일 순서** (분석 코드와 같음)
- RunPod arm → \`runpod/diag/<데이터셋>/<unit>.csv\`, 없으면 \`runpod/runs/SGGL/\`
- Ubuntu arm → \`diag/<데이터셋>/<unit>.csv\`, 없으면 \`runs/SGGL/\`

단위별 값은 \`reports/causal_units_<arm>.csv\`에 있다. 표 아래 링크로 열 수 있고, Unit 탐색에서도 단위별로 보인다.

판정은 "FLOWVEL 일반화 기준"과 "새 arm 판정" 카드의 기준을 따른다. 이 페이지는 숫자만 보여 주고 판정 라벨을 붙이지 않는다.`,
  },
  "m9.forest": {
    title: "forest plot 읽는 법",
    body: `한 줄이 (arm, 범위) 하나다. 점은 단위 평균 ΔJ&F, 수염은 영상 군집 bootstrap 95% CI다. 세로선이 0이다.

- 색: 검정 = 전체(pooled), 파랑 = MOSEv2, 주황 = VOST
- 빈 마름모: dev/conf 분할 행. 분할 행은 표본이 작아 CI가 넓다
- 수염 전체가 0 오른쪽이면 그 범위에서 arm이 SGGL보다 낫다는 근거가 있다. 다만 일반화 판정은 CI 하나가 아니라 기준 5개를 모두 본다`,
  },
  "m9.col.n_units": { title: "units / videos", body: `집계에 들어간 작은 단위 수 / 서로 다른 영상 수. CI와 부호 검정은 영상 단위로 계산한다.` },
  "m9.col.dJF": { title: "ΔJ&F", body: `단위별 (arm J&F 평균 − 같은 기계 SGGL J&F 평균)의 단위 평균. frame 1..n−1, prompt frame 제외. 양수면 arm이 낫다.` },
  "m9.col.CI_video": { title: "CI (video)", body: `영상 군집 bootstrap 95% 신뢰구간(2,000회). 영상째로 다시 뽑아 같은 영상 단위끼리의 상관을 반영한다. 영상이 하나뿐이면 계산하지 않는다.` },
  "m9.col.dJ": { title: "ΔJ / ΔF", body: `J(영역 IoU)와 F(경계 정확도)를 따로 본 차이. 하나만 오르면 이득의 성격이 다르다. 예를 들어 F만 오르면 경계가 조금 나아진 것이지 대상을 되찾은 것은 아니다.` },
  "m9.col.videos_up": { title: "영상 ↑/↓ 0.01", body: `영상 평균 Δ가 +0.01보다 큰 영상 수 / −0.01보다 작은 영상 수. FLOWVEL 일반화 기준 3번은 ↑ 영상 5개 이상이다.` },
  "m9.col.sign_test_p": { title: "sign p", body: `영상 부호 검정의 정확 p값(양측). Δ가 0이 아닌 영상만 센다. 효과 크기와 무관하게 "오른 영상이 내린 영상보다 많은가"만 본다.` },
  "m9.col.holm_p": { title: "Holm p", body: `데이터셋별 행에만 있다. 영상 부호 뒤집기 순열 검정 p값(10,000회)에 데이터셋 수만큼 Holm 보정을 한 값이다. 사전 등록의 다중 비교 보정이다.` },
  "m9.col.lovo_min": { title: "LOVO min", body: `영상을 하나씩 빼고 다시 구한 ΔJ&F 중 가장 작은 값. 이 값이 음수면 영상 하나가 결과의 부호를 정하고 있다는 뜻이다.` },
  "m9.col.top2": { title: "상위 2 비중", body: `영상 평균 Δ가 가장 큰 2개 영상에 속한 단위들의 Δ 합 / 전체 단위 Δ 합. 일반화 기준 4번은 0.5 미만이다. 전체 Δ가 0 이하면 계산하지 않는다(—).` },
  "m9.col.worst_unit": { title: "최악 단위", body: `가장 많이 떨어진 단위의 Δ. 마우스를 올리면 단위 이름이 보인다. −0.05 미만이면 중단 규칙에 걸린다.` },
  "m9.col.n_lt": { title: "< −0.05", body: `Δ가 −0.05 미만인 단위 수. FLOWVEL 일반화 기준 5번은 1개 이하, 새 arm은 새로 생기면 안 된다.` },
  "m9.col.tiny_dJF": { title: "tiny ΔJ&F", body: `극소 단위(area0 < 0.1%)만의 ΔJ&F 평균. 극소 단위가 없으면 —.` },
  "m9.col.coabs": { title: "co-abs Δ/frame", body: `기준선 SGGL에서 동시 부재(G, L score 모두 ≤ 0)였던 frame들의 J&F 차이 합 / 그 frame 수. 놓친 동안의 회복을 직접 본다. FLOWVEL과 PCCARRY는 이 값이 올라야 메커니즘 근거가 있다.` },
  "m9.col.fp": { title: "거짓 재획득 arm / SGGL", body: `GT가 빈 frame 중 J < 1인 frame 수를 arm과 기준선에서 각각 센 값. arm 쪽이 눈에 띄게 많으면 중단 규칙에 걸린다.` },
  "m9.col.latency_s": { title: "latency s", body: `arm의 frame당 평균 지연(초). 기계마다 다르므로 같은 기계 arm끼리만 비교한다. PCCARRY 성공 기준은 FLOWVEL보다 작을 것.` },
  "m9.col.peak_vram_gb": { title: "VRAM GB", body: `해당 범위 단위들의 최대 GPU 메모리(GB).` },
  "m9.unit": {
    title: "M9 단위 (MOSEv2 · VOST)",
    body: `선택기 위쪽 그룹이 M9 단위다. 이름은 \`<데이터셋>__<데이터셋>_<영상 id>__obj<k>\` 꼴이다.
- \`mosev2__mosev2_006e6b64__obj1\`: MOSEv2 train 영상 006e6b64의 객체 1
- \`vost__vost_1169_peel_onion__obj1\`: VOST val 영상 1169_peel_onion의 객체 1

괄호 안은 그 단위의 결과가 있는 곳이다. \`runpod/diag\`나 \`ubuntu/SGGL\`만 있으면 아직 기준선만 있고 arm 결과는 없다.

VOST의 frame 번호(video_idx)는 정답 frame 순번이다. 원본 영상의 6 frame 간격(5 fps)이다.`,
  },
  "m9.unitBaseline": {
    title: "M9 기준선 매칭",
    body: `M9 arm은 같은 기계의 SGGL 궤적과 비교한다. 순서는 \`scripts/m9_analysis.py\`와 같다.

| arm이 돈 기계 | 기준선 |
|---|---|
| RunPod | \`runpod/diag/<데이터셋>/<unit>.csv\` (Stage 1 진단 행). 없으면 \`runpod/runs/SGGL/\` |
| Ubuntu | \`diag/<데이터셋>/<unit>.csv\`. 없으면 \`runs/SGGL/<unit>.csv\` |

진단 행도 SGGL 궤적 그대로다. 진단은 후보를 기록만 하고 출력에 개입하지 않는다.`,
  },
  "m9.extraSeries": {
    title: "frame별 crop 진단 열",
    body: `M9 실행 행에 새로 들어간 열이다. 있는 열만 고를 수 있다. 굵은 회색은 기준선 SGGL이다.

| 열 | 뜻 |
|---|---|
| crop_side | 그 frame에서 쓴 L crop의 한 변(px). SGGL은 거의 고정, SCALE/JOINT는 변한다 |
| gt_in_crop | GT 픽셀 중 crop 안에 든 비율. 0.5 미만이면 실패 분류의 POSITION 조건(점선) |
| gt_long_over_side | GT 긴 변 / crop 변. 0.6 초과면 SCALE 조건(점선) |
| pc_dx, pc_dy | t−1 → t 위상 상관 전역 이동(원래 해상도 px). PCCARRY는 출력이 빈 동안 이만큼 crop을 옮긴다 |

RunPod 진단 행에는 \`gt_in_crop\`만 있다. GT가 빈 frame에서는 비율 값이 비어 선이 끊긴다.`,
  },
}
