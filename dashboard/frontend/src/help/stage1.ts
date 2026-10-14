import type { HelpEntry, HelpMap } from "./types"

// keys: "stage1.<element>"
// 출처: scripts/m8.py (generators, diag_unit, trigger), scripts/m8_analysis.py (stage1, rankers), src/spatial/{rescue,motion,ranker,sggl}.py,
// docs/m8/STAGE1_DIAGNOSTIC_RESULTS.md, EXPERIMENT_TOURNAMENT_PLAN.md, EXPANDED_CANDIDATE_PORTFOLIO.md, preregistration_amend{1,2}.json

const SRC = "`outputs/milestone8/stage1_report.json` (생성: `uv run python scripts/m8_analysis.py stage1`)"

const FRAMES = `**대상 frame:** 동시 부재(coabs) trigger frame 중 GT가 있는 frame이 분모다. 2026-10-14 기준 GT 있음 131 frame (11영상), GT 없음 355 frame이다. 진짜 부재 frame이 2.7배 많다는 점이 이후 모든 순위화 결과를 좌우한다.`

const PROBE = `**비저장 probe:** 모든 생성기는 살아 있는 SGGL 궤적의 해당 frame에서 SAM 2를 한 번 더 호출하지만, 그 결과를 메모리에 쓰지 않는다 (\`rescue.probe\`: \`cached_features\`를 잠시 바꿨다가 복원). 따라서 생성기끼리 서로 영향을 주지 않고, SGGL 궤적 자체도 바뀌지 않는다. 그 대신 "후보를 출력했으면 다음 frame이 어떻게 됐을까"(메모리 오염, crop 재배치 효과)는 Stage 1에서 측정되지 않는다. 그건 Stage 2 인과 arm의 몫이다.`

const OUTCOME_NOTE = "수치는 2026-10-14 기준 `stage1_generators.csv` / `STAGE1_DIAGNOSTIC_RESULTS.md` 값이며, 표의 실시간 값이 우선이다."

const help_: HelpMap = {
  "stage1.about": {
    title: "Stage 1 진단이란",
    body: `**무엇을 보여주나:** M8 Stage 1 진단 결과 전체 페이지다. SGGL이 대상을 놓친 frame에서, 25개 "rescue 생성기"가 대상을 다시 찾아낼 수 있는지(생성), 그리고 찾은 후보를 믿을지 말지 정할 수 있는지(순위화·존재 판정)를 분리해서 측정한다.

**어떻게 돌렸나 (\`scripts/m8.py diag\`):**
1. 단위(객체 1개 × 영상 1개)마다 SGGL(G = 전체 frame + P5 prompt 메모리, L = prompt box 4배 정사각 crop, self gate)을 그대로 실행한다. 이것이 "살아 있는 SGGL 궤적"이다.
2. 매 frame t에서 \`trigger(G, L, iou_lg, t)\`로 frame 유형을 정한다:
   - **coabs (동시 부재):** G와 L의 object score가 모두 ≤ 0. SGGL 출력이 비어 있다. **Stage 1의 주 대상.**
   - **disagree (불일치):** 한쪽 score만 > 0이거나, 두 view 마스크의 IoU < 0.5.
   - **regular (정기):** 위 둘이 아니고 t % 10 == 0 인 frame. 정상 추적 중인 쉬운 frame에서 생성기가 대상을 찾는지 보는 대조군.
   - 그 외 frame에서는 생성기를 돌리지 않는다.
3. trigger frame마다 25개 생성기 전부를 **같은 frame에서** 실행해 후보 마스크를 만든다. 그래서 생성기 간 비교가 같은 frame끼리 짝지어진다(paired).
4. 각 후보에 대해 J(GT 대비), SAM 2 object score, 예측 IoU, frame-0 특징과의 코사인 등을 \`*.cand.csv\`에 기록한다. CPU 분석(\`m8_analysis.py stage1\`, \`rankers\`)이 이것을 요약한다.

${PROBE}

**규모 (2026-10-14 기준, \`stage1_report.json\`):** 64단위 / 39영상 / 5082 frame. trigger: coabs 486 (GT 있음 131, GT 없음 355), disagree 242, regular 399. 53단위는 Ubuntu RTX 2080 (\`outputs/milestone8/diag/\`), 11단위는 RunPod RTX 4090 (\`outputs/milestone8/runpod/diag/\`)에서 기록했다.

**연구 질문과의 연결:** M7의 가장 큰 실패 범주가 동시 부재다. Stage 1은 "두 view가 모두 실패할 때 새 증거를 더하면 되살릴 수 있는가"를 싸게(비저장, 짝 비교) 거르는 단계다. 사전 등록된 진출 조건(recall ≥ 0.25, ≥ 2 × max(Gp, Lp), ≥ 4영상, REP에서 고정한 ranker/threshold로 DEV 거짓 양성 ≤ 0.10)을 통과한 생성기만 Stage 2 인과 실행으로 간다. 2026-10-14 기준 **통과한 생성기·순위화기는 없다.** 그래서 RESC_OCLR 등은 "진단적 인과 arm"으로 별도 표시해 돌렸다.

**주의:**
- 모든 Stage 1 결과는 **탐색적**이다. 약 25개 생성기와 10여 개 짝 비교를 같은 개발 영상(DEV + REP)으로 수행했다.
- 사전 등록 일탈 2건이 공개돼 있다: 수정 1 (\`_pt\` 점 prompt 변형 추가), 수정 2 (H_s8 특징 버그 수정). 둘 다 일부 결과를 본 뒤의 결정이다.
- GT가 있는 동시 부재 frame의 적중이 소수 영상(drone, kite-surf)에 몰려 있다. 영상 수 기준으로 보면 표본이 매우 작다.`,
  },

  "stage1.recallChart": {
    title: "생성기 recall vs 원시 거짓 양성",
    body: `**무엇을 보여주나:** 생성기 계열마다 두 막대. 파랑 = recall, 두 번째 색 = GT가 없는 동시 부재 frame에서의 원시 FP 비율. 점선 = 기준 recall (Gp, Lp 중 큰 값). 출처: ${SRC}의 \`generators_coabs\`, \`reference_recall_max_Gp_Lp\`.

**계산 (\`m8_analysis.per_frame\`, \`stage1\`):**
- 후보 이름의 \`#i\` 접미사를 떼어 계열로 묶는다 (\`fam\`: \`A_cos#0\`, \`A_cos#1\`, \`A_cos#2\` → \`A_cos\`). 그래서 A_cos, B_att, H_s8은 frame당 후보 3개(top-3 peak), 나머지는 1개다.
- **recall** = GT 있는 coabs frame 중, 그 계열 후보 중 **하나라도 J > 0.5**인 frame의 비율. 즉 "oracle이 그 계열 안에서 가장 좋은 후보를 골라줬을 때"의 회수율이다.
- **원시 FP (fp_rate_gt_empty)** = GT 없는 coabs frame 중, 그 계열 후보 중 하나라도 "존재"(\`present = score > 0 and area > 0\`)를 낸 frame의 비율. 순위화·threshold를 적용하기 **전**의 값이다.
- **기준선** = 같은 실행에서 Gp, Lp (M7 보호 메모리 probe)의 recall 중 최댓값. 사전 등록 진출 조건 "recall ≥ 2 × 기준"의 기준이다.

${FRAMES}

**왜 점 prompt 생성기의 원시 FP가 ~100%인가:**
- A_cos, B_att, H_s8, G_neg, D_ref, C_flowbox, \`F_*_pt\`는 SAM 2에 양성 점(또는 box)을 준다. SAM 2 decoder는 클릭을 받으면 거의 항상 그 근처에 마스크를 만들고 object score를 양수로 낸다. 2026-10-14 기준 진짜 부재 frame에서 B_att#0, A_cos#0의 score > 0 비율은 1.00이다.
- 즉 점 prompt 후보는 "대상이 있다"는 판단을 스스로 하지 않는다. 존재 판정은 전적으로 뒤의 순위화기·threshold가 맡아야 한다. 그래서 이 막대만 보고 "FP 1.0이니 쓸모없다"고 결론 내리면 안 된다. 사전 등록 수정 1도 이 점을 공개하고, FP 기준은 원시값이 아니라 REP에서 고정한 ranker/threshold 이후 값으로 판정하도록 했다.
- 반대로 prompt 없는 view 생성기(F_oclr8_L, C_flowcrop, E_beam, Lp 등)는 SAM 2 자체의 존재 판정을 그대로 쓰므로 원시 FP가 낮다. 대신 recall도 낮다.

**읽는 법:**
- 오른쪽으로 긴 파랑 + 짧은 FP 막대가 이상적이다. 실제로는 "recall 높음 ↔ FP ~1.0"(점 prompt 계열)과 "FP 낮음 ↔ recall 낮음"(view 계열)으로 갈린다. 예외적으로 F_oclr8_L이 recall 0.176, FP 0.014였다 (2026-10-14 기준). 이것이 RESC_OCLR 진단 arm의 근거다.
- 점선보다 짧은 생성기는 M7의 단순 보호 메모리 probe보다도 못 찾는다는 뜻이다.
- recall은 oracle 선택이다. top-3 계열(A/B/H)은 후보 3개 중 아무거나 맞으면 hit이므로, 후보 1개짜리 계열보다 구조적으로 유리하다.

**주의:** recall 분모 131 frame 중 상당수가 drone(45), kite-surf(23) 두 영상이다. 계열 간 차이가 수 frame 수준이면 한두 영상의 사정일 수 있다. 영상 분포는 아래 히트맵을 보라.`,
  },

  "stage1.locChart": {
    title: "위치 정확도 (localisation, loc1 / loc3)",
    body: `**무엇을 보여주나:** 전역 탐색 heatmap 3종(A_cos, B_att, H_s8)의 peak이 실제 대상 위에 떨어졌는지. SAM 2 마스크를 만들기 전 단계, "어디를 찍을지"의 정확도다. 출처: ${SRC}의 \`localisation_coabs\`.

**계산 (\`scripts/m8.py: diag_unit\`, \`m8_analysis.stage1\`):**
- 대상 frame: GT가 있는 동시 부재 frame (2026-10-14 기준 131).
- heatmap: G 상태의 현재 frame 특징 위에서 계산. 전체 frame을 1024 × 1024로 (가로세로 비율 무시하고) 줄인 SAM 2 입력 기준 격자다.
  - A_cos, B_att: **stride-16** 특징 → 64 × 64 격자. peak NMS 반경 3 cell.
  - H_s8: **stride-8** FPN neck 특징 → 128 × 128 격자. peak NMS 반경 6 cell (같은 48 px).
- GT 격자: GT 마스크를 격자 크기로 area 보간 후 > 0 (대상 픽셀이 조금이라도 있는 cell) → **한 cell 팽창** (stride-16: 3 × 3 max-pool). stride-8 격자는 5 × 5 max-pool로 2 cell 팽창한다. 1024 공간에서 둘 다 16 px의 같은 물리적 여유다.
- **loc1** = top-1 peak cell이 팽창된 GT cell 안에 있는 frame 비율.
- **loc3** = top-3 peak 중 하나라도 그 안에 있는 frame 비율.

**읽는 법:**
- loc는 recall의 상한에 가깝다. peak이 대상에 없으면 그 점 prompt로는 대상을 분할할 수 없다. 2026-10-14 기준 A_cos loc3 0.68인데 recall 0.389다. 점이 대상 위에 있어도 SAM 2가 J > 0.5 마스크를 못 만드는 경우가 꽤 있다는 뜻이다 (작은 대상, 일부만 분할, 이웃 객체로 번짐).
- A_cos와 B_att가 거의 같다면, 대조 attention이 단일 prototype보다 나은 탐색을 하지 못한다는 뜻이다 (반증 시험 \`B_att_vs_A_cos_loc1\`).
- H_s8이 낮다면 "stride-16 표현에서 작은 대상이 점유하는 cell이 너무 적어서 못 찾는다"(표현 점유율) 가설을 지지하지 않는다.

**주의:**
- stride-8 격자는 cell이 4배 많고 GT cell 집합도 더 촘촘하다. 같은 16 px 여유를 줬지만 우연히 맞힐 확률이 낮아져 완전히 공정한 비교는 아니다.
- H_s8은 사전 등록 수정 2에서 버그를 고친 값이다 (원래는 SAM 2가 이미 64-d conv_s1 투영으로 바꿔 놓은 특징을 쓰고 있었다. 수정 후 \`raw_s8\`이 256-d neck 특징을 다시 계산한다).
- 격자가 가로세로 비율을 무시하므로 넓은 영상에서는 cell이 가로로 길다.`,
  },

  "stage1.tests": {
    title: "짝지은 반증 검정 (forest plot)",
    body: `**무엇을 보여주나:** 사전 등록된 반증 시험들의 효과 크기(점)와 95% CI(가로 막대). 각 행은 "생성기 X − 생성기 Y"의 차이다. 출처: ${SRC}의 \`paired_tests\`. 아래 칩의 ?를 누르면 시험별 가설과 해석이 나온다.

**계산 (\`m8_analysis.stage1: paired\`):**
- **같은 frame 짝 설계:** GT 있는 동시 부재 frame마다 두 계열의 지표(hit = 그 계열 후보 중 J > 0.5가 있는가 (0/1), 또는 best_J)를 구해 frame별 차이 d = X − Y를 만든다. 두 계열 모두 같은 frame, 같은 SGGL 상태에서 비저장 probe로 돌았으므로 frame 난이도가 상쇄된다.
- **Δ (점)** = d의 평균. recall 시험이면 recall 차이와 같다.
- **CI** = 영상 군집 bootstrap (\`rfeas.risk.cluster_bootstrap\`, 2000회, seed 0): 영상 단위로 복원 추출해 평균을 다시 계산하고 2.5 / 97.5 백분위를 쓴다. 한 영상의 여러 frame은 서로 독립이 아니므로 frame 단위 bootstrap보다 넓고 정직하다.
- loc1 시험 2개는 frame 표(\`loc1_*\` 열)에서 같은 방식으로 계산한다.
- 색: CI가 0을 완전히 벗어나면 파랑(+) 또는 빨강(−), 0을 포함하면 회색.

**읽는 법:**
- 각 시험은 "X가 Y보다 낫다"는 가설을 **반증**하려는 것이다. 점이 0 근처이거나 음수이고 CI 상한이 작으면, 그 메커니즘이 주는 이득은 없다(또는 있어도 작다)는 근거다.
- 2026-10-14 기준 유의하게 양수인 시험은 없다. 유의하게 음수인 것은 H_s8 loc1, G_neg, D_ref다.

**주의:**
- 영상 수가 11개뿐이고 대부분의 hit이 drone, kite-surf에 몰려 있다. bootstrap 재표본에 drone이 몇 번 뽑히느냐에 CI가 크게 흔들린다. CI 끝이 정확히 0.000인 경우(예: [−0.24, 0.00])는 차이가 소수 영상에서만 생겼다는 신호다.
- 다중 비교 보정이 없다. 13개 시험 중 하나가 우연히 유의할 수 있다.
- 두 생성기가 모두 바닥(recall ≈ 0)인 시험은 "차이 없음"이 정보가 아니다. 둘 다 안 된다는 뜻일 뿐이다.
- 후보 개수가 다른 계열끼리의 비교(top-3 vs 1개)는 후보 수 효과가 섞인다. 해당 시험 설명에 적었다.`,
  },

  "stage1.test.B_att_vs_A_cos_recall": {
    title: "B_att − A_cos recall (attention의 탐색 이득)",
    body: `**가설 (사전 등록):** 대조 token attention(B_att: frame-0 fg 8개 + bg 32개 token에 softmax)은 단일 prototype 코사인(A_cos)보다 대상을 더 잘 찾는다. bg token이 배경과 닮은 방해물을 억제하기 때문이다.

**비교:** 둘 다 G 상태 stride-16 특징에서 heatmap → top-3 peak → G 메모리에 점 prompt. 후보 수(3개)와 prompt 방식이 같고 heatmap만 다르다. 따라서 깨끗한 attention 효과 측정이다.

**결과 읽기:** Δ ≈ 0이고 CI가 0을 포함하면 기각(attention이 탐색에 이득을 주지 않음). 2026-10-14 기준 Δ = 0.000 [−0.014, +0.030]. 문서 해석: 두 방법이 같은 frozen 특징 위에서 같은 peak를 고르고, 실패하는 frame은 둘 다 실패한다. "어떤 cell이 대상과 닮았나"라는 질문 자체가 이 특징으로는 풀리지 않는다는 뜻이다 (\`ATTENTION_MECHANISM_COMPARISON.md\`).`,
  },
  "stage1.test.B_att_vs_A_cos_loc1": {
    title: "B_att − A_cos 위치 정확도 top-1",
    body: `**가설:** attention heatmap의 top-1 peak이 prototype heatmap보다 더 자주 대상 위에 떨어진다.

**계산:** GT 있는 동시 부재 frame마다 \`loc1_B_att − loc1_A_cos\` (각각 0/1), 영상 군집 bootstrap CI. 두 heatmap 모두 64 × 64 격자, GT 한 cell 팽창.

**결과 읽기:** 2026-10-14 기준 Δ = 0.000 [−0.106, +0.084]. CI가 넓다. 같은 결론(차이 없음)이지만 표본이 작아 ±0.1 정도의 차이는 배제하지 못한다.`,
  },
  "stage1.test.H_s8_vs_A_cos_recall": {
    title: "H_s8 − A_cos recall (표현 점유율 가설)",
    body: `**가설 (사전 등록 'H > A'):** 작은 대상은 stride-16 격자에서 1–2 cell만 차지해 특징이 배경과 섞인다. stride-8 특징을 쓰면 대상이 4배 많은 cell을 차지해 더 잘 찾을 것이다 (표현 점유율).

**비교:** 같은 prototype 매칭, 같은 top-3 점 prompt, G 메모리. 특징 해상도만 다르다 (stride-8 FPN neck 256-d vs stride-16).

**결과 읽기:** 2026-10-14 기준 Δ = −0.153 [−0.242, 0.000]. stride-8이 오히려 나쁘다 → 가설 기각. CI 상한이 정확히 0인 것은 일부 영상에서만 차이가 났다는 뜻이다. stride-8 특징은 frozen 상태에서 의미 정보가 약하고 질감에 민감한 것으로 해석된다.

**주의:** 사전 등록 수정 2(H_s8 특징 버그 수정, shard 0 표를 본 뒤 결정) 이후의 값이다.`,
  },
  "stage1.test.H_s8_vs_A_cos_loc1": {
    title: "H_s8 − A_cos 위치 정확도 top-1",
    body: `**가설:** stride-8 heatmap의 top-1 peak이 stride-16 heatmap보다 대상 위에 더 자주 떨어진다.

**계산:** \`loc1_H_s8 − loc1_A_cos\`. H_s8은 128 × 128 격자에서 GT 2 cell 팽창(16 px), A_cos는 64 × 64에서 1 cell 팽창(16 px).

**결과 읽기:** 2026-10-14 기준 Δ = −0.435 [−0.612, −0.167]. CI가 0을 완전히 벗어난 큰 음수다. 표현 점유율 가설에 대한 가장 강한 반증이다.

**주의:** 128 격자는 cell이 4배 많아 우연히 맞힐 확률이 낮다. 효과의 일부는 이 격자 차이일 수 있지만, 크기(−0.44)를 다 설명하지는 못한다.`,
  },
  "stage1.test.cpfr8_vs_oclr8_G_recall": {
    title: "CPFR8 − OCLR8 recall, G 메모리 (문맥 보존 가설)",
    body: `**가설 (사전 등록 'CPFR > OCLR, 같은 상태·같은 배율'):** 같은 배율로 확대할 때, 주변을 잘라내는 crop(OCLR)보다 주변을 압축해 전체 문맥을 남기는 foveation(CPFR)이 SAM 2 memory attention의 재식별을 돕는다.

**비교:** 둘 다 광류로 운반한 box 중심, 대상 긴 변이 8 stride-16 cell(128 px)을 차지하도록 맞춘 같은 배율. 메모리는 G 상태(전체 frame 메모리). prompt 없음. 차이는 view 기하(잘라내기 vs 주변 압축)뿐이다.

**결과 읽기:** 2026-10-14 기준 Δ = +0.008 [0.000, +0.031]. 차이 없음. 그러나 **두 생성기 모두 recall ≈ 0.01**로 바닥이다. G 메모리는 전체 frame 좌표로 쌓였으므로, 확대·왜곡된 view에서는 memory attention이 대상을 다시 찾지 못한다(사전 등록 수정 1의 관찰). 바닥끼리의 "차이 없음"이라 문맥 보존 가설에 대한 정보는 적다.

**주의:** 수정 1은 점 prompt 변형(\`_pt\`)을 F 시험의 주 시험으로 보고하겠다고 적었지만, 현재 \`stage1_report.json\`의 짝 시험은 prompt 없는 변형만 계산한다. \`_pt\` 변형도 recall 0–0.04로 바닥이라 결론은 같을 가능성이 크지만, 이 일탈은 기록해 둔다.`,
  },
  "stage1.test.cpfr8_vs_oclr8_G_bestJ": {
    title: "CPFR8 − OCLR8 최고 J, G 메모리",
    body: `**가설:** 위 recall 시험과 같다. 다만 0/1 hit 대신 연속 지표인 best_J(그 계열 후보의 J)로 비교해, J > 0.5 문턱 아래의 부분적 개선도 잡는다.

**결과 읽기:** 2026-10-14 기준 Δ = +0.003 [−0.015, +0.024]. 차이 없음. 두 view 모두 G 메모리에서 바닥이라 정보가 적다는 점은 recall 시험과 같다.`,
  },
  "stage1.test.cpfr8_vs_oclr8_L_recall": {
    title: "CPFR8 − OCLR8 recall, L 메모리 (문맥 보존 가설의 핵심 시험)",
    body: `**가설:** 같은 배율이면 문맥 보존 foveation(CPFR)이 잘라낸 crop(OCLR)보다 재식별을 잘한다.

**비교:** 둘 다 L 상태(crop 좌표로 쌓인 메모리), 광류 box 중심, 대상이 8 cell을 차지하는 같은 배율, prompt 없음. **이 시험이 F 계열에서 가장 정보가 많다.** 두 생성기 모두 recall이 바닥이 아니기 때문이다 (2026-10-14 기준 OCLR 0.176, CPFR 0.168).

**결과 읽기:** 2026-10-14 기준 Δ = −0.008 [−0.031, 0.000]. CPFR이 recall을 늘리지 못했다 → 문맥 보존 가설 지지 안 됨. 더 중요한 차이는 FP 쪽에 있다: 원시 FP는 OCLR 0.014 vs CPFR 0.110이다. 주변 문맥이 보이면 비슷한 방해물(miami-surf의 다른 서퍼들)을 '존재'로 잡는 일이 많아진다. 이것이 RESC_OCLR / RESC_CPFR 진단 arm을 나눠 돌린 이유다.

**주의:** 적중은 거의 drone 한 영상(22 frame)에서 나온다. 사실상 영상 1개의 비교다.`,
  },
  "stage1.test.cpfr8_vs_oclr8_L_bestJ": {
    title: "CPFR8 − OCLR8 최고 J, L 메모리",
    body: `**가설:** L 메모리 recall 시험과 같고, 연속 지표 best_J로 본다.

**결과 읽기:** 2026-10-14 기준 Δ = −0.008 [−0.031, +0.001]. 차이 없음 → 문맥 보존 가설 지지 안 됨 (\`STAGE1_DIAGNOSTIC_RESULTS.md\` §2).`,
  },
  "stage1.test.cpfr4_vs_oclr4_G_recall": {
    title: "CPFR4 − OCLR4 recall, G 메모리",
    body: `**가설:** 배율을 낮춰(대상이 4 cell을 차지, 즉 crop 폭 ≈ 대상 크기의 16배) 왜곡이 덜하면 G 메모리에서도 foveation이 crop보다 나을 수 있다.

**비교:** G 상태, 광류 box 중심, 같은 4-cell 배율, prompt 없음. 4-cell 변형은 G 메모리에만 있다 (L은 8-cell만).

**결과 읽기:** 2026-10-14 기준 Δ = −0.008 [−0.040, +0.014]. 차이 없음. 두 생성기 모두 recall 0.01–0.02로 바닥이라 정보가 적다.`,
  },
  "stage1.test.G_neg_vs_B_att#0_recall": {
    title: "G_neg − B_att recall (음성 점 prompt)",
    body: `**가설:** 방해물에 음성 점을 찍어주면(G_neg) SAM 2가 대상과 방해물을 구별해 recall이 오른다. PerSAM 계열 선행연구의 아이디어다.

**비교:** G_neg는 B_att top-1 peak에 양성 점 + "A_cos가 대상처럼 보지만 B_att 사후확률 < 0.5이고 양성 점에서 3 cell 넘게 떨어진" A_cos peak들에 음성 점. 후보 1개.

**결과 읽기:** 2026-10-14 기준 Δ = −0.084 [−0.136, −0.010]. 음성 점이 해롭다. 문서는 선행연구 보고와 같은 방향이라고 적었다.

**주의 (키 이름과 실제 계산의 차이):** 키에는 \`B_att#0\`이라고 적혀 있지만 코드(\`paired("G_neg", "B_att")\`)는 B_att **계열 전체(top-3 후보)**와 비교한다. 후보 1개 vs 3개의 차이가 섞여 있다. 2026-10-14에 diag CSV에서 따로 계산하면 B_att#0 단독 recall은 0.359, G_neg는 0.305로, top-1끼리 비교해도 G_neg가 약 −0.05 낮다. 결론(음성 점이 해롭다)은 유지되지만 효과 크기는 이 차트 값보다 작다.`,
  },
  "stage1.test.D_ref_vs_B_att_recall": {
    title: "D_ref − B_att recall (frame-0 메모리만 읽기)",
    body: `**가설:** 놓친 구간의 메모리는 오염됐을 수 있다. 같은 B_att top-1 점 prompt를 주되 memory attention이 frame-0(prompt frame) 메모리와 object pointer만 읽게 하면(D_ref) 재식별이 좋아진다.

**결과 읽기:** 2026-10-14 기준 Δ = −0.038 [−0.058, −0.008]. 문서 결론: frame-0 메모리만 쓰면 약간 나쁘다.

**주의 (후보 수 차이):** D_ref는 후보 1개(B_att top-1 peak)이고, 비교 대상은 B_att 계열 전체(top-3)다. 2026-10-14에 diag CSV에서 따로 계산하면 B_att#0 단독 recall 0.359, D_ref 0.351로 거의 같다. 즉 이 −0.038의 대부분은 "후보 1개 vs 3개" 효과이고, 메모리를 frame-0으로 제한한 효과 자체는 0에 가깝다. "메모리 제한이 해롭다"보다 "메모리 제한이 돕지 않는다"가 정확한 해석이다.`,
  },
  "stage1.test.C_flowcrop_vs_Lp_recall": {
    title: "C_flowcrop − Lp recall (광류 운동 증거)",
    body: `**가설 (사전 등록 'C/E > Lp'):** 외형과 무관한 운동 증거(RAFT 광류로 운반한 box 중심에 L crop을 옮기기)가 M7의 보호 메모리 probe(Lp)보다 대상을 더 잘 되찾는다.

**비교:** 둘 다 L 상태, prompt 없음, SAM 2 자체 존재 판정. C_flowcrop은 crop 위치를 광류 box 중심으로 옮기고 전체 메모리를 쓴다. Lp는 현재 crop 위치에서 prompt frame 메모리만 쓴다.

**결과 읽기:** 2026-10-14 기준 Δ = −0.046 [−0.136, +0.016]. 운동 생성기가 새 증거를 더하지 못했다. 이유 중 하나: 대상이 사라지면 광류 box가 배경 흐름을 따라 표류한다 (shard 0 측정에서 GT와 겹치는 비율 19%).

**주의:** 이것은 "놓친 뒤 광류로 다시 찾기"의 실패다. 같은 광류를 **놓치기 전부터** crop 중심 추종에 쓰는 FLOWVEL(Stage 2)은 PROMISING이다. 같은 정보원이라도 쓰는 시점에 따라 결과가 다르다.`,
  },
  "stage1.test.E_beam1_vs_Lp_recall": {
    title: "E_beam1 − Lp recall (속도 외삽 가설)",
    body: `**가설:** 마지막 신뢰 위치 + 속도 × 경과 frame으로 외삽한 곳에 L crop을 놓으면(E_beam1) 대상을 되찾는다.

**결과 읽기:** 2026-10-14 기준 Δ = −0.061 [−0.163, 0.000]. E_beam1 recall이 0이라 Δ는 곧 −(Lp recall)이다. 등속 외삽은 동시 부재 구간의 대상 위치를 맞히지 못했다. M7 M3 계열(속도 기반 중심)의 rescue 버전이 실패한 것이다.`,
  },

  "stage1.rankerChart": {
    title: "순위화기 pooled held-out 순 이득",
    body: `**무엇을 보여주나:** 후보를 만든 뒤 "어느 후보를 출력할지, 아니면 아무것도 출력하지 않을지"를 정하는 순위화기 R0–R3의 교차 적합 held-out 순 이득. 묶음(GLOBAL, LOCAL, POOL, pool25)별 막대 4개. 출처: \`outputs/milestone8/rankers_{GLOBAL,LOCAL,POOL,pool25}.json\`의 \`<ranker>.pooled_heldout.net_dJ_frames\` (생성: \`m8_analysis.py rankers <gens> <name>\`).

**결정 규칙 (\`_accept_eval\`):** 동시 부재 frame마다 그 묶음 후보 중 순위 점수가 가장 높은 후보 하나를 고르고, 점수 > threshold면 출력, 아니면 SGGL처럼 빈 출력을 유지한다.

**net_dJ_frames (순 이득, frame J 합):**
- 출력했고 GT가 있는 frame: + J (그 후보의 J. SGGL은 빈 출력이라 J = 0이었으므로 이득 = J. J ≤ 0.5인 "틀린 구조"도 그 작은 J만큼 더해진다).
- 출력했는데 GT가 없는 frame: −1 (둘 다 비면 J = 1로 치므로, 무언가 출력하면 1 → 0).
- 순 이득 = Σ(+J) − (거짓 양성 frame 수). 0보다 크면 SGGL보다 낫다.

**교차 적합 (cross-fitting):**
- REP→DEV (사전 등록): REP에서 학습·threshold 고정 → DEV에서 평가.
- DEV→REP (추가, 공개된 일탈): DEV에서 학습·threshold 고정 → REP에서 평가. 인과 arm을 DEV + REP 전체에서 표본 밖으로 평가하기 위해 추가했다.
- **pooled held-out** = 두 방향 held-out 결과의 합. 모든 동시 부재 frame이 정확히 한 번, 자기를 보지 않은 모델로 평가된다.

**threshold 규칙 (\`_thr_best_net\`):** 학습 split의 동시 부재 frame에서 **거짓 양성률(진짜 부재 frame 중 출력 비율) ≤ 0.10을 지키는 threshold 중 순 이득이 최대인 것**. 사전 등록 문구는 "FP ≤ 0.10"뿐이었고 "그중 순 이득 최대"는 공개된 구체화다. R3는 학습 split에서도 out-of-fold 점수로 threshold를 고른다.

**왜 음수인가 (2026-10-14 기준 모든 묶음·순위화기가 ≤ 0):**
1. **비용 비대칭:** 거짓 양성은 정확히 −1, 구조는 최대 +1이고 평균은 그보다 작다.
2. **기저율:** 동시 부재 frame 중 진짜 부재가 355, 대상 있음이 131 (2.7배). 존재 판별 AUC가 0.60–0.71 정도로는(문서 \`STAGE1_DIAGNOSTIC_RESULTS.md\` §3) 회수 이득보다 거짓 재획득 비용이 크다. 반면 frame 안에서 "맞는 후보 vs 틀린 후보" AUC는 0.815다. 병목은 순위가 아니라 **존재 판정**이다.
3. **split 간 전이 실패:** DEV에는 GT 있는 동시 부재 frame이 25개(23개가 kite-surf), 진짜 부재 47개뿐이다. DEV에서 FP ≤ 0.10이면 진짜 부재 4–5 frame까지 허용하는 threshold라 매우 불안정하다. 그래서 대부분의 손실이 DEV→REP 방향(REP 106 / 308 frame에 적용)에서 나온다. REP→DEV 방향은 REP에서 고른 높은 threshold 때문에 DEV에서 거의 아무것도 출력하지 않아 0에 가깝다.
4. **적중의 영상 집중:** 회수 가능한 frame이 drone(REP)과 kite-surf(DEV)에 몰려 있어, 한 split에서 배운 것이 다른 split의 다른 영상으로 옮겨가지 않는다.

**읽는 법:**
- 0이 최선인 경우가 많다. 0 = "항상 기권" = SGGL과 같다. 예: R3 GLOBAL이 0인 것은 이긴 게 아니라 아무것도 출력하지 않은 것이다.
- 음수가 작을수록 덜 해롭다. 양의 막대가 하나라도 생기면, 그것이 Stage 2로 갈 후보 순위화기다.

**주의:** 이것은 frame 단위 비저장 계산이다. 실제로 출력하면 crop 재배치·메모리 기록으로 다음 frame이 바뀐다(좋게도 나쁘게도). 그 효과는 Stage 2 인과 arm에서만 보인다.`,
  },

  "stage1.ranker.R0_score": {
    title: "R0: SAM 2 object score",
    body: `**메커니즘:** 학습 없음. 순위 점수 = 후보를 만들 때 SAM 2가 낸 object score logit 그대로. threshold만 학습 split에서 고른다.

**의미:** "SAM 2 자신의 존재 판정을 믿는다"는 기준선. 점 prompt 후보는 거의 항상 score > 0이라, threshold를 높게 잡아야 FP ≤ 0.10을 지킨다.

**2026-10-14 기준:** GLOBAL −9.0, LOCAL −41.0, POOL −13.0 (pooled held-out). 어느 묶음에서도 양수가 아니다.`,
  },
  "stage1.ranker.R1_trainfree": {
    title: "R1: 학습 없는 외형 마진",
    body: `**메커니즘 (\`ranker.r1\`):** SAM 2 score ≤ 0인 후보는 제외(−∞). 나머지는 \`cos_fg − cos_bg\`로 순위를 매긴다. cos_fg = 후보 마스크로 pool한 현재 frame G 특징(stride-16)과 frame-0 대상 prototype의 코사인, cos_bg = frame-0 배경 prototype과의 코사인.

**의미:** 학습 파라미터가 없어(threshold 하나만) split 간 과적합 위험이 가장 작다. "frame-0 대상과 더 닮고 배경과 덜 닮은 후보"를 고른다.

**2026-10-14 기준:** GLOBAL −28.9, LOCAL −41.1, POOL −28.4. 외형 마진은 맞는 후보를 고르는 데는 쓸 만하지만(frame 내 AUC 0.815), 대상이 아예 없는 frame에서도 가장 닮은 무언가는 마진이 높게 나온다.`,
  },
  "stage1.ranker.R2_logistic": {
    title: "R2: 로지스틱 회귀",
    body: `**메커니즘 (\`ranker.LR\`):** 후보마다 독립적으로 점수를 매기는 L2 정규화 로지스틱 회귀(LBFGS, 표준화 입력). 특징 7개 (\`ranker.SCALARS\`): score, iou(SAM 2 예측 IoU), cos_fg, cos_bg, margin(= cos_fg − cos_bg), log_area(후보 면적 / frame-0 면적의 로그), dist_flow(후보 중심과 광류 box 중심 거리 / crop 변).

**학습 데이터:** 학습 split의 coabs + disagree + regular frame 후보 전부. 라벨 = J > 0.5면 1, J < 0.1이거나 GT가 없으면 0, 그 사이는 제외.

**2026-10-14 기준:** 가장 해롭다. POOL −77.6 (회수 26 / 거짓 양성 99), GLOBAL −46.6. 후보를 서로 비교하지 않고 따로 점수화하므로 "이 frame에 대상이 있는가"를 판단하지 못하고, disagree / regular frame(대상이 대부분 있는 frame)에서 배운 점수 척도가 동시 부재 frame에 그대로 옮겨지지 않는다.`,
  },
  "stage1.ranker.R3_set": {
    title: "R3: set-attention 순위화기 (기권 가능)",
    body: `**메커니즘 (\`ranker.SetRanker\`, \`m8_analysis._fit_set\`):**
- 입력: frame의 후보 전부를 token으로. 각 token = 후보 마스크로 pool한 256-d 특징 + 표준화된 7개 스칼라 + 생성기 embedding.
- 후보끼리 self-attention(경쟁) → frame-0 fg/bg k-means token에 cross-attention(신원) → 후보별 logit + 학습된 "none"(기권) token logit.
- 학습: frame 하나가 예제 하나. 정답 = J가 가장 높은 후보(그 J > 0.5일 때), 아니면 none. cross-entropy, AdamW, 1500 step, seed 0.
- 순위 점수 = 후보 logit − none logit.
- **out-of-fold:** 학습 split의 영상을 2개 fold로 나눠(영상 단위), 학습 split 후보에는 자기를 보지 않은 fold 모델 점수를 쓴다. threshold는 이 out-of-fold 점수로 고른다. 평가 split에는 두 fold 모델 점수의 평균을 쓴다 (같은 점수 척도 유지).

**의미:** attention을 "후보 경쟁 + 존재 판정"에 쓰는 시도(\`ATTENTION_MECHANISM_COMPARISON.md\` 위치 ③). 기권 token이 있어 로지스틱처럼 거짓 양성을 쏟아내지 않는다.

**2026-10-14 기준:** GLOBAL 0.0 (항상 기권 = SGGL과 동일), POOL −11.9, LOCAL −33.1. R2보다 덜 해롭지만 양의 이득은 없다. 존재를 확신할 만큼 판별하지 못해 결국 기권하는 쪽으로 학습된다.`,
  },

  "stage1.rcol.set": {
    title: "set (생성기 묶음)",
    body: `순위화기가 고를 수 있는 후보 생성기의 묶음. 각 묶음마다 R0–R3를 따로 학습·평가한다 (\`rankers_<set>.json\`의 \`gens\`).

| 묶음 | 생성기 | 의도 |
|---|---|---|
| GLOBAL | A_cos, B_att, D_ref, G_neg | 전역 탐색 + 점 prompt 계열만. recall은 높지만 원시 FP ~1.0 |
| LOCAL | F_oclr8_L, F_cpfr8_L, B_att_crop, A_cos_crop, Lp, Gp | crop/view를 바꾸고 SAM 2 자체 존재 판정을 쓰는 계열. 원시 FP가 낮음 |
| POOL | 25종 전부 | 모든 증거를 한 순위화기에 |
| pool25 | 25종 전부 (POOL과 같은 목록) | 이름 없이 기본 태그로 돌린 **이전 실행**. 2026-10-14 기준 분모가 119 / 291 frame으로 POOL(131 / 355)보다 작아, 일부 단위가 들어오기 전 시점의 결과로 보인다. 비교에는 POOL을 쓰라 |`,
  },
  "stage1.rcol.ranker": {
    title: "ranker (R0–R3)",
    body: `R0 = SAM 2 score, R1 = 학습 없는 외형 마진, R2 = 로지스틱, R3 = set-attention(기권 가능). 각 행의 ?에 자세한 정의가 있다. R0, R1은 학습 파라미터가 없고 threshold만 고른다. R2, R3는 학습 split의 후보로 학습한다.`,
  },
  "stage1.rcol.pooled": {
    title: "pooled net ΔJ",
    body: `\`pooled_heldout.net_dJ_frames\` = (REP→DEV의 DEV 평가 순 이득) + (DEV→REP의 REP 평가 순 이득). 순 이득 = 출력한 GT 있는 frame의 J 합 − 출력한 GT 없는 frame 수. 단위는 "frame 수"(J 합)다. 0 = SGGL과 같음. 양수만 의미 있는 이득이다. 단위 평균 J&F 변화로 바꾸려면 단위·frame 수로 나눠야 하므로 이 숫자를 dJF와 직접 비교하지 말 것.`,
  },
  "stage1.rcol.rescued": {
    title: "rescued (회수 frame 수)",
    body: `두 held-out 평가에서 **출력했고, GT가 있고, 출력 후보의 J > 0.5**인 frame 수의 합 (\`rescue_rate_present × frames_present\`의 합). J ≤ 0.5인 "틀린 구조"는 여기 세지 않지만 순 이득에는 그 J가 더해진다. 분모는 GT 있는 동시 부재 frame 131 (2026-10-14 기준).`,
  },
  "stage1.rcol.fp": {
    title: "FP frames (거짓 양성 frame 수)",
    body: `두 held-out 평가에서 GT가 없는데 무언가를 출력한 frame 수의 합 (\`fp_rate_empty × frames_empty\`의 합). 한 frame당 순 이득 −1. 분모는 진짜 부재 동시 부재 frame 355 (2026-10-14 기준). threshold는 학습 split에서 FP ≤ 10%로 골랐지만, held-out에서는 이를 넘을 수 있다. 그것이 전이 실패의 증상이다.`,
  },
  "stage1.rcol.fitREP": {
    title: "fitREP→DEV net",
    body: `REP(DAVIS train)에서 학습·threshold 고정 → DEV(DAVIS val + PUMaVOS A/B + SegTrack)에서 평가한 순 이득 (\`fitREP.heldout_eval.net_dJ_frames\`). 사전 등록된 방향이다. DEV의 GT 있는 동시 부재 frame은 25개뿐(대부분 kite-surf)이라 대부분 0 근처다. 0은 대개 "REP threshold가 높아 DEV에서 아무것도 출력하지 않음"을 뜻한다.`,
  },
  "stage1.rcol.fitDEV": {
    title: "fitDEV→REP net",
    body: `DEV에서 학습·threshold 고정 → REP에서 평가한 순 이득 (\`fitDEV.heldout_eval.net_dJ_frames\`). 공개된 추가 방향이다. DEV는 학습 표본이 작아(GT 있는 동시 부재 25, 진짜 부재 47) threshold가 불안정하고, REP의 많은 진짜 부재 frame(308)에서 거짓 양성이 쌓인다. pooled 손실의 대부분이 여기서 나온다.`,
  },

  "stage1.hits": {
    title: "영상별 hit 수 히트맵",
    body: `**무엇을 보여주나:** 행 = 생성기 계열, 열 = 영상. 칸 숫자 = 그 계열이 hit(후보 중 J > 0.5)한 GT 있는 동시 부재 frame 수. 진할수록 많다. 출처: ${SRC}의 \`hits_by_video\` (\`P[P.hit].groupby(["fam", "video"]).size()\`). 한 번이라도 hit이 있는 계열·영상만 나온다.

**읽는 법:**
- 이 히트맵이 Stage 1의 가장 중요한 단서다. 2026-10-14 기준 hit이 있는 영상은 5개뿐(drone, kite-surf, dog-gooses, lady-running, skate-park)이고, 대부분이 **drone과 kite-surf**에 몰려 있다. diag CSV에서 다시 세면 전체 생성기 합집합 hit 67 frame 중 51개(76%)가 이 두 영상이다 (문서에는 이전 시점 값 "64 중 48, 75%").
- drone은 REP, kite-surf는 DEV에 있다. 그래서 학습형 순위화기가 한 split에서 배운 것이 다른 split로 옮겨가지 않는다.
- recall이 높은 계열(A_cos, B_att, D_ref, G_neg, H_s8)의 hit은 거의 전부 drone + kite-surf다. "recall 0.39"는 사실상 두 영상에서 잘 된다는 뜻이다.
- 사전 등록 진출 조건의 "≥ 4영상"은 이 행에서 0이 아닌 열 수다. 1 frame짜리 열도 한 영상으로 센다 (예: B_att의 dog-gooses 1, lady-running 1).
- 다른 계열이 다른 영상에서 hit하면(예: Lp는 dog-gooses, Gp·crop 계열은 skate-park) 결합의 여지가 있다는 뜻이지만, 어느 후보를 언제 믿을지 정하는 문제는 그대로 남는다.

**주의:** 영상이 5개라 어떤 일반화 주장도 할 수 없다. 대상이 사라지는 방식(작은 비행체, 수면 반사, 가림)이 영상마다 달라, 생성기 우열은 영상 구성에 크게 좌우된다.`,
  },

  "stage1.genTable": {
    title: "생성기 표",
    body: `**무엇을 보여주나:** 생성기 계열(25종)마다 동시 부재 frame 지표와, disagree / regular trigger frame의 recall. 출처: \`outputs/milestone8/stage1_generators.csv\` (= \`stage1_report.json\`의 \`generators_coabs\`)와 \`disagree_recall\`, \`regular_recall\`. recall 내림차순 정렬. 생성기 이름 옆 ?에 메커니즘·계열·검증 가설·결과가 있다. 열 머리의 ?에 열 정의가 있다.

**계열 기호:** A 전역 prototype 탐색, B attention, C 광류 운동, D 메모리, E 다중 가설 beam, F 척도(crop vs foveation), G prompt, H 표현. \`_crop\` = heatmap peak로 L crop 배치, \`_G\` / \`_L\` = 어느 SAM 2 상태의 메모리로 예측하나, \`_pt\` = 양성 점 prompt 추가(사전 등록 수정 1).

${FRAMES}

**주의:** 이 표의 지표는 모두 oracle(GT로 그 계열 최고 후보를 고름) 기준이다. 실제로 쓸 수 있는지는 순위화기 결과로 판단해야 한다.`,
  },
  "stage1.col.gen": {
    title: "gen (생성기 계열)",
    body: `\`scripts/m8.py: generators()\`의 후보 이름에서 \`#i\` 접미사를 뗀 계열 이름 (\`m8_analysis.fam\`). A_cos, B_att, H_s8은 frame당 top-3 peak 후보 3개를 하나의 계열로 묶는다. 이름 옆 ?에 각 생성기 설명이 있다.`,
  },
  "stage1.col.recall": {
    title: "recall",
    body: `GT 있는 동시 부재 frame 중, 그 계열 후보 중 하나라도 J > 0.5인 frame의 비율 (\`hit.mean()\`). oracle 선택 기준이다. 사전 등록 진출 조건: ≥ 0.25이고 ≥ 2 × max(Gp, Lp recall). 후보가 3개인 계열(A_cos, B_att, H_s8)이 구조적으로 유리하다.`,
  },
  "stage1.col.mean_best_J": {
    title: "mean_best_J",
    body: `GT 있는 동시 부재 frame마다 그 계열 후보 중 가장 높은 J를 구해 평균 낸 값. hit 문턱(0.5) 아래의 부분적 성공도 반영한다. recall이 같아도 mean_best_J가 높으면 놓친 frame에서도 대상 일부를 잡았다는 뜻이다. 진짜 부재 frame은 포함하지 않는다.`,
  },
  "stage1.col.top_J": {
    title: "top_J",
    body: `GT 있는 동시 부재 frame에서 그 계열의 **첫 번째 후보**의 J 평균 (\`J.first\`). A_cos, B_att, H_s8에서는 top-1 heatmap peak(#0) 후보다. 후보가 1개인 계열은 mean_best_J와 같다. mean_best_J − top_J 차이가 크면 top-3 중 2·3번째 peak이 자주 정답이었다는 뜻이다. 즉 oracle 없이 top-1만 쓰면 그만큼 잃는다. 예: H_s8은 2026-10-14 기준 mean_best_J 0.225 vs top_J 0.127로 top-1이 특히 약하다.`,
  },
  "stage1.col.videos_hit": {
    title: "videos_hit",
    body: `hit(J > 0.5)한 frame이 하나라도 있는 영상 수. 사전 등록 진출 조건은 ≥ 4영상이다. 1 frame만 hit해도 1영상으로 센다. 그래서 4가 나와도 실제 근거가 drone, kite-surf 두 영상에 몰려 있을 수 있다. 히트맵에서 영상별 frame 수를 같이 보라. GT 있는 동시 부재 frame이 있는 영상은 2026-10-14 기준 11개뿐이다.`,
  },
  "stage1.col.units_hit": {
    title: "units_hit",
    body: `hit한 frame이 하나라도 있는 단위(객체 × 영상) 수. 한 영상에 객체가 여럿이면 videos_hit보다 크다. 동시 부재 frame이 있는 단위는 2026-10-14 기준 24개(DEV 3, REP 21)다.`,
  },
  "stage1.col.fp_rate_gt_empty": {
    title: "fp_rate_gt_empty (원시 거짓 양성률)",
    body: `GT 없는 동시 부재 frame(2026-10-14 기준 355) 중, 그 계열 후보 중 하나라도 '존재'(SAM 2 object score > 0이고 마스크 면적 > 0)를 낸 frame의 비율. 순위화·threshold 적용 **전** 값이다.

점 prompt 생성기(A_cos, B_att, H_s8, G_neg, D_ref, C_flowbox, \`_pt\`)는 SAM 2가 클릭을 받으면 거의 항상 마스크를 내므로 ~1.0이다. 이 경우 존재 판정은 순위화기의 몫이다. prompt 없는 view 생성기는 SAM 2 자체 판정을 쓰므로 낮다. 사전 등록 FP 기준(≤ 0.10)은 원시값이 아니라 REP에서 고정한 ranker/threshold 이후의 DEV 값으로 판정한다.`,
  },
  "stage1.col.adv_recall": {
    title: "adv_recall (recall 진출 조건 충족)",
    body: `\`recall ≥ 0.25\` 그리고 \`recall ≥ 2 × max(Gp, Lp recall)\` 그리고 \`videos_hit ≥ 4\`이면 ✓ (\`m8_analysis.stage1\`). 사전 등록 진출 조건 중 **recall 부분만**이다. 나머지 조건(REP에서 고정한 ranker/threshold로 DEV 진짜 부재 frame 거짓 양성 ≤ 0.10, 그리고 양의 순 이득)은 이 열에 없고 순위화기 결과로 봐야 한다.

2026-10-14 기준 B_att만 ✓다. 그러나 4영상 중 2개(dog-gooses, lady-running)는 1 frame씩이고, 어떤 순위화기도 FP/순 이득 조건을 통과하지 못해 Stage 2로 정식 진출한 생성기는 없다.`,
  },
  "stage1.col.disagree": {
    title: "disagree recall",
    body: `**불일치 trigger frame**(G, L 중 한쪽만 score > 0, 또는 두 마스크 IoU < 0.5)의 GT 있는 frame에서 같은 recall 정의(후보 중 J > 0.5). 출처: \`stage1_report.json\`의 \`disagree_recall\`. 2026-10-14 기준 불일치 frame 242개.

**의미:** 한 view는 아직 대상을 보고 있는 경우가 많은, 동시 부재보다 쉬운 frame이다. 여기서도 recall이 낮으면 생성기 자체가 약하다. 동시 부재에서만 낮으면 "대상이 정말 안 보이는" 문제다. 순위화기 학습 데이터(coabs + disagree + regular)의 일부이기도 하다.`,
  },
  "stage1.col.regular": {
    title: "regular recall",
    body: `**정기 trigger frame**(정상 추적 중, t % 10 == 0)의 GT 있는 frame에서 같은 recall. 출처: \`regular_recall\`. 2026-10-14 기준 399 frame.

**의미:** 대상이 잘 보이는 쉬운 frame에서의 sanity check다. 대부분 0.95 이상이어야 정상이다. 낮으면(예: D_ref, Gp 약 0.80) 그 생성기가 쉬운 frame에서도 대상을 놓친다는 뜻이다. frame-0 메모리만 읽는 생성기(D_ref, Gp, Lp)가 낮은 것은, 대상 외형이 frame 0에서 많이 바뀐 경우 최근 메모리가 필요하다는 뜻이다.`,
  },
}

// ----------------------------------------------------------------------------- generator entries ("stage1.gen.<FAMILY>")
const g = (title: string, body: string): HelpEntry => ({ title, body: `${body}\n\n_${OUTCOME_NOTE}_` })

const POINT_FP = "점 prompt를 주므로 SAM 2가 거의 항상 '존재'를 출력한다. 원시 FP ~1.0은 결함이 아니라 구조적 성질이고, 존재 판정은 순위화기에 맡겨진다."

const gens: HelpMap = {
  "stage1.gen.A_cos": g("A_cos: frame-0 prototype 코사인 전역 탐색", `**계열:** A (전역 탐색). 후보 ID A1. 선행연구 근접도 높음 (PerSAM, Matcher, SAMITE).

**메커니즘 (\`rescue.proto\`, \`heat_proto\`, \`peaks\`):**
1. frame 0에서 GT 마스크로 G 상태 stride-16 특징을 가중 평균 → L2 정규화한 대상 prototype 1개.
2. 현재 frame t의 G 특징(64 × 64 격자) 각 cell과 prototype의 코사인 → heatmap.
3. top-3 peak (NMS 반경 3 cell) → 각 peak cell 중심을 원본 frame 좌표로 바꿔 **G 상태(전체 메모리)에 양성 점 1개**로 비저장 probe. 후보 3개 (\`A_cos#0..2\`).

**검증하는 것:** 현재 frame 전역에서 외형만으로 대상을 다시 찾을 수 있는가. B_att, H_s8, G_neg 시험의 공통 기준선이다.

**Stage 1 결과:** recall 0.389, 원시 FP 1.00, 위치 정확도 top-1 0.58 / top-3 0.68. hit은 drone 40, kite-surf 10, lady-running 1 frame (3영상 → 진출 조건 4영상 미달). ${POINT_FP} GLOBAL 묶음 순위화기로는 양의 순 이득이 나오지 않았다.`),

  "stage1.gen.B_att": g("B_att: 대조 token attention 전역 탐색", `**계열:** B (attention, 위치 ① 전역 탐색). 후보 ID B1.

**메커니즘 (\`rescue.tokens\`, \`heat_contrast\`):**
1. frame 0 G 특징에서 대상 cell을 k-means로 fg token 8개, 배경 cell(마스크를 max(4, √면적) px 팽창한 영역 밖)을 bg token 32개로 요약.
2. 현재 frame 각 cell이 40개 token에 softmax attention (온도 τ = 0.1). fg token에 간 attention 질량 = 대상 사후확률 heatmap. A_cos와 달리 bg token과 닮은 cell(배경 방해물)은 억제된다.
3. top-3 peak → G 상태에 양성 점 prompt. 후보 3개.

**검증하는 것:** 사전 등록 시험 "B_att > A_cos" — 대조 attention이 단일 prototype보다 탐색을 잘하는가.

**Stage 1 결과:** recall 0.389 (A_cos와 동일, Δ 0.000 [−0.014, +0.030]), 위치 정확도도 동일 → **attention의 탐색 이득 없음.** 4영상 hit(drone 39, kite-surf 10, dog-gooses 1, lady-running 1)이라 유일하게 adv_recall ✓지만, 순위화 후 FP/순 이득 조건을 통과하지 못했다. ${POINT_FP}`),

  "stage1.gen.H_s8": g("H_s8: stride-8 특징 prototype 매칭", `**계열:** H (표현). 후보 ID H1 (진단적).

**메커니즘 (\`scripts/m8.py: raw_s8\`):** A_cos와 같은 prototype 코사인 매칭을, SAM 2 이미지 인코더의 **stride-8 FPN neck 특징(256-d, 128 × 128 격자)**에서 한다. top-3 peak (NMS 반경 6 cell = 48 px) → G 상태에 양성 점 prompt. 후보 3개.

**검증하는 것:** 표현 점유율 가설 — 작은 대상이 stride-16 격자에서 너무 적은 cell을 차지해 못 찾는 것이라면, 더 촘촘한 특징이 나아야 한다 ("H_s8 > A_cos").

**Stage 1 결과:** recall 0.237 (A_cos 대비 −0.153), 위치 정확도 top-1 0.15 (A_cos 대비 −0.435, CI가 0을 완전히 벗어남) → **가설 기각.** top_J(top-1 후보)가 특히 낮다. ${POINT_FP}

**주의:** 사전 등록 수정 2: 원래 코드가 SAM 2가 이미 64-d conv_s1 투영으로 바꿔 놓은 특징을 읽고 있었다. shard 0 표를 본 뒤 고쳤고, shard 0 단위는 stage-1b 재실행으로 H_s8을 다시 계산했다.`),

  "stage1.gen.G_neg": g("G_neg: 방해물 음성 점 prompt", `**계열:** G (prompt). 후보 ID G1. 선행연구 근접 (PerSAM).

**메커니즘:** 양성 점 = B_att top-1 peak. 음성 점 = A_cos peak 중 (a) 양성 점에서 Chebyshev 거리 3 cell 초과이고 (b) B_att 사후확률 < 0.5인 것 — 즉 "prototype은 대상처럼 보지만 대조 attention은 배경으로 보는" 방해물. G 상태에 한 번에 prompt. 후보 1개. 해당 peak이 없으면 양성 점만 준다.

**검증하는 것:** 방해물에 음성 점을 주면 SAM 2가 대상과 방해물을 구별해 recall이 오르는가.

**Stage 1 결과:** recall 0.305, B_att 계열 대비 −0.084 [−0.136, −0.010] → **음성 점이 해롭다.** 단, 이 비교는 후보 1개 vs B_att top-3이다. top-1끼리(B_att#0 0.359, 2026-10-14 diag CSV 재계산) 비교해도 약 −0.05 낮다. ${POINT_FP}`),

  "stage1.gen.D_ref": g("D_ref: frame-0 메모리만 읽는 재prompt", `**계열:** D (메모리) × B. 후보 ID B4 (M7 M1 변형).

**메커니즘:** B_att top-1 peak에 양성 점 prompt를 주되, planner \`sggl._cond_only\`로 SAM 2 memory attention이 **prompt frame(frame 0)의 공간 메모리와 object pointer만** 읽게 한다. 최근 frame 메모리는 무시. G 상태. 후보 1개.

**검증하는 것:** 놓친 구간의 최근 메모리가 오염돼 재식별을 방해하는가. 오염되지 않은 초기 참조만 쓰면 나아지는가 (attention 위치 ④ 메모리 판독).

**Stage 1 결과:** recall 0.351, B_att 계열 대비 −0.038 [−0.058, −0.008]. 그러나 같은 점을 전체 메모리로 쓴 B_att#0 단독 recall은 0.359 (2026-10-14 diag CSV 재계산)로 거의 같다. 즉 **메모리 제한의 효과는 0에 가깝고**, 차이는 대부분 후보 수(1 vs 3) 때문이다. regular recall이 0.80으로 낮다 — 쉬운 frame에서는 최근 메모리가 필요하다. ${POINT_FP}`),

  "stage1.gen.A_cos_crop": g("A_cos_crop: prototype peak로 L crop 배치", `**계열:** A (전역 탐색 → 위치 제어). 후보 ID A2.

**메커니즘:** A_cos heatmap top-1 peak을 중심으로 L 상태의 crop(변 = SGGL L과 같은 prompt box의 4배)을 \`vcam.place\`로 놓고, 그 crop 이미지로 **L 상태(전체 메모리)**에서 prompt 없이 비저장 probe. 존재 판정은 SAM 2 자신이 한다.

**검증하는 것:** 전역 탐색 위치를 점 prompt 대신 "L이 볼 곳"으로만 쓰면, SAM 2 자체 존재 판정이 살아 있어 FP가 낮아지는가.

**Stage 1 결과:** recall 0.099, 원시 FP 0.135. hit은 drone 5, kite-surf 3, skate-park 5. 점 prompt 계열보다 FP가 크게 낮지만 recall도 크게 낮다. LOCAL 묶음 후보. score > 0 규칙으로 재생한 순 이득은 음수(문서 −38.0).`),

  "stage1.gen.B_att_crop": g("B_att_crop: attention peak로 L crop 배치", `**계열:** B (attention, 위치 ② crop 제어). 후보 ID B2.

**메커니즘:** B_att heatmap top-1 peak에 L crop(변 = prompt box의 4배)을 놓고, L 상태(전체 메모리)에서 prompt 없이 probe.

**검증하는 것:** A_cos_crop 대비 — attention 위치가 crop 배치에서 더 나은가.

**Stage 1 결과:** recall 0.130, 원시 FP 0.124 (A_cos_crop 0.099 / 0.135보다 약간 낫다). hit: drone 9, kite-surf 3, skate-park 5. score > 0 규칙 재생 순 이득 −30.2 (회수 17, 거짓 양성 44; \`STAGE1_DIAGNOSTIC_RESULTS.md\` §4) → 약간 낫지만 둘 다 순손실.`),

  "stage1.gen.E_beam1": g("E_beam1: 1× 속도 외삽 crop", `**계열:** E (다중 가설 beam). 후보 ID E1. M7 M3 확장, 선행연구 근접 (SAM2Long).

**메커니즘 (\`motion.Track.extrapolate\`):** 마지막으로 SGGL 출력이 비어 있지 않았던 frame의 box 중심 + 1 × 속도 × (t − 그 frame). 속도 = 마지막 두 신뢰 box 중심 차이 / frame 간격. 그 위치에 L crop(변 그대로)을 놓고 L 상태에서 prompt 없이 probe.

**검증하는 것:** "C/E > Lp" — 등속 운동 가설로 사라진 대상의 위치를 맞힐 수 있는가.

**Stage 1 결과:** recall 0.000, 원시 FP 0.003. 동시 부재 frame에서 한 번도 대상을 되찾지 못했다 → 등속 외삽 rescue 실패. Lp 대비 −0.061.`),

  "stage1.gen.E_beam2": g("E_beam2: 2× 속도 외삽 crop", `**계열:** E (다중 가설 beam). 후보 ID E1의 두 번째 가설.

**메커니즘:** E_beam1과 같고 속도 배수만 2 (마지막 신뢰 중심 + 2 × 속도 × 경과 frame). 대상이 가속하거나 SGGL 중심이 뒤처지는 경우를 겨냥한 두 번째 beam이다. L 상태, prompt 없음.

**Stage 1 결과:** recall 0.000, 원시 FP 0.003. E_beam1과 같이 회수 없음.`),

  "stage1.gen.C_flowcrop": g("C_flowcrop: 광류 운반 box 중심으로 L crop", `**계열:** C (운동). 후보 ID C1. 선행연구 근접 (SAM-PT, MoSAM).

**메커니즘 (\`motion.Flow\`, \`Track.step_flow\`):** RAFT-small(torchvision, VOS 데이터 미사용)로 frame t−1 → t 광류(긴 변 640으로 축소)를 구한다. 추적 box를 box 중앙 절반 영역의 median 광류만큼 옮긴다. SGGL 출력이 비어 있지 않으면 box를 그 출력으로 다시 고정한다. 동시 부재 중에는 광류로만 계속 운반된다. 그 box 중심에 L crop(변 그대로)을 놓고 L 상태에서 prompt 없이 probe.

**검증하는 것:** 외형과 독립인 운동 증거가 보호 메모리 probe(Lp)보다 대상을 잘 되찾는가 ("C/E > Lp").

**Stage 1 결과:** recall 0.015 (2영상), 원시 FP 0.003. Lp 대비 −0.046. 대상이 사라지면 광류 box가 배경 흐름을 따라 표류한다 (shard 0: GT와 겹치는 비율 19%). score > 0 규칙 재생 순 이득 +0.9 (회수 2, 거짓 양성 1).

**연결:** 같은 광류를 **놓치기 전부터** L crop 중심 추종에 쓰는 Stage 2 arm FLOWVEL은 PROMISING이다. 사후 재탐색이 아니라 놓치는 일 자체를 줄인다.`),

  "stage1.gen.C_flowbox": g("C_flowbox: 광류 box를 G box prompt로", `**계열:** C (운동) × G (prompt). 후보 ID C2.

**메커니즘:** C_flowcrop과 같은 광류 운반 box를 각 변 10% + 2 px 넓혀 **G 상태에 box prompt**(SAM 2 라벨 2/3 = box 모서리)로 준다. 후보 1개.

**검증하는 것:** 운동 위치 + prompt의 강제력으로 대상을 분할할 수 있는가 ("recall ≤ A1"이면 반증).

**Stage 1 결과:** recall 0.046 (drone 6 frame만), 원시 FP 0.80. box prompt도 SAM 2가 거의 항상 무언가를 분할하게 만든다. box가 표류해 대상과 겹치지 않으면 엉뚱한 것을 분할한다.`),

  "stage1.gen.Gp": g("Gp: G 보호 메모리 probe (M7 기준)", `**계열:** D (메모리, M7 M1 재사용). 후보 ID D1. **기준선.**

**메커니즘 (\`sggl.Dual.probe(..., "protected")\`):** G 상태(전체 frame view)에서 prompt 없이, memory attention이 prompt frame 메모리와 object pointer만 읽게 한 비저장 예측. 존재 판정은 SAM 2 자신.

**역할:** M7에서 이미 시험한 가장 단순한 rescue. 차트의 점선(기준 recall) = max(Gp, Lp) recall. 사전 등록 진출 조건 "recall ≥ 2 × 기준"의 분모다.

**Stage 1 결과:** recall 0.069 (kite-surf 8, skate-park 1), 원시 FP 0.245. regular recall 0.82로 쉬운 frame에서도 약하다.`),

  "stage1.gen.Lp": g("Lp: L 보호 메모리 probe (M7 기준)", `**계열:** D (메모리, M7 M1 재사용). 후보 ID D1. **기준선.**

**메커니즘:** L 상태의 **현재 crop 위치**에서 prompt 없이, prompt frame 메모리와 object pointer만 읽는 비저장 예측.

**역할:** 기준 recall(max Gp/Lp)의 다른 한쪽. 운동 시험(C_flowcrop − Lp, E_beam1 − Lp)의 비교 대상이다: crop 위치를 운동으로 옮기는 것이 같은 위치에서 메모리만 바꾸는 것보다 나은가.

**Stage 1 결과:** recall 0.061 (dog-gooses 2, kite-surf 6), 원시 FP 0.076. Gp와 hit 영상이 다르다.`),
}

// F family: occupancy-controlled crop (OCLR) vs context-preserving foveation (CPFR) at matched magnification
const F_MECH = (fov: boolean, cells: number, mem: "G" | "L", pt: boolean) => `**계열:** F (척도). 후보 ID ${fov ? "F2 (CPFR, 선행연구 근접: Learning to Zoom, FOVEA, LZU)" : "F1 (OCLR, 보고서 1순위)"}.${pt ? " `_pt` = 사전 등록 수정 1로 추가한 점 prompt 변형." : ""}

**메커니즘 (\`rescue.occupancy_alpha\`, \`axis_map\`, \`view_image\`, \`view_to_frame\`):**
- 중심 = 광류로 운반한 box 중심 (C_flowcrop과 같은 \`Track.flow_center\`).
- 배율 α = ${cells} × 16 / d̂ (d̂ = 마지막 신뢰 출력 box의 긴 변, px). 대상 긴 변이 1024 입력에서 **${cells}개 stride-16 cell(${cells * 16} px)**을 차지하도록 맞춘다. α가 전체 frame 배율보다 작으면 그냥 전체 frame.
- ${fov
    ? `**CPFR (context-preserving foveation):** 축마다 분리된 구간 선형 사상. 중심 주변 폭 1024/(2α) 구간(= ${32 / cells} × d̂)을 배율 α로 view의 절반(512 px)에 펼치고, 나머지 frame 전체를 남은 절반에 같은 비율로 압축한다. **같은 배율인데 frame 전체가 보인다** (문맥 보존).`
    : `**OCLR (occupancy-controlled crop):** 폭 1024/α (= ${64 / cells} × d̂)의 일반 crop. frame 밖으로 나가면 안쪽으로 민다. 주변 문맥은 잘린다.`}
- 이 view 이미지를 SAM 2 인코더에 통과시키고 **${mem} 상태의 메모리**로 비저장 예측${pt ? ", 그리고 광류 중심에 **양성 점 1개** prompt" : ", prompt 없음 (존재 판정은 SAM 2 자신)"}. 예측 logit을 역사상으로 원본 frame에 되돌린다 (view 밖 픽셀은 미관측).
- ${mem === "G" ? "G 메모리는 전체 frame 좌표로 쌓였다. 확대·왜곡된 view와 메모리 좌표계가 맞지 않는다." : "L 메모리는 prompt box 4배 crop(대상 ≈ 16 cell) 좌표로 쌓였다. 대상 크기가 prompt 때와 같다면 8-cell view는 그보다 약 2배 넓게 본다."}

**검증하는 것:** ${fov ? "같은 배율의 OCLR과 짝 비교 — 문맥 보존이 재식별을 돕는가 (\"CPFR > OCLR\")." : "대상 점유율(cell 수)을 맞춘 확대가 놓친 대상을 되찾는가. 같은 배율 CPFR과의 짝 비교 기준."}${pt ? " `_pt`: 메모리만으로는 재식별이 안 될 때 점 prompt로 강제하면 회수되는가." : ""}`

type FOut = { r: string; fp: string; note: string }
const F_OUT: Record<string, FOut> = {
  F_oclr8_L: { r: "0.176", fp: "0.014", note: "hit: drone 22, lady-running 1. **prompt 없는 생성기 중 가장 좋은 recall-FP 조합.** score > 0 규칙(학습 파라미터 없음)으로 재생한 순 이득 +14.3 (채택 31, 회수 23, 거짓 양성 5) — Stage 1에서 판단 규칙 없이 양의 순 이득을 낸 유일한 생성기다. 사전 진출 조건(4영상)은 통과하지 못해, Stage 2에서 '진단적 인과 arm' RESC_OCLR(+ADMIT)로 돌렸다 (결과는 Arms 페이지)." },
  F_cpfr8_L: { r: "0.168", fp: "0.110", note: "hit: drone 22. OCLR8_L과 같은 대상을 회수하지만(recall Δ −0.008, CI가 0 포함) 원시 FP가 8배다. score > 0 규칙 재생 순 이득 −20.8 (회수 22, 거짓 양성 39, 그중 miami-surf 33) — 문맥이 보이면 비슷한 방해물(다른 서퍼)을 '존재'로 잡는다. **문맥 보존 가설 지지 안 됨.** Stage 2 진단 arm RESC_CPFR." },
  F_oclr8_G: { r: "0.008", fp: "0.020", note: "G 메모리에서는 거의 '존재'를 내지 않는다. 확대된 view와 전체 frame 메모리 좌표가 맞지 않아 memory attention이 대상을 다시 찾지 못한다 (수정 1의 관찰)." },
  F_cpfr8_G: { r: "0.015", fp: "0.048", note: "OCLR8_G와 같이 바닥. CPFR8 − OCLR8 (G) 시험은 바닥끼리의 비교라 정보가 적다." },
  F_oclr4_G: { r: "0.015", fp: "0.006", note: "바닥. score > 0 규칙 재생 순 이득 +0.6 (회수 2, 거짓 양성 2)." },
  F_cpfr4_G: { r: "0.008", fp: "0.014", note: "바닥. CPFR4 − OCLR4 (G) 시험도 차이 없음." },
  F_oclr8_L_pt: { r: "0.023", fp: "0.910", note: "점 prompt를 주면 '존재'는 거의 항상 출력되지만 recall은 prompt 없는 F_oclr8_L(0.176)보다 오히려 크게 낮다. 광류 중심이 대상에서 벗어나 있으면 그 점이 엉뚱한 것을 분할하게 만든다." },
  F_cpfr8_L_pt: { r: "0.000", fp: "0.946", note: "회수 없음, 원시 FP ~0.95." },
  F_oclr8_G_pt: { r: "0.031", fp: "0.921", note: "G 메모리의 좌표 불일치를 점 prompt로 강제해도 recall이 거의 오르지 않는다." },
  F_cpfr8_G_pt: { r: "0.000", fp: "0.938", note: "회수 없음." },
  F_oclr4_G_pt: { r: "0.038", fp: "0.924", note: "`_pt` 변형 중 가장 높지만 바닥 수준(2영상, dog-gooses 위주)." },
  F_cpfr4_G_pt: { r: "0.015", fp: "0.927", note: "바닥." },
}

for (const [name, o] of Object.entries(F_OUT)) {
  const m = name.match(/^F_(oclr|cpfr)(\d)_(G|L)(_pt)?$/)!
  const fov = m[1] === "cpfr", cells = Number(m[2]), mem = m[3] as "G" | "L", pt = !!m[4]
  gens[`stage1.gen.${name}`] = g(
    `${name}: ${fov ? "문맥 보존 foveation" : "점유율 맞춤 crop"} (${cells} cell, ${mem} 메모리${pt ? ", 점 prompt" : ""})`,
    `${F_MECH(fov, cells, mem, pt)}\n\n**Stage 1 결과:** recall ${o.r}, 원시 FP ${o.fp}. ${o.note}`,
  )
}

export const help: HelpMap = { ...help_, ...gens }
