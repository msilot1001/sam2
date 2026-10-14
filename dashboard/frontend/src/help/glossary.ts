import type { HelpMap } from "./types"

// keys: "glossary.<group>.<term>" (shown on the 용어집 tab, grouped by <group>) and "glossary.ui.<element>" (UI helps, not listed there)
export const help: HelpMap = {
  // ───────────────────────────── SAM 2 기초
  "glossary.sam2.sam2": {
    title: "SAM 2.1 Hiera-T (frozen)",
    body: `**무엇:** Meta의 영상 분할 모델 SAM 2.1 중 가장 작은 backbone(Hiera-Tiny, 설정 파일 \`sam2.1_hiera_t.yaml\`)이다. 이 프로젝트의 모든 방법(SGGL, 모든 arm, 모든 생성기)은 **같은 가중치를 고정(frozen)한 채** 쓴다. 미세조정은 하지 않는다.

**한 frame의 처리 흐름:**
1. **image encoder (Hiera + FPN neck):** 1024×1024 입력 → stride 4/8/16 특징 (FPN 단계 참조). 최상위 stride-16 특징은 64×64 격자다.
2. **memory attention:** 현재 frame 특징이 메모리 뱅크의 과거 frame 특징·object pointer에 attention해서 '대상이 어디인지'를 조건화한다.
3. **mask decoder:** 1×1×256×256 저해상도 logit, predicted IoU, object score logit을 낸다. logit을 원래 해상도로 키워 마스크를 만든다.
4. **memory encoder:** 출력 마스크와 특징을 합쳐 메모리 뱅크에 넣는다(저장하지 않는 probe는 이 단계를 건너뜀).

**이 연구에서의 의미:**
- 작은 대상은 1024 입력에서 몇 개 stride-16 cell만 차지한다. 그래서 전체 frame만 보는 공식 SAM 2는 작은 대상을 놓치기 쉽다. SGGL은 crop 상태(L)를 하나 더 돌려 이 문제를 줄인다.
- 실행 조건: CUDA FP32, \`fill_hole_area = 0\`(hole filling 끔). RTX 5090/4090 worker에는 \`sam2._C\`가 있어 기본값이면 hole filling이 켜지므로, 기계 간 일치를 위해 명시적으로 0으로 둔다(\`docs/m7/MIGRATION_RTX5090.md\`).

**주의:** 가중치가 고정이므로 '모델을 더 잘 학습시킨' 결과는 없다. 모든 차이는 **어떤 view를 주고, 무엇을 메모리에 넣고, 어떤 출력을 고르느냐**에서 나온다.`,
  },
  "glossary.sam2.memory_bank": {
    title: "메모리 뱅크 (memory bank)",
    body: `**무엇:** SAM 2 상태(state)가 frame마다 쌓는 과거 정보 저장소다. 각 항목은 memory encoder가 만든 공간 메모리 특징(64×64×64)과 object pointer다.

**구성 (Hiera-T 설정 \`num_maskmem = 7\`):** prompt frame(조건 frame) 메모리 + 최근 frame 메모리 최대 6개를 다음 frame의 memory attention이 읽는다.

**왜 중요한가:**
- 다음 frame에서 대상을 '다시 찾는' 근거가 전부 메모리에 있다. 잘못된 마스크가 메모리에 들어가면 이후 frame이 그 잘못된 대상을 따라간다(**신원 전환**, identity switch).
- G와 L은 **서로 다른 상태**라 메모리를 공유하지 않는다. L의 메모리는 crop 좌표계에만 존재한다.
- M7 Gp/Lp(보호 메모리), M8 D_ref(frame-0 메모리만 읽기), ADMIT(rescue를 메모리에 기록)는 모두 '무엇을 메모리로 읽고 쓰는가'를 바꾸는 실험이다.

**관찰 (M8 Stage 1, 2026-10-14 기준):** 확대·왜곡된 view를 G 메모리로 돌리면 대상을 거의 찾지 못했다(recall ≤ 0.015). 메모리의 좌표계·배율과 현재 view가 맞지 않으면 memory attention이 대상을 다시 찾지 못한다.`,
  },
  "glossary.sam2.object_score": {
    title: "object score logit (객체 점수)",
    body: `**무엇:** SAM 2 mask decoder가 frame마다 내는 스칼라 logit으로, '이 frame에 대상이 보이는가'에 대한 모델의 판단이다.

**규칙:**
- **> 0** → 대상 존재. 마스크 logit을 그대로 쓴다.
- **≤ 0** → 대상 부재. SAM 2가 마스크 logit을 매우 큰 음수로 덮어써서 **출력 마스크가 비게 된다**.

**이 연구에서 쓰이는 곳:**
- SGGL self gate의 첫 조건(L score > 0).
- 동시 부재: G와 L의 score가 모두 ≤ 0인 frame. SGGL 출력이 비는 유일한 경우다.
- Stage 1 후보의 '존재 주장' 판정: \`present = score > 0 그리고 마스크 면적 > 0\`. 원시 FP 계산에 쓰인다.
- 순위화기 R0는 이 score만으로 후보를 고르고, RESC_* arm은 \`score > 0\`이면 rescue를 채택한다.

**해석 함정:** score는 '확률'이 아니라 logit이다. 0 근처 값은 불확실하고, 큰 양수라도 **비슷한 방해물을 대상으로 확신하는 경우**가 있다(miami-surf의 서퍼들). Stage 1에서 score 기반 존재 판정의 AUC는 0.60–0.71 수준이었다(\`STAGE1_DIAGNOSTIC_RESULTS.md\` §3).`,
  },
  "glossary.sam2.pred_iou": {
    title: "predicted IoU (예측 IoU)",
    body: `**무엇:** SAM 2 mask decoder가 출력 마스크마다 함께 내는 '이 마스크가 GT와 얼마나 겹칠 것 같은가'의 자기 추정치(0–1)다. multimask 출력에서 어떤 마스크를 고를지 정할 때도 쓰인다.

**이 연구에서:**
- \`View.iou\` 필드로 기록된다(\`src/spatial/sggl.py\`).
- Stage 1 순위화기 R2/R3의 입력 특징 중 하나다(\`ranker.SCALARS\`: score, iou, cos_fg, cos_bg, margin, log_area, dist_flow).

**주의:** predicted IoU는 **대상이 맞다는 전제에서의 마스크 품질** 추정이다. 엉뚱한 물체를 잘 분할하면 높게 나온다. 그래서 '맞는 대상인가'를 가르는 신호로는 약하다. 존재 판정에는 object score가 주로 쓰인다.`,
  },
  "glossary.sam2.prompt_frame": {
    title: "prompt frame (frame 0)",
    body: `**무엇:** 각 unit의 첫 frame(frame 0)이다. GT 마스크를 SAM 2에 prompt로 주는 유일한 frame이다.

**규칙:**
- **GT는 frame 0 prompt와 오프라인 채점에만 쓴다.** 이후 frame의 GT나 미래 정보는 알고리즘에 들어가지 않는다(누설 금지, M7 규칙).
- 지표는 **frame 1..n−1**에서만 계산한다. frame 0은 GT를 받았으므로 제외한다.
- 크기 분층(small / tiny)은 frame 0 prompt 마스크의 면적 비율(\`area0\`)로 정한다.
- L crop의 변 길이도 frame 0 prompt box로 정해진다(crop 변 4×).
- B_att, A_cos 등의 frame-0 prototype/token도 이 마스크에서 뽑는다.`,
  },
  "glossary.sam2.p5": {
    title: "P5 prompt 메모리",
    body: `**무엇:** M5.5에서 도입한 수정이다. prompt frame의 메모리를 SAM 2의 256×256 저해상도 logit 경로가 아니라 **전체 해상도 GT 마스크**로 인코딩한다.

**왜:** 기본 경로는 prompt 마스크를 256 logit으로 줄였다가 메모리에 넣는다. 작은 대상은 이 과정에서 공간 상세를 잃는다(\`MILESTONE_5_5.md\`).

**이 연구에서:**
- SGGL의 G와 L 모두 P5로 prompt한다.
- M7 표의 'Δ vs P5'는 crop 없이 전체 frame + P5만 돌린 기준선(B1) 대비 차이다. 예: SGGL은 DEV+REP 작은 객체에서 P5 대비 +0.011 [−0.010, +0.027](\`docs/m7/FINAL_ALGORITHM_DECISION.md\`, miami-surf obj5 −0.477 포함).
- crop이 frame 전체보다 커지는 대상(변 ≥ min(H, W))은 L 없이 P5만 돈다.`,
  },
  "glossary.sam2.stride16": {
    title: "stride-16 특징 / feature cell",
    body: `**무엇:** SAM 2 image encoder 최상위 특징은 입력 1024×1024를 16배 줄인 **64×64 격자**다. 격자의 칸 하나(16×16 입력 px)를 **feature cell**(이하 cell)이라 부른다.

**왜 중요한가:** 작은 대상이 몇 cell을 차지하느냐가 SAM 2가 대상을 '볼 수 있는가'에 영향을 준다는 가설이 있다(occupancy).
- 전체 frame(G)에서 prompt 면적 < 1%인 대상은 한 변이 몇 cell에 불과할 수 있다.
- L crop은 변 = 4× prompt box이고 1024로 리사이즈되므로, prompt 시점의 대상 긴 변은 1024/4 = 256 px ≈ **16 cell**이 된다.

**쓰이는 곳:**
- A_cos/B_att heatmap은 stride-16 cell 단위로 계산하고, peak도 cell 좌표로 찾는다(\`rescue.peaks\`).
- F 계열 이름의 숫자(\`F_oclr8\`, \`F_oclr4\`)는 대상 긴 변이 **8 또는 4 cell**이 되도록 배율을 맞춘다는 뜻이다(\`rescue.occupancy_alpha = cells × 16 / 대상 긴 변 px\`).`,
  },
  "glossary.sam2.fpn": {
    title: "FPN 단계 (stride 4 / 8 / 16)",
    body: `**무엇:** SAM 2 image encoder의 neck(FPN)이 내는 다중 해상도 특징이다. Hiera-T 설정(\`scalp: 1\`)에서는 가장 거친 단계를 버리고 **stride 4, 8, 16** 세 단계를 쓴다.
- stride 16 (64×64): memory attention과 mask decoder의 주 입력.
- stride 4, 8: mask decoder가 고해상도 상세를 복원할 때 쓰는 보조 특징(\`use_high_res_features_in_sam\`).

**이 연구에서:** H 계열 \`H_s8\`은 A_cos와 같은 prototype 매칭을 **stride-8 특징**으로 한다. '해상도가 높은 특징이면 작은 대상을 더 잘 찾는다(표현 점유율 가설)'를 시험하는 진단이다.

**결과 (2026-10-14 기준, \`STAGE1_DIAGNOSTIC_RESULTS.md\` §2):** H_s8은 A_cos보다 recall −0.153, 위치 정확도 −0.435로 **더 나빴다**. stride-8 특징은 memory/decoder가 쓰도록 학습된 표현이 아니라 매칭 공간으로 덜 적합하다는 해석과 일치한다. 이 비교에는 수정 2(H_s8 특징 버그 수정, 사전 등록 참조)가 적용됐다.`,
  },

  // ───────────────────────────── SGGL
  "glossary.sggl.sggl": {
    title: "SGGL (Self-Gated Global-Local)",
    body: `**무엇:** 현재 기준 알고리즘(incumbent). M6에서 A2x로 정의되었고 M7 최종 결정이 **KEEP_SGGL**이었다. M8의 모든 arm은 SGGL과 비교한다.

**구성:** 고정된 SAM 2.1 Hiera-T 상태 두 개를 한 영상에 동시에 돌린다.
- **G**: 전체 frame, P5 prompt 메모리.
- **L**: 대상 주변의 고정 크기 정사각 crop(변 = 4× prompt box). 중심은 직전 출력을 따라간다.

**frame t의 출력:**
1. G와 L이 각자 마스크와 object score를 낸다(각자 자기 메모리에 저장).
2. self gate 통과 → L 마스크를 frame 좌표로 붙여 출력. 아니면 G 마스크.
3. 다음 crop 중심 = 출력 마스크 box 중심. 출력이 비면 중심을 그대로 둔다.

**비용:** crop이 적용된 객체는 frame마다 SAM 2 track step 2회(G, L). RTX 2080에서 ≈0.44 s/frame, 최대 VRAM 1.1–1.6 GB(M7 측정).

**알려진 실패 (M7–M8 분석):**
- 동시 부재: 두 view 모두 대상을 놓쳐 출력이 비는 frame. 가장 큰 손실 범주다.
- 거짓 재획득: 진짜 부재 뒤 L이 비슷한 방해물을 높은 확신으로 잡는 경우(miami-surf obj5).
- crop 이탈: 대상이 빨리 움직여 crop 밖으로 나감(drone obj1).

**인터페이스:** \`src/spatial/sggl.py\` (\`Dual\`, \`self_gate\`, \`a2x\`).`,
  },
  "glossary.sggl.g": {
    title: "G (전역 상태, global view)",
    body: `**무엇:** SGGL의 첫 번째 SAM 2 상태. 원본 frame 전체를 1024 입력으로 보고, frame 0을 P5로 prompt한다. crop이 항등 box면 공식 SAM 2 frame과 비트 단위로 같다(M6 테스트).

**역할:**
- L이 self gate를 통과하지 못할 때의 출력(fallback).
- 넓은 시야라 대상이 crop 밖으로 나가도 원칙적으로 볼 수 있다. 그러나 작은 대상은 몇 cell만 차지해 자주 놓친다.

**관련 기록 열:** \`score_G\`(G object score), \`JF_G\`(G 마스크만 썼을 때의 J&F).

**변형:** \`Gp\`(G 보호 메모리 probe: prompt 메모리만 읽기)는 M7 M1에서 나온 진단으로, M8 Stage 1의 기준 생성기다(L2 / Lp / Gp).`,
  },
  "glossary.sggl.l": {
    title: "L (국소 crop 상태, local view)",
    body: `**무엇:** SGGL의 두 번째 SAM 2 상태. frame에서 대상 주변의 정사각 crop(가상 카메라, \`src/spatial/vcam.py\`)을 잘라 1024로 키운 영상을 본다. 이 가상 영상은 '한 배율로 팬만 하는 평범한 영상'이라 SAM 2 메모리가 모두 crop 좌표에 있다.

**crop 규칙:**
- 변 길이 s는 frame 0에서 한 번 정하고 고정한다(crop 변 4×).
- 중심은 정책이 정한다. SGGL은 직전 출력 box 중심(\`center = "out"\`), M3vel은 감쇠 속도, FLOWVEL은 광류. 출력이 비면 SGGL은 중심을 그대로 둔다.
- box는 frame 안으로 밀어 넣는다.

**강점과 약점:**
- 대상이 ≈16 cell을 차지하므로 작은 대상의 분할·검출이 G보다 훨씬 좋다.
- 시야가 좁아, 대상이 crop 밖으로 나가면 L은 대상을 볼 수 없다. M8 분석: GT가 있는 동시 부재 frame에서 대상이 **완전히 crop 밖**인 비율이 47%였다(\`FAILURE_EPISODE_ANALYSIS.md\`, 2026-10-14 기준).

**관련 기록 열:** \`score_L\`, \`acc\`(L이 self gate를 통과해 출력으로 쓰였는지).`,
  },
  "glossary.sggl.crop_side": {
    title: "crop 변 = 4× prompt box",
    body: `**정의 (\`vcam.crop_side\`):**
- s = 4 × (frame 0 prompt 마스크 box의 긴 변).
- 하한: s ≥ min(H, W) / 4 (\`MAX_ZOOM = 4\`, M6 사전 등록). 너무 작은 crop으로 과도하게 확대하지 않는다.
- s ≥ min(H, W)이면 crop을 쓰지 않고 P5(G만)로 돈다.

**결과:** 1024로 리사이즈하면 prompt 시점 대상 긴 변 ≈ 256 px ≈ 16 stride-16 cell. 대상 주변에 대상 크기의 약 1.5배 여백이 사방으로 남는다.

**왜 '고정'인가:** 변이 바뀌면 가상 영상의 배율이 바뀌어 메모리 좌표계가 어긋난다. 그래서 대상이 커지거나 작아져도 s는 그대로다. L2는 같은 중심에 변 2배(8×)인 두 번째 crop이다.

**함의:** 대상이 한 frame에 crop 반 변(≈2× box) 이상 움직이면 crop 밖으로 나간다. 이것이 M3vel/FLOWVEL 같은 crop 중심 제어가 겨냥하는 실패다.`,
  },
  "glossary.sggl.l2_lp": {
    title: "L2 / Lp / Lt / Gp (M7 대체 후보)",
    body: `M7에서 '두 view가 모두 틀릴 때 다른 후보가 맞힐 수 있는가'를 보려고 기록한 추가 view다(\`docs/m7/OPPORTUNITY_AND_FAILURE_AUDIT.md\`).

| 이름 | 정의 |
|---|---|
| **L2** | L과 같은 중심, 변 2배(2× crop) 상태. 시야가 넓다 |
| **Lp** (L-보호) | L 상태에서 **prompt 메모리만** 읽는 probe |
| **Lt** (L-신뢰) | G와 L이 일치한 frame의 메모리만 읽는 probe |
| **Gp** (G-보호) | G 상태에서 prompt 메모리만 읽는 probe |

**M7 결과:** {G, L, L2, Lp, Lt} 선택 oracle은 +0.033이지만, 각각을 단독으로 L 대신 쓰면 오히려 나쁘다(L2 −0.008, Lp −0.076). 즉 '가끔 맞히지만 언제 맞는지 모른다'.

**M8에서:** Gp, Lp는 Stage 1의 **기준 생성기**다. 진출 조건이 '같은 실행의 max(Gp, Lp) recall의 2배 이상'을 요구한다. 2026-10-14 기준 Gp/Lp recall은 0.069 / 0.061이다.`,
  },
  "glossary.sggl.self_gate": {
    title: "self gate (자기 gate)",
    body: `**정의 (\`sggl.self_gate\`):** frame t에서 L 출력을 채택하는 조건. 세 가지를 **모두** 만족해야 한다.
1. L의 object score > 0
2. L 마스크가 비어 있지 않음
3. L 마스크가 frame 경계가 아닌 crop 경계에 닿지 않음

통과하면 L 마스크를 frame 좌표에 붙여 출력하고, 아니면 G 마스크를 출력한다. 외부 학습 파라미터나 threshold가 없다('self' = SAM 2 자신의 신호만 씀).

**왜 이 정도로 충분한가:** M6.1 측정에서 self gate는 {G, L} 중 더 나은 쪽을 고르는 선택 상한의 **89%**에 이르렀다. 그래서 M7은 선택 규칙을 더 정교하게 만드는 것(S 계열)으로는 이득이 거의 없음을 확인했고, M8은 '두 view가 다 틀린 경우'에 집중했다.

**한계:** gate는 '맞는 대상인가'를 모른다. L이 비슷한 방해물을 높은 score로 잡으면 그대로 통과한다(miami-surf obj5). 이를 막으려던 SGGL-V는 효과가 한 영상에 몰려 선택 사항으로 내려갔다.`,
  },
  "glossary.sggl.edge_touch": {
    title: "내부 crop 경계 접촉 (interior edge touch)",
    body: `**정의 (\`vcam.touches_border\`):** L 마스크가 crop의 가장자리 중 **frame 가장자리와 겹치지 않는 쪽**에 닿으면 '접촉'이다.
- 예: crop 위쪽 변이 frame 맨 위(y0 = 0)와 같으면, 마스크가 그 변에 닿아도 접촉이 아니다. 대상이 frame 밖으로 나간 것이지 crop에 잘린 것이 아니기 때문이다.
- crop 위쪽 변이 frame 내부(y0 > 0)인데 마스크 첫 행에 화소가 있으면 접촉이다.

**왜:** 내부 경계에 닿은 마스크는 대상이 crop 밖으로 이어진다는 뜻이라 **잘린(불완전한) 마스크**일 가능성이 크다. 이때는 전체를 보는 G 출력이 더 안전하다.

**부작용:** 대상이 crop 가장자리로 밀려나는 빠른 운동 구간에서 L이 자주 거부된다. 그러면 G가 출력하고, G도 작은 대상을 놓치면 출력이 빈다. crop 중심 제어(M3vel, FLOWVEL)가 crop을 대상 쪽으로 미리 옮겨 두면 이런 frame이 줄어든다.`,
  },
  "glossary.sggl.sgglv": {
    title: "SGGL-V (거부권 변형)",
    body: `**정의 (\`sggl.a2v\`, M6.1 A2v):** SGGL self gate에 조건 하나를 더한다. **G의 object score > 0일 때만** L을 채택한다. G가 '대상 없음'이라고 하면 L이 무엇을 잡았든 거부한다(거부권, veto).

**의도:** 진짜 부재 뒤 L이 방해물을 잡는 거짓 재획득(miami-surf obj5)을 막는다.

**M7 결과 (\`docs/m7/FINAL_ALGORITHM_DECISION.md\`):**
- DEV+REP 작은 객체: +0.0069 [−0.003, +0.024]. 그러나 **miami-surf를 빼면 −0.0019**. 이득이 한 영상에서만 난다.
- VK: 거부권이 1객체 5 frame에서만 발동, 효과 −0.00002.
- 거부권의 비용: G 지지 없이 일어난 **올바른** 재획득도 막는다(dog-gooses obj5 t82).

**결론:** 기본값이 아닌 선택 사항(ablation)으로 내렸다. M8 arm들의 기준선은 SGGL-V가 아니라 **SGGL**이다.`,
  },

  // ───────────────────────────── 지표·통계
  "glossary.metric.jf": {
    title: "J, F, J&F",
    body: `DAVIS 2017 평가 방식을 numpy로 재구현한 지표다(\`src/baseline/metrics.py\`).

**J (region, 영역 IoU):** J = |예측 ∩ GT| / |예측 ∪ GT|.
- **둘 다 비어 있으면 J = 1.** GT가 빈 frame에서 아무것도 출력하지 않으면 만점이다.
- GT가 빈 frame에 무엇이든 출력하면 J = 0. 이 비대칭 때문에 거짓 재획득과 rescue의 원시 FP가 비싸다.

**F (boundary, 경계 F-measure):** 예측 경계와 GT 경계를 비교한다. 경계 화소가 상대 경계에서 bound_pix = ceil(0.008 × 영상 대각선) 이내에 있으면 일치로 본다. F = 2PR / (P + R).
- 둘 다 비면 P = R = 1(F = 1). 예측만 비면 R = 0, GT만 비면 P = 0 → F = 0.
- \`F_strict\`는 허용 거리를 2 px로 고정한 보조 지표다.

**J&F = (J + F) / 2.** frame 단위로 계산한다.

**unit 점수:** 한 unit의 J&F = frame 1..n−1(prompt frame 제외)의 frame J&F **평균**.

**해석:** 작은 대상은 경계 허용 거리(대각선의 0.8%)가 대상 크기에 비해 커서 F가 J보다 관대할 수 있다. 그래서 ΔJ와 ΔF를 함께 본다(예: FLOWVEL ΔJ +0.0106 / ΔF +0.0154, 2026-10-14 기준).`,
  },
  "glossary.metric.small_tiny": {
    title: "small / tiny (작은 / 극소 객체)",
    body: `**정의:** frame 0 prompt 마스크의 면적 비율 \`area0\` = 대상 화소 수 / frame 화소 수.
- **small (작은):** area0 < 0.01 (1%)
- **tiny (극소):** area0 < 0.001 (0.1%). tiny는 small의 부분집합이다.

**쓰이는 곳:**
- **판정은 small unit만으로 한다** (\`_pilot\`: \`S = U[U.small]\`). 2026-10-14 기준 DEV+REP에서 57 unit / 34 영상.
- tiny는 7 unit뿐이다. \`tiny_dJF\`는 분층 보고용이며 판정에 쓰지 않는다. 7개라 한두 unit이 값을 좌우한다.
- \`all_units_dJF\`는 small이 아닌 unit까지 포함한 64 unit 평균으로, 큰 대상에서 회귀가 없는지 확인용이다.

**주의:** 크기는 **frame 0 기준**이다. 이후 대상이 가려지거나 작아져도 분층은 바뀌지 않는다. 동시 부재 frame의 45%는 대상이 초기 면적의 절반 미만으로 줄어든 상태였다(\`FAILURE_EPISODE_ANALYSIS.md\`).`,
  },
  "glossary.metric.djf": {
    title: "dJF (ΔJ&F vs SGGL)",
    body: `**정의 (\`scripts/m8_analysis.py: causal\`):**
1. unit마다 d = (arm의 unit J&F) − (**같은 기계** SGGL의 unit J&F). 같은 frame끼리 맞춰 병합하며, frame 수가 다르면 실패(assert)한다.
2. **dJF = small unit들의 d 평균.** unit 가중 평균이다(영상 가중이 아님).

같은 방식으로 \`dJ\`, \`dF\`도 낸다.

**기준선 규칙:** RunPod 4090 arm은 \`outputs/milestone8/runpod/runs/SGGL\`, Ubuntu 2080 arm은 Ubuntu diag 궤적(\`outputs/milestone8/diag/*.csv\`)과 짝짓는다(machine-matched).

**읽는 법:**
- 양수 = SGGL보다 좋음. +0.005가 PROMISING 최소 크기다.
- 크기 감각: 동시 부재 frame을 **전부** 완벽히 고쳐도 객체 평균 +0.028이다. +0.013은 이 상한의 절반 가까운 크기다.
- unit 가중이라 객체가 많은 영상(miami-surf 등)이 평균에 더 크게 기여한다. 그래서 LOVO, miami-surf 제외, 영상 ↑/↓를 함께 본다.
- 평균이 소수 unit의 큰 이득에서 나올 수 있다. 예: FLOWVEL의 drone obj1 +0.379(2026-10-14 기준).`,
  },
  "glossary.metric.bootstrap_ci": {
    title: "영상 군집 bootstrap CI",
    body: `**정의 (\`rfeas.risk.cluster_bootstrap\`):**
1. small unit들의 **영상 목록**에서 영상을 복원 추출로 같은 수만큼 뽑는다.
2. 뽑힌 영상의 unit을 모두 모아 d 평균을 계산한다.
3. 2000회 반복(seed 0)하고, 분포의 2.5%·97.5% 백분위를 95% CI로 보고한다.

**왜 영상 단위인가:** 한 영상의 여러 객체는 같은 카메라 운동·조명·방해물을 공유해 독립이 아니다. unit 단위로 뽑으면 CI가 지나치게 좁아진다. 실질 표본 수는 unit 57개가 아니라 **영상 34개**다.

**읽는 법:**
- 하한 > 0: 영상 표본 변동을 고려해도 평균이 양수. FLOWVEL [+0.0010, +0.0285]가 M7–M8에서 처음 하한이 0을 넘은 사례다(2026-10-14 기준).
- 하한이 0에 '걸친' 경우(M3vel −0.0000)는 seed에 따라 부호가 바뀔 수 있는 수준이다.

**주의:**
- 이 CI는 **같은 개발 영상에서 여러 arm을 비교한 뒤의** 단일 arm CI다. 다중 비교 보정이 없다.
- 영상이 적고 이득이 한두 영상에 몰리면 백분위 CI가 불안정하다.
- VK는 scene이 2개뿐이라 영상 대신 **객체 군집** CI를 쓰며 서술용이다.`,
  },
  "glossary.metric.lovo": {
    title: "LOVO (영상 하나씩 빼기) / miami-surf 제외",
    body: `**정의:** small unit이 있는 영상 v마다, v의 unit을 모두 뺀 나머지의 d 평균을 계산한다. 그 값들의 **최솟값**이 \`lovo_min\`이다(leave-one-video-out).

**의미:** '어떤 영상 하나를 빼도 평균이 양수로 남는가'. 이득이 한 영상에서만 오면, 그 영상을 뺐을 때 평균이 0 이하로 떨어져 LOVO ≤ 0이 된다.

**판정에서:**
- PROMISING 조건: lovo_min > 0.
- REJECTED 조건의 일부: (+0.01 이상 오른 영상 ≤ 2) **그리고** lovo_min ≤ 0.

**예 (2026-10-14 기준, \`STAGE2_CAUSAL_RESULTS.md\`):** FLOWVEL lovo_min +0.0067. 가장 큰 기여 영상인 drone을 빼도 양수다. RESC_OCLR_ADMIT은 평균 +0.0046이지만 lovo_min −0.0019라 한 영상에 의존한다.

**주의:** LOVO는 '한 영상'에 대한 견고성만 본다. 두 영상(drone + kite-surf)을 함께 빼는 경우는 보지 않는다.

**함께 보는 값 — miami-surf 제외 (\`without_miami\`):** \`davis/miami-surf\`의 unit을 모두 뺀 small unit의 d 평균. PROMISING 조건은 이 값 > 0이다.
- miami-surf는 비슷한 서퍼가 여럿 있는 영상이다. SGGL의 대표 실패(obj5: L이 다른 서퍼를 잡는 거짓 재획득, P5 대비 −0.477)가 여기서 났다.
- M6.1–M7에서 '이 한 영상만 고치는' 방법이 평균을 크게 움직이는 일이 반복됐다(SGGL-V: miami-surf를 빼면 −0.0019). 그래서 M7 pilot 규칙부터 이 조건을 넣었다.
- 전체 dJF와 비슷하면 이득이 이 영상에 의존하지 않는다. 예(2026-10-14 기준): FLOWVEL 전체 +0.0130, miami 제외 +0.0138.`,
  },
  "glossary.metric.worst_unit": {
    title: "최악 단위 (worst unit)",
    body: `**정의:** small unit 중 d가 가장 작은 unit과 그 값(\`worst_unit = [unit, d]\`).

**판정에서:** PROMISING이려면 최악 단위 ≥ **−0.05**. 평균이 좋아도 어떤 객체 하나를 크게 망치는 방법은 통과하지 못한다(재앙적 손실 금지).

**왜:** rescue나 메모리 기록은 잘못되면 **신원 전환**(다른 물체를 대상으로 착각하고 끝까지 따라감)을 일으킨다. 이런 손실은 평균에 묻히기 쉽지만 실제 사용에서는 치명적이다.

**예 (2026-10-14 기준):**
- RESC_CPFR: disc-jockey obj3 **−0.330** (foveation이 방해물을 잡아 신원 전환)
- RESC_OCLR_ADMIT: miami-surf obj4 −0.116 (잘못된 rescue가 메모리에 기록됨)
- FLOWVEL: scooter-board obj2 −0.026 (기준 통과)

**함께 보는 값:** \`best_unit\`(가장 큰 이득), \`units_up_0.01\` / \`units_down_0.01\`.`,
  },
  "glossary.metric.videos_up": {
    title: "영상 ↑ / ↓ (±0.01)",
    body: `**정의:** small unit의 d를 영상별로 평균한 뒤,
- \`videos_up_0.01\`: 영상 평균 > +0.01인 영상 수
- \`videos_down_0.01\`: 영상 평균 < −0.01인 영상 수

±0.01 사이 영상은 '변화 없음'으로 세지 않는다.

**판정에서:**
- PROMISING: 오른 영상 ≥ **3**.
- REJECTED: 오른 영상 ≤ 2 **그리고** LOVO ≤ 0.

**왜:** 이득이 2개 영상에 몰린 방법은 우연이나 영상 특이성일 가능성이 크다. M7의 M3-vel이 정확히 이 이유(skate-park, sheep 2개)로 INCONCLUSIVE에 머물렀다.

**예 (2026-10-14 기준):** FLOWVEL 5 / 2 (drone, kite-surf, skate-park, sheep, lady-running ↑ / scooter-board, lab-coat ↓).`,
  },
  "glossary.metric.false_reacq": {
    title: "거짓 재획득 (false re-acquisition)",
    body: `**정의 (인과 arm, \`causal_per_unit.csv\`의 \`fp_arm\` / \`fp_base\`):** GT가 비어 있는데 출력 J < 1인 frame 수. J 규칙상 GT가 빈 frame에서 J < 1은 '무언가를 출력했다'는 뜻이다.
- \`false_reacq_frames\`: arm의 합계
- \`false_reacq_frames_SGGL\`: 같은 unit·같은 기계 SGGL의 합계

**읽는 법:** arm 값이 SGGL보다 크면, 그 arm이 진짜 부재 구간에서 방해물을 잡는 일이 늘었다는 뜻이다.

**예 (2026-10-14 기준):** FLOWVEL 93 / 93(증가 없음), RESC_CPFR 118 / 93(foveation이 방해물 거짓 양성을 늘림).

**Stage 1의 원시 FP와 차이:** 원시 FP는 후보 하나가 '존재'를 주장했는지(비율)이고, 거짓 재획득은 인과 rollout의 **최종 출력**이 GT가 빈 frame에 무언가를 냈는지(frame 수)다.`,
  },
  "glossary.metric.multiple_comparisons": {
    title: "다중 비교 / 탐색적 결과",
    body: `**무엇:** 같은 개발 영상(DEV+REP)에서 많은 방법을 시험하면, 효과가 없어도 몇 개는 우연히 좋아 보인다.

**이 연구의 상황:**
- Stage 1: 생성기 약 25종 + 짝 비교 10개를 같은 영상에서.
- Stage 2/3: arm 10여 개를 같은 57 unit / 34 영상에서.
- DEV와 REP는 M6–M8 동안 여러 번 본 데이터다.
- CI와 판정 규칙은 **arm 하나씩** 적용하며 다중 비교 보정이 없다.

**그래서:**
- 사전 등록 문서(\`EXPERIMENT_TOURNAMENT_PLAN.md\`)는 Stage 1 결과를 모두 탐색적이라고 명시한다.
- 인과 승자도 **새 데이터나 봉인 세트로 확인되기 전까지** 탐색적이다. FLOWVEL은 출력 전에 고정했지만 PROMISING은 '확인할 가치가 있음'이지 '확인됨'이 아니다.
- 새 tiny 데이터(LVOS v2, SA-V, VOST 후보)는 다운로드 승인 대기(BLOCKED)다. PUMaVOS C는 봉인 상태다.

**읽을 때:** 'PROMISING'·'CI 하한 > 0'을 보면 항상 '탐색적, 개발 영상, 보정 없음'을 함께 떠올린다.`,
  },

  // ───────────────────────────── 데이터
  "glossary.data.unit": {
    title: "unit (단위)",
    body: `**정의:** 한 영상 안의 한 객체. 평가의 기본 단위다. 이름 형식은 \`<source>__<video>__obj<k>\`(예: \`davis__davis/car-turn__obj1\`). 파일명에서는 \`/\`가 \`_\`로 바뀐다(\`davis__davis_car-turn__obj1.csv\`).

**구성 (2026-10-14 기준):** DEV 21 + REP 43 = **64 unit**, 그중 small 57 unit / 34 영상. VK 28 unit은 별도.

**unit마다 기록되는 것 (arm CSV 한 행 = 한 frame):** \`video_idx\`, \`score_G\`, \`score_L\`, \`acc\`, \`gt_empty\`, \`J\`, \`F\`, \`JF\`, \`JF_G\`, \`rescued\`, \`gen\`, \`calls\`, \`latency_s\`, \`peak_vram_bytes\` 등.

**unit → 기계 배정:** shard 규칙(M7_SHARD)으로 고정되며, 같은 unit의 모든 arm과 기준선은 **같은 기계**에서 돈다.

**주의:** 같은 영상의 unit들은 독립이 아니다(같은 카메라 운동·방해물). 그래서 통계는 영상 군집으로 낸다.`,
  },
  "glossary.data.dev": {
    title: "DEV 세트",
    body: `**구성:** M6 개발 crop 객체 21개. DAVIS 2017 val, PUMaVOS A/B, SegTrack v2에서 왔다.

**역할:** 평가. M6부터 반복해 본 데이터라 **탐색적**이다.

**학습형 순위화기와의 관계:** 사전 등록은 'REP로 학습 → DEV로 판정'(REP→DEV)이었다. M8은 DEV→REP 방향도 추가해 두 split 모두 held-out 평가가 가능하게 했다(교차 적합).

**주의:** DEV의 라벨은 학습에 쓰지 않는다(M7 규칙). DEV→REP 방향은 사후에 추가한 것으로 공개되어 있다.`,
  },
  "glossary.data.rep": {
    title: "REP 세트",
    body: `**구성:** DAVIS 2017 **train** split의 crop 객체 43개 / 23 영상.

**역할:**
- 평가 (DEV와 합쳐 판정에 사용).
- **학습형 순위화기(R2, R3)의 학습 데이터**와 threshold 고정. 학습형 결과는 DEV로만 판정하는 것이 사전 등록이었다.

**주의:**
- miami-surf는 REP에서 발견된 실패 사례다.
- REP도 M7부터 여러 번 본 데이터라 탐색적이다. 영상이 23개라 학습 데이터로는 작다. M7 T 계열 head와 M8 R2/R3가 일반화에 실패한 원인 중 하나다.`,
  },
  "glossary.data.vk": {
    title: "VK (VKITTI2 Scene01/06)",
    body: `**구성:** Virtual KITTI 2(합성 주행 영상)의 Scene01, Scene06 clone에서 뽑은 crop 객체 28개. 같은 종류의 방해물(차량)이 많다.

**역할:** 결선 후보를 고정한 뒤 **한 번만** 돌리는 확인 세트(\`preregistration_vk.json\`). 결과는 **서술적**이다.

**왜 서술적인가:**
- scene이 2개뿐이라 영상 군집 CI가 의미 없다. 그래서 **객체 군집** CI를 서술용으로 낸다.
- M2–M4의 개발 scene이라 프로젝트 전체로는 새 데이터가 아니다.
- prompt 면적이 ≥ 0.1%로 선택되어 tiny 객체가 없다.

**판정 대신 보는 것:** 비회귀(\`non_regression_pass\`: dJF ≥ −0.005), scene별 값, 객체 ↑/↓.

**예 (2026-10-14 기준):** FLOWVEL VK +0.0002 [−0.0006, +0.0008] — 비회귀 통과, 이득 없음. VK에는 crop 이탈 사건이 거의 없다.

**규칙:** VKITTI Scene02/18/20은 보류 scene이라 쓰지 않는다.`,
  },
  "glossary.data.pumavos": {
    title: "PUMaVOS A / B / C (C는 봉인)",
    body: `**PUMaVOS:** 부분 영역·비정형 마스크(얼굴 일부, 옷 일부 등)를 대상으로 하는 VOS 데이터셋이다. 이 프로젝트는 이를 A/B/C 세 부분으로 나눠 쓴다.

**A / B:** DEV 세트의 일부. 작업자(worker)에는 A/B clip만 보낸다.

**C (봉인, sealed):** 11개 객체.
- **열지 않는다:** worker로 복사하지 않고, 마스크를 보지 않고, 튜닝에 쓰지 않는다(\`AGENTS.md\` §2).
- 용도: 최종 단일 방법이 사전 등록한 최종 규칙을 통과했을 때만 **한 번** 확인용으로 쓴다.
- M7에서 통과한 방법이 없어 봉인을 유지했다. M8도 2026-10-14 기준 봉인 상태다.

**왜 중요한가:** 개발 데이터를 반복해서 본 탓에 모든 결과가 탐색적이다. C는 그 편향이 없는 몇 안 되는 평가 자원이라, 한 번 쓰면 다시 '새 데이터'가 될 수 없다. 대시보드도 원시 데이터 경로(\`data\`)를 노출하지 않는다.`,
  },
  "glossary.data.davis": {
    title: "DAVIS 2017 / SegTrack v2",
    body: `**무엇:** 영상 객체 분할(VOS)의 표준 벤치마크. 영상마다 여러 객체에 frame별 GT 마스크가 있다. J/F 지표 정의도 이 벤치마크에서 왔다.

**이 연구에서:**
- **val** split의 일부 → DEV (M6 개발 crop 객체).
- **train** split의 일부 → REP (43 객체 / 23 영상).
- 작은 대상 실패 사례 대부분이 DAVIS 영상이다: drone, kite-surf, skate-park, miami-surf, dog-gooses, lady-running, sheep, scooter-board, lab-coat, disc-jockey 등.

**unit 이름:** \`davis__davis/<영상>__obj<k>\`.

**SegTrack v2:** 오래된 소규모 VOS 데이터셋으로, 작은 대상과 빠른 운동이 있는 짧은 clip이 많다. DEV 세트의 일부이며, RunPod worker에는 실제로 쓰는 **14개 clip**만 보낸다(\`docs/GPU_WORKER_OPERATIONS.md\`).`,
  },

  // ───────────────────────────── Stage 1
  "glossary.stage1.trigger": {
    title: "trigger frame (진단 대상 frame)",
    body: `**정의 (\`scripts/m8.py: trigger\`):** SGGL의 **현재 frame 출력만**(G, L의 score와 마스크) 보고 인과적으로 정한다. GT는 쓰지 않는다. 우선순위 순서:
1. **coabs (동시 부재):** G score ≤ 0 그리고 L score ≤ 0.
2. **disagree (불일치):** 한 view만 '존재'이거나, 둘 다 존재인데 IoU(L, G) < 0.5.
3. **regular (정기):** 위에 해당하지 않는 frame 중 t % 10 == 0 (보정용 표본).

**Stage 1에서:** trigger frame마다 25개 생성기를 비저장 probe로 돌린다. 주 분석 대상은 coabs다. disagree/regular 후보는 순위화기 학습 데이터와 '어려움 맥락'으로 쓴다.

**규모 (2026-10-14 기준, 64 unit):** coabs 486 (GT 있음 131 / 없음 355), disagree 242, regular 399.

**인과 arm에서:** RESC_* / ORACLE / K_* arm은 coabs에서만 생성기를 돈다. 나머지 frame은 SGGL과 비트 단위로 같다.`,
  },
  "glossary.stage1.coabs": {
    title: "동시 부재 (co-absence)",
    body: `**정의:** G와 L의 object score가 **모두 ≤ 0**인 frame. 두 view가 모두 '대상 없음'이라고 하므로 SGGL 출력이 빈 마스크다.

**규모 (M7 기록 재구성, \`FAILURE_EPISODE_ANALYSIS.md\`, 2026-10-14 기준):**

| 항목 | 값 |
|---|---|
| 동시 부재 frame | 486 (전체 5082의 9.6%) |
| GT 있음 (놓친 frame) | 131 |
| GT 없음 (진짜 부재: 가림·이탈) | 355 |
| 연속 사례 | 40 (GT 있는 사례 20) |
| 완전 회수 시 상한 | 객체 평균 J&F **+0.028** |

**왜 중요한가:** M7의 가장 큰 실패 범주다. M7은 '이미 맞는 위치의 분할 개선'(①)의 상한이 +0.002–0.006임을 보였고, M8은 이 ②번 실패를 겨냥했다.

**핵심 어려움:** 진짜 부재가 놓친 frame의 **2.7배**다. GT가 빈 frame에 무언가를 출력하면 J가 1 → 0이 되어(−1), 회수 하나의 이득(+0.5~0.8)보다 손실이 크다. 그래서 '찾기'보다 '정말 있는지 판단하기'가 병목이었다.

**FLOWVEL과의 관계:** FLOWVEL은 놓친 동시 부재 frame 자체를 131 → 63으로 줄였다(RunPod 같은 기계 기준, 2026-10-14). 사후 회수가 아니라 예방이다.`,
  },
  "glossary.stage1.probe": {
    title: "비저장 probe (non-storing probe)",
    body: `**정의:** frame t에서 SAM 2 예측을 **추가로** 한 번 돌리되, 결과를 어떤 상태의 메모리나 출력 dict에도 쓰지 않는 예측이다(\`rescue.probe\`, \`sggl\` probe). 건드린 image feature cache도 복원한다.

**보장:** \`tests/test_m7.py\`, \`tests/test_m8.py\`가 'probe를 끼운 rollout의 production 출력이 probe 없는 rollout과 **비트 단위로 같다**'를 확인한다. 그래서 Stage 1에서 생성기 25종을 같은 frame에 모두 돌려도 SGGL 궤적은 변하지 않는다.

**할 수 있는 것:** 다른 view 영상(crop, foveation), 다른 메모리 계획(보호 메모리 등), point/box prompt를 한 번에 줄 수 있다. 같은 view 영상의 backbone은 한 번만 계산해 공유한다(비트 동일, 약 1.7× 빠름).

**한계 (중요):** probe는 '그 frame에서 후보를 낼 수 있는가'만 잰다. 그 후보를 실제로 출력하고 crop을 옮기고 메모리에 쓰면 다음 frame이 달라진다. 이 되먹임은 인과 rollout으로만 잴 수 있다.`,
  },
  "glossary.stage1.generator": {
    title: "생성기 (rescue generator)",
    body: `**정의:** 동시 부재 frame에서 '대상일 수 있는 후보 마스크'를 만드는 방법. Stage 1에서 25개 계열(변형 포함 31개 probe)을 같은 trigger frame에 모두 돌렸다.

**이름 읽기:**
- 앞부분 = 계열과 메커니즘. 예: \`A_cos\`(A: 전역 코사인 탐색), \`F_oclr8_L\`(F: occupancy crop, 8 cell, L 메모리).
- \`#k\` = 같은 생성기의 k번째 후보(예: \`A_cos#0..2\` = top-3 peak). 분석은 \`#\` 앞 이름으로 묶는다(\`fam\`).
- \`_G\` / \`_L\` = probe에 쓴 메모리 상태. \`_pt\` = point prompt 변형(수정 1).
- \`_crop\` = heatmap peak에 L crop을 다시 놓는 변형(prompt 없음).

**후보마다 기록되는 것 (\`*.cand.csv\`):** score, iou, area, J(채점용), cos_fg/cos_bg(frame-0 외형 유사도), dist_flow(광류 box와의 거리) 등.

**목록과 상세:** \`docs/m8/EXPANDED_CANDIDATE_PORTFOLIO.md\`, 코드 \`scripts/m8.py: generators\`, \`src/spatial/rescue.py\`.`,
  },
  "glossary.stage1.recall": {
    title: "recall (회수율, Stage 1)",
    body: `**정의 (\`m8_analysis.per_frame\`, \`stage1\`):** GT가 **있는** 동시 부재 frame 중, 해당 생성기 계열의 후보 **하나라도** J > 0.5인 frame의 비율.
- 분모: GT 있는 coabs frame (2026-10-14 기준 131).
- '하나라도'이므로 순위화와 존재 판정을 하기 **전**의 상한이다. 실제로 그 후보를 고를 수 있는지는 별개다.

**함께 보는 값 (\`stage1_generators.csv\`):**
- \`mean_best_J\`: 후보 중 최고 J의 평균
- \`top_J\`: 첫 번째 후보 J의 평균
- \`videos_hit\`, \`units_hit\`: 적중이 난 영상·unit 수
- 위치 정확도 top-1/top-3: heatmap peak이 GT 대상 위에 있는 비율(A/B/H)

**예 (2026-10-14 기준):** A_cos·B_att 0.389, 전체 생성기 합집합(oracle) 0.511.

**함정:** recall이 높아도 원시 FP가 1.00이면(항상 '존재'를 주장) 그대로 쓸 수 없다. 적중의 75%가 drone·kite-surf 두 영상에 몰려 있어, recall 숫자가 소수 영상의 성질을 반영한다.`,
  },
  "glossary.stage1.raw_fp": {
    title: "원시 FP (raw false-positive rate)",
    body: `**정의:** GT가 **비어 있는** 동시 부재 frame(진짜 부재, 2026-10-14 기준 355) 중, 해당 계열의 후보 하나라도 '존재'를 주장한 frame의 비율. '존재' = score > 0 그리고 마스크 면적 > 0. CSV 열 \`fp_rate_gt_empty\`.

**'원시'인 이유:** 순위화기나 threshold를 적용하기 **전** 값이다.

**왜 중요한가:** 진짜 부재 frame에 무언가를 출력하면 J가 1 → 0(−1)이다. 진짜 부재가 놓친 frame의 2.7배라, FP가 조금만 있어도 회수 이득을 넘어선다.

**예 (2026-10-14 기준):**
- point prompt를 주는 생성기(A_cos, B_att, G_neg 등)는 거의 항상 '존재'를 출력한다(0.92–1.00). 점을 찍으면 SAM 2가 무언가를 분할한다.
- \`F_oclr8_L\`은 0.014로 매우 보수적이다. 같은 배율 foveation \`F_cpfr8_L\`은 0.110(약 8배).

**인과 단계의 대응:** 거짓 재획득.`,
  },
  "glossary.stage1.rankers": {
    title: "순위화기 R0–R3",
    body: `**역할:** 동시 부재 frame의 후보들 중 하나를 고르고(순위), 그 점수가 threshold를 넘을 때만 출력한다(존재 판정). 넘지 않으면 기권(빈 출력 유지).

| 이름 | 방식 | 학습 |
|---|---|---|
| **R0** | SAM 2 object score 그대로 | 없음 |
| **R1** | score > 0인 후보 중 외형 margin(cos_fg − cos_bg) 최대 | 없음 |
| **R2** | 후보별 로지스틱 회귀(score, iou, cos_fg, cos_bg, margin, log_area, dist_flow) | 있음 |
| **R3** | set-attention: 후보 token끼리 self-attention(경쟁) + frame-0 fg/bg token cross-attention(신원) + **기권 token** | 있음 |

**학습 라벨:** J > 0.5 → 1, J < 0.1 또는 GT 없음 → 0, 그 사이는 제외.

**생성기 묶음:** GLOBAL(A, B, D, G), LOCAL(F_L, crop 배치, Lp, Gp), POOL(전체). 파일 \`rankers_{GLOBAL,LOCAL,POOL}.json\`.

**결과 (2026-10-14 기준):** held-out 순 이득이 **양수인 순위화기 없음**. R3 GLOBAL은 항상 기권(0.0)했다. 학습형이 실패한 이유: 적중의 75%가 두 영상에 몰려, 한 split에서 배운 것이 다른 split로 옮겨가지 않는다.`,
  },
  "glossary.stage1.net_dj": {
    title: "net_dJ_frames (순 이득)",
    body: `**정의 (\`m8_analysis._accept_eval\`):** 동시 부재 frame에서 '최고 점수 후보의 점수 > threshold면 출력'을 적용했을 때 SGGL의 빈 출력 대비 frame J 변화의 **합**.

\`net_dJ_frames = Σ(채택 ∧ GT 있음) J − Σ(채택 ∧ GT 없음) 1\`

- GT 있는 frame: SGGL 빈 출력의 J는 0이므로, 채택한 후보 J만큼 이득.
- GT 없는 frame: SGGL 빈 출력의 J는 1(빈=빈)이므로, 무언가 출력하면 −1.
- 채택하지 않은 frame: 0.

**읽는 법:** 단위는 'frame J 합'이다(unit 평균 J&F가 아님). 양수여야 rescue가 이득이다. 표의 괄호 (a / b)는 (회수 frame / 거짓 양성 frame)이다.

**예 (2026-10-14 기준):** \`F_oclr8_L\` + score > 0 규칙의 재생 결과 +14.3 (회수 23 / FP 5). 같은 배율 foveation \`F_cpfr8_L\`은 −20.8 (22 / 39).

**주의:** 재생(replay) 계산이라 되먹임을 반영하지 않는다. F를 무시하고 J만 쓴다. 회수한 다음 frame에서 다시 놓치는 효과는 인과 rollout에서만 보인다(RESC_OCLR 인과 dJF −0.0001).`,
  },
  "glossary.stage1.crossfit": {
    title: "교차 적합 (cross-fitting, REP→DEV / DEV→REP)",
    body: `**정의 (\`m8_analysis.rankers\`):** 순위화기를 한 split에서 학습하고 threshold를 고정한 뒤, **다른 split**에서 평가한다. 두 방향을 모두 한다.
- **REP→DEV:** REP로 학습·threshold → DEV에서 held-out 평가. **사전 등록된 방향.**
- **DEV→REP:** 반대 방향. 사후 추가(공개됨). 모든 64 unit에 대해 out-of-sample 평가가 가능해진다.
- \`pooled_heldout\`: 두 held-out 결과의 합.

**threshold 규칙:** 학습 split의 coabs frame에서 '진짜 부재 FP율 ≤ 0.10을 지키면서 net_dJ_frames 최대'인 값. 사전 등록 문구는 'FP ≤ 0.10'뿐이었고 '순 이득 최대'는 추가 규정이다(공개됨). R3는 학습 split 안에서 out-of-fold 점수로 threshold를 정한다.

**왜:** 같은 데이터로 학습·평가하면 과적합이 이득처럼 보인다. 교차 적합은 '처음 보는 영상에서도 통하는가'를 잰다.

**결과 (2026-10-14 기준):** 모든 순위화기의 held-out 순 이득 ≤ 0.`,
  },
  "glossary.stage1.advance": {
    title: "Stage 1 → 인과 진출 조건",
    body: `**사전 등록 규칙 (\`preregistration.json\` \`advance_to_causal\`):** 생성기 계열이 인과 단계로 가려면 **모두** 만족해야 한다.
1. recall(J > 0.5) ≥ **0.25**
2. recall ≥ 같은 실행의 **max(Gp, Lp) recall의 2배**
3. **4개 이상 영상**에서 적중
4. REP에서 고정한 순위화기/threshold로 DEV 진짜 부재 frame의 FP율 ≤ **0.10**

CSV의 \`adv_recall\`은 1–3번(recall 관련)만 확인한 값이다.

**결과 (2026-10-14 기준):** **통과한 계열 없음.** A_cos/B_att는 recall은 높지만 3영상 적중이고 FP 조건을 못 지켰다.

**그 다음:** 사전 등록에 따라, 통과가 없으면 정보가 가장 큰 탈락 후보를 '진단적 인과 실행'으로 명시해 돌릴 수 있다. 그래서 Stage 1 재생에서 순 이득이 양수였던 유일한 조합(F_oclr8_L + score > 0)과 그 짝(같은 배율 foveation)을 RESC_OCLR / RESC_CPFR로 고정했다. 판정에는 M7 pilot 규칙을 그대로 적용했다.`,
  },
  "glossary.stage1.falsification": {
    title: "반증 시험 (paired falsification tests)",
    body: `**무엇:** 각 계열이 기대는 가설을 **같은 frame 짝 비교**로 반박할 수 있게 사전 등록한 시험이다. 차이(Δ)와 영상 군집 95% CI를 낸다.

| 시험 | 가설 | 결과 (2026-10-14 기준) |
|---|---|---|
| B_att − A_cos | attention의 탐색 이득 | Δ 0.000 → 기각 |
| H_s8 − A_cos | 표현 점유율(고해상도 특징) | Δ −0.153 → 기각(더 나쁨) |
| CPFR − OCLR (같은 배율, G/L) | 문맥 보존 foveation | Δ ≈ 0 → 지지 안 됨 |
| G_neg − B_att | 음성 점 prompt | −0.084 → 해로움 |
| D_ref − B_att | frame-0 메모리만 읽기 | −0.038 → 약간 나쁨 |
| C_flowcrop / E_beam1 − Lp | 운동 증거 | 0 이하 → 새 증거 없음 |

**읽는 법:** 같은 frame에서 짝을 지었으므로 영상·frame 난이도 차이가 상쇄된다. CI가 0을 포함하면 '차이 없음'이다.

**주의:** 반증 시험은 비저장 probe 기준(한 frame의 후보 품질)이다. 인과 rollout의 결과와 다를 수 있다. 실제로 Stage 1에서 운동 생성기는 실패했지만, 운동을 **crop 제어**에 쓴 FLOWVEL은 인과 단계에서 PROMISING이었다.`,
  },

  // ───────────────────────────── 인과 실행
  "glossary.causal.rollout_vs_replay": {
    title: "인과 rollout vs 재생(replay)",
    body: `**재생 (replay):** 이미 기록된 SGGL 궤적 위에서 **출력만** 바꿔 지표를 다시 계산한다. CPU로 빠르게 할 수 있다. 그러나 바뀐 출력이 crop 중심·메모리를 바꾸는 효과(되먹임)를 반영하지 못한다. Stage 1 순 이득, M7 S/F 계열이 재생이다.

**인과 rollout (causal):** 영상 전체를 처음부터 다시 돈다. frame t의 출력 → 다음 crop 중심 → frame t+1의 L 입력 → … 이 이어진다. 메모리 기록(admit)도 실제로 일어난다. GPU 비용이 크다. Stage 2/3 arm이 인과다.

**왜 둘이 다른가 (예, 2026-10-14 기준):**
- F_oclr8_L + score > 0: 재생 순 이득 +14.3 → 인과 RESC_OCLR dJF −0.0001. 회수한 다음 frame에서 다시 놓치는 경우가 많았다.
- 운동 생성기: Stage 1(재생·probe) 실패 → FLOWVEL(인과 crop 제어) PROMISING.

**규칙 (M7부터):** 출력을 바꾸는 후보는 인과 rollout으로 확인해야만 판정 대상이 된다.`,
  },
  "glossary.causal.machine_matched": {
    title: "같은 기계 기준선 (machine-matched baseline)",
    body: `**규칙:** arm의 각 unit은 **같은 기계에서 돈** SGGL 결과와만 비교한다.
- RunPod RTX 4090 arm ↔ \`outputs/milestone8/runpod/runs/SGGL/\`
- Ubuntu RTX 2080 arm ↔ Ubuntu diag 궤적 \`outputs/milestone8/diag/*.csv\` (diag 실행이 SGGL 궤적과 지표를 함께 기록한다)
- unit → 기계 배정(shard)은 모든 arm에서 같다.

**왜:** GPU 종류(2080/4090/5090)마다 부동소수점 결과가 미세하게 다르다. SAM 2 rollout은 되먹임이 있어 작은 차이가 frame이 지나며 커질 수 있다. 다른 기계의 기준선과 비교하면 '방법의 효과'와 '기계 차이'가 섞인다.

**같게 맞춘 설정:** \`fill_hole_area = 0\`, FP32, TF32 끔. M7에서 Ubuntu diag와 Vast SGGL의 unit 평균 차이는 ≤ 2.4e-4(5 unit)였다(\`COMPUTE_AND_PERFORMANCE_AUDIT.md\`).

**대시보드에서:** \`causal_per_unit.csv\`의 \`machine\` 열이 짝지은 기계, \`causal_report.json\`의 \`machines\`가 arm별 기계 분포다.`,
  },
  "glossary.causal.arm": {
    title: "arm (인과 실험 조건)과 arms.json",
    body: `**정의:** 인과 rollout으로 실행하는 하나의 방법 설정. 모든 arm은 **첫 출력 전에** \`outputs/milestone8/arms.json\`에 고정된다(\`scripts/m8.py: freeze_arm\`). 이미 고정된 arm을 다른 내용으로 바꾸려 하면 거부하고, 출력이 있는 arm은 새로 고정할 수 없다.

**spec 필드:**
- \`gens\`: coabs frame에서 돌릴 생성기 목록(비어 있으면 rescue 없음)
- \`ranker\`: 후보 선택 규칙(예: \`{"kind": "score", "thr": 0.0}\` = score > 0이면 채택, \`oracle\` = GT 선택)
- \`center\`: crop 중심 정책(\`out\` = SGGL, \`m3vel\`, \`flowvel\`, \`fv_present\`, \`flowcarry\`, \`camcarry\`)
- \`admit\`: rescue를 메모리에 기록할지
- \`why\`: 고정 이유, \`frozen_kst\`: 고정 시각(KST)

**arm 종류:**
- 운동: FLOWVEL, M3vel
- rescue: RESC_OCLR, RESC_OCLR_ADMIT, RESC_CPFR
- 상한: ORACLE_POOL, ORACLE_POOL_ADMIT
- 결합(Stage 3, **K_***): K_OCLR_M3, K_OCLR_FLOWVEL, K_ADMIT_FLOWVEL(FLOWVEL·RESC_OCLR_ADMIT 결과를 본 뒤 고정 — 사후 선택으로 공개됨)
- FLOWVEL ablation: FV_PRESENT, FLOWCARRY, CAMCARRY
- SGGL: 각 기계의 기준선

**결과 위치:** \`outputs/milestone8/{runs,runpod/runs}/<arm>/<unit>.csv\`.`,
  },
  "glossary.causal.oracle": {
    title: "ORACLE (GT 선택 상한, 배포 불가)",
    body: `**정의 (\`ORACLE_POOL\`):** 동시 부재 frame에서 31개 후보 probe를 모두 돌리고, **GT로 채점해** J가 가장 높고 J > 0.5인 후보를 출력한다(없으면 빈 출력 유지). 출력 후 crop을 그 후보로 옮긴다. \`ORACLE_POOL_ADMIT\`는 그 후보를 메모리에도 기록한다.

**의미:** '순위화와 존재 판정이 완벽하다면 rescue로 얼마까지 얻을 수 있나'의 **인과 상한**이다. 미래 GT를 쓰므로 **배포 불가**이고 판정 대상이 아니다.

**예 (2026-10-14 기준):** ORACLE_POOL +0.0136 [+0.0020, +0.0297], 영상 5↑ / 0↓, 추가 SAM 2 호출 ≈3.1 / frame. FLOWVEL(+0.0130)이 순위화 없이, SAM 2 추가 호출 없이 이 상한과 같은 크기를 냈다.

**해석 함정:** ORACLE도 '후보 풀에 맞는 후보가 있을 때'만 회수한다. 풀의 recall 상한(0.511)과 되먹임 때문에 동시 부재 상한(+0.028)에는 못 미친다.`,
  },
  "glossary.causal.admit": {
    title: "admit (메모리 기록, memory admission)",
    body: `**정의:** rescue로 채택한 후보 마스크를 그 후보를 만든 상태(예: F_oclr8_L이면 L)의 **메모리 뱅크에 기록**하는 것. arm spec의 \`"admit": true\`. 기록하지 않으면 출력만 바뀌고 메모리는 SGGL과 같다.

**기대 효과:** 회수한 대상이 메모리에 들어가 다음 frame에서도 계속 추적된다(재생에는 없는 장기 효과).

**위험:** 잘못된 rescue(방해물)가 메모리에 들어가면 이후 frame이 그 방해물을 '대상'으로 따라간다 = **신원 전환**.

**예 (2026-10-14 기준):** RESC_OCLR_ADMIT − RESC_OCLR = +0.0047(장기 효과 있음). 그러나 miami-surf obj4에서 −0.116 신원 전환이 났다.

**관련 선행연구:** DAM4SAM, SENTRY 등 '검증 후 기록' 계열이 정확히 이 위험을 겨냥한다. M7 M2(메모리 허용 분리)는 인과 근거 부족으로 구현하지 않았다.`,
  },
  "glossary.causal.recentre": {
    title: "re-centre (crop 재배치)",
    body: `**정의:** rescue 후보를 채택해 출력하면, 다음 frame의 L crop 중심을 **그 후보 마스크의 중심**으로 옮기는 것. RESC_* arm의 설명 'output + re-centre'가 이것이다.

**왜:** SGGL은 출력이 비면 crop 중심을 그대로 둔다. 대상이 crop 밖에 있어 놓친 경우, 회수 후 crop을 옮기지 않으면 다음 frame에서 L은 여전히 대상을 못 본다.

**한계 (2026-10-14 기준):** RESC_OCLR은 re-centre를 해도 회수 후 다음 frame에서 다시 놓치는 일이 많았다(11 frame 회수, 거짓 재획득 +6, dJF −0.0001). 사후에 한 번 옮기는 것보다, FLOWVEL처럼 **놓치기 전·놓친 동안 계속** 대상 쪽으로 crop을 옮기는 편이 효과가 컸다.

**crop 중심 정책 일반:** L, FLOWVEL, M3vel 참조.`,
  },
  "glossary.causal.flowvel": {
    title: "FLOWVEL과 ablation (FV_PRESENT / FLOWCARRY / CAMCARRY)",
    body: `**FLOWVEL 정의 (\`arms.json\`, \`scripts/m8.py\` \`center == "flowvel"\`):**
- c0 = 현재 출력 마스크 중심. 출력이 비면 **광류로 운반한 대상 box의 중심**.
- 다음 crop 중심 = c0 + 0.5 × (대상 영역의 RAFT 광류 변위). 0.5는 감쇠 계수(damp).
- rescue·순위화 없음, SAM 2 추가 호출 없음. frame마다 RAFT-small 1회(긴 변 640).

**M3vel과의 차이:** 마스크 중심 차분 대신 dense flow를 쓴다(극소 마스크의 중심 흔들림에 덜 민감). 그리고 **출력이 빈 동안에도** crop을 움직인다.

**결과 (2026-10-14 기준, \`STAGE2_CAUSAL_RESULTS.md\`):** +0.0130 [+0.0010, +0.0285], 영상 5↑/2↓, LOVO +0.0067, 최악 −0.026, 거짓 재획득 93 = 93 → **PROMISING**(현재 유일한 배포 가능 PROMISING arm). GT 있는 동시 부재 frame을 131 → 63으로 줄였다.

**ablation (FLOWVEL PROMISING 후, 출력 전 고정):**

| arm | 남긴 요소 |
|---|---|
| FV_PRESENT | 광류 속도를 **출력이 있을 때만** 사용. 부재 중에는 crop 고정(SGGL처럼) |
| FLOWCARRY | 속도 없음. 부재 중 광류 운반 box만 따라감 |
| CAMCARRY | 부재 중 **카메라 운동**(대상 box 밖 median flow)만큼만 이동 |

어느 요소가 이득을 만드는지 분리한다. 결과는 Causal arms 탭과 \`NON_ATTENTION_MECHANISM_COMPARISON.md\` §4.

**주의:** 탐색적 결과다. VK에서는 +0.0002(이득 없음, crop 이탈 사건이 거의 없음).`,
  },
  "glossary.causal.m3vel": {
    title: "M3vel (M3-vel, M7 도전자)",
    body: `**정의 (\`scripts/m8.py: m3_center\`):**
- c = 현재 출력 마스크 box 중심, p = 직전 출력 중심.
- 다음 crop 중심 = c + 0.5 × (c − p). 감쇠된 등속 예측이다.
- 출력이 비면 예측하지 않는다(중심 유지, 이전 중심 기록 초기화).

**역사:** M7의 가장 강한 도전자였다. M7 결과 +0.0046 [−0.0001, +0.0115], 이득이 2개 영상(skate-park, sheep)에 몰려 INCONCLUSIVE / INSUFFICIENT_EVIDENCE. 이득 형태는 '대상이 다시 나타날 때 10–18 frame 빠른 재획득'.

**M8에서:** 출력 전에 고정한 **같은 기계 대조군**이다. RunPod에서도 +0.0046으로 Vast와 같은 값이 재현됐다(2026-10-14 기준). 여전히 INCONCLUSIVE.

**FLOWVEL과 비교하는 이유:** 둘 다 crop 중심 제어다. M3vel은 출력 중심만 쓰고 부재 중에는 멈추며, FLOWVEL은 광류를 쓰고 부재 중에도 움직인다. 이 차이가 동시 부재 감소(M3vel 108 vs FLOWVEL 63)로 나타났다.`,
  },

  // ───────────────────────────── 후보 계열
  "glossary.family.families": {
    title: "후보 계열 A–I",
    body: `M8 포트폴리오(\`EXPANDED_CANDIDATE_PORTFOLIO.md\`, 33개 후보)를 메커니즘으로 나눈 9개 계열이다.

| 계열 | 이름 | 핵심 아이디어 | 대표 생성기·arm |
|---|---|---|---|
| **A** | 전역 탐색 | frame-0 prototype과 현재 frame 전체의 코사인 heatmap → peak에 point prompt | \`A_cos\`, \`A_cos_crop\` |
| **B** | attention | 대조 token attention(fg 8 + bg 32 token) heatmap, set-attention 순위화기, 메모리 판독 | \`B_att\`, \`B_att_crop\`, R3, \`D_ref\` |
| **C** | 운동 | RAFT 광류로 대상 box를 운반해 crop/prompt에 사용 | \`C_flowcrop\`, \`C_flowbox\`, **FLOWVEL** |
| **D** | 메모리 | 보호 메모리, 메모리 기록(admit) | \`Gp\`, \`Lp\`, \`*_ADMIT\` |
| **E** | 다중 가설 | 속도 외삽 crop beam, 지연 확정 | \`E_beam1/2\`, M3vel(C/E) |
| **F** | 척도·시야 | 대상이 N cell을 차지하도록 배율을 맞춘 crop(OCLR) vs 같은 배율 foveation(CPFR) | \`F_oclr*\`, \`F_cpfr*\`, RESC_* |
| **G** | prompt | 방해물 위치에 음성 점 | \`G_neg\` |
| **H** | 표현 | stride-8 특징, 다른 사전학습 특징 | \`H_s8\` |
| **I** | 구조 | 형태·운동 사전을 순위 특징으로, 마스크 결합 | ranker 특징 \`dist_flow\` |

**역할 구분:** ① = 이미 위치가 맞는 대상의 분할 개선, ② = 두 view가 모두 실패할 때 새 후보 생성. M8의 주 대상은 ②다.

**2026-10-14 기준 요약:** Stage 1 진출 계열 없음. 인과에서는 **C(운동, crop 제어)**만 PROMISING. F는 REJECTED/INCONCLUSIVE.`,
  },
  "glossary.family.raft": {
    title: "RAFT 광류 (optical flow)",
    body: `**무엇:** 연속 두 frame 사이 모든 화소의 이동 벡터(dense optical flow)를 추정하는 신경망이다. torchvision 0.29.1의 **RAFT-small**(가중치 4 MB, BSD, FlyingChairs+Things로 학습 — VOS 데이터와 겹치지 않음)을 쓴다.

**사용 방식 (\`src/spatial/motion.py\`):** 긴 변 640으로 줄인 frame 쌍에 frame마다 1회. 결과로
- 대상 영역의 평균 변위 → FLOWVEL·FV_PRESENT의 속도
- 광류로 대상 box를 frame마다 운반 → 부재 중 crop 중심(FLOWVEL, FLOWCARRY), C 계열 생성기
- 대상 box 밖 median flow = 카메라 운동 → CAMCARRY

**비용:** RTX 2080 0.028–0.047 s/frame. SAM 2 1 frame(≈0.45 s)보다 훨씬 작다.

**장점:** 외형과 무관한 정보원이다. SAM 2가 대상을 '못 보는' frame에도 화소는 움직인다.

**한계 (2026-10-14 기준):** 대상이 오래 사라지면 운반 box가 배경 흐름을 따라 표류한다. GT 있는 동시 부재 frame에서 광류 box가 GT와 겹친 비율은 19%뿐이었다(shard 0). 그래서 광류 box를 **직접 prompt로** 쓴 C_flowbox는 실패했고, crop을 **대략 그쪽으로 옮기는** 데 쓴 FLOWVEL은 성공했다.`,
  },
  "glossary.family.occupancy": {
    title: "occupancy (cell 점유) / OCLR",
    body: `**가설:** 작은 대상 실패는 대상이 stride-16 특징 격자에서 1–2 cell만 차지해 SAM 2가 표현하지 못하기 때문이다. 대상이 N cell을 차지하도록 확대하면 다시 찾을 수 있다.

**OCLR (occupancy crop, \`F_oclr{N}_{G|L}\`):**
- 배율 α = N × 16 / (대상 긴 변 추정 px) (\`rescue.occupancy_alpha\`)
- 그 배율의 평범한 정사각 crop(주변 문맥 잘림)을 광류 운반 중심에 놓고 G 또는 L 메모리로 probe.
- N = 8 또는 4.

**저장소 사실과의 충돌 (\`REPORT_INGESTION_AND_REPO_AUDIT.md\`):** L crop에서 대상은 이미 ≈16 cell이다. 그래서 L 메모리로 8-cell crop을 쓰면 실제로는 **축소**(시야 확대)다. 또한 GT 있는 동시 부재 frame의 절반은 대상이 crop 밖이었다. L에게는 occupancy보다 **위치** 문제가 컸다.

**결과 (2026-10-14 기준):** \`F_oclr8_L\`은 recall 0.176, 원시 FP 0.014(매우 보수적). 인과 RESC_OCLR −0.0001(REJECTED). G 메모리 변형은 확대가 메모리 좌표계와 어긋나 거의 회수하지 못했다.`,
  },
  "glossary.family.foveation": {
    title: "foveation / CPFR (문맥 보존 확대)",
    body: `**무엇:** 사람 눈의 중심와(fovea)처럼, 관심 영역은 확대하고 주변은 압축해 **전체 문맥을 한 view에 유지**하는 비균일 샘플링이다.

**CPFR 구현 (\`F_cpfr{N}_{G|L}\`, \`rescue.axis_map(fovea=True)\`):**
- 축(가로·세로)마다 분리된 1-D 사상. 중심 주변 창을 배율 α로 확대해 view의 절반(fovea)에 놓고, 나머지 영역은 양옆에 압축해 넣는다.
- 출력 logit은 역변환으로 frame 좌표에 되돌린다.
- **같은 배율의 OCLR과 짝**을 지어, '문맥 보존' 하나만 다르게 한다.

**가설:** 문맥이 보이면 대상을 더 잘 찾는다(첨부 보고서의 1순위 FORGE-SGGL 아이디어).

**결과 (2026-10-14 기준):**
- Stage 1: CPFR − OCLR recall 차이 ≈ 0(CI가 0 포함). 회수 이득 없음.
- 원시 FP 0.110 vs 0.014: 주변의 비슷한 방해물(miami-surf 서퍼들)을 '존재'로 잡음.
- 인과 RESC_CPFR: −0.0070, 거짓 재획득 118 vs 93, disc-jockey obj3 −0.330 신원 전환 → **REJECTED**. 문맥 보존 가설은 반증 방향이다.

**선행연구:** Learning to Zoom(ECCV'18), FOVEA(ICCV'21), LZU(CVPR'23).`,
  },

  // ───────────────────────────── 판정·사전 등록
  "glossary.verdict.labels": {
    title: "판정 라벨: PROMISING / INCONCLUSIVE / REJECTED / STRONG_WIN",
    body: `M7 pilot 규칙을 그대로 쓴다(\`preregistration.json\` \`stage2_causal_rule\`, 코드 \`m8_analysis._pilot\`). 모두 small unit 기준이다.

| 라벨 | 조건 |
|---|---|
| **PROMISING** | dJF ≥ +0.005 **그리고** +0.01 이상 오른 영상 ≥ 3 **그리고** LOVO 최솟값 > 0 **그리고** miami-surf 제외 > 0 **그리고** 최악 unit ≥ −0.05 |
| **REJECTED** | dJF ≤ 0, **또는** (오른 영상 ≤ 2 **그리고** LOVO ≤ 0) |
| **INCONCLUSIVE** | 위 둘 다 아님 |
| **STRONG_WIN** | dJF ≥ +0.03 **그리고** CI 하한 > 0 (다른 라벨을 덮어씀) |

**STRONG_WIN이 사실상 불가능한 이유:** 동시 부재 frame을 전부 완벽히 고쳐도 객체 평균 +0.028이다. rescue만으로는 +0.03에 닿을 수 없음을 **사전에 공개**했다.

**읽는 법:**
- PROMISING = '사전 기준을 넘었으니 새 데이터로 확인할 가치가 있음'. 확정 승리가 아니다(다중 비교).
- INCONCLUSIVE = 평균은 양수일 수 있지만 영상 수·견고성이 부족함(예: M3vel 영상 2개).
- PROMISING 규칙에는 CI 조건이 없다. CI 하한 > 0은 별도로 확인한다.
- ORACLE arm에는 라벨이 붙어도 배포 불가라 판정에 쓰지 않는다.
- M7에서는 F1, M1처럼 pilot 규칙상 INCONCLUSIVE여도 실질 효과가 0 이하이면 계열 판정을 REJECTED로 덮어쓴 경우가 있다(근거 문서 공개).`,
  },
  "glossary.verdict.prereg": {
    title: "사전 등록과 수정 (pre-registration / amendments)",
    body: `**사전 등록:** 실험 출력을 보기 **전에** 데이터 역할, 지표, 진출·판정 규칙, 반증 시험을 파일로 고정한다. 결과를 본 뒤 규칙을 바꿔 '이긴 것처럼' 만드는 일을 막는다.

**M8 파일 (\`outputs/milestone8/\`):**

| 파일 | 내용 |
|---|---|
| \`preregistration.json\` | 본 사전 등록 (\`scripts/m8.py: PREREG\`) |
| \`preregistration_amend1.json\` | **수정 1:** F view에 점 prompt 변형(\`*_pt\`) 추가. 단위 1개를 본 뒤 결정 |
| \`preregistration_amend2.json\` | **수정 2:** H_s8 특징 버그 수정. shard 0 표를 본 뒤 결정 |
| \`preregistration_vk.json\` | VK 결선 실행 사전 등록 |
| \`arms.json\` | 인과 arm spec. arm마다 첫 출력 전 고정(\`frozen_kst\`) |

**공개된 기타 일탈 (\`STAGE1_DIAGNOSTIC_RESULTS.md\`, \`STAGE2_CAUSAL_RESULTS.md\`):** DEV→REP 교차 적합 추가, threshold 규칙의 '순 이득 최대' 보충, 진출 계열이 없어 '진단적 인과 실행' 지정, K_ADMIT_FLOWVEL의 사후 선택.

**규칙:** 고정 파일은 덮어쓰지 않는다. arm마다 사전 등록된 수정 1회까지 허용. 일탈은 모두 문서에 공개한다(\`AGENTS.md\` §2).`,
  },
  "glossary.verdict.keep_sggl": {
    title: "KEEP_SGGL (M7 최종 결정)",
    body: `**무엇:** M7 최종 결정(\`docs/m7/FINAL_ALGORITHM_DECISION.md\`). 18개 후보 계열, 33개 prototype을 시험했지만 사전 등록 최종 규칙(DEV+REP 영상 CI 하한 > 0)으로 SGGL을 이긴 방법이 없었다. 그래서 SGGL을 주 알고리즘으로 유지했다.

**M3-vel:** INSUFFICIENT_EVIDENCE. CI 하한이 −0.0001로 0에 걸렸고, 이득이 2개 영상에 몰렸다. '다음 확인 단계의 지정 도전자'로 남았다.

**M7의 교훈 (M8 설계로 이어짐):**
- 선택(G vs L)은 거의 포화(self gate = 선택 상한의 89%).
- 마스크 결합·attention 보정(①)의 상한은 +0.002–0.006.
- 남은 큰 손실은 **두 view가 모두 실패하는 frame**이다 → M8은 후보 생성(②)과 crop 제어로 이동.

**M8에서의 위치 (2026-10-14 기준):** 결정은 유지된다. FLOWVEL은 PROMISING이지만 탐색적이며 새 데이터 확인 전이다. 다음 단계는 새 tiny 데이터 다운로드 승인이다.`,
  },

  // ───────────────────────────── 운영
  "glossary.ops.ubuntu_truth": {
    title: "Ubuntu = 유일한 원본 (source of truth)",
    body: `**규칙 (\`AGENTS.md\` §1, 최우선):** 로컬 Ubuntu 서버(\`/home/sojak/sam2\`, RTX 2080 8 GB)가 연구 결과의 **유일한 공식 저장소**다.
- 결과 CSV/JSON/NPZ, 그림, 사전 등록, 설정, 실행 출처(커밋·환경), 보고서는 모두 Ubuntu에 영구 보관한다.
- 클라우드 GPU(RunPod, Vast)는 계산 worker일 뿐이며 디스크·network volume은 영구 저장소가 아니다.
- 분석, 보고서, 커밋은 Ubuntu에서 한다.
- '저장됐다'는 말은 sync ledger에 기록된 뒤에만 한다.
- 연구는 Ubuntu 파일만으로 이어갈 수 있어야 한다.

**대시보드와의 관계:** 이 사이트는 Ubuntu 저장소를 **읽기 전용**으로 보여준다. worker에서 아직 당겨오지 않은 결과는 여기 보이지 않는다.

**Ubuntu도 실험 기계다:** 2080에서 돈 Stage 1 diag(53 unit)와 일부 arm은 Ubuntu 결과이며, 같은 기계 기준선은 Ubuntu diag 궤적이다.`,
  },
  "glossary.ops.runpod": {
    title: "RunPod worker (RTX 4090)",
    body: `**무엇 (2026-10-14 기준, \`docs/GPU_WORKER_OPERATIONS.md\`):** M8의 GPU worker. RunPod RTX 4090 24 GB, \`/workspace\`(network volume)에 저장소 사본과 venv가 있다. 작업은 pod의 tmux 세션 \`m8\`에서 돈다.

**하는 일:** 인과 arm rollout, 일부 diag(11 unit), RunPod SGGL 기준선(64 unit). 결과는 \`outputs/milestone8/runpod/<실험>/\`에 쓴다.

**절차:**
1. \`gpu_sync.py push-code\`로 코드·사전 등록·arms.json 전송(\`CODE_SYNC.json\`에 Ubuntu HEAD 기록).
2. 필요한 데이터만 전송하고 md5로 확인(PUMaVOS C는 절대 보내지 않음).
3. Ubuntu tmux \`m8:sync\`가 300초마다 결과를 당겨옴.
4. 종료 전 audit이 exit 0이어야 함.

**Vast RTX 5090:** M7의 worker. M8 시작 시 연결이 거부되어 M8에서는 쓰지 않았다.

**기계 차이:** RunPod 결과는 RunPod SGGL과만 비교한다(같은 기계 기준선).`,
  },
  "glossary.ops.ledger": {
    title: "sync ledger 상태와 gpu_sync audit",
    body: `**sync 방식 (\`scripts/gpu_sync.py\`):** Ubuntu가 당겨온다(worker는 밀지 않음). 디렉터리마다
1. worker가 모든 파일의 (경로, 크기, sha256) manifest 출력
2. tar stream으로 Ubuntu staging에 수신
3. 파일마다 크기·sha256 확인(실패 시 3회 재시도, backoff)
4. **덮어쓰지 않고** 병합
5. 확인된 파일을 \`outputs/SYNC_LEDGER.tsv\`에 기록하고, 확인 목록을 worker의 \`<dir>/.ubuntu_verified.tsv\`로 돌려보냄

**ledger \`status\` 값:**

| 상태 | 의미 |
|---|---|
| **NEW** | Ubuntu에 없던 파일. 확인 후 옮겨 넣음 |
| **SAME** | Ubuntu에 같은 내용(sha256 동일)이 이미 있음. 건너뜀 |
| **APPENDED** | Ubuntu 파일이 새 파일의 앞부분과 같음(로그처럼 덧붙여 커짐). Ubuntu 바이트를 접두로 유지하며 갱신 |
| **CONFLICT** | 같은 경로에 내용이 다름. Ubuntu 사본은 그대로 두고 worker 판을 \`<name>.conflict-<ts>\`로 저장해 보고 |

**\`SYNC_STATUS.json\`:** 디렉터리별 \`last_ok\`, \`last_new\`, \`conflicts\`, \`last_fail\`(실패 사유). 실패는 성공으로 기록되지 않는다. \`last_fail\`이 \`last_ok\`보다 오래됐으면 이후 복구된 것이다.

**audit (\`gpu_sync.py audit <dirs>\`):** worker의 모든 파일이 Ubuntu에서 확인됐는지 검사. 빠진 파일이 있으면 목록과 함께 exit 1, 모두 확인되면 exit 0. **worker를 멈추거나 지우기 전 exit 0이 필수**이고, \`.ubuntu_verified.tsv\`에 없는 worker 결과는 지우지 않는다.`,
  },

  // ───────────────────────────── UI 도움말 (용어집 목록에는 나오지 않음)
  "glossary.ui.search": {
    title: "전체 검색 사용법",
    body: `**무엇을 찾나:** 저장소에서 대시보드가 허용한 경로의 **파일 경로(이름)**만 찾는다. **파일 내용은 검색하지 않는다.** 용어 뜻은 '용어집' 탭의 검색을 쓴다.

**규칙 (\`dashboard/server.py: search\`):**
- 대소문자 구분 없는 **부분 문자열** 일치.
- 공백으로 나눈 단어는 **모두** 경로에 있어야 한다(AND). 예: \`causal csv\` → \`causal_per_unit.csv\`; \`FLOWVEL drone\` → FLOWVEL arm의 drone unit CSV.
- 2글자 이상부터, 입력을 멈춘 뒤 0.2초 후 검색한다.
- 정렬: \`.md\` 문서 먼저, 그다음 경로가 짧은 순. 상위 200개만 보이고 전체 일치 수를 함께 표시한다.
- 파일 목록은 최대 1분마다 다시 훑는다. 방금 생긴 파일은 최대 60초 늦게 나타날 수 있다.

**결과 클릭:** 앞의 라벨대로 탭이 열린다. \`문서\`(.md) → 리포트, \`코드\`(.py/.sh 또는 scripts/src/tests 아래) → 코드, 나머지 → 원시 데이터.

**검색 범위:** \`docs/m7\`, \`docs/m8\`, \`docs/GPU_WORKER_OPERATIONS.md\`, \`AGENTS.md\`, \`outputs/milestone7\`, \`outputs/milestone8\`, \`outputs/SYNC_LEDGER.tsv\`, \`outputs/SYNC_STATUS.json\`, \`scripts\`, \`src/spatial\`, \`tests\`. 숨김 경로는 원시 데이터 탭의 숨김 규칙과 같다.

**유용한 검색어:** \`arms.json\`, \`causal_report\`, \`stage1_generators\`, \`preregistration\`, \`TASKS\`, \`gpu_sync.log\`, unit 이름 일부(예: \`miami-surf\`).`,
  },
  "glossary.ui.hidden_paths": {
    title: "보이는 경로와 숨겨진 경로",
    body: `이 사이트는 저장소를 **읽기 전용**으로, 허용 목록 안에서만 보여준다(\`dashboard/server.py\` \`ALLOW\`).

**허용 루트:** \`docs/m7\`, \`docs/m8\`, \`docs/GPU_WORKER_OPERATIONS.md\`, \`AGENTS.md\`, \`outputs/milestone7\`, \`outputs/milestone8\`, \`outputs/SYNC_LEDGER.tsv\`, \`outputs/SYNC_STATUS.json\`, \`scripts\`, \`src/spatial\`, \`tests\`.
- '원시 데이터' 탭: 코드 루트를 뺀 나머지.
- '코드' 탭: \`scripts\`, \`src/spatial\`, \`tests\`, \`AGENTS.md\`.

**허용 루트 안에서도 숨기는 것 (\`DENY_PARTS\`):** 경로 어느 부분이든 \`tcache\`, \`checkpoints\`, \`data\`, \`.ssh\`, \`__pycache__\`, \`.git\`이거나, 이름이 \`.env\`로 시작하거나 끝나면 목록·검색·열기 모두 불가.

**왜 숨기나:**
- **비밀:** \`.gpu_worker.env\`(worker 주소·키), SSH 키, 토큰. 이 사이트는 외부(Cloudflare Tunnel)로도 열린다.
- **봉인 데이터 보호:** 원본 데이터셋 경로(\`data\`)에는 봉인된 PUMaVOS C가 있다.
- **크기:** \`tcache\`(M7 7.5 GB 특징 캐시), 체크포인트는 볼 이유가 없고 무겁다.

**그 밖의 제한:**
- 심볼릭 링크는 실제 경로로 풀어 허용 루트 안인지 다시 확인한다.
- \`.npz/.npy/.pt\`는 목록에 크기만 나오고 내용은 제공하지 않는다.
- 8 MiB가 넘는 텍스트 파일은 통째로 열 수 없고 끝부분(tail)만 볼 수 있다.

**원하는 파일이 안 보이면:** worker에서 아직 sync되지 않았거나(\`SYNC_STATUS.json\` 확인), 위 규칙에 걸린 것이다.`,
  },
  "glossary.ui.file_csv": {
    title: "CSV / TSV 보기",
    body: `**표시 방식:**
- 서버가 파일 전체를 읽고, 2000행씩 페이지로 보낸다('이전'/'다음').
- **열 머리글 클릭 = 정렬.** 현재 페이지 2000행 안에서만 정렬한다(전체 정렬 아님). 두 값이 모두 숫자면 수치 비교, 아니면 문자열 비교.
- 소수점 5자리 이상인 숫자는 보기 좋게 줄여 표시한다. 칸에 마우스를 올리면 원래 값이 보인다.
- 30초마다 다시 읽는다(실행 중 커지는 CSV도 따라감).

**'열 요약' (펼치기):** **전체 행** 기준이다(현재 페이지가 아님).
- \`n\`: 빈 칸이 아닌 값의 수
- 모든 값이 숫자로 읽히면 \`mean\`, \`min\`, \`max\`
- 하나라도 숫자가 아니면 \`unique\`(서로 다른 값의 수)만
- \`True/False\` 열은 문자열이라 unique = 2로 나온다

**자주 보는 CSV:**
- \`causal_per_unit.csv\`: arm × unit 한 행. \`d\`/\`dJ\`/\`dF\` = 같은 기계 SGGL 대비 차이, \`small\`/\`tiny\`, \`coabs_n\`, \`fp_arm\`/\`fp_base\`(거짓 재획득), \`rescued\`, \`lat\`/\`lat_base\`, \`calls\`, \`vram\`, \`machine\`.
- \`stage1_generators.csv\`: 생성기별 \`recall\`, \`mean_best_J\`, \`videos_hit\`, \`fp_rate_gt_empty\`, \`adv_recall\`.
- \`runs/<arm>/<unit>.csv\`: frame 한 행. \`score_G\`, \`score_L\`, \`acc\`, \`gt_empty\`, \`J\`, \`F\`, \`JF\`, \`JF_G\`, \`rescued\`, \`gen\`.
- \`diag/<unit>.cand.csv\`: Stage 1 후보 한 행.
- \`SYNC_LEDGER.tsv\`: sync 기록(상태 열 → 용어집 'sync ledger').

**주의:** \`d\`의 mean은 **모든 unit**(small 아닌 것 포함, 여러 arm 섞임) 평균일 수 있다. 판정 값은 \`causal_report.json\`을 본다.`,
  },
  "glossary.ui.file_json": {
    title: "JSON 보기",
    body: `**표시 방식:** 접을 수 있는 트리다.
- 최상위 한 단계만 펼쳐진 채로 시작한다. ▸를 눌러 펼친다.
- 접힌 노드 옆 \`{n}\`은 key 수, \`[n]\`은 원소 수다.
- 스칼라 12개 이하 배열(예: CI \`[하한, 상한]\`)은 한 줄로 보인다.
- 파란색 = key, 초록 = 문자열, 갈색 = 숫자·불리언·null.
- 30초마다 다시 읽는다.

**자주 보는 JSON (\`outputs/milestone8/\`):**
- \`causal_report.json\`: arm별 판정 요약 — \`dJF\`, \`CI_video\`, \`videos_up_0.01\`, \`lovo_min\`, \`without_miami\`, \`worst_unit\`, \`verdict\`, \`per_video\`, \`machines\`. VK는 \`<arm>@VK\` key.
- \`arms.json\`: 고정된 arm spec과 \`frozen_kst\`.
- \`stage1_report.json\`: trigger 수, recall, 반증 시험(\`tests\`), 지연 시간.
- \`rankers_*.json\`: 순위화기별 \`fitREP\`/\`fitDEV\`의 threshold와 held-out 평가, \`pooled_heldout\`.
- \`preregistration*.json\`: 사전 등록과 수정.
- \`TASKS.json\`: 작업 진행 현황(agent가 갱신).

**'원본' 링크:** 브라우저에서 원래 JSON 텍스트를 연다.`,
  },
  "glossary.ui.file_tail": {
    title: "로그·텍스트 보기 (tail)",
    body: `CSV/JSON/MD/이미지/바이너리가 아닌 파일(\`.log\`, \`.txt\`, \`.sh\` 등)은 **끝부분만** 보여준다.

**규칙:**
- 기본 마지막 500줄. 버튼으로 200 / 500 / 2000 / 10000줄 선택.
- 서버는 파일의 **마지막 2 MiB**만 읽고 그 안에서 줄을 센다. 아주 긴 줄이 많으면 요청한 줄 수보다 적게 보일 수 있다.
- **15초마다 자동 갱신**(갱신 시각 표시). 실행 중인 로그를 지켜보는 용도다.

**자주 보는 로그 (\`outputs/milestone8/logs/\`, \`runpod/logs/\`):**
- \`run_<ARM>_<machine>.log\`: 인과 arm 실행. unit마다 진행이 찍힌다.
- \`diag_*.log\`: Stage 1 진단. \`*.aborted_h8bug.log\`는 수정 2 이전 중단된 실행이다.
- \`gpu_sync.log\`: Ubuntu sync 루프. \`GAVE UP\`, \`verification failed\`는 실패이며 \`SYNC_STATUS.json\`에도 남는다.

**주의:** RunPod 로그는 sync된 사본이다. 최대 5분(sync 주기) 늦을 수 있다.`,
  },
  "glossary.ui.file_binary": {
    title: "바이너리 파일 (.npz / .npy / .pt)",
    body: `\`.npz\`, \`.npy\`, \`.pt\`는 목록에 **크기만** 나오고 내용은 열 수 없다. 서버가 이 확장자의 파일 요청을 거부한다.

**왜:** 크기가 크고(특징 벡터, 학습 가중치), 브라우저에서 읽을 수 있는 형식이 아니다. \`.pt\`는 pickle 기반이라 안전하게 다루기 어렵다.

**이 프로젝트의 예:**
- \`diag/<unit>.vec.npy\`, \`vec1b.npy\`: Stage 1 후보의 특징 벡터(순위화기 R3 입력).
- \`diag/<unit>.refs.pt\`: frame-0 참조 token(fg/bg).
- \`rankers/*.pt\`: R3 set-attention 순위화기 가중치.

내용이 필요하면 Ubuntu에서 \`uv run python\`으로 직접 연다.`,
  },
  "glossary.ui.file_view": {
    title: "파일 보기 방식 (확장자별)",
    body: `오른쪽 파일 보기는 확장자로 형식을 정한다.

| 확장자 | 보기 |
|---|---|
| \`.csv\`, \`.tsv\` | 표 + 전체 행 열 요약, 2000행 페이지, 페이지 내 정렬 |
| \`.json\` | 접을 수 있는 트리 |
| \`.md\` | 문서 렌더링. 문서 안의 상대 링크는 사이트 안 해당 탭으로 이동 |
| \`.png\`, \`.jpg\`, \`.gif\`, \`.webp\` | 이미지 |
| \`.npz\`, \`.npy\`, \`.pt\` | 크기만 (내용 제공 안 함) |
| 그 밖 (\`.log\`, \`.txt\`, …) | 마지막 N줄, 15초마다 갱신 |

**'원본' 링크:** 가공하지 않은 파일을 새 탭으로 연다. 8 MiB가 넘는 텍스트는 통째로 열 수 없다(tail만).

**주소:** 선택한 파일이 주소 \`#raw:<경로>\`에 남는다. 북마크와 뒤로 가기가 된다.`,
  },
  "glossary.ui.code": {
    title: "코드 탭",
    body: `\`scripts\`, \`src/spatial\`, \`tests\`, \`AGENTS.md\`를 **읽기 전용**으로, 줄 번호와 구문 강조와 함께 보여준다. 8 MiB가 넘는 파일은 열 수 없다.

**정의를 확인할 때 먼저 볼 파일:**

| 파일 | 내용 |
|---|---|
| \`scripts/m8.py\` | 사전 등록(PREREG), trigger, 생성기, diag, arm 고정(\`freeze_arm\`), 인과 rollout과 crop 중심 정책(\`m3vel\`, \`flowvel\` 등) |
| \`scripts/m8_analysis.py\` | Stage 1 recall/FP/반증 시험, 순위화기 교차 적합, \`net_dJ_frames\`, 인과 판정(\`_pilot\`), 같은 기계 기준선(\`_base_frames\`) |
| \`src/spatial/sggl.py\` | SGGL 두 상태 rollout, \`self_gate\`, \`a2x\`(SGGL), \`a2v\`(SGGL-V) |
| \`src/spatial/vcam.py\` | crop 변 규칙, 내부 경계 접촉(\`touches_border\`) |
| \`src/spatial/rescue.py\` | heatmap(A/B), OCLR/CPFR view 사상, occupancy 배율, 비저장 probe |
| \`src/spatial/motion.py\` | RAFT 광류, 대상 box 운반 |
| \`src/spatial/ranker.py\` | R1–R3 |
| \`scripts/gpu_sync.py\` | worker → Ubuntu sync, ledger, audit |
| \`tests/test_m8.py\` | 비저장 probe 비트 동일성, 비활성 rescue == SGGL 등 |
| \`AGENTS.md\` | 운영 규칙(원본 = Ubuntu, 봉인, 같은 기계 비교) |

이 사이트에서 코드를 고칠 수는 없다.`,
  },
  "glossary.ui.reading_order": {
    title: "M8 문서 읽는 순서",
    body: `왼쪽 목록은 \`docs/m8\` → \`docs/m7\` → 운영 문서 순으로, 각 폴더 안에서는 이름순이다. 아래 순서로 읽기를 권한다.

**1. 무엇을 왜 하는가**
1. \`m7/FINAL_ALGORITHM_DECISION.md\` — 출발점: KEEP_SGGL, SGGL 정의, M3-vel이 도전자인 이유
2. \`m8/REPORT_INGESTION_AND_REPO_AUDIT.md\` — 첨부 보고서 제안과 저장소 사실의 충돌, M8 방향 조정
3. \`m8/EXPERIMENT_TOURNAMENT_PLAN.md\` — 사전 등록 요약: 데이터, 단계, 진출·판정 규칙, 다중 비교

**2. 실패 분석과 후보**
4. \`m8/FAILURE_EPISODE_ANALYSIS.md\` — 동시 부재 486 frame의 구성, 위치형/표현형/가림형
5. \`m8/EXPANDED_CANDIDATE_PORTFOLIO.md\` — 33개 후보, 계열 A–I

**3. 결과**
6. \`m8/STAGE1_DIAGNOSTIC_RESULTS.md\` — 생성기 recall/FP, 반증 시험, 순위화기가 실패한 이유
7. \`m8/STAGE2_CAUSAL_RESULTS.md\` — 인과 arm 결과표, FLOWVEL 견고성 **(핵심)**
8. \`m8/NON_ATTENTION_MECHANISM_COMPARISON.md\` — 운동/척도/메모리 계열 비교, FLOWVEL ablation
9. \`m8/ATTENTION_MECHANISM_COMPARISON.md\` — attention 삽입 위치별 결과

**4. 필요할 때**
- \`m8/CROSS_FAMILY_COMBINATIONS.md\` — 결합 행렬(실험 전 작성)
- \`m8/COMPUTE_AND_PERFORMANCE_AUDIT.md\` — 비용, 최적화, 같은 기계 기준선
- \`m8/PRIOR_ART_AND_NOVELTY_AUDIT.md\` — 선행연구와 새로움 위험(\`PRIOR_ART_raw.md\`는 원자료)
- \`m8/PARALLEL_EXECUTION_PLAN.md\` — 작업 의존 그래프, 기계 배치
- \`GPU_WORKER_OPERATIONS.md\`, \`AGENTS.md\` — worker 운영과 규칙
- \`m7/*\` — M7 개별 track 문서와 검토

**주의:** 문서의 숫자는 작성 시점 값이다. 실험이 진행 중이면 최신 값은 개요·Causal arms 탭(원자료 \`causal_report.json\`)을 기준으로 한다. 일부 문서가 링크하는 \`COMBINATION_RESULTS.md\`, \`FINAL_PORTFOLIO_ASSESSMENT.md\`는 2026-10-14 기준 아직 없다.`,
  },
  "glossary.ui.figures": {
    title: "그림 탭",
    body: `\`outputs/milestone8/viz\`, \`outputs/milestone7/viz\`의 PNG를 모두 모아 보여준다. 30초마다 목록을 갱신해 새 그림이 생기면 나타난다.

**사용법:** 그림을 누르면 크게 열린다. ←/→ 버튼이나 키보드 화살표로 넘긴다. 제목은 파일명에서 번호와 \`_\`를 뺀 것이다.

**M8 그림 (\`scripts/m8_viz.py\`로 생성):**
- \`30_rescue_events\`: 동시 부재 frame 몇 개에서 SGGL 상태(G/L score), A/B heatmap(stride 16), OCLR8/CPFR8 view, 후보 마스크(A_cos#0, OCLR8_L, CPFR8_L), 오프라인 GT를 한 행에 나란히 보여준다.
- \`31_causal_timelines\`: drone obj1, kite-surf obj2, skate-park obj2, scooter-board obj2, lab-coat obj2의 frame별 J&F. 굵은 검정 = RunPod SGGL, 가는 선 = FLOWVEL·M3vel·RESC_OCLR, 점 = rescue frame, 회색 띠 = GT가 빈 구간. 예: drone obj1에서 FLOWVEL이 GT 공백 뒤 바로 재획득.

**M7 그림:** \`00–07\`은 실패·성공 사례 frame(miami-surf 거짓 재획득, drone crop 이탈 등), \`20\`은 M3-vel 시간선, \`21\`은 VK 예시. 설명은 \`docs/m7/FINAL_ALGORITHM_DECISION.md\` §4.

**주의:** 사례 그림은 고른 예시다. 평균 효과는 표와 CI로 판단한다.`,
  },
  "glossary.ui.glossary": {
    title: "용어집 사용법",
    body: `이 사이트에서 반복해서 나오는 용어를 묶음별로 모았다. 각 항목은 다른 탭의 '?' 설명과 같은 출처(\`src/help/glossary.ts\`)를 쓴다.

**검색:** 제목과 본문 모두에서 대소문자 구분 없이 찾는다. 공백으로 나눈 단어는 모두 들어 있어야 한다. 예: \`LOVO\`, \`동시 부재\`, \`machine\`, \`FP\`.

**숫자에 대해:** 결과 수치는 '예'로 들었으며 출처 문서와 '2026-10-14 기준'을 함께 적었다. 실험이 진행되면 값이 바뀔 수 있다. 최신 값은 개요·Causal arms·Stage 1 탭을 본다.`,
  },
}
