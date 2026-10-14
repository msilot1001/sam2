import type { HelpMap } from "./types"

// keys: "m7.<element>"
// 출처: docs/m7/*.md, outputs/milestone7/*.json, scripts/m7_replay.py, scripts/m7_reports.py, src/spatial/policies.py
// 수치 예시는 모두 "2026-10-14 기준" M7 문서 값이다. M7은 종료된 milestone이라 값이 바뀌지 않아야 하지만, 화면 값이 우선이다.

const CAVEAT_EXPLORATORY = `- DEV(DAVIS val + PUMaVOS A/B + SegTrack)와 REP(DAVIS train)는 M6부터 여러 번 본 **개발 데이터**다. 여기 나오는 모든 결과는 **탐색적**이다.
- 작은 객체 57개가 영상 34개에 몰려 있다. 효과가 한두 영상(특히 miami-surf, skate-park, drone)에 집중되기 쉽다.
- PUMaVOS C는 봉인되어 있다. M7에서 최종 규칙을 통과한 방법이 없어 열지 않았다.`

export const help: HelpMap = {
  // ------------------------------------------------------------------ page-level
  "m7.summary": {
    title: "M7 요약: 최종 결정 KEEP_SGGL, 그리고 M8로 이어진 이유",
    body: `## 무엇을 보여 주나
- M7(SGGL을 대체할 알고리즘 탐색 milestone)의 최종 결정과 근거 문서 목록이다.
- 버튼은 \`docs/m7/*.md\` 문서이고, 누르면 Reports 탭에서 연다. 먼저 읽을 것: \`FINAL_ALGORITHM_DECISION.md\` → \`OPPORTUNITY_AND_FAILURE_AUDIT.md\` → \`STAGE1_EXPERIMENTS.md\` → \`STAGE2_REVISIONS.md\` → Track A/B/C 문서.
- 이 페이지의 카드는 \`outputs/milestone7/\`의 JSON 5개를 \`/api/m7\`(dashboard/server.py \`m7()\`)로 읽어 그린다: \`causal_pilots_report\`, \`teval_report\`, \`vk_report\`, \`m3r_report\`, \`replay_report\`. 이 JSON들은 \`scripts/m7_replay.py summary\`와 \`scripts/m7_reports.py all\`이 frame별 CSV에서 CPU로 다시 만든다.

## 결정: KEEP_SGGL (2026-10-14 기준, FINAL_ALGORITHM_DECISION.md)
- **SGGL**: frozen SAM 2.1 Hiera-T 상태 두 개. G = 전체 frame(P5 prompt memory), L = 변이 prompt box의 4배인 정사각 crop(중심이 출력을 따라감). L의 object score > 0, 마스크 비어 있지 않음, crop 내부 경계에 닿지 않음이면 L, 아니면 G.
- 18개 후보 계열(S 선택, F 결합, M 메모리, R 해상도, T 학습 head, O 사건 실행)을 screening했고, 실제로 실행한 prototype은 33종이다(재생 19, 인과 3 + 개정 1, 학습 head 5, 조합 4, R1 probe 1).
- **최종 규칙**(사전 등록): DEV + REP의 영상 군집 CI 하한 > 0으로 SGGL을 이겨야 교체. **통과한 방법은 없다.** Stage 1 pilot 규칙의 ADVANCE도 0개다.
- 가장 가까운 후보 **M3-vel**(crop 중심 감쇠 속도 예측): +0.0046 [−0.0001, +0.0115], 영상 하나씩 빼기 최솟값 +0.0017. 그러나 이득이 영상 2개(skate-park, sheep)에 몰려 있어 pilot 규칙의 '영상 ≥ 3'을 못 채웠다. 판정은 INCONCLUSIVE이고, 최종 문구는 INSUFFICIENT_EVIDENCE(지정 도전자)다.
- SGGL 자체는 VKITTI(개발 scene 2개)에서 P5 대비 +0.017로 개발 세트와 같은 방향·크기였다(서술용).
- SGGL-V(G 부재 시 L 거부권)는 miami-surf 하나에만 효과가 있어 선택 사항으로 내렸다.
- PUMaVOS C는 봉인 유지.

## 왜 M8(후보 생성 포트폴리오)로 이어졌나
주된 근거는 Oracle 분해(아래 'Oracle 상한' 카드)다.

| 상한 (GT 사용, 배포 불가) | SGGL 대비 | 뜻 |
|---|---|---|
| {G, L} frame별 선택 | +0.018 | 같은 두 마스크를 완벽히 골라도 이 정도 |
| + 마스크 결합 | 선택 위에 +0.002 | 섞기는 거의 가치 없음 |
| + 성분 단위 편집 | 선택 위에 +0.006 | 이것도 작음 |
| {G, L, L2, L-보호, L-신뢰} 선택 | +0.033 (선택 위에 +0.015) | **새 후보**가 있어야 큰 이득 |

- 실패 분류에서 가장 큰 손실은 '7 두 view 모두 틀림'(SGGL J&F 손실 합 156.6, 14개 영상)이고, 대부분은 G와 L이 **둘 다 '없다'고 말하는** frame이다(동시 부재).
- 결론: 개선 여지는 '같은 두 마스크를 더 잘 고르거나 섞는 것'이 아니라 '두 view가 모두 틀린 frame에서 맞는 후보를 만드는 것'이다. 그래서 M8은 rescue 생성기 25종(Stage 1), ranker, 인과 arm으로 구성했다. M3-vel은 M8에서 **M3vel arm**(같은 기계 대조)으로 그대로 들어갔다.
- M8 시작 감사(docs/m8/REPORT_INGESTION_AND_REPO_AUDIT.md)가 M7 기록에서 추가로 잰 사실:
  - 동시 부재 frame은 486개(전체의 9.6%)다. 그중 131개만 GT가 있고 355개는 정말 비어 있다. rescue는 진짜 부재에서 아무것도 출력하지 않아야 하므로 거짓 양성률이 1차 지표가 됐다.
  - GT가 있는 동시 부재 frame에서 GT가 crop 안에 든 비율의 중앙값은 0.54다. 절반은 대상이 crop **밖**에 있다는 뜻이다. '해상도'보다 '위치' 문제이고, 이것이 운동 기반 crop 제어(FLOWVEL 등)로 이어졌다.
  - 동시 부재 frame을 전부 완벽히 고쳐도 객체 평균 +0.028이다. 그래서 STRONG_WIN(+0.03)은 이 데이터에서 도달 불가능하다고 미리 공개했다.
- 2026-10-14 기준 M8에서는 FLOWVEL(RAFT 광류로 crop 중심 이동, 부재 중에도 계속 따라감)이 유일한 PROMISING 배포 가능 arm이다. 자세한 것은 M8 페이지.

## 주의
${CAVEAT_EXPLORATORY}
- 기계: M7 기록·재생(Stage 0)은 Ubuntu RTX 2080, 인과 pilot·VK·M3-vel은 Vast RTX 5090에서 돌렸다. 모든 비교는 **같은 기계에서 만든 SGGL 기준선**과 짝지었다(P5 출력이 객체마다 정확히 같은지 assert로 확인).
- 극소 대상(< 0.1%)은 7객체뿐이라 극소 분층 결론은 약하다.`,
  },

  "m7.rule": {
    title: "M7 pilot 판정 규칙 (ADVANCE / REJECT / INCONCLUSIVE)",
    body: `## 무엇인가
M7이 모든 pilot(재생, 인과, T-head)에 똑같이 적용한 사전 등록 규칙이다. 코드: \`scripts/m7_replay.py: pilot()\`. 판정은 JSON의 \`verdict\` 필드에 있지만 이 페이지 표에는 판정 열이 없다. 아래 규칙은 표의 열을 보고 직접 확인하면 된다.

| 판정 | 조건 (작은 객체, SGGL 대비) | 표에서 볼 열 |
|---|---|---|
| ADVANCE | ΔJ&F ≥ +0.005 **그리고** 0.01 넘게 오른 영상 ≥ 3 **그리고** LOVO min > 0 **그리고** miami-surf 제외 > 0 **그리고** 최악 객체 ≥ −0.05 | ΔJ&F, vid ↑, LOVO min, w/o miami, worst unit |
| REJECT | ΔJ&F ≤ 0 **또는** (오른 영상 ≤ 2 **이면서** LOVO min ≤ 0) | ΔJ&F, vid ↑, LOVO min |
| INCONCLUSIVE | 나머지 | |

- M8은 같은 규칙을 쓰되 ADVANCE를 **PROMISING**으로 이름만 바꿨다. M8에는 STRONG_WIN(ΔJ&F ≥ +0.03, CI 하한 > 0)이 추가됐다.
- M7 **최종 교체 규칙**은 더 엄격하다: DEV + REP 영상 군집 CI 하한 > 0.

## 예시 (2026-10-14 기준, STAGE1_EXPERIMENTS.md)
- M3-vel(인과): +0.0046, 오른 영상 2, LOVO min +0.0017. 영상 ≥ 3을 못 채워 ADVANCE가 아니고, LOVO min > 0이라 REJECT도 아니다. 그래서 **INCONCLUSIVE**다.
- F1-blend.7(인과): +0.0043 < 0.005, 오른 영상 2, LOVO min −0.0046이라 **REJECT**.
- F1-blend(재생): +0.0013, 오른 영상 3, LOVO min −0.0086이라 규칙상 INCONCLUSIVE지만 miami-surf를 빼면 −0.0086이라 계열 판정은 REJECTED로 덮어썼다.

## 왜 이런 조건인가
- 평균만 보면 한 객체의 큰 이득(예: miami-surf obj5 +0.48)이 전체를 끌어올린다. '영상 ≥ 3', 'LOVO min > 0', 'miami-surf 제외 > 0'은 이득이 **한 영상에만 기대는** 경우를 걸러 낸다. miami-surf는 M7 데이터에서 거짓 재획득이 일어난 유일한 영상이라 따로 지정했다.
- '최악 객체 ≥ −0.05'는 평균 이득 뒤에 숨은 큰 회귀를 막는다.

## 주의
- 같은 개발 영상에서 수십 개 정책을 비교했다(다중 비교). INCONCLUSIVE는 '거의 통과'가 아니라 '근거 부족'이다.`,
  },

  "m7.forest": {
    title: "Forest plot 읽는 법 (점 = 추정치, 선 = 95% CI)",
    body: `## 무엇을 보여 주나
- 한 행이 정책(또는 비교) 하나다. 점은 SGGL 대비 평균 ΔJ&F(작은 객체 단위 평균), 가로선은 95% 신뢰구간, 세로선은 0이다. 행은 추정치가 큰 순서로 정렬된다(\`charts.tsx: forestRows\`).
- 색: **파랑** = CI 하한 > 0(0보다 확실히 큼), **빨강** = CI 상한 < 0(확실히 나쁨), **회색** = CI가 0을 포함. CI가 없는 행은 기본색.

## 어떻게 계산했나
- CI는 \`rfeas.risk.cluster_bootstrap\`의 **영상 군집** bootstrap 2000회다. 같은 영상의 객체들이 함께 뽑히므로 실제 표본 크기는 객체 수가 아니라 영상 수다.
- 예외: VKITTI 카드는 scene이 2개뿐이라 **객체 군집** bootstrap이고 서술용이다.

## 읽는 법과 함정
- 회색이라고 '효과 없음'이 아니다. 표본이 작아 구별하지 못한다는 뜻이다.
- 파랑이라도 한 영상이 끌어올린 것일 수 있다. 표의 'vid ↑/↓', 'LOVO min', 'w/o miami'를 함께 본다.
- 두 행의 CI가 겹친다고 같은 것은 아니고, 안 겹친다고 차이가 유의한 것도 아니다. 두 정책을 직접 비교하려면 짝지은 차이(예: T-head 단계 간 차이 카드)가 필요하다.
- 점추정치가 같아도 CI 폭이 크게 다를 수 있다(예: F1-blend는 넓고, M1-trusted는 매우 좁다). 폭이 넓으면 소수 객체의 큰 변화, 좁으면 거의 모든 객체가 0 근처라는 뜻이다.`,
  },

  // ------------------------------------------------------------------ cards
  "m7.m3r": {
    title: "M3-vel-r: M3-vel 개정판의 인과 재실행 (m3r_report.json)",
    body: `## 무엇을 보여 주나
- \`m3r_report.json\`은 이름과 달리 단순 재실행이 아니라 **M3-vel의 사전 등록 개정판(M3-vel-r)** 결과다(STAGE2_REVISIONS.md). 행은 하나이고, 열의 뜻은 Causal pilots 표와 같다.

## M3-vel과 개정 내용 (\`src/spatial/policies.py\`)
- **M3-vel**: 출력 선택은 SGGL과 같다. 다음 frame의 crop 중심만 바꾼다: 중심 = 현재 출력 box 중심 + 0.5 × (직전 중심 이동량). 감쇠를 준 등속 예측이다. 출력이 비면 crop은 제자리에 머문다(부재 중에는 따라가지 않음).
- **M3-vel-r** (preregistration_r3.json, 출력 전 고정, 개정 1회 제한):
  1. 속도는 최근 두 출력이 **모두 채택된 L crop 출력**일 때만 쓴다(G/L 출처 전환 때 중심이 튀는 것을 막으려는 의도).
  2. 예측 이동을 crop 변의 1/8로 제한한다. damp 0.5는 그대로.
- 성공 조건: SGGL 대비 ADVANCE, 그리고 표적 회귀 3객체(lab-coat obj2, scooter-board obj2, lady-running obj2)의 손실을 줄이면서 skate-park·sheep 이득의 50% 이상 유지.

## 어떻게 계산했나 (\`scripts/m7_reports.py: m3r()\`)
- \`runs/M3-vel-r/*.csv\`의 frame별 JF를 객체 평균한다.
- 객체 0–39번은 Ubuntu RTX 2080, 40–63번은 Vast RTX 5090에서 실행했다(\`runs/M3-vel-r/PROVENANCE.json\`). 기준 SGGL도 객체마다 같은 기계 것을 쓴다(Ubuntu: \`rec/\`, Vast: \`runs/SGGL/\`). P5(G) 출력이 정확히 같은지 assert로 확인한다.
- 작은 객체(prompt 면적 < 1%) 57개, 영상 34개에 pilot 규칙을 적용한다.
- 화면에 없는 JSON 필드 \`minus_M3vel\`: M3-vel-r − M3-vel의 짝지은 차이와 CI. M3-vel은 Vast에서만 돌았으므로 Ubuntu 객체에서는 **기계 간 비교**라는 caveat가 붙어 있다.

## 결과 (2026-10-14 기준)
| | ΔJ&F | CI | 영상 ↑/↓ | LOVO min | 최악 |
|---|---|---|---|---|---|
| M3-vel | +0.0046 | [−0.0001, +0.0115] | 2 / 1 | +0.0017 | lab-coat obj2 −0.014 |
| M3-vel-r | +0.0044 | [−0.0001, +0.0114] | 2 / 1 | +0.0015 | lab-coat obj2 −0.018 |
| 차이 | −0.0001 | [−0.0007, +0.0003] | | | |

- **개정 실패**: lab-coat obj2는 오히려 나빠졌다(−0.0145 → −0.0176). 이득 영상은 여전히 2개다. 손실 사례에서도 출처가 바뀌지 않았으므로 '출처 전환' 가설은 맞지 않는다.
- 사전 등록대로 추가 변형은 만들지 않았고, M3-vel-r은 VK 결선에 넣지 않았다.

## 연구 질문과의 연결
- M3 계열의 이득 형태는 '대상이 사라졌다 다시 나타날 때 10–18 frame 빨리 재획득'이다(\`viz/20_M3vel_timelines.png\`). 사라지기 직전 속도로 crop이 이동 방향 앞에 놓였기 때문으로 보인다.
- 이 관찰이 M8의 운동 계열로 이어졌다: M8은 원본 M3-vel을 그대로 M3vel arm(대조)으로 두고, 부재 중에도 광류로 crop을 계속 옮기는 FLOWVEL을 시험했다.

## 주의
- CI 하한 −0.0001은 bootstrap 잡음 수준이라 seed에 따라 부호가 바뀔 수 있다. '거의 유의'로 읽으면 안 된다. 주된 문제는 이득이 영상 2개에 몰린 것이다.
${CAVEAT_EXPLORATORY}`,
  },

  "m7.causal": {
    title: "Causal pilots: 인과 rollout으로 확인한 3개 정책 (causal_pilots_report.json)",
    body: `## 무엇을 보여 주나
- 재생 screening 뒤 실제 SAM 2 rollout으로 다시 돌린 정책 3개: **F1-blend.7**, **M3-vel**, **M1-always**. forest plot과 표의 행이 각각 한 정책이다.

## 재생과 인과의 차이
- **재생**: Stage 0에 기록한 SGGL 궤적(crop 위치, 메모리)을 그대로 두고 출력만 정책대로 바꿔 채점한다. CPU로 빠르지만 '출력이 바뀌면 다음 crop 중심과 메모리도 바뀐다'는 피드백을 무시한다.
- **인과(causal)**: 정책을 넣고 처음부터 다시 SAM 2를 돌린다. crop 중심이 **그 정책 자신의 출력**을 따라간다. crop 중심을 바꾸는 정책(M3-vel)은 인과로만 평가할 수 있다.

## 어떻게 계산했나 (\`scripts/m7_reports.py: pilots()\`)
- Vast RTX 5090에서 \`runs/<정책>/*.csv\`와 같은 기계의 \`runs/SGGL/*.csv\`를 객체 단위 평균 JF로 합친다.
- 같은 기계 확인: 각 객체에서 정책 실행의 G(=P5) JF와 SGGL 실행의 G JF가 **정확히 같아야** 한다(assert). 다르면 보고서를 만들지 않는다.
- 작은 객체(prompt 면적 < 1%) 57개, 영상 34개에 pilot 규칙(\`m7.rule\`)을 적용한다. J&F는 prompt frame 0을 뺀 frame 1..n−1 평균, CI는 영상 군집 bootstrap 2000회.

## 사전 등록에서 벗어난 점
- 원래 인과 pilot은 F1-blend.7, **S2-observe**, M1-always였다. 재생에서 S2-observe가 SGGL-V와 1 frame만 달라서, 첫 pilot 출력(12:19 KST) 전인 11:45 KST에 **M3-vel로 교체**했다(\`preregistration_r2_amend.json\`). 그래서 S 계열은 인과 확인이 없다.

## 결과 (2026-10-14 기준, STAGE1_EXPERIMENTS.md §3)
| 정책 | ΔJ&F | CI | 영상 ↑/↓ | LOVO min | 재생 추정 | 판정 |
|---|---|---|---|---|---|---|
| M3-vel | +0.0046 | [−0.0001, +0.0115] | 2 / 1 | +0.0017 | 재생 불가 | INCONCLUSIVE |
| F1-blend.7 | +0.0043 | [−0.007, +0.021] | 2 / 7 | −0.0046 | +0.0048 | REJECT |
| M1-always | +0.0003 | [−0.003, +0.003] | 1 / 1 | −0.0009 | +0.0003 | REJECT |

## 읽는 법
- **재생과 인과가 0.0005 이내로 일치했다**(F1-blend.7, M1-always). 출력만 바꾸는 정책에서는 재생 screening이 타당했다는 근거다.
- F1-blend.7의 양의 평균은 miami-surf obj5(+0.48) 한 객체 덕이다. 'w/o miami'는 음수다.
- M3-vel은 miami-surf 의존이 없고 LOVO min이 양수인 **유일한** 후보지만, 이득이 skate-park obj2(+0.169), sheep obj4(+0.060)에 몰려 있다.

## 주의
${CAVEAT_EXPLORATORY}
- M3-vel은 M8에서 RunPod RTX 4090으로도 재현됐다(+0.0046, docs/m8/STAGE2_CAUSAL_RESULTS.md).`,
  },

  "m7.teval": {
    title: "T-head 평가 (T0–T4): 학습형 특징 결합 head (teval_report.json)",
    body: `## 무엇을 보여 주나
- Track C(목표 조건 특징 결합)의 학습 head 5개를 DEV 영상에서 평가한 결과. forest plot과 표의 열은 다른 pilot 카드와 같다.

## head 구조 (TRACK_C_TARGET_ATTENTION.md, \`src/spatial/feature_fusion.py\`, \`target_attention.py\`)
- 공통: frozen SAM 2.1 Hiera-T에서 L의 메모리 조건 특징(256×64×64)과 L logit(256×256)을 받아, L logit에 **더할 잔차**를 낸다. 마지막 층을 0으로 초기화해서 **처음 출력은 L logit과 같다**(테스트로 확인). crop이 채택된 frame에서만 쓰고, 메모리에는 쓰지 않는다(출력 전용).

| head | 추가 입력 | 파라미터 |
|---|---|---|
| T0 | 없음. L만 정제 (전이 없는 대조) | 115k |
| T1 | + G 특징·logit을 crop 격자로 정확히 재표본화해 CNN 결합 | 155k |
| T2 | + frame-0 목표 token 4개(fg/bg × L/G)가 L, G 특징을 따로 cross-attention으로 읽음, FiLM | 355k |
| T3 | + 기권 gate α ∈ [0,1] (G 쪽 채널 전체에 곱함) | 355k |
| T4 | + 신원 제약: 목표와 닮지 않은 G 셀은 전이하지 않음 | 355k |

## 학습과 평가
- 학습: REP(DAVIS 2017 train) crop 객체 43개, SGGL 궤적에서 2 frame마다 캐시. 영상 hash 4:1 분할 → fit 18영상 1,198 frame, val 5영상 237 frame. AdamW lr 1e-3, 3,000 step, BCE + dice. 모든 head 같은 설정(preregistration_r2.json).
- 평가: **DEV만**, SGGL 궤적 위 출력 전용, 작은 객체 16개 / 영상 12개. \`scripts/m7_reports.py: teval()\`이 \`outputs/milestone7/teval/*.csv\`에서 객체 평균 후 pilot 규칙 적용. 'Δ vs P5' 열은 G(P5) 대비.

## 양성 대조 (positive control)
- 고정 64개 표본을 1,500 step 외우게 했을 때 J가 0.869 → 0.966(T1), 0.965(T4)로 올랐다. 통과 기준은 +0.05다. 기울기, 라벨 정렬, 모양은 정상이고 **학습 자체는 된다**. M6.1 head처럼 '학습이 안 되는 버그'가 아니다.

## 왜 18개 영상으로는 일반화하지 않았나
- val(학습에 안 쓴 영상 5개)에서 **loss는 내려가는데 J는 입력(0.848) 아래로 떨어진다**(예: T2 0.849 → 0.842).
- soft BCE + dice는 경계의 애매한 픽셀 확률을 조정하는 것만으로도 줄어든다. 그래도 이진 마스크는 나아지지 않는다. 학습 loss가 낮다고 VOS에서 일반화되는 것은 아니다.
- 학습 데이터가 18개 영상(1,198 frame)뿐이라 외형·운동 다양성이 부족하다. 그래서 결론은 '구조가 불가능하다'가 아니라 '이 데이터 규모에서는 일반화하지 않는다'다. SA-V급 대규모 학습이면 결론이 바뀔 수 있지만, 그 방향은 SAM2Refiner / HQ-SAM 2 같은 선행연구와 겹친다.

## 결과 (2026-10-14 기준)
- 5개 모두 SGGL보다 나쁘다(−0.0008 ~ −0.0134), 모두 REJECT. T2/T3이 가장 덜 나쁜 이유는 거의 수정하지 않기 때문이다(최악 객체 −0.006 수준).
- 메모리 되쓰기 실험은 '출력 전용으로 이득이 있을 때만'이라는 사전 조건을 못 채워 하지 않았다.

## 연구 질문과의 연결
- 이 head들은 L의 logit을 **고치는** 구조라, L이 '없다'고 말하는 frame(동시 부재)은 원리상 거의 살리지 못한다. M7의 가장 큰 손실을 겨냥하지 못한 셈이다. M8은 attention을 마스크 보정이 아니라 **후보 탐색·순위화**(B_att 생성기, R3 set-attention ranker)에 다시 썼다.

## 주의
- 16객체 / 12영상이라 CI가 넓고, 다른 pilot 카드(57객체)와 직접 비교하면 안 된다.
${CAVEAT_EXPLORATORY}`,
  },

  "m7.tdelta": {
    title: "T-head 단계 간 차이 (Tk − Tj)와 attention 주장 규칙",
    body: `## 무엇을 보여 주나
- head 사이의 **짝지은 차이**: 같은 작은 객체(DEV 16개)에서 객체별 (Tk − Tj)의 평균과 영상 군집 bootstrap CI. 각 단계가 구조 요소 하나를 더한 것이므로, 차이는 그 요소의 기여다. 코드: \`scripts/m7_reports.py: teval()\`.

| 비교 | 더한 요소 | 2026-10-14 기준 값 | 의미 |
|---|---|---|---|
| T1 − T0 | 전역(G) 특징 전이 | −0.0007 [−0.002, +0.002] | 교차 view 특징이 L 단독 정제보다 정보를 더하지 않는다 |
| T2 − T1 | 목표 token cross-attention | +0.0121 [+0.003, +0.029] | attention이 CNN 결합보다 **덜 해롭다** |
| T3 − T2 | 기권 gate | +0.0005 [−0.001, +0.002] | 차이 없음 |
| T4 − T3 | 신원 제약 | −0.0073 [−0.016, −0.002] | 신원 제약은 오히려 해롭다 |
| T2 − T0 | attention + 전이 전체 | +0.0114 [+0.002, +0.028] | T2가 T0보다 덜 나쁨 |

## attention 주장 규칙 (사전 등록)
- 'attention이 도움이 된다'고 말하려면 T2/T3/T4가 T1보다 **+0.005 이상, CI 하한 > 0**이어야 한다.
- T2 − T1은 이 조건을 만족한다. **그러나** T2 자체가 SGGL보다 나쁘다(−0.0013). 그래서 주장할 수 있는 것은 'attention이 CNN 결합보다 덜 해롭다'뿐이고, attention의 **이득**은 주장할 수 없다.
- 짝지은 차이가 양수여도 기준선(SGGL)을 넘지 못하면 '개선'이 아니다. 이 카드에서 가장 흔히 잘못 읽는 부분이다.

## 주의
- 16객체 / 12영상, 비교 5개(다중 비교). T2가 덜 나쁜 이유는 잔차를 작게 만들어 거의 수정하지 않기 때문으로 보인다. 'attention이 더 똑똑하다'는 증거가 아니다.`,
  },

  "m7.vk": {
    title: "VKITTI 결선 (vk_report.json): 서술용, 개발 scene",
    body: `## 무엇을 보여 주나
- 합성 데이터 VKITTI2 Scene01/06의 28개 객체(1,736 frame)에서 SGGL과 고정 후보를 비교한 결선이다. 사전 등록: \`preregistration_r4_vk.json\`.
- 설명 줄의 \`note\`, \`units\`, \`frames\`는 JSON 그대로다.

## 어떻게 계산했나 (\`scripts/m7_reports.py: vk()\`)
- \`rec_vk/*.csv\`: Vast RTX 5090에서 SGGL을 실시간으로 돌리며 SGGL JF, G(=P5) JF, 공식 SAM 2(B0)의 J를 함께 기록.
- \`runs/SGGL-V/vkitti*.csv\`, \`runs/M3-vel/vkitti*.csv\`: 같은 기계 arm. P5가 rec_vk와 1e-9 이내로 같은지 assert.
- 객체 단위 평균의 차이를 비교마다 계산한다. 코드는 **작은 객체 필터 없이 28객체 전부**를 쓴다.
- CI는 **객체 군집** bootstrap 2000회(\`CI_by_unit_descriptive\`). scene이 2개뿐이라 영상(scene) 군집 CI는 의미가 없다. 객체끼리는 같은 scene 안에서 강하게 상관돼 있으므로 이 CI는 **실제 불확실성보다 좁다**. 그래서 서술용이다.
- 'J:'로 시작하는 행은 J&F가 아니라 **J만**이다. 공식 SAM 2(B0)는 J만 기록했기 때문이다.
- \`veto_frames_total\` / \`veto_units\`: SGGL은 crop을 택했지만 SGGL-V는 G 부재로 거부한 frame 수 / 그런 객체 수.

## 결과 (2026-10-14 기준, STAGE2_REVISIONS.md §4)
- SGGL − P5 +0.017 (Scene01 +0.017, Scene06 +0.020, 22객체 ↑ / 1 ↓). 개발 세트(+0.011–0.018)와 같은 방향·같은 크기.
- 유일한 큰 손실 obj57(−0.073): crop이 이웃 차량으로 넘어간 오대상 추적(\`viz/21_VK_examples.png\`).
- SGGL-V − SGGL ≈ 0: 거부권이 1객체 5 frame에서만 발동.
- M3-vel − SGGL +0.0013: 비회귀 조건(≥ −0.005)은 통과하지만 이득이 작다.
- M3-vel-r은 개정 성공 조건을 못 채워 넣지 않았다.

## 읽는 법과 주의
- **'독립 재현'이 아니다.** Scene01/06은 M2–M4의 개발 scene이었다. SGGL에는 처음 적용했지만 프로젝트 전체로는 새 데이터가 아니다. 프로젝트 규칙상 보류 scene(02, 18, 20)은 쓰지 않았다.
- 실질 표본은 scene 2개다. 말할 수 있는 것은 '개발 세트와 같은 방향의 서술적 일치'뿐이다.
- 합성 주행 장면이라 DAVIS류 실제 영상과 분포가 다르다. 극소 대상(< 0.1%)은 VK에 없다.`,
  },

  "m7.replay": {
    title: "Replay pilots: 기록된 SGGL 궤적 위 출력 정책 screening (replay_report.json)",
    body: `## 무엇을 보여 주나
- Stage 1에서 재생으로 screening한 정책 19종(SGGL 제외)의 결과. 열은 Causal pilots 표와 같다. 각 정책의 정의는 행 이름 옆 '?'를 본다.

## 재생이란 (인과와 비교)
1. **기록(Stage 0, Ubuntu RTX 2080, \`scripts/m7.py rec\`)**: SGGL을 실시간으로 돌리며 frame마다 G, L, L2(2배 crop), L-보호(prompt 메모리만), L-신뢰(일치 frame 메모리만), G-보호, 해상도 probe의 logit·score·마스크를 모두 저장. probe는 출력·메모리에 영향이 없음을 비트 단위로 테스트했다.
2. **재생(\`scripts/m7_replay.py\`, CPU)**: 저장된 관측에 정책(\`src/spatial/policies.py\`)을 적용해 출력만 다시 만들고 GT로 채점한다. **crop 위치와 메모리는 SGGL 것 그대로**다.
- 그래서 재생은 '출력만 바꾸는 정책'의 근사다. 실제로는 출력이 바뀌면 다음 crop 중심(출력을 따라감)도 바뀐다. 이 피드백까지 보려면 인과 rollout이 필요하다(Causal pilots 카드). crop 중심을 직접 바꾸는 M3-vel은 재생할 수 없다.
- 근사가 괜찮았는지: 인과로 다시 돌린 F1-blend.7, M1-always는 재생 추정과 0.0005 이내로 일치했다.

## 재생 충실도 (설명 줄의 값)
- \`replay_fidelity.max_abs_J_diff_SGGL_replay_vs_live\`: 재생으로 다시 만든 SGGL의 J와 실시간 SGGL의 J의 frame별 최대 차이. 2026-10-14 기준 0.0009인데, logit을 fp16으로 저장해서 생긴 차이다. 평균 JF 차이는 2e-7이다. 재생 파이프라인은 SGGL을 거의 그대로 재현한다.

## 계산
- 작은 객체 57 / 영상 34(DEV 16 + REP 41 객체), pilot 규칙(\`m7.rule\`), 영상 군집 bootstrap 2000회.
- 화면에 없는 \`pilots_by_set\`: DEV, REP 각각의 평균 Δ. 예: F1-blend.7은 DEV −0.0043, REP +0.0083이고 REP 양수는 miami-surf obj5 한 객체다.

## 결과 요약 (2026-10-14 기준)
- ADVANCE 0개. INCONCLUSIVE는 F1-blend(실질은 음), M1-trusted(효과 0)뿐.
- **S 계열**(선택 증거)의 양의 점추정치는 모두 miami-surf 한 영상에서 온다. 그 영상을 빼면 전부 음수. 감사에서 거짓 재획득이 일어난 영상이 1개뿐이라 이 계열은 검정력이 없다.
- **F 계열**(마스크 결합)은 블렌드를 제외하고 모두 CI가 0 아래. 상한 자체가 작다(Oracle 카드).
- **M1 계열**(보호 메모리)은 |Δ| ≤ 0.0004로 정보가 없다. 보호 메모리가 L을 고치는 frame은 26, 망치는 frame은 361이었다.
- 조합 K1–K5는 \`combos_report.json\`에 따로 있고 이 화면에는 없다(모두 REJECT).

## 주의
${CAVEAT_EXPLORATORY}`,
  },

  "m7.oracle": {
    title: "Oracle 상한 (작은 객체, SGGL 대비): 선택 / 결합 / 성분 / 추가 후보",
    body: `## 무엇을 보여 주나
- 'GT를 보고 frame마다 가장 좋은 후보를 골랐다면 SGGL보다 얼마나 좋았을까'의 상한. **배포 불가능**한 진단이다. 개선 여지가 어디에 있는지 나누어 보는 데 쓴다.

## 어떻게 계산했나 (\`scripts/m7_replay.py: summary()\`)
- 기록된 후보마다 frame별 JF_k = (J_k + F_k)/2를 구한다.
- oracle: frame마다 후보 집합의 max를 구하고, 객체별로 평균한 다음, 객체마다 SGGL을 뺀 값을 평균한다(작은 객체 57 / 영상 34). CI는 영상 군집 bootstrap 2000회.

| 행 | 후보 집합 / 정의 | 2026-10-14 기준 |
|---|---|---|
| or_sel | {G, L} 중 frame별 최선 = **선택** 상한 | +0.018 [+0.004, +0.040] |
| or_fuse | {G, L, 평균(blend), 합집합, 교집합} | +0.020 (선택 위 +0.002) |
| or_comp | max(or_sel, 성분 oracle). 성분 oracle = G/L이 다른 연결 성분을 GT로 하나씩 켜고 끔 | +0.024 (선택 위 +0.006) |
| or_sel5 | {G, L, L2, L-보호(Lp), L-신뢰(Lt)} | **+0.033** [+0.016, +0.057] |
| JF_blend | oracle 아님: 항상 확률 평균을 쓸 때 | −0.006 |
| JF_L2 | oracle 아님: L 대신 항상 2배 crop L2 | −0.008 |
| JF_Lp | oracle 아님: L 대신 항상 L-보호 | −0.076 |
| JF_Lt | oracle 아님: L 대신 항상 L-신뢰 | −0.004 |

## 분해와 해석
- **선택**(같은 두 마스크 중 고르기): +0.018. SGGL의 P5 대비 이득(+0.011)보다 크지만 상당 부분이 miami-surf 하나에 있다.
- **결합**(두 마스크 섞기): 선택 위에 +0.002. M6.1의 '+0.003'을 다시 확인했다. 결합 규칙에 투자할 근거가 없다(Track A도 실제로 전부 실패했다).
- **성분 편집**: 최대 +0.006. 이것도 작다.
- **추가 후보**(L2, Lp, Lt 생성): 선택 위에 +0.015, 합계 +0.033. 남은 여지 대부분이 **새 후보를 만드는 것**에 있다.
- 그런데 추가 후보를 **항상** 쓰면 모두 L보다 나쁘다(JF_L2, JF_Lp, JF_Lt가 음수). 가치는 '어느 frame에서 쓸지'를 아는 데 있고, M7은 GT 없이 그 frame을 고를 신호를 찾지 못했다.

## 왜 M8로 이어졌나
- M8이 selection에서 **candidate generation**으로 방향을 바꾼 근거가 이 분해다. M8 Stage 1은 25개 rescue 생성기를 동시 부재 frame에 돌려 recall(J > 0.5)과 진짜 부재 frame의 거짓 양성을 재고, ranker(R0–R3)로 '언제 쓸지'를 학습했다. M8의 ORACLE_POOL arm이 같은 질문(완벽한 순위화의 상한)을 인과 rollout에서 다시 잰다.

## 주의
- 후보가 많으면 frame별 max는 그것만으로도 커진다(우연히 맞은 frame도 포함). 그래서 or_sel5가 크다는 것은 '좋은 후보가 있다'와 '후보가 많다'가 섞인 값이다.
- frame별 oracle은 피드백을 무시한다. 실제로 다른 출력을 고르면 다음 crop 중심과 메모리가 바뀐다.
- 추가 후보의 이득은 skate-park, kite-surf, sheep, dog-gooses 영상에 몰려 있다.`,
  },

  "m7.taxonomy": {
    title: "실패 분류 (taxonomy): 범주별 SGGL J&F 손실 합",
    body: `## 무엇을 보여 주나
- SGGL이 틀린 frame을 실패 메커니즘 12(+1) 범주로 나누고, 범주별로 SGGL이 잃은 J&F를 합한 막대 그래프다. 막대가 길수록 그 메커니즘을 고쳤을 때 얻을 수 있는 몫이 크다.

## 어떻게 계산했나 (\`scripts/m7_replay.py: taxonomy()\`)
1. 대상 frame(DEV + REP **64객체 전체**, 작은 객체만이 아님): G·L 존재 판단 불일치, 또는 |J_L − J_G| > 0.3, 또는 GT가 있는데 SGGL J < 0.5, 또는 GT가 비었는데 SGGL 출력이 있음.
2. 각 frame에 아래 조건으로 **다중 라벨**을 붙이고, 우선순위(11 → 3 → 3b → 6 → 9 → 2 → 1 → 4 → 8 → 7 → 5 → 10)의 첫 라벨을 1차 범주로 정한다. 아무것도 안 붙으면 12_unresolved.
3. 막대 값 = 1차 범주 frame들의 Σ(1 − SGGL JF). 단위는 'frame-JF'이고, 1.0은 한 frame을 완전히 잃은 것이다. GT는 오프라인에서 범주를 나눌 때만 썼다.
- JSON에는 \`frames_primary\`, \`frames_any\`(다중 라벨 기준), \`units\`, \`videos\`, \`top_videos\`도 있다.

| 범주 | 조건 (GT 있음 = gp) | 뜻 |
|---|---|---|
| 1_G_absent_resolution | gp, G 부재, J_L ≥ 0.5, G-보호도 실패, 해상도 낮춘 L도 후보를 잃음 | G가 해상도 때문에 놓침 |
| 2_G_absent_state | gp, G 부재, G-보호는 맞음(J ≥ 0.5) | G의 누적 메모리 상태 실패 |
| 3_L_false_reacq | L 존재, J_L < 0.1, GT 없음 또는 L이 다른 주석 객체 위, 재획득 상태 | L의 거짓 재획득 |
| 3b_L_wrong_object_tracking | 3과 같되 재획득 상태가 아님 | L이 계속 다른 객체를 추적 |
| 4_correct_L_vetoed | gp, J_L ≥ 0.5, G 부재, SGGL은 채택했지만 SGGL-V면 거부 | 거부권의 비용 |
| 5_complementary | gp, J_L·J_G ≥ 0.3, 결합이 둘 중 최선보다 0.05 이상 좋음 | 상보적 마스크 |
| 6_different_targets | G, L 모두 존재, IoU_LG < 0.1, 한쪽이 다른 주석 객체 위 | 두 view가 다른 대상 |
| 7_both_wrong | gp, J_L < 0.3 **그리고** J_G < 0.3 | 두 view 모두 틀림 |
| 8_crop_escape | gp, GT의 crop 내부 비율 < 0.9, J_L < J_G | 대상이 crop 밖으로 |
| 9_memory_feedback | L 존재, J_L < 0.3, L-보호는 J ≥ 0.5 | 메모리 양성 되먹임 |
| 10_boundary_thin | gp, IoU_LG ≥ 0.5, SGGL J < 0.8 | 경계 / 얇은 구조 |
| 11_annotation_absence | GT 없음, G·L 모두 존재하고 IoU_LG > 0.5 | 주석상 부재(알고리즘 실패라기보다 주석 문제) |
| 12_unresolved | 어느 조건에도 안 맞음 | 미분류 |

## 결과 (2026-10-14 기준, OPPORTUNITY_AND_FAILURE_AUDIT.md §2)
- 1위 **7_both_wrong** 156.6 (165 frame, 20객체, 14영상; drone 63 frame). 2위 12_unresolved 61.2(miami-surf 47 frame). 3위 11_annotation_absence 35.0(drone 28).
- 3_L_false_reacq는 7 frame, **영상 1개(miami-surf)**뿐이다. 그래서 S 계열과 M1(거짓 재획득 겨냥)은 원리상 검정력이 없었다.
- 9_memory_feedback은 영상 2개뿐이라 M2(메모리 허용 분리)는 구현하지 않았다.
- 8_crop_escape는 손실은 7.5로 작지만 9개 영상에 가장 넓게 퍼져 있어 M3(crop 중심 제어)를 인과로 평가한 이유가 됐다.

## 주의
- 1차 범주는 우선순위 순서에 따라 바뀐다. 예를 들어 crop 이탈이면서 두 view 모두 틀린 frame은 8로 먼저 잡힌다(8이 7보다 앞). 그래서 '7'은 crop 이탈 조건(J_L < J_G 등)에 안 걸린 동시 실패다. 다중 라벨 수는 \`frames_any\`를 본다.
- 손실이 소수 영상(drone, miami-surf)에 크게 몰려 있다.`,
  },

  "m7.coabsence": {
    title: "'두 view 모두 틀림'과 동시 부재(co-absence)",
    body: `## 무엇인가
- **7_both_wrong**(M7 분류): GT가 있는데 J_L < 0.3 **그리고** J_G < 0.3인 frame. M7에서 가장 큰 손실 범주다.
- **동시 부재(co-absence)**(M8 용어): G와 L의 object score가 **둘 다 ≤ 0** → SGGL 출력이 빈 마스크인 frame. GT 유무와 상관없이 정의된다.
- 둘은 겹치지만 같지 않다. M7 감사에 따르면 7_both_wrong의 대부분이 '두 view가 함께 없다고 말하는데 작은 GT(32–200 px)가 있는' 경우였다(L 존재율 0–17%). 대부분이 GT가 있는 동시 부재라는 뜻이다.

## 왜 SGGL이 못 고치나
- SGGL의 gate는 G와 L 중 하나를 **고를** 뿐이다. 둘 다 비어 있으면 무엇을 골라도 빈 출력이다. 선택·결합 계열(S, F, M1)이 모두 실패한 근본 이유다.

## 사례 (OPPORTUNITY_AND_FAILURE_AUDIT.md)
- drone obj1: GT의 crop 내부 비율 < 0.9인 frame이 100%. 대상이 crop 밖으로 나갔고 G도 놓친다.
- skate-park obj2, kite-surf obj2: crop 안팎을 오가며 작아진다.
- 이 frame들에서 L2(2배 crop)가 가끔 맞힌다(skate-park J 0.38, sheep 0.23). '다른 후보가 있으면 살릴 수 있다'는 뜻으로, Oracle 카드의 or_sel5와 같은 신호다.

## M8과의 연결 (2026-10-14 기준, docs/m8/REPORT_INGESTION_AND_REPO_AUDIT.md)
- 동시 부재 frame 486개 = GT 있음 131 + GT 진짜 비어 있음 355. rescue는 355개에서 **아무것도 출력하지 않아야** 하므로 거짓 양성률이 1차 지표가 됐다(M8 Stage 1의 raw FP, net_dJ_frames = 구한 J − 거짓 양성 1개당 1).
- GT가 있는 동시 부재 frame에서 GT가 crop 안에 든 비율의 중앙값은 0.54다. 절반은 대상이 crop 밖에 있다. 해상도보다 위치 문제라서 운동 기반 crop 제어로 이어졌다(FLOWVEL: GT가 있는 동시 부재를 같은 기계 기준 131 → 63으로 줄임).
- 이 frame을 전부 완벽히 고쳐도 객체 평균 +0.028이다. 그래서 rescue만으로는 STRONG_WIN(+0.03)에 닿을 수 없다.

## 주의
- drone 한 영상이 7_both_wrong frame의 약 40%(165 중 63)를 차지한다. 일반화할 때 조심한다.`,
  },

  // ------------------------------------------------------------------ pilot table columns
  "m7.col.pilot": {
    title: "pilot (정책 이름)",
    body: `- 평가한 정책의 이름. 이름은 \`src/spatial/policies.py\`의 \`name\`과 \`docs/m7/18_CANDIDATE_MATRIX.md\`의 ID를 따른다. 첫 글자가 계열이다: S = 선택 증거, F = 마스크 결합, M = 메모리 조건, R = 해상도·배율, T = 학습 head, SGGL-V = 기존 거부권 변형.
- 각 행의 이름 옆 '?'에 정의와 결과 해석이 있다.
- 표는 ΔJ&F가 큰 순서로 정렬된다. 위쪽이라고 '좋은' 것이 아니다. 판정은 \`m7.rule\`의 조건을 모두 봐야 한다.`,
  },
  "m7.col.units": {
    title: "units / videos",
    body: `- 평가에 들어간 **작은 객체**(prompt 면적 < 프레임의 1%) 수 / 그 객체들이 속한 영상 수(\`n_units\`, \`n_videos\`).
- DEV + REP 재생·인과 pilot은 57 / 34, T-head는 DEV만이라 16 / 12다.
- 실질 표본은 영상 수다. CI는 영상 단위로 다시 뽑는 bootstrap이라 영상이 적을수록 넓어진다.
- 한 영상에 객체가 여럿이면(예: dog-gooses obj4, obj5) 그 영상의 비중이 커진다.`,
  },
  "m7.col.dJF": {
    title: "ΔJ&F vs SGGL",
    body: `## 정의
- 작은 객체마다 (정책 J&F − 같은 기계 SGGL J&F)를 구하고 객체 평균을 낸 값(\`dJF_vs_SGGL\`).
- 객체의 J&F = frame 1..n−1(prompt frame 0 제외)의 (J + F)/2 평균. J = 마스크 IoU, F = 경계 F-measure.

## 읽는 법
- +0.005 이상이어야 ADVANCE 후보(\`m7.rule\`). 0 이하면 REJECT.
- 규모 감각: SGGL의 P5 대비 이득이 +0.011 정도다(재생, 작은 객체). +0.005는 그 절반이다.
- 평균은 한 객체에 크게 흔들린다. miami-surf obj5 하나가 ±0.48 움직이면 57객체 평균이 약 0.008 움직인다. 'worst/best unit', 'w/o miami'를 함께 본다.`,
  },
  "m7.col.ci": {
    title: "CI (video): 영상 군집 95% 신뢰구간",
    body: `- \`CI_video\`: 객체별 Δ를 **영상 단위로 묶어** 복원 추출하는 bootstrap 2000회의 2.5% / 97.5% 분위수(\`rfeas.risk.cluster_bootstrap\`).
- 같은 영상의 객체들은 서로 상관돼 있으므로 영상 단위로 뽑아야 불확실성을 과소평가하지 않는다.
- M7 최종 교체 규칙은 'DEV + REP에서 CI 하한 > 0'이다. 통과한 정책은 없다.
- 하한이 −0.0001처럼 0에 붙어 있으면(M3-vel) seed에 따라 부호가 바뀔 수 있는 수준이다. '거의 유의'로 읽지 않는다.
- 다중 비교 보정은 없다. 수십 개 정책 중 하나가 우연히 CI > 0이 될 수 있다.`,
  },
  "m7.col.vid": {
    title: "vid ↑/↓: 0.01 넘게 오르거나 내린 영상 수",
    body: `- 객체 Δ를 영상별로 평균한 뒤, +0.01 초과인 영상 수 / −0.01 미만인 영상 수(\`videos_up_0.01\`, \`videos_down_0.01\`).
- pilot 규칙에서 **↑ ≥ 3**이 ADVANCE 조건이고, ↑ ≤ 2이면서 LOVO min ≤ 0이면 REJECT다.
- 예: M3-vel은 2 / 1이다. 이득이 skate-park, sheep 두 영상에만 있고, 이 조건 하나 때문에 INCONCLUSIVE다.
- ↓가 ↑보다 많은데 평균이 양수면(F1-blend.7: 2 / 7) 소수 영상의 큰 이득이 다수 영상의 작은 손실을 덮고 있다는 뜻이다.`,
  },
  "m7.col.unit": {
    title: "unit ↑/↓: 0.01 넘게 오르거나 내린 객체 수",
    body: `- 작은 객체 중 Δ > +0.01인 수 / Δ < −0.01인 수(\`units_up_0.01\`, \`units_down_0.01\`).
- 판정 규칙에는 직접 쓰지 않는 보조 지표다. 영상 단위 집계에 가려진 개별 회귀를 본다.
- 0 / 0이면(M1-trusted, R2-scale, T2, T3) 정책이 거의 아무것도 바꾸지 않았다는 뜻이다. 평균 효과가 0인 이유가 '좋고 나쁨이 상쇄'인지 '아무 일도 안 함'인지 여기서 구별된다.`,
  },
  "m7.col.lovo": {
    title: "LOVO min: 영상 하나씩 빼기 최솟값",
    body: `- leave-one-video-out: 영상 하나를 통째로 빼고 나머지 작은 객체의 평균 Δ를 다시 구한다. 모든 영상에 대해 반복한 결과 중 최솟값이 \`lovo_min\`이다(JSON에는 \`lovo_max\`도 있다).
- **LOVO min > 0**: 어느 영상 하나를 빼도 평균이 양수라는 뜻이다. 이득이 한 영상에만 기대지 않는다. ADVANCE 조건이다.
- LOVO min ≤ 0이면 '그 영상 하나가 없으면 효과가 사라진다'는 뜻. 대개 그 영상은 miami-surf다.
- 한계: 한 번에 하나만 뺀다. 이득이 두 영상에 몰려 있으면(M3-vel) LOVO min은 양수일 수 있다. 그래서 '영상 ↑ ≥ 3' 조건이 따로 있다.`,
  },
  "m7.col.miami": {
    title: "w/o miami: miami-surf를 뺀 평균",
    body: `- \`davis/miami-surf\` 영상의 객체를 모두 빼고 구한 평균 Δ(\`without_miami\`). ADVANCE 조건은 > 0이다.
- 왜 이 영상만 따로 보나: M7 데이터에서 **거짓 재획득**이 일어난 영상은 miami-surf 하나뿐이다. obj5에서 L이 닮은 surfer(방해물)를 잡는다. G도 frame 31–33에 같은 방해물을 잡는다. 거부권·결합 계열이 이 한 객체에서 +0.48을 얻어 평균을 부풀린다.
- 예(2026-10-14 기준): SGGL-V +0.0069 → miami-surf 제외 −0.0019. F1-blend.7 인과 +0.0043 → −0.0046.
- 이 영상을 빼면 거의 모든 S·F 정책이 음수가 된다. 반대로 M3-vel은 이 영상과 무관하다(+0.0045).`,
  },
  "m7.col.worst": {
    title: "worst unit: 가장 크게 손해 본 객체",
    body: `- Δ가 가장 작은 객체의 id와 Δ(\`worst_unit\`). id 형식은 \`<세트>__<데이터셋>/<영상>__obj<번호>\`.
- ADVANCE 조건: 최악 객체 ≥ −0.05. 평균이 좋아도 한 객체를 크게 망치면 탈락이다.
- 자주 보이는 이름:
  - dog-gooses obj5(−0.05 ~ −0.06): G 지지 없이 일어난 **올바른** 재획득을 거부권·결합이 망친다. G의 확률이 물 배경까지 퍼져 블렌드가 마스크를 부풀린다.
  - lab-coat obj2(M3 계열): 극소 대상이라 box 중심이 흔들려 속도 예측이 잡음이 된다.
  - lab-coat obj1(T0/T1/T4): head가 크게 망친 객체.
- 글자가 잘리면 마우스를 올려 전체 id를 본다.`,
  },
  "m7.col.best": {
    title: "best unit: 가장 크게 이득 본 객체",
    body: `- Δ가 가장 큰 객체의 id와 Δ(\`best_unit\`). 판정 규칙에는 쓰지 않는다.
- 이득이 어디서 오는지 빠르게 보는 용도다. 예: 거의 모든 S·F 정책의 best는 miami-surf obj5(+0.48)다. 평균 이득이 이 한 객체에서 온다는 신호다.
- M3-vel의 best는 skate-park obj2(+0.169): 대상이 사라졌다 다시 나타날 때 crop이 이동 방향 앞에 있어 10–18 frame 빨리 재획득한다.
- best 하나가 평균의 대부분을 설명하면(best / 57 ≈ 평균) 효과는 사실상 한 객체에서 나온 것이다.`,
  },
  "m7.col.p5": {
    title: "Δ vs P5",
    body: `- 작은 객체에서 (정책 J&F − P5 J&F)의 평균(\`dJF_vs_P5\`). P5는 SGGL의 G 상태, 즉 전체 frame + prompt 메모리만으로 추적한 출력이다. 같은 실행의 G 출력을 기준으로 쓰므로 같은 기계다.
- 참고값: SGGL 자체의 P5 대비 이득이 재생 작은 객체에서 +0.011 [−0.010, +0.027]이다(miami-surf obj5 −0.477 포함). 정책의 Δ vs P5가 이보다 작으면 SGGL보다 나쁜 것이다.
- 의미: '이 정책이 crop을 쓰는 이점을 얼마나 지켰나'. F3-sym, F4-instance, F-union은 +0.000–0.001로 crop의 이점을 거의 다 잃었다.
- T-head 카드에서는 DEV 16객체 기준이라 값의 규모가 다르다.`,
  },

  // ------------------------------------------------------------------ VK table columns
  "m7.vk.col.cmp": {
    title: "비교 (A − B)",
    body: `- 각 행은 '앞 − 뒤'의 객체별 차이다. 'J:'로 시작하면 J&F가 아니라 J(IoU)만이다.
- SGGL-P5: crop 이점. SGGL-V-SGGL: 거부권 효과. M3-vel-SGGL: 최선 도전자. J:SGGL-B0, J:P5-B0: 공식 SAM 2 대비. 행 이름 옆 '?'에 자세히 있다.
- 출처: \`vk_report.json\`, \`scripts/m7_reports.py: vk()\`. 28객체 전부, scene 2개.`,
  },
  "m7.vk.col.mean": {
    title: "mean (VK)",
    body: `- 28개 VKITTI 객체의 객체별 차이 평균. 작은 객체 필터는 없다.
- 개발 세트(DEV + REP 작은 객체) 수치와 비교할 때는 객체 구성(합성 주행 장면, 차량 위주, 극소 대상 없음)이 다르다는 것을 감안한다.`,
  },
  "m7.vk.col.ci": {
    title: "CI (unit, descriptive)",
    body: `- 객체 단위 bootstrap 2000회의 95% 구간(\`CI_by_unit_descriptive\`). scene이 2개뿐이라 scene 군집 bootstrap을 할 수 없어서 객체 단위로 뽑았다.
- 같은 scene의 객체끼리는 상관돼 있으므로 이 구간은 **실제보다 좁다**. 이름 그대로 '서술용'이며 통계적 검정으로 쓰면 안 된다.`,
  },
  "m7.vk.col.scene": {
    title: "scene별",
    body: `- Scene01, Scene06 각각의 객체 평균 차이(\`by_scene\`, 소수 넷째 자리 반올림).
- 실질 표본이 이 두 값이다. 두 scene에서 부호가 같으면 '방향 일치' 정도를 말할 수 있다. 예(2026-10-14 기준): SGGL − P5는 +0.0165 / +0.0204로 둘 다 양수.`,
  },
  "m7.vk.col.updown": {
    title: "↑/↓ 0.01 (객체 수)",
    body: `- 차이가 +0.01 초과인 객체 수 / −0.01 미만인 객체 수(\`up_0.01\`, \`down_0.01\`). 개발 세트 표와 달리 **영상이 아니라 객체** 단위다.
- 예: SGGL − P5는 22 / 1로, 대부분 객체에서 crop이 이득이다. SGGL-V − SGGL은 0 / 0으로, 거부권이 사실상 발동하지 않았다.`,
  },
  "m7.vk.col.worst": {
    title: "worst (VK)",
    body: `- 차이가 가장 작은 객체 id와 값.
- SGGL − P5의 최악 obj57(−0.073): crop이 이웃 차량으로 넘어가 다른 차를 추적한 오대상 추적이다(\`outputs/milestone7/viz/21_VK_examples.png\`). 개발 세트의 3b(L이 다른 객체 추적) 범주와 같은 종류의 실패다.`,
  },

  // ------------------------------------------------------------------ VK rows
  "m7.vkrow.SGGL-P5": {
    title: "SGGL − P5 (VK)",
    body: `- SGGL의 crop(L) 상태가 전체 frame 단독(P5)보다 얼마나 나은가. SGGL을 쓰는 이유가 되는 효과를, SGGL을 처음 적용한 scene에서 다시 확인한 것이다.
- 2026-10-14 기준 +0.017, 두 scene 모두 양수, 22 ↑ / 1 ↓. 개발 세트(+0.011–0.018)와 방향·크기가 같아 KEEP_SGGL 결정의 보조 근거가 됐다. 단 scene이 2개뿐이고 개발 scene이라 독립 재현은 아니다.`,
  },
  "m7.vkrow.SGGL-V-SGGL": {
    title: "SGGL-V − SGGL (VK)",
    body: `- G가 대상이 없다고 하면 L을 거부하는 SGGL-V의 효과. 설명 줄 위의 \`veto_frames_total\`/\`veto_units\`가 실제 거부 frame 수다.
- 2026-10-14 기준 −0.00002: 1객체 5 frame에서만 발동. 개발 세트에서도 miami-surf 하나에만 효과가 있었으므로 SGGL-V는 선택 사항(ablation)으로 내렸다.`,
  },
  "m7.vkrow.M3-vel-SGGL": {
    title: "M3-vel − SGGL (VK)",
    body: `- 최선 도전자 M3-vel(crop 중심 감쇠 속도 예측)의 VK 결과. 같은 기계(Vast) 인과 실행.
- 2026-10-14 기준 +0.0013 [−0.0001, +0.0029], Scene01 +0.0015 / Scene06 +0.0002, 1 ↑ / 0 ↓. 비회귀 조건(≥ −0.005)은 통과하지만 이득이 작다. 교체 근거는 되지 못했다.`,
  },
  "m7.vkrow.J:SGGL-B0": {
    title: "J: SGGL − 공식 SAM 2 (B0)",
    body: `- 공식 SAM 2 video predictor(B0, 전체 frame 1회, 프로젝트 수정 없음) 대비 SGGL의 J(IoU) 차이. B0는 J만 기록해서 J&F가 아니다.
- 2026-10-14 기준 +0.034, 24 ↑ / 1 ↓. '우리 방법이 공식 기준선보다 낫다'를 VK에서 확인하는 행이다.`,
  },
  "m7.vkrow.J:P5-B0": {
    title: "J: P5 − 공식 SAM 2 (B0)",
    body: `- P5(전체 frame + prompt 메모리, SGGL의 G 상태)와 공식 SAM 2의 J 차이. 대조 행이다.
- 2026-10-14 기준 −0.001로 거의 0이다. VK에서 SGGL의 B0 대비 이득(+0.034)은 P5의 메모리 설정이 아니라 **crop(L) 상태**에서 온다는 뜻이다.`,
  },

  // ------------------------------------------------------------------ pilot rows (causal / replay / T-head)
  "m7.pilot.M3-vel": {
    title: "M3-vel: crop 중심 감쇠 속도 예측 (인과 전용)",
    body: `- 정의(\`policies.py: M3Velocity\`): 출력 선택은 SGGL과 같다. 다음 crop 중심 = 현재 출력 box 중심 + 0.5 × (직전 중심 이동). 출력이 비면 crop은 그대로.
- crop 위치를 바꾸므로 재생할 수 없고 인과 rollout으로만 평가했다(Vast, 같은 기계 SGGL).
- 2026-10-14 기준 +0.0046 [−0.0001, +0.0115], 영상 2 ↑ / 1 ↓, LOVO min +0.0017, miami-surf 제외 +0.0045라 **INCONCLUSIVE**다. M7의 지정 도전자.
- 이득: skate-park obj2 +0.169, sheep obj4 +0.060(재등장 시 10–18 frame 빠른 재획득). 손실: lab-coat obj2 −0.014, scooter-board obj2 −0.011, lady-running obj2 −0.009(극소·흔들리는 대상).
- 이득과 crop 이탈 비율의 상관은 0.08로 약해 '이탈 방지'만으로 설명되지 않는다.
- M8에서 M3vel arm으로 변경 없이 재실행해 같은 값(+0.0046)이 나왔다.`,
  },
  "m7.pilot.M3-vel-r": {
    title: "M3-vel-r: M3-vel 개정판",
    body: `- M3-vel + (1) 최근 두 출력이 모두 채택된 L 출력일 때만 속도 사용, (2) 이동을 crop 변의 1/8로 제한. 사전 등록 1회 개정(\`preregistration_r3.json\`).
- 2026-10-14 기준 +0.0044 [−0.0001, +0.0114], M3-vel과 차이 −0.0001. 표적 회귀를 줄이지 못해 **개정 실패**. 자세한 것은 카드 제목 옆 '?'.`,
  },
  "m7.pilot.F1-blend.7": {
    title: "F1-blend.7: L 가중 logit 혼합 (0.7 L + 0.3 G)",
    body: `- 정의: SGGL이 crop을 채택한 frame에서 출력 = (0.7 × l_L + 0.3 × l_G > 0). 아니면 G. crop 밖은 L이 관측하지 않은 영역이라 배경 증거로 쓰지 않는다.
- 재생 +0.0048, 인과 +0.0043 [−0.007, +0.021]. 영상 2 ↑ / 7 ↓, miami-surf 제외 −0.0046(인과)이라 **REJECT**. 재생과 인과가 0.0005 이내로 일치했다.
- 이득은 miami-surf obj5(+0.48) 한 객체: G의 강한 음의 logit이 L의 방해물 마스크를 지웠다. 결합이 아니라 **연속값 거부권**으로 작동한 것이다. 최악은 dog-gooses obj5(−0.058).
- 픽셀 회계: 새 TP 104,531 px vs 새 FP 122,456 px(Track A §4).`,
  },
  "m7.pilot.F1-blend": {
    title: "F1-blend: 동일 가중 확률 평균",
    body: `- 정의: crop 채택 frame에서 (σ(l_L) + σ(l_G))/2 > 0.5. 가중치가 같으면 logit 평균 > 0과 수학적으로 같다.
- 재생 +0.0013 [−0.011, +0.020], 영상 3 ↑ / 11 ↓, LOVO min −0.0086이라 규칙상 INCONCLUSIVE지만 miami-surf 제외 −0.0086이라 계열 판정은 REJECTED.
- 회귀 영상이 11개로 많다. 블렌드가 L의 정확한 작은 마스크를 G의 퍼진 확률 쪽으로 부풀린다(dog-gooses obj5 −0.061).`,
  },
  "m7.pilot.F2-boundary": {
    title: "F2-boundary: 경계 띠만 혼합",
    body: `- 정의: SGGL 마스크 경계 ±2 px 띠 안에서만 동일 가중 혼합, 나머지는 L 그대로.
- 재생 −0.0062 [−0.009, −0.003], 회귀 영상 9개라 REJECT. 경계 보정은 순손실이다(정답 px −67,346). '개선이 경계에 있다'는 가설을 지지하지 않는다.`,
  },
  "m7.pilot.F3-asym": {
    title: "F3-asym: 비대칭 추가/삭제 편집",
    body: `- 정의: G에만 있는 성분은 SGGL 마스크에 붙어 있고 L이 부정하지 않을 때(L 확률 ≥ 0.2)만 추가. SGGL에만 있는 성분은 G가 관측 가능하고 G 확률 < 0.05, L 확률 < 0.8일 때만 삭제.
- 재생 −0.0023 [−0.004, −0.001]으로 REJECT.
- 비대칭성 자체는 효과가 있었다. 대칭판(F3-sym −0.0110)보다 덜 나쁘고, 잘못 지운 TP가 6,554 px 대 256,035 px다. 하지만 삭제가 거의 안 일어나 이득도 없고, 추가는 FP가 TP보다 많다.`,
  },
  "m7.pilot.F3-sym": {
    title: "F3-sym: 대칭 편집 (F3 대조)",
    body: `- 정의: G 성분이 인접하고 p_G > 0.5면 추가, p_G < 0.5면 삭제. F3-asym의 비대칭성이 효과가 있는지 보기 위한 대조.
- 재생 −0.0110, 회귀 영상 22개로 REJECT. 최악 −0.099. crop 이점(Δ vs P5)을 거의 다 잃는다.`,
  },
  "m7.pilot.F4-instance": {
    title: "F4-instance: 같은 instance 성분만 합집합",
    body: `- 정의: SGGL 마스크(5 px 팽창)에 닿는 G 성분만 합친다. 떨어진 G 영역은 거부.
- 재생 −0.0105 [−0.016, −0.006]으로 REJECT. 떨어진 성분은 막지만 붙어 있는 G 가장자리 확장이 대부분 FP다(FP/TP 1.74).`,
  },
  "m7.pilot.F-union": {
    title: "F-union: L ∪ G",
    body: `- 정의: crop 채택 frame에서 L과 G의 합집합.
- 재생 −0.0105로 REJECT. 새 TP 31만 px를 더하지만 FP 53만 px도 더한다. G와 L에 상보적 정답 픽셀은 있지만, G의 거친 가장자리에 TP와 FP가 섞여 픽셀 증거만으로는 둘을 나눌 수 없다.`,
  },
  "m7.pilot.F-inter": {
    title: "F-inter: L ∩ G",
    body: `- 정의: crop 채택 frame에서 G score > 0이면 L과 G의 교집합, 아니면 L.
- 재생 −0.0083, 회귀 영상 10개로 REJECT. G가 작은 대상 일부를 놓쳐 맞는 픽셀(TP)을 지운다.`,
  },
  "m7.pilot.SGGL-V": {
    title: "SGGL-V: G 부재 시 L 거부권",
    body: `- 정의: SGGL의 self gate에 'G object score > 0' 조건을 더한다. G가 대상이 없다고 하면 L을 쓰지 않는다.
- 재생 +0.0069 [−0.003, +0.024], 영상 1 ↑ / 4 ↓, miami-surf 제외 −0.0019라 REJECT. 이득은 miami-surf obj5(거짓 재획득) 하나, 비용은 dog-gooses obj5(−0.056: G 지지 없는 올바른 재획득을 막음).
- VK에서도 5 frame만 발동. M7 결정에서 선택 사항(ablation)으로 내렸다.`,
  },
  "m7.pilot.S1-K3": {
    title: "S1-K3: 재획득 후 3 frame 연속 확인 (지속성 대조)",
    body: `- 정의: 출력이 비었다가 다시 잡힌 후보는 유효한 frame이 3번 연속될 때만 채택. 증거 없이 '기다리기'만 하는 단순 대조.
- 재생 −0.0005로 REJECT. 올바른 재획득도 늦춘다.`,
  },
  "m7.pilot.S1-state": {
    title: "S1-state: 재획득 중에만 G 지지 요구",
    body: `- 정의: 평소에는 SGGL과 같고, 출력이 비었다가 다시 잡히는 미확인 재획득 동안에만 G가 같은 위치를 지지(IoU > 0.1)해야 채택. 항상 거부하는 SGGL-V보다 좁게 쓴다.
- 재생 −0.0000으로 REJECT. miami-surf에서는 G도 같은 방해물을 잡아(frame 31–33) 거짓 확인을 주고, dog-gooses obj5의 올바른 재획득(G 지지 없음)은 막는다(−0.051).`,
  },
  "m7.pilot.S2-observe": {
    title: "S2-observe: G가 관측 가능할 때만 거부권",
    body: `- 정의: G 부재가 L을 거부하는 것은 G가 대상을 볼 수 있었을 때(G 입력에서 대상 변 ≥ 24 px, G가 최근 신뢰됨)만.
- 재생 +0.0070으로 REJECT. SGGL-V와 frame 1개만 다르다: SGGL-V가 손해 보는 4객체에서 '관측 불가'가 한 번도 나오지 않았다. 그 객체들의 G 부재는 크기로 설명되지 않는다.
- 원래 인과 pilot이었으나 이 이유로 M3-vel로 교체됐다(S 계열은 인과 확인 없음).`,
  },
  "m7.pilot.S3-abs": {
    title: "S3-abs: frame-0 참조 신원 (절대)",
    body: `- 정의: L 후보와 frozen frame-0 참조(stride-16 backbone 특징)의 유사도가 자기 과거 채택 이력보다 3 sd 넘게 낮으면 거부.
- 재생 +0.0003으로 REJECT. 외형 신원 신호는 거의 아무것도 바꾸지 않는다.`,
  },
  "m7.pilot.S3-rel": {
    title: "S3-rel: frame-0 참조 신원 (상대 margin)",
    body: `- 정의: crop 안의 다른 영역이 후보보다 frame-0 참조와 **더** 닮았고(음의 방해물 margin) G 지지도 없으면 거부.
- 재생 +0.0007로 REJECT. miami-surf에서 닮은 surfer는 신원 검사도 '맞다'고 판정한다. 외형이 헷갈리는 문제는 외형 신호로 풀 수 없다.`,
  },
  "m7.pilot.S4-evidence": {
    title: "S4-evidence: 증거 분류 (양성 / 신뢰 음성 / 충돌 / 불충분)",
    body: `- 정의: G의 판단을 네 종류로 나눈다. 신뢰할 만한 음성(G 관측 가능 + 부재)이나 충돌(G가 다른 곳에 있음)이면서 신원 근거가 없을 때만 L을 지운다. 불충분한 전역 증거로는 지우지 않는다. 사실상 S2 + S3.
- 재생 −0.0010으로 REJECT.`,
  },
  "m7.pilot.M1-verify": {
    title: "M1-verify: 보호 메모리로 재획득 확인",
    body: `- 정의: 미확인 재획득은 G 지지 **또는** L-보호(prompt 메모리만 쓴 L 예측)가 같은 후보를 찾을 때(IoU > 0.3, score > 0)만 채택.
- 재생 −0.0004로 REJECT. 보호 메모리가 L을 고치는 frame은 26, 망치는 frame은 361(감사 2×2 표)이라 정보가 없다.`,
  },
  "m7.pilot.M1-always": {
    title: "M1-always: 보호 메모리 불일치 시 항상 거부",
    body: `- 정의: L-보호와 불일치하고(IoU ≤ 0.1 또는 부재) G 지지도 없으면 항상 L을 거부.
- 재생 +0.0003, 인과 +0.0003으로 REJECT. 재생과 인과가 일치했다. 최악 lady-running obj2 −0.043.
- L과 L-보호의 불일치는 시간이 지나며 늘지만(31–60 frame 19%), 대부분 L이 맞는 정상적인 외형 적응이다. 불일치는 신원 신호가 아니라 '최근 메모리가 필요하다'는 신호다.
- 직접 선례: Matthews et al. 2003 'The Template Update Problem'. 이 선례의 음성 결과가 SAM 2에서도 나온 것이다.`,
  },
  "m7.pilot.M1-trusted": {
    title: "M1-trusted: 신뢰 메모리 예측으로 대체",
    body: `- 정의: G 지지 없는 미확인 재획득에서는 L-신뢰(L과 G가 일치한 frame의 메모리만 쓴 예측)를 대신 출력.
- 재생 +0.0001, CI가 매우 좁음, 객체 0 ↑ / 0 ↓라 규칙상 INCONCLUSIVE지만 **효과 0**이라 계열 판정은 REJECTED. 거의 발동하지 않는다.`,
  },
  "m7.pilot.R2-scale": {
    title: "R2-scale: 2배 crop 상태와의 일치",
    body: `- 정의: 미확인 재획득은 G 지지 또는 2배 넓은 crop(L2, 독립 메모리)의 일치(IoU > 0.3)로 확인. 배율 간 일관성.
- 재생 +0.0009, 객체 0 ↑ / 0 ↓로 REJECT. 상태 하나(+50% 계산)를 더하지만 바뀌는 것이 거의 없다.
- 관련: L2는 '두 view 모두 틀림' frame에서 가끔 맞힌다. 확인 신호로는 쓸모없지만 **후보**로는 가치가 있다. 이 관찰이 Oracle 카드(or_sel5)와 M8로 이어졌다.`,
  },
  "m7.pilot.T0": {
    title: "T0: L만 정제하는 head (전이 없는 대조)",
    body: `- L 특징과 logit만 받아 잔차를 낸다(115k 파라미터). G 정보가 없으므로 T1 이후의 '교차 view 전이' 효과를 재는 기준이다.
- DEV −0.0127 [−0.028, −0.004]로 REJECT. 최악 lab-coat obj1 −0.135. 학습 head 자체가 이 데이터 규모에서 마스크를 망친다.`,
  },
  "m7.pilot.T1": {
    title: "T1: 정렬된 G 특징 CNN 결합",
    body: `- T0 + G의 메모리 조건 특징과 logit을 crop 격자로 정확히 재표본화(셀 중심 오차 < 1e-3 px)해 CNN으로 결합(155k).
- DEV −0.0134로 REJECT. T1 − T0 ≈ 0: 전역 특징은 정보를 더하지 않는다.`,
  },
  "m7.pilot.T2": {
    title: "T2: 목표 token cross-attention",
    body: `- frame-0 목표 token 4개(fg/bg × L/G)가 L, G 특징을 **따로** cross-attention으로 읽고(전체 해상도 all-to-all 없음), FiLM과 token 유사도 지도를 추가(355k).
- DEV −0.0013 [−0.003, +0.001]으로 REJECT. T2 − T1 = +0.012로 attention 주장 규칙(T1보다 +0.005, CI > 0)은 만족하지만 SGGL보다 나빠서 **이득은 주장할 수 없다**. 거의 수정하지 않아서 덜 나쁠 뿐이다.`,
  },
  "m7.pilot.T3": {
    title: "T3: 기권 gate",
    body: `- T2 + 기권 gate α ∈ [0, 1]. G 쪽 모든 채널에 곱해 G가 믿을 만하지 않으면 전이를 끈다. 입력은 G logit 크기, G score, G token 지도.
- DEV −0.0008로 REJECT. T3 − T2 ≈ 0: gate가 의미 있는 차이를 내지 않는다.`,
  },
  "m7.pilot.T4": {
    title: "T4: 신원 제약 attention",
    body: `- T3 × σ(a·(cos_fg_G − cos_bg_G) + b): 목표와 닮지 않은 G 셀은 전이하지 않는다.
- DEV −0.0081로 REJECT. T4 − T3 = −0.007: 신원 제약은 오히려 해롭다. 양성 대조(J 0.869 → 0.965)는 통과했으므로 학습 실패가 아니라 일반화 실패다.`,
  },
}
