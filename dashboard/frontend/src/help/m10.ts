import type { HelpMap } from "./types"

// keys: "m10.<element>"
// Sources: outputs/milestone10/preregistration*.json, scripts/m10.py (run_unit, box_unit, rewrite_memory),
// scripts/m10_analysis.py (table, states, pool, episodes, parity, or9), docs/m10/ORACLE_DEFINITIONS.md.

const ONLINE_OFFLINE = `
| 종류 | 무엇을 하나 | 해석 |
|---|---|---|
| **온라인 (rollout)** | GT 정보를 주고 영상 전체를 **다시 추적**한다. 개입이 기억·crop을 통해 이후 frame에도 영향을 준다 | 원인 개입의 실제 효과. 현재 frame과 미래 frame에 미치는 영향이 모두 들어 있다 |
| **오프라인 (사후)** | OR0가 이미 낸 frame별 결과에서 GT로 '더 나은 것'을 고른다. 다시 추적하지 않는다 | 그 frame 하나의 상한. 미래 frame을 개선한다는 주장은 할 수 없다 |

두 종류의 숫자를 같은 성격으로 놓고 비교하면 안 된다. 그래서 그림에서 오프라인은 **빈 마름모**로 표시한다.`

const arm = (title: string, body: string) => ({ title, body })

export const help: HelpMap = {
  "m10.about": {
    title: "M10 Oracle 병목 분해란",
    body: `M9 결론: FLOWVEL과 crop 기하 조절은 새 데이터에서 효과가 없었고, 놓친 손실의 54–68%가 '출력이 대상 밖에 있음'이었다. M10은 이 관찰을 **개입 실험**으로 확인하거나 반증한다.

**방법:** GT로 병목 하나만 '완벽하게' 고쳐 주고, SGGL이 얼마나 회복하는지 본다.
- 위치, 크기, 존재, 후보 선택, 대상 영역 판별, 기억 기록, 기억 내용, 분할 용량을 하나씩 다룬다.
- 회복량이 큰 병목이 실제로 성능을 묶고 있는 병목이다.

**지키는 원칙**
- 출력은 항상 SAM 2가 예측한 mask나 빈 mask다. GT mask를 출력으로 복사하지 않는다(그건 자명하게 1.0이다).
- 한 번에 병목 하나만 고치고, 나머지 조건은 OR0와 같게 둔다.
- dev split만 쓴다. PUMaVOS C는 봉인 상태다.
- Oracle 수치는 진단용 상한이다. 실제 방법의 성능이 아니다.

**최종 질문 (Q1–Q8):** 가장 큰 병목은 무엇인가? 신원 이탈이 주원인인가? 후보가 없는가, 선택을 못 하는가? 기억 기록을 완벽히 하면 얼마나 회복되나? 기억과 기하 중 어느 쪽이 더 큰가? tiny 모델이 한계인가? 다음에 무엇을 만들고 무엇을 그만둘까?

문서: \`docs/m10/ORACLE_DEFINITIONS.md\``,
  },
  "m10.progress": {
    title: "실행 현황",
    body: `막대 하나 = arm 하나 × 데이터셋 하나. 끝난 단위 CSV 수 / dev 단위 수.

| 줄 | 뜻 |
|---|---|
| \`runpod/OR*\` | RunPod RTX 4090에서 돈 단일 oracle (\`outputs/milestone10/runpod/runs/<arm>/\`) |
| \`runpod/OR9\` | GT box prompt 분할 용량 (이미지 모드, GT가 있는 frame만) |
| \`ubuntu/t, b+, l, s\` | 용량 진단(수정 1). Ubuntu RTX 2080에서 SAM 2.1 tiny / base+ / large / small로 SGGL을 실행. 지금은 VOST dev부터 |

**dev 단위 수:** MOSEv2 62(48영상), VOST 31(19영상), LVOS v2 16(12영상, 첫 300 frame).

**작업 분배:** 워커들이 잠금 파일로 같은 작업 큐를 나눠 가진다. arm 우선순위 순서(OR0 → OR7 → OR7W → OR8 → OR4 → OR5M → OR6 → OR2 → OR3 → OR1M → OR5S → OR8C0 → OR7R)로 진행하고, 단위는 긴 것부터 처리한다.`,
  },
  "m10.parity": {
    title: "기준선 재현 (OR0 vs M9 SGGL)",
    body: `OR0는 SGGL을 새 Pod에서 다시 돌린 결과다. 로그 항목을 늘렸을 뿐 계산은 같다. M9의 같은 frame 결과와 J&F가 **완전히 같아야** 이후 비교가 M9와 이어진다.

- **결과:** MOSEv2 4,806 frame과 VOST 3,619 frame 모두 최대 |Δ| = 0.
- **2080 재현:** 용량 진단의 tiny 실행도 M9 2080 SGGL과 완전히 같다.
- **의미:** 새 Pod(같은 GPU 모델과 소프트웨어)는 M9 Pod와 비트 단위로 같은 결과를 낸다. 그러니 M9 결과와 직접 이어서 읽을 수 있다.`,
  },
  "m10.oracleTable": {
    title: "Oracle 상한표",
    body: `\`outputs/milestone10/reports/oracle_table.json\` (\`scripts/m10_analysis.py table\`).

**값:** 데이터셋마다 Δ = (oracle J&F − OR0 J&F).

**영상 동일 가중**
1. frame 평균으로 단위 J&F를 구한다.
2. 단위 평균으로 영상 J&F를 구한다.
3. 영상 평균을 낸다.

영상 하나에 단위가 많아도 그 영상이 결과를 지배하지 않게 하려는 것이다.

${ONLINE_OFFLINE}

**표 읽기**
- **ΔJ&F [CI]:** 굵은 파랑 = CI 하한 > 0, 빨강 = CI 상한 < 0.
- **회복:** ΔJ&F / (1 − OR0 J&F). 기준선이 잃은 J&F 중 몇 %를 되찾았는지.
- **기억 기록률 G/L:** 그 arm에서 G·L view의 frame 기억이 실제로 기록된 비율. OR7류에서만 1보다 작다.
- **\\*:** 아직 일부 단위만 끝남. 숫자가 바뀐다.

**주의:** 서로 다른 oracle의 회복량을 더해서 '전체 손실의 몇 %'라고 말하면 안 된다. 효과가 겹친다. 겹침은 조합 실험(단계 3)으로 따로 잰다.`,
  },
  "m10.forest": {
    title: "숲 그림 읽는 법",
    body: `- **줄:** arm × 데이터셋 하나.
- **점(마름모):** 영상 동일 가중 ΔJ&F.
- **수염:** 영상 단위 bootstrap 95% CI. 영상을 10,000번 다시 뽑아 계산했다.
- **가로선 0:** 효과 없음. 수염 전체가 0보다 오른쪽이면 그 데이터셋에서 개선이 확실하다는 뜻이다.
- **색:** 데이터셋(MOSEv2 / VOST / LVOS v2).
- **빈 마름모:** 오프라인(사후) oracle. 온라인 arm과 성격이 다르다.

데이터셋별로 먼저 읽는다. 합산 값은 보조 정보다.`,
  },
  "m10.lossStates": {
    title: "OR0 손실의 상태별 비중",
    body: `\`reports/states.json\` → \`OR0_loss_share\`. OR0의 frame마다 (1 − J&F)를 손실로 두고, 아래 상태별로 더한 비중이다. 분류는 GT로 사후에 한다.

| 상태 | 정의 |
|---|---|
| correct | GT 있음, J ≥ 0.5. 맞혔지만 경계가 덜 맞아서 생긴 잔여 손실 |
| partial | GT 있음, J < 0.5, 출력 precision ≥ 0.5. **대상의 일부만** 분할 |
| empty | GT 있음, 출력이 비어 있음. 놓침 |
| wrong_region | GT 있음, 출력 precision < 0.5. **출력 대부분이 대상 밖** |
| wrong_object | wrong_region 중 출력의 50% 이상이 **주석된 다른 객체** 위에 있음. 주석된 방해물로 넘어간 경우만 '다른 객체'라고 부른다 |
| false_present | GT 없음(대상이 안 보임), 그런데 출력이 있음 |

wrong_region + wrong_object가 M9에서 말한 '신원·대상 영역 이탈'이다.`,
  },
  "m10.pool": {
    title: "놓친 frame: 정답 후보는 어디에 있나",
    body: `\`reports/pool.json\` (\`scripts/m10_analysis.py pool\`). 대상: GT가 있는데 SGGL 출력 J < 0.5인 frame.

**후보 7개:** SAM 2가 frame마다 이미 만드는 것들이다. GT로 새 후보를 만들지 않는다.
- G view의 원시 가설 3개 (디코더 multimask 출력, 존재 점수로 지우기 전)
- L view의 원시 가설 3개
- 빈 mask

**분류** (위에서부터 처음 맞는 것)

| 분류 | 뜻 | 고칠 곳 |
|---|---|---|
| own_view_misselect | 출력한 view가 '있음'이라 했는데, 고르지 않은 다른 가설이 맞음 (J ≥ 0.5) | 가설 선택(SAM 2 IoU 예측) |
| own_view_suppressed | 출력한 view가 '없음'이라 해서 mask가 지워졌는데, 원시 가설 중 하나가 맞음 | 존재 점수 |
| other_view_output | 다른 view의 실제 출력이 맞음. 게이트가 view를 잘못 고름 | G/L 게이트 |
| other_view_hyp | 다른 view의 고르지 않은 가설만 맞음 | 선택 |
| **none** | 7개 모두 J < 0.5 | **후보 생성 자체의 실패**. 선택을 아무리 잘해도 못 고침 |

막대 = 각 분류가 놓친 손실에서 차지하는 비중. none이 크면 '찾고도 못 고르는' 것이 아니라 '찾지 못하는' 것이다.`,
  },
  "m10.recovery": {
    title: "어느 상태의 손실을 회복하는가",
    body: `\`reports/states.json\` → \`recovery\`. arm을 OR0와 frame 단위로 짝지은 뒤, **OR0에서 그 상태였던 frame들**의 손실 합 중 arm이 되찾은 비율을 계산한다.

- 100% = 그 상태의 손실을 모두 회복. 음수 = 그 상태에서 더 나빠짐.
- 예: OR1off(존재 oracle)는 정의상 false_present만 100% 회복한다.
- 온라인 arm은 궤적이 바뀌므로, 'OR0에서 그 상태였던 frame'이 arm에서는 다른 상태일 수 있다. 회복 비율은 같은 frame의 J&F 변화다.

어떤 oracle이 어떤 실패를 고치는지 보여 준다. 예를 들어 기억 기록 oracle이 wrong_region을 얼마나 고치는지 볼 수 있다.`,
  },
  "m10.episodes": {
    title: "이탈 에피소드",
    body: `\`reports/episodes.json\` (\`scripts/m10_analysis.py episodes\`).

**에피소드 정의**
- **이탈:** 출력이 대상 밖(wrong_region 또는 wrong_object)인 frame이 연속된 구간.
- **손실 구간:** GT가 있는데 J < 0.5인 frame이 연속된 구간.

**칸 = 에피소드 수 · 평균 길이(frame) · 손실 합.** 비교 대상은 **arm이 끝낸 단위와 같은 단위의 OR0**다.

**읽는 법**
- 에피소드 수가 줄었다 → 이탈이 **시작되는 것**을 막는다.
- 평균 길이가 줄었다 → 이탈에서 **빨리 회복**한다.
- 예: OR7(기억 기록 허용)은 수는 비슷하거나 늘고 길이가 크게 준다. 기억 오염을 막으면 이탈을 예방하기보다 회복이 빨라진다는 뜻이다.`,
  },
  "m10.or9": {
    title: "OR9 분할 용량 상한 (GT box prompt)",
    body: `수정 2(\`preregistration_amend2.json\`). SAM 2.1 tiny **이미지 모드**를 쓴다. 기억 없이 frame마다 독립적으로, GT bounding box를 prompt로 준다. GT가 있는 frame만 계산한다.

| 변형 | 입력 |
|---|---|
| box 전체 frame | 전체 frame + GT box |
| box + 완벽 crop | GT 중심, 그 단위의 SGGL crop 크기로 자른 영상 + GT box (위치·크기가 완벽한 view) |

**질문:** 위치와 범위를 완벽히 알려 줘도 SAM 2.1 tiny가 그 대상을 분할하지 못하는가?
- OR9 J&F가 낮으면, 그 손실은 추적이나 기억의 문제가 아니라 **분할 능력(모델 용량)의 한계**다.
- 같은 frame의 OR0 J&F와 비교한다. 추적 결과가 아니므로 다른 arm과 같은 표에 넣지 않는다.`,
  },
  "m10.capacity": {
    title: "모델 크기 진단 (수정 1)",
    body: `SGGL 코드를 그대로 두고 SAM 2.1 체크포인트만 tiny / small / base+ / large로 바꾼다. Pod가 바빠서 **Ubuntu RTX 2080 한 대**에서 모두 돌리고, 같은 기계 안에서만 비교한다.
- **체크포인트:** 공식 092824. sha256은 \`checkpoints/sha256_m10.txt\`.
- **보는 것:** 전체 ΔJ&F만이 아니다. **어느 실패 상태가 줄어드는지**를 본다(wrong_region, partial, empty).
- **판단:** 큰 모델에서 이탈이 크게 줄면, 연구의 기반을 큰 모델로 옮길지 검토한다.
- **범위:** VOST dev부터 시작한다(데이터가 Ubuntu에 있음).`,
  },
  "m10.prereg": {
    title: "사전 등록",
    body: `\`outputs/milestone10/preregistration.json\`은 **M10 결과가 하나라도 나오기 전**에 고정했다(09:02 KST, 커밋 \`378f331\`). 담긴 내용:
- arm 정의
- 데이터 부분집합
- 주 지표(영상 동일 가중 ΔJ&F)
- CI 방법
- 중단 규칙

수정(amendment)은 해당 출력이 나오기 전에 별도 파일로 추가한다.
- **수정 1:** 모델 크기 진단
- **수정 2:** OR9 GT box prompt

수정 1·2의 \`created_kst\`는 손으로 잘못 적었던 시각을 커밋 시각으로 바로잡았다(파일 안에 기록, 다른 필드는 그대로).`,
  },
  "m10.logs": {
    title: "최근 로그",
    body: `\`outputs/milestone10/runpod/logs/w<번호>.log\`: Pod 워커 로그. 단위가 끝날 때마다 한 줄씩 남긴다(arm, 단위, frame 수, J&F, 처리 속도).

\`or9.log\`는 GT box prompt 워커의 로그다. Pod 결과는 Ubuntu \`m8:sync10\`이 300초마다 sha256을 검증하며 가져온다. 리포트 JSON은 \`m8:m10rep\`이 10분마다 다시 만든다.`,
  },
  "m10.figures": {
    title: "실패 에피소드 그림",
    body: `\`scripts/m10_viz.py\`. 대표 단위를 같은 Pod에서 다시 실행해 mask를 저장한 뒤 그린다. 실행이 결정론적이라 원래 결과와 같다.

**배치**
- **행:** GT, 그다음 arm들.
- **열:** OR0에서 손실이 처음 시작되는 frame 앞뒤, 그리고 영상 끝까지 고르게.

**표시**
- 빨강 = 그 arm의 출력 mask
- 초록 선 = GT 경계
- 노랑 사각형 = L crop
- 캡션 \`mem=\`: 대문자 G/L은 그 view의 기억이 기록됨, '-'는 기록이 막힘, '*'는 기억을 다시 부호화함
- 아래 띠 = frame별 J&F 곡선

**고르는 기준:** 성공 사례만 고르지 않는다. 모든 oracle이 실패한 영상도 넣는다.`,
  },
  "m10.col.kind": { title: "종류: 온라인 / 오프라인", body: ONLINE_OFFLINE },
  "m10.col.recovered": {
    title: "회복 비율",
    body: `ΔJ&F / (1 − OR0 J&F). 데이터셋마다 영상 동일 가중으로 계산한다.

예: OR0 J&F가 0.72이면 잃은 몫은 0.28이다. Δ = +0.056이면 회복은 20%다. 기준선이 못 맞힌 몫 중 그 oracle이 되찾은 비율이다.`,
  },
  "m10.col.admit": {
    title: "기억 기록률 G / L",
    body: `frame t의 예측이 그 view의 기억(memory bank)에 기록된 비율이다.
- 기본 SGGL은 매 frame 기록하므로 100%다.
- OR7은 '맞는 예측만 기록'이라 이 비율이 낮다.
- OR7R(대조)은 OR7과 **같은 개수**의 frame을 무작위로 막는다.

OR7 − OR7R이 '어떤 frame을 막느냐'의 효과다. 단순히 '덜 기록한다'는 효과와 구분된다.`,
  },
  "m10.arm.OR0": arm("OR0 — SGGL 기준선", "SGGL(G 전체 frame view + L crop view + 자기 게이트) 그대로. 같은 Pod에서 다시 돌렸고, M9과 frame 단위로 완전히 같다. 다른 모든 arm은 이것과 단위·frame을 짝지어 비교한다."),
  "m10.arm.OR1off": arm("OR1off — 완벽한 존재 판단 (출력만, 오프라인)", "GT가 없는 frame의 출력을 비운다. 나머지는 OR0와 같다. 대상이 있는지 없는지만 완벽히 알 때의 상한이다. 기억에는 영향이 없다(사후 계산)."),
  "m10.arm.OR1M": arm("OR1M — 존재 정보로 기억 기록 막기 (온라인)", "GT가 없는 frame에서 비어 있지 않은 mask를 낸 view는 그 frame을 기억에 기록하지 않는다. 출력과 crop은 SGGL 그대로다. '없는 대상을 있다고 기록한 것'이 이후 frame을 오염시키는지 본다."),
  "m10.arm.OR2": arm("OR2 — 완벽한 위치", "L crop의 중심을 **예측할 frame의 GT box 중심**으로 둔다. crop 크기와 나머지는 SGGL 그대로다. 매 frame 미래 GT를 쓰므로 비인과적 상한이다. GT가 안 보이면 이전 중심을 유지한다."),
  "m10.arm.OR3": arm("OR3 — 완벽한 크기", "L crop의 변을 그 frame GT에 M6 crop 크기 규칙을 적용한 값으로 둔다(GT 긴 변 / 0.25, 하한 min(H,W)/4). 중심은 SGGL 규칙(출력을 따름)이다. M9 SCALE 조절기의 최대 가능 효과다."),
  "m10.arm.OR4": arm("OR4 — 완벽한 기하 (위치 + 크기)", "OR2 + OR3. crop이 매 frame 대상을 정확히 담는다. 이것으로도 실패가 남으면 crop 제어 연구의 우선순위를 낮출 근거가 된다. OR2와 OR3 효과의 합과 비교해 상호작용을 본다."),
  "m10.arm.OR5off": arm("OR5off — 후보 선택 best-of-7 (오프라인)", "frame마다 기존 후보 7개(G 원시 가설 3, L 원시 가설 3, 빈 mask) 중 J가 가장 높은 것을 고른 J&F. 기존 후보 안에서 얻을 수 있는 선택의 상한이다. 사후 계산이라 미래 frame 효과는 없다. '정답 후보가 있는데 못 고르는가'를 잰다."),
  "m10.arm.OR5S": arm("OR5S — 후보 선택 (온라인, 출력만)", "OR5off처럼 GT로 최선 후보를 출력한다. crop 중심이 그 출력을 따르므로 이후 frame에도 영향이 있다. 기억은 SAM 2가 원래 고른 대로 기록한다. 동점이면 SGGL 출력을 유지한다."),
  "m10.arm.OR5M": arm("OR5M — 후보 선택 + 기억", "OR5S에 더해, 각 view의 frame 기억을 **그 view에서 J가 가장 높은 원시 가설**로 다시 부호화한다(SAM 2의 mask 입력 경로). GT mask는 쓰지 않는다. 후보 안에서 고른 기억이다. OR8(GT mask 기억)과의 차이가 '후보 생성의 한계'다."),
  "m10.arm.OR6off": arm("OR6off — 완벽한 대상 영역 판별 (오프라인)", "규칙(위에서부터): 출력이 맞으면 유지 → 다른 view가 맞으면 교체 → 출력이 대상 밖(precision < 0.5)이면 비움 → 아니면 유지(대상 일부). 판별이 완벽해도 맞는 대체 후보가 없으면 회복할 수 없다. '판별'과 '복구'가 다른 문제라는 것을 잰다."),
  "m10.arm.OR6": arm("OR6 — 대상 영역 판별 + 기록 차단 (온라인)", "OR6off의 출력 규칙에, 맞지 않은 view의 기억 기록을 막는 OR7 규칙을 더했다. 완벽한 판별기로 할 수 있는 실제 행동(거부, 다른 후보 선택, 기록 차단)만 쓴다."),
  "m10.arm.OR7": arm("OR7 — 기억 기록 허용 oracle", "view의 frame t 예측이 그 view 영역 안에서 **맞을 때만**(J ≥ 0.5, 또는 GT와 출력이 둘 다 비었을 때) 그 view의 기억에 기록한다. 막힌 frame의 빈자리는 이전에 허용된 frame이 채운다(backfill). GT mask를 기억에 넣지 않는다. 출력은 SGGL 그대로라서 효과는 **미래 frame에만** 나타난다."),
  "m10.arm.OR7W": arm("OR7W — 대상 밖 기록만 차단", "view mask가 비어 있지 않고 precision < 0.5일 때(출력 대부분이 대상 밖)만 기록을 막는다. 대상 일부(partial)와 빈 예측은 기록한다. 기억 오염이 '이탈'에서 오는지, '불완전한 분할'에서 오는지 OR7과 비교해 가른다."),
  "m10.arm.OR7R": arm("OR7R — 대조: 같은 횟수 무작위 차단", "단위·view마다 OR7이 막은 것과 **같은 개수**의 frame을 seed 고정 무작위로 막는다. OR7 − OR7R = 정확한 frame을 고른 효과. OR7R − OR0 = 기록 횟수만 줄인 효과."),
  "m10.arm.OR8": arm("OR8 — 완벽한 기억", "frame t의 출력을 정한 뒤, 각 view의 frame t 기억을 **GT mask로** 다시 부호화한다(add_new_mask와 같은 SAM 2 경로). 현재 frame 출력에는 GT를 쓰지 않는다. 다음 frame부터 '과거가 모두 정답인 기억'으로 예측한다. 기억 내용의 상한이다. OR7(기록 거부)이 약하고 OR8이 강하면, 올바른 대상 정보를 다시 공급하는 문제가 남아 있다는 뜻이다."),
  "m10.arm.OR8C0": arm("OR8C0 — frame 0 기억만", "두 view 모두 처음 prompt frame의 기억만 읽는다. 이후 기록은 무시한다. GT는 쓰지 않는다. 최근 기억이 도움이 되는지, 오염원인지 본다."),
  "m10.arm.OR9": arm("OR9 — GT box prompt 분할 용량", "이미지 모드 SAM 2에 frame마다 GT box를 준다. 추적이 아니라 분할 능력의 상한이다. 별도 카드에서 같은 frame의 OR0와 비교한다."),
}
