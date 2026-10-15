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
  "m10.interactions": {
    title: "조합 실험: 독립 · 조건부 · 상호작용",
    body: `\`reports/interactions.json\` (\`scripts/m10_analysis.py interact\`). 수정 3에서 고정한 조합 4개를 본다.

**계산 범위:** 데이터셋마다, OR0·A·B·A+B가 **모두 끝난 같은 단위**에서만 영상 동일 가중으로 계산한다.

| 열 | 뜻 |
|---|---|
| ΔA, ΔB | 각 oracle 단독 효과 (독립 상한) |
| ΔA+B | 둘을 함께 적용한 효과 [95% CI] |
| B \\| A | ΔA+B − ΔA. A가 이미 완벽할 때 B가 **추가로** 주는 회복 (조건부 상한) |
| A \\| B | ΔA+B − ΔB |
| 상호작용 | ΔA+B − ΔA − ΔB |

**상호작용 읽는 법**
- 0이면 효과가 더해진다.
- 음수면 겹친다. 같은 손실을 두 oracle이 함께 고친다.
- 양수면 서로를 필요로 한다.

**조합**
- **C48** = 기하 + 완벽한 기억. 기억이 완벽할 때 crop 기하가 아직 의미가 있는가?
- **C58** = 후보 선택 + 완벽한 기억. 기억이 완벽해도 남는 이탈 시작을 선택으로 고칠 수 있는가?
- **C47** = 기하 + 기억 기록 허용.
- **C75** = 후보 선택 + 기억 기록 허용. GT mask를 어디에도 쓰지 않는 조합이다.

CI는 영상을 다시 뽑아 네 값을 함께 계산한 bootstrap이다. 서로 다른 oracle의 회복을 단순히 더해 '전체 손실의 몇 %'라고 말하지 않는다. 겹침이 이 표의 상호작용이다.`,
  },
  "m10.arm.C48": arm("C48 — 기하 + 완벽한 기억 (수정 3)", "OR4(GT 중심·크기 crop) + OR8(매 frame GT로 기억 재부호화). 기억이 완벽할 때 남는 손실이 crop 기하로 더 줄어드는지 본다. 남은 손실은 frame별 분할·디코딩 한계다."),
  "m10.arm.C58": arm("C58 — 후보 선택 + 완벽한 기억 (수정 3)", "OR5S(7개 후보 중 GT 최선을 출력) + OR8. 완벽한 기억에서도 이탈이 '시작'되는데(에피소드 수가 거의 그대로), 그 시작 frame에 맞는 후보가 이미 있는지 본다."),
  "m10.arm.C47": arm("C47 — 기하 + 기억 기록 허용 (수정 3)", "OR4 + OR7. 기하 교정과 기록 거부가 겹치는지 더해지는지 본다."),
  "m10.arm.C75": arm("C75 — 후보 선택 + 기억 기록 허용 (수정 3)", "OR5S + OR7. GT mask를 어디에도 쓰지 않는다. '맞는 후보를 고르고, 틀린 예측을 기억에 안 쓰는' 판별기가 완벽할 때의 상한이다. 실제로 만들 수 있는 방법(검증기)에 가장 가까운 oracle이다."),
  "m10.arm.OR8K10": arm("OR8K10 — 10 frame마다 완벽한 기억 갱신 (수정 3)", "OR8과 같지만, GT 재부호화를 t % 10 == 0인 frame에서만 한다. 나머지 frame은 SAM 2가 원래 하던 대로 기록한다. 드문 교정(사람의 간헐적 클릭, 신뢰할 수 있는 재검출)이 얼마나 효과가 있는지 본다."),
  "m10.gate": {
    title: "개선 단계: GT 없는 선택기 오프라인 관문",
    body: `\`reports/selector_gate.json\` (\`scripts/m10_analysis.py gate\`). 사전 기준은 \`preregistration_amend5.json\`이다(외부 비판 검토 11d를 그대로 채택).

**질문:** SAM 2가 매 frame 내놓는 원시 가설 6개(G 3개, L 3개) 중에서, **GT 없이** SAM 2 자신의 선택(view 안에서 예측 IoU 최대)보다 나은 것을 고를 수 있는가?

**범위**
- SGGL 출력이 비어 있지 않은 frame에서만 작동한다.
- 빈 mask는 고르지 않는다. 존재 판단은 M9에서 반증된 문제라 건드리지 않는다.

**스위치와 정밀도**
- 규칙 점수가 SAM 2의 선택보다 τ 이상 높을 때만 바꾼다(스위치).
- **스위치 정밀도** = 스위치 중 실제로 J가 오른 비율이다.
- 온라인에서는 고른 가설이 기억에 쓰인다. 그래서 틀린 스위치는 오염을 증폭한다(OR7R 교훈). 정밀도가 핵심이다.

**τ\***
- MOSEv2와 VOST를 합친 정밀도가 0.70에 처음 도달하는 τ다.
- 도달하지 못하면 그 규칙은 탈락한다.

**통과 기준 (τ\*에서, MOSEv2와 VOST 각각)**

| 조건 | 기준 |
|---|---|
| GT 있는 frame의 ΔJ | 영상 CI 하한 > 0 |
| 스위치 정밀도 | ≥ 0.6 |
| OR0가 이탈 중이던 frame의 ΔJ | > 0 |
| OR0가 맞던 frame의 ΔJ | ≥ −0.005 |

**표 읽기**
- 'τ=0' 열: 아무 문턱 없이 규칙을 그대로 쓸 때의 ΔJ, 스위치 수, 정밀도 / 악화 비율.
- 정밀도 곡선: τ를 올려 스위치를 줄였을 때 정밀도가 오르는지 보여 준다.

**통과 규칙이 없을 때:** 온라인 검증기 arm은 돌리지 않는다. 결론은 '현재 신호로는 GT 없는 선택기가 불가능하다'이다(M9 결정 트리 Q1의 아래 가지).`,
  },
  "m10.rule.R_piou": arm("R_piou — 예측 IoU (두 view 통합)", "6개 가설 중 디코더 예측 IoU가 가장 큰 것을 고른다. SAM 2는 view 안에서만 이렇게 고른다. 이 규칙은 G와 L을 넘나들며 고른다."),
  "m10.rule.R_sum4": arm("R_sum4 — 네 신호 합", "frame 안에서 표준화한 z(예측 IoU) + z(frame 0 cosine) + z(직전 출력과의 IoU) − z(직전 출력 중심과의 거리). 직전 출력 신호는 이탈 중이면 이탈을 굳히는 자기강화 위험이 있다(검토 11a)."),
  "m10.rule.R_anchor": arm("R_anchor — 기준점 신호만", "z(예측 IoU) + z(frame 0 cosine) + z(다른 view 출력과의 IoU). 직전 출력에 기대지 않는 신호만 쓴다(검토 11a 권고)."),
  "m10.rule.R_cos": arm("R_cos — frame 0 외형 cosine", "G stride-16 특징을 mask로 평균한 벡터와 frame 0 prompt 원형의 cosine이 가장 큰 가설을 고른다."),
  "m10.rule.R_lr": arm("R_lr — 로지스틱 회귀", "예측 IoU, cosine, 직전 IoU, 거리, 교차 view IoU, log 면적, view, view 객체 점수로 'J ≥ 0.5' 확률을 예측한다. 같은 데이터셋 안에서 영상 단위 2-fold로 교차 적합했다(numpy ridge)."),
  "m10.arm.OR0F": arm("OR0F — OR0 + 가설별 신호 기록 (수정 4)", "SGGL 결정은 OR0과 같다(12,989 frame 전부 |Δ| = 0). 6개 원시 가설마다 GT 없이 쓸 수 있는 신호를 기록한다: 디코더 예측 IoU, 면적, 직전 출력과의 IoU·중심 거리, 다른 view와의 IoU, frame 0 외형 cosine. 선택기 관문의 입력이다. ΔJ&F는 정의상 0이다."),
  "m10.arm.VXO": arm("VXO — 두 view가 다를 때 다른 view로 전환, 출력만 (수정 6, 탐색)", "GT 없음. SGGL 출력이 있고, 6개 가설 중 예측 IoU 최대가 다른 view에 있으며 SAM 2 자신의 선택보다 높고, SAM 2의 선택이 다른 view 출력과 IoU < 0.5일 때(두 view가 다를 때) 출력을 그 가설로 바꾼다. 기억은 SAM 2 원래대로다. crop 중심은 출력을 따라간다. 이 조건은 dev를 보고 정한 사후 조건이라 conf만이 편향 없는 시험이다."),
  "m10.arm.VXM": arm("VXM — VXO + 기억 재부호화 (수정 6, 탐색)", "VXO와 같이 바꾸고, 출력을 내던 view의 frame 기억도 고른 가설로 다시 부호화한다(SAM 2 mask 입력 경로). 오라클에서는 기억에 쓰면 이득이 약 2배였다(OR5off → OR5M). 다만 틀린 선택도 증폭된다(OR7R 교훈)."),
  "m10.arm.N1M": arm("N1M — 대조: 같은 frame에서 자기 예측 재부호화 (수정 6)", "출력은 SGGL 그대로다. VXM이 바꾼 바로 그 frame에서, 출력 view의 기억을 자기 원래 mask로 같은 경로를 거쳐 다시 쓴다. '다시 쓰는 행위' 자체의 효과(신뢰도 고정, 이진화)를 잰다. VXM − N1M = 쓰는 '내용'의 효과."),
  "m10.arm.N2X": arm("N2X — 대조: 같은 횟수 무작위 전환 (수정 6)", "단위마다 VXM과 같은 횟수를 seed 고정 무작위 frame에서 바꾼다. 다른 view의 무작위 가설로 출력을 바꾸고 기억도 다시 쓴다. VXM − N2X = '언제·무엇으로 바꾸는지'의 효과."),
  "m10.arm.OR0T": arm("OR0T — OR0 + 디코더 쪽 신호 기록 (수정 7, 탐색)", "결정은 OR0과 같다. 가설마다 다음을 추가로 기록한다: 안정성 점수, 양의 logit 평균, object pointer와 frame 0 prompt pointer의 cosine, 다툼 영역(그 가설에만 있는 영역 vs SAM 2 선택에만 있는 영역)의 frame 0 외형 cosine 차이, 디코더 token 6×256. 학습 검증 head(옵션 b)의 오프라인 분석용이다. 분석 계획은 수정 7에 미리 고정했다."),
  "m10.degraded": {
    title: "선택기 정밀도 vs 기억 쓰기 이득 (degraded oracle)",
    body: `\`reports/degraded.json\` (\`scripts/m10_analysis.py degraded\`). 수정 8.

**질문:** 정밀도 p인 선택기가 고른 가설을 기억에 쓰면 이득이 어떻게 되는가?
- GT 선택(OR5M)은 정밀도 1이고 +0.17 / +0.30이다.
- 정보 없는 기억 개입(OR7R)은 0 또는 음수다.
- 그 사이를 처음으로 실측한 것이 VXM이다.

**VXM:** 두 view가 다를 때 예측 IoU가 더 높은 다른 view 가설로 출력을 바꾼다. 그 가설로 출력 view의 기억도 다시 쓴다.

**정밀도 두 가지**
- **오프라인:** OR0 궤적(OR0F)에서 같은 전환 조건이 걸린 frame 중 J가 오른 비율이다. MOSEv2 0.24, VOST 0.61, LVOS 0.85.
- **온라인 실측:** VXM이 실제로 전환한 frame에서의 비율이다. 기억을 다시 쓰면 궤적이 바뀌어, 전환이 걸리는 frame과 정밀도가 오프라인과 다를 수 있다.
- **동점 제외:** 둘 다 J = 0인 전환을 뺀 값이다.

**읽는 법**
- y(ΔJ&F)가 x(정밀도)와 함께 오르면 "기억 쓰기는 정밀도 p 이상에서만 이득"이라는 문장을 데이터로 쓸 수 있다.
- 실험 전 사용자 예상은 MOSEv2 음수, LVOS 양수, VOST 경계였다.
- 같은 전환을 출력에만 적용한 VXO가 대비점이다.
- 대조군 N1M(같은 frame 자기 재부호화)과 N2X(같은 횟수 무작위)는 '쓰는 내용'의 효과를 분리한다.

**주의:** 데이터셋 3개 = 점 3개다. 기울기를 추정하기에는 부족하다. 방향만 읽는다.`,
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
  "m10.baselines": {
    title: "기준선: 공식 SAM 2.1 → P5 → SGGL (수정 9)",
    body: `\`reports/baselines.json\`, \`docs/m10/BASELINES.md\` (\`scripts/m10_analysis.py baselines\`). 논문 주 표의 세 칸.

- **B0:** 공식 SAM 2.1 Hiera-T 비디오 예측기. frame 0 GT mask prompt, 전체 frame, crop 없음, FP32, fill_hole_area=0. 새 RunPod 4090에서 dev 109단위를 돌렸다.
- **P5:** B0 + frame 0 prompt 기억을 full-res GT mask로 다시 부호화. SGGL의 G 상태가 L을 전혀 읽지 않음을 코드로 확인했으므로, OR0의 \`JF_G\`가 곧 P5다. 독립 실행 P5와 차이 0.
- **SGGL:** OR0.
- **기계:** OR0는 이전 Pod에서 나왔다. 새 Pod에서 OR0 2단위를 다시 돌려 저장값과 J·F 차이 0을 확인했다.
- **Δ:** 같은 frame끼리 짝지은 단위 평균 → 영상 평균 → 영상 동일 가중 평균. CI는 영상 bootstrap 10k. ↑/↓는 Δ가 ±0.01을 넘는 영상 수, LOVO min은 영상 하나를 뺐을 때 평균의 최솟값.
- Fable 서브에이전트가 SGGL − B0와 P5 = JF_G를 독립 재계산해 일치를 확인했다.`,
  },
  "m10.baselines.t1": {
    title: "표 1. 전체 단위",
    body: "데이터셋별 세 방법의 절대 J&F(영상 동일 가중)와 짝지은 차이 세 개. SGGL − B0가 음수면 SGGL이 공식 SAM 2.1보다 나쁘다는 뜻이다. VOST에서 CI가 0 아래에 있다.",
  },
  "m10.baselines.t2": {
    title: "표 2. 극소 단위 (area0 < 0.001)",
    body: "frame 0 객체 면적이 화면의 0.1% 미만인 단위만. SGGL이 노린 대상이다. 단위·영상 수가 작아(VOST 5/3, LVOS 8/6) CI가 넓다.",
  },
  "m10.baselines.t3": {
    title: "표 3. 상태별 손실 분해",
    body: `각 frame을 GT로 사후 분류하고 (1 − J&F)를 상태별로 더한 뒤 frame 수로 나눴다. 열을 모두 더하면 1 − frame 평균 J&F다(영상 가중이 아니라 frame 가중).

- **correct:** J ≥ 0.5인 frame의 남은 손실. **partial:** 출력이 GT와 겹치지만 J < 0.5. **empty:** GT가 있는데 출력이 빔.
- **wrong_region:** 출력의 절반 이상이 GT 밖. **wrong_object:** 그중 절반 이상이 주석된 다른 객체 위. **false_present:** GT가 없는데 출력이 있음.
- **P5:** OR0에 G 출력과 다른 객체의 교집합이 기록되지 않아 wrong_region과 wrong_object를 나눌 수 없다. 합친 열만 비교한다.`,
  },
  "m10.arm.B0": arm("B0 — 공식 SAM 2.1", "기준선(수정 9). crop·P5 없는 공식 비디오 예측기. 오라클 표의 B0 행은 B0 − SGGL이다(다른 arm과 같은 방향). 논문 표는 기준선 카드를 본다."),
  "m10.head": {
    title: "학습한 선택 head (수정 10, 탐색)",
    body: `\`scripts/m10_head.py\`, \`docs/m10/HEAD_RESULTS.md\`. SAM 2 가중치는 그대로 두고, 디코더 내부 출력으로 6개 가설(G 3개, L 3개) 중 무엇을 고를지 점수를 매기는 외부 head를 학습했다.

- **H1 (view 보정):** 두 view 각각의 SAM 2 선택 중 하나를 고른다. 배율 차이로 L의 예측 IoU가 부풀려졌다는 가설이다.
- **H2 (view 안 선택):** SGGL 출력 view의 3가설 중에서 고른다.
- **모델:** a = view별 affine 보정 (4 파라미터), b = 스칼라 MLP, c = MLP + mask/IoU token. null = 같은 frame 안에서 라벨을 섞어 학습한 영 대조. R_piou = 원시 예측 IoU.
- **학습 데이터:** LVOS v2 여분 영상만 썼다. MOSEv2·VOST 여분 영상에는 적격 단위가 없었다. 학습 덤프는 이 Pod의 GPU 오류 때문에 두 번 계산해 비트 단위로 일치한 것만 썼다.
- **τ*:** 학습 영상 5-fold OOF에서 '전환 중 개선 비율 ≥ 0.70'이 되는 가장 작은 마진이다. dev는 τ 선택에 쓰지 않았다.
- **관문 (dev):** MOSEv2+VOST 합동 정밀도 ≥ 0.70, VOST·LVOS ΔJ CI 하한 > 0, MOSEv2 ≥ −0.005, null보다 우세.
- **결과:** 두 회차 모두 불통과라 온라인 arm은 없다. 1회차 b(H1)만 τ*에 도달했지만 dev 정밀도는 0.33이었다.
- **표의 최대 정밀도:** 전환 20회 이상인 τ 중 가장 높은 정밀도다(참고용, 관문 수치가 아님).`,
  },
}
