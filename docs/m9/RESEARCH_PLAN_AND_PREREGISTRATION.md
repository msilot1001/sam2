# M9 연구 계획과 사전 등록

- **고정 파일:** `outputs/milestone9/preregistration.json`, `outputs/milestone9/arms.json` (SCALE, JOINT, PCCARRY 고정 시각 23:13 KST)
- **고정 시점:** 새 데이터에서 SAM 2 출력이 하나도 나오기 전. 첫 출력은 MOSEv2 진단과 VOST 연기 시험이다.
- **근거 문서:**
  - Deep Research 보고서 'M8 Expansion to Broader Datasets: Twelve-Hour Experimental Plan' (보고서 2)
  - M8 이전에 쓰인 'M8 Portfolio Audit' (보고서 1, 일부 낡음)
  - M8 최종 문서

## 질문

1. M8의 가장 강한 메커니즘(FLOWVEL: 두 view가 놓친 동안 crop을 운동에 맞춰 옮김)이 M6–M8에서 반복해서 본 영상 밖에서도 유지되는가?
2. 새 데이터에서 놓친 frame의 주된 원인은 무엇인가? 위치, 크기, 존재·신원, 관측 불가 중 하나를 찾는다.
3. 그 원인에 맞는 최소한의 메커니즘이 SGGL보다 낫고, 더 단순한 부모보다도 나은가?

## 데이터와 표본

`DATASET_AUDIT.md`와 `DATASET_MANIFEST_AND_SPLITS.md`를 따른다.
- MOSEv2 train: 139단위 / 102영상
- VOST val: 74단위 / 44영상
- LVOS v2: 받지 못함
- PUMaVOS C: 봉인

## arm

| arm | 정의 | 고정 |
|---|---|---|
| SGGL | 기준 시스템 | M7 |
| FLOWVEL | 출력 중심 + 0.5 × 대상 영역 RAFT 광류 변위. 출력이 비면 광류 box를 따라감 | M8 (2026-10-14 16:14) |
| FLOWCARRY | 출력이 비었을 때만 광류 box를 따라감 | M8 사후 ablation (보조) |
| PCCARRY | 출력이 비었을 때만 위상 상관 전역 이동으로 crop을 옮김. RAFT 없음 | M9 (23:13) |
| SCALE | crop 한 변 = 4 × (최근 믿을 만한 출력 box의 긴 변). EMA 0.3, frame당 ×[0.8, 1.25], 하한은 SGGL과 같은 min(H,W)/4, 상한 min(H,W). 출력이 비거나 믿기 어려우면 유지 | M9 (23:13) |
| JOINT | FLOWVEL + SCALE (같은 파라미터) | M9 (23:13) |

## Stage 1 최소 진단

- **생성기:** A_cos(top-3), B_att(top-3), C_flowcrop, C_flowbox, F_oclr8_L, 그리고 참조용 Gp, Lp.
- **진출 기준:**
  - recall ≥ 0.25이고 4영상 이상에서 적중
  - attention은 B_att가 A_cos보다 나아야 함
  - 순위화는 held-out 순 이득 > 0
  - R3 attention 순위화는 적중 영상이 20개 이상일 때만 시도

## 실패 분류 (H4 관문)

- **대상:** SGGL 궤적에서 GT는 있지만 J < 0.5인 frame. 다음 순서로 판정한다.
  1. **관측 불가:** GT 면적 < prompt 면적의 0.25배, 또는 64 px 미만
  2. **스케일:** GT 긴 변 > crop 변의 0.6배, 또는 면적 > prompt의 4배
  3. **위치:** GT의 crop 내부 비율 < 0.5
  4. **존재·신원:** 위에 해당하지 않는 경우. 출력이 비었는지(부재 주장), 다른 대상을 잡았는지(오대상)로 나눈다.
- **지배적 범주:** 놓친 손실의 50% 이상을 차지하고 3영상 이상에 걸쳐 있어야 한다. 그렇지 않으면 MIXED로 본다.

## 판정 기준

| 대상 | 기준 |
|---|---|
| FLOWVEL 일반화 | 새 데이터 전체 영상 CI 하한 > 0, 2개 이상 데이터셋에서 평균 > 0, +0.01 이상 상승 영상 ≥ 5개(2개 데이터셋에 걸쳐), 상위 2영상 비중 < 50%, −0.05 미만 단위 ≤ 1개 |
| SCALE 성공 | 스케일 범주 놓친 frame을 30% 이상 줄이고 ΔJ&F ≥ 0(dev), −0.05 미만 단위가 새로 생기지 않음. 그다음 conf에서 확인 |
| JOINT 시너지 | JOINT − max(SCALE, FLOWVEL)의 CI 하한 > 0 |
| 값싼 운동(PCCARRY) | FLOWVEL 효과의 70% 이상을 내면서 지연이 더 적음 |
| 중단 | 평균 음수, 단순한 부모보다 나쁨, 거짓 재획득 증가, −0.05 미만 단위, 이득이 2영상 이하에 몰림 → 중단. 수정은 메커니즘마다 1번, threshold 탐색 금지 |

## 통계

- **주 지표:** 데이터셋별 작은 객체의 짝지은 ΔJ&F. 비교 상대는 같은 기계의 SGGL이다.
- **보고 항목:** 영상 군집 bootstrap 95% CI, 하나씩 빼기 최솟값, 영상 부호 검정, 상위 1·2영상 비중, 최악 단위, −0.05 미만 단위 수, J/F, 극소 분층, 동시 부재 frame 회복, 거짓 재획득, 지연, VRAM.
- **다중 비교:** 데이터셋별 FLOWVEL 검정에 Holm 보정을 적용한다.
