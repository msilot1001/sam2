# M8 비-attention 메커니즘 비교

원자료: `stage1_report.json`(진단), `causal_report.json`, `causal_per_unit.csv`(인과, RunPod 4090, 같은 기계 SGGL 대비, 작은 객체 57단위 / 34영상).

## 1. 운동 (계열 C) — **가장 강한 결과**

| arm | 메커니즘 | ΔJ&F vs SGGL | 95% CI (영상 군집) | 영상 ↑ / ↓ | 하나씩 빼기 최솟값 | 최악 단위 | 판정 |
|---|---|---|---|---|---|---|---|
| **FLOWVEL** | 다음 crop 중심 = 현재 대상 중심 + 0.5 × (대상 영역의 RAFT 광류 변위). 출력이 비면 광류로 운반한 box를 계속 따라간다 | **+0.0130** | **[+0.0010, +0.0285]** | 5 / 2 | +0.0067 | scooter-board obj2 −0.026 | **PROMISING** (사전 pilot 규칙 통과, CI 하한 > 0) |
| M3vel (M7 대조) | 다음 중심 = 출력 중심 + 0.5 × (출력 중심 차분) | +0.0046 | [−0.0000, +0.0109] | 2 / 1 | +0.0017 | lab-coat obj2 −0.014 | INCONCLUSIVE (M7 Vast의 +0.0046과 같음) |
| C_flowcrop (진단, 동시 부재 frame에서만) | 광류 box 중심에 L crop | Stage 1 recall 0.015 | — | — | — | — | 진단 실패 |
| C_flowbox (진단) | 광류 box → G box prompt | recall 0.046, 원시 FP 0.80 | — | — | — | — | 진단 실패 |
| E_beam (진단) | 속도 외삽 crop beam | recall 0.000 | — | — | — | — | 진단 실패 |

**FLOWVEL 상세:**
- J/F: ΔJ +0.0106, ΔF +0.0154. 극소 7단위 +0.046. 작은 객체가 아닌 단위를 포함한 64단위 전체 +0.0115.
- **이득이 난 곳:**
  - drone obj1 +0.379
  - skate-park obj2 +0.173
  - kite-surf obj2 +0.132
  - sheep obj4 +0.055
  - miami-surf obj2 +0.044
- **손실이 난 곳:** scooter-board obj2 −0.026, lab-coat obj2 −0.024, miami-surf obj4 −0.011.
- **거짓 재획득:** SGGL과 같다(93 frame 대 93 frame).
- **추가 비용:** frame마다 RAFT-small 1회(긴 변 640). 2080 측정 0.028–0.047 s. 최대 VRAM 1.45 GB. SAM 2 추가 호출 없음.
- **메커니즘 (그림 `viz/31_causal_timelines.png`):**
  - drone obj1에서는 GT가 비는 구간(frame 18–23)이 끝나고 대상이 다시 나타났다. FLOWVEL은 바로 재획득해 36 frame을 추적했다. 같은 구간에서 SGGL과 M3vel은 끝까지 놓쳤다.
  - kite-surf에서는 frame 21–28을 회수했다.
  - skate-park에서는 M3vel과 함께 17 frame 일찍 재획득했다.
  - 동시 부재 frame 자체가 줄었다. RunPod 같은 기계 기준으로 전체 486 → 417이다.
  - **그중 GT가 있는 것(놓친 frame)은 131 → 63으로 절반 이하가 됐다.** M3vel은 108이었다. Stage 1의 rescue가 '이미 놓친 뒤 찾기'였다면, FLOWVEL은 '놓치기 전에, 그리고 놓친 동안에도 crop을 대상 쪽으로 옮겨 두기'다.
- **M3vel과의 차이:**
  - 출력 마스크 중심의 차분 대신 dense flow를 쓴다. 그래서 극소 마스크의 중심 흔들림에 덜 민감하다.
  - 출력이 비어 있는 동안에도 crop을 움직인다.
  - 두 요소의 기여는 ablation(§4)으로 분리한다.

## 2. 척도·시야 (계열 F)

| arm | 메커니즘 | ΔJ&F | CI | 영상 ↑ / ↓ | 최악 단위 | 거짓 재획득 (arm / SGGL) | 판정 |
|---|---|---|---|---|---|---|---|
| RESC_OCLR | 동시 부재 → L 메모리, 8-cell 평범한 crop, 광류 중심. score > 0이면 출력하고 crop 재배치 | −0.0001 | [−0.0034, +0.0029] | 2 / 1 | −0.058 | 99 / 93 | REJECTED |
| RESC_OCLR_ADMIT | + rescue를 L 메모리에 기록 | +0.0046 | [−0.0066, +0.0193] | 3 / 1 | **−0.116** | 105 / 93 | INCONCLUSIVE (하나씩 빼기 −0.0019) |
| RESC_CPFR | 같은 배율의 문맥 보존 foveation | −0.0070 | [−0.0258, +0.0091] | 2 / 3 | **−0.330** | 118 / 93 | REJECTED |

**해석:**
- **foveation (보고서 1순위):**
  - 같은 배율에서 문맥을 보존해도 대상 회수율은 늘지 않았다(Stage 1 Δ ≈ 0).
  - 대신 주변의 비슷한 방해물을 '존재'로 잡는 거짓 양성이 크게 늘었다. 인과 rollout에서는 이것이 신원 전환으로 이어져 최악 단위가 −0.33이 됐다.
  - 현재 데이터는 **문맥 보존 가설에 반대되는 쪽**을 가리킨다.
- **occupancy 조절:** 이득은 drone 한 영상에 몰린다. 출력만 바꾸는 rescue는 다음 frame에서 다시 잃는다(RESC_OCLR −0.0001).

## 3. 메모리 (계열 D)

| 비교 | 결과 |
|---|---|
| D_ref (frame-0 메모리만) vs B_att#0 (전체 메모리, 같은 peak, 후보 1개) (Stage 1) | recall −0.008 [−0.016, 0.000]: 차이 없음 |
| RESC_OCLR_ADMIT − RESC_OCLR (메모리 기록 효과, 인과) | +0.0047. 회수 후 추적이 이어지는 효과가 있지만, 잘못된 rescue도 기록되어 최악 단위 −0.116 |
| ORACLE_POOL (GT 선택, 출력만) | +0.0136 [+0.0020, +0.0297], 5 / 0 — 배포 불가 상한 |
| ORACLE_POOL_ADMIT (GT 선택 + 기록) | +0.0169 [+0.0022, +0.0394], 5 / 0. 완벽한 순위화라면 메모리 기록이 +0.0033을 더한다(회수 frame 61 → 14: 기록된 회수가 이후의 동시 부재를 막는다) |

## 4. FLOWVEL ablation (출력 전 고정, `arms.json`)

| arm | 남긴 요소 | ΔJ&F | 판정 |
|---|---|---|---|
| FV_PRESENT | 광류 속도 (출력이 있을 때만). 부재 중에는 crop 고정 | +0.0016 [−0.0046, +0.0095] | REJECTED (최악 −0.104) |
| FLOWCARRY | 부재 중 광류 box 추적만 (속도 없음) | **+0.0141 [+0.0020, +0.0296]** | PROMISING |
| CAMCARRY (C3) | 부재 중 카메라 운동만큼 이동 | **+0.0137 [+0.0017, +0.0294]** | PROMISING |

메커니즘은 '부재 중 crop 운반'이다. 부재 중 운반 box 안의 광류는 대부분 배경 흐름이므로, FLOWCARRY ≈ CAMCARRY(+0.0003)는 설계상 당연한 결과다. 이 비교는 대상 광류와 카메라 운동을 분리하지 못한다. 자세한 내용은 `STAGE2_CAUSAL_RESULTS.md` §2.1에 있다.

## 5. 단순 기준선과의 비교

M8의 비-attention 방법은 모두 같은 기계의 SGGL과 비교했다. 그 위에 같은 계열의 단순 기준선과도 비교했다.
- 운동: M3vel
- 척도: L2 / Lp
- 메모리: Gp / Lp

계산량 대조: FLOWVEL의 추가 연산은 광류 1회이고 SAM 2 호출은 늘지 않는다. 그래서 '추가 SAM 2 계산 때문에 좋아졌다'는 설명은 성립하지 않는다. rescue arm은 동시 부재 frame에서만 SAM 2를 1회 더 호출한다.
