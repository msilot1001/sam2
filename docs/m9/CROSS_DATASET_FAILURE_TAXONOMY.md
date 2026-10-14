# M9 교차 데이터셋 실패 분류와 Stage 1 진단

**재생성 명령:**
- 실패 분류: `uv run python scripts/m9_analysis.py taxonomy <ds>`
- Stage 1: `uv run python scripts/m9_analysis.py stage1 <ds>`

**원자료:**
- `outputs/milestone9/reports/{taxonomy,stage1}_<ds>.json`, `taxonomy_frames_<ds>.csv`, `rankers_mosev2.json`
- GT 형상: `outputs/milestone9/gtbox/<ds>.csv`

**정의:** 사전 등록(`preregistration.json` `failure_taxonomy`)을 따른다.
- 사후에 추가한 것이 하나 있다. 출력이 있고 0.1 ≤ J < 0.5인 frame은 사전 등록 문구에 범주가 없어서 **PARTIAL_SEGMENTATION**으로 따로 셌다(공개).

## 1. MOSEv2 (139단위 / 102영상, RunPod, SGGL 궤적)

- SGGL 작은 객체 단위 평균 J&F: **0.726**
- frame 구성: 10,320 frame 중 GT 있음 6,928, GT 없음 3,392
- 놓친 frame(GT 있음, J < 0.5): 2,308개, 손실(1 − J&F 합) 1,923.7

| 범주 | frame | 손실 | 손실 비중 | 단위 | 영상 | 그중 동시 부재 frame | 최대 영상 비중 |
|---|---|---|---|---|---|---|---|
| **SCALE** | 993 | 852.4 | **44.3%** | 41 | 38 | 244 | 11% |
| POSITION | 367 | 358.4 | 18.6% | 44 | 37 | 141 | 15% |
| PRESENCE_WRONG_TARGET | 270 | 246.4 | 12.8% | 29 | 23 | 0 | 16% |
| OBSERVABILITY | 235 | 187.4 | 9.7% | 56 | 46 | 83 | 12% |
| PRESENCE_ABSENT_CLAIM | 145 | 145.0 | 7.5% | 26 | 24 | 141 | 17% |
| PARTIAL_SEGMENTATION | 298 | 134.1 | 7.0% | 56 | 44 | 0 | 12% |

**거짓 재획득:**
- GT가 빈 3,392 frame 중 **495 frame(14.6%)**에서 SGGL이 마스크를 냈다. 손실 495, 40영상, 최대 영상 비중 13%다.
- 출처: L(gate 통과) 270, G(L gate 실패 후) 225. G 출처 중 L도 '존재'라고 한 것은 32뿐이다.

**판정 (사전 규칙):**
- 손실 50% 이상인 범주가 없어 **MIXED**다.
- 스케일이 위치의 2.4배로 가장 크고, 38개 영상에 고르게 퍼져 있다. 한 영상이 11% 넘게 차지하지 않는다.
- 거짓 재획득 손실을 더하면 존재·신원 계열(오대상 + 부재 주장 + 거짓 재획득 = 886)이 스케일(852)과 비슷해진다.

### 새로 드러난 메커니즘: 첫 frame 크기 불일치

스케일 손실이 큰 단위를 보면, 많은 대상이 **첫 정답 frame에서 화면에 막 들어오거나 일부만 보이다가 몇 frame 안에 크게 커진다**.

| 대상 | prompt frame | 1 frame 뒤 | 몇 frame 뒤 |
|---|---|---|---|
| 006e6b64/obj2 | 251 px | 637 | 5 frame 뒤 3,392 |
| 36cc4055/obj1 | 212 px | — | 5 frame 뒤 3,425 |
| b6520a94/obj5 | 2,403 px | 6,571 (긴 변 103 → 318 px) | — |

- prompt 크기에서 정한 고정 crop(box의 4배)은 곧 대상보다 좁아진다. GT 긴 변이 crop 변의 0.6–0.8배가 된다. 그러면 L 마스크가 crop 가장자리에 닿아 self gate가 실패하고, 해상도가 낮은 G로 넘어간다.
- 그림: `outputs/milestone9/viz_mosev2_scale_failures.png`
- M8의 DAVIS에서는 이 현상이 드물었다. VKITTI의 차량 2대(면적 6.4배 성장)가 같은 유형이었다.

### Stage 1 최소 진단 (MOSEv2, 동시 부재 trigger frame)

- 동시 부재 frame: GT 있음 609 / GT 없음 2,851. 진짜 부재가 **4.7배**다. M8 DAVIS는 2.7배였다.

| 생성기 | recall (J > 0.5) | 적중 영상 | 원시 FP (GT 없음) | score>0 규칙 재생 순이득 |
|---|---|---|---|---|
| A_cos (top-3 point prompt) | 0.227 | 20 | 0.999 | −2,769 |
| B_att (top-3) | 0.225 | 25 | 0.998 | −2,762 |
| Gp | 0.089 | 11 | 0.088 | −205 |
| C_flowbox | 0.066 | 15 | 0.583 | −1,695 |
| Lp | 0.057 | 6 | 0.090 | −225 |
| F_oclr8_L | 0.031 | 9 | 0.058 | −146 (M8에서는 +14.3) |
| C_flowcrop | 0.010 | 5 | 0.013 | −33 |
| 모든 생성기 합집합 | 0.366 | 30 | — | — |

**관문 판정:**
- **탐색 (recall ≥ 0.25):** 모두 **불통과**다. 가장 높은 것이 0.227이다.
- **attention (B_att > A_cos):**
  - recall 차이 −0.002 [−0.044, +0.043]
  - hit@1 차이 +0.005 [−0.034, +0.042]
  - 차이가 없어 **attention 작업을 중단**한다.
- **존재 판정 / 순위화** (dev→conf, conf→dev 교차 적합):
  - R0/R1/R2 held-out 순이득은 −1 / −8 / −1이다. 사실상 모두 기권했다.
  - 대상 존재 frame과 진짜 부재 frame의 판별 AUC는 0.57(SAM 2 score), 0.65(frame-0 코사인)다.
  - 따라서 **불통과**다.
- **R3 attention 순위화:**
  - 적중 영상(20–25)은 사전 조건(20영상 이상)을 넘는다.
  - 그러나 탐색 recall이 기준에 못 미치고, 단순 순위화도 분리 신호를 찾지 못했으며, attention 탐색이 대조군과 같았다. 그래서 **실행하지 않는다.**
- **결론:** M8에서 이미 실패한 '후보 생성 → 순위화' 경로는 새 데이터에서도 실패했다. 진짜 부재가 더 많아 오히려 더 나빠졌다.

## 2. VOST

(SGGL 궤적 완료 후 추가)

## 3. H4 결정

(아래 FINAL과 `FLOWVEL_GENERALIZATION.md`, `SCALE_CONTROL_RESULTS.md`에 이어짐)
