# M5-Pre — 메모리 개입의 인과 효과 검증 (탐색적)

- 브랜치: `exp/m5-pre-memory-intervention` (`exp/risk-memory-feasibility`에서 분기, 병합하지 않음)
- 모델: 공식 SAM 2.1 Hiera-Tiny, CUDA **FP32**. 가중치와 구조는 바꾸지 않았다.
- 제어 범위: 메모리 검색 함수 하나만 *인스턴스 단위*로 교체했다(공식 코드 복사 + 계획 주입). upstream 파일은 수정하지 않았다.
- 데이터: 개발 데이터(DAVIS 2017 val, VKITTI Scene01/06)만 사용했다. 시험 장면은 열지 않았다.
- PART B 개입 manifest는 실행 전에 커밋해 고정했다(`2e69220`).

## 1. CUDA 환경과 정합성 (Gate 0)

| 항목 | 값 |
|---|---|
| OS / GPU | Ubuntu 24.04.5, NVIDIA GeForce **RTX 2080 8 GB** (Turing), driver 595.99.02 |
| PyTorch / CUDA | 2.14.1+cu130, CUDA runtime 13.0, cuDNN 9.24. 시스템 nvcc 없음(필요 없음) |
| Python / 환경 | 3.12.15, uv 0.13.0, `uv.lock` 동일 |
| SAM 2 | upstream `2b90b9f5`, 체크포인트 sha256 `7402e0d8…34be69` (M2와 동일) |
| CUDA 확장 `sam2._C` | **없음.** MPS 실험과 같은 조건이라 hole-filling 생략도 동일 |
| 정밀도 | FP32, autocast 없음, TF32 비활성(Turing은 원래 TF32 미지원) |

**정합성 확인 결과:**
- 같은 CUDA 실행을 반복하면 비트 단위로 같다.
- **제어 구현(native 계획) + 프레임별 실행 루프 = 공식 `propagate_in_video`, 비트 단위로 같다**(pigs 79 frame, 테스트 clip).
- **A1 계획 = 예전 pop 방식 게이트, 비트 단위로 같다.**
- 그래서 **Gate 0을 통과**한 뒤에 개입 실험을 시작했다.

**CUDA vs 기존 MPS (같은 FP32, 다른 장치):**
- M2 8개 track: frame별 CUDA–MPS J 최소 0.994, 최대 20 px 차이(t86). track 지표 차이 < 6e-5. 전체 J&F 0.9511로 같다.
- 누락 시퀀스(M4 t86 burst_30 s0): 63 frame 중 62 frame이 동일하고, GT 대비 J 0.9303으로 같다.
- 두 장치 결과는 거의 같지만 동일한 실험 조건으로 취급하지 않는다. 이번 실험은 모두 CUDA로 다시 수행했다.

**공식 메모리 구조 (코드로 확인):**
- spatial memory: 직전 6개 수신 frame(고정 offset)과 prompt.
- object pointer: 직전 최대 15개 frame과 prompt. 위치 인코딩은 frame 차이 / 15.
- 시간 인코딩: 실제 나이가 아니라 slot 위치로 정해진다. **나이가 7 이상인 메모리를 위한 인코딩은 없다.**

## 2. PART A — 메모리 고갈 원인 분리 (DAVIS pigs obj2, 이전 MPS 실행의 skip frame {46,47,48,62–67}을 그대로 사용)

| 조건 | 평균 J&F | J<0.5 frame | J<0.1 | 최장 실패 연속 | 다른 돼지 포착 (> 0.3) | 재진입 후 최소 spatial / pointer 수 |
|---|---|---|---|---|---|---|
| A0 native | 0.907 | 1 | 0 | 1 | 0 | 6 / 15 |
| **A1 spatial + pointer 모두 제거** | 0.838 | 12 | 5 | **11** | **11 frame (최대 0.98)** | 0 / 9 |
| **A2 spatial만 제거** | **0.812** | **19** | 4 | **11** | **11 frame** | 0 / 15 |
| **A3 pointer만 제거** | 0.902 | 3 | 0 | 3 | 0 | 6 / 9 |
| **A4 spatial backfill (pointer 제거)** | **0.909** | **0** | 0 | 0 | 0 | 6 / 9 |
| A5 spatial backfill (pointer 유지) | 0.876 | 7 | 0 | 6 | 0 | 6 / 15 |

**backfill 규칙:** 비어 있는 spatial slot을, 제거되지 않았고 아직 쓰이지 않은 가장 최근 과거 메모리로 채운다.
- 후보 범위: 직전 L = 32 frame
- 제약: 미래 frame 금지, 같은 메모리 중복 금지
- 활성 메모리 수: native와 같게 유지
- 시간 위치: 실제 나이 기준(나이 ≥ 7이면 가장 먼 slot t_pos = 1)
- 사용된 메모리의 최대 나이: 12 frame

해석:
1. **CUDA에서도 원래 실패가 재현된다**(A1). 대상이 다른 돼지와 합쳐지고, J가 0.01까지 떨어진다.
2. **spatial memory만 제거해도 같은 실패가 생긴다**(A2). **pointer만 제거하면 거의 영향이 없다**(A3). 따라서 실패의 주원인은 **spatial memory 고갈**이다.
3. **단순 backfill이 정확도를 완전히 복원한다**(A4: 실패 0, native보다 약간 높음).
4. A5(backfill + pointer 유지)와 A2에서 frame 58–63에 저하가 나타난다. skip 46–48 이후 궤적이 이미 달라졌기 때문으로, pointer 효과로 단정할 수 없다(혼돈적 발산).
5. 원인은 **메모리 부재**다. 오래된 정상 메모리로 채우기만 해도 해결되므로, 별도의 '품질 선택' 없이 개수만 맞추면 충분했다.

**한계:** 우리가 일부러 만든 고갈, 한 사례에 대한 결과다. 일반적인 개선으로 해석하지 않는다.

## 3. PART B — 메모리 하나의 반사실적 효과 (pre-state 동일)

**설계:**
- 사전 고정: Group A(알려진 사례) 5개 sequence + Group B(hash 대조군) 7개 sequence, sequence당 hash로 고른 6 frame. 총 **72 frame, 2,160개 즉시 개입**.
- 각 frame에서 native 상태를 그대로 둔 채 그 frame의 예측만 다시 계산했다(저장하지 않음).
- **즉시 효과:** B1 하나 제거, B2 가장 오래된 후보로 교체, B3 무작위 교체(sha256 seed), B4 사용 안 된 가장 최근 후보로 교체, B5 위치만 변경(같은 메모리를 t_pos = 1로).
- **rollout:** 같은 pre-state에서 dict를 복제해 10 frame을 진행했다. 변형은 native / LOO 가장 최근 / LOO 가장 오래된 / B4 / B3.
- 상태 복제 검증: 같은 개입을 두 번 하면 같은 결과가 나오고, 원본 상태는 변하지 않으며, native 분기는 실제 진행과 같다(테스트).
- 효과 판정 기준 |ΔJ| > 0.01은 결과를 보기 전에 정했다.

| 그룹 | 개입 | n | 도움 / 해 / 무시 | mask가 바뀐 비율 | 평균 ΔJ [sequence bootstrap CI] | 평균 \|ΔJ\| | 최대 / 최소 ΔJ |
|---|---|---|---|---|---|---|---|
| A 알려진 사례 | B1 하나 제거 | 180 | 5 / 6 / 169 | 99% | +0.0011 [−0.0006, +0.0030] | 0.0038 | +0.280 / −0.036 |
| | B2 가장 오래된 후보 | 180 | 5 / 9 / 166 | 100% | +0.0008 [−0.0004, +0.0022] | 0.0044 | +0.323 / −0.036 |
| | B3 무작위 | 180 | 8 / 11 / 161 | 99% | +0.0010 [−0.0002, +0.0025] | 0.0051 | +0.296 / −0.081 |
| | B4 최근 대체 | 180 | 5 / 4 / 171 | 100% | +0.0015 [−0.0001, +0.0042] | 0.0032 | +0.292 / −0.023 |
| | B5 위치만 | 180 | 1 / 2 / 177 | 82% | +0.0000 | 0.0008 | +0.011 / −0.022 |
| B 대조군 | B1 | 252 | 2 / 4 / 246 | 100% | −0.0002 [−0.0009, +0.0006] | 0.0018 | +0.045 / −0.035 |
| | B2 | 252 | 4 / 5 / 243 | 100% | −0.0002 | 0.0025 | +0.046 / −0.035 |
| | B3 | 252 | 3 / 4 / 245 | 100% | −0.0001 | 0.0022 | +0.047 / −0.038 |
| | B4 | 252 | 1 / 7 / 244 | 100% | **−0.0007** [−0.0012, −0.0002] | 0.0020 | +0.043 / −0.038 |
| | B5 | 252 | 1 / 0 / 251 | 83% | +0.0003 | 0.0010 | +0.046 / −0.005 |

- **B1–B4 1,728개 개입:** 도움 33(1.9%), 해 50(2.9%), **무시 1,645(95.2%)**.
- 메모리를 바꾸면 mask는 거의 항상 달라지지만, 정확도 변화는 대부분 무시할 수준이다.
- 효과가 있는 경우는 어려운 frame(native J < 0.7)에서 7.7%로, 쉬운 frame의 3.0%보다 많다.
- **가장 큰 효과**(+0.28–0.32)는 모두 pigs frame 66에서 **나이 1인 메모리(frame 65, native J 0.57)를 빼는** 개입이다. 무엇으로 대체해도 비슷하다. 따라서 '좋은 대체 메모리를 고른 효과'가 아니라 '나쁜 메모리를 뺀 효과'다.
- **위치 인코딩만 바꾼 효과(B5)는 가장 작다.** 교체 효과는 대부분 내용 또는 메모리 부재에서 나온다.
- **oracle 상한 (사후 GT로 frame마다 최선의 개입을 고른 값, 실제로 쓸 수 없음):** frame당 평균 +0.015(A), +0.005(B). +0.01을 넘는 frame은 30개 중 4개, 42개 중 4개다.
- **rollout (10 frame):**
  - 즉시 효과는 거의 전파되지 않는다. 예: A의 LOO 가장 최근은 즉시 +0.0085 → 10 frame 평균 +0.0013.
  - pigs 66의 +0.28은 다음 frame에서 native도 스스로 회복해 사라진다.
  - 실패 frame 수는 변하지 않는다(B: 120 → 120, A: 2 → 2–3).

## 4. PART C — 메모리 효용을 추론 중에 예측할 수 있는가

- 사전 기준(효과 있는 개입 ≥ 30, sequence ≥ 4)은 충족했다(83개, 11개 sequence). 하지만 하나 제거(LOO)에서 양성은 '유용' 10건(7개 sequence), '해로움' 7건(4개 sequence)뿐이다.
- 학습 모델은 만들지 않았고, 단순 1차원 점수만 평가했다. 신뢰구간은 sequence 단위 cluster bootstrap이다.

| 라벨 | 최신일수록 높은 점수 | 메모리의 predicted IoU | 메모리의 객체 점수 |
|---|---|---|---|
| 제거하면 나빠짐 (유용한 메모리) | **0.74** [0.55, 0.92] | 0.26 | 0.40 |
| 제거하면 좋아짐 (해로운 메모리) | 0.44 | **0.75** (IoU가 낮을수록) [0.58, 0.97] | 0.42 |

- 메모리 품질 특징은 이전 MPS native 실행의 신호를 썼다(메모리가 기록될 때 얻을 수 있는 값이므로 인과적이다. CUDA와의 차이는 ≤ 20 px).
- 유용한 메모리는 **최근성**으로 구분된다. 이는 SAM 2의 기본 동작 그대로다.
- 해로운 메모리는 **낮은 predicted IoU**로 어느 정도 구분된다. 이는 SAM2Long, MoSAM, SAMURAI의 품질 게이트와 같다.
- 결론: 새로운 '효용 예측기'가 기존 단순 정책보다 낫다는 근거가 없다.

## 5. 추가 진단 — 후보 mask 선택 문제인가 (사후 GT, 실제로 쓸 수 없음)

- PART B 12개 sequence를 native로 다시 실행하고, SAM 2의 다중 mask 후보 3개를 GT와 비교했다.
- 실패 frame(J < 0.5) 133개(6개 sequence) 중 **다른 후보가 J ≥ 0.5인 경우는 7개뿐**이다. 평균 개선 여지는 +0.025다.
- 실패의 대부분은 두 sequence에서 나왔다: paragliding-launch obj3(J 0.10–0.33, 79 frame), kite-surf obj1(0.28–0.53). 둘 다 **prompt frame에서부터** J가 낮다(0.28, 0.66). 즉 얇은 구조물 같은 분할 단위·해상도 문제이지, 메모리나 후보 선택 문제가 아니다.
- 이번 자연 데이터에서는 '잘못된 대상/후보 선택'이 주된 실패 유형이라는 근거도 없다. 대상 합침·바뀜은 우리가 유도한 고갈(A1/A2)과 M4.5 스트레스 조건에서만 관찰되었다.

## 6. 선행연구 비교 (조사 에이전트 결과 + 저장소 확인)

| 방법 | 제어하는 것 | 학습 | online GT | 고정 예산 | 메모리별 실제 효용 측정 | 우리와 겹치는 부분 |
|---|---|---|---|---|---|---|
| SAM2Long (ICCV'25) | 쓰기/선택: IoU ≤ 0.3 또는 occlusion ≤ 0 frame을 건너뛰며 과거로 탐색, tree | 없음 | 없음 | 예 | 아니오 (휴리스틱) | **backfill과 사실상 같음** (신뢰할 수 없는 최근 frame 대신 과거 frame) |
| SAMURAI (arXiv'24) | 점수 기준 메모리 선택 + Kalman | 없음 | 없음 | 예 | 아니오 | backfill (암묵적) |
| MoSAM (arXiv'25) | IoU > 0.7·occlusion 기준 선택 + motion | 있음 | 없음 | 예 | 아니오 | backfill, 품질 게이트 |
| SENTRY (ECCV'26) | 쓰기 검증 (tracklet 일관성) | 없음 | 없음 | 명시 없음 | 아니오 | 쓰기 게이트 |
| DAM4SAM (CVPR'25) | 쓰기: RAM + distractor 메모리 | 없음 | 없음 | 예 | 아니오 | 장기 보존 |
| SAM2Dual (arXiv'26) | 단기 + 장기 bank 읽기 융합 | 없음 | 없음 | **아니오 (약 2배)** | 아니오 | 장기 메모리 |
| TSMS-SAM2 (arXiv'25) | 단기/장기 분리 + 유사도 기반 정리 | 있음 | 없음 | 예 | 아니오 | 보존·정리 |
| Lean-SAM2 (arXiv'26) | token 정리 + 'insurance bank' (occlusion > 0.7 보관) | 일부 | 없음 | 예 | 아니오 (attention 점수) | 보존 |
| SAM2RL (arXiv'25) | RL로 skip/교체, 보상 = GT IoU | 영상별 학습 | **있음 (학습 시)** | 예 | 정책 학습 (측정 아님) | 교체 |

- **메모리 하나의 반사실적 효용 측정**, **spatial / pointer 분리**는 공개된 사례를 찾지 못했다(부재를 증명하지는 못함).
- **backfill**은 SAM2Long, SAMURAI, MoSAM이 암묵적으로 이미 수행한다. 고갈을 일부러 만든 통제 비교라는 형식만 새롭다.

## 7. 실행 시간과 GPU 메모리 (RTX 2080, FP32)

- native: 약 **0.20 s/frame** (MPS 약 1.0 s 대비 5배). PART A 조건당 79 frame에 16–17 s. 제어 계획을 써도 overhead는 측정 오차 수준이다(0.188–0.203 s).
- PART B 전체(12 sequence, 개입 2,160 + rollout 3,960 frame): 1,287 s.
- CUDA 최대 할당 2.84 GB (8 GB 중).

## 8. 질문별 답 (측정 / 가설 / 한계 구분)

- **Q1. 해로운 게이트 결과의 원인은?**
  - 측정: **spatial memory 고갈**이 주원인이다(A2 ≈ A1, A3 ≈ native). 한 사례 결과다.
- **Q2. backfill이 정확도를 복원하는가?**
  - 측정: 그렇다(A4 실패 0).
  - 다만 우리가 일부러 만든 고갈에서만 확인했다. 자연 데이터에서 고갈이 실제로 일어나는지는 확인하지 않았다. M4.5의 긴 누락 실패는 메모리가 차 있는 상태에서 일어났다.
- **Q3. 메모리 교체가 재현 가능한 개선을 주는가?**
  - 측정: 아니다. 95%가 무시할 수준이고, 도움과 해가 비슷하며, 평균 효과의 CI가 0을 포함하거나 약간 음수다. rollout에서는 사라진다.
- **Q4. 도움과 해가 되는 메모리를 추론 중에 구분할 수 있는가?**
  - 측정: 유용함은 최근성(native 동작)으로, 해로움은 낮은 predicted IoU(기존 게이트)로 일부 구분된다.
  - 표본이 매우 작다(양성 7–10건).
- **Q5. 기존 단순 정책으로 같은 결과를 얻는가?**
  - 측정 + 문헌: 그렇다. 최근성 + IoU 게이트 + 과거 frame backfill은 SAM2Long, SAMURAI, MoSAM의 조합이다.
- **Q6. 새 메모리 관리 알고리즘을 개발할 근거가 있는가?**
  - 아니다(현재 증거로는).
  - 이 데이터에서 자연적인 실패는 메모리 조작으로도(PART B), 후보 선택으로도(5절) 거의 고쳐지지 않는다.

**한계:**
- 12개 sequence / 72 frame이고, 효과가 있는 사례가 소수 sequence에 몰려 있다.
- DAVIS가 SAM 2 학습 데이터와 겹칠 수 있다.
- 자연 실패가 적다. MOSE 계열 데이터를 쓰지 못했다.
- PART A는 한 사례다.
- PART C의 품질 특징은 MPS 실행의 로그다.

## 9. 권고

**결정: INSUFFICIENT_EVIDENCE**

| 경로 | 근거 | 판정 |
|---|---|---|
| A. 메모리 효용 연구 | 교체 효과의 95%가 무시할 수준. 예측 가능한 부분은 최근성과 IoU로, 기존 방법과 같다 | 지지하지 않음 |
| B. 메모리 고갈·보존 연구 | backfill이 *우리가 유도한* 고갈을 완전히 고친다. 하지만 자연 실패가 고갈 때문이라는 근거가 없고, 메커니즘은 SAM2Long, SAMURAI, MoSAM에 이미 있다 → 연구 기여보다는 공학적 교정에 가깝다 | 근거 부족 |
| C. 대상·후보 선택 연구 | 자연 실패 133 frame 중 다른 후보로 고칠 수 있는 것은 7개. 대상 합침·바뀜은 유도된 조건에서만 관찰 | 근거 부족 |

**필요한 최소 추가 실험:**
1. 실패가 풍부한 자연 영상(MOSEv2 train 전체 58 GB, 다운로드는 사용자 승인 필요)에서 **자연 실패 구간 50개 이상 / 영상 20개 이상**을 사전 고정한 표본으로 확보한다.
2. 그 실패를 **유형별로 분류**한다(대상 합침·바뀜 / 얇은 구조·해상도 / 가림·재등장 / 메모리 고갈). 사전 고정 기준을 쓴다.
3. 같은 PART B 개입과 후보 oracle 진단을 **실패 구간 내 frame**에 적용한다.
4. 어떤 유형이 메모리 개입에 반응하는지 확인한 뒤 경로를 정한다.

## 10. 재현

```bash
uv run python scripts/m5pre.py partA        # outputs/milestone5pre/partA/
uv run python scripts/m5pre.py manifest     # 이미 고정되어 있으면 변경 거부
uv run python scripts/m5pre.py partB        # 재개 가능 (sequence별 완료 기록)
uv run python scripts/m5pre.py analyze      # reports/partB_*.csv, partB_report.json
uv run python scripts/m5pre.py candidates   # reports/candidate_oracle_*.{csv,json}
uv run python scripts/run_sam2_baseline.py --device cuda --out outputs/milestone5pre/cuda_m2_baseline
uv run pytest                               # 221 passed
```

**산출물** (`outputs/milestone5pre/`, manifest만 git에 포함):
- `manifest.json`
- `reports/cuda_vs_mps_equivalence.json`
- `cuda_m2_baseline/`
- `partA/{summary.csv, per_frame.csv, partA_report.json}` (frame별 spatial·pointer 출처 frame ID 포함)
- `partB/*.json`
- `reports/partB_{interventions,rollouts,families,rollout_summary}.csv`, `partB_report.json`
- `reports/partC_loo_features.csv`
- `reports/candidate_oracle_{frames.csv,report.json}`
- `partB.log`
