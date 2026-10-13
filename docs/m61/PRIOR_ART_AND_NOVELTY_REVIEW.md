# M6.1 선행연구와 새로움 검토

**출처와 판독 수준:**
- 원자료: [`PRIOR_ART_raw.md`](PRIOR_ART_raw.md) (agent R, M6.1). M6 원자료 [`../m6/PRIOR_ART_NOTES_raw.md`](../m6/PRIOR_ART_NOTES_raw.md)도 참조했다.
- 판독 수준:
  - **READ:** 방법 절을 읽었다(요약 도구 경유가 포함되므로 인용 전에 수치를 재확인할 것).
  - **ABSTRACT** / **SNIPPET:** 초록이나 검색 요약만 읽었다.

## 1. 결선 후보별 가장 가까운 연구

### 1.1 현직(incumbent): 자기 신호 gate 전역–국소 SAM 2 (A2x)

| 연구 | 상태 | 메커니즘 | 이미 알려진 구성요소 | 우리와의 차이 |
|---|---|---|---|---|
| CTU 학사논문 (Rozumnyi 2025) | 학위논문, 미심사, READ | SAM2.1-L crop 창(H/4, H/2)과 전체 frame 병렬. crop 채택 조건: IoU(crop, 전체) > γ, bbox/창 ≤ 0.1 | **병렬 전역–국소 + frame별 선택** | 수용 gate가 crop 자신의 신호(object score, 내부 경계)이고 객체 상대 고정 배율을 쓴다. M6 측정: CTU gate 대비 +0.008 [+0.001, +0.014] |
| AdaSAM2 (Sensors 2026) | 심사, READ | RL(Double-DQN)로 crop 배율 선택, 실패 시 crop 확대 | crop 배율 적응, 복구 | 학습 없음, 객체 상대 규칙, 전역 상태를 병렬로 유지 |
| HQTrack (VOTS2023 보고) | 미심사, READ | 추적기 마스크 box로 HQ-SAM 재분할. IoU > 0.1일 때 채택 | 추적기 출력 → SAM 재분할, 일치 gate | 확대 없음. HQ-SAM은 별도 모델 |
| SAMURAI | 프리프린트, READ | SAM2 예측 IoU + Kalman box 점수로 후보 마스크 선택 | 품질 점수 기반 선택 | 배율 간 선택이 아님. 보정 없음 |
| Alpha-Refine (CVPR'21) | 심사, READ | 추적기 box 약 2배 crop을 학습된 정제기로 처리 | 고정 객체 상대 crop | 별도 학습 모델. SAM2 메모리 없음 |

**추가 선행 계열 (M6.1 독립 검토자 지식 기반, 원문 미확인):**
- 장기 추적의 국소 탐색 + 전역 재탐지: LTMU (CVPR'20), SPLT (ICCV'19), GlobalTrack (AAAI'20), Siam R-CNN (CVPR'20)
- 작은 객체 검출을 위한 slicing 추론: SAHI
- 상호작용 교정 경로: SAM 2 자체가 교정 마스크를 메모리에 넣는다. 되쓰기(W)와 가깝다.

**새로움 위험: 높음.**
- 'crop 창 + 전역 병렬 + 선택'은 CTU에 있고, '고정 객체 상대 crop'은 Alpha-Refine과 f-BRS에 있다.
- 남는 차별점은 두 가지뿐이다.
  1. 수용 gate (자기 신호 vs 일치). 측정상 작은 차이다.
  2. mask VOS(J&F, 크기 분층)와 Hiera-T 평가.
- **M6.1 Track D:** 보정된 학습 gate로 이 선택을 더 잘하려는 시도는 실패했다(아래 §2). 그래서 '배율 인식 신뢰도 추정'을 새 기여로 주장할 근거가 오늘 밤 오히려 약해졌다.

### 1.2 도전자 B: frozen SAM 2 출력 쪽 학습 정제 head

| 연구 | 상태 | 메커니즘 | 겹침 |
|---|---|---|---|
| SAM2Refiner (ICCV'25) | 심사, READ (M6) | frozen SAM 2 + 다중 scale 특징 융합 1024 정제. 이미지로 학습. LVOS J&F +1.6–3.7 | **높음**: frozen backbone + 고해상도 정제 |
| HQ-SAM 2 | 코드 공개, README | HQ 출력 token. DAVIS J&F +1.2 (SAM2.1-L) | **높음** |
| CascadePSP (CVPR'20) | 심사, READ | 모델 무관 RGB + 마스크 정제(전역 → 국소). 경계 loss | 높음: 출력 정제기. SAM2 VOS에 적용한 보고는 못 찾음 |
| SegRefiner (NeurIPS'23) | 심사, READ | 이산 diffusion 정제 | 중간 |
| BPR (CVPR'21) | 심사, ABSTRACT | 경계 patch 정제 | 중간 |

**새로움 위험: 높음.**
- 우리 head의 입력은 SAM 2 내부 특징(stride-4/8) + decoder logit + RGB다. 메모리에는 쓰지 않고 출력에만 적용한다.
- 이것은 SAM2Refiner와 HQ-SAM 2의 '가벼운 변형'이다.
- 독립 기여가 되려면 둘 중 하나를 보여야 한다.
  - 명확히 다른 목적: 예를 들어 '정제 결과를 메모리에 다시 쓸 때의 시간 효과', 또는 작은 대상 전용 감독
  - 같은 비용에서 더 나은 결과

### 1.3 도전자 A: 추적기 prompt 국소 재분할(TPLR), 기각됨

- **가장 가까운 연구:** HQTrack, Track Anything(TAM, 메모리 되쓰기), SAM-PD, Alpha-Refine, f-BRS Zoom-In.
- **이미 알려진 것:** 구성요소는 모두 알려져 있다. '같은 SAM 2의 자기 추적기 box → 같은 SAM 2 image decoder로 확대 crop 재분할'이라는 조합은 찾지 못했다.
- **M6.1 실험 결과:** 정보 추가 +0.001, 작은 객체 −0.004로 실패했다. 메모리 없는 재분할은 부분 대상의 정의(눈 주변 vs 얼굴)를 잃는다.

### 1.4 도전자 C: 시간 교정, 검정력 부족

- **가장 가까운 연구:** ViperSAM(frozen SAM2 + Farnebäck flow + fb 일관성 + entropy gate, 학습 없음)이 거의 같은 메커니즘이다. 다만 안정성 지표만 보고하고 J&F는 없다. 이 밖에 SAM-PT, MoSAM, TAM이 있다.
- **M6.1 진단:** 4개 이상 독립 영상에서 교정 가능한 범주가 없다.

### 1.5 도전자 D: 배율 인식 보정 gate, 실패

- **가장 가까운 연구:**
  - SAMURAI: 원시 IoU + 운동 점수
  - LIT: SAM2 IoU를 '실제로 신뢰할 수 없음, 데이터셋 의존'으로 보고
  - RUAC (ICML'26): 별도 불확실도 head
  - StreamDAM: 학습된 존재 gate
- **찾지 못한 것:** 배율별 보정 + 전역/국소 선택기.
- **M6.1 실험 결과:** 보정 gate는 self gate보다 −0.0016이었다. 틈은 있지만 **효과가 없다**.

## 2. 새로움 판정 요약

| 후보 | 이미 알려진 것 | 차별점 | 측정된 차별 효과 | 판정 |
|---|---|---|---|---|
| A2x (현직) | 병렬 전역–국소 + 선택 (CTU) | 자기 신호 gate, 객체 상대 고정 배율, VOS 평가 | CTU gate 대비 +0.008 (개발, 탐색) | 점진적. 'VOS 재검증 + gate 비교 + 상한 분석'으로만 정직하게 주장할 수 있음 |
| B head | frozen 정제 head (SAM2Refiner, HQ-SAM 2) | 내부 특징 + logit 입력, 출력 전용, 저비용 | 결과는 [`EXPERIMENT_COMPARISON.md`](EXPERIMENT_COMPARISON.md) 참조 | 높은 선행연구 위험 |
| D 보정 gate | 원시 IoU gate가 널리 쓰임, LIT 비판 | 배율별 보정 | −0.0016 (없음) | 새 기여로 주장할 근거 없음 |

**주의:**
- VOTS 2023–2025 참가 추적기의 부록은 읽지 않았다. 'crop 후 SAM 재분할'의 직접 선례가 있을 가능성이 가장 높은 곳이다.
- 최종 주제를 확정하기 전에 확인해야 한다(로드맵 TASK-M7-001).
