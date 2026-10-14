# M7 선행연구와 새로움

**원자료:**
- [`PRIOR_ART_raw.md`](PRIOR_ART_raw.md): S, M2 계열과 장기 추적
- [`PRIOR_ART_raw2.md`](PRIOR_ART_raw2.md): F, M, T, R, O 계열
- M6.1 [`../m61/PRIOR_ART_AND_NOVELTY_REVIEW.md`](../m61/PRIOR_ART_AND_NOVELTY_REVIEW.md)

**판독 수준과 주의:**
- **READ:** 방법 절을 읽었지만 요약기를 거쳤다. Matthews 2003만 PDF 원문을 직접 읽었다.
- **ABSTRACT / SNIPPET:** 초록이나 검색 요약만 읽었다.
- 인용 전에 수치를 원문과 다시 대조해야 한다.
- VOTS2025 참가 추적기 설명은 여전히 읽지 않았다. 'crop + 전체 frame + 검증'의 직접 선례가 있을 가능성이 가장 높은 곳이다.

## 1. 결정 대상 알고리즘별 비교 (연산 수준)

| 알고리즘 | 가장 가까운 연구 (상태, 판독) | 겹치는 연산 | 남는 차이 | 새로움 위험 |
|---|---|---|---|---|
| **SGGL (최종)** | CTU 학사논문 Rozumnyi 2025 (학위논문, READ): SAM2.1-L 병렬 crop 창 + 전체 frame, IoU 일치 gate | 두 SAM 2 상태 병렬, frame별 선택 | 자기 신호 gate(crop score, 내부 경계 접촉), 객체 상대 고정 배율 가상 카메라, P5 prompt 메모리, mask VOS 크기 분층 평가. 측정 차이: CTU식 gate 대비 +0.008 (M6) | 높음 |
| | AdaSAM2 (Sensors 2026, 심사, READ): RL로 crop 배율 선택 | crop 배율 적응 | 학습 없음, 전역 상태 병렬 유지 | |
| | Alpha-Refine (CVPR'21), f-BRS Zoom-In | 객체 상대 고정 crop | SAM 2 메모리 VOS에서의 측정 | |
| SGGL-V 거부권 (선택 사항) | StreamDAM, DAM4SAM (단일 view 존재 gate), LTMU / GlobalTrack / Siam R-CNN (검증기) | 존재 판단으로 출력 억제 | 다른 view(전역)의 존재 판단을 쓴다 | 중간 |
| M3-vel (INCONCLUSIVE) | DaSiamRPN (실패 시 search 확장), KeepTrack, 일반적인 운동 모델 search region | 예측 운동으로 search 영역 배치 | SAM 2 가상 카메라의 중심 제어에 적용 | 높음 |

## 2. 기각된 계열의 선행연구 위치

| 계열 | 가장 가까운 연구 | 위험 | M7 결과와의 관계 |
|---|---|---|---|
| S1 상태 인식 재획득 | SatSAM2 (preprint, READ): reset 뒤 12 frame score > 0.3 지속 요구. KeepTrack (ICCV'21, READ) | 높음 | 측정: 0 / −0.0005. 우리 데이터의 거짓 재획득은 1 영상 |
| S3 신원 검증 | SAMOSA (preprint, READ): SAM 2 Hiera 특징 mask pool 코사인 + prototype 고정. LTMU, SPLT, TLD | 높음 | 측정: 효과 없음. 방해물이 외형상 비슷하면 신원 검증은 원리상 실패 |
| F1–F4 결합 | Tao 2020 (학습형 픽셀 gate), CGFSeg (LSVOS'25), PointRend, Mask Transfiner, SAM2Refiner (ICCV'25), MiVOS (추가 / 삭제 분리) | 높음 (F3는 중간) | 측정: 상한 +0.002–0.006, 전부 음수 |
| M1 보호 메모리 반사실 | **Matthews, Ishikawa & Baker 2003/2004 "Template Update Problem" 전략 3 (직접 선례, PDF READ).** 장기 추적 검증기 | 중간 | 측정: 정보 없음 (26 vs 361 frame) |
| M2 메모리 허용 | DAM4SAM, SAMURAI, SAM2Long, SENTRY, Cutie | 높음 | 인과 근거가 없어 평가하지 않음 |
| R1 해상도 민감도 | Rottmann & Schubert 2019 (CVPRW, ABSTRACT): 중첩 crop 예측 분산으로 오분할 검출 | 낮음–중간 | 진단 결과: 맞는 / 틀린 후보를 구별하지 못함 |
| T1–T4 특징 결합 / attention | SAM2Refiner (frozen SAM 2, 확대 사분면 token과 전역 token cross-attention), Cutie object query, GLNet, Flamingo gate | 구성요소는 각각 높음, 조합은 중간 | 측정: 일반화 실패 (18개 학습 영상) |
| O1 사건 기반 실행 | Viola–Jones cascade, DVSNet, SPLT, LTMU | 높음 | 비용 분석만 |

## 3. 주장할 수 있는 것과 없는 것

**주장할 수 없음 (사용자 목록 그대로):**
두 SAM 2 상태 실행, 일반 crop 추적, 단순 마스크 평균, 일반 cross-attention, 목표 template, object token, 메모리 검증, 시간 지속성, 사건 기반 계산.

**새 알고리즘이라고 할 근거:**
- M7에서 **실험으로 뒷받침되는 새 알고리즘 차이는 만들지 못했다.**
- SGGL의 구체적 연산(자기 신호 gate + 객체 상대 고정 배율 가상 카메라 + P5)은 CTU와 Alpha-Refine 계열 위의 **공학적 변형**이다.

**연구 기여로 정직하게 내세울 수 있는 것 (측정 기반, 탐색적):**
1. **크기 분층 mask VOS에서 전역–국소 SAM 2 추론의 효과 측정**
   - P5 대비 개발 세트 +0.011, 새 합성 세트 +0.017 (22개 오름 / 1개 내림).
   - 같은 계산량의 flip 앙상블 대비 +0.018, CTU식 gate 대비 +0.008 (M6).
2. **'선택 대 결합 대 후보 생성' 상한 분해**
   - {G, L} 선택 +0.018, 결합 +0.002 추가, 성분 단위 +0.006 추가, 대체 상태 후보(2× crop, 보호 / 신뢰 메모리) +0.015 추가.
3. **실패 메커니즘 분석**
   - 가장 큰 손실은 '두 view가 함께 부재를 말함'이다.
   - 거짓 재획득은 메모리 오염이 아니라 외형 모호성이다(보호 메모리도 방해물을 잡는다).
   - G가 작은 대상을 놓치는 원인은 화소 해상도가 아니라 특징 격자 점유다(해상도 개입 probe).
4. **음성 결과 (재현 가능한 코드와 함께):**
   - 메모리 반사실(Matthews 전략의 SAM 2판)은 정보를 주지 않는다.
   - 18개 영상 규모의 frozen-feature 결합 head는 일반화하지 않는다.
