# M6 선행연구 비교 (2026-10-13)

**출처와 판독 수준:**
- 원자료는 [`PRIOR_ART_NOTES_raw.md`](PRIOR_ART_NOTES_raw.md)에 있다. 웹 조사 agent가 작성했고 숫자는 원문에서 그대로 옮겼다.
- 판독 수준 표시:
  - **READ:** 방법 절을 원문으로 읽음
  - **ABSTRACT:** 초록만 읽음
  - **UNVERIFIED:** 직접 확인하지 못함
- 이 문서는 세 트랙과 최종 후보를 위 원자료와 대조한 것이다.

## 1. 트랙별 요약

| 트랙 | 가장 가까운 연구 (판독 수준) | 우리 방법과 겹치는 부분 | 남는 틈 |
|---|---|---|---|
| **A 소형 대상 확대 (virtual camera)** | **CTU 학사논문 Rozumnyi 2025** (READ) / **AdaSAM2, Sensors 2026** (READ, EuropePMC) / Lean-SAM2 (READ, 목적이 속도) | **매우 높음.** CTU는 다음을 이미 했다: 학습 없는 SAM2.1 crop 창(직전 마스크 중심), **고정 크기 창 + 전체 frame 병렬 + 인과적 선택 규칙**(우리 A2와 같은 구조). AdaSAM2는 다음을 이미 했다: RL로 배율 선택, 추적 실패 시 crop 확대 | (1) 첫 frame 마스크로 한 번 정하는 **객체 상대 고정 배율** (2) crop 좌표계에서 메모리 배율 일관성을 **명시적으로 시험** (3) VOS(J&F, 크기별) 평가 (4) 언제 해가 되는지에 대한 분석. 두 연구 모두 메모리 처리를 서술하지 않음 |
| **B 가림·재획득** | DAM4SAM (READ) / StreamDAM (READ) / SAMOSA (요약) / SENTRY, SAM2Long, SeC, SAM2Dual (ABSTRACT) / SAM3 재prompt PVUW 보고 (ABSTRACT) | **높음.** 다음이 모두 이미 있다: 빈 마스크를 메모리에 쓰지 않기, 학습된 존재 head가 출력 억제와 재탐지를 제어(StreamDAM), 첫 frame prototype으로 후보 재검증(SAMOSA), 외부 detector 전역 탐색 | 학습 없이 frozen SAM2 신호만으로 '부재 vs 실패'를 가르는 분류기. 그러나 M6에서 SAM2 object score AUROC가 세트 단위 0.994–0.9999, 객체 내 최저 0.88로 이미 높아 개선 여지가 작다(GT 빈 frame은 9개 객체에 편중) |
| **C 세부·해상도** | SAM2Refiner, ICCV'25 (READ) / HQ-SAM 2 (README) / MGD-SAM2 (READ, 이미지 전용) | **높음.** frozen SAM2 + 학습된 고해상도 정제 head(1024 마스크)와 영상 수치가 이미 있다 | 영상에서 성분 단위 소형 구조 재현율 평가. 학습 없는 경로(=A)로 세부를 개선하는 것 |

## 2. 최종 후보와 가장 가까운 3개 연구

- **최종 후보:** 「자기 신호 gate 전역–국소 추론(SGGL)」 (잠정)
- 근거 문서: [`FINAL_RESEARCH_TOPIC_DECISION.md`](FINAL_RESEARCH_TOPIC_DECISION.md) §F, [`PROPOSED_ALGORITHM_SPEC.md`](PROPOSED_ALGORITHM_SPEC.md)

| 연구 | 그 연구가 한 것 | 정확한 차이 | 독창성을 위협하는 겹침 |
|---|---|---|---|
| **CTU 학사논문** (Rozumnyi, Matas 지도, 2025, 미심사) | **crop:** SAM2.1-L crop 창(직전 bbox × N, 매 frame 변함) **병렬:** H/4, H/2 고정 창 + 전체 frame 추적기 **수용 규칙:** IoU(crop, 전체) > γ 그리고 bbox/창 ≤ 0.1일 때만 crop 채택 **결과:** VOT2022ST EAO 0.691 → 0.720 | **수용 gate:** crop 상태의 object score + 내부 경계 접촉 (전체 frame 예측과 무관) **배율:** 객체 상대 고정 1개 **평가:** mask VOS(J&F, 크기 분층, Hiera-T) **M6 측정:** 같은 틀에서 CTU gate(A2c) 대비 **+0.008 [+0.001, +0.014]**, 작은 객체 5개 우세 / 0개 열세, crop 채택률 0.89 vs 0.57 | **높음.** 병렬 + 선택 골격이 같다. 차별점은 gate 하나이고 효과가 작다 |
| **AdaSAM2** (Sensors 26:5819, 2026, 심사) | **기반:** SAMITE **배율:** Double-DQN이 이미지 비율 배율을 초기화와 '불안정' 시점에 선택 **복구:** 실패 시 crop 점진 확대 **결과:** VOT 계열 Success +1 내외 | **배율:** 학습 없는 객체 상대 규칙 vs RL 정책(학습 필요) **복구 방식:** 전역 상태를 병렬로 유지 vs crop 확대 **평가:** mask VOS vs box | 중간 |
| **SAM2Refiner** (ICCV'25) | frozen SAM2에 1024 해상도 다중 scale 정제 모듈을 학습. LVOS J&F +1.6–3.7 | 학습 없음, 입력 배율 vs 출력 쪽 학습 head | 중간 (같은 문제, 다른 메커니즘. 함께 쓸 수 있음) |

## 3. 독창성 판정 (정직한 평가)

**새롭지 않은 것:**
- crop/zoom 입력 (CTU, AdaSAM2, 고전 SOT의 search region)
- 전역 + 국소 병렬 후 선택 (CTU)
- 실패 시 복구 (AdaSAM2)

**M5.6 제안서의 새로움 주장 철회:** '고정 배율로 메모리 일관성 유지'는 M6 A1 vs A1v에서 근거가 나오지 않았다(결론 불가).

**남는 주장 (모두 작거나 측정 수준이다):**
1. 극소 대상에서 '일치 gate'가 crop을 버리는 문제와 자기 신호 gate의 개선 (+0.008, 개발 세트, 탐색)
2. mask VOS에서 크기 의존 효과 (시사적)
3. 과도한 확대가 정지 영상 분할 자체를 망친다는 측정 (oracle 0.819 → 0.782)
4. SAM 2 예측 IoU가 확대 입력에서 부풀려진다는 측정 (S1 실패 원인)

**결론:** 기여는 '새 방법'이라기보다 **'기존 아이디어의 VOS 재검증 + gate 개선 + 상한 분석'**에 가깝다. 극소 대상 데이터에서 +0.03 이상과 CTU 대비 우위가 확인되어야 독립 연구로 성립한다.

## 4. 확인하지 못한 것

- SAMURAI, SAM2Long, SENTRY, SAM2Dual, SeC는 초록만 읽었다.
- HQ-SAM 2의 학습 방식은 확인하지 못했다.
- AdaSAM2의 backbone 크기와 RL 학습 데이터는 원문에 없다.
- FocusTrack은 검색 요약으로만 접했다.
- 다음 단계에서 **CTU 원형(H/4, H/2 병렬)을 재현하고, 공개되어 있으면 저자 코드로 메모리 처리를 확인해야 한다** (로드맵 단계 4).
