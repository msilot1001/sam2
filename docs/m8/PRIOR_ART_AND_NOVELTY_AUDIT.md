# M8 선행연구와 새로움 감사

- **원자료:** [`PRIOR_ART_raw.md`](PRIOR_ART_raw.md). 백그라운드 agent가 2026-10-14에 웹을 조사해 작성했고, 인용마다 판독 수준(READ / ABSTRACT / UNVERIFIED)을 적었다.
- 이 문서는 그 요약에 M8 실험 결과를 연결한다. §3의 '주장 가능 범위'는 결과에 따라 정해진다.

## 1. 후보별 가장 가까운 선행연구 (연산 수준)

| 후보 | 가장 가까운 연구 | 겹치는 연산 | 남는 차이 | 위험 |
|---|---|---|---|---|
| A_cos 전역 재탐지 → point prompt | PerSAM, Matcher (ICLR'24), SAMITE, Track Anything | 마스크 pooling prototype → 코사인 map → peak 점 → SAM prompt | 동시 부재에만 발동. 실행 중인 SAM 2 상태에 prompt | 높음 |
| B_att 대조 token attention | VideoMatch (ECCV'18, READ), FEELVOS, STM/XMem/Cutie readout | frame-0 fg/bg 집합 대비 픽셀 매칭 → softmax 사후확률 | frozen SAM 2 특징, 재탐지 전용 | 높음 |
| H_s8 stride-8 매칭 | SAM2Refiner, FEELVOS | 고해상도 특징 수준 매칭 | A_cos의 해상도 ablation일 뿐 | 주장할 것 없음 |
| G_neg 방해물 peak에 음성 점 | PerSAM, SAM-PT, FSS-meets-SAM3 (2026, 자동 음성 점이 일관되게 해롭다고 보고) | 자동 음성 점 | 2차 prototype peak(방해물)에 음성 점을 둔다 | 중간 |
| D_ref frame-0 메모리 재prompt | DAM4SAM, SAM2Refiner | 초기 frame 메모리 고정 | 없음 | 높음 (진단 전용) |
| A/B_crop heatmap 유도 crop 배치 | DaSiamRPN local-to-global, SPLT, LTMU, AdaSAM2 | 전역 위치 추정 → 국소 탐색창 배치 | SAM 2 특징 heatmap이 SGGL crop 상태를 옮김 | 중상 |
| E_beam 속도 beam | SAM2Long, SAM-PD, SAMURAI/MoSAM | 다중 가설, 속도 외삽 | 마스크/메모리가 아니라 입력 crop 위치에 대한 beam | 중간 |
| C_flow 광류 box | MoSAM (흐름으로 warp한 마스크 → box prompt), SAM-PT, FlowSAM | 광류/점 추적 → SAM 2 prompt | 동시 부재에서 crop view 이동에만 사용 | 높음 |
| FLOWVEL 광류 crop 제어 | MoSAM, SAMURAI(Kalman), DaSiamRPN | 운동 모델로 탐색 영역 이동 | 출력 마스크 중심 차분 대신 dense flow 변위 | 높음 |
| F_oclr 점유율 crop | ZoomTrack Eq.1, Siamese context factor, AdaSAM2 | 대상 크기로 crop 크기 결정 | stride-16 cell 단위 표현 | 높음 |
| **F_cpfr foveation** | **ZoomTrack (NeurIPS'23, READ) — 직접 선례**, FOVEA (ICCV'21), LZU (CVPR'23) | 이전 box 중심의 분리형 비균일 resize, 배율 제어, 시야 보존(crop 없음), 역변환, 기성 추적기에 test-time 적용 | 메모리 전파형 VOS(SAM 2)에서 매 frame 다른 warp 아래 메모리를 쓰고, 마스크를 역변환 | 연산 수준 높음 |
| R3 set-attention ranker | KeepTrack (ICCV'21, READ) | score map 후보 간 attention + no-match slot | 후보가 SAM 2 rescue 마스크, frame-0 token 조건 | 중상 |
| 메모리 기록(admit) | DAM4SAM, SAMURAI, SAM2Long, SENTRY, StreamDAM, Track Anything | gate된 메모리 쓰기, 외부 마스크 기록 | 동시 부재 rescue의 기록 | 높음 |

**VOTS2025 (보고서 전체 READ):**
- 유효 참가 15개 중 80%가 SAM 2 기반이다.
- crop view와 전체 frame SAM 2 view를 결합한 참가자는 없었다.
- 재탐지 방식은 메모리, Grounding DINO / Qwen-VL, CoTracker 점 prompt였다.
- 동시 부재 trigger에 가장 가까운 것은 DAM-RMemAOT의 '모든 병렬 추적기 동시 실패 시 복구'다. AOT 기반이고, 복구는 형태학 연산이다.

## 2. 첨부 보고서와 다른 점

보고서는 "context-preserving foveated rescue와 정확히 같은 구성은 확인하지 못했다"고 했다. 이번 조사에서 **ZoomTrack**이 연산 수준의 직접 선례로 확인되었다. 분리형 비균일 확대, 시야 보존, 역변환, 기성 추적기 test-time 적용이 모두 겹친다. 남는 차이는 두 가지뿐이다.
- 마스크 메모리를 전파하는 SAM 2 안에서 쓰는 점
- 마스크를 역변환하는 점

따라서 CPFR의 새로움은 '확대 + 문맥 보존' 자체가 아니다. 그것을 메모리 전파 모델에 넣었을 때 생기는 문제와 그 해법이어야 한다.

## 3. 주장 가능 범위 (결과 반영)

(실험 결과 후 작성: `FINAL_PORTFOLIO_ASSESSMENT.md` §5에도 요약)
