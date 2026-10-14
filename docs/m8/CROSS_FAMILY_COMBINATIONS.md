# M8 계열 간 결합 행렬 (16개)

- 실험 전 작성. 결합은 **구성요소가 각자의 Stage 1 반증 시험을 통과했을 때만** 실행한다(사전 등록). 실행 여부와 결과는 [`COMBINATION_RESULTS.md`](COMBINATION_RESULTS.md)에 있다.
- 모든 결합은 SGGL 위에서 **트리거(두 view 동시 부재) 때만** 작동한다. 그 밖의 frame은 SGGL과 비트 단위로 같다(`tests/test_m8.py`).
- 비용은 trigger frame당 추가 SAM 2 호출 수다. encoder는 crop/foveated 영상의 backbone 1회, decoder는 memory attention + mask decoder 1회다. 2080 실측: decoder ≈ 0.05 s, encoder 포함 ≈ 0.15 s.
- 공통 시너지 ablation: 결합 X+Y마다 X 단독, Y 단독, X+Y, 계산량을 맞춘 대조를 같은 기계에서 인과 rollout한다. '시너지' 주장은 X+Y − max(X, Y) > 0이고 영상 군집 CI가 0을 넘을 때만 한다.

| # | 구성 | 각자 기여하는 정보 | 중복이 아닌 이유 | 겨냥 실패 사례 | 선행 통과 조건 | 시너지 ablation | 추가 비용 / 메모리 | 실패 방식 |
|---|---|---|---|---|---|---|---|---|
| K1 | A1 전역 재탐지 + G1 음성 prompt | A1: 어디(전역 외형) / G1: 방해물 배제 | 위치 제안과 분할 제약은 다른 단계 | drone, miami-surf 같은 비슷한 방해물 | A1 recall 통과 | A1 vs A1+neg (FP, J) | +1 decoder | 음성 점이 실제 대상 위에 찍힘 |
| K2 | B1 대조 attention 탐색 + B3 set ranker | B1: 후보 위치 / B3: 후보 간 경쟁·기권 | 생성과 선택을 나눔 | 동시 부재 + 진짜 부재 구분 | B1 통과, B3 > R2 (DEV) | B1+R2 vs B1+B3 | 무시 가능 | 18영상 학습 ranker의 일반화 실패 (M7 T 계열과 같은 위험) |
| K3 | C1 광류 crop + B2 attention crop | 운동(외형 독립) / 외형 | 서로 다른 오류 원인(가림 vs 외형 변화) | crop 이탈(drone, skate-park) | C1, B2 각각 L보다 recall 높음 | 단독 2개 vs 합(후보 풀) | +2 encoder | 둘 다 같은 방해물로 감 |
| K4 | E1 속도 beam + B3 ranker | 다중 운동 가설 / 신원 선택 | beam만으로는 고를 수 없음 | 빠른 이탈 후 재등장 | E1 recall 통과 | E1+R1 vs E1+B3 | +2 encoder | beam 모두 빗나감 |
| K5 | F2 foveation + B3 ranker (보고서 CPFR+TTPS) | 문맥 보존 확대 / 신원 | 관측 생성과 선택 | 작아진 대상 | F2 > F1 (같은 배율) | F2+R1 vs F2+B3 | +1~2 encoder | 왜곡된 관측에서 신원 특징이 무너짐 |
| K6 | F2 foveation + M3-vel (보고서 CPFR+M3) | 확대 관측 / 중심 예측 | 관측과 제어 | 이탈 + 축소 | F2 통과, M3-vel 유지 | F2, M3, F2+M3 | +1 encoder | 예측 중심이 틀리면 fovea가 빗나감 |
| K7 | A1 + C 광류 거리 순위(I1) | 외형 후보 / 운동 사전 | 외형만으로는 방해물을 구분할 수 없음 | 같은 종류 방해물(VK 차량, miami-surf) | A1 통과 | A1+R1 vs A1+R1+dist | 0 (flow는 계속 실행) | 카메라 급회전 시 광류 사전이 틀림 |
| K8 | rescue + D2 메모리 기록 | 출력 회수 / 장기 추적 재개 | 출력만 바꾸면 다음 frame에서 다시 잃을 수 있음 | 긴 부재 구간(drone 45 frame) | rescue arm 인과 양수 | admit vs no-admit | 0 | 잘못된 후보가 메모리를 오염(장기 identity switch) |
| K9 | rescue + E2 지연 확정 | 회수 / 거짓 재획득 억제 | 신호 vs 시간 일관성 | 진짜 부재 frame(355개) | rescue arm FP > 0.1 | 즉시 vs 지연 | 0 | 회수가 1 frame 늦어 짧은 사건 손실 |
| K10 | B1 + H1 다척도 heatmap 합의 | stride-16 의미 / stride-8 세부 | 서로 다른 척도 특징 | 극소 대상 | H1 > A1 (loc) | B1 vs B1×H1 | +0 (heatmap만) | 척도 간 peak 불일치 |
| K11 | A1/B1 전역 후보 + E3 독립 가설 분기 | 후보 / 반사실 rollout으로 확정 | 한 frame 순위보다 시간적 증거가 강함 | 비슷한 방해물 | A1 recall@3 > recall@1 + 0.1 | top-1 vs fork-k | +k 상태 (5090 전용) | 계산량 급증, 메모리 격리 버그 |
| K12 | C2 광류 box prompt + G1 음성점 | 운동 box / 방해물 배제 | box는 위치, 음성점은 경계 | 인접 방해물 | C2 통과 | C2 vs C2+neg | +0 | box 안에 방해물 포함 |
| K13 | D_ref(B4) + D2 메모리 기록 | 오염 없는 읽기 / 장기 재개 | 읽기와 쓰기를 나눔 | 메모리 오염 가설 사례 | B4 > B1 (recall) | B1+admit vs B4+admit | +1 decoder | 외형 변화가 큰 대상에서 frame-0 메모리가 맞지 않음 |
| K14 | 사건 트리거(F4) + 모든 rescue | 비용 제어 | 정확도 메커니즘이 아님 | — | 항상 적용 | always-on 대비 지연 | 절감 | 트리거 누락 사건 |
| K15 | C3 카메라 보정 + E1 beam | 카메라 운동 / 대상 운동 | 두 운동을 나눔 | 카메라 패닝 영상 | E1 통과 | E1 vs E1+cam | +flow | 카메라 추정 오류 |
| K16 | A1 + B1 + C1 + F2 후보 풀 + B3 ranker ("전체 풀") | 외형·대조·운동·확대 | 서로 다른 정보원의 후보 합집합 | 동시 부재 전체 | 구성요소 2개 이상 통과 | 풀 vs 최강 단일 생성기, 계산량 맞춘 대조(같은 수의 E-beam) | +6~8 호출 | 후보가 많을수록 거짓 양성 증가 |

## 실행 선택 규칙 (사전 등록)

1. Stage 1 진단 후 구성요소 통과 여부로 위 행을 거른다.
2. 남은 행 가운데 3–6개를 고른다. 기준은 정보 이득, 메커니즘끼리 보완하는 정도(서로 다른 정보원), 계산 가능성이다.
3. attention/비-attention 결합(K2, K4, K5, K7, K16 중 하나 이상)을 구성요소 근거가 허락하면 포함한다.
4. 각 결합은 SGGL, 가장 강한 단일 구성요소, 그리고 가능하면 계산량을 맞춘 대조와 비교한다.
