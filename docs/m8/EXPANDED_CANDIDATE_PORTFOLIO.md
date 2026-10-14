# M8 확장 후보 포트폴리오 (33개, 9계열)

- §1은 실험 전(2026-10-14) 분류다. '상태' 열은 실험 후 갱신하며, 최종 값은 [`EXECUTION_STATUS.json`](EXECUTION_STATUS.json)과 [`FINAL_PORTFOLIO_ASSESSMENT.md`](FINAL_PORTFOLIO_ASSESSMENT.md)를 따른다.
- **구분 기호:** NEW = M7에 없던 메커니즘 / M7≈ = M7 후보의 재명명·변형 / ENG = 공학적 변형 / DIAG = 진단 도구 / COMBO = 기존 구성요소의 결합 / PA = 선행연구가 가까움(상세는 [`PRIOR_ART_AND_NOVELTY_AUDIT.md`](PRIOR_ART_AND_NOVELTY_AUDIT.md)).
- **역할:** ①=이미 위치가 맞는 대상의 분할 개선, ②=두 view가 모두 실패할 때 새 증거/후보 생성. M8의 주 대상은 ②다(M7: ①의 상한 +0.002–0.006).
- 생성기 이름은 코드(`scripts/m8.py: generators`)의 이름과 같다.

## 1. 후보 표

| ID | 계열 | 메커니즘 | 역할 | 새 정보원 | 구분 | 비용 (trigger frame당) | 반증 시험 | 상태 |
|---|---|---|---|---|---|---|---|---|
| A1 | A 전역 탐색 | `A_cos`: frame-0 prototype 코사인 heatmap(G stride-16) → top-3 peak → G 메모리에 point prompt | ② | 현재 frame 전역 외형 대응 | NEW (PA 높음: PerSAM/Matcher/SAMITE) | +3 SAM 2 decoder | recall ≤ 2× max(Gp,Lp) 또는 4영상 미만 | Stage 1 |
| A2 | A | `A_cos_crop`: heatmap top-1에 L crop 재배치(prompt 없음) | ② | 위치 | NEW | +1 crop encoder | L 대비 recall 무증가 | Stage 1 |
| A3 | A/E | 전역 top-k peak를 독립 SAM 2 상태로 k frame 추적 후 선택 | ② | 시간적 일관성 | NEW (PA: SAM2Long) | +k 상태 | A1 recall이 낮으면 무의미 | PROPOSED (A1 결과 대기) |
| A4 | A | 작은 창 sliding template matching(stride-4) | ② | 고해상도 외형 | ENG | 높음 | H1이 A1보다 못하면 생략 | PROPOSED |
| B1 | B attention | `B_att`: 각 cell이 frame-0 fg(8 k-means) + bg(32) token에 softmax attention → fg 질량 = 대상 사후확률 | ② 탐색 | 대조(배경 유사 억제) | NEW (PA 중: 판별 추적기) | +3 decoder | B ≤ A (loc, recall; 짝지은 영상 CI) | Stage 1 |
| B2 | B | `B_att_crop`: attention peak로 crop 배치 | ② 위치 제어 | 위치 | NEW | +1 encoder | A2 대비 무차이 | Stage 1 |
| B3 | B | R3 set-attention ranker: 후보 token 간 self-attention(경쟁) + frame-0 token cross-attention(신원) + abstain token | 순위화 | 후보 간 비교 | NEW (PA 중: KeepTrack, DETR no-object) | 무시 가능 | DEV에서 R2(로지스틱) 이하 | Stage 1 (학습 REP) |
| B4 | B/D | `D_ref`: B1 peak point prompt + frame-0 메모리만 읽는 memory attention | ② | 오염 없는 메모리 판독 | NEW 조합 (M1 변형) | +1 decoder | recall ≤ B1 | Stage 1 |
| B5 | B | 학습형 frame-0 token → 현재 특징 localization head | ② | 학습 | NEW | +head | REP 18영상, M7 T 계열과 같은 일반화 위험 | NOT_RUN (데이터 부족; 새 데이터 필요) |
| B6 | B | G↔L 교차 척도 attention(로짓 보정) | ① | — | M7≈ (T2) | — | M7: −0.0013 | REJECTED (M7) |
| B7 | B/D | 메모리 쓰기 slot attention(학습형 admission) | ② | 학습 | NEW (PA 높음) | — | 오염 인과 근거 부족 | NOT_RUN |
| C1 | C 운동 | `C_flowcrop`: RAFT-small 광류로 운반한 box → L crop 중심 | ② | 광류(외형 독립) | NEW (PA 중: SAM-PT, MoSAM) | +flow/frame, +1 encoder | L 대비 recall 무증가 | Stage 1 |
| C2 | C/G | `C_flowbox`: 광류 box → G box prompt | ② | 광류 + prompt | NEW (PA 중) | +1 decoder | recall ≤ A1 | Stage 1 |
| C3 | C | 카메라 운동 보정 M3-vel(전경 밖 median flow 차감) | ① 위치 | 카메라 운동 | ENG (M3 변형) | +flow | M3-vel 대비 무차이 | PROPOSED |
| C4 | C | 점 추적(CoTracker/PIPs) → prompt | ② | 장기 점 대응 | PA 높음 (SAM-PT) | 큼 | — | BLOCKED (가중치 라이선스 CC-BY-NC, 다운로드 승인 필요) |
| C5 | C/I | 정방향-역방향 광류 순환 일관성으로 후보 검증 | 검증 | 순환 일관성 | ENG | +flow | — | PROPOSED |
| D1 | D 메모리 | `Gp`/`Lp`: 보호 메모리 probe | ② | 초기 참조 | M7≈ (M1) | +1 | 기준선 | Stage 1 (참조) |
| D2 | D | rescue 후보를 메모리에 즉시 기록(admit) vs 출력만 | ② 장기 | 메모리 쓰기 | NEW 조합 (PA 높음: DAM4SAM, SENTRY) | 0 | admit ≤ no-admit | Stage 2 (조건부) |
| D3 | D | 거부된 후보로 온라인 distractor exemplar bank → B1 bg token에 추가 | 순위화 | 온라인 음성 | NEW (PA: DAM4SAM) | 무시 가능 | B1 FP 감소 없음 | PROPOSED |
| E1 | E 다중 가설 | `E_beam1/2`: 마지막 신뢰 중심 + 1×/2× 속도 외삽 crop beam | ② | 운동 가설 | M7≈ 확장 (M3) / PA 높음 (SAM2Long) | +2 encoder | L, M3-vel 대비 무증가 | Stage 1 |
| E2 | E | 지연 확정: rescue를 다음 frame 확인 후에만 출력 | 거짓 재획득 | 시간 | ENG (PA: SatSAM2, SENTRY) | 0 | FP 감소 없이 회수 감소 | PROPOSED |
| E3 | E | rescue 시 상태 분기(fork) 후 k frame 경쟁, 승자만 유지 | ② | 반사실 rollout | NEW 조합 (PA 높음) | +k× | — | NOT_RUN (비용; 1단계 근거 대기) |
| F1 | F 척도 | `F_oclr{8,4}_{G,L}`: target이 N stride-16 cell을 차지하도록 배율 맞춘 crop | ② | 표현 점유율 | NEW (보고서 1순위) | +1 encoder | CPFR와 짝 비교 | Stage 1 |
| F2 | F | `F_cpfr{8,4}_{G,L}`: 같은 배율의 분리형 foveation(주변 압축, 전체 문맥 유지) + 역변환 | ② | 문맥 보존 확대 | NEW (PA 중: Learning to Zoom, FOVEA, LZU) | +1 encoder | 같은 배율 OCLR 대비 recall 차 CI가 0 포함 | Stage 1 |
| F3 | F | L2 (2× crop) 대체 | ② | 넓은 시야 | M7≈ (R2/M71-04) | +50% | — | 참조 (M7) |
| F4 | F | 사건 기반 계산(트리거) | 비용 | — | M7≈ (O1) | 절감 | always-on 대비 같은 정확도에서 지연 감소 | 모든 arm에 내장 |
| G1 | G prompt | `G_neg`: B1 양성 + B 사후확률 < 0.5인 A peak에 음성 점 | ② 신원 | 방해물 음성 prompt | NEW (PA 중: PerSAM) | +1 decoder | recall/FP가 B1 단독과 같음 | Stage 1 |
| G2 | G | 반복 prompt 정제(후보 mask 중심 + box로 재prompt) | ① | — | ENG | +1 | — | PROPOSED |
| H1 | H 표현 | `H_s8`: stride-8 FPN 특징으로 같은 prototype 매칭 | ② | 고해상도 특징 | NEW (진단적) | +3 decoder | H ≤ A (loc) → 표현 점유율 가설 기각 | Stage 1 |
| H2 | H | DINOv2 등 다른 pretrained 특징 | ② | 다른 사전학습 데이터 | PA 높음 | +ViT | — | NOT_RUN (추가 가중치; 사전학습 데이터 효과와 분리 불가 → 우선순위 낮음) |
| H3 | H | REP로 학습한 경량 projection adapter(대조 학습)로 매칭 | ② | 학습 | NEW | 작음 | DEV에서 A1 이하 | PROPOSED (A1/B1 결과 대기) |
| I1 | I 구조 | 광류 box 거리/크기 비율을 후보 순위 특징으로(형태·운동 사전) | 순위화 | 운동 사전 | ENG | 0 | R2 계수 무의미 | Stage 1 (ranker 특징으로 내장) |
| I2 | I | 마스크 결합·경계 보정 | ① | — | M7≈ (F1–F4) | — | M7 상한 +0.002 | REJECTED (M7) |

**계열별 수:** A 4, B 7, C 5, D 3, E 3, F 4, G 2, H 3, I 2 = 33. attention 역할은 서로 다른 5개 위치(B1 탐색, B2 crop 제어, B3 후보 경쟁·순위, B4 메모리 판독, B5 학습 localization)와 M7의 로짓 보정(B6)이다.

## 2. 선별 원칙 (실험 전)

- Stage 1은 한 번의 GPU 통과로 **모든 생성기를 같은 trigger frame에서** 비저장 probe로 실행한다. 생성기 간 비교는 같은 frame에서 짝을 짓는다.
- 인과 단계로는 사전 등록 규칙(`outputs/milestone8/preregistration.json` `advance_to_causal`)을 통과한 생성기만 간다. 통과 수가 4 미만이면 그 사실을 보고하고, 계열 다양성을 위해 가장 정보가 큰 탈락 후보 1–2개를 '진단적 인과 실행'으로 명시해 돌릴 수 있다(판정에 쓰지 않음).
- M7에서 재현 가능한 음성 결과가 있는 것(B6, I2, S 계열)은 재실행하지 않는다.
