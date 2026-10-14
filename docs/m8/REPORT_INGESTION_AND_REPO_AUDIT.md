# M8 첨부 보고서 수용과 저장소 감사

작성: 2026-10-14, M8 첫 단계 (실험 출력 이전). 브랜치 `exp/m8-diverse-algorithm-portfolio`는 `exp/m7-algorithm-research` `b5c5091`에서 분기했다.

## 1. 첨부 보고서("M7 심층 연구 분석과 M7.1 연구 방향")가 실제로 제안하는 것

| 항목 | 보고서 내용 |
|---|---|
| 결론 | KEEP_SGGL 유지. 연구 중심을 selection → **candidate generation**으로 이동 |
| 근거 | 선택 oracle +0.018, 결합 추가 +0.002, 성분 +0.006, 새 후보(L2/Lp/Lt) oracle +0.033. 최대 손실 = 두 view 동시 실패(156.6 frame-JF / 14영상) |
| 1순위 | FORGE-SGGL: Feature-Occupancy Rescue (OCLR), Context-Preserving Foveated Rescue (CPFR) |
| 성능 후보 | Event Search Beam (MSB, M3-vel의 다중 가설 확장) |
| attention 재도전 | Target-Token Proposal Scorer (TTPS): attention을 mask 보정이 아니라 후보 순위화에 사용 |
| 강한 대조군 | Global Reference Re-prompt (GRR), M3-vel |
| 조건부 | Deferred Memory Admission (DMA), Event Trigger (O1 발전형) |
| 결합 | CPFR+TTPS, MSB+TTPS, CPFR+M3-vel만 의미 있다고 판단 |
| 판정 규칙 | STRONG_WIN(+0.03, CI>0), ADVANCE_TO_STAGE2(전체 ≥+0.010 & event subset ≥+0.03 & ≥3영상), regression kill(<−0.05), 런타임 +30% |
| 필수 테스트 | disabled rescue == SGGL, foveated round trip, occupancy, branch 격리, memory admission 격리, no-future, report 재생성 |
| 데이터 | PUMaVOS C 봉인, 새 tiny 데이터(≥20객체/12영상)가 알고리즘 튜닝보다 먼저 |
| 선행연구 | DAM4SAM, SENTRY, SAMURAI, SAM2Long, Lean-SAM2, StreamDAM, SAM2Dual, SAM2Refiner, MoSAM, AdaSAM2, SAMITE, VOTS2025 |

M8 지시문은 이 보고서를 확장해 9개 계열(A–I) 25–35 후보, 12–18 결합, 다중 attention 삽입 위치를 요구한다.

## 2. 저장소에 실제로 있는 것 (확인함)

- 코드: `src/spatial/{sggl,policies,evidence,mask_fusion,feature_fusion,target_attention,vcam,detail}.py`, `scripts/m7*.py`, `tests/test_m7.py`(11개). M7 문서 12개 + 검토 문서 모두 존재.
- 출력: `outputs/milestone7/rec/`(64 단위 frame별 G/L/L2/Lp/Lt logit, 2.3 GB), `runs/`(SGGL, SGGL-V, M3-vel, M3-vel-r, F1-blend.7, M1-always), `tcache`(7.5 GB), `teval`, `heads`, VK 결과.
- 데이터(로컬): DAVIS 2017, PUMaVOS(A/B/C, C는 봉인), SegTrack v2, VKITTI2. **그 외 VOS 데이터 없음.**
- 하드웨어(측정): 로컬 RTX 2080 8 GB, 10 vCPU, RAM 25 GB, 디스크 여유 41 GB. Vast RTX 5090(89.221.67.161:16328)은 **M8 시작 시 연결 거부** (사용자가 시작 중).
- 라이브러리: OpenCV/kornia 없음. torchvision 0.29.1의 RAFT-small(4 MB, BSD, FlyingChairs+Things 학습 — VOS 데이터 비중첩) 사용 가능.

## 3. 보고서와 저장소 사실의 충돌, 그리고 조정

| # | 보고서 주장/가정 | 실제 (측정/코드) | 조정 |
|---|---|---|---|
| 1 | 실패 원인 = G에서 target이 1–2 cell (feature occupancy). OCLR은 target이 6–10 cell이 되도록 확대 | L crop은 변 = 4× prompt box → target이 이미 **≈16 cell**. 동시 부재 frame(GT 있음 131개)의 GT crop 내부 비율 중앙값 0.54 — **절반은 target이 crop 밖**. L에게는 occupancy가 아니라 위치 문제 | OCLR/CPFR을 G 메모리(확대)와 L 메모리(o*=8 → 사실상 축소) **양쪽**에서, 같은 배율로 짝지어 실험. 위치 가설(운동/전역 탐색)과 분리 |
| 2 | event ≈ 4.8% frame | 4.8%는 '불일치' 사건(242/5082). **동시 부재는 486 frame(9.6%)**이고 그중 355는 GT가 실제로 비어 있음(진짜 가림/이탈) | rescue는 진짜 부재 frame에서 침묵해야 함 → 거짓 양성률을 1차 지표로 사전 등록 |
| 3 | STRONG_WIN +0.03 | 동시 부재 frame을 **전부 완벽히** 고쳐도 객체 평균 +0.028 | +0.03은 이 데이터에서 rescue 단독으로 도달 불가. 공개하고 그대로 보고 |
| 4 | 결과 구조 `docs/m71/`, `scripts/m71.py` | 지시문은 `docs/m8/`, `outputs/milestone8/` | M8 경로 사용, 보고서 구조(모듈 분리)는 유지: `rescue.py`, `motion.py`, `ranker.py`, `scripts/m8.py` |
| 5 | T2 positive control J 0.869→0.966 | M7 기록 0.869→0.965 | 사소 (반올림) |
| 6 | 새 tiny-object 데이터 우선 | 데이터 다운로드는 사용자 승인 필요, 승인 없음 | DEV+REP(탐색적)로 진행, 새 데이터는 BLOCKED로 표기 |
| 7 | 보고서는 foveation을 "현재 확인된 동일 구성 없음"으로 봄 | Learning to Zoom(ECCV'18), FOVEA(ICCV'21), LZU(CVPR'23) 등 비균일 샘플링 선행연구가 있음 | 선행연구 감사에 포함 (`PRIOR_ART_AND_NOVELTY_AUDIT.md`) |
| 8 | mask prompt로 rescue 후보 생성 가능성 | SAM 2.1 설정 `use_mask_input_as_output_without_sam=True` → mask 입력은 SAM을 거치지 않고 그대로 출력 | mask prompt 계열(flow-warp mask prompt)은 의미 없음 → point/box prompt만 사용 |
| 9 | (보고서 없음) | 기존 `Dual.probe(kind='image')`가 메모리 고정 + 영상 교체 probe를 이미 지원, 단 prompt 불가 | `rescue.probe`로 prompt + 영상 교체 + planner를 한 번에 지원, 비저장 테스트 추가 |

## 4. 재사용과 신규

- 재사용: `sggl.Dual`(상태), `vcam`, `m7.units/gt/paths/shard`, `m55.jf`, `memint.controlled`(planner), M7 판정 규칙과 bootstrap(`rfeas.risk.cluster_bootstrap`).
- 신규: `src/spatial/rescue.py`(기하·heatmap·prompt probe), `motion.py`(RAFT, Track), `ranker.py`(R1–R3), `scripts/m8.py`(prereg, diag, refs, run), `tests/test_m8.py`.
- 하지 않은 것: M7의 S/F/M/T 실험 재실행(재현 가능한 음성 결과로 재사용).
