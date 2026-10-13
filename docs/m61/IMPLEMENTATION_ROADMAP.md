# 구현 로드맵 (M7, 1–2주): 현재 저장소 상태에서 시작

**출발점 (실제 상태):**
- 브랜치 `exp/m61-overnight-deep-research`
- 실험 코드는 여러 곳에 흩어져 있다:
  - `scripts/m6.py`: `run_unit` arm A2x. 전역 상태는 **기록된 B1**에서 읽는다.
  - `scripts/m61.py`: `run_dual` arm A2d / A2v / A3 / A3s. 두 상태를 실시간으로 돌린다.
- 공용 모듈은 아직 없다.
- PUMaVOS C는 열지 않았다.
- 극소 대상 데이터는 거의 없다(개발 관문 세트 극소 객체 3개 / 영상 2개, 개발 전체 4개 / 영상 3개).

**전제:**
- 사용자가 주제를 승인해야 시작한다.
- 큰 데이터나 checkpoint를 내려받기 전에는 매번 승인을 받는다.
- master에 병합하지 않는다.

| ID | 목표 | 파일·모듈 | 의존 | 수용 테스트 | 기대 결과 | 완료 기준 |
|---|---|---|---|---|---|---|
| **TASK-M7-001** | 직접 선례 확인: VOTS 2023–2025 참가 추적기 부록, CTU 저자 코드의 메모리 처리 | `docs/m7/PRIOR_ART_VOTS.md` | — | 읽은 문서 목록과 판독 수준 | 'crop + SAM 재분할 / 메모리 되쓰기'의 직접 선례 유무 확인 | 직접 선례가 있으면 주제를 재검토(중단 조건) |
| **TASK-M7-002** | 극소 대상 데이터 선정 **(승인 필요)**: SA-V val/test, LVOS v2 val, VOST의 라이선스·크기·mask 공개 여부 확인. GT만으로 prompt 면적 < 0.1% 객체 수 집계(SAM 2 실행 금지) | `scripts/m7_data.py`, `docs/m7/DATA.md` | 사용자 승인 | 집계 스크립트 재실행 결과 동일 | 극소 객체 ≥ 20개, 영상 ≥ 12개 | 데이터 manifest 커밋 |
| **TASK-M7-003** | 분할과 기각 관문 고정: 영상 hash로 개발 / 확인 절반. 관문은 FINAL_RESEARCH_DIRECTION §2 기각 조건의 숫자 그대로 | `outputs/milestone7/{manifest,preregistration}.json` | 002 | `_freeze` 재실행 시 변경 거부 | — | 추론 전 커밋 |
| **TASK-M7-004** | `src/spatial/sggl.py`: `run_dual`을 단일 모듈로 옮긴다. gate = self+presence(기본) / self / agreement / none, write_back = none / l2g / l2g_reliable | `src/spatial/sggl.py`, `tests/test_sggl.py` | — | (a) crop 미적용 = P5 (b) gate=self, write_back=none이 기록된 A2x와 같음, gate=self+presence가 기록된 A2v와 같음 (c) write_back=l2g가 기록된 A3와 같음 (d) 인과성 (e) L 메모리에 G 쓰기 없음 (f) l2g_reliable은 G score ≤ 0이면 쓰지 않음 | 테스트 6개 통과 | 개발 21개 객체에서 A2x와 A3 기록이 비트 단위로 재현됨 |
| **TASK-M7-005** | 기준선: CTU 원형(H/4, H/2 창 병렬, 일치 gate γ = 0.5, β = 0.1)과 F2 flip 앙상블 | `src/spatial/baselines.py` | 004 | DAVIS val 1개 영상 sanity, 결정론 | — | 두 기준선이 개발 세트에서 실행됨 |
| **TASK-M7-006** | **기각 관문 실행 (새 데이터 개발 절반)**: 주 비교는 B0, B1, A1, **SGGL-V(= A2v)**, SGGL(= A2x), A2c, CTU 원형, F2. 부재 후 재등장 사건은 별도 집계한다. W arm 3종(끔 / 항상 / 신뢰도 gate)은 ablation | `scripts/m7.py run` | 003, 004, 005 | 실행 장부, 객체별·영상별 표, 영상 수 표기, 영상 하나씩 빼기 분석 | 극소 객체에서 SGGL 또는 SGGL-V(공동 1차) ≥ +0.03이고 F2와 CTU 원형보다 나음 | FINAL §2 기각 조건 1–4b 판정. **실패하면 TASK-M7-010으로** |
| **TASK-M7-007** | ablation (명세서 §9): 배율 목표와 최대 배율, W arm 3종, gate 3종 | `scripts/m7.py ablate` | 006 통과 | 사전 등록 | 메커니즘별 효과 분리. W 판정은 FINAL §2 조건 5 | ablation 보고서 |
| **TASK-M7-008** | Hiera-L 검사 **(ckpt 다운로드 승인 필요)** | 같은 스크립트, `--cfg` | 006 통과 | — | 효과가 절반 이하로 줄면 위험 | backbone 의존성 판정 |
| **TASK-M7-009** | **확인 1회:** 코드와 파라미터를 고정해 커밋한 뒤 새 확인 절반 + PUMaVOS C를 실행한다. 모든 객체를 보고하고 이후 튜닝하지 않는다 | `scripts/m7.py confirm` | 007, 008 | 확인 전 커밋 hash 기록 | — | 확인 보고서. FINAL §2 기각 조건 판정 |
| **TASK-M7-010** | (기각 시) 범위 전환 제안: 'SAM 2.1의 학습 없는 추론 단계 개선의 상한 분석'. M4.5–M6.1 결과 정리 | `docs/m7/SCOPE_CHANGE.md` | 006 실패 | — | — | 사용자 승인 |
| **TASK-M7-011** | 분석과 원고: 크기별 효과, escape와 채택률, 되쓰기 피드백 위험, 예측 IoU의 배율 편향, 실패 사례 | `docs/m7/` | 009 | — | — | 논문 초안 |

**일정 (승인 후):**

| 기간 | TASK |
|---|---|
| 1일차 | 001, 002 |
| 2일차 | 003, 004 |
| 3일차 | 005, 006 (**관문**) |
| 4–5일차 | 007, 008 |
| 6일차 | 009 |
| 7–10일차 | 011 |

**계산 추정 (Hiera-T, 극소 대상 세트 약 10k frame 가정):**
- TASK-006: arm 7개 × 0.2–0.45 s/frame ≈ 4–5시간
- TASK-007: 약 4시간
- TASK-008: 미측정 (Hiera-L은 3–4배 느릴 것으로 예상)
