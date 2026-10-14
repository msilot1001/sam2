# M7 연구 기반 (Research Foundation)

**시작점:**
- 브랜치 `exp/m7-algorithm-research`, base commit `9d6f14b`(M6.1 최종)
- 이전 M7 시도("disagreement-aware selection", 로컬 브랜치 `exp/m7-disagreement-aware-selection`)는 커밋 없이 사용자가 중단했다. 그 시도의 선행연구 검토 원자료만 [`PRIOR_ART_raw.md`](PRIOR_ART_raw.md)로 재사용한다.
- 환경: RTX 2080 8 GB, SAM 2.1 Hiera-T, CUDA FP32, TF32 꺼짐, 영상 CPU offload

## 1. 읽은 문서와 이어받는 사실

| 문서 | M7에 가져오는 사실 |
|---|---|
| `MILESTONE_5_5.md` | P5(전체 해상도 prompt 메모리) 수정. SAM 2의 256 logit 경로가 작은 대상의 공간 상세를 잃는다 |
| `MILESTONE_5_6.md` | 실패 감사: 작은 대상 실패의 주원인은 입력 해상도. crop 기반 방향 제안 |
| `m6/FINAL_RESEARCH_TOPIC_DECISION.md`, `m6/OVERNIGHT_EXPERIMENT_REPORT.md` | A2x(SGGL) 정의, 개발 세트 작은 객체 +0.018, CTU식 일치 gate 대비 +0.008 |
| `m61/FINAL_RESEARCH_DIRECTION.md`, `FINAL_ALGORITHM_SPEC.md` | SGGL / SGGL-V 공동 1차. 거부권 이득은 miami-surf obj5 하나(사후) |
| `m61/EXPERIMENT_COMPARISON.md`, `OVERNIGHT_DEEP_RESEARCH_REPORT.md` | 기각: TPLR(메모리 없는 재분할), 보정 gate(선택 포화: self gate가 선택 상한의 89%), 정제 head(수렴 실패, 결론 불가), 양방향 되쓰기. 한 방향 되쓰기 A3 +0.002 |
| `m61/REVIEW_raw.md`, `REVIEW2_raw.md` | 거부권은 miami-surf를 빼면 −0.0015. 극소 객체에서 발동 0회. 비용 쪽 기각 조건 필요 |
| `m61/IMPLEMENTATION_ROADMAP.md` | TASK-M7-004(단일 모듈화), -005(기준선) 계획 |

**M6.1의 정량 사실 중 M7 설계에 직접 쓰는 것:**
- **(a) 선택은 거의 포화:** self gate가 {G, L} 선택 상한의 89%에 이른다.
- **(b) 마스크 결합 상한은 작다:** 결합 상한이 선택 상한을 넘는 폭은 +0.003뿐이다. 이번에 정식 측정으로 다시 확인한다.
- **(c) 치명 실패의 원인:** 진짜 부재 뒤 L이 높은 확신으로 다른 물체를 다시 잡는 경우다. 이번에 독립 사건이 몇 개인지 센다.
- **(d) 메모리 없는 재분할은 실패:** 부분 대상의 정의를 잃는다.

## 2. 재사용한 기반과 새로 만든 모듈

| 기능 | 위치 | 상태 |
|---|---|---|
| 프레임 단위 SAM 2 진행, 메모리 계획 제어 | `src/memint/engine.py`, `controlled.py`, `planners.py` | 재사용. planner로 메모리 허용(M2)과 보호 메모리(M1)를 구현 |
| 가상 카메라 crop | `src/spatial/vcam.py` | 재사용 |
| **두 상태 rollout + 정책 + 진단 probe** | **`src/spatial/sggl.py` (신규)** | `scripts/m61.py:run_dual`을 통합. 기록된 A2x/A2v를 비트 단위로 재현한다(테스트) |
| S 계열 증거 (상태, 관측 가능성, 신원) | `src/spatial/evidence.py` (신규) | |
| F 계열 결합 | `src/spatial/mask_fusion.py` (신규) | |
| 후보 정책 20종 | `src/spatial/policies.py` (신규) | 재생과 실시간에 같은 코드를 쓴다 |
| T 계열 head (T0–T4) | `src/spatial/feature_fusion.py`, `target_attention.py` (신규) | |
| 실행 스크립트 | `scripts/m7.py` (기록, 인과 rollout, T 캐시와 학습), `scripts/m7_replay.py` (CPU 재생, 감사) | |
| 지표, 영상 군집 bootstrap | `baseline/metrics.py`, `rfeas/risk.py` | 재사용 |

## 3. 데이터 역할 (사전 등록, `outputs/milestone7/preregistration.json`)

| 세트 | 단위 | 역할 |
|---|---|---|
| DEV | M6 개발 crop 객체 21개 (DAVIS val, PUMaVOS A/B, SegTrack) | 개발. 여러 번 본 데이터 |
| REP | DAVIS 2017 train crop 객체 43개 / 영상 23개 | 개발 + T head 학습(영상 단위 분리). miami-surf는 발견 사례 |
| VK | VKITTI2 Scene01/06 clone crop 객체 28개 (scene 2개, 합성) | SGGL에는 처음 쓰는 세트. 결선 후보를 고정한 뒤 1회만 실행. 같은 종류의 방해물(차량)이 많다 |
| PUMaVOS C | 11개 객체 | 손대지 않음. 최종 단일 방법이 최종 규칙을 통과할 때만 1회 실행 |

- **추가 데이터:** 이전 시도에서 LVOS v2 메타데이터 확인을 시작했지만 사용자가 그 agent를 중단했다. 그래서 새 데이터는 내려받지 않는다.
- **알려진 한계:** VKITTI 객체는 prompt 면적 ≥ 0.1% 기준으로 골랐다. 그래서 극소(< 0.1%) 대상은 이후 frame에서만 나타나고, scene이 2개라 영상 군집 CI는 의미가 없다.

## 4. 규칙 (사용자 지시 그대로 적용)

- **누설 금지:** GT는 frame 0 prompt와 오프라인 채점에만 쓴다. 미래 frame과 미래 GT는 쓰지 않는다. 정책은 현재와 과거의 모델 출력만 본다.
- **진단 분기 격리:** probe는 저장하지 않는 예측이고, 건드린 이미지 feature cache를 복원한다. production 출력이 비트 단위로 같음을 테스트로 확인했다.
- **재생과 인과 rollout 구분:** 재생은 SGGL 궤적 위에서 출력만 바꾼 근사다. 출력을 바꾸는 후보는 인과 rollout으로 다시 확인해야 ADVANCE할 수 있다.
- **학습 데이터:** DAVIS train(REP)만 쓰고 영상 단위로 분리한다. DEV, VK, C의 라벨은 쓰지 않는다.
- **GPU:** 한 번에 한 작업. 초기 상한은 8–12 GPU-시간이다.
