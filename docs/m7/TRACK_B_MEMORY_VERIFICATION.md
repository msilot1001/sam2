# Track B: 메모리 조건 반사실 검증 (Memory-Conditioned Candidate Verification)

**판정: REJECT.** 보호 메모리 반사실은 일반적인 외형·시간 신호가 주지 못하는 신원 정보를 주지 않는다.
- 감사 2×2 표에서 보호 메모리가 L을 고치는 경우는 26 frame, 망치는 경우는 361 frame이다.
- 대표 실패(miami-surf obj5)는 메모리 오염이 아니라 외형이 닮은 방해물이다.
- M1 정책 3종(재생)과 M1-always(인과)는 모두 |Δ| ≤ 0.0004다.
- M2(메모리 허용)는 인과 근거가 없어 평가하지 않았다. M3(crop 중심 제어)는 별도로 평가했다. M3-vel은 INCONCLUSIVE이고, 개정판은 [`STAGE2_REVISIONS.md`](STAGE2_REVISIONS.md)에 있다.

## 1. 반사실의 정의와 상태 일관성 (`src/spatial/sggl.py: Dual.probe`)

| 조건 | 바꾸는 것 | 그대로 두는 것 |
|---|---|---|
| B0 정상 | — (production L 상태) | — |
| B1 보호 초기 메모리 (`protected`) | 메모리 계획을 **prompt frame의 공간 메모리와 object pointer만**으로 제한(`_cond_only` planner) | 현재 crop 이미지와 그 이미지 특징(cache), crop 좌표, 모델 파라미터, 출력 좌표 |
| B2 신뢰 메모리 (`trusted`) | 비조건 메모리를 'L과 G가 일치한 frame'(두 score > 0이고 IoU > 0.5)으로만 제한. 빈 슬롯은 32 frame 이내의 신뢰 frame으로 채움(`planners.backfill`) | 같음 |

**구현 원칙:**
- SAM 2의 공식 검색 경로(`controlled.py`, 공식 코드와 비트 동일로 테스트된 사본)의 계획만 바꾼다.
- 공간 메모리, 위치 부호, 시간 위치 부호(`maskmem_tpos_enc`), object pointer, 조건/비조건 저장소, frame 번호가 모두 원본 저장소에서 일관되게 읽힌다. 불완전한 메모리를 이식하지 않는다.

**격리 (`tests/test_m7.py::test_recording_rollout_reproduces_sggl_probes_are_isolated_and_causal`):**
- probe를 넣은 rollout과 넣지 않은 rollout을 비교했다. G/L/L2 logit, 유사도 지도, object pointer, crop box, J&F가 비트 단위로 같다.
- probe는 출력 dict에 저장하지 않는다. 이미지 특징 cache를 바꾸는 해상도 probe는 cache를 원래대로 복원한다.
- 시간을 자른 rollout(n = 5와 n = 9)의 앞부분이 같다. 미래 의존이 없다는 뜻이다.

## 2. 측정 (Stage 0 기록, L이 존재하는 frame 4,530개)

| | L-보호 맞음 | L-보호 틀림 |
|---|---|---|
| **L 맞음** | 3,932 | 361 |
| **L 틀림** | **26** (객체 10 / 영상 8) | 211 |

**L과 L-보호의 불일치(IoU < 0.5) 비율, 경과 시간별:**

| 경과 frame | 1–10 | 11–30 | 31–60 | 60+ |
|---|---|---|---|---|
| 불일치 비율 | 1.3% | 7.4% | 19.1% | 9.1% |

- 불일치는 시간이 지나며 늘고, 대부분 L이 맞는 쪽이다(361 vs 26). **정상적인 외형 적응**을 반영한다.
- **대체 후보로 쓸 때:**
  - L-보호를 L 대신 쓰면 −0.076이고, L-신뢰는 −0.004다.
  - 선택 상한에는 각각 +0.006–0.008을 더한다. 하지만 그 frame들을 GT 없이 고를 신호를 찾지 못했다.

**대표 사례 miami-surf obj5:**
- L-보호의 object score는 방해물 위에서 6–7이다. prompt 메모리만으로도 방해물을 대상으로 판단한다.
- 따라서 반사실 메모리 비교는 원리상 이 실패를 구별하지 못한다.
- 같은 사례에서 G도 frame 31–33에 같은 방해물을 잡는다(score 0.21–0.79). 즉 이것은 view나 메모리의 문제가 아니라 **외형 모호성**이다.

## 3. 강한 대조와의 비교 (재생; 인과는 M1-always만)

작은 객체 57 / 영상 34, Δ vs SGGL:

| 정책 | 정의 | ΔJ&F | 판정 |
|---|---|---|---|
| M1-verify | 미확인 재획득은 G 지지 또는 L-보호 일치(IoU > 0.3)로만 채택 | −0.0004 | REJECT |
| M1-always | L-보호와 불일치하고(IoU ≤ 0.1 또는 부재) G 지지도 없으면 항상 거부 | 재생 +0.0003, 인과 +0.0003 | REJECT |
| M1-trusted | 미확인 재획득에 G 지지가 없으면 신뢰 메모리 예측을 대신 출력 | +0.0001 | INCONCLUSIVE (효과 0) |
| 대조: SGGL-V | — | +0.0069 (miami-surf 제외 −0.0019) | REJECT |
| 대조: S1-state | — | −0.0000 | REJECT |
| 대조: S3-abs (frame-0 외형 신원) | — | +0.0003 | REJECT |
| 대조: S1-K3 (단순 지속성) | — | −0.0005 | REJECT |
| 대조: M3-vel (최근 운동 연속성, 인과) | — | +0.0046 | INCONCLUSIVE |

**주 질문의 답:**
- 질문: 메모리 반사실 개입이 일반 외형·시간 신호가 주지 못하는 정보를 주는가?
- 답: **아니다.** M1 계열은 외형 신원(S3)이나 지속성(S1-K3)보다도 낫지 않다. 모두 0 근처다.

## 4. 조건부 후속 (M2, M3)

- **M2 (메모리 허용 분리):**
  - 사전 조건은 '메모리가 오류를 지속시키는 인과 근거'였다. 감사 범주 9(메모리 양성 되먹임)는 1차 7 frame, 영상 2개뿐이다.
  - 그래서 **평가하지 않았다(IMPLEMENTED_NOT_EVALUATED).** 허용 planner는 `Dual.excluded`와 `planners.backfill`로 구현돼 있다.
- **M3 (crop 중심 제어):**
  - crop 이탈(범주 8)이 영상 9개에 걸쳐 가장 넓게 퍼진 실패라 인과 rollout으로 평가했다.
  - M3-vel: +0.0046 [−0.000, +0.012], INCONCLUSIVE. 개정판 M3-vel-r은 Stage 2에서 다룬다.

## 5. 선행연구

- Matthews, Ishikawa & Baker, "The Template Update Problem"(BMVC'03 / TPAMI'04)의 전략 3이 직접 선례다. 처음 template로 다시 추적해 비교하고, 차이가 크면 갱신하지 않는다.
- SAM 2 계열에서 같은 비교를 쓴 선례는 찾지 못했다([`PRIOR_ART_raw2.md`](PRIOR_ART_raw2.md)).
- 이번 결과는 그 아이디어를 SAM 2에 옮겨도 이 데이터에서는 정보가 없다는 **음성 결과**다.
