# M9 최종 평가: 교차 데이터셋 검증

- **기간:** 2026-10-14 23:01 – 2026-10-15 06:40 KST
- **브랜치:** `exp/m9-cross-dataset-validation` (병합하지 않음)
- **시작 HEAD:** `e0c1b44`

## 1. 질문과 답

| 질문 | 답 |
|---|---|
| M8 FLOWVEL이 독립 데이터에서 유지되는가? | **아니오.** MOSEv2 −0.0142 [−0.0284, −0.0016], VOST +0.0046 [−0.0181, +0.0272], LVOS v2 −0.0021 [−0.0110, +0.0071]. 기준을 하나도 충족하지 못했다 |
| 놓친 frame의 주된 원인은? | 사전 범주로는 세 데이터셋 모두 **MIXED**다(SCALE 44% / 36% / 33%로 가장 큼). 출력 관계로 다시 보면 **출력이 대상 밖에 있음(신원 이탈)**이 놓친 손실의 54% / 68% / 68%다 (사후 분석, 공개) |
| 근거가 있는 곳에서 개선했는가? | **아니오.** SCALE·CONSENSUS·PCCARRY 모두 사전 기준을 통과하지 못했다. 기준 시스템은 SGGL 그대로다 |

## 2. 데이터

| 데이터셋 | 상태 | 단위 / 영상 | dev / conf | 극소 | frame | SGGL 작은 객체 J&F |
|---|---|---|---|---|---|---|
| MOSEv2 train (180영상 사전 추출) | 사용 | 139 / 102 | 62 / 77 | 51 | 10,459 | 0.726 |
| VOST val | 사용 | 74 / 44 | 31 / 43 | 11 | 8,396 | 0.487 |
| LVOS v2 valid | 늦게 사용 (수정 4, 첫 300 frame) | 32 / 27 | 16 / 16 | 11 | 8,854 | 0.847 |
| MOSEv1 val | 사용 안 함 | — | — | — | — | 정답이 첫 frame만 있음 |
| SA-V | 사용 안 함 | — | — | — | — | SAM 2 학습 데이터 |
| PUMaVOS C | **봉인** | — | — | — | — | 전송·열람·실행 없음 |

- 모든 manifest는 GT만으로, SAM 2 추론 전에 고정했다(`DATASET_MANIFEST_AND_SPLITS.md`).
- 어댑터 검증:
  - VOST는 prompt 겹침 그림과 연기 시험 4단위로 확인했다.
  - MOSEv2는 분할 파일 sha256을 공식값과 대조했다.
  - Pod 재실행은 diag 행과 완전히 일치했다.

## 3. 결과 요약 (작은 객체, 같은 기계 SGGL 대비 ΔJ&F, 영상 군집 95% CI)

| arm | MOSEv2 | VOST | LVOS v2 | 판정 |
|---|---|---|---|---|
| FLOWVEL (M8 고정) | −0.0142 [−0.0284, −0.0016] | +0.0046 [−0.0181, +0.0272] | −0.0021 [−0.0110, +0.0071] | 일반화 실패 |
| SCALE (스케일만) | +0.0116 [−0.0003, +0.0258] | −0.0073 [−0.0389, +0.0199] | −0.0015 [−0.0473, +0.0329]; dev는 기준 통과, conf 불통과 | 사전 기준 불통과 |
| JOINT (위치 + 스케일) | NOT_RUN | NOT_RUN | NOT_RUN | 위치 성분이 해로워 실행 안 함 |
| PCCARRY (값싼 운동) | −0.0031 [−0.0102, +0.0037] | NOT_RUN | NOT_RUN | 중단 |
| CONSENSUS (존재 합의, 수정 1) | −0.0278 [−0.0585, −0.0010]; conf −0.0503 | 취소 | NOT_RUN | 반증 |
| NULLJIT (영 대조, 수정 3) | −0.0011 [−0.0116, +0.0101] | +0.0089 [−0.0123, +0.0281] (2080) | −0.0073 [−0.0340, +0.0153] | 기준선 |

**세부 문서:**
- `FLOWVEL_GENERALIZATION.md`
- `SCALE_CONTROL_RESULTS.md`
- `CONSENSUS_RESULTS.md`
- `MOTION_ESTIMATOR_ABLATION.md`
- `REGRESSION_AND_FAILURE_AUDIT.md`
- `CROSS_DATASET_FAILURE_TAXONOMY.md`

## 4. 핵심 발견

1. **FLOWVEL은 DAVIS 특화였다.**
   - 세 독립 데이터셋 모두에서 평균이 0 근처이거나 음수다.
   - 극소 객체에서도 이득이 없다.
   - 2080 재현과 단위별 상관이 0.999로, 결과는 기계에 의존하지 않는다.
2. **'스케일' 범주의 실체는 이탈이다.**
   - 스케일 범주 놓친 frame에서 출력 precision 중앙값은 0.07(MOSEv2), 0.00(VOST), 0.36(LVOS v2)이다. '대상 밖'이 59–73%다.
   - '대상 일부만 분할' 가설은 기각됐다(수정 2).
   - crop 크기 조절이 그 범주를 18% / 2% / 34%(LVOS v2: dev 36%, conf 27%)만 회복한 이유가 이것이다.
3. **놓친 손실의 절반 이상이 '출력이 대상이 아닌 곳에 있음'이다.**
   - MOSEv2 54%(64영상), VOST 68%(38영상), LVOS v2 68%(18영상)다.
   - 점수 기반 존재 규칙(CONSENSUS)과 순위화(Stage 1 AUC 0.57–0.65)는 이를 가르지 못했다.
4. **단위 수준 결과는 crop 기하에 대해 혼돈적이다.**
   - 정보 없는 ±15% 흔들림만으로 ±0.05 넘는 단위가 8–9개씩 생긴다(MOSEv2 9 / 8, VOST 9 / 7). LVOS v2에서는 −0.05 미만이 3개였다.
   - 사전 기준 '−0.05 미만 단위 ≤ 1개'는 메커니즘을 가르지 못하는 기준이었다. 다음 사전 등록부터 영 대조 대비 기준으로 바꾼다.
5. **값싼 운동 추정은 RAFT를 대신할 수 있지만, 대신할 효과가 없다.** 위상 상관 오차 중앙값은 3 px이고 지연은 1/14이다. 그러나 PCCARRY는 −0.0031이었다.

## 5. 범주별 결론

| 범주 | 값 |
|---|---|
| CURRENT_INCUMBENT | SGGL |
| FLOWVEL_GENERALIZATION | NOT_GENERALIZED (MOSEv2 −0.014 CI < 0; VOST +0.005, LVOS v2 −0.002, CI가 0을 포함) |
| BEST_NEW_DATASET_EVIDENCE | 출력 관계 분석: 놓친 손실의 54% / 68% / 68% (MOSEv2 / VOST / LVOS v2)가 '대상 밖'. SGGL 재실행과 완전히 일치하는 측정 |
| DOMINANT_FAILURE_MODE | 사전 범주: MIXED (SCALE 최대). 출력 관계: 신원 이탈 (사후) |
| BEST_TRANSLATION_METHOD | 없음. SGGL 유지 (FLOWVEL·PCCARRY 모두 SGGL 이하) |
| BEST_SCALE_METHOD | 없음. SCALE은 사전 기준 불통과 (MOSEv2에서만 영 대조 +0.013, p = 0.09) |
| BEST_JOINT_METHOD | NOT_RUN |
| BEST_PRESENCE_IDENTITY_METHOD | 없음 (CONSENSUS 반증, 순위화 held-out 순이득 ≤ 0) |
| BEST_CAUSAL_CHALLENGER | 없음. 굳이 꼽으면 SCALE (탐색적, 데이터셋 간 부호 불일치) |
| WORST_REGRESSION | CONSENSUS MOSEv2 8bex1hz5/obj1 −0.741 (기권 뒤 재획득 실패) |
| CHEAP_MOTION_RESULT | 추정 일치 양호(3 px, 1/14 지연), 인과 효과 −0.0031 → 채택 안 함 |
| ATTENTION_STATUS | Stage 1 관문에서 중단 (B_att − A_cos recall −0.002 [−0.044, +0.043]) |
| PUMAVOS_C_STATUS | SEALED |
| RUNPOD_SYNC_AUDIT | PASS. `gpu_sync.py audit outputs/milestone9/runpod` exit 0 (06:24), Pod 파일 1,804개 sha256 검증, 충돌 0. 동기화 디렉터리 밖 14개도 일치. Pod는 06:27에 정지(삭제 안 함) |
| COMPUTE_USED | RunPod RTX 4090 7.4시간(가동 23:02–06:27, 실제 연산 약 6.6 h), Ubuntu RTX 2080 약 5 h |
| NOT_RUN_OR_BLOCKED | JOINT (NOT_RUN), PCCARRY VOST / LVOS v2 (NOT_RUN), CONSENSUS VOST / LVOS v2 (취소 / NOT_RUN), VOST·LVOS v2 Stage 1 생성기 진단 (NOT_RUN), R3 attention 순위화 (NOT_RUN), LVOS v2 300 frame 이후 장기 구간과 나머지 70단위 (DEFERRED) |
| NEXT_RESEARCH_PRIORITY | 신원 분리 신호의 오프라인 측정 (`NEXT_DECISION_TREE.md` Q1) |

## 6. 사전 등록에서 벗어난 점 (모두 공개)

| 항목 | 내용 |
|---|---|
| 수정 1 (00:50) | CONSENSUS arm 추가. MOSEv2 dev 재생만으로 설계했고 conf로 판정했다 |
| 수정 2 (02:50) | 면적 진단 추가. 메커니즘 없이 측정만 했다 |
| 수정 3 (03:25) | NULLJIT 영 대조 추가 |
| 수정 4 (05:08) | LVOS v2를 연산 예산 manifest로 추가. 첫 300 frame만 써서 장기 추적 구간은 빠졌다 |
| 사후 범주 | PARTIAL_SEGMENTATION, 출력 관계 분석 |
| VOST 진단 | 2080 속도 때문에 Stage 1 생성기 진단을 하지 않았다. 실패 분류는 SGGL 궤적과 GT 형상만으로 했다 |
| VOST 주 검정 기계 | 결과를 보기 전에 RunPod을 주 검정으로 정했다. 2080은 재현이다 |
| VOST 실패 분류 | 2080 SGGL 궤적 기준이다. 기계 간 차이는 평균 0.0006이다 |
| JOINT | 사전 등록했지만 실행하지 않았다 |
| LVOS 확인 지연 | 02:26에 받았지만 04:58에 확인했다 |

## 7. 운영

| 항목 | 내용 |
|---|---|
| RunPod 구성 | RTX 4090 24 GB, quota 13.6 vCPU, RAM 124 GB, 4 worker × 3 thread (4.39 frame/s 벤치마크, 실제 7–8 frame/s 합산) |
| 지연 / VRAM (단일 worker, 4 동시) | SGGL 계열 0.26–0.37 s/frame, 1.2–1.35 GB. FLOWVEL 0.34–0.43 s/frame, 1.3–1.45 GB |
| 동기화 | Ubuntu tmux `m8:sync9`에서 300초마다 pull. 최종 audit exit 0 |
| Pod 상태 | 06:27 정지(stop). 볼륨 `/workspace`에 LVOS v2 압축과 M8/M9 환경이 남아 있다. 컨테이너 디스크의 추출 데이터는 정지와 함께 사라진다(manifest·sha256으로 재현 가능) |
| 비밀 정보 | API 키, 대시보드 토큰, SSH 키를 출력하거나 커밋하지 않음. 경로와 이름만 문서화 |
| PUMaVOS C | 봉인. Pod의 PUMaVOS clip 15개는 모두 A/B 허용 목록(개수만 대조) |
