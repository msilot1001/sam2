# M8 실험 토너먼트 계획 (사전 등록)

고정 파일: `outputs/milestone8/preregistration.json` (M8 출력 전에 생성, `scripts/m8.py: PREREG`). 이 문서는 그 요약이다. 인과 arm 정의는 첫 출력 전에 `outputs/milestone8/arms.json`에 고정한다.

## 데이터

| 세트 | 내용 | 용도 |
|---|---|---|
| DEV | M6 개발 crop 단위 21개 (DAVIS val, PUMaVOS A/B, SegTrack) | 평가 (반복해서 본 데이터라 탐색적) |
| REP | DAVIS 2017 train crop 단위 43개 / 23영상 | 평가 + **학습형 ranker의 학습 데이터** (학습형 결과는 DEV로만 판정) |
| VK | VKITTI2 Scene01/06 28단위 | 고정된 결선 후보만 1회 (서술적, scene 2개) |
| PUMaVOS C | 봉인 | 열지 않음 |
| 새 tiny 데이터 | LVOS v2 / SA-V / VOST 후보 | BLOCKED (다운로드 승인 필요) |

주 지표는 작은 객체(prompt 면적 < 1%) 단위별 J&F 평균(prompt frame 제외)의 SGGL 대비 차이다. 영상 군집 bootstrap 2000회로 CI를 낸다. 기준선은 **같은 기계**의 SGGL이다. 각 단위의 진단 실행(diag)이 SGGL 궤적과 지표를 함께 기록하고, 단위별 기계 배정(M7_SHARD)은 모든 arm에서 같다.

## 단계

| 단계 | 내용 | 산출 |
|---|---|---|
| 0 | M7 기록으로 실패 사례와 상한 재구성, SGGL 재현 확인, 성능 점검 | `FAILURE_EPISODE_ANALYSIS.md`, `COMPUTE_AND_PERFORMANCE_AUDIT.md` |
| 1 | 진단: 실제로 돌아가는 SGGL 궤적의 trigger frame마다 생성기 25개를 비저장 probe로 실행해 recall, 거짓 양성, 위치 정확도, 짝지은 반증 시험을 측정. ranker R0–R3 (R2/R3는 REP 학습, DEV 평가) | `stage1_report.json`, `rankers_*.json`, `STAGE1_DIAGNOSTIC_RESULTS.md` |
| 2 | 인과 rollout: 통과한 생성기(+ranker)로 rescue arm 구성. 출력 + crop 재배치 (+선택적 메모리 기록). M3-vel 대조 | `runs/<arm>/`, `STAGE2_CAUSAL_RESULTS.md` |
| 3 | 결합 3–6개 (구성요소 근거가 있을 때만) | `COMBINATION_RESULTS.md` |
| 4 | 결선 VK 1회, 최종 평가 | `FINAL_PORTFOLIO_ASSESSMENT.md` |

## 규칙

- **Stage 1 → 인과 진출:** recall(J > 0.5) ≥ 0.25, 그리고 같은 실행의 max(Gp, Lp) recall의 2배 이상, 그리고 4영상 이상에서 적중, 그리고 REP에서 고정한 ranker/threshold로 DEV 진짜 부재 frame의 거짓 양성률 ≤ 0.10.
- **인과 판정:**
  - PROMISING: M7 pilot 규칙. ΔJ&F ≥ +0.005, +0.01 이상 상승 영상 ≥ 3, 영상 하나씩 빼기 최솟값 > 0, miami-surf 제외 시 > 0, 최악 단위 ≥ −0.05.
  - STRONG_WIN: ΔJ&F ≥ +0.03이고 CI 하한 > 0. 동시 부재 frame 상한이 +0.028이라 이 데이터에서는 도달할 수 없다는 점을 미리 밝힌다.
  - REJECTED: ΔJ&F ≤ 0, 또는 이득 영상 ≤ 2이면서 하나씩 빼기 최솟값 ≤ 0.
- **반증 시험 (같은 frame, 영상 군집 CI):**
  - B_att > A_cos (attention의 탐색 이득)
  - H_s8 > A_cos (표현 점유율)
  - CPFR > OCLR, 같은 상태·같은 배율 (문맥 보존)
  - C/E > Lp (운동 증거)
- **수정:** arm마다 사전 등록된 수정 1회까지. threshold는 인과 실행 전에 REP에서 고정한다.
- **보고 항목:**
  - 영향: J, F, J&F, 단위별·영상별 결과, 극소(< 0.1%) 분층, 동시 부재 subset 결과
  - 오류: 거짓 재획득(GT가 비었는데 출력이 있는 frame), 최악 단위
  - 비용: 지연, 최대 VRAM, 추가 SAM 2 호출 수

## 다중 비교

Stage 1에서 약 25개 생성기와 10개 짝 비교를 같은 개발 영상에서 한다. Stage 1 결과는 모두 탐색적이다. 인과 승자도 새 데이터나 봉인 세트로 확인되기 전까지는 탐색적이다.
