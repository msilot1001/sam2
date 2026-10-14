# M8 결합 실험 결과 (Stage 3)

**실행 조건:**
- 결합 후보 16개 중 실행할 것은 [`CROSS_FAMILY_COMBINATIONS.md`](CROSS_FAMILY_COMBINATIONS.md)의 실행 규칙에 따라 골랐다(구성요소가 Stage 1/2를 통과해야 한다).
- 모두 RunPod 4090에서 실행했다. 비교 대상은 같은 기계의 SGGL과 각 구성요소다. 작은 객체 57단위 / 34영상.
- 원자료: `causal_report.json`, `pairwise_report.json`.

## 1. 무엇을 실행했고 왜

| 결합 | 구성요소 | 정보원 | 실행 근거 | 고정 시점 |
|---|---|---|---|---|
| K_OCLR_M3 (행렬 K6 변형) | RESC_OCLR (F 척도 rescue) + M3vel (중심 예측) | 넓힌 L 시야의 회수 + 출력 중심 운동 | 두 구성요소가 겨냥하는 사례가 다르다: drone 이탈 vs skate-park·sheep 재획득 | 18:38 KST, 모든 구성요소의 인과 결과가 나오기 전 (M3vel 부분 결과만 있었음) |
| K_OCLR_FLOWVEL | RESC_OCLR + FLOWVEL | 회수 + 광류 crop 제어 | 같음 | 18:38 KST (FLOWVEL 결과 이전) |
| K_ADMIT_FLOWVEL | RESC_OCLR_ADMIT + FLOWVEL | 메모리에 기록되는 회수 + 광류 crop 제어 | rescue 변형 중 평균이 양수인 것은 메모리 기록뿐이다 | **19:43 KST: FLOWVEL·RESC_OCLR_ADMIT 결과를 본 뒤 (사후, 공개)** |

**실행하지 않은 결합과 이유:**

| 결합 | 실행하지 않은 이유 |
|---|---|
| K1, K2, K4, K5, K7, K10, K11, K12, K16 | 전역 탐색·attention 생성기를 순위화해도 held-out 순이득 > 0을 얻지 못했다(Stage 1). 그래서 구성요소 조건을 통과하지 못했다 |
| K3, K15 | 광류 crop 생성기와 속도 beam 생성기의 recall이 ≈ 0이었다 |
| K13 | D_ref가 B_att#0보다 낫지 않았다 |
| K8 (rescue + 메모리 기록) | 단일 arm으로 실행했다(RESC_OCLR_ADMIT) |
| K9 (지연 확정) | 실행하지 않았다(NOT_RUN). §3 참고 |

## 2. 결과

| 결합 | ΔJ&F vs SGGL | 95% CI | 영상 ↑/↓ | 최악 단위 | 거짓 재획득 (vs SGGL 93) | 가장 강한 구성요소 대비 | 시너지 |
|---|---|---|---|---|---|---|---|
| K_OCLR_M3 | +0.0071 | [−0.0009, +0.0172] | 3/1 | miami-surf obj5 −0.058 | 100 | vs M3vel +0.0025 [−0.0032, +0.0112]; vs RESC_OCLR +0.0072 [+0.0003, +0.0159] | **없음** (M3vel 대비 CI가 0 포함) |
| K_OCLR_FLOWVEL | +0.0121 | [−0.0010, +0.0295] | 4/3 | miami-surf obj5 −0.058 | 99 | vs FLOWVEL −0.0009 [−0.0043, +0.0027] | **없음** (약간 나쁨) |
| K_ADMIT_FLOWVEL | +0.0103 | [−0.0031, +0.0278] | 4/3 | miami-surf obj4 −0.127 | 105 | vs FLOWVEL −0.0027 [−0.0084, +0.0013] | **없음, 위험 증가** |

계산량 비교: 결합은 동시 부재 frame에서만 SAM 2를 1회 더 호출한다(평균 +0.09 호출/frame). FLOWVEL 단독은 SAM 2 추가 호출이 0이다. 같은 계산량에서도 결합은 이득이 없다.

## 3. 해석

1. **운동 제어(C)와 rescue 생성(F)은 상보적이지 않았다.**
   - FLOWVEL·FLOWCARRY·CAMCARRY는 놓친 frame(GT가 있는 동시 부재)을 131에서 62–63으로 줄인다.
   - 그러고 나면 rescue가 고칠 frame이 거의 남지 않는다. 남은 rescue는 대부분 방해물을 잡는다. 회수는 6 frame인데 거짓 재획득이 6–12 늘었다.
   - FLOWVEL과 결합하면 '서로 다른 실패를 겨냥한다'는 사전 가설이 성립하지 않았다. drone 유형(이탈)과 skate-park 유형(재획득)을 운동 제어가 이미 둘 다 해결했기 때문이다.
2. **메모리 기록은 결합에서도 꼬리 위험을 만든다.** 잘못된 rescue가 기록되면 이후 frame이 계속 틀린다. miami-surf obj4에서 −0.127이었다.
3. **attention/비-attention 결합은 실행하지 못했다.** attention 순위화의 held-out 이득이 양수라는 구성요소 근거가 없어서, 사전 규칙에 따라 실행하지 않았다.
4. **가장 효과적인 결합:** 없음. 가장 효과적인 단일 방법(FLOWCARRY/CAMCARRY)보다 나은 결합이 없다.
5. **아직 시험할 가치가 있는 결합:** 운동 제어 + '검증된 경우에만 메모리 기록'(K9 지연 확정형)이다. ORACLE_POOL_ADMIT 결과를 보면 판정이 완벽할 때 메모리 기록이 +0.003을 더한다(+0.0169 vs +0.0136). 다만 그런 판정을 배포 가능한 신호로 만들 방법은 아직 없다.
