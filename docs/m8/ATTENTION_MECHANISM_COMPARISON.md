# M8 attention 메커니즘 비교 (삽입 위치별)

M7의 attention(T2–T4)은 SGGL이 L을 채택한 frame에서 **L logit을 보정**했다. M8은 attention을 기능이 다른 4개 위치에 넣어 시험했다. 모두 같은 동시 부재 frame에서 짝 비교했다(원자료: `stage1_report.json`, `rankers_*.json`).

## 1. 위치별 결과

| 위치 | 구현 | query / key / value | 새 공간 증거를 더하나 | 비교 대조 | 결과 | 판정 |
|---|---|---|---|---|---|---|
| ① 전역 탐색 heatmap | `B_att` (`rescue.heat_contrast`) | query = 현재 frame의 각 cell 특징. key = frame-0 fg token 8개(k-means) + bg token 32개. value = fg/bg 표시. fg에 대한 attention 질량 = 대상 사후확률 | 예 (crop 밖 전역 위치) | A_cos (단일 prototype 코사인, attention 없음) | 위치 정확도 top-1 0.58 = 0.58. recall 0.389 = 0.389 (Δ 0.000 [−0.014, +0.030]) | **이득 없음** |
| ② crop 위치 제어 | `B_att_crop` | ①의 peak에 L crop을 다시 놓는다 | 예 (L 메모리에 새 시야) | A_cos_crop | recall 0.130 vs 0.099. 거짓 양성 0.12 vs 0.14. score>0 규칙의 순 이득 −30.2 vs −38.0 | 약간 낫지만 둘 다 순손실 |
| ③ 경쟁 후보 순위화 | R3 `SetRanker` | 후보 token끼리 self-attention(경쟁) → frame-0 fg/bg token에 cross-attention(신원) → 기권 token | 아니오 (선택만) | R2 로지스틱 (후보를 독립적으로 점수화) | held-out 순 이득: GLOBAL 0.0 (항상 기권) vs −46.6. POOL −11.9 vs −77.6. LOCAL −33.1 vs −43.9 | 로지스틱보다 덜 해롭지만 **양의 이득 없음** |
| ④ 메모리 판독 | `D_ref` | SAM 2 memory attention이 frame-0 메모리만 읽는다 + ①의 peak prompt | 예 | B_att (전체 메모리) | D_ref vs B_att#0(둘 다 후보 1개): recall 0.351 vs 0.359 (Δ −0.008 [−0.016, 0.000]) | **차이 없음** (frame-0 메모리 제한은 효과 없음) |
| (M7) logit 보정 | T2 | frame-0 token이 G/L 특징을 읽어 FiLM | 아니오 | T1 CNN | M7: SGGL 대비 −0.0013 | M7 기각 |
| 미실행 | B5 학습형 localization, B7 메모리 쓰기 attention | — | — | — | 학습 데이터 부족 (REP 23영상, 동시 부재 적중은 5영상) | NOT_RUN |

## 2. 해석

1. **attention이 탐색에서 아무것도 더하지 못한 이유.** 대조 token attention과 단일 prototype 코사인은 같은 frozen 특징 위에서 같은 peak를 고른다. 위치 정확도와 recall이 frame 단위로 일치한다.
   - 실패하는 frame은 두 방법 모두에서 실패한다. 대상이 보이지 않거나 표현이 방해물과 구별되지 않는 경우다.
   - 문제는 attention 방식이 아니다. '어떤 cell이 대상과 닮았나'라는 질문 자체가 풀리지 않는다.
2. **attention이 가장 쓸모 있었던 곳은 순위화(③)다.**
   - 경쟁 구조와 기권 token 덕분에 로지스틱처럼 거짓 양성을 쏟아내지 않았다.
   - 그러나 대상이 있다고 확신할 만큼 판별하지 못해 결국 기권했다. 존재 판별 AUC는 0.6–0.7이다.
3. **위치별 차이.** 같은 attention도 위치에 따라 결과가 달랐다. 그러나 4개 위치 중 SGGL을 넘는 데 필요한 '양의 순 이득'을 낸 곳은 없다.

## 3. 인과 rollout

- **정정 (독립 검토):** B_att는 사전 등록 진출 조건을 문구대로 보면 통과한다.
  - recall 0.389, 4영상 적중, `adv_recall=True`.
  - REP에서 고정한 R0/R1/R3 threshold로 DEV 거짓 양성이 0이다.
  - 그러나 그 threshold에서 DEV 회수도 0이다(아무것도 출력하지 않아 거짓 양성이 0인 형식적인 통과).
- 이 조건을 '양의 순 이득'으로 해석해 attention 단독 인과 arm은 돌리지 않았다. 이 해석은 사전 등록 문구에 없던 것이며, 여기서 공개한다. 돌렸다면 SGGL과 같은 결과(Δ = 0)가 나왔을 것이다.
- R3 GLOBAL은 held-out에서 항상 기권한다. 인과 rollout 결과가 SGGL과 비트 단위로 같으므로 실행해도 얻을 정보가 없다.
- 이 결정은 Stage 1 결과를 따랐다. 사전 규칙이 허용하는 '진단적 인과 실행' 대상(Stage 1 순 이득 양수)에도 attention 계열은 없었다.

**BEST_ATTENTION_DIRECTION:** 존재 판정과 결합된 후보 경쟁 순위화(③). 단, 위치형 실패가 여러 영상에 있는 데이터에서만 시험할 가치가 있다. 현재 데이터에서는 적중 영상이 5개다.
