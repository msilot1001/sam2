# M8 Stage 1 진단 결과 (생성기 25종, 64단위, 비저장 probe)

**원자료와 재생성:**
- 실행 결과: `outputs/milestone8/diag/` (Ubuntu RTX 2080, 53단위), `outputs/milestone8/runpod/diag/` (RunPod RTX 4090, 11단위)
- 요약: `stage1_report.json`, `stage1_generators.csv`, `rankers_{GLOBAL,LOCAL,POOL}.json`
- 재생성: `uv run python scripts/m8_analysis.py stage1`, `... rankers <gens> <name>`

**사전 등록 일탈 (모두 공개):**
- 수정 1 (`preregistration_amend1.json`): F view에 점 prompt 변형(`*_pt`)을 추가했다. 단위 1개를 본 뒤에 정했다. 수정 1이 주 시험으로 지정한 `_pt` 짝 비교는 처음 분석에서 빠졌고, 검토 후 추가했다(§2).
- 진출 조건 해석: B_att는 조건을 문구대로 보면 통과했다(형식적인 통과, `ATTENTION_MECHANISM_COMPARISON.md` §3).
- 수정 2 (`preregistration_amend2.json`): H_s8 특징 버그를 고쳤다. shard 0 표를 본 뒤에 정했다.
- ranker 교차 적합: 사전 등록은 REP→DEV 한 방향이었다. DEV→REP 방향을 추가했다.
- threshold 규칙: 'REP에서 FP ≤ 0.10을 지키면서 순 이득이 가장 큰 값'으로 정했다. 사전 등록 문구는 'FP ≤ 0.10'뿐이었다.
- trigger frame: 동시 부재 486 (GT 있음 131 / GT 없음 355), 불일치 242, 정기 399.

## 1. 생성기별 결과 (동시 부재 frame)

recall = 후보 중 하나라도 J > 0.5인 비율 (GT 있음 131 frame 기준). 원시 FP = GT가 없는 355 frame에서 후보가 '존재'를 출력한 비율이다. 순위화나 판정 규칙을 적용하기 전 값이다.

| 생성기 | 계열 | recall | 평균 최고 J | 적중 영상 (frame) | 원시 FP |
|---|---|---|---|---|---|
| A_cos (prototype 코사인 → point prompt, top-3) | A | 0.389 | 0.365 | drone 40, kite-surf 10, lady-running 1 | 1.00 |
| B_att (대조 token attention → point prompt) | B | 0.389 | 0.371 | drone 39, kite-surf 10, dog-gooses 1, lady-running 1 | 1.00 |
| D_ref (B peak + frame-0 메모리만) | D/B | 0.351 | 0.319 | drone 36, kite-surf 9, lady-running 1 | 1.00 |
| G_neg (B 양성 + 방해물 음성 점) | G | 0.305 | 0.299 | 3영상 | 1.00 |
| H_s8 (stride-8 FPN 매칭) | H | 0.237 | 0.225 | 3영상 | 0.99 |
| F_oclr8_L (L 메모리, 8-cell crop, 광류 중심) | F | 0.176 | 0.147 | drone 22, lady-running 1 | **0.014** |
| F_cpfr8_L (같은 배율 foveation) | F | 0.168 | 0.139 | drone 22 | 0.110 |
| B_att_crop / A_cos_crop (heatmap peak로 L crop 배치) | B/A | 0.130 / 0.099 | — | drone, skate-park, kite-surf | 0.12 / 0.14 |
| Gp / Lp (M7 보호 메모리) | D (M7) | 0.069 / 0.061 | — | kite-surf, skate-park / dog-gooses, kite-surf | 0.25 / 0.08 |
| C_flowbox (광류 box → G box prompt) | C | 0.046 | — | drone | 0.80 |
| F_*_G, F_*_pt (G 메모리 또는 점 prompt) | F | 0.000–0.038 | — | — | 0.006–0.95 |
| C_flowcrop (광류 중심 L crop) | C | 0.015 | — | 2영상 | 0.003 |
| E_beam1/2 (속도 외삽 beam) | E | 0.000 | — | — | 0.003 |
| **전체 생성기 합집합 (oracle)** | | **0.511** | | | |

**위치 정확도 (GT 있는 동시 부재 frame에서 heatmap peak이 대상 위인 비율):**

| 생성기 | top-1 | top-3 |
|---|---|---|
| A_cos | 0.58 | 0.68 |
| B_att | 0.58 | 0.64 |
| H_s8 | 0.15 | 0.30 |

## 2. 사전 등록 반증 시험 (같은 frame 짝 비교, 영상 군집 95% CI)

| 시험 | Δ | CI | 판정 |
|---|---|---|---|
| B_att − A_cos recall (attention의 탐색 이득) | 0.000 | [−0.014, +0.030] | **기각**: 대조 token attention은 단일 prototype보다 낫지 않다 |
| B_att − A_cos 위치 정확도 top-1 | 0.000 | [−0.106, +0.084] | 같음 |
| H_s8 − A_cos recall / 위치 정확도 | −0.153 / −0.435 | [−0.242, 0.000] / [−0.612, −0.167] | **기각**: stride-8 특징은 더 나쁘다. 표현 점유율 가설을 지지하지 않는다 |
| CPFR8 − OCLR8 recall, G 메모리 | +0.008 | [0.000, +0.031] | 차이 없음 |
| CPFR8 − OCLR8 recall, L 메모리 | −0.008 | [−0.031, 0.000] | 차이 없음 |
| CPFR8 − OCLR8 최고 J, L 메모리 | −0.008 | [−0.031, +0.001] | 차이 없음 → **문맥 보존 가설을 지지하지 않는다** |
| CPFR4 − OCLR4 recall, G 메모리 | −0.008 | [−0.040, +0.014] | 차이 없음 |
| **(수정 1의 주 시험) CPFR8_pt − OCLR8_pt recall, G 메모리** | −0.031 | [−0.121, 0.000] | foveation이 낫지 않다. 오히려 낮은 쪽 |
| **CPFR8_pt − OCLR8_pt recall, L 메모리** | −0.023 | [−0.051, 0.000] | 같음 |
| **CPFR4_pt − OCLR4_pt recall, G 메모리** | −0.023 | [−0.075, 0.000] | 같음 |
| G_neg − B_att#0 recall (둘 다 후보 1개) | −0.053 | [−0.097, 0.000] | 음성 점이 약간 해롭다 (선행연구 보고와 같은 방향). 정정: 처음에는 B_att top-3 묶음과 비교해 −0.084로 과대 추정했다 |
| D_ref − B_att#0 recall (둘 다 후보 1개) | −0.008 | [−0.016, 0.000] | 차이가 거의 없다. 메모리를 frame 0으로 제한해도 효과 없음. 정정: 처음 −0.038은 top-3 묶음과 비교한 착오였다 |
| C_flowcrop − Lp, E_beam1 − Lp recall | −0.046 / −0.061 | CI가 0을 포함하거나 0 이하 | 운동 생성기는 새 증거를 더하지 못한다. **일탈:** 사전 등록은 '실제 L view 대비'였다. 동시 부재 frame에서 L은 정의상 부재(recall 0)라, 정보가 있는 Lp와 비교했다 |

**추가 관찰:**
- **잘못된 메모리 좌표:** prompt 없는 F view를 G 메모리로 돌리면 대상을 '존재'로 거의 판정하지 않는다(recall ≤ 0.015). 확대나 왜곡이 메모리 좌표계와 맞지 않으면 SAM 2 memory attention이 대상을 다시 찾지 못한다.
- **점 prompt의 대가:** 점 prompt를 주면 어떤 view든 거의 항상 '존재'를 출력한다(원시 FP 0.92–1.00).
- **광류 운반 box:** GT 있는 동시 부재 frame에서 GT와 겹치는 비율이 19%뿐이다(shard 0 측정). 대상이 사라지면 box가 배경 흐름을 따라 밀려간다.

## 3. 순위화: 생성은 되는데 '구조할지 말지'를 정하지 못한다

| 생성기 묶음 | R0 SAM 2 score | R1 학습 없음 | R2 로지스틱 | R3 set-attention |
|---|---|---|---|---|
| GLOBAL (A, B, D, G) | −9.0 (0 / 9) | −28.9 (0 / 32) | −46.6 (2 / 48) | 0.0 (0 / 0: 항상 기권) |
| LOCAL (F_L, crop 배치, Lp, Gp) | −41.0 (4 / 44) | −41.1 (7 / 46) | −43.9 (25 / 65) | −33.1 (26 / 53) |
| POOL (25종 전부) | −13.0 (0 / 13) | −28.4 (2 / 32) | −77.6 (26 / 99) | −11.9 (0 / 12) |

- 표의 숫자는 교차 적합 held-out 순 이득(frame J 합)이다. 괄호는 (회수 frame / 거짓 양성 frame)이다.
- 순 이득은 회수 frame의 +J에서 진짜 부재 frame에 무언가를 출력한 비용 −1을 뺀 값이다.

**결과: held-out 순 이득이 양수인 ranker는 없다. 사전 등록 진출 조건을 통과한 생성기 계열도 없다.**

**판별력 (AUC, 학습 없음, 동시 부재 frame):**

| 판별 대상 | AUC |
|---|---|
| frame 수준 '대상 존재 vs 진짜 부재' (후보 최대 SAM 2 score 또는 frame-0 코사인) | 0.60–0.71 |
| frame 안에서 '맞는 후보(J > 0.5) vs 틀린 후보(J < 0.1)' (frame-0 코사인, GT 있는 frame의 전체 풀 후보) | **0.815** (`m8_analysis.py auc` → `auc_report.json`) |

**해석:**
- 후보 생성은 회수율 51%까지 가능하고, 후보 사이의 순위도 어느 정도 매길 수 있다.
- 병목은 그 frame에 대상이 정말 있는지(존재)를 판정하는 일이다.
- 진짜 부재 frame이 2.7배 많다. AUC 0.7 수준으로는 거짓 재획득 비용이 회수 이득을 넘는다.
- 전체 생성기 합집합의 적중 frame 67개 중 51개(76%)가 drone·kite-surf 두 영상에 몰려 있다. 그래서 학습형 ranker가 한 split에서 배운 것이 다른 split로 옮겨가지 않는다.

## 4. 판단 규칙 없이 양의 순 이득을 낸 유일한 조합

규칙: SAM 2 score > 0 이면 채택 (적합한 파라미터 없음). 재생 결과(동시 부재 486 frame)는 다음과 같다.

| 생성기 | 채택 | 회수 | 거짓 양성 | 순 이득 | 회수 영상 |
|---|---|---|---|---|---|
| **F_oclr8_L** | 31 | 23 | 5 | **+14.3** | drone 22, lady-running 1 |
| C_flowcrop | 7 | 2 | 1 | +0.9 | 2영상 |
| F_oclr4_G | 11 | 2 | 2 | +0.6 | 2영상 |
| F_cpfr8_L (같은 배율 foveation) | 67 | 22 | **39** | −20.8 | drone 22 (거짓 양성: miami-surf 33) |
| B_att_crop | 66 | 17 | 44 | −30.2 | 3영상 |

- **foveation의 실패 방식:** 같은 배율 foveation은 같은 대상을 회수한다. 그러나 주변 문맥이 보이면서 비슷한 방해물(miami-surf 서퍼들)을 '존재'로 잡는 거짓 양성이 8배로 늘었다.
- **Stage 2 진단적 인과 arm:** 문맥을 지운 crop이 존재 판정을 보수적으로 만든다는 관찰을 바탕으로 RESC_OCLR(+ADMIT)와 RESC_CPFR를 '진단적 인과 실행'으로 정해 두었다(`arms.json`).
  - 사전 진출 조건(4영상 이상)은 통과하지 못했다.
  - 판정에는 M7 pilot 규칙을 그대로 적용한다.

## 5. 비용 (diag, trigger frame당 생성기 31종 전부)

| 기계 | SGGL 1 frame | 광류 1 frame | 생성기 31종 / trigger frame | 최대 VRAM |
|---|---|---|---|---|
| 2080 (혼합 평균, GPU 공유 구간 포함) | 0.59 s | 0.047 s | 6.0 s | < 2 GB |
| 2080 단독 | 0.45 s | — | 2.9 s | — |
