# 표 5. 주 결과: 공식 SAM 2.1 (B0) / P5 / SGGL

MOSEv2·VOST·LVOS v2는 M10 dev split(같은 frame 짝, 영상 동일 가중, 영상 bootstrap CI). ↑/↓ = Δ > +0.01 / < −0.01인 영상 수, LOVO min = 영상 하나씩 뺀 평균의 최솟값. 빈 칸 = 그 milestone에서 측정하지 않음.

| 데이터셋 | 단위/영상 | B0 | P5 | SGGL | SGGL − B0 [CI] | ↑/↓ | LOVO min | SGGL − P5 [CI] | ↑/↓ | LOVO min | P5 − B0 [CI] | ↑/↓ | LOVO min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOSEv2 | 62/48 | 0.671 | 0.684 | 0.725 | +0.054 [+0.009, +0.109] | 23/8 | +0.037 | +0.041 [+0.007, +0.082] | 23/8 | +0.028 | +0.013 [-0.008, +0.046] | 5/4 | -0.001 |
| VOST | 31/19 | 0.537 | 0.545 | 0.500 | -0.036 [-0.082, -0.002] | 5/8 | -0.041 | -0.044 [-0.090, -0.007] | 6/9 | -0.050 | +0.008 [-0.002, +0.027] | 1/1 | -0.001 |
| LVOS v2 | 16/12 | 0.813 | 0.815 | 0.825 | +0.012 [-0.069, +0.127] | 2/5 | -0.036 | +0.010 [-0.068, +0.121] | 2/5 | -0.037 | +0.002 [+0.000, +0.006] | 1/0 | +0.001 |
| DAVIS val + PUMaVOS B (M6.1 관문 세트)ᵃ | 67/36 |  | 0.895 |  |  |  |  | +0.004 [+0.002, +0.007] | 12/0 |  | +0.016 [+0.001, +0.038] | 5/0 |  |
| DAVIS DEV + REP (M7, 작은 객체)ᵇ | 57/34 |  |  |  |  |  |  | +0.011 [-0.010, +0.027] |  |  |  |  |  |
| VKITTI2 (M7)ᶜ | 28/2 scene |  |  |  | +0.034 [+0.024, +0.044] (J) | 24/1 |  | +0.017 [+0.009, +0.025] | 22/1 |  | -0.001 [-0.003, +0.000] (J) | 0/1 |  |

출처: `outputs/milestone10/reports/baselines.json` (MOSEv2·VOST·LVOS v2), `outputs/milestone61/comparison_report.json` (ᵃ), `outputs/milestone7/replay_report.json` (ᵇ), `outputs/milestone7/vk_report.json` (ᶜ). 해설 문서: `docs/m10/BASELINES.md`, `docs/m61/EXPERIMENT_COMPARISON.md`, `docs/m7/FINAL_ALGORITHM_DECISION.md`.

- ᵃ J&F, 단위 = 객체(DAVIS val + PUMaVOS B 전체 객체), P5 = `JF_B1`. SGGL = A2x. ↑/↓는 객체 수(±0.01), LOVO 없음. M6.1에서 B0는 P5 대비 -0.016 [-0.038, -0.001]만 측정했다(B0·SGGL 절대값과 SGGL − B0는 기록 없음). P5 − B0 칸은 이 값의 부호를 바꾼 것이다(CI 양 끝도 바꿈).
- ᵇ J&F, M7 DEV(M6 dev crop 객체) + REP(DAVIS 2017 train crop 객체) 중 작은 crop 객체, 영상 CI. M7은 이 세트에서 B0를 다시 돌리지 않았다. 여러 번 본 개발 데이터라 탐색적이다.
- ᶜ VKITTI2 Scene01, Scene06 (scene별 SGGL − P5: Scene01 +0.017, Scene06 +0.020). SGGL − P5는 J&F, SGGL − B0와 P5 − B0는 **J만** 기록되어 있다(표시 (J)). CI는 객체 단위 서술용이고 실효 표본은 scene 2개다. ↑/↓는 객체 수. M2–M4 개발 scene이다.
- 행마다 실행 기계가 다르다. 기계가 다른 행의 절대값은 서로 비교하지 않는다(같은 행 안의 차이만 짝지은 비교).
