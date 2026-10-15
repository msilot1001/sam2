# M10 amend10: 학습 선택 헤드 오프라인 게이트 (dev)

- 학습: ['outputs/milestone10/runpod/runs/OR0K_train'] (50 단위 / 36 영상 / 13777 프레임, scope 12811). 영상 단위 5-fold OOF 점수로만 tau* 선택. dev 점수 = 전체 학습 행 재적합 모델.
- dev: {'lvosv2': 16, 'mosev2': 62, 'vost': 31} 단위. dJ = 전환 프레임에서 cJ(pick) − cJ(k_sggl), 그 외 0; GT 존재 프레임 평균, 단위→영상→영상 동일 가중; 영상 bootstrap 95% CI (10k).
- 통과 = (1) MOSEv2+VOST 합산 정밀도 ≥ 0.70, (2) VOST·LVOS CI 하한 > 0, (3) MOSEv2 점추정 ≥ −0.005, (4) null 헤드보다 우세. tau* 없음 = 실패.
- 하이퍼파라미터 고정: {'opt': 'Adam', 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 30, 'batch': 512, 'seed': 0, 'loss': 'MSE to cJ', 'hidden': {'b': [64, 64], 'c': [128, 128]}, 'folds': 5, 'fold_seed': 0, 'null_shuffle_seed': 0, 'bootstrap_seed': 10}

## H1: 뷰 보정 (G vs L, 각 뷰의 SAM 2 선택 사이 전환)

| 모델 | tau* | 학습 OOF 정밀도 | dev 합산 정밀도 (MOSEv2+VOST) | MOSEv2 dJ [CI] | VOST dJ [CI] | LVOS v2 dJ [CI] | 전환 수 (M/V/L) | 통과 |
|---|---|---|---|---|---|---|---|---|
| R_piou | — | — | — | — | — | — | — | — |
| a | — | — | — | — | — | — | — | 실패 (tau* 없음) |
| b | 0.415 | 0.706 | 0.329 | -0.002 [-0.013, +0.008] | +0.010 [+0.003, +0.020] | +0.026 [+0.000, +0.066] | 109/101/150 | 실패 |
| c | — | — | — | — | — | — | — | 실패 (tau* 없음) |
| null_b | — | — | — | — | — | — | — | — |
| null_c | — | — | — | — | — | — | — | — |

선택: **없음**

## H2: 뷰 내 선택 (SGGL 출력 뷰의 3 가설)

| 모델 | tau* | 학습 OOF 정밀도 | dev 합산 정밀도 (MOSEv2+VOST) | MOSEv2 dJ [CI] | VOST dJ [CI] | LVOS v2 dJ [CI] | 전환 수 (M/V/L) | 통과 |
|---|---|---|---|---|---|---|---|---|
| R_piou | — | — | — | — | — | — | — | — |
| a | — | — | — | — | — | — | — | — |
| b | — | — | — | — | — | — | — | 실패 (tau* 없음) |
| c | — | — | — | — | — | — | — | 실패 (tau* 없음) |
| null_b | — | — | — | — | — | — | — | — |
| null_c | — | — | — | — | — | — | — | — |

선택: **없음**

## 이탈/정상 프레임 dJ (tau*)

| 모델 | 가설 | 데이터셋 | 이탈 프레임 dJ | 정상 프레임 dJ | 정밀도 |
|---|---|---|---|---|---|
| b | H1 | LVOS v2 | +0.145 | +0.000 | +1.000 |
| b | H1 | MOSEv2 | +0.017 | -0.012 | +0.147 |
| b | H1 | VOST | +0.011 | -0.001 | +0.525 |

정밀도–tau 곡선 (학습 OOF, dev)과 fold 배정은 outputs/milestone10/reports/head_offline.json. 재적합 모델: outputs/milestone10/head/<model>.pt.
