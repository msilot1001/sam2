# M8 최종 포트폴리오 평가 (2026-10-14)

- **브랜치:** `exp/m8-diverse-algorithm-portfolio`. 병합하지 않는다.
- **원자료:** `outputs/milestone8/` — `causal_report.json`, `causal_per_unit.csv`, `pairwise_report.json`, `stage1_report.json`, `rankers_*.json`, `arms.json`, `preregistration*.json`
- **재생성:** `uv run python scripts/m8_analysis.py stage1|rankers|causal|pairwise`

## 1. 한 줄 결론

동시 부재(두 view가 모두 대상을 놓침)를 고치는 데 효과가 있었던 것은 **새 후보를 찾아 대상을 되찾는 방법이 아니었다.** 효과가 있었던 것은 **놓친 동안 crop을 영상 좌표에 고정하지 않고 장면의 움직임에 맞춰 옮겨 두는 것**이었다.

**결과:**
- 출력 전에 고정한 FLOWVEL: SGGL 대비 +0.0130 [+0.0010, +0.0285], 5영상 상승, 재앙적 손실 없음
- 결과를 본 뒤 ablation으로 만든 FLOWCARRY와 CAMCARRY: +0.0141, +0.0137 (FLOWVEL과 구별되지 않음)
- 이 크기는 GT로 후보를 고르고 존재 여부까지 GT로 판정한 rescue의 상한(ORACLE_POOL +0.0136)과 같다.

**약한 점 (독립 검토, `REVIEW_raw.md`):**
1. 배포 가능 arm 11개에 대한 Bonferroni 보정을 하면 CI 하한이 약 0이 된다(FLOWVEL −0.0011).
2. 이득의 64–69%가 drone·kite-surf 두 영상에서 나온다. 두 영상을 빼면 +0.004–0.006이다.
3. VK에서는 효과가 없다. VK의 SGGL은 GT가 있는 동시 부재 frame이 89/1,736(5.1%)으로 DEV/REP(131/5,082, 2.6%)보다 많다. 그러나 VK 진단(§6) 결과, 이 frame은 거의 전부 두 차량이 초기보다 6배 커지는 크기 변화에서 나온다. crop을 옮겨서 고칠 수 있는 유형이 아니다.

그래서 **SGGL을 대체한다고 판정하지 않는다.**

## 2. 실제로 구현하고 실행한 것

| 단계 | 내용 | 규모 |
|---|---|---|
| 0 | M7 기록으로 동시 부재 사례 재구성 (486 frame, 131 GT 있음, 사례 40개), 상한 +0.028 | CPU |
| 1 | 생성기 25종(9계열 중 A, B, C, D, E, F, G, H의 8계열)을 살아 있는 SGGL 궤적의 trigger frame 1,127개에서 비저장 probe로 실행. 순위화기 R0–R3 교차 적합 | 64단위, 2080 + 4090 |
| 2 | 인과 arm: M3vel, FLOWVEL, RESC_OCLR(+ADMIT), RESC_CPFR, ORACLE_POOL(+ADMIT), ablation 3개 (FV_PRESENT, FLOWCARRY, CAMCARRY), 계산 재현 FLOWVEL_U (같은 코드·데이터를 다른 GPU에서) | arm 11개 × 64단위 |
| 3 | 결합 3개: K_OCLR_M3, K_OCLR_FLOWVEL, K_ADMIT_FLOWVEL | 3 × 64단위 |
| 4 | VK 결선: SGGL, FLOWVEL, M3vel, FLOWCARRY | 4 × 28단위 |

모든 인과 비교는 같은 기계의 SGGL과 했다.

## 3. 계열별 실험 범위

| 계열 | 실험한 것 | 결과 |
|---|---|---|
| A 전역 탐색 | A_cos, A_cos_crop | 생성 recall 0.39, 그러나 존재 판정 실패 |
| B attention | B_att(탐색), B_att_crop(crop 배치), R3(경쟁 순위화), D_ref(메모리 판독) | 4개 위치 모두 이득 없음 (`ATTENTION_MECHANISM_COMPARISON.md`) |
| C 운동 | RAFT 광류: C_flowcrop/box(생성), FLOWVEL/FLOWCARRY/FV_PRESENT(crop 제어), CAMCARRY(카메라 보정) | **유일하게 효과가 있었던 계열** |
| D 메모리 | Gp/Lp, D_ref, ADMIT(메모리 기록) | 기록은 장기 효과를 만들지만 꼬리 위험(최악 −0.116, −0.127) |
| E 다중 가설 | E_beam(속도 beam) | recall 0. 독립 상태 분기(E3)는 실행하지 않음 |
| F 척도·시야 | OCLR vs CPFR (같은 배율, G/L 메모리, prompt 유무) | foveation은 이득이 없고 해로움 |
| G prompt | point/box/음성 point prompt | 점 prompt는 항상 '존재'를 출력. 음성 점은 약간 해로움 |
| H 표현 | stride-8 FPN 매칭 | 더 나쁨 |
| I 구조 | 광류 거리·크기 비율을 순위 특징으로 사용 | 순위화 실패에 포함 |

## 4. 결정 표

| 범주 | 판정 | 근거 |
|---|---|---|
| **CURRENT_INCUMBENT** | **SGGL** | 대체 판정에 필요한 독립 확인이 없다 (새 데이터 / C) |
| **BEST_CAUSAL_CHALLENGER** | **FLOWVEL** (출력 전 고정. 부재 중 crop 운반) | +0.0130 [+0.0010, +0.0285], 5/2영상. 사후 ablation인 FLOWCARRY(+0.0141)·CAMCARRY(+0.0137)와 구별되지 않는다(짝 비교 차이 ≤ 0.0011). 단서: Bonferroni 보정 시 하한 ≈ 0, 이득의 69%가 두 영상, VK 효과 없음. 2080 실행(+0.0161)은 같은 코드·데이터의 **계산 재현**일 뿐 독립 재현이 아니다 |
| **BEST_ATTENTION_DIRECTION** | 후보 경쟁 순위화 + 기권 (R3) | 로지스틱보다 덜 해롭지만 양의 이득이 없다. 데이터가 커질 때만 재시도 |
| **BEST_NON_ATTENTION_DIRECTION** | 두 view가 모두 부재일 때 crop을 장면 움직임에 맞춰 옮기기 | '카메라 운동 보정'은 가설이다. 카메라 운동을 따로 기록하지 않았다. 부재 중에는 운반 box가 대상 위에 있지 않으므로 FLOWCARRY ≈ CAMCARRY는 구성상 당연하다. 지금 구현은 세 형태 모두 전체 frame RAFT가 필요하다(지연 +14%) |
| **BEST_CROSS_FAMILY_COMBINATION** | 없음 | 결합 3개 모두 가장 강한 구성요소를 넘지 못했다 (K_OCLR_FLOWVEL − FLOWVEL −0.0009) |
| **HIGHEST_VALUE_UNRESOLVED_HYPOTHESIS** | '검증된 rescue만 메모리 기록' (근거 약함) | ORACLE_POOL_ADMIT − ORACLE_POOL = +0.0032 [−0.0001, +0.0099]. 이 차이는 SGGL 위에서 잰 것이고, 운동 운반 위에서는 재지 않았다. 반대 근거: K_ADMIT_FLOWVEL은 FLOWVEL보다 −0.0027이었고 최악 단위는 −0.127이었다 |
| **NEXT_EXPERIMENT_PRIORITY** | ① VK 진단(실행함, §6): VK의 놓친 frame은 대상이 6배 커지는 크기 변화(2단위). 크기 적응 crop은 별도 후보. ② 새 극소 대상 데이터(승인 필요)에서 **FLOWVEL 하나를** 사전 등록해 확인. ③ 그다음 C 1회 | `NEXT_RESEARCH_DECISION_TREE.md` |
| **SEALED_TEST_STATUS** | PUMaVOS C 봉인 유지 | Worker에 전송하지 않았다. manifest의 C 행도 제거했다 |
| **COMPUTE_USED** | 2080 ≈ 6.5 GPU-h (VK 진단 포함). RunPod 4090 Pod 4.8 h (GPU ≈ 3.5 h). Vast 0 | `COMPUTE_AND_PERFORMANCE_AUDIT.md` §4 |
| **NOT_RUN_ITEMS** | B5, B7, C4(라이선스), C5, D3, E2/E3, G2, H2/H3, A3/A4, K9 등 | `EXPANDED_CANDIDATE_PORTFOLIO.md` §3 |

## 5. 주요 음성 결과

1. **rescue 계열(생성 → 순위 → 출력):**
   - 회수율 51%까지 생성할 수 있었다.
   - 그러나 '그 frame에 대상이 정말 있는가'를 판별하지 못했다. AUC 0.6–0.7이고, 진짜 부재 frame이 2.7배 많다.
   - 그래서 순위화기 R0–R3의 held-out 순이득이 모두 0 이하였다. 배포형 rescue는 이득이 없다.
2. **foveation (첨부 보고서의 1순위 가설):** 같은 배율에서 회수 이득이 0이고, 방해물 거짓 양성이 늘었다. 인과 rollout에서 최악 단위 −0.33이었다.
3. **attention:** 4개 위치 모두 attention 없는 대조와 같거나 나빴다.
4. **feature occupancy 가설:** stride-8 특징은 더 나빴다. 같은 배율의 crop과 foveation도 같았다.
5. **결합:** 운동 보정이 놓친 frame의 절반 이상을 미리 없애기 때문에, 남은 rescue는 대부분 방해물을 잡는다.

## 6. 통계와 집중도

- **FLOWCARRY의 이득은 5개 영상에서 나온다 (영상 평균 Δ):**

  | 영상 | 영상 평균 Δ | 해당 단위 Δ |
  |---|---|---|
  | kite-surf | +0.135 | obj2 +0.135 |
  | drone | +0.126 | obj1 +0.380 |
  | skate-park | +0.086 | obj2 +0.171 |
  | lady-running | +0.037 | obj2 +0.037 |
  | sheep | +0.020 | obj4 +0.061 |

  - 영상 하나씩 빼기 최솟값 +0.008. miami-surf를 빼도 +0.015.
  - −0.01 아래로 떨어진 영상은 scooter-board 하나다(−0.031).

- **탐색적 결과다.**
  - 개발 영상은 M6 이후 반복 검사됐다.
  - M8에서 arm 15개를 같은 영상에서 비교했다.
  - FLOWCARRY와 CAMCARRY는 FLOWVEL 결과를 본 뒤 ablation으로 고정했다.
- **VK (scene 2개, 개발 scene):** 효과가 없다(+0.0002). 이유는 아래 'VK 진단'에 있다. VK에서 놓친 frame은 위치가 아니라 크기 변화 때문이다.
- **재앙적 회귀:** 운동 보정 계열에는 없다(최악 −0.031). rescue + 메모리 기록 계열에는 있다(−0.116, −0.127). foveation도 있다(−0.330).

### VK 진단 (독립 검토 지적 1 대응, Ubuntu 2080, 생성기 없이 frame 정보만 기록)

원자료: `outputs/milestone8/diag_vk/`, 요약 `outputs/milestone8/vk_coabs_diag.json`

| 항목 (SGGL 궤적, GT 있는 동시 부재 frame) | DEV+REP | VK |
|---|---|---|
| frame 수 / 전체 | 131 / 5,082 | 89 / 1,736 |
| 해당 단위 / 영상 | 14 / 11 | **4 / 1** (Scene01 obj59 58, obj91 29) |
| GT가 crop 밖에 완전히 있음 | 47% | 85% |
| GT 면적 / 초기 면적 (중앙값) | **0.68** | **6.4** |
| GT 면적 < 초기의 절반 | 45% | 2% |
| 광류 운반 box가 GT와 겹침 | 16% | 34% |

**해석:**
- VK에서 놓친 frame은 거의 전부 Scene01의 두 차량에서 나온다.
- 이 차량들은 면적이 초기의 중앙값 6.4배로 커진다(카메라 쪽으로 다가옴). 크기가 고정된 crop(초기 box의 4배)에 들어가지 않는다.
- crop을 옮기는 운동 운반으로는 해결되지 않는 **크기 변화 실패**다.
- DEV/REP의 놓친 frame은 대상이 작아지거나 위치가 바뀐 경우였다. 그래서 VK의 무효과는 FLOWVEL의 메커니즘과 모순되지 않는다. 다만 효과가 '대상이 crop 크기 안에 있으면서 위치만 바뀌는 사건'에만 나타난다는 한계를 보여 준다.
- 크기 변화에 맞춰 crop 크기를 바꾸는 방법(AdaSAM2 계열)은 M8에서 시험하지 않았다. 다음 후보다(`NEXT_RESEARCH_DECISION_TREE.md` Q0).

## 7. 새로움

부재 중 crop 운반은 고전적 **카메라 운동 보정**이다. 방법 자체는 새롭지 않다.

M8의 기여는 측정이다.
- object-relative crop을 쓰는 frozen SAM 2 dual-view 추적에서, 동시 부재 손실의 절반이 이 결함에서 온다.
- ablation으로 그 원인을 분리했다.
- 생성·순위·인과를 나눈 진단을 했다.

자세한 비교는 `PRIOR_ART_AND_NOVELTY_AUDIT.md` §3에 있다.

## 8. 재개 명령

```bash
# 분석 재생성 (CPU, Ubuntu)
uv run python scripts/m8_analysis.py stage1 && uv run python scripts/m8_analysis.py causal && uv run python scripts/m8_analysis.py pairwise
# 새 worker에서 arm 실행 (예: CAMCARRY, 64단위를 4 worker로)
uv run python scripts/gpu_sync.py push-code
#   worker: source /workspace/m8env.sh; for i in 0 1 2 3; do M7_SHARD=$i/4 OMP_NUM_THREADS=3 $PY scripts/m8.py run CAMCARRY DEV,REP & done
uv run python scripts/gpu_sync.py loop 300 outputs/milestone8/runpod   # Ubuntu tmux
uv run python scripts/gpu_sync.py audit outputs/milestone8/runpod      # worker 정지 전 exit 0 확인
```

## 9. 독립 검토 반영

- 검토 원문: [`REVIEW_raw.md`](REVIEW_raw.md). 대응: [`REVIEW_RESPONSE.md`](REVIEW_RESPONSE.md).
- 추가 통계: `robustness_report.json` (`m8_analysis.py robustness`), `auc_report.json` (`m8_analysis.py auc`).
