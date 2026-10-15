# 연구 데이터 색인 (Claude Research / 외부 검토용)

인증 없이 열리는 링크다(공개 GitHub 저장소, raw 파일).

- **브랜치:** `exp/m10-oracle-bottleneck-decomposition`
- **기반 이력:** M7·M8·M9 브랜치를 모두 조상으로 포함한다.
- **원본 저장소:** Ubuntu 서버. 결과는 sha256으로 검증한 뒤 커밋한다.
- **대시보드:** https://research.sojak.io/#m10 (로그인 필요)

## 1. 먼저 읽을 것

- [docs/m10/DEEP_RESEARCH_BRIEF.txt](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/DEEP_RESEARCH_BRIEF.txt): M10 중간 브리핑 (모든 수치, 잠정 결론, 질문)
- [docs/m10/ORACLE_DEFINITIONS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/ORACLE_DEFINITIONS.md): oracle 정의, 추적 상태, 통계
- [docs/m10/EXECUTION_LOG.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/EXECUTION_LOG.md): 진행 기록, 사고, 우선순위 변경
- [docs/m9/FINAL_ASSESSMENT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/FINAL_ASSESSMENT.md), [docs/m9/NEXT_DECISION_TREE.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/NEXT_DECISION_TREE.md): 직전 마일스톤 결론

## 2. M10 사전 등록

- [outputs/milestone10/preregistration.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/preregistration.json)
- [outputs/milestone10/preregistration_amend1.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/preregistration_amend1.json)
- [outputs/milestone10/preregistration_amend2.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/preregistration_amend2.json)
- [outputs/milestone10/preregistration_amend3.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/preregistration_amend3.json)
- [outputs/milestone10/preregistration_amend4.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/preregistration_amend4.json)

## 3. M10 분석 리포트 (JSON, `scripts/m10_analysis.py`)

- [outputs/milestone10/reports/episodes.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/reports/episodes.json)
- [outputs/milestone10/reports/interactions.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/reports/interactions.json)
- [outputs/milestone10/reports/or9.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/reports/or9.json)
- [outputs/milestone10/reports/oracle_table.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/reports/oracle_table.json)
- [outputs/milestone10/reports/parity.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/reports/parity.json)
- [outputs/milestone10/reports/pool.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/reports/pool.json)
- [outputs/milestone10/reports/states.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/reports/states.json)

## 4. M10 frame별 원자료 (단위 1개 = CSV 1개)

| arm | 단위 수 (MOSEv2 / VOST / LVOS v2) | 폴더 | 예시 CSV |
|---|---|---|---|
| C48 | 0 / 0 / 10 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/C48) | [lvosv2__lvosv2_7K7WVzGG__obj3.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/C48/lvosv2__lvosv2_7K7WVzGG__obj3.csv) |
| C58 | 62 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/C58) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/C58/mosev2__mosev2_0442a954__obj1.csv) |
| C75 | 62 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/C75) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/C75/mosev2__mosev2_0442a954__obj1.csv) |
| OR0 | 62 / 31 / 16 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR0) | [lvosv2__lvosv2_7K7WVzGG__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR0/lvosv2__lvosv2_7K7WVzGG__obj1.csv) |
| OR4 | 62 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR4) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR4/mosev2__mosev2_0442a954__obj1.csv) |
| OR5M | 62 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR5M) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR5M/mosev2__mosev2_0442a954__obj1.csv) |
| OR5S | 62 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR5S) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR5S/mosev2__mosev2_0442a954__obj1.csv) |
| OR6 | 6 / 12 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR6) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR6/mosev2__mosev2_0442a954__obj1.csv) |
| OR7 | 62 / 31 / 16 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR7) | [lvosv2__lvosv2_7K7WVzGG__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR7/lvosv2__lvosv2_7K7WVzGG__obj1.csv) |
| OR7R | 62 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR7R) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR7R/mosev2__mosev2_0442a954__obj1.csv) |
| OR7W | 62 / 31 / 2 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR7W) | [lvosv2__lvosv2_7K7WVzGG__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR7W/lvosv2__lvosv2_7K7WVzGG__obj1.csv) |
| OR8 | 62 / 31 / 16 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR8) | [lvosv2__lvosv2_7K7WVzGG__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR8/lvosv2__lvosv2_7K7WVzGG__obj1.csv) |
| OR8K10 | 62 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR8K10) | [mosev2__mosev2_0442a954__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR8K10/mosev2__mosev2_0442a954__obj1.csv) |
| OR9 | 62 / 31 / 16 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR9) | [lvosv2__lvosv2_7K7WVzGG__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/runpod/runs/OR9/lvosv2__lvosv2_7K7WVzGG__obj1.csv) |
| OR0 (모델 b+, RTX 2080) | 0 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_b+/OR0) | [vost__vost_1169_peel_onion__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_b+/OR0/vost__vost_1169_peel_onion__obj1.csv) |
| OR0 (모델 l, RTX 2080) | 0 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_l/OR0) | [vost__vost_1169_peel_onion__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_l/OR0/vost__vost_1169_peel_onion__obj1.csv) |
| OR0 (모델 s, RTX 2080) | 0 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_s/OR0) | [vost__vost_1169_peel_onion__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_s/OR0/vost__vost_1169_peel_onion__obj1.csv) |
| OR0 (모델 t, RTX 2080) | 0 / 31 / 0 | [폴더](https://github.com/msilot1001/sam2/tree/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_t/OR0) | [vost__vost_1169_peel_onion__obj1.csv](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone10/ubuntu/runs_t/OR0/vost__vost_1169_peel_onion__obj1.csv) |

※ OR6, C48은 사용자 결정으로 보류(부분 결과). 진행 중인 arm은 단위 수가 늘어난다.

### CSV 열 사전

| 열 | 뜻 |
|---|---|
| set, split, unit, video, arm, video_idx | 데이터셋:split, dev/conf, 단위(영상/객체), 영상, arm, frame 번호 (1부터, 0은 prompt) |
| J, F, JF | 출력 mask의 J (IoU), F (경계), J&F |
| JF_G, JF_L, J_G, J_L | G view(전체 frame)와 L view(crop을 frame에 붙인 것) 각각의 J&F / J |
| Jv_L | L 출력 vs crop 영역 안 GT의 J |
| acc | SGGL 자기 게이트 통과 여부 (True면 출력 = L, 아니면 G) |
| src | 실제 출력 출처: L, G, E(빈 mask) |
| score_G, score_L, iou_G, iou_L | SAM 2 객체 점수 logit, 디코더 예측 IoU (선택된 가설) |
| cJ0–cJ2, cJ3–cJ5, cJ6 | G 원시 가설 3개, L 원시 가설 3개, 빈 mask의 GT J (존재 점수로 지우기 전) |
| best, JF_best | 7개 중 J 최대인 후보 번호와 그 J&F |
| ok_G, ok_L | view 예측이 그 view 영역 안 GT에 대해 맞음 (J ≥ 0.5, 둘 다 비면 맞음) |
| adm_G, adm_L | 그 frame이 view 기억에 기록됐는지 (OR7류에서 False 가능) |
| rew_G, rew_L | 그 frame 기억이 다시 부호화됐는지 (OR8: GT mask, OR5M: 최선 가설) |
| crop_y0, crop_x0, crop_side | L crop 위치·크기 (frame 좌표) |
| gt_empty, gt_area, gt_in_crop, area0 | GT가 비었는지, GT 면적, GT 중 crop 안 비율, prompt 면적 비율 |
| out_area, out_inter_gt, out_inter_other, other_area | 출력 면적, 출력∩GT, 출력∩주석된 다른 객체, 다른 객체 면적 |
| G_area, G_inter_gt, L_area, L_inter_gt | view별 면적과 GT 교집합 |
| latency_s, peak_vram_bytes, wall_s, machine | 지연, 최대 VRAM, 단위 실행 시간, 기계 |
| OR9 행 | J_full/F_full/JF_full (GT box, 전체 frame), *_crop (GT box + 완벽 crop), score_* |


## 5. 데이터셋 manifest (GT만으로 추론 전에 고정, M9)

- [outputs/milestone9/manifests/lvosv2_units.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/manifests/lvosv2_units.json)
- [outputs/milestone9/manifests/mosev2_units.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/manifests/mosev2_units.json)
- [outputs/milestone9/manifests/vost_units.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/manifests/vost_units.json)

## 6. M9 리포트

- [outputs/milestone9/reports/anchor.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/anchor.json)
- [outputs/milestone9/reports/catmiss.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/catmiss.json)
- [outputs/milestone9/reports/causal.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/causal.json)
- [outputs/milestone9/reports/pairs.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/pairs.json)
- [outputs/milestone9/reports/rankers_mosev2.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/rankers_mosev2.json)
- [outputs/milestone9/reports/stage1_mosev2.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/stage1_mosev2.json)
- [outputs/milestone9/reports/taxonomy_lvosv2.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/taxonomy_lvosv2.json)
- [outputs/milestone9/reports/taxonomy_mosev2.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/taxonomy_mosev2.json)
- [outputs/milestone9/reports/taxonomy_vost.json](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/outputs/milestone9/reports/taxonomy_vost.json)

## 7. 문서 (마일스톤별)

- **m10:** [docs/m10/CORE_CONCLUSIONS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/CORE_CONCLUSIONS.md), [docs/m10/EXECUTION_LOG.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/EXECUTION_LOG.md), [docs/m10/ORACLE_DEFINITIONS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/ORACLE_DEFINITIONS.md), [docs/m10/RESEARCH_DATA_INDEX.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/RESEARCH_DATA_INDEX.md), [docs/m10/DEEP_RESEARCH_BRIEF.txt](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/DEEP_RESEARCH_BRIEF.txt), [docs/m10/RESEARCH_BUNDLE.txt](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m10/RESEARCH_BUNDLE.txt)
- **m9:** [docs/m9/COMPUTE_AND_THROUGHPUT_AUDIT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/COMPUTE_AND_THROUGHPUT_AUDIT.md), [docs/m9/CONSENSUS_RESULTS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/CONSENSUS_RESULTS.md), [docs/m9/CROSS_DATASET_FAILURE_TAXONOMY.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/CROSS_DATASET_FAILURE_TAXONOMY.md), [docs/m9/DATASET_AUDIT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/DATASET_AUDIT.md), [docs/m9/DATASET_MANIFEST_AND_SPLITS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/DATASET_MANIFEST_AND_SPLITS.md), [docs/m9/EXECUTION_LOG.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/EXECUTION_LOG.md), [docs/m9/FINAL_ASSESSMENT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/FINAL_ASSESSMENT.md), [docs/m9/FLOWVEL_GENERALIZATION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/FLOWVEL_GENERALIZATION.md), [docs/m9/MOTION_ESTIMATOR_ABLATION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/MOTION_ESTIMATOR_ABLATION.md), [docs/m9/NEXT_DECISION_TREE.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/NEXT_DECISION_TREE.md), [docs/m9/REGRESSION_AND_FAILURE_AUDIT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/REGRESSION_AND_FAILURE_AUDIT.md), [docs/m9/RESEARCH_PLAN_AND_PREREGISTRATION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/RESEARCH_PLAN_AND_PREREGISTRATION.md), [docs/m9/SCALE_CONTROL_RESULTS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m9/SCALE_CONTROL_RESULTS.md)
- **m8:** [docs/m8/ATTENTION_MECHANISM_COMPARISON.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/ATTENTION_MECHANISM_COMPARISON.md), [docs/m8/COMBINATION_RESULTS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/COMBINATION_RESULTS.md), [docs/m8/COMPUTE_AND_PERFORMANCE_AUDIT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/COMPUTE_AND_PERFORMANCE_AUDIT.md), [docs/m8/CROSS_FAMILY_COMBINATIONS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/CROSS_FAMILY_COMBINATIONS.md), [docs/m8/EXPANDED_CANDIDATE_PORTFOLIO.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/EXPANDED_CANDIDATE_PORTFOLIO.md), [docs/m8/EXPERIMENT_TOURNAMENT_PLAN.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/EXPERIMENT_TOURNAMENT_PLAN.md), [docs/m8/FAILURE_EPISODE_ANALYSIS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/FAILURE_EPISODE_ANALYSIS.md), [docs/m8/FINAL_PORTFOLIO_ASSESSMENT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/FINAL_PORTFOLIO_ASSESSMENT.md), [docs/m8/NEXT_RESEARCH_DECISION_TREE.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/NEXT_RESEARCH_DECISION_TREE.md), [docs/m8/NON_ATTENTION_MECHANISM_COMPARISON.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/NON_ATTENTION_MECHANISM_COMPARISON.md), [docs/m8/PARALLEL_EXECUTION_PLAN.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/PARALLEL_EXECUTION_PLAN.md), [docs/m8/PRIOR_ART_AND_NOVELTY_AUDIT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/PRIOR_ART_AND_NOVELTY_AUDIT.md), [docs/m8/PRIOR_ART_raw.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/PRIOR_ART_raw.md), [docs/m8/REPORT_INGESTION_AND_REPO_AUDIT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/REPORT_INGESTION_AND_REPO_AUDIT.md), [docs/m8/REVIEW_RESPONSE.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/REVIEW_RESPONSE.md), [docs/m8/REVIEW_raw.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/REVIEW_raw.md), [docs/m8/STAGE1_DIAGNOSTIC_RESULTS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/STAGE1_DIAGNOSTIC_RESULTS.md), [docs/m8/STAGE2_CAUSAL_RESULTS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m8/STAGE2_CAUSAL_RESULTS.md)
- **m7:** [docs/m7/18_CANDIDATE_MATRIX.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/18_CANDIDATE_MATRIX.md), [docs/m7/COMBINATION_EXPERIMENTS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/COMBINATION_EXPERIMENTS.md), [docs/m7/FINAL_ALGORITHM_DECISION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/FINAL_ALGORITHM_DECISION.md), [docs/m7/MIGRATION_RTX5090.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/MIGRATION_RTX5090.md), [docs/m7/NEXT_MILESTONE_ROADMAP.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/NEXT_MILESTONE_ROADMAP.md), [docs/m7/OPPORTUNITY_AND_FAILURE_AUDIT.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/OPPORTUNITY_AND_FAILURE_AUDIT.md), [docs/m7/PRIOR_ART_AND_NOVELTY.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/PRIOR_ART_AND_NOVELTY.md), [docs/m7/PRIOR_ART_raw.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/PRIOR_ART_raw.md), [docs/m7/PRIOR_ART_raw2.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/PRIOR_ART_raw2.md), [docs/m7/RESEARCH_FOUNDATION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/RESEARCH_FOUNDATION.md), [docs/m7/REVIEW_RESPONSE.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/REVIEW_RESPONSE.md), [docs/m7/REVIEW_raw.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/REVIEW_raw.md), [docs/m7/STAGE1_EXPERIMENTS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/STAGE1_EXPERIMENTS.md), [docs/m7/STAGE2_REVISIONS.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/STAGE2_REVISIONS.md), [docs/m7/TRACK_A_MASK_FUSION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/TRACK_A_MASK_FUSION.md), [docs/m7/TRACK_B_MEMORY_VERIFICATION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/TRACK_B_MEMORY_VERIFICATION.md), [docs/m7/TRACK_C_TARGET_ATTENTION.md](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/docs/m7/TRACK_C_TARGET_ATTENTION.md)

## 8. 코드

- [scripts/m10.py](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/scripts/m10.py): oracle runner (`run_unit`, `rewrite_memory`, `box_unit`)
- [scripts/m10_analysis.py](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/scripts/m10_analysis.py): 통계 (영상 동일 가중, 영상 bootstrap)
- [src/spatial/sggl.py](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/src/spatial/sggl.py): SGGL (G·L 이중 view, 자기 게이트)
- [scripts/m8.py](https://raw.githubusercontent.com/msilot1001/sam2/exp/m10-oracle-bottleneck-decomposition/scripts/m8.py): M8/M9 arm runner

