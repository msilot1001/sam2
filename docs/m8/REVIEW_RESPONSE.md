# M8 독립 검토 대응 ([`REVIEW_raw.md`](REVIEW_raw.md))

검토에서는 모든 headline 수치(arm 14개, 짝 비교 11개)를 원시 frame CSV에서 재현했고, 배포 가능 arm에 GT나 미래 frame이 새는 일은 없다고 판정했다. KEEP_SGGL은 그대로다. 지적 14건과 조치는 다음과 같다:

| # | 등급 | 지적 | 조치 |
|---|---|---|---|
| 1 | HIGH | 'VK에는 crop 이탈 사건이 거의 없다'는 설명이 데이터와 맞지 않는다. VK SGGL의 GT 있는 동시 부재는 89/1,736(5.1%)로 DEV/REP의 2배이고, FLOWVEL에서는 97로 늘었다 | FINAL, STAGE2, 결정 트리를 정정했다(원인은 '미상'으로 적음). VK 진단을 2080에서 실행해 결과를 FINAL §6에 추가했다. 원인은 두 차량이 6배 커지는 크기 변화(위치 이탈 아님)였다 |
| 2 | MEDIUM | 'CI 하한 > 0'은 근거로 약하다 | `m8_analysis.py robustness`를 추가했다(`robustness_report.json`). 결과: Bonferroni(배포 가능 arm 11개) 하한 ≈ 0, 영상 부호 검정 p = 0.15–0.39, 두 영상이 이득의 64–69%, 두 영상 제외 +0.004–0.006. STAGE2 §3과 FINAL §1에 적었다 |
| 3 | MEDIUM | FLOWVEL_U는 독립 재현이 아니라 계산 재현 | 모든 문서와 결정 표에서 '계산 재현'으로 바꿨다 |
| 4 | MEDIUM | 사후에 고른 FLOWCARRY를 대표로 썼다 | 대표를 출력 전에 고정한 FLOWVEL로 바꿨다. FLOWCARRY·CAMCARRY는 사후 ablation으로 표기했다. C 후보도 FLOWVEL로 정정했다 |
| 5 | MEDIUM | B_att가 진출 조건을 형식상으로는 통과했는데(형식적인 통과) 공개하지 않았다 | ATTENTION §3과 STAGE1 일탈 목록에 공개했다. 결론은 같다(DEV 회수 0) |
| 6 | MEDIUM | FLOWCARRY ≈ CAMCARRY는 설계상 당연하다. 카메라 운동은 기록하지 않았다. '카메라에 끌려간다'는 표현은 실제와 반대다 | STAGE2 §2.1, NON_ATTENTION, PRIOR_ART, FINAL을 정정했다: 'SGGL은 출력이 비면 crop을 영상 좌표에 고정한다. 부재 중 crop을 움직이는 것이 원인. 카메라 운동 보정은 가설'. FLOWVEL − FV_PRESENT CI가 seed에 따라 달라지는 것도 적었다 |
| 7 | MEDIUM | '메모리 기록 +0.003 여지'의 근거가 약하다 | FINAL 결정 표에 CI [−0.0001, +0.0099], 'SGGL 위에서 잰 값', K_ADMIT_FLOWVEL 반대 근거를 적었다 |
| 8 | LOW–MED | ORACLE은 완벽한 존재 판정도 포함한다. 보고서 판정이 PROMISING으로 나온다 | 설명을 정정했다. `causal_report.json`에서 ORACLE 판정을 `DIAGNOSTIC_ORACLE`로 바꿨다(규칙대로 낸 판정은 `verdict_rule_only`에 남겼다) |
| 9 | LOW–MED | 'FLOWVEL은 출력 전 고정'은 정확하지 않다 (사전 등록 이후, diag 7단위 이후) | STAGE2를 '인과 출력 전 고정'으로 정정했다(16:14, 첫 인과 출력 18:41) |
| 10 | LOW | 운동 반증 시험을 L이 아니라 Lp와 비교했고, 사전 등록한 '동시 부재 subset dJF'를 보고하지 않았다 | STAGE1에 일탈을 공개했다. 동시 부재 subset(SGGL의 동시 부재 486 frame) frame당 ΔJ&F: FLOWVEL +0.117, M3vel +0.032 (STAGE2 §3) |
| 11 | LOW | AUC를 만드는 스크립트가 없다. 0.815가 재현되지 않는다 | `m8_analysis.py auc`를 추가했다. 정의(J > 0.5 대 J < 0.1, 전체 후보, GT 있는 동시 부재 frame)를 명시하면 0.815로 재현된다 |
| 12 | LOW | FLOWCARRY 고정 시각(19:08 vs '~19:25'), arm 수(15 vs 10여 개), 영상 수(64단위 = 39영상), FLOWCARRY·CAMCARRY 지연 +14% 미보고, CAMCARRY도 RAFT 필요 | 고정 시각은 `preregistration_vk_amend1.json`에 정정 필드로 추가했다. arm 수, 지연, RAFT 필요 여부도 정정했다. 39 / 34 구분은 모든 문서에 '작은 객체 57단위 / 34영상'으로 적었다 |
| 13 | LOW | VK 실행이 FLOWCARRY의 DEV/REP ENV 파일을 덮어썼다(`.conflict` 사본 존재). arm마다 코드 버전이 다르다 | 공개만 한다. 검토에서 SGGL과 같은 경로가 정확히 일치해 영향이 없음을 확인했다. `scripts/m8.py run`의 ENV 파일명에 세트 이름을 넣도록 고쳤다 |
| 14 | LOW | '손실의 절반' 주장에는 '개발 세트에서'라는 한정이 필요하다. CMC 선행연구는 확인하지 않았다 | PRIOR_ART §3을 정정했다(개발 세트, 두 영상 집중, VK 미재현). CMC 인용은 '원문 미확인'을 유지했다 |

결정 표 변경 사항(FINAL §4):
- BEST_CAUSAL_CHALLENGER를 FLOWVEL로 바꾸고 단서를 달았다.
- BEST_NON_ATTENTION_DIRECTION을 '부재 중 crop 운반'으로 바꾸고, 카메라 운동 보정은 가설로 표기했다.
- HIGHEST_VALUE_UNRESOLVED_HYPOTHESIS에는 근거가 약하다고 표기했다.
- NEXT_EXPERIMENT_PRIORITY에서는 VK 진단을 맨 앞에 두었다.
