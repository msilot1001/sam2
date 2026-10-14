# 독립 검토 대응 ([`REVIEW_raw.md`](REVIEW_raw.md))

검토는 모든 headline 수치를 원시 frame CSV에서 재현했다. 누설은 없다고 판정했고, KEEP_SGGL은 고정된 규칙을 그대로 따른 정당한 결과라고 봤다. HIGH 등급 지적은 없다. 수정 요청 10건과 조치는 다음과 같다.

| # | 지적 | 조치 |
|---|---|---|
| 1 | VK를 '새 영상'으로, 결과를 '재현'으로 쓴 것은 과장 | FINAL과 STAGE2를 고쳤다: Scene01/06은 M2–M4 개발 scene이고, 실질 n은 scene 2개, CI는 서술용 |
| 2 | M3-vel / M3-vel-r CI 하한 반올림 (−0.000) | −0.0001로 표기하고, bootstrap 잡음 수준에서 실패한다고 명시했다. FINAL_DECISION 아래에 "M3-vel vs SGGL: INSUFFICIENT_EVIDENCE, 지정 도전자" 줄 추가 |
| 3 | report 재생성 스크립트 없음, 결정에 쓴 출력이 커밋되지 않음 | `scripts/m7_reports.py all` 추가. 이전 report와 수치가 같음을 확인했다. runs/M3-vel-r, SGGL-V, VK, rec_vk CSV, report, combos 코드를 커밋 |
| 4 | S2를 M3-vel로 바꾼 사실을 적지 않음, "모든 객체 동일" 주장 오류 | STAGE1 §3과 FINAL에 일탈로 기록했다(S 계열 인과 확인 없음). '1 frame 차이'로 정정. 사전 등록 파일 자체는 고정 기록이라 수정하지 않았다 |
| 5 | 메커니즘 주장 과장 | PRIOR_ART, 감사 §5, FINAL R1 행, TRACK_B, COMBINATION을 표현을 '가설과 일치', '1 영상 사례'로 낮췄다 |
| 6 | 상태 용어 | M2를 NOT_IMPLEMENTED로, REJECT를 REJECTED로 바꿨다. F1과 M1 행의 INCONCLUSIVE 덮어쓰기를 명시하고 실행 방식(재생/인과/학습) 열 추가 |
| 7 | M3-vel과 VK 시각 증거 없음 | `viz/20_M3vel_timelines.png`, `viz/21_VK_examples.png`를 추가하고 FINAL §4에서 참조. timeline에서 새 관찰: M3-vel 이득은 '재등장 시 빠른 재획득' |
| 8 | M3-vel-r 실행 위치 일탈, ENV의 커밋 기록 부정확 | STAGE2 §3과 FINAL에 일탈로 기록. 실제 커밋을 적은 ENV 파일 4개를 나중에 작성(ENV_vast_SGGL, teval, rec_vk, heads)하고 MIGRATION §9 표 추가 |
| 9 | 작은 불일치 | prototype 수 33으로 정정(정의 명시), VK 거부권 '1객체 5 frame', TRACK_A CI 반올림을 STAGE1과 통일, STAGE2 §1과 §3 모순 해소, `evidence.py` docstring 정정, pre-freeze 재생 11개 공개, MIGRATION과 장부 갱신 |
| 10 | 테스트 부족 | `M3VelocityR.center`(이동 제한, 출처 일관성)와 `evidence.identity` / `Hist` 테스트 추가. 선택 항목이던 L2 격리 테스트는 하지 않았다. 기존 격리 테스트는 이미 scale2 = 2.0인 L2 상태를 포함한다 |
