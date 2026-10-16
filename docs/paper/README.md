# 논문 집필 자료 (`docs/paper/`)

모든 수치는 스크립트가 원자료(json·csv)에서 읽어 만든다. 손으로 옮긴 수치는 없다. 다시 만들려면 아래 명령을 Ubuntu에서 실행한다(GPU 불필요, 그림 6 마스크 재실행만 2080 사용).

| 파일 | 내용 | 생성 스크립트 | 원자료 |
|---|---|---|---|
| `table5_main.md` | 표 5 주 결과: B0 / P5 / SGGL (MOSEv2·VOST·LVOS v2, DAVIS 계열, VKITTI2) | `scripts/paper_tables.py` | `outputs/milestone10/reports/baselines.json`, `outputs/milestone61/comparison_report.json`, `outputs/milestone7/replay_report.json`, `outputs/milestone7/vk_report.json` |
| `table6_tiny.md` | 표 6 극소 단위 (area0 < 0.001) | `scripts/paper_tables.py` | `baselines.json` |
| `table8_states.md` | 표 8 상태별 손실 분해 B0 / P5 / SGGL | `scripts/paper_tables.py` | `baselines.json` |
| `table9_oracle.md` | 표 9 oracle 상한 (LVOS 열 포함, 상위 2영상 비중 ≥ 0.5 각주) | `scripts/paper_tables.py` | `outputs/milestone10/reports/oracle_table.json` |
| `table10_selection.md` | 표 10 GT 없는 선택 신호 시험: 수정 5 손규칙, 수정 6·8 VXO·VXM·N1M·N2X, 수정 10 head, 수정 11·12 진단 | `scripts/paper_tables.py` | `selector_gate.json`, `selector_offline.json`, `degraded.json`, `head_offline.json`, `head_offline_a01.json`, `head_upper.json`, `head_within.json` |
| `fig3_states.csv` | 그림 3 데이터: 데이터셋 × arm(B0/SGGL) × 상태 → 손실 | `scripts/paper_tables.py` | `baselines.json` |
| `fig4_oracle.csv` | 그림 4 데이터: arm × 데이터셋 → ΔJ&F, CI, 회복 비율 | `scripts/paper_tables.py` | `oracle_table.json` |
| `fig5_episodes.csv` | 그림 5 데이터: arm × 데이터셋 → 에피소드 수, 평균 길이 | `scripts/paper_tables.py` | `outputs/milestone10/reports/episodes.json` |
| `fig6_examples.png` | 그림 6 예시 frame: B0 / SGGL / GT, crop 사각형 | `scripts/paper_fig6.py select` → `scripts/m10.py rerun` (2080) → `scripts/paper_fig6.py plot` | `runs/{B0,OR0}` (선택), `outputs/milestone10/fig6/` (2080 마스크) |
| `fig6_selection.md` | 그림 6 단위·frame 선택 규칙과 근거, 2080/4090 J&F 대조 | `scripts/paper_fig6.py` | 위와 같음 |
| `fig3.png/.svg`, `fig4.png/.svg`, `fig5.png/.svg` | 그림 3 상태별 손실(B0 대 SGGL), 그림 4 oracle 상한(점 + CI, 회복 비율), 그림 5 이탈 에피소드 수 대 길이. 300 dpi, NanumGothic(OFL, `data/fonts/`, sha256은 스크립트에), Okabe-Ito 팔레트 + 빗금·마커 모양 | `scripts/paper_figs.py` | `fig3_states.csv`, `fig4_oracle.csv`, `fig5_episodes.csv` |
| `captions.md` | 그림 3·4·5 캡션 초안 (수치는 CSV에서 읽음) | `scripts/paper_figs.py` | 위와 같음 |
| `appendixB_amendments.md` | 부록 B 사전 등록·수정 이력 M6 – M10 수정 12 | `scripts/paper_appendixB.py` (+ 출처를 밝힌 수동 행 11개) | `outputs/milestone*/preregistration*.json`, `outputs/milestone{8,9}/arms.json`, `docs/m{9,10}/EXECUTION_LOG.md`, git 커밋 시각 |

```
uv run python scripts/paper_tables.py
uv run python scripts/paper_appendixB.py
uv run python scripts/paper_figs.py
uv run python scripts/paper_check.py      # 문서 수치 대 reports/*.json, 불일치 0이면 exit 0
uv run python scripts/paper_fig6.py select
M9_DATA=data/m9 M10_SAVE_MASKS=outputs/milestone10/fig6 uv run python scripts/m10.py rerun outputs/milestone10/fig6/rerun.txt
M9_DATA=data/m9 uv run python scripts/paper_fig6.py plot
```

**주의**
- report 스냅샷은 커밋 `b212cc9`가 최종본이다. bootstrap 난수는 비교 쌍마다 고정돼(`m10_analysis.pair_rng`) 같은 입력이면 바이트 단위로 같고, 같은 비교는 어느 report에서나 같은 CI다. 자동 갱신 루프는 껐다.
- 그림 6의 MOSEv2 frame은 공개 HF `FudanCVL/MOSEv2` train 분할을 Ubuntu에서 한 번 스트리밍해 해당 영상 하나만 추출했다(분할 sha256은 M9 기록과 대조, `outputs/milestone10/fig6/dl/`).
- 같은 데이터셋 진단(수정 11·12)은 dev로 학습했으므로 채택 근거가 아니다.
