# 회귀와 실패 감사

**분석 명령:** `uv run python scripts/m9_analysis.py causal <arm>`

**원자료:** `outputs/milestone9/reports/causal.json`, `causal_units_<arm>.csv`

## 1. arm별 회귀 요약 (작은 객체, 같은 기계 SGGL 대비)

| arm | 데이터셋 | 단위 | ΔJ&F | 95% CI | > +0.05 단위 | < −0.05 단위 | 최악 단위 | 거짓 재획득 (arm / SGGL) |
|---|---|---|---|---|---|---|---|---|
| FLOWVEL | MOSEv2 | 139 | −0.0142 | [−0.0284, −0.0016] | 8 | 14 | 381hg4vf/obj1 −0.694 | 501 / 495 |
| FLOWVEL | VOST | 74 | +0.0046 | [−0.0181, +0.0272] | — | 8 | 1200_cut_onion/obj1 −0.291 | 269 / 275 |
| SCALE | MOSEv2 | 139 | +0.0116 | [−0.0003, +0.0258] | 10 | 10 | z6kk712z/obj1 −0.130 | 497 / 495 |
| SCALE | VOST | 74 | −0.0073 | [−0.0389, +0.0199] | 9 | 10 | 1179_cut_onion/obj2 −0.695 | 236 / 275 |
| CONSENSUS | MOSEv2 | 139 | −0.0278 | [−0.0585, −0.0010] | 13 | **24** | **8bex1hz5/obj1 −0.741** | 199 / 495 |
| PCCARRY | MOSEv2 | 139 | −0.0031 | [−0.0102, +0.0037] | 3 | 6 | sjl7jcdm/obj1 −0.301 | 486 / 495 |

- **JOINT:** NOT_RUN. 위치 성분(FLOWVEL)이 단독으로 해로웠다.
- **CONSENSUS·PCCARRY의 VOST 실행:** NOT_RUN. MOSEv2에서 반증되었거나 양수가 아니었다.

## 2. 가장 큰 회귀 사례와 메커니즘

| arm | 단위 | SGGL → arm | 메커니즘 |
|---|---|---|---|
| CONSENSUS | MOSEv2 8bex1hz5/obj1 | 0.87 → 0.13 | t=3에서 L이 '없음'이라 기권했다. 그 frame은 GT도 비어 있어 맞았다. 그러나 이후 대상이 다시 나타났을 때 재획득하지 못했고, 나머지 frame의 89%가 J < 0.1이었다. SGGL은 G 출력을 유지해 곧바로 잡았다 |
| SCALE | VOST 1179_cut_onion/obj2 | 0.93 → 0.23 | t=28에서 crop이 SGGL보다 7% 컸다(604 vs 564 px). L이 높은 점수(3.7)로 **다른 양파 조각**으로 바꿔 잡았고 돌아오지 않았다. 이후 frame의 85%가 J < 0.1이었다 |
| FLOWVEL | MOSEv2 381hg4vf/obj1 | 0.80 → 0.11 | t=8부터 광류 속도로 옮긴 crop에서 L이 다른 대상을 높은 점수로 잡았다. 이후 frame의 93%가 J < 0.1이었다 |
| SCALE | MOSEv2 z6kk712z/obj1 | 0.79 → 0.66 | t=44에서 crop이 커진 뒤 L이 오대상으로 넘어갔다 |

- **공통점:** 큰 회귀는 거의 모두 **한 frame에서 생긴 신원 전환이 끝까지 이어진 것**이다.
- **계기:** crop 위치나 크기가 조금 바뀐 것이다. 메커니즘 자체의 논리와는 무관하다.
- **방향:** 같은 크기의 큰 상승도 반대 방향으로 생긴다(1절의 > +0.05 열).

## 3. 단위 수준 변동은 결정론적이다

- **FLOWVEL VOST, Pod vs 2080:** 단위별 Δ의 상관이 0.999다. |Δ| > 0.05인 15단위는 모두 두 기계에서 부호가 같았다.
- **SGGL VOST, Pod vs 2080:** 평균 |ΔJ&F| 0.0006, 최대 0.036이다. ±0.05를 넘은 단위는 0개다.
- **결론:** 단위별 큰 변동은 부동소수점 잡음이 아니다. 궤적이 바뀌어서 생긴 결정론적 결과다.
- **남는 문제:** 그 변동이 메커니즘의 정보 때문인지, 아무 교란이나 같은 정도를 만드는지는 이것만으로 가를 수 없다. 이를 위해 NULLJIT 영 대조를 돌렸다(`preregistration_amend3.json`, 4절).

## 4. NULLJIT 영 대조

(실행 중. 결과가 나오면 채운다.)

## 5. 무결성 감사

| 항목 | 결과 |
|---|---|
| arm 행과 기준 행 길이 | 모든 단위에서 frame 수가 같음 (`_unit_deltas`의 assert 통과) |
| manifest | GT만으로 추론 전에 고정. 덮어쓰기 거부 로직 있음 (`m9_manifest.py`) |
| 같은 기계 비교 | MOSEv2는 Pod diag 궤적, VOST 주 검정은 Pod SGGL. 2080 결과는 재현으로만 보고 |
| 사전 등록 이후 변경 | 수정 1(CONSENSUS), 수정 2(면적 진단), 수정 3(NULLJIT). 모두 출력 전에 고정하고 공개 |
| 실패 분류 범주 추가 | PARTIAL_SEGMENTATION (사후, 공개) |
| PUMaVOS C | Pod의 `data/pumavos/clips`는 15개이고, M8 worker manifest의 A/B 목록 밖 clip은 0개다. 이름은 출력하지 않고 개수만 비교했다. C는 전송·열람·실행하지 않았다 |
| LVOS v2 | Google Drive 할당량 거부. 1차 6회와 2차 재시도 모두 실패 → BLOCKED_USER_ACTION |
