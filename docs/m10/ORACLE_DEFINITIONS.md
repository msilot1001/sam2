# M10 Oracle 정의

- **사전 등록:** `outputs/milestone10/preregistration.json` (09:02 KST 고정, `378f331`). 추가 항목은 수정 1–2.
- **코드:** `scripts/m10.py` (`run_unit`, `box_unit`)

## 1. 공통 조건

| 항목 | 내용 |
|---|---|
| 기반 시스템 | SGGL. 동결된 SAM 2.1 Hiera-T(FP32, `fill_hole_area=0`), 전체 frame view G와 crop view L, 자기 게이트 |
| 데이터 | M9에서 고정한 manifest의 **dev split만** 사용. MOSEv2 62단위 / 48영상(극소 21), VOST 31 / 19(극소 5), LVOS v2 16 / 12(극소 8, 첫 300 frame). PUMaVOS C 봉인 |
| 기계 | RunPod RTX 4090 Pod 1대에서 모든 단일 oracle을 실행. 비교는 같은 Pod의 OR0와만 한다 |
| 기준선 OR0 | 같은 Pod에서 SGGL을 재실행하며 로그를 확장. M9 행과 frame 단위로 **완전히 일치**(MOSEv2·VOST 8,425 frame 모두 \|Δ\| = 0, `reports/parity.json`) |
| '맞음' (view v, mask m) | v 영역 안 GT가 비고 m도 빔, 또는 J(m, v 영역 안 GT) ≥ 0.5 |
| 출력 | 항상 SAM 2가 예측한 mask이거나 빈 mask다. GT mask를 출력에 복사하지 않는다 |

## 2. 실행 arm (온라인 rollout)

| arm | 질문 | GT가 하는 일 | 바뀌는 것 | 인과성 |
|---|---|---|---|---|
| OR0 | 기준 | 없음 (frame 0 prompt만) | — | — |
| OR1M | 존재 정보가 기억 오염을 줄이는가 | GT가 없는 frame에서 비어 있지 않은 view의 기억 기록을 막음 | 기억만 | 미래 frame |
| OR2 | 위치를 완벽히 알면? | L crop 중심 = 예측할 frame의 GT box 중심 | crop 위치 | 비인과(매 frame 미래 GT) |
| OR3 | 크기를 완벽히 알면? | L crop 변 = 그 frame GT에 M6 crop 크기 규칙을 적용 | crop 크기 | 비인과 |
| OR4 | 기하 전체 | OR2 + OR3 | crop 위치·크기 | 비인과 |
| OR5S | 선택만 완벽하면? | 7개 후보 {G 원시 가설 3, L 원시 가설 3, 빈 mask} 중 J 최대를 출력. 동점이면 SGGL 유지 | 출력(그리고 출력을 따르는 crop) | 현재 frame |
| OR5M | 선택한 후보를 기억에도 쓰면? | OR5S + 각 view의 기억을 그 view에서 J가 최대인 가설로 재부호화 (SAM 2 mask 입력 경로) | 출력 + 기억 | 현재 + 미래 |
| OR6 | 대상 영역 판별이 완벽하면? | 맞으면 유지 → 다른 view가 맞으면 교체 → 출력이 대상 밖(precision < 0.5)이면 기권 → 아니면 유지. 맞지 않은 view의 기억 기록 차단 | 출력 + 기억 기록 | 현재 + 미래 |
| OR7 | 틀린 예측을 기억에 안 쓰면? | view의 예측이 맞을 때만 그 view 기억에 기록(backfill로 이전 frame이 빈자리를 채움) | 기억 기록 | 미래 frame만 |
| OR7W | 대상 밖 예측만 막으면? | view mask가 비어 있지 않고 precision < 0.5면 기록 차단 | 기억 기록 | 미래 frame만 |
| OR7R | 대조: 기록 횟수만 줄이면? | 같은 단위·view에서 OR7과 **같은 개수**의 frame을 seed 고정 무작위로 차단 | 기억 기록 | 미래 frame만 |
| OR8 | 기억이 완벽하면? | frame t 출력 뒤, 각 view의 frame t 기억을 GT mask로 재부호화(add_new_mask 경로) | 기억 내용 | 미래 frame만 |
| OR8C0 | frame 0 기준 기억만 쓰면? | 없음. 처음 prompt 기억만 읽음 | 기억 읽기 | 미래 frame |
| OR9 (수정 2) | 분할 용량 상한 | SAM 2 이미지 모드에 GT box prompt. 전체 frame과, GT 중심의 SGGL 크기 crop | 기억 없음 | 비인과, 추적 아님 |
| 용량 (수정 1) | 모델 크기 | 없음. SGGL을 tiny / small / base+ / large로 실행 (Ubuntu 2080, 같은 기계 비교) | 모델 | — |

## 3. 오프라인 oracle (OR0 행에서 계산, rollout 없음)

| 이름 | 정의 | 주의 |
|---|---|---|
| OR1off | GT가 없는 frame의 출력을 비움 | 출력만 바꾼다 |
| OR5off | frame마다 7개 후보 중 J 최대인 후보의 J&F | 사후(hindsight). 미래 상태를 개선했다는 주장을 할 수 없다 |
| OR6off | OR6의 출력 규칙을 기록된 G·L mask에 적용 | 기억 효과 없음 |

## 4. 추적 상태 (OR0 frame, GT 기반, 사후)

| 상태 | 정의 |
|---|---|
| correct | GT 있음, J ≥ 0.5 |
| partial | GT 있음, J < 0.5, 출력 precision ≥ 0.5 (대상의 일부) |
| empty | GT 있음, 출력 없음 |
| wrong_region | GT 있음, 출력 precision < 0.5 |
| wrong_object | wrong_region 중 출력의 50% 이상이 **주석된 다른 객체** 위에 있음 (주석된 방해물에 한해서만 '다른 객체'라고 말한다) |
| false_present | GT 없음, 출력 있음 |
| absent_ok | GT 없음, 출력 없음 |

## 5. 통계

- **주 지표:** 데이터셋별로 영상 동일 가중 평균 ΔJ&F(arm − OR0). 단위 평균을 먼저 내고 그다음 영상 평균을 낸다.
- **95% CI:** 영상 단위 bootstrap 10,000회
- **함께 보고:**
  - 하나씩 빼기(LOVO) 최솟값·최댓값
  - 상위 2영상 비중
  - ±0.01 영상 수
  - 극소 단위 ΔJ&F
  - 단위 가중 ΔJ&F
- **회복 비율:** ΔJ&F / (1 − OR0 J&F)
- **데이터셋 처리:** 데이터셋별로 먼저 보고한다. 합산 값은 보조로만 쓴다.
- **JSON 출력:** `allow_nan=False`로 쓴다. 비유한 값은 null이다.
