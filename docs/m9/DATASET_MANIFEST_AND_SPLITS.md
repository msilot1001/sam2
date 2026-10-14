# M9 평가 단위 manifest와 분할

모든 manifest는 **그 데이터셋에서 SAM 2를 한 번도 돌리기 전에** GT만으로 고정했다. 파일이 이미 있으면 `scripts/m9_manifest.py`가 덮어쓰기를 거부한다.

## 규칙 (M8 정의 그대로, `scripts/m9_manifest.py`)

1. **prompt frame:** 대상이 처음 보이는 정답 frame. LVOS처럼 늦게 나타나는 객체도 그 frame부터 시작한다.
2. **길이:** prompt 이후 정답 frame이 10개 이상이어야 한다.
3. **crop 단위:** `vcam.crop_side(prompt 마스크)`가 정의된다. box 한 변의 4배가 frame의 짧은 변보다 작아야 한다(M6–M8과 같음).
4. **작은 객체만 포함:** prompt 면적 1% 미만. 극소는 0.1% 미만이다.
5. **영상당 상한:** 영상마다 최대 3개, seed 9로 무작위 선택.
6. **분할:** 영상 id 해시로 dev/conf를 나눈다. 한 영상의 객체는 모두 같은 쪽에 들어간다.
7. **frame:** 정답 파일이 있는 frame만 쓴다. VOST는 5 fps 정답 frame이고, 정답을 보간하지 않는다.

MOSEv2는 영상 수가 많아, 마스크를 보기 전에 meta(frame 수)만으로 표본을 먼저 고정했다.
- 대상: 40–200 frame인 영상 2,531개
- 표본: seed 9로 180개 (`mosev2_extraction_sample.json`, 23:0x KST)

## 고정된 표본

| 데이터셋 | 훑은 영상 | 훑은 객체 | 평가 단위 (작은 객체) | 단위가 있는 영상 | dev / conf 단위 | 극소 | 총 frame | 커지는 객체 (최대 면적 > 2× prompt) | 부재 frame이 있는 단위 |
|---|---|---|---|---|---|---|---|---|---|
| MOSEv2 train | 180 | 436 | **139** | **102** | 62 / 77 | 51 | 10,459 | 85 | 105 |
| VOST val | 70 | 115 | **74** | **44** | 31 / 43 | 11 | 8,396 | 36 | 47 |
| LVOS v2 valid (수정 4, 예산) | 140 | 239 | **32** | **27** | 16 / 16 | 11 | 8,854 | 27 | 25 |
| 합계 | 390 | 790 | **245** | **173** | 109 / 136 | 73 | 27,709 | 148 | 177 |

목표였던 '극소·작은 객체 ≥ 20개, 독립 영상 ≥ 12개'를 크게 넘는다.

**파일:**
- `outputs/milestone9/manifests/{mosev2,vost}_units.json`: 평가 단위
- `*_all_objects.json`: 훑은 모든 객체의 GT 통계
- `mosev2_meta_train.json`

**LVOS v2 예산 규칙 (사전 등록 수정 4):**
- 단위 규칙은 같다. 조건을 만족하는 단위는 97개 / 74영상이었다.
- 각 단위는 첫 300 정답 frame만 쓴다.
- 영상을 seed 9 순서로 넣되, 총 frame이 9,000을 넘지 않을 때까지 영상 단위로 넣었다. 그 결과 27영상이 들어갔다.
- '커지는 객체'와 '부재 frame' 열은 자르기 전 전체 길이 기준이다.
- 명령: `M9_DATA=/root/m9data python scripts/m9_manifest.py lvosv2 "" 300 9000` (Pod)

## 분할 사용 규칙 (사전 등록 `outputs/milestone9/preregistration.json`)

- **M8에서 고정된 arm(SGGL, FLOWVEL, FLOWCARRY):** 바꿀 것이 없으므로 dev와 conf 모두에서 평가하고 따로 보고한다.
- **M9 새 arm(SCALE, JOINT, PCCARRY):** 파라미터는 출력 전에 고정했다. 수정은 메커니즘마다 1번만 허용한다. 수정할 때는 dev 결과만 보고, 판정은 conf로만 한다.
