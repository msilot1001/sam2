# M5.5 — SAM 2.1 공간 디테일 보존: 실제 영상 타당성 연구 (탐색적)

- 브랜치: `exp/m55-spatial-detail-feasibility` (`exp/m5-pre-memory-intervention` `0ea50c3`에서 분기, 병합하지 않음)
- 모델: 공식 SAM 2.1 Hiera-Tiny, CUDA **FP32**(TF32 끔), 가중치 고정. upstream 파일은 수정하지 않았다(인스턴스 단위 hook과 공개 메서드 호출만 사용).
- 데이터:
  - DAVIS 2017 val 30개 영상, 61개 객체. 주 평가이며 SAM 2 학습 데이터와 겹칠 수 있다.
  - SegTrack v2 14개 영상, 24개 객체. SAM 2 학습에 쓰이지 않았다(zero-shot).
  - VKITTI 시험 장면은 열지 않았다.
- 사전 고정 순서:
  1. `manifest.json` (`aee4348`): 205개 DAVIS 객체, strata, 실패 분류 규칙, 단일 요인 ablation 격자
  2. `manifest_segtrack.json` (`338d0e4`)
  3. `manifest_proto.json` (`10fe67c`): 프로토타입 하이퍼파라미터, 비교 조건
  - 각 manifest는 해당 추론 전에 커밋했다. 하이퍼파라미터는 **DAVIS train**에서만 골랐다.
- 실행 시간: native 약 0.20 s/frame. 조건 하나당 최대 할당 3.3 GB(DAVIS), 긴 SegTrack 영상(최대 279 frame)은 7.7 GB.

## 1. Prompt frame 감사 (paragliding-launch obj3, kite-surf obj1)

M5-Pre에서 prompt frame J가 0.28과 0.66으로 낮게 나온 원인을 단계별로 추적했다(`outputs/milestone55/audit/prompt_audit.json`).

| 단계 | paragliding-launch obj3 | kite-surf obj1 |
|---|---|---|
| GT PNG (palette P 모드, id {0,1,2,3}, void 255 없음, 480×854 = JPEG 크기) | 3,799 px, 98%가 얇은 구조(줄), 120개 조각 | 3,350 px, 59%가 얇은 구조(줄 + harness), 5개 조각 |
| prompt frame 이름 / video_idx | `00000` / 0, JPEG 이름 0..n-1 연속 | 같음 |
| ① 1024×1024 입력 변환 (bilinear + antialias, ≥ 0.5) 후 복원 J | **0.997** | **1.000** |
| ② `_use_mask_as_output`의 256×256 출력 → 영상 해상도 J | **0.283** | **0.658** |
| 공식 prompt 출력 = ② 재구성 (비트 단위) | 예 | 예 |
| ③ prompt **메모리 인코더 입력** (256 → 1024 bilinear, 이진화) J | **0.276** | **0.648** |
| 공식 메모리 인코더 입력과 ③ 재구성 차이 | 최대 9.5e-7 | 최대 6.0e-8 |
| 정렬 확인: ±3 px 이동 중 최고 J를 주는 이동 | (0, 0) | (0, 0) |
| 첫 전파 frame J / frame 1–20 J | 0.229 / 0.230 | 0.533 / 0.404 |
| frame 1–20의 256 상한 J (GT → 256 logit → 영상 해상도) | 0.229 | 0.585 |
| decoder 경로 (`use_mask_input_as_output_without_sam=False`) | 출력이 **비어 있음** (J 0) | 비어 있음 (J 0) |

**결론:**
- 평가 오류가 아니다. 객체 ID, frame 인덱스, 해상도, 좌표 정렬 모두 맞다.
- 낮은 IoU의 원인은 **SAM 2 추론 경로가 마스크를 256×256으로 재표본화하는 손실**이다. GT를 1024로 바꾸는 단계에서는 손실이 없다(≥ 0.997).
- 손실은 두 군데에서 생긴다.
  - (a) **출력 해상도:** decoder와 prompt 출력이 stride 4(256×256)라서 1–2 px 줄을 표현하지 못한다. paragliding obj3는 모델이 이 상한(0.229)에 **정확히** 붙어 있다.
  - (b) **prompt 메모리:** 공식 video predictor는 `add_new_mask`에서 메모리 인코더를 건너뛴다. 그 뒤 `propagate_in_video_preflight`에서 **256으로 줄였다가 다시 1024로 키운 마스크**로 prompt 메모리를 만든다. 그래서 얇은 구조와 작은 객체가 메모리에 들어가기 전에 사라진다.
- **학습 코드와의 불일치 (설치된 `training/model/sam2.py`에서 확인):** 학습 때 prompt frame 메모리는 `_use_mask_as_output`의 **1024 해상도 마스크**(±10 logit)로 만든다. 추론에서만 256 왕복이 들어간다. 즉 (b)는 **학습과 추론 사이의 불일치**다.
- decoder 경로(P6)는 61개 객체 모두에서 출력이 비었다. 공식 설정에서 쓰지 않는 경로이고 교정 수단으로도 쓸 수 없다.

**이전 결론 수정 (M5-Pre §5):**
- 원래 결론: "paragliding, kite-surf 실패는 메모리나 후보 문제가 아니라 분할 단위·해상도 문제다."
- 수정:
  - prompt frame J가 낮은 것은 모델의 분할 품질 문제가 아니라 **256 출력 경로에서 생기는 결정론적 손실**이다.
  - kite-surf obj1(그리고 lab-coat obj2)의 이후 실패는 일부가 **해상도가 떨어진 prompt 메모리** 때문이다(§5에서 P5로 교정됨).
  - 따라서 이 사례들에서는 "메모리 문제가 아니다"가 맞지 않다. 정확히는 '메모리 선택'이 아니라 '**prompt 메모리의 공간 충실도**' 문제다.
- M5-Pre의 나머지 수치(PART A/B/C)는 영향을 받지 않는다.

## 2. 실제 영상 데이터 평가 (조사 에이전트 결과, 1차 출처 확인)

| 데이터 | 오프라인 GT | 라이선스 | 크기 / 선택 다운로드 | SAM 2 학습과 겹침 | 판정 |
|---|---|---|---|---|---|
| **DAVIS 2017 val** (보유) | 모든 frame | 연구용 | 보유 | SAM 2 논문: 학습 혼합에 DAVIS 약 1.3% 포함. 배포 체크포인트는 "SA-V manual + Internal + SA-1B"로 학습했다고 하지만, 어느 split인지 밝히지 않아 **겹침 가능으로 취급** | 주 평가 |
| **SegTrack v2** (이번에 받음, 209 MB, sha256 `947bea61…`) | 모든 frame (저해상도 ≤ 640×360) | **명시 없음** (연구용으로만 사용, git 제외) | 단일 zip | 학습에 없음 | zero-shot 대조. 디테일 객체가 1개(birdfall)뿐 |
| MOSE v1 | train만 (valid는 서버 채점) | CC BY-NC-SA 4.0 | 26 GB, 단일 zip | train이 학습에 포함 | 부적합 |
| MOSEv2 | train만 | 표기가 서로 다름 | train 약 58 GB, valid 26.7 GB | 없음 | 사용자 승인 필요 |
| YouTube-VOS | train 6 fps (valid는 첫 frame만) | 비상업 | 수 GB | 학습에 포함 | 부적합 |
| LVOS v2 val | 6 fps 공개 | CC BY 4.0 비상업 | 크기 미확인, split 단위 | zero-shot | 후보 (승인 필요) |
| VOST | 5 fps | CC BY-NC-SA 4.0 | 54 GB 단일 | zero-shot | 너무 큼 |
| **PUMaVOS** | 24개 영상, 30 fps 전 frame | CC BY 4.0 | 2.8 GB, Drive 폴더에서 영상별 다운로드 가능 | zero-shot | **다음 단계 최우선 후보** (부분·얇은 마스크, 승인 필요) |
| BURST, VISOR, LV-VIS, SA-V val/test | 희소하거나 형식 변환이 필요 | 다양 | 크다 | 대부분 zero-shot | 보류 |
| DAVIS test-dev | 첫 frame만 | — | 316 MB | — | 오프라인 평가 불가 |

- 수 GB 데이터는 받지 않았다.
- **한계:**
  - 주 결과(DAVIS)는 학습 데이터와 겹칠 수 있다.
  - zero-shot 데이터(SegTrack)는 해상도가 낮고 얇은·작은 객체가 거의 없다. 그래서 공간 디테일 가설을 검정하는 힘이 약하다.

## 3. 자연 실패 분포 (DAVIS val 61개 객체 전수, 3,868 visible frame)

**실패 기준과 분류:**
- 실패: J < 0.5. 분류 규칙은 실행 전에 고정했고, 앞에서부터 처음 맞는 유형 하나로 정한다.
- **F (가림·정체성):** GT 면적이 객체 중앙값의 25% 미만이거나, 예측 픽셀의 30% 이상이 다른 객체 위에 있음.
- **D (출력 해상도):** 256 상한 J < 0.5.
- **BC (인코더·decoder):** 고정된 SAM 2 이미지 모드에 GT box(또는 box + 내부 점)를 주고 3개 후보 중 최고 J를 본 값(사후 oracle, 실제로는 쓸 수 없음)이 < 0.5.
- **E (메모리 전파):** oracle ≥ 0.5이고 oracle ≥ 영상 J + 0.2.
- **A (prompt 전처리):** 해당 없음. 1024 표현 J가 61개 모두 ≥ 0.989다(최소: shooting obj3).
- **G (주석 모호성):** 자동 판별 불가. 아래에 수동 검토 결과를 적었다.

| 유형 | 실패 frame | 객체 수 | 주요 객체 |
|---|---|---|---|
| D 출력 해상도 | 81 | 2 | paragliding-launch obj3 (79) |
| BC 인코더·decoder | 72 | 7 | kite-surf obj1 (35), lab-coat obj2, kite-surf obj2, bmx-trees |
| F 가림·정체성 | 63 | 9 | india obj2 (15), lab-coat obj1 (13), kite-surf obj1 (11) |
| E 메모리 전파 | 59 | 5 | lab-coat obj2 (29), paragliding obj2 (16), kite-surf obj2 (10) |
| 미해결 | 12 | 4 | |
| **합계** | **287** | **13** | |

- 실패가 이어질 확률: P(다음 frame 실패 | 실패) = 0.70.
- 수동 검토(`figs/failure_examples.png`):
  - 실패는 대부분 **작은 객체**(lab-coat obj2의 휴대폰, kite-surf obj2의 보드)이거나 **얇은 줄**(kite-surf, paragliding)이다.
  - kite-surf obj1은 harness와 연 줄을 한 객체로 묶은 주석이다. 분할 단위가 모호하다는 점에서 G 성격이 있다.
- **한계:**
  - BC 판정에 쓴 이미지 oracle은 box 프롬프트라서 줄처럼 가는 대상에서는 약하다(전체 평균 0.846으로 영상 J 0.844와 비슷).
  - 그래서 BC와 E의 경계는 거칠다. B(인코더)와 C(decoder)는 학습 없이 분리할 수 없어 합쳐서 보고했다.
- **SegTrack v2:** 실패 frame 11개, 객체 4개(E 6, BC 2, F 1, 미해결 2). 256 상한 J가 0.999라서 해상도 실패가 없다(영상이 작기 때문).

## 4. 공간 디테일 측정 (DAVIS val, 객체 단위 평균)

strata는 prompt frame GT 메타데이터로만 정했다:
- small: 면적이 frame의 1% 미만
- thin: opening r = 2로 사라지는 비율 ≥ 0.30
- multi: 20 px 이상 조각이 3개 이상
- detail = small, thin, multi 중 하나 이상. 61개 중 19개.

| 집단 | n | J | F | F_strict (2 px) | **256 상한 J** | 상한 − J | 얇은 구조 recall (상한) | 조각 유지율 (상한) |
|---|---|---|---|---|---|---|---|---|
| 전체 | 61 | 0.844 | 0.900 | 0.778 | 0.955 | 0.110 | 0.37 (0.62) | 0.82 (0.95) |
| **detail** | 19 | **0.687** | 0.805 | 0.674 | **0.878** | **0.190** | 0.37 (0.58) | 0.62 (0.85) |
| plain | 42 | 0.915 | 0.943 | 0.825 | 0.989 | 0.074 | 0.36 (0.64) | 0.91 (0.99) |
| thin | 5 | 0.270 | 0.449 | 0.350 | 0.631 | 0.361 | 0.22 (0.50) | 0.29 (0.66) |
| small | 13 | 0.653 | 0.756 | 0.640 | 0.845 | 0.192 | 0.37 (0.59) | 0.61 (0.86) |

| GT 면적 (frame별) | frame | J | 256 상한 J | F_strict |
|---|---|---|---|---|
| < 500 px | 103 | **0.183** | 0.739 | 0.280 |
| 500–2k | 131 | 0.506 | 0.900 | 0.579 |
| 2k–8k | 866 | 0.733 | 0.875 | 0.727 |
| 8k–32k | 1,635 | 0.919 | 0.990 | 0.835 |
| > 32k | 1,133 | 0.949 | 0.992 | 0.844 |

- **256 출력 해상도가 설명하는 오차 비율:** 오차 합 기준 (1 − 상한 J) / (1 − J)로 계산했다. 전체 30%, **detail 42%**, plain 13%.
- 나머지 오차는 상한보다 아래에 있다. 같은 256 표현으로도 가능한 정확도에 모델이 못 미친다.
- **시간에 따른 저하:** 상한과 J의 차이가 첫 10 frame 0.076에서 마지막 10 frame 0.131로 커진다. 61개 중 15개 객체에서 0.05 이상 증가했다.
- **경계 일관성:** 연속 frame 사이 F_strict가 예측 0.472, GT 0.484다. 예측이 GT보다 약간 덜 안정적이지만 차이는 작다.

## 5. 전처리 ablation (DAVIS val 61개 객체, prompt + 20 frame, 단일 요인)

| 조건 | 1024 표현 J | prompt 출력 J | **prompt 메모리 입력 J** | 메모리 입력 얇은 구조 recall | 첫 frame ΔJ | **20 frame ΔJ&F** [영상 bootstrap CI] | detail / plain ΔJ&F | ΔF_strict | 개선 / 악화 객체 (\|Δ\| > 0.01) |
|---|---|---|---|---|---|---|---|---|---|
| P0 공식 | 0.9996 | 0.932 | 0.931 | 0.57 | — | — | — | — | — |
| P1 nearest 리사이즈 | 0.9997 | 0.936 | 0.934 | 0.58 | +0.002 | +0.001 [+0.000, +0.002] | +0.003 / 0.000 | +0.001 | 2 / 0 |
| P2 bilinear, antialias 없음 | 0.9996 | 0.932 | 0.931 | 0.57 | 0.000 | 0.000 | 0 / 0 | 0.000 | 0 / 0 |
| P3 임계값 0.25 | 0.979 | 0.944 | 0.943 | 0.79 | +0.005 | **+0.018** [+0.000, +0.042] | +0.059 / −0.000 | +0.022 | 6 / 0 |
| P4 임계값 0.75 | 0.977 | 0.914 | 0.915 | 0.35 | −0.001 | −0.003 | −0.011 / 0.000 | −0.005 | 0 / 2 |
| **P5 full-res prompt 메모리** | 0.9996 | 0.932 | **0.9996** | **0.99** | +0.008 | **+0.021** [+0.001, +0.046] | **+0.065 / +0.000** | **+0.028** | **6 / 0** |
| P6 decoder 경로 (진단) | 0.9996 | 0.000 | 0.000 | 0.00 | −0.877 | −0.893 | — | — | 0 / 60 |

- 리사이즈 방식(P1, P2)은 효과가 거의 없다. 손실은 1024 리사이즈가 아니라 **256 왕복**에서 생기기 때문이다.
- P3는 마스크를 두껍게 만들어 256 왕복을 일부 버티게 한다. P5는 왕복 자체를 없앤다. P5가 더 깔끔하고 효과도 크다.
- P5의 효과는 detail 객체에 집중된다. 20 frame 기준:
  - lab-coat obj2 **+0.733**: 작은 휴대폰이 prompt 메모리에서 사라져 추적이 처음부터 시작되지 못했다.
  - kite-surf obj1 +0.292
  - paragliding obj3 +0.125
  - lab-coat obj1 +0.036
  - mbike-trick obj2 +0.015
- **악화된 객체는 없다.**
- 실행 시간 차이는 없다(0.207 s/frame). P5는 prompt frame에서 메모리 인코더를 한 번 더 돌릴 뿐이다.
- **분류:** P5는 학습과 추론의 불일치를 교정하는 **공학적 교정**이다(약 5줄). 새 연구 기여가 아니다.

## 6. 최소 프로토타입 (사전 고정, 전체 길이)

**진단으로 확인된 문제:**
- (i) prompt 메모리 해상도 손실 → P5로 교정.
- (ii) 256 출력 해상도 상한: detail 오차의 42%.
- (iii) 상한 아래에 남는 격차: detail 0.19이며 시간이 지날수록 커진다.

**학습 없는 프로토타입 1개(두 가지 조건)로 (ii)를 시험했다:**
- **유도 업샘플링 (`src/spatial/proto.py`):** 256 logit을 bilinear로 키운 확률 지도에, 현재 frame 영상(gray)을 guide로 하는 guided filter(He et al.)를 적용한다. 미래 frame은 쓰지 않는다.
  - `guided_out`: 출력에만 적용
  - `guided_mem`: 다듬은 지도로 그 frame의 메모리를 다시 인코딩. 미래 frame에 미치는 영향을 보기 위한 조건이다.
- 하이퍼파라미터는 **DAVIS train**의 hash로 고정한 24개 객체(detail 12 + plain 12) × 20 frame에서만 골랐다.
  - 격자: r ∈ {1,2,4,8}, eps ∈ {1e-4,1e-3,1e-2,1e-1}
  - 선택: r = 1, eps = 1e-4. J&F 0.9253, 공식 0.9247. r ≥ 4는 확실히 나빠졌다.
- 비교 조건:
  - 공식
  - 단순 교정(P5). val 20 frame ablation에서 골랐다. 평가 객체와 같으므로 누출 가능성이 있음을 밝힌다. SegTrack은 이 선택에 쓰이지 않았다.
  - `guided_out`, `guided_mem`
  - P5 + `guided_mem` 조합 (두 요인 동시이므로 따로 표시)

**DAVIS val 61개 객체, 모든 frame (3,923), 공식 대비 paired:**

| 조건 | J&F | ΔJ&F [CI] | detail / plain | ΔF_strict | Δ얇은 구조 recall | Δ조각 유지율 | Δ frame 1–20 / 21–끝 | 개선 / 악화 (> 0.01) | 지연 (s/frame) | 최대 VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| 공식 | 0.8765 | — | — | — | — | — | — | — | 0.209 | 3.2 GB |
| **단순 교정 P5** | **0.8937** | **+0.0172** [+0.0006, +0.0401] | **+0.0550 / +0.0001** | +0.023 | +0.020 | +0.027 | +0.021 / **+0.019** | **5 / 0** | 0.208 | 3.3 GB |
| guided_out | 0.8765 | −0.0000 [−0.0006, +0.0004] | −0.001 / +0.000 | −0.001 | +0.004 | −0.001 | 0.000 / −0.000 | 0 / 1 | 0.209 | 3.2 GB |
| guided_mem | 0.8635 | **−0.0130** [−0.0309, +0.0004] | −0.034 / −0.003 | −0.014 | +0.025 | −0.001 | −0.003 / −0.020 | 4 / **14** | 0.213 | 3.3 GB |
| P5 + guided_mem | 0.8906 | +0.0141 [−0.0073, +0.0396] | +0.052 / −0.003 | +0.018 | +0.049 | +0.040 | +0.021 / +0.020 | 7 / 12 | 0.214 | 3.3 GB |

**SegTrack v2 24개 객체 (zero-shot), 모든 frame:**

| 조건 | J&F | ΔJ&F [CI] | 개선 / 악화 |
|---|---|---|---|
| 공식 | 0.9298 | — | — |
| P5 | 0.9300 | +0.0002 [−0.0000, +0.0006] | 0 / 0 |
| guided_out | 0.9303 | +0.0005 [−0.0002, +0.0010] | 0 / 0 |
| guided_mem | 0.9289 | −0.0009 [−0.0055, +0.0058] | 1 / 2 |
| P5 + guided_mem | 0.9290 | −0.0007 | 1 / 2 |

**해석:**
1. **P5는 실제 미래 frame을 개선한다.**
   - 21번째 frame 이후 +0.019. 효과가 사라지지 않는다.
   - 악화된 객체가 없다(DAVIS 0, SegTrack 0).
   - 하지만 개선은 **5개 객체(4개 영상)에 몰려 있다.** lab-coat obj2 +0.707, kite-surf obj1 +0.214, paragliding obj3 +0.073, mbike-trick obj2 +0.019, lab-coat obj1 +0.017.
   - SegTrack에서는 효과가 0이다. 해상도가 낮고 대상이 덩어리 모양이라 손실이 없기 때문이며, 일반화를 시험하지 못했다.
2. **유도 업샘플링 출력(`guided_out`)은 효과가 없다.**
   - 256 logit에 이미 없는 줄과 작은 조각은 영상 edge만으로 복원할 수 없다.
3. **다듬은 마스크를 메모리에 넣으면(`guided_mem`) 해롭다.**
   - 14개 객체가 악화됐다. lab-coat obj1 −0.54, drift-chicane obj1 −0.19이며, 둘 다 21번째 frame 이후 크게 나빠졌다.
   - 메모리 인코더가 학습 때 보지 못한 형태(filter 처리된 확률 지도)의 입력이 누적 오차를 만든 것으로 보인다(가설).
   - 미래 frame 영향은 **음수**다.
4. 조합(P5 + guided_mem)은 P5 단독보다 나쁘다.

## 7. 선행연구와 겹치는 부분 (조사 에이전트 결과, 웹 출처)

| 방법 | 바꾸는 모듈 | 학습 | 영상·메모리 반영 | 디테일 대상 |
|---|---|---|---|---|
| HQ-SAM (NeurIPS'23) | mask decoder (HQ token + 전역·국소 특징 융합) | 0.5% 미만 추가 파라미터, HQSeg-44K | frame 단위 | 얇은 구조 |
| HQ-SAM 2 (sam-hq2 저장소) | decoder (MaskDecoderHQ). decoder 출력이 메모리 인코더로 들어감. **prompt frame은 여전히 `use_mask_input_as_output_without_sam: true`** | 학습 | 일부 | 암묵적 |
| SAM2Refiner (ICCV'25) | 인코더 국소 강화 + prompt retargeting + 다중 스케일 refinement head | SAM 2 고정, HQSeg-44K, V100 8대 | 메모리 반영 불명확 | 예 (ThinObject) |
| MGD-SAM2, SAM2-UNeXT, SAM2-Adapter, HRSAM, Pi-SAM | 인코더 adapter / 고해상도 decoder | 학습 | 이미지 | 고해상도 디테일 |
| SAMRefiner, SegRefiner, CascadePSP, CRM | 사후 refiner | 대부분 학습 | frame 단위, 메모리 미반영 | 경계 |
| OASIS (ICCV'25) | Cutie 계열 메모리 + 구조 refinement (Canny) | 학습 | 메모리 반영 | 경계 (SAM 2 아님) |
| SAM2Long, SAMURAI, DAM4SAM, MoSAM | 어떤 frame을 메모리에 넣을지 선택 | 대부분 학습 없음 | 예 | 아니오 |

- **출력 해상도 문제(D)는 이미 많이 다뤄졌다.** HQ-SAM 2, SAM2Refiner, MGD-SAM2가 256 출력 손실을 언급하고, 모두 *학습된* 출력 쪽 head로 해결한다.
- 학습 없는 crop-zoom이나 guided 업샘플링을 SAM 2 영상에 적용한 사례는 찾지 못했다. 다만 이번 실험에서 guided 업샘플링은 효과가 없었다.
- **prompt 마스크가 256 왕복을 거쳐 메모리에 들어간다는 지적과 full-res prompt 메모리는 문헌에서 찾지 못했다**(부재를 증명하지는 못함). 그러나 학습 코드와 비교하면 이것은 **추론 구현의 불일치**이고, 교정은 5줄 정도다. 연구 기여가 아니라 공학적 교정으로 분류한다.
- 남은 열린 질문: 학습된 고해상도 출력(HQ-SAM 2류)이 만든 고충실 마스크를 메모리에 넣으면 미래 frame이 좋아지는가? 이번 학습 없는 시도(guided_mem)는 오히려 나빠졌다.

## 8. 가장 유망한 SAM 2 개선 대상

| 방향 | 메커니즘 | 근거 | 최소 변경 모듈 | 비용 (RTX 2080) | 선행연구 | 부작용 | 작은 인과 검증 |
|---|---|---|---|---|---|---|---|
| **A. 고충실 prompt·메모리 표현** | prompt와 작은 객체의 마스크를 메모리에 손실 없이 넣는다 | **P5 +0.055 (detail), 미래 frame까지 유지, 악화 없음** | `propagate_in_video_preflight`의 prompt 메모리 입력 (공학적 교정) | 없음 | 문헌에는 없으나 학습 경로와 같음 | 없음 (확인 범위 내) | 이미 수행. zero-shot 고해상도 데이터에서 재현 필요 |
| B. 다중 스케일 특징 융합 | stride 4/8 특징으로 256보다 고운 출력 | 상한 아래 격차 0.19, 작은 객체 J 0.18 vs 상한 0.74 | mask decoder upscaling | head 학습. 1024 입력, batch 1–2로 8 GB 안에서 가능할 것으로 봄 (미측정) | **HQ-SAM 2, SAM2Refiner, MGD-SAM2와 직접 겹침** | 메모리 분포 이동 (guided_mem 사례) | frozen 백본 + HQ token head를 DAVIS train에서 학습하고 detail 객체 J를 확인 |
| C. 경계 인식 refinement | 영상 edge로 경계 보정 | **guided_out 효과 0** | 사후 처리 | 낮음 | SAMRefiner, CascadePSP | — | 이미 음성 결과 |
| D. 시간적 디테일 일관성 | 메모리를 통해 디테일 누적 손실을 막는다 | 상한 격차가 0.076 → 0.131로 증가 | 메모리 인코더 입력 | 학습 필요 | 직접 사례 없음 | **guided_mem이 해로움** | 학습 없는 버전은 음성 결과 |

**판정:**
- 증거가 뒷받침하는 것은 A뿐이고, A는 이미 단순 교정으로 해결된다.
- 남은 큰 격차(B, D)는 학습된 고해상도 head가 필요하다. 이 영역은 HQ-SAM 2와 SAM2Refiner가 이미 다룬다.
- 우리만의 차별점이 될 수 있는 '고충실 마스크를 메모리에 넣었을 때 미래 frame에 미치는 영향'은 학습 없는 시도에서 **음수**였다.

## 9. 계산 가능성

- native 추론: 0.20–0.21 s/frame. P5는 지연 증가가 없고, guided filter는 +0.004 s/frame.
- 메모리: DAVIS 최대 3.3 GB. 279 frame SegTrack에서는 `init_state`가 모든 frame을 GPU에 올려 7.7 GB까지 쓴다(긴 영상은 `offload_video_to_cpu`가 필요).
- 전체 계산량: 이번 milestone 전체 약 4.5 GPU 시간. 전수조사 + oracle 약 50분, ablation 약 60분, 프로토타입 약 2시간, 튜닝 약 10분.
- 방향 B(HQ head 학습)는 frozen 백본과 batch 1–2로 RTX 2080에서 가능할 것으로 추정한다. 측정하지 않았고, HQSeg-44K(이미지)가 필요하다.

## 10. 다음 milestone 권고

**질문별 답:**
- **공간 디테일 손실이 SAM 2의 실제 한계인가?** 그렇다.
  - detail 객체의 J는 0.687, plain은 0.915다.
  - 256 출력 해상도가 detail 오차의 42%를 차지한다.
  - 500 px 미만 객체의 J는 0.18이다.
  - 다만 주된 근거는 DAVIS val(학습과 겹칠 수 있음)의 19개 detail 객체이고, 실패는 13개 객체에 몰려 있다.
- **알고리즘 개발을 정당화할 만큼 큰가?**
  - 평균으로 보면 작다. 전체 J&F 기준으로 P5가 회복한 것은 +0.017이다.
  - detail 객체에서는 크다(+0.055, 상한까지 남은 격차 0.19).
  - 하지만 표본이 작고 소수 객체에 집중되어 있다.
- **단순 교정으로 이미 해결되는가?** 일부는 그렇다.
  - prompt 메모리 손실은 P5로 해결된다. P5는 공학적 교정이고, 회귀가 없으며, 미래 frame에도 효과가 유지된다.
  - 출력 해상도 상한과 상한 아래 격차는 해결되지 않는다.
- **선행연구를 넘는 개선이 그럴듯한가?**
  - 출력 쪽 개선은 HQ-SAM 2와 SAM2Refiner가 이미 다룬다.
  - 메모리 쪽 고충실화는 문헌에 없지만, 학습 없는 시도는 해로웠다.
  - 현재 증거로는 그럴듯하다고 말할 수 없다.
- **제안한 변경이 실제 미래 frame을 개선하는가?**
  - P5: 그렇다(frame 21 이후 +0.019, CI 하한 > 0, 악화 0).
  - guided 업샘플링: 출력에만 적용하면 효과 0, 메모리에 넣으면 음수.
- **최종 연구에서 바꿀 정확한 구성요소는?**
  - 현재 증거가 지목하는 곳은 **mask → memory encoder 입력 경로**(prompt frame 및 작은·얇은 객체)다.
  - 하지만 그 교정은 이미 끝났다(P5). 그 다음 단계(학습된 고해상도 출력과 그 결과의 메모리 반영)는 아직 정당화되지 않았다.

**결정: NEED_MORE_REAL_VIDEO_DATA**

이유:
- 주 근거가 DAVIS(학습과 겹칠 수 있음)이고, 효과가 3–5개 객체에 몰려 있다.
- zero-shot 데이터(SegTrack)는 해상도가 낮고 detail 객체가 1개라서, 공간 디테일 가설과 P5 일반화를 검정하지 못했다.

**필요한 최소 추가 진단:**
1. 사용자 승인 후, 해상도가 높고 디테일 객체가 많은 zero-shot 데이터를 받는다. 우선순위는 **PUMaVOS**(CC BY 4.0, 영상별 다운로드, 전체 2.8 GB) 또는 **LVOS v2 val**이다.
2. 받은 데이터로 manifest를 새로 고정하고, 이번과 같은 전수조사(256 상한, 실패 분류)와 **P5 대 공식** 비교를 다시 한다.
3. 판단 기준 (사전 고정):
   - P5가 detail 객체 ≥ 10개에서 개선되고 악화가 없으면, P5를 기본값으로 채택한다.
   - 상한 아래 격차가 detail 객체에서 계속 ≥ 0.1이면, B 방향(frozen 백본 + 고해상도 head)과 '그 출력을 메모리에 넣었을 때의 미래 frame 효과'를 다음 연구 주제로 검토한다.
4. 최종 알고리즘 구현은 시작하지 않았다.

## 재현

```bash
uv run python scripts/segtrack_prepare.py            # data/segtrackv2/raw -> clips (zip은 수동 다운로드, sha256 위)
uv run python scripts/m55.py audit                   # audit/prompt_audit.{json,_frames.csv}
uv run python scripts/m55.py manifest                # 고정됨: 바뀌면 거부
uv run python scripts/m55.py manifest --split segtrack
uv run python scripts/m55.py survey --split val      # 재개 가능 (객체별 csv + npz)
uv run python scripts/m55.py survey --split segtrack
uv run python scripts/m55.py ablate                  # P0-P6, prompt + 20 frame
uv run python scripts/m55.py tune                    # DAVIS train만 사용
uv run python scripts/m55.py proto_manifest --simple P5_fullres_prompt_memory
uv run python scripts/m55.py proto --split val
uv run python scripts/m55.py proto --split segtrack
uv run python scripts/m55.py analyze                 # reports/*
uv run pytest
```

**산출물** (`outputs/milestone55/`; manifest와 `reports/proto_tune.json`만 git에 포함):
- `audit/`
- `survey/{val,segtrack}/` (객체별 frame 지표 csv, 예측 mask packbits npz)
- `ablate/`
- `proto/{val,segtrack}/`
- `reports/`:
  - `survey_*_{report.json,units.csv,failures.csv}`
  - `ablation_{summary,per_unit}.csv`
  - `proto_*_{summary,per_unit}.csv`
  - `proto_tune*.{json,csv}`
- `figs/failure_examples.png`
- `*.log`
