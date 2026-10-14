# M7 최종 알고리즘 결정

## FINAL_DECISION: KEEP_SGGL

**결정 요지:**
- 18개 후보 계열을 모두 screening했다. 실제로 실행한 prototype은 32종이다(재생 20, 인과 3 + 개정 1, 학습 head 5, 조합 4).
- **SGGL을 사전 등록한 최종 규칙(DEV + REP 영상 CI 하한 > 0)으로 이긴 방법은 없다.**
- 가장 가까운 후보는 M3-vel(crop 중심 속도 예측)이다. +0.0046 [−0.000, +0.012]이고, 영상 하나씩 빼기 최솟값이 +0.0017이다. 그러나 이득이 2개 영상에 몰려 있고, 개정판도 실패했다. VK에서는 +0.0013이다.
- 반대로 **SGGL 자체는 처음 적용한 세트(VKITTI2, 합성, 28객체 / scene 2개)에서 P5 대비 +0.017 [+0.009, +0.025]로 재현됐다.** 22개 객체가 오르고 1개가 내렸다.
- 따라서 더 복잡한 알고리즘으로 교체하지 않는다. SGGL을 주 알고리즘으로 유지한다.
- SGGL-V의 거부권은 개발 데이터에서 miami-surf 하나에만 효과가 있다. VK에서는 발동하지 않았다. 그래서 선택 사항(ablation)으로 내린다.
- PUMaVOS C는 최종 규칙을 통과한 방법이 없어 **봉인을 유지한다.**

**정직한 범위:**
- DEV와 REP는 여러 번 본 개발 데이터다. 모든 Stage 1–3 결과는 탐색적이다.
- VK는 합성이고 scene이 2개라 독립 확인이라기보다 '새 영상에서의 서술적 재현'이다.
- 극소 대상(< 0.1%)은 여전히 7객체뿐이다.

## 1. 최종 비교 (작은 객체, J&F)

| 방법 | DEV + REP (57객체 / 영상 34) Δ vs SGGL | Δ vs P5 | VK (28 / scene 2) Δ vs SGGL | VK Δ vs P5 | 추가 SAM 2 호출 / frame |
|---|---|---|---|---|---|
| 공식 SAM 2 (B0) | — (M6.1: gate set 전체 −0.016, 극소 −0.242 vs P5) | — | J −0.034 vs SGGL | J +0.001 vs P5 | 0 (전체 frame 1회) |
| P5 (B1) | −0.011 [−0.027, +0.010] | 0 | −0.017 | 0 | 0 |
| **SGGL (= A2x)** | **0** | **+0.011 [−0.010, +0.027]** (miami-surf obj5 −0.477 포함) | **0** | **+0.017 [+0.009, +0.025]** | +1 상태 (crop) |
| SGGL-V | +0.0069 [−0.003, +0.024], miami-surf 제외 −0.0019 | +0.018 | −0.00002 | +0.017 | +1 |
| M6.1 flip 2× 대조 (F2) | M6.1: SGGL이 +0.018 우세 | | — | | +1 |
| M6.1 CTU식 gate | M6.1: SGGL이 +0.008 우세 | | — | | +1 |
| M3-vel (최선 도전자) | +0.0046 [−0.000, +0.012] | +0.016 | +0.0013 [−0.000, +0.003] | +0.019 | +1 (SGGL과 같음) |
| M3-vel-r | +0.0044 [−0.000, +0.011] | | — | | +1 |
| F1-blend.7 | +0.0043 [−0.007, +0.021] | | — | | +1 |
| T2 (최선 attention head) | DEV만 −0.0013 [−0.003, +0.001] | | — | | +1 + head 15 ms |

## 2. 18개 후보 결과표

**표기 규칙:**
- 독립 영상 수는 그 후보 평가에 쓴 작은 객체 영상 수다.
- 지연 시간은 같은 부하에서 비교할 수 없어 '추가 SAM 2 호출 수'로 적었다. 측정한 벽시계 시간은 §4에 있다.
- VRAM은 모두 production 경로 최대 1.25 GB(RTX 5090)다. head를 쓰면 +0.5 GB다.

| ID | 메커니즘 | 구현 | 테스트 | 영상 | Δ vs SGGL | Δ vs P5 | 주 개선 | 주 회귀 | 추가 비용 | 가장 가까운 선행연구 | 새로움 위험 | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 재획득 상태에서만 교차 view 확인 (+ K-지속 대조) | 정책 (재생) | 재생 동등성 | 34 | −0.0000 / K3: −0.0005 | +0.011 | — (miami-surf에서 G도 방해물을 잡아 거짓 확인) | dog-gooses obj5 −0.051 | 0 | SatSAM2, KeepTrack, DeepSORT | 높음 | REJECT |
| S2 | G 관측 가능성 거부권 | 정책 (재생) | 같음 | 34 | +0.0070 (miami-surf 제외 −0.0019) | +0.018 | miami-surf obj5 | dog-gooses obj5 −0.056 | 0 | Küppers 2020 (크기별 보정), CTU | 중간 | REJECT (규칙이 손해 객체에서 한 번도 발동하지 않음) |
| S3 | frame-0 참조 신원 (절대 / 상대 margin) | 정책 (재생) | 같음 | 34 | +0.0003 / +0.0007 | +0.012 | miami-surf 일부 | dog-gooses −0.056 | 무시 가능 | SAMOSA, LTMU, SPLT, TLD | 높음 | REJECT |
| S4 | 양성 / 신뢰 음성 / 충돌 / 불충분 증거 | 정책 (재생) | 같음 | 34 | −0.0010 | +0.010 | — | dog-gooses −0.056 | 0 | (S2 + S3 조합) | 중간 | REJECT |
| F1 | 픽셀 logit 혼합 (0.5 / 0.7) | 정책 (재생 + 인과) | 경계 테스트 | 34 | +0.0013 / +0.0043 (인과) | +0.016 | miami-surf obj5 (연속값 거부권) | dog-gooses −0.058, 회귀 30객체 | 0 | Tao 2020, CGFSeg | 높음 | REJECT |
| F2 | 경계 띠 보정 | 정책 (재생) | 같음 | 34 | −0.0062 | +0.005 | — | 회귀 9 영상 | 0 | PointRend, BPR, CascadePSP | 높음 | REJECT |
| F3 | 비대칭 추가 / 삭제 편집 | 정책 (재생) | 경계 테스트 | 34 | −0.0023 (대칭 −0.0110) | +0.009 | 잘못 지운 TP가 대칭 대비 1/39 | 추가 FP > TP (27만 vs 17.5만 px) | 0 | MiVOS, hysteresis | 중간 | REJECT (비대칭성 효과는 측정됨) |
| F4 | 성분 단위 같은 instance 합집합 | 정책 (재생) | 경계 테스트 | 34 | −0.0105 | +0.001 | — | FP / TP 1.74 | 0 | DAM4SAM 성분, panoptic merge | 중상 | REJECT |
| M1 | 보호 / 신뢰 메모리 반사실 | probe + 정책 (재생 + 인과) | **격리 테스트**, 재생 동등성 | 34 | +0.0003 (인과) | +0.011 | 거의 없음 (26 frame) | lady-running −0.043 | +1 메모리 attention | Matthews 2003 (직접 선례) | 중간 | REJECT |
| M2 | 메모리 허용 분리 | planner 구현 (`Dual.excluded`) | — | — | 측정 안 함 | — | — | — | 0 | DAM4SAM, SAMURAI, SAM2Long | 높음 | IMPLEMENTED_NOT_EVALUATED (인과 근거 없음: 범주 9가 영상 2개) |
| M3 | crop 중심 제어 (속도 예측; 개정판) | 정책 (인과 전용) | 기계 간 동일 | 34 | **+0.0046** / 개정판 +0.0044 | +0.016 | skate-park +0.169, sheep +0.060 | lab-coat obj2 −0.014 ~ −0.018 | 0 | DaSiamRPN, KeepTrack (search region) | 높음 | **INCONCLUSIVE** (가장 강한 도전자, 영상 2개) |
| R1 | 같은 crop·메모리에서 해상도만 낮춤 | probe | 격리 | 39 (DEV + REP 전체) | (진단) | — | 원인 발견: G 실패는 화소가 아니라 특징 격자 점유 문제 | 맞는 / 틀린 후보 구별 불가 (4.3% vs 2.3%) | 사건 frame +encoder | Rottmann & Schubert 2019 | 낮음–중간 | DIAGNOSTIC_ONLY |
| R2 | 2× crop 상태 일치 (재획득 확인) | 상태 + 정책 (재생) | 재생 | 34 | +0.0009 | +0.012 | — | — | +1 상태 | multi-scale 일관성 | 낮음–중간 | REJECT |
| T1 | 정렬 특징 CNN 결합 | head + 학습 | 항등 초기화, 기울기, 정렬, PC | 12 (DEV) | −0.0134 | — | — | −0.125 | +head 14 ms | SAM2Refiner, GLNet | 높음 | REJECT |
| T2 | 목표 token cross-attention | head + 학습 | 같음 | 12 | −0.0013 | — | T1 대비 +0.012 | — | +16 ms | SAM2Refiner + Cutie 조합 | 중간 | REJECT |
| T3 | 기권 gate | head + 학습 | gate 범위 테스트 | 12 | −0.0008 | — | T2 대비 +0.0005 | — | 같음 | Flamingo gate, Tao | 중간 | REJECT |
| T4 | 신원 제약 attention | head + 학습 | 같음 | 12 | −0.0081 | — | — | T3 대비 −0.007 | 같음 | OSTrack 후보 제거 | 중간 | REJECT |
| O1 | 사건 기반 실행 | 비용 분석 | — | — | — | — | 사건 frame 4.8% → 검증 비용 약 5%로 줄일 수 있음 | 검증 자체에 이득이 없음 | 절감 | Viola–Jones 계열 | 높음 | DIAGNOSTIC_ONLY |
| K1–K5 | 조합 4종 | 정책 (재생) | 같음 | 34 | −0.0005 ~ −0.0011 | | — | dog-gooses −0.056 | — | | | REJECT |

## 3. 최종 알고리즘: SGGL

### 3.1 모듈 인터페이스 (`src/spatial/sggl.py`)

```python
d = sggl.Dual(predictor, frame_paths, gt0)       # G: full frame, P5 prompt memory; L: crop side s (None -> P5 only)
for t in 1..T-1:
    G = d.view("G", t)                            # View(out, logit 1x1x256x256, mask HxW bool, score, iou, box=None)
    L = d.view("L", t)                            # same; mask s x s bool, box = (y0, y1, x0, x1) in frame coords
    out, used_crop = sggl.a2x(G, L, (H, W))       # out: HxW bool
    d.set_center(vcam.center_of(out))             # next crop centre (unchanged if out empty)
d.close()
```

### 3.2 의사코드와 상태 전이

```text
input : frames 1..T-1 (RGB, H x W), prompt mask M0 (frame 0 only)
s     <- clamp(4 * longest side of box(M0), min(H,W)/4, ...) ; if s >= min(H,W): run P5 only (crop disabled)
G     <- SAM2 state on the full frame, prompt M0, memory encoded from the full-res mask (P5)
c     <- box centre of M0
L     <- SAM2 state on the virtual camera video (square s x s crop centred at c, resized to 1024), prompt M0 cropped (P5)
for t = 1 .. T-1:
    box_t <- square of side s centred at c, shifted inside the frame          (depends only on outputs <= t-1)
    (m_G, z_G) <- G.track(frame t)              ; stored in G's memory
    (m_L, z_L) <- L.track(crop(frame t, box_t)) ; stored in L's memory (crop coordinates only)
    accept <- z_L > 0  and  m_L non-empty  and  m_L does not touch a crop edge that is not a frame edge
    out_t  <- paste(m_L, box_t) if accept else m_G
    c      <- box centre(out_t) if out_t non-empty else c
    emit out_t
```

**상태 전이와 불변 조건:**
- G와 L은 메모리를 교환하지 않는다.
- GT는 frame 0 prompt에만 쓴다.
- L의 메모리는 crop 좌표에만 있다.
- crop 위치는 직전 출력에만 의존한다(인과 테스트로 확인).

### 3.3 모양과 비용 (Hiera-T, 1024 입력)

**모양:**
- 상태당 입력은 3×1024×1024, decoder 출력은 1×1×256×256 logit이다.
- object score는 스칼라, 메모리 특징은 64×64×64(bf16)다.

**비용:**
- crop이 적용된 객체는 frame마다 SAM 2 track step 2회다(G, L).
- 적용되지 않은 객체는 P5와 같은 1회다.

**측정 (production 경로, 측정 부하는 §4 참조):**

| GPU | s/frame | 최대 VRAM |
|---|---|---|
| RTX 2080 | 0.44 | 1.1–1.6 GB |
| RTX 5090 | 0.14 (단일, 4 thread) | 1.25 GB |

**선택 사항 (기본 off):**
- `a2v`: SGGL-V 거부권.
- `M3-vel`: crop 중심 속도 예측. `Policy.center` 한 줄 교체로 바꿀 수 있다.

## 4. 측정 조건 메모 (공정성)

- 벽시계 지연 시간은 실행 시점의 동시 부하에 따라 0.24–0.49 s/frame(RTX 5090)로 달라진다. 그래서 후보 간 비교에 쓰지 않았다.
- 같은 부하에서 측정한 profile(SGGL 1 process, 4 thread): SAM 2 GPU 133 ms, JPEG decode·crop 65 ms, J/F 채점 28 ms.
- 비교는 기계 간 수치 차이를 피하려고 같은 기계 기준선과 짝지었다([`MIGRATION_RTX5090.md`](MIGRATION_RTX5090.md)).
