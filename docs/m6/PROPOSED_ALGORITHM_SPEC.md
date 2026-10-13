# 제안 알고리즘 명세: Self-Gated Global–Local SAM 2 (SGGL)

> **상태: 잠정.**
> - 개발 세트에서 측정한 효과는 작은 객체 ΔJ&F +0.018 [+0.009, +0.025], 회귀 0이다. 사전 관문 +0.03에는 미달했다.
> - 이 형태(A2x)는 M6 2회차 **사후 탐색**에서 나왔으므로 새 데이터에서 사전 등록 검증이 필요하다.
> - 현재 코드(`scripts/m6.py run_unit(arm="A2x")`)는 기록된 B1 출력을 전역 상태로 재사용한다. 결정론적이어서 동등하다. 아래는 두 상태를 한 루프에서 함께 돌리는 **배포용** 명세다.

## 1. 구조

```mermaid
flowchart LR
  F[frame t RGB, 원본 해상도] --> G[상태 G: SAM 2 전체 frame<br/>P5 prompt 메모리]
  F --> C[crop: 정사각 s×s, 중심 c_(t-1)] --> L[상태 L: SAM 2 crop 영상<br/>P5 prompt 메모리, crop 좌표]
  G -->|mask_G, score_G| S{gate:<br/>score_L > 0 ∧ mask_L ≠ ∅<br/>∧ 내부 경계 미접촉}
  L -->|mask_L, score_L| S
  S -->|예: paste mask_L| O[출력 mask_t]
  S -->|아니오: mask_G| O
  O -->|box 중심| U[c_t 갱신] -.-> C
```

**두 상태의 분리:**
- G와 L은 메모리를 **공유하지 않는다.**
- L은 언제나 같은 배율(s → 1024)의 '팬만 하는 가상 카메라' 영상을 본다.
- 좌표 변환은 출력 마스크 붙여 넣기에만 쓴다.

## 2. 의사코드

```python
def sggl(frames, gt0, predictor, target=0.25, max_zoom=4.0):
    H, W = gt0.shape
    s = crop_side(gt0, target, max_zoom)            # None -> 전체 frame만 (P5와 동일)
    G = init_state_p5(predictor, frames, gt0)        # m56.start_dir(p5=True)
    if s is None:
        yield from propagate(G); return
    c = box_center(gt0)
    Lf = LazyCropFrames(frames); Lf.box = place(c, s, (H, W))
    L = init_state_p5(predictor, Lf, crop(gt0, Lf.box), video_res=(s, s))
    for t in 1..T-1:
        Lf.box = place(c, s, (H, W))                 # t-1까지의 출력만 사용 (인과)
        mG, scoreG = step(predictor, G, t)           # 각 상태의 메모리에 t 저장
        mL, scoreL = step(predictor, L, t)
        ok = scoreL > 0 and mL.any() and not touches_interior_edge(mL, Lf.box, (H, W))
        out = paste(mL, Lf.box, (H, W)) if ok else mG
        c = box_center(out) or c
        yield t, out
```

## 3. 구성요소와 인터페이스 (기존 코드 재사용)

| 구성요소 | 위치 | 인터페이스 |
|---|---|---|
| 배율 규칙 | `src/spatial/vcam.py:crop_side` | `(gt0, target) -> int or None` |
| crop 배치 | `vcam.place` | `(center, side, hw) -> (y0, y1, x0, x1)`. frame 안으로 이동시킴 |
| 지연 crop frame | `vcam.LazyCropFrames` + `vcam.lazy_loader` | SAM 2 `init_state`의 frame 텐서 대체. identity box면 공식 로더와 비트 단위로 같음 |
| P5 prompt 메모리 | `scripts/m56.py:start_dir(p5=True)` | 상태 초기화 |
| frame 진행 | `memint.engine.predict` + `non_cond_frame_outputs[t]` 저장 | 상태별 1 frame |
| gate | `vcam.touches_border` + object score | bool |
| 붙여 넣기 | `vcam.paste` | crop 마스크 → 원본 |

**새로 구현할 것:**
- `src/spatial/sggl.py` 한 파일: 두 상태를 한 루프에서 돌리는 위 의사코드.
- gate 함수를 분리한다(ablation용). 후보: `self` / `agreement`(CTU) / `none`.

## 4. 필요한 상태 정보

| 상태 | 내용 |
|---|---|
| G | SAM 2 inference_state 1개 (전체 frame) |
| L | SAM 2 inference_state 1개 (crop 영상, video_res = s×s) |
| 그 밖 | crop 중심 c (float 2개), crop 한 변 s (상수), 직전 출력 마스크 (원본 해상도 bool) |

- GT는 frame 0의 prompt에만 쓴다.
- 미래 frame은 쓰지 않는다.

## 5. 메모리와 계산 (RTX 2080, Hiera-T, FP32, 측정값)

| 항목 | P5 단독 | SGGL |
|---|---|---|
| 지연 (crop 적용 객체) | 0.208 s/frame | 약 0.43 s/frame (두 상태 + crop JPEG 디코딩) |
| 지연 (미적용 객체) | 0.208 | 0.208 (L 없음) |
| 최대 VRAM | 0.85–1.0 GB | 약 1.0–1.3 GB 예상 (두 상태. 영상은 CPU offload, 특징은 frame별 캐시 1개) |
| CPU RAM | 전체 frame 1024² 텐서 (300 frame에 약 3.8 GB) | + crop frame은 지연 생성이라 추가 거의 없음 |

## 6. 통제 ablation 계획

| # | 비교 | 시험할 가설 | 사전 판정 기준 |
|---|---|---|---|
| 1 | gate: self vs agreement(CTU) vs none(A1) | H-gate, H-safety | self − agreement 영상 CI > 0 |
| 2 | 배율 목표 0.25 vs 0.35. clamp 4× vs 8× | H-scale (극소 대상에 최대 확대 제한이 걸리는지) | 극소 대상 Δ 단조 증가 여부 |
| 3 | crop 중심: 융합 출력 vs crop 출력 | 재획득 경로 | escape 비율 |
| 4 | 경계 접촉 조건 있음 vs 없음 | escape 감지 | 보통 객체 회귀 |
| 5 | 고정 배율 vs 가변 배율(A1v 방식) | 메모리 배율 일관성 (M6에서 결론 불가) | 탐색용 |
| 6 | Hiera-T vs Hiera-L | backbone 의존성 | L에서 효과가 절반 이하이면 위험 |
| 7 | CTU 원형(H/4, H/2 창 병렬, 일치 gate) | 최근접 선행연구 재현 | 기준선 |

## 7. 정확성 테스트 (구현 시 필수)

- s = None이면 P5 출력과 비트 단위로 같다 (기존 `test_identity_crop_equals_B1...` 확장).
- G 단독 경로는 B1과 같다.
- 인과성: n frame 실행과 n+k frame 실행의 앞 n frame이 같다(A2 경로 포함).
- 상태 격리: L 메모리에 G의 출력이 들어가지 않는다(출력 dict 객체 id 검사).
