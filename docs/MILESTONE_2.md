# Milestone 2 — 순정 SAM 2.1 Hiera-Tiny baseline (VKITTI2 개발 장면)

이번 단계의 범위:
- 공식 SAM 2.1 Hiera-Tiny를 **코드·가중치·메모리 동작을 바꾸지 않고** 실행했다. 프레임 누락은 없다.
- 개발 장면 Scene01·06만 사용했다. 시험 장면 02·18·20은 읽지 않았다.
- 메모리 선택 정책은 아직 넣지 않았다. M1에서 확인한 confound(이동 중 카메라에서는 recent-K = pose-only)는 그대로 남아 있다. 이 단계의 결과로 자세 기반 개선에 대해 어떤 결론도 내리지 않는다.

## 1. 실행 환경 (2026-10-12)

| 항목 | 값 |
|---|---|
| 하드웨어 | Apple M4, 16 GB unified memory, macOS 27.0.1 arm64 |
| Python / 환경 | 3.12.15, uv (`uv.lock`) |
| PyTorch | 2.14.1, torchvision 0.29.1 |
| 공식 SAM 2 | `facebookresearch/sam2` @ `2b90b9f5ceec907a1c18123530e92e794ad901a4` (2024-12-15). pyproject의 `sam2` 의존성 그룹에서 git 패키지로 설치하며, 소스는 저장소에 복사하지 않는다 |
| 모델 설정 | 패키지 내장 `configs/sam2.1/sam2.1_hiera_t.yaml` |
| 체크포인트 | `sam2.1_hiera_tiny.pt` (092824 배포본), 156,008,466 B, sha256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` (로컬에서 계산. 공식 배포처는 checksum을 제공하지 않는다), `checkpoints/`에 두며 git에서 제외 |
| 장치 / 정밀도 | **MPS**, **float32** (autocast 없음) |
| CUDA 확장 `sam2._C` | 없음 (`SAM2_BUILD_CUDA=0`) |

정밀도와 장치 설정의 근거:
- 공식 노트북은 CUDA일 때만 bfloat16 autocast를 쓴다. MPS·CPU에서는 float32와 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 쓰며, 이 실행도 그대로 따랐다.
- 계획서의 FP16 조건은 CUDA 기준이므로 이 실행에는 적용하지 않았다.
- 실행 중 MPS에서 CPU로 넘어간 연산은 경고 기준으로 없었다.

## 2. 모델과 추론 설정 (공식 기본값 그대로)

- 생성 방식: `build_sam2_video_predictor(cfg, ckpt, device, apply_postprocessing=True)`. 공식 기본값이며 다음 설정이 함께 켜진다.
  - `dynamic_multimask_via_stability=true` (delta 0.05, thresh 0.98)
  - `binarize_mask_from_pts_for_mem_enc=true`
  - `fill_hole_area=8`
- 모델 값(실행 중 확인): `num_maskmem=7`, `memory_temporal_stride_for_eval=1`, `max_cond_frames_in_attn=-1`, `use_obj_ptrs_in_encoder=true`, `max_obj_ptrs_in_encoder=16`, `add_tpos_enc_to_obj_ptrs=true`, `image_size=1024`.
  - 즉 메모리는 조건 프레임(prompt) + 최근 6프레임이고, object pointer와 시간 위치 인코딩도 기본값 그대로다.
- 프롬프트: `add_new_mask(frame_idx=0, obj_id=1, mask=GT)`를 **한 번만** 호출한다. 이후 `propagate_in_video`로 순방향 추적한다. 예측 mask는 logits > 0이다.
- 공식 파일이 수정되지 않았음을 sha256으로 확인했다(`tests/test_sam2_adapter.py`). 대상 파일: `sam2_base.py`, `sam2_video_predictor.py`, `memory_attention.py`, `memory_encoder.py`, `build_sam.py`, `sam2.1_hiera_t.yaml`.
- **알려진 차이:** `_C` 확장이 없어서 공식 코드가 hole-filling 후처리(`fill_hole_area=8`)를 경고와 함께 건너뛴다. 공식 코드에 들어 있는 fallback 경로이지만, CUDA에서 실행한 결과와는 작은 구멍 처리에서 다를 수 있다.

## 3. 데이터 어댑터와 인덱스 분리 (`src/baseline/sam2_vos.py`)

- 공식 loader는 `<int>.jpg` 이름만 읽는다. 그래서 track마다 `outputs/milestone2/clips/<scene>_trackNNN/{video_idx:05d}.jpg` 형태의 symlink를 만들어 원본 VKITTI JPEG을 가리키게 했다. SAM2는 원본과 같은 바이트를 디코딩한다.
- `mapping.csv`와 `reports/video_index_mapping.csv`에 `video_idx ↔ original_frame_id`를 저장했다. 예: Scene01 track 90은 video_idx 0 = 원본 frame 130.
- `run_clip(predictor, video_dir, prompt_mask)`는 데이터셋을 받지 않는다. 받는 GT는 prompt mask 하나뿐이다. 나머지 프레임의 GT는 추론이 끝난 뒤 평가 단계에서만 읽는다.
- 대상 차량 GT pose(pose.txt)는 어디에서도 읽지 않는다.
- clip 범위는 M1의 `first_area` 규칙을 따른다: prompt frame부터 마지막 관측 frame까지 연속이다.
  - 끝점은 GT로 정한 평가 구간이다. 순방향 추론에서 frame t의 예측은 frame t 이하만 사용한다.
  - 공식 `init_state`는 clip 전체 프레임을 미리 메모리에 올린다. 하지만 아직 처리하지 않은 프레임의 특징이나 예측은 메모리에 들어가지 않는다.

## 4. 평가 프로토콜 (`src/baseline/metrics.py`)

- **J:** IoU. 예측과 GT가 모두 비어 있으면 1이다(DAVIS 관례).
- **F:** DAVIS-2017 boundary F (`_seg2bmap` 경계, disk dilation 허용오차 = ceil(0.008 × 대각선) = **11 px**).
  - 경계가 둘 다 없으면 P = R = 1, 예측만 비면 P = 1·R = 0, GT만 비면 P = 0·R = 1이다.
  - 구현은 numpy만 쓰며, disk dilation이 brute force 결과와 일치하는 것을 테스트로 확인했다.
- **J&F** = (J + F) / 2.
- **보조 지표 F_strict:** 허용오차 2 px. DAVIS 지표가 아니다. 작은 차량에서 11 px 허용오차로는 F가 거의 1에 붙어 버려서 따로 기록한다.
- prompt frame(video_idx 0)은 평균에서 제외하고, 그 J는 따로 기록한다.
- **GT가 빈 프레임**(완전 가림 등)도 주 평균에 DAVIS 관례대로 포함한다. 예측도 비면 1, 예측이 있으면 0이다. GT가 보이는 프레임만의 평균(`*_visible`)과, GT가 빈 프레임에서 예측이 나온 false positive 수도 함께 기록한다.
- **시야 이탈:** 마지막 관측 frame 이후는 평가하지 않는다. clip 중간에 잠시 사라지는 경우는 GT가 빈 프레임으로 처리한다.
- **tracking loss:** J < 0.1 (계획서 §9). 첫 손실 frame도 기록한다.
- **통계 단위:** track이다. 프레임을 독립 표본으로 보지 않는다. 장면 값은 track 평균의 평균, 전체 값은 8개 track 평균의 평균이다. 프레임 수준 신뢰구간은 계산하지 않았다.

## 5. Track과 프레임 범위

| 장면 | track | 원본 frame 범위 | 평가 frame (prompt 제외) | GT가 빈 frame |
|---|---|---|---|---|
| Scene01 | 90 (Van, static) | 130–235 | 105 | 0 |
| Scene01 | 86 (Car, moving) | 339–425 | 86 | 0 |
| Scene01 | 63 (Car, static) | 217–297 | 80 | 0 |
| Scene01 | 54 (Car, static) | 156–228 | 72 | 0 |
| Scene06 | 12 (Car, mixed) | 85–220 | 135 | 7 (92–95, 138–140) |
| Scene06 | 9 (Truck, moving) | 69–139 | 70 | 0 |
| Scene06 | 11 (Van, moving) | 123–178 | 55 | 0 |
| Scene06 | 13 (Van, moving) | 185–239 | 54 | 0 |

## 6. 정확도 결과 (실측, MPS)

| 장면 | track | J | F | **J&F** | J&F (GT 보이는 frame만) | F_strict (2 px) | J < 0.1 frame | prompt frame J |
|---|---|---|---|---|---|---|---|---|
| Scene01 | 90 | 0.936 | 0.989 | **0.963** | 0.963 | 0.868 | 0 | 0.942 |
| Scene01 | 86 | 0.913 | 0.983 | **0.948** | 0.948 | 0.905 | 1 (frame 380) | 0.908 |
| Scene01 | 63 | 0.866 | 0.998 | **0.932** | 0.932 | 0.850 | 0 | 0.951 |
| Scene01 | 54 | 0.920 | 1.000 | **0.960** | 0.960 | 0.922 | 0 | 0.942 |
| Scene06 | 12 | 0.860 | 1.000 | **0.930** | 0.926 | 0.961 | 0 | 0.938 |
| Scene06 | 9 | 0.926 | 0.993 | **0.959** | 0.959 | 0.890 | 0 | 0.986 |
| Scene06 | 11 | 0.900 | 0.998 | **0.949** | 0.949 | 0.920 | 0 | 0.971 |
| Scene06 | 13 | 0.935 | 1.000 | **0.968** | 0.968 | 0.959 | 0 | 0.968 |
| **Scene01 (track 평균)** | 4 track, 343 frame | 0.909 | 0.993 | **0.951** | 0.951 | 0.886 | | |
| **Scene06 (track 평균)** | 4 track, 314 frame | 0.905 | 0.998 | **0.952** | 0.951 | 0.933 | | |
| **전체 (8 track 평균)** | 657 frame | 0.907 | 0.995 | **0.951** | 0.951 | 0.909 | | |

- GT가 빈 7개 frame(Scene06 track 12)에서는 SAM2도 빈 mask를 냈다(false positive 0).
- GT가 보이는 650개 frame 중 J < 0.5는 1개, J < 0.7은 7개다.
- **MPS vs CPU 교차검증:** Scene01 track 90 전체(106 frame)를 CPU float32로 다시 실행했다. 105개 frame은 완전히 같았고, 나머지 1개도 1픽셀만 달랐다. J&F는 소수점 4자리까지 같았다(`outputs/milestone2_cpu_check/`). 10-frame smoke test에서도 두 장치의 결과가 완전히 같았다.

## 7. 그래프 (`outputs/milestone2/figures/`, git 제외)

| 내용 | 파일 |
|---|---|
| RGB · GT · 예측 · overlay · 확대 (track별: prompt, 첫/중간/마지막 평가 frame, 최저 J&F frame) | `<scene>_trackNNN_side_by_side.png` (8개) |
| J/F/J&F 시간 변화 (회색 = GT가 빈 frame) | `jf_over_time.png` |
| track별 J/F/J&F | `per_track_jf.png` |
| 성공 사례 | `success_Scene01_track063_frame0029{5,6,7}.png` |
| 실패 사례 | `failure_Scene01_track086_frame00380.png`, `failure_Scene06_track012_frame00117.png`, `failure_Scene06_track012_frame00137.png` |

수치 파일: `reports/per_frame_metrics.csv`, `per_track_metrics.csv`, `per_scene_metrics.csv`, `video_index_mapping.csv`, `runtime.csv`, `<clip>_latency.csv`, `environment.json`, `m2_report.json`. 예측 mask는 `predictions/*.npz`에 있다.

## 8. 실행 시간 (MPS float32, Apple M4. CUDA 수치와 비교하면 안 됨)

| 항목 | 값 |
|---|---|
| 모델 생성 | 0.27 s |
| clip 초기화 (프레임 로딩 + prompt 인코딩) | 0.86–2.57 s (clip 길이에 비례) |
| 프레임당 지연 (prompt 제외, track별 중앙값) | 0.96–1.04 s → 처리 속도 **약 1.0 FPS** |
| 메모리 bank가 찬 뒤 평균 지연 (video_idx ≥ 7) | 0.98–1.04 s |
| MPS driver 할당량 최대 (캐시 포함, 측정값) | Scene01 3.81 GB, Scene06 5.55 GB. 공식 `init_state`가 clip 전체 프레임을 1024² float32로 미리 올리기 때문에 clip 길이에 따라 커진다 |
| 프로세스 최대 RSS | 2.16 GB |

10-frame 진단 측정(계측 래퍼만 씌우고 코드는 수정하지 않음)의 구성 요소별 시간:

| 구성 요소 | MPS | CPU |
|---|---|---|
| 이미지 인코더 | 195 ms | 461 ms |
| memory attention | **569 ms** | 389 ms |
| SAM heads | 14 ms | 22 ms |
| memory encoder | 15 ms | 50 ms |

전체 속도가 CPU(중앙값 1.06 s/frame)와 비슷한 이유는 MPS의 memory attention이 CPU보다 느리기 때문이다. 원인(RoPE 연산 또는 attention 커널)은 확인하지 않았다.
- **이 머신 기준 추정:** M7까지 같은 실행을 반복하면, 개발 장면 657 frame × 정책 4 × 누락 조건 5 × seed 3 ≈ 39k frame으로 약 11시간이 걸린다.

## 9. 테스트

`uv run pytest`: **66개 통과**. 이 중 M2 관련 테스트는 13개다.
- `test_metrics.py`:
  - J 값과 빈 mask 처리 관례
  - `seg2bmap`의 정확한 경계 픽셀(손으로 계산한 18픽셀과 일치)
  - disk dilation = brute force
  - F 허용오차 안/밖, shape 오류
- `test_sam2_adapter.py`:
  - clip mapping과 결정론성
  - **공식 loader가 mapping 순서대로 프레임을 읽음**
  - fake predictor로 확인한 **GT prompt 1회, frame 0에만** 주어지는 것과 호출 순서, 출력 mask 크기와 dtype
  - `run_clip` 시그니처에 데이터셋 인자가 없음
  - 공식 파일 sha256 일치
  - 실제 체크포인트·데이터로 CPU 3-frame 통합 테스트 (메모리 기본 설정 확인, J > 0.5)

## 10. 연구 검토

### 관찰 (실제 실행 결과)
- 누락이 없는 조건에서 순정 SAM 2.1 Tiny는 선택된 8개 track 모두를 끝까지 추적했다. track별 J&F는 0.930–0.968이고, tracking loss(J < 0.1)는 frame 1개뿐이다.
- 그 1개 frame은 Scene01 track 86의 frame 380이다. 대상이 흰 SUV 뒤에 거의 다 가려져 GT가 37 px(bbox occupancy 0.007)일 때 예측이 비었다. 다음 frame 381에서 바로 다시 잡았다(J 0.84).
- Scene06 track 12는 완전 가림이 두 번 있었다(4 frame, 3 frame). 가려진 동안 예측은 비어 있었다. 다시 나타난 첫 frame의 J는 0.63(frame 96), 0.75(frame 141)였고, 1–2 frame 뒤에 0.89(frame 97), 0.91(frame 143)로 회복했다.
- 정확도는 대상 크기와 강하게 관련된다. GT가 보이는 frame에서 J와 GT 픽셀 수의 Spearman 상관은 0.86이다. 크기 구간별 J 평균은 다음과 같다.

  | GT 크기 | J 평균 | frame 수 |
  |---|---|---|
  | ≤ 1000 px | 0.83 | 192 |
  | 1000–3000 px | 0.90 | 208 |
  | 3000–10000 px | 0.93 | 133 |
  | > 10000 px | 0.98 | 117 |
- Scene01 track 90은 처음에 작을 때(약 500 px) J가 0.70–0.80이었고, 차량이 커지면서 0.95 이상으로 올라갔다.
- 부분 가림 구간에서 J가 떨어진다. 예: track 12 frame 117–118에서 GT가 50–110 px로 얇게 보일 때 J 0.55–0.59.
- 이 해상도와 대상 크기에서는 DAVIS F(11 px)가 거의 포화된다(평균 0.995). 정책 간 차이를 보려면 J와 F_strict가 더 민감하다.

### 가설 (아직 검증하지 않음)
- 이 baseline은 이미 0.95 수준이라 누락 없는 조건에서는 개선 여지가 작다. 평균 J&F보다 누락 직후 회복 지표가 차이를 드러낼 가능성이 크다.
- 짧은 완전 가림(3–4 frame)에서 바로 회복했으므로, 비슷한 길이(4)의 burst 누락만으로는 성능 저하가 작을 수 있다. M4에서 확인이 필요하다.
- 누락이 어려울 수 있는 조건: 작은 대상(≤ 1000 px), 부분 가림 직전·직후, 빠르게 움직이는 차량의 진입·이탈(track 86, 12), 대상이 처음 나타나 작은 구간.
- 정지한 카메라 + 이동 차량(Scene06)에서는 대상의 외형 변화를 카메라 자세로 설명하지 못한다(M1 결과와 일치).

### 구현상 제한
- `_C` 확장이 없어 hole-filling 후처리를 건너뛴다. 실행 장치도 MPS·float32라 CUDA bfloat16 공식 환경과 수치가 다를 수 있다(이 머신에서 MPS와 CPU 결과는 사실상 같았다).
- torch 2.14.1은 upstream 시점(torch ≥ 2.5.1 요구)보다 훨씬 새 버전이다.
- 실행 시간은 이 Mac에서만 측정했고, MPS 메모리 수치는 캐시를 포함한다.
- 개발 장면 track 8개, 누락 없는 조건, 1회 실행이다. 일반화하려면 더 많은 데이터가 필요하다.
- 공식 `init_state`는 clip 전체 프레임을 미리 올린다. M3에서는 수신된 frame만 폴더에 넣어 누락을 표현해야 한다.

## 11. Milestone 3 준비 상태

- 준비된 것:
  - baseline 파이프라인 (원본 frame ↔ video_idx mapping, GT prompt 1회, 평가, 그래프)
  - 개발 track 8개의 누락 없는 기준값
  - 회귀 테스트
- M3에서 할 일:
  - clip 폴더에 **수신된 frame만** symlink하고 mapping에 원본 frame 번호를 유지한다. 그러면 공식 predictor는 수정하지 않아도 된다.
  - 평가는 계획서 §9에 따라 수신된 frame만 대상으로 한다.
- M1의 confound는 여전히 해결되지 않았다. 정책 비교(M5 이후)에 들어가기 전에 멘토와 결정해야 한다.
