# Milestone 0 — Virtual KITTI 2 데이터 파이프라인

범위: 데이터 준비, 정합성 검증, 카메라 pose 처리, 차량 instance mask 추출과 시각화. SAM 2.1 추론·학습은 포함하지 않는다. 연구 설계의 기준은 [mentor_plan_revision.md](../guide-build/mentor_plan_revision.md)다.

## 1. 개발 컴퓨터 (2026-10-12 확인)

| 항목 | 값 |
|---|---|
| OS | macOS 27.0.1 (arm64 네이티브, Rosetta 미사용) |
| CPU / RAM | Apple M4 (10코어 GPU, Metal 4) / 16 GB |
| GPU 백엔드 | CUDA 없음. 향후 PyTorch는 MPS 백엔드만 가능 |
| 디스크 | 228 GB 중 약 90 GB 여유 (작업 전) |
| Python | 프로젝트 3.12.15 (uv 관리), 시스템 python3는 3.14 |
| 도구 | git 2.54, gh (msilot1001 인증), Homebrew, uv 0.13.0 |

Milestone 0은 CPU만 사용한다. SAM 2.1 Hiera-Tiny 추론은 이 Mac의 MPS로 시도할 수 있지만, 계획서의 FP16·peak VRAM·latency 측정은 CUDA GPU 기준이므로 최종 비용 측정에는 CUDA 환경을 별도로 준비하는 것이 안전하다.

## 2. Python 환경

```bash
brew install uv          # 없을 때만
uv sync                  # .python-version(3.12) + uv.lock 대로 .venv 생성
```

의존성은 `pyproject.toml`과 `uv.lock`으로 고정한다: numpy, pandas, pillow, matplotlib, pyyaml, (dev) pytest. OpenCV·PyTorch는 아직 설치하지 않는다.

## 3. 데이터셋 다운로드

공식 페이지: <https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/>
라이선스: **CC BY-NC-SA 3.0, 비상업적 용도만 허용**. 다운로드하면 약관에 동의한 것으로 간주된다. 연구에 사용할 때 Cabon et al. 2020(Virtual KITTI 2)과 Gaidon et al. 2016을 인용하고 Naver를 원작자로 표기한다.

```bash
python3 scripts/setup_vkitti.py                                    # 기본값: Scene01 / clone / Camera_0
python3 scripts/setup_vkitti.py --scenes Scene01 Scene06 Scene02 Scene18 Scene20   # 이후 마일스톤용
```

| 아카이브 | 크기 (bytes) | 공식 MD5 |
|---|---|---|
| vkitti_2.0.3_instanceSegmentation.tar | 172,298,240 | c79684c6344e50663587b8da848535e7 |
| vkitti_2.0.3_textgt.tar.gz | 24,451,078 | 855f7f37746e5508f094e522c9cf41f4 |
| vkitti_2.0.3_rgb.tar | 7,532,472,320 | 1e00a143a397c2c53aa9720a868fe34a |

depth, classSegmentation, flow, scene flow 아카이브는 받지 않는다. 다운로드는 `curl -C -`로 `*.part` 파일에서 이어받으며 10회까지 재시도하고, 1 KB/s 미만 속도가 60초 이어지면 timeout으로 처리한다. 완료되면 크기와 MD5를 검증한 뒤에만 정식 파일명으로 바꾼다. 압축 해제는 tar 전체를 풀지 않고 요청한 scene/variation/camera만 스트리밍으로 추출한다. 같은 크기의 파일이 이미 있으면 건너뛰고, 쓰는 중에는 `.tmp`에 기록한 뒤 이름을 바꾼다. 다시 실행해도 안전하다(재실행 시 약 5초).

## 4. 데이터 디렉터리 구조 (아카이브 내부 경로를 실제로 확인함)

```
data/vkitti2/                       # .gitignore 대상
  archives/*.tar[.gz]               # 원본 아카이브 (~7.7 GB, 이후 장면 추출용으로 보존)
  Scene01/clone/
    frames/rgb/Camera_0/rgb_00000.jpg ... rgb_00446.jpg
    frames/instanceSegmentation/Camera_0/instancegt_00000.png ...
    extrinsic.txt intrinsic.txt bbox.txt info.txt colors.txt pose.txt
```

공식 페이지에는 `frames/instancesegmentation/`(소문자)로 적혀 있지만, 실제 아카이브의 경로는 `frames/instanceSegmentation/`이다. 아카이브 경로는 `Scene01/...`에서 바로 시작한다. textgt에는 두 카메라의 정보가 한 파일에 들어 있다. `pose.txt`도 함께 추출되지만 파이프라인은 이 파일을 읽지 않는다.

## 5. 실행 명령

```bash
uv run pytest                                  # 28개 fixture 테스트 + 1개 실데이터 통합 테스트
uv run python scripts/inspect_vkitti.py        # Milestone 0 smoke test (약 13초)
uv run python scripts/visualize_vkitti.py --track 86 --frames 339 382 425
```

`inspect_vkitti.py`는 하나라도 실패하면 exit 1을 반환한다. 설정은 `configs/vkitti_m0.yaml`에 있다.

## 6. 데이터 로더 반환 형식

```python
from vkitti.dataset import VKittiDataset
ds = VKittiDataset(root="data/vkitti2", scene="Scene01", variation="clone", camera_id=0)
s = ds.get_frame(frame_id=50, track_id=4)
```

| 키 | 형식 |
|---|---|
| `rgb` | (H, W, 3) uint8, RGB 순서 |
| `mask` | (H, W) bool, `instance == track_id + 1` |
| `original_frame_id`, `track_id`, `camera_id` | int (원본 VKITTI2 번호, 0부터 시작) |
| `track_visible` | 이 프레임에 mask 픽셀이 1개 이상 있는지 |
| `extrinsic` | 4×4 world→camera |
| `camera_position` | (3,) 월드 좌표의 카메라 중심 |
| `camera_rotation` | 3×3 camera→world 회전 |
| `scene`, `variation` | str |

- 생성 시 RGB·mask 프레임 번호 집합이 같은지, 번호가 연속되는지, 모든 프레임에 해당 카메라의 extrinsic이 있는지 확인한다. 총 프레임 수는 파일을 스캔해 정하며 하드코딩하지 않는다.
- 범위를 벗어난 frame은 `KeyError`, 정수가 아닌 frame은 `TypeError`, info.txt에 없는 track은 `KeyError`, 해상도 불일치는 `ValueError`, 누락 파일은 `FileNotFoundError`를 낸다.
- info.txt에는 있지만 해당 프레임에 보이지 않는 track은 오류가 아니다. 빈 mask와 `track_visible=False`를 반환한다.
- instance PNG는 mode `P`의 원본 인덱스를 그대로 읽는다. `convert("RGB")`를 쓰면 팔레트 색으로 바뀌어 track ID를 잃는다. 이 실수를 잡는 테스트를 넣어 두었다.
- 프레임 번호는 언제나 원본 VKITTI2 번호이고, 로더는 번호를 다시 매기지 않는다. 누락 시뮬레이션에서 수신 순번을 붙일 때도 이 값을 그대로 함께 들고 다녀야 한다.

## 7. 좌표계와 extrinsic

- `extrinsic.txt`의 한 행 = `frame cameraID` + row-major 4×4 행렬이다. 조회 키는 `(original_frame_id, camera_id)`다.
- **변환 방향: world→camera (`X_cam = R X_world + t`)이며, 실데이터로 두 가지를 확인했다.**
  1. Scene01/clone frame 0에서 `T @ [pose.txt world_space_XYZ, 1]`이 pose.txt의 `camera_space_XYZ`와 1e-6 이내로 일치한다. pose.txt는 이 검증에만 사용했다.
  2. 447개 전 프레임에서 Camera_1 중심이 Camera_0 좌표계의 +x 방향으로 0.532725 m 떨어져 있다(공식 baseline, 최대 오차 1.0e-4 m로 파일의 반올림 수준). 파일을 camera→world로 잘못 읽으면 이 검사가 실패하며, 테스트로도 확인한다.
- 카메라 pose: `T_c2w = inverse(T_w2c)`를 rigid 역행렬 `[Rᵀ, −Rᵀt]`로 정확히 계산한다. `camera_position = T_c2w[:3,3] = −Rᵀt`, `camera_rotation = Rᵀ`.
- 행렬 검사 항목: 4×4 크기, 마지막 행 `[0 0 0 1]`, `RᵀR = I`, `det R = +1`(허용 오차 1e-4), NaN/Inf, 프레임 간 이동 3 m·회전 15° 초과 여부.
- 카메라 좌표계는 x 오른쪽, y 아래, z 앞이다. 월드 y축도 아래를 향한다(카메라 y ≈ −112.65, 차량 y ≈ −111). 그래서 top view는 x–z 평면에 그린다.
- **데이터 누수 방지:** `vkitti` 패키지는 `pose.txt`(대상 차량 GT)를 읽지 않는다. 메모리 선택 입력으로 쓸 수 있는 pose는 `ds.camera(frame)`가 돌려주는 카메라 pose뿐이다.

## 8. Track 선택 기준 (실험 전에 고정)

계획서 §8을 다음과 같이 구현했다(`src/vkitti/tracks.py`). 모든 값은 실제 instance mask에서 센다.

1. prompt frame = 해당 track의 mask 픽셀이 처음으로 1개 이상 나타나는 프레임
2. prompt frame의 mask 면적 ≥ 0.1% × H × W (Scene01: 465.75 px)
3. mask가 보이는 프레임 수 ≥ 40
4. 정렬 기준: 보이는 프레임 수 내림차순 → prompt 면적 내림차순 → track ID 오름차순. 장면마다 상위 4개를 고른다.

입력 순서와 상관없이 결과가 같다는 것을 테스트로 확인했다.

## 9. 테스트

`tests/conftest.py`가 실제 2.0.3 헤더와 경로 구조를 그대로 따르는 6×8, 3프레임, 2카메라 합성 데이터를 만든다. 이 데이터로 RGB shape/dtype, indexed PNG 값 보존, trackID+1 규칙, 빈 mask, 해상도 불일치, 프레임–RGB–pose 대응, 카메라 0/1 구분, extrinsic 파싱, rigid 역행렬, 잘못된 행렬 검출, 궤적 연속성, stereo 방향 검사, track 선택과 결정론성, 누락 파일과 잘못된 ID 처리를 검증한다. `test_real_scene01_clone_cam0`은 실데이터가 있을 때만 실행된다. 정답 대신 일부러 버그를 넣은 코드(trackID+1 누락, RGB 변환)로 테스트가 실패하는 것도 확인했다.

## 10. 실제 결과 (Scene01 / clone / Camera_0, 2026-10-12)

| 항목 | 결과 |
|---|---|
| 프레임 | 447개 (0–446), RGB·mask·extrinsic 모두 대응, 1242×375 |
| track | info.txt 93개, 모두 mask에 등장. mask에 info.txt에 없는 ID 없음 |
| bbox `number_pixels` vs mask 픽셀 | 6947개 (frame, track) 쌍 모두 **정확히 일치** (불일치 0) |
| extrinsic | 447개 모두 유효한 rigid 행렬, 점프 없음 |
| 카메라 이동 | 총 332.5 m, 프레임당 평균 0.75 m(최대 1.25 m), 회전 최대 2.78°/frame(약 frame 200–230의 좌회전 구간), frame 400 이후 거의 정지 |
| 선택된 track | 86 (Car, prompt 339, 87프레임), 63 (Car, 217, 81), 83 (Car, 340, 70), 87 (Truck, 330, 46) |

출력물(`outputs/milestone0/`, git 제외):
- `overlays/Scene01_clone_Camera_0_track{ID}_frame{F}.png` — track별 prompt·중간·마지막 프레임
- `trajectories/Scene01_clone_Camera_0_trajectory_{2d,3d,timeseries}.png`
- `reports/Scene01_clone_Camera_0_{m0_report.json, tracks.csv, selected_tracks.csv, camera_trajectory.csv}`

## 11. 알려진 제한사항과 결정이 필요한 사항

- **선택 기준 해석:** 대부분의 track은 멀리서 아주 작게 처음 등장한다. 그래서 "첫 등장 프레임에서 0.1% 이상" 조건을 만족하는 track은 93개 중 5개뿐이다(40프레임 조건만 보면 75개). 가장 오래 보이는 track 90(236프레임)도 첫 프레임 면적이 6 px이라 제외된다. 다른 해석으로 "면적이 처음 0.1% 이상이 되는 프레임을 prompt로 삼는다"가 있지만 계획서 문장과 달라 적용하지 않았다. 바꾸려면 개발 장면 결과를 보기 전에 멘토와 확정해야 한다.
- 선택된 4개 중 3개는 `isMoving=True` 비율이 1.0인 이동 차량이다. 계획서 연구질문 4의 정지/이동 비교에는 장면별 구성이 치우칠 수 있다.
- 프레임당 이동거리에 약 0.1 m 폭의 주기적 진동이 있다. 회전에서는 보이지 않는다. 시뮬레이션 샘플링에서 생긴 것으로 보이지만 원인은 확인하지 않았다. 자세 비용 정규화(sT)에 주는 영향은 작을 것으로 예상한다.
- bbox.txt의 2D box는 mask의 tight extent와 1–2 px 다를 수 있다(track 63 frame 217에서 확인). box가 필요하면 mask에서 계산한다.
- **FPS:** 공식 페이지에는 원본 FPS가 적혀 있지 않다. Δt를 초로 바꾸려면 출처를 확인해야 한다. 원본 KITTI tracking은 10 Hz이지만 이 데이터에서는 검증하지 않았다.
- 지금은 Scene01만 추출했다. Scene06과 시험 장면 02·18·20은 `setup_vkitti.py --scenes`로 같은 아카이브에서 추출할 수 있지만 아직 검증하지 않았다.

## 12. Milestone 1 전에 준비할 것

- 위 선택 기준 해석을 확정한 뒤 Scene06을 추출하고 `inspect_vkitti.py`(config의 scene만 변경)로 같은 검증을 실행한다.
- Δt용 FPS의 근거를 확보한다.
- 공식 SAM 2.1 설치와 Hiera-Tiny checkpoint를 준비한다. 결정할 것: MPS로 개발할지, 측정용 CUDA 환경을 둘지.
- 프레임 누락 패턴(seed 3개)을 생성하고, 원본 frame ID ↔ 수신 순번 대응을 함께 저장하도록 설계한다.
