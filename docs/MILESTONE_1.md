# Milestone 0.5 + 1 — Track 선정 수정, Scene06, Memory Selector, 정책 분리도 분석

기준 문서: [mentor_plan_revision.md](../guide-build/mentor_plan_revision.md)를 source of truth로 따른다. Google Docs "SAM 2 성능 개선 선행연구 조사 및 과제연구 설계안"의 '연구 과정' 탭은 M1의 목적과 판정 기준(정책 간 선택이 약 95% 이상 겹치면 B 경로)을 정할 때만 참고했다.
범위: SAM 2.1은 설치하지도 수정하지도 않았다. 프레임 누락이 없는 조건만 다루며, 개발 장면은 Scene01과 Scene06이다. 시험 장면 02·18·20은 읽지 않았다.
**여기서 보이는 선택 결과의 차이는 SAM 2.1의 분할 정확도 차이를 뜻하지 않는다.**

## 1. M0 정리

- M0 브랜치(`df48449`)에서 테스트 29개가 모두 통과한 것을 확인했다. 이후 [PR msilot1001/sam2#1](https://github.com/msilot1001/sam2/pull/1)을 master에 merge commit(`f338c4d`)으로 병합했다. 저장소에 CI는 없다.
- M1 작업은 `feat/m1-memory-selector` 브랜치에서 진행했다. `data/`와 `outputs/`는 git에서 제외되어 있다.

## 2. Track 선정 기준 수정 (M0.5)

`src/vkitti/tracks.py`에 두 규칙을 모두 두고, `select_tracks(rule=...)`로 고른다. 기본값은 `first_area`다.

| 규칙 | prompt frame | 관측 조건 |
|---|---|---|
| `first_area` (새 기본값) | mask 면적이 **처음으로** H·W의 0.1% 이상이 되는 프레임 | prompt frame **이후**(prompt frame 제외) mask가 1픽셀 이상인 프레임 ≥ 40 |
| `first_visible` (M0, 비교용으로 보존) | track이 처음 보이는 프레임 (그 프레임의 면적 ≥ 0.1%) | prompt frame을 포함해 보이는 프레임 ≥ 40 |

정렬 기준은 두 규칙이 같다: 관측 프레임 수 내림차순 → prompt 면적 내림차순 → track ID 오름차순, 장면당 최대 4개. 판정은 mask 통계만 사용하며 SAM 2.1 결과는 보지 않는다. 같은 코드와 설정(`configs/m1_memory.yaml`, `configs/vkitti_m0.yaml`)을 시험 장면에도 그대로 쓴다.

| 장면 | track (info.txt) | 적격 track: M0 → M0.5 | 적격 track의 구성 (M0 → M0.5) |
|---|---|---|---|
| Scene01 | 93 | 5 → **23** | 정지 1·이동 4 → **정지 19·이동 4** |
| Scene06 | 15 | 7 → **11** | 이동 6·혼합 1 → **이동 10·혼합 1** |

`motion`은 bbox.txt의 `isMoving` 값으로 나눈다. 전 프레임 True이면 moving, 전 프레임 False이면 static, 섞여 있으면 mixed다.

선정 결과 (`first_area`):

| 장면 | track | 종류 | 첫 등장 frame (면적) | prompt frame (면적) | prompt 이후 관측 | 마지막 frame | motion |
|---|---|---|---|---|---|---|---|
| Scene01 | 90 | Van | 0 (6 px) | 130 (474) | 105 | 235 | static |
| Scene01 | 86 | Car | 339 (702) | 339 (702) | 86 | 425 | moving |
| Scene01 | 63 | Car | 217 (754) | 217 (754) | 80 | 297 | static |
| Scene01 | 54 | Car | 134 (4 px) | 156 (476) | 72 | 228 | static |
| Scene06 | 12 | Car | 85 (1086) | 85 (1086) | 128 | 220 | mixed (96% 이동) |
| Scene06 | 9 | Truck | 68 (73 px) | 69 (10292) | 70 | 139 | moving |
| Scene06 | 11 | Van | 123 (1517) | 123 (1517) | 55 | 178 | moving |
| Scene06 | 13 | Van | 185 (588) | 185 (588) | 54 | 239 | moving |

M0 규칙으로 고른 track은 Scene01 [86, 63, 83, 87], Scene06 [12, 11, 13, 10]이다. 비교 결과는 `outputs/milestone0/reports/*_tracks.csv`에 있다.

**구성 편향:** Scene01에서 고른 track은 대부분 정지 차량이고 카메라는 움직인다. Scene06은 반대로 카메라가 거의 정지해 있고 track은 모두 이동 차량이다. 그래서 연구질문 4(정지 vs 이동)의 효과가 장면 효과와 섞인다(confounded).

## 3. Scene06 정합성 검증

`uv run python scripts/inspect_vkitti.py --scene Scene06`의 6개 검사가 모두 PASS다(Scene01도 다시 실행해 PASS).

| 항목 | Scene01 | Scene06 |
|---|---|---|
| 프레임 (RGB = mask = extrinsic) | 447 (0–446) | 270 (0–269) |
| info.txt track / mask에 없는 ID | 93 / 0 | 15 / 0 |
| bbox `number_pixels` vs mask 불일치 | 0 | 0 |
| extrinsic rigid·연속성 | 정상 | 정상 |
| stereo baseline 오차 | 1.0e-4 m | 1.0e-4 m |
| 카메라 이동 총거리 / 프레임당 최대 | 332.5 m / 1.25 m | 51.9 m / 1.02 m |

Scene06의 카메라는 대부분 정지해 있다. VKITTI1 논문(Gaidon et al. 2016, §3.2)도 원본 KITTI 0006을 'stationary camera at a busy intersection'으로 설명한다. 궤적 그림은 `outputs/milestone0/trajectories/Scene06_clone_Camera_0_trajectory_*.png`에 있다.

## 4. FPS 근거

| 출처 | 내용 | 판정 |
|---|---|---|
| VKITTI2 공식 페이지 | 촬영 FPS·timestamp 언급 없음 | 근거 없음 |
| VKITTI2 논문 (arXiv 2001.10773) | 언급 없음 | 근거 없음 |
| VKITTI1 논문 (CVPR 2016) | 가상 카메라 경로가 KITTI MOT 원본 시퀀스를 따른다(§3.2). 언급된 "5-8 FPS"는 렌더링 속도다 | 간접 근거 |
| KITTI 논문 (Geiger et al., IJRR 2013) | 카메라가 10 Hz로 회전하는 Velodyne에 맞춰 트리거된다(Synchronization 절) | KITTI 원본에 대한 근거 |

**결론:** VKITTI2 자체에서 촬영 FPS를 공식적으로 밝힌 근거는 찾지 못했다. 10 Hz는 KITTI를 거친 추정일 뿐이므로 확정값으로 쓰지 않는다. M1의 Δt와 시간 구간은 **원본 frame 번호 차이**만 사용한다. 실제 경과시간이 필요해지면 그때 근거를 추가하고 따로 표기한다.

## 5. MemorySelector (`src/memsel/selector.py`)

SAM2와 독립된 모듈이다.

```python
bank = MemoryBank(prompt=Memory(f0, pos, rot), L=32)    # 최초 prompt는 별도로 고정
sel = MemorySelector("pose+dt", scale=PoseScale(s_t, s_r_deg, w_t=0.5), K=7)
s = sel.select(current=Memory(t, pos_t, rot_t), bank=bank)   # s.active (≤7), s.chosen (≤5)
bank.add(Memory(t, pos_t, rot_t))   # 프레임 t를 예측하고 메모리를 인코딩한 뒤에만 호출
```

- `Memory`는 `original_frame_id`, `camera_position`, `camera_rotation` 세 필드만 가진다(frozen, slots). GT mask나 대상 차량의 GT pose를 넣을 자리가 아예 없고, 모듈은 pose.txt·mask·bbox를 읽지 않는다.
- `MemoryBank.add()`는 직전보다 큰 frame ID만 받는다. `select()`는 bank 안에 현재 프레임 이상의 ID가 있으면 오류를 낸다. 따라서 현재 프레임이나 미래 프레임은 후보가 될 수 없다. 후보 큐가 L개를 넘으면 가장 오래된 것부터 버린다(FIFO).
- active set은 prompt + 가장 최근 수신 메모리 + 정책 선택 최대 5개다. bank가 비어 있으면 최근 메모리가 곧 prompt이므로 한 번만 센다. 후보가 부족하면 있는 것만 쓰고, 가짜나 중복 메모리는 만들지 않는다. 결과는 원본 시간순으로 정렬한다.
- 정책은 최근 메모리를 제외한 후보에서 고른다.
  - **recent-K:** 가장 최근 5개.
  - **time-distributed:** age 구간 [1–2], [3–5], [6–10], [11–20], [21+]에서 하나씩 고른다. 닫힌 구간은 구간 중앙에, 21+ 구간은 해당 구간 후보 age의 중앙값에 가장 가까운 후보를 고른다. 빈 구간은 이미 active인 메모리(prompt·recent·이미 고른 것)와의 최소 시간 간격이 가장 큰 후보로 채운다.
  - **pose-only:** 자세 비용 C가 가장 작은 5개.
  - **pose+Δt:** 구간마다 C가 가장 작은 후보 1개. 빈 구간은 아직 고르지 않은 후보 중 C가 작은 순서로 채운다.
  - 모든 동률은 더 최근 후보를 고른다.
- 자세 비용: C = w_T·clip(dT/sT, 0, 1) + (1−w_T)·clip(dR/sR, 0, 1). dT는 두 카메라 중심 사이의 거리, dR은 두 회전 사이의 geodesic 각도다.
  - **sT, sR:** 개발 장면 01·06의 모든 (현재 프레임, 직전 32프레임) 쌍에서 구한 95백분위다. sT = 28.99 m, sR = 33.31°이며 `estimate_pose_scale`로 계산한다.
  - **하한값:** sT ≥ 0.01 m, sR ≥ 0.01°(계획서 §13).
  - **w_T = 0.5:** M6에서 정할 때까지 임시로 쓰는 값이다.
- **계획서에 명시되지 않아 확인이 필요한 구현 선택:**
  - (a) time-distributed의 빈 구간을 채울 때 기준 시각에 prompt와 recent를 포함했다.
  - (b) sT·sR을 장면별이 아니라 개발 장면 전체를 합쳐 하나로 정했다. 장면별 값은 Scene01이 31.1 m / 28.5°, Scene06이 16.1 m / 38.8°이며 보고서에 함께 기록했다.

## 6. 분석 방법 (`scripts/analyze_memory_selection.py`)

- 선택된 track마다 prompt frame+1부터 마지막 관측 프레임까지 프레임을 하나씩 흘려보낸다. 프레임마다 네 정책이 선택한 뒤에 그 프레임을 bank에 넣는다. 정책별 예측이 다르면 메모리 내용이 달라질 수 있지만, 이 단계는 frame ID와 pose 선택만 보므로 bank를 정책끼리 공유해도 결과가 같다.
- **overlap** = |A∩B| / max(|A|,|B|)이며, 정책이 고른 5개 자리만 비교한다(prompt·recent는 모든 정책에서 같다). 장면마다 전체 프레임과 **full bank**(후보 32개가 모두 찬 상태) 프레임을 따로 집계하고, 본문에는 full bank 결과를 쓴다. 분석 대상은 Scene01 343개 프레임(full bank 215), Scene06 314개 프레임(full bank 186)이다.
- 현재 프레임은 카메라 상태로 나눈다. 직전 대비 이동 < 0.05 m이면 stopped, 회전 > 0.5°이면 turning, 나머지는 straight다. 이 구분은 설명용이며 어떤 정책에도 쓰이지 않는다.
- **시간 분산 효과 통제 (비교 B):** time-distributed와 pose+Δt는 둘 다 구간마다 1개씩 고른다. 따라서 같은 구간 안에서 두 선택이 다르다면 원인은 자세 비용뿐이다. 이를 확인하려고 구간마다 세 값을 계산했다.
  - 두 정책의 일치율
  - 우연 일치율 E[1/후보 수]
  - **pose+Δt가 그 구간에서 가장 최근 후보를 고른 비율**. 이 값이 높으면 자세 정보가 사실상 '구간 안의 최근 프레임'이라는 시간 규칙과 같다는 뜻이다.

## 7. 결과

### 7.1 정책 간 overlap (full bank)

| | Scene01 | Scene06 |
|---|---|---|
| **A. recent-K vs pose-only** | **0.977** (같은 집합 93.5%) | **0.737** (같은 집합 41.4%) |
| **B. time-distributed vs pose+Δt** | **0.204** (같은 집합 0%) | **0.282** (같은 집합 0%) |
| recent-K vs time-distributed | 0.400 | 0.400 |
| recent-K vs pose+Δt | 0.592 | 0.520 |
| pose-only vs pose+Δt | 0.600 | 0.613 |
| 후보 age와 자세 비용의 Spearman 상관 (프레임 평균) | 0.989 | 0.865 |

### 7.2 카메라 상태별 A·B overlap (full bank)

| 장면 / 상태 | 프레임 | A (recent vs pose) | B (td vs pose+Δt) | 후보 pool 안의 최대 dT / dR (중앙값) |
|---|---|---|---|---|
| Scene01 straight | 98 | **1.000** | 0.200 | 23.8 m / 3.2° |
| Scene01 turning | 90 | **1.000** | 0.200 | 8.7 m / 26.2° |
| Scene01 stopped | 27 | 0.815 | 0.230 | 2.9 m / 1.9° |
| Scene06 straight | 26 | **1.000** | 0.208 | 2.4 m / 1.7° |
| Scene06 turning | 30 | **1.000** | 0.200 | 14.1 m / 19.3° |
| Scene06 stopped | 130 | 0.623 | 0.315 | **0.015 m / 0.03°** |

pose-only가 recent-K와 다르게 고른 프레임은 Scene01 14개, Scene06 109개이고 **모두 stopped 상태**다. 이때 pose-only가 추가로 고른 메모리는 현재 자세와 거리 차이 중앙값이 Scene01 3.8 cm, Scene06 0.5 cm(최대 11 cm)이고, 회전 차이는 최대 0.17°다.

### 7.3 구간별 time-distributed vs pose+Δt (full bank)

| 장면 | 구간 | 후보 수 | 일치율 | 우연 일치 | **pose+Δt = 구간 내 최신** | dT 평균 td → pose+Δt |
|---|---|---|---|---|---|---|
| Scene01 | 3–5 | 3 | 0.000 | 0.333 | **0.991** | 1.73 → 1.30 m |
| Scene01 | 6–10 | 5 | 0.005 | 0.200 | **0.958** | 3.46 → 2.59 m |
| Scene01 | 11–20 | 10 | 0.014 | 0.100 | **0.940** | 6.56 → 4.77 m |
| Scene01 | 21+ | 12 | 0.000 | 0.083 | **0.986** | 11.88 → 9.38 m |
| Scene06 | 3–5 | 3 | 0.199 | 0.333 | 0.683 | 0.56 → 0.42 m |
| Scene06 | 6–10 | 5 | 0.097 | 0.200 | 0.602 | 1.05 → 0.81 m |
| Scene06 | 11–20 | 10 | 0.048 | 0.100 | 0.532 | 1.75 → 1.37 m |
| Scene06 | 21+ | 12 | 0.065 | 0.083 | 0.435 | 2.47 → 2.19 m |

[1–2] 구간은 최근 메모리를 빼면 후보가 frame t−2 하나뿐이어서, 두 정책이 항상 같은 프레임을 고른다. 그래서 B의 overlap 0.20 중 0.20 전체가 이 구간에서 나온다.

### 7.4 시간 구간 후보 수와 빈 구간

- full bank에서는 프레임 누락이 없으므로 구간마다 후보 수가 1/3/5/10/12개로 고정되고, 빈 구간은 0%다.
- 빈 구간은 bank가 아직 덜 찬 초반에만 생긴다. 전체 프레임 기준 빈 구간 비율은 다음과 같다.
  - Scene01: 2.3% / 3.5% / 7.0% / 12.8% / 24.5%
  - Scene06: 2.5% / 3.8% / 7.6% / 14.0% / 26.8%
- 고른 메모리의 age 중앙값(전체 프레임 기준): recent-K 4, pose-only 4–5, pose+Δt 6, time-distributed 8.

## 8. 그래프와 파일 (`outputs/milestone1/`, git 제외)

| 그래프 | 경로 |
|---|---|
| 카메라 궤적 + 선택된 메모리 | `figures/Scene01_trajectory_selected_frame00416.png`, `figures/Scene06_trajectory_selected_frame00159.png` |
| 현재 frame vs 선택된 memory frame | `figures/Scene01_track090_current_vs_selected.png`, `figures/Scene06_track012_current_vs_selected.png` |
| 정책 overlap matrix | `figures/policy_overlap_matrix.png` |
| memory age 분포 | `figures/memory_age_distribution.png` |
| 선택된 memory의 자세 차이 (dT, dR, C) | `figures/selected_memory_pose_difference.png` |
| 장면별 정책 비교 | `figures/scene_policy_comparison.png` |
| 현재 RGB + 정책별 선택 RGB | `figures/Scene01_rgb_current_vs_selected_frame00416.png`, `figures/Scene06_rgb_current_vs_selected_frame00159.png` |
| 수치 | `reports/m1_report.json`, `m1_scene_summary.csv`, `m1_within_bin_td_vs_posedt.csv`, `m1_picks.csv`, `m1_frames.csv` |

예시 프레임은 full bank 프레임 중 recent-K와 pose-only의 overlap이 가장 낮은 프레임을 정해진 규칙으로 골랐다.
- **Scene01 frame 416 (track 86):** 카메라는 교차로에 정지해 있고 대상 차량이 진입하는 장면이다. pose-only는 6–10프레임 전 메모리를 고르는데, 배경은 거의 같지만 대상 차량의 위치가 달라져 있다.

## 9. 테스트

`uv run pytest`: **53 passed**.
- `tests/test_memory_selector.py` (21개): 과거 프레임만 선택, K·L 제한, prompt 고정, 최근 메모리 고정, 중복 제거, 초기 프레임, 빈 구간 채우기(두 정책), 구간 중앙·중앙값 동률, pose 동률 시 recent-K와 일치, 결정론성, 자세 거리·각도·비용 계산과 clip·하한값, 개발 데이터 scale 추정, GT 누수 방지(필드 구성, 속성 추가 불가, 소스에 pose.txt·mask·bbox 참조 없음), 잘못된 설정 처리.
- `tests/test_track_selection.py`: 새 규칙(prompt 정의, prompt 이후 관측 수, 면적 기준을 한 번도 넘지 못한 track)과 M0 규칙 보존, 정렬과 결정론성.
- 의도적으로 버그를 넣어 테스트가 잡아내는지 확인했다. 동률 방향을 반대로 하면 1개, 최근 메모리를 후보에서 빼지 않으면 6개가 실패한다.
- 분석 스크립트는 누락이 없을 때 후보 pool이 연속적이라는 가정을 assert로 확인한다.

## 10. 연구 판단

1. **카메라 자세를 고려한 선택이 최근 프레임 선택과 실제로 다른가?** 거의 같다.
   - 카메라가 움직이는 프레임(straight·turning)에서는 두 장면 모두 pose-only가 recent-K와 **100% 같다**.
   - 차이는 카메라가 정지한 프레임에서만 생기고, 그때의 자세 차이는 cm 이하·0.2° 이하의 수치 잡음 수준이다.
   - Scene01의 overlap 97.7%는 '연구 과정' 탭의 B 판정선(약 95% 이상)을 넘는다.
2. **시간 분산을 통제하면 자세 정보가 선택에 영향을 주는가?**
   - 겉으로는 영향이 커 보인다. 구간 안에서 time-distributed와의 일치율이 우연 수준보다 낮다.
   - 하지만 Scene01에서는 pose+Δt가 구간 안의 **가장 최근 후보**를 고른 비율이 94–99%다. 즉 '구간 중앙 대신 구간 안의 최신 프레임'이라는 **시간 규칙**과 거의 같다.
   - Scene06에서는 이 비율이 44–68%이지만, 차이가 나는 경우는 대부분 카메라가 정지해 자세 차이가 의미 없는 구간이다.
   - **결론:** 현재 개발 데이터에서는 자세 정보가 시간 정보와 독립적으로 영향을 준다는 증거가 약하다.
3. **Scene01과 Scene06의 차이:**
   - Scene01은 카메라가 계속 이동하며 정지 차량을 지나간다. 그래서 자세 비용이 시간과 거의 단조 관계다(Spearman 0.99).
   - Scene06은 카메라가 대부분 정지해 있고(stopped 130/186프레임) 대상은 모두 이동 차량이다. 자세 비용이 거의 0이라 정책 간 차이가 잡음으로 생긴다.
   - 두 장면 모두 같은 장소를 다시 지나가거나(loop·재방문) 크게 되돌아오는 구간이 없다.
4. **카메라 이동 경로 때문에 pose-only와 recent-K가 거의 같아지는가?** 그렇다.
   - 직진이든 회전이든 재방문 없는 단조 경로에서는 시간적으로 가까운 프레임이 자세로도 가장 가깝다.
   - 회전 구간(Scene01 turning, pool 안 최대 회전 26°)에서도 overlap은 1.000이었다.
   - 정지 구간에서는 자세가 같아져 동률 처리 규칙이나 잡음이 선택을 결정한다.
5. **이후 SAM2 실험에 줄 영향 (예상):**
   - pose-only와 recent-K는 SAM2에 거의 같은 메모리를 넣으므로 **성능 차이가 작을 것으로 예상**한다.
   - pose+Δt와 time-distributed의 차이는 대부분 '구간 안에서 최신이냐 중앙이냐'라는 시간 배치의 차이로 해석해야 한다. pose+Δt가 이기더라도 이를 '자세 정보의 효과'로 주장하기 어렵다.
   - 프레임 누락(M3)이 생기면 각 구간의 후보가 줄어 정책 간 차이가 더 작아질 수 있다.
   - 정지 카메라 + 이동 대상(Scene06)에서는 카메라 자세가 대상의 외형 변화를 설명하지 못한다(Scene01 frame 416 예시 참고).
   - 이것은 선택 결과에 대한 분석일 뿐이며, 분할 정확도에 대한 결론이 아니다.

## 11. 연구 설계상 발견한 문제

1. **자세와 시간이 거의 같은 정보다:** VKITTI 개발 장면에는 재방문 경로가 없다. 그래서 핵심 가설('오래되었지만 자세가 비슷한 메모리가 유리하다')을 이 데이터에서 직접 검증하기 어렵다. 공개된 VKITTI1 설명으로 보면 시험 장면(0002 도심→교차로, 0018 숲길, 0020 고속도로)도 재방문 경로가 아닐 가능성이 크다. 다만 시험 데이터는 열어 보지 않았다.
2. **정지 카메라에서의 동률 처리:** 자세 차이가 수 mm 수준일 때도 비용 순서가 선택을 정한다. 'pose 차이가 이 값보다 작으면 동률로 본다'는 허용오차 ε를 정할지 결정해야 한다. ε를 두면 정지 구간의 pose-only는 recent-K와 같아진다.
3. **비교 B에 대조군이 없다:** time-distributed와 pose+Δt의 차이에는 '구간 안의 최신 후보 선택'이라는 시간 효과가 섞여 있다. 자세 정보만의 효과를 분리하려면 `recent-per-bin`(구간마다 최신 1개) 같은 순수 시간 대조군이 필요하다. 정책을 추가하는 일이므로 멘토 확인이 필요하다.
4. **정지/이동 track 구성이 장면과 섞인다:** Scene01은 정지 차량 위주, Scene06은 이동 차량뿐이다.
5. **정규화 기준:** sT = 29 m는 L=32 후보 창 전체에 맞춘 값이라, 직전 몇 프레임 사이의 비용 차이는 매우 작다. 단조 경로에서는 순서가 바뀌지 않으므로 결론에는 영향이 없지만, M6에서 sT·sR 정의를 확정할 때 고려해야 한다.

## 12. M2 진행 판단

'연구 과정' 탭의 기준으로 보면 **현재 개발 데이터는 B 경로**에 해당한다: Scene01에서 recent-K와 pose-only의 overlap이 97.7%이고, 이동 중인 프레임에서는 100%다.

권장:
- **M2(순정 SAM 2.1 baseline)는 진행한다.** M2–M4(baseline 재현, 프레임 누락 생성, 누락에 따른 성능 저하 확인)는 메모리 선택 정책과 무관하게 필요하다. 이 단계에서 '프레임 누락이 실제로 SAM2를 어렵게 하는가'를 먼저 확인할 수 있다.
- **M5 이후로 넘어가기 전에 멘토와 다음을 결정한다.**
  - 위 11장 2·3번 항목의 처리 방법
  - 재방문·loop 경로가 있는 데이터를 확보할지(VKITTI 15/30-deg 변형은 회전된 카메라일 뿐 재방문 경로는 아니다. 시뮬레이터 여부는 M10의 판단으로 미룬다)
  - 연구 질문을 '시간 배치 효과' 중심으로 다시 쓸지
