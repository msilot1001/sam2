# Milestone 3 — 결정론적 프레임 누락 생성과 검증

이번 단계의 범위:
- 누락 패턴 생성기, manifest, 누락 clip 구성, 대응 평가(paired evaluation), burst 회복 사건 추출을 만들었다.
- SAM2로는 짧은 smoke test(clip 3개)만 실행했다. **4조건 × 3 seed 전체 실험(M4)은 실행하지 않았다.**
- 공식 SAM 2.1 코드와 가중치, 추론 동작은 그대로다. 핵심 파일 sha256과 체크포인트 sha256이 M2와 같다.
- M1의 confound(이동 중 카메라에서 recent-K = pose-only)는 여전히 남아 있다.

## 1. 누락 조건과 생성 규칙 (`src/baseline/frame_drops.py`)

| 조건 | 종류 | 요청 누락률 | 누락 수 결정 방식 |
|---|---|---|---|
| `random_30` | random | 0.30 | round_half_up(0.30 · n) 프레임 |
| `random_50` | random | 0.50 | round_half_up(0.50 · n) 프레임 |
| `burst_30` | burst (길이 4) | 0.30 | k = round_half_up(0.30 · n / 4)개 burst |
| `burst_50` | burst (길이 4) | 0.50 | k = round_half_up(0.50 · n / 4)개 burst |

- **모집단:** prompt 이후 프레임 n개(= clip 프레임 수 − 1). prompt 프레임은 절대 누락하지 않는다. `actual_drop_rate` = 누락 수 / n.
- **반올림:** `round_half_up(x) = floor(x + 0.5)`. Python 기본 `round`의 은행가 반올림은 쓰지 않는다.
- **burst 규칙:**
  - 원본 frame 번호 기준으로 정확히 연속된 4프레임이다.
  - burst와 burst 사이에는 수신 프레임이 1개 이상 있어서(`min_gap_between_bursts = 1`) 두 burst가 합쳐질 수 없다.
  - 부분 burst는 만들지 않는다. 요청한 k개가 들어갈 자리가 없으면 들어가는 최대 개수로 줄이고 `request_fits = false`로 기록한다.
  - 배치는 가능한 모든 배치 중 균등 추출이다(stars and bars: slack + k칸 중 k칸 선택, start_i = c_i + i·(L + gap − 1)).
  - clip 끝에 걸치는 burst도 허용한다. 이 경우 그 사건은 관찰 구간이 짧거나 없어서 censoring된다.
- **난수:**
  - 난수 라이브러리를 쓰지 않는다. 후보를 `sha256("<rng_key>|<i>")` 순서로 정렬해 앞에서부터 k개를 고른다. 머신, Python, numpy 버전과 무관하게 같은 결과가 나오며, 균등 무작위 부분집합에 해당한다.
  - `rng_key = "m3|<scene>|<variation>|cam<c>|track<t>|<condition>|seed<s>"`. Python `hash()`는 쓰지 않는다.
  - GT mask, 면적, 이동 여부, 가시성은 입력으로 쓰지 않는다.
- **seed:** 계획서에는 '세 seed'만 있고 값이 없어서 **0, 1, 2**로 정했다(`configs/m3_frame_drop.yaml`).
  - 조건마다 독립적으로 생성하므로 `random_30`이 `random_50`의 부분집합인 것은 아니다.
  - 정책 비교(M4/M5)에서는 네 정책이 같은 manifest를 공유한다(계획서 §8).

## 2. 실데이터 manifest 생성 결과 (개발 track 8개 × 4조건 × 3 seed = 96개, 모두 검증 통과)

| 조건 | 실제 누락률 (최소 / 평균 / 최대) | burst 수 범위 | 요청이 모두 들어감 |
|---|---|---|---|
| random_30 | 0.296 / 0.303 / 0.309 | – | 예 |
| random_50 | 0.500 / 0.502 / 0.509 | – | 예 |
| burst_30 | 0.278 / 0.291 / 0.305 | 4–10 | 예 |
| burst_50 | 0.495 / 0.507 / 0.519 | 7–17 | 예 |

- track별 prompt 이후 프레임 수 n: 105, 86, 80, 72 (Scene01) / 135, 70, 55, 54 (Scene06). 각 track의 clip은 M2의 no-drop clip과 같다.
- **재현성:** `generate`를 다시 실행해도 96개 manifest의 CSV·JSON이 바이트 단위로 같다. 저장된 manifest를 불러올 때마다 생성기로 다시 만들어 비교한다.
- **burst 관찰 구간** (burst 뒤 다음 burst나 clip 끝까지의 수신 프레임 수):

| 조건 | burst 사건 | 평균 구간 길이 | 구간 < 2 (회복 판정 불가) | 구간 ≥ 5 (5프레임 이내 회복 판정 가능) |
|---|---|---|---|---|
| burst_30 | 144 | 8.6 | 19 | 90 |
| burst_50 | 249 | 3.7 | **69** | **76** |

## 3. Manifest 형식 (M4 실험의 기준 데이터)

경로: `outputs/milestone3/manifests/<scene>_trackNNN/<condition>_seed<s>.{json,csv}`

- **CSV (원본 프레임마다 1행):** `original_frame_id, is_prompt, is_received, is_dropped, received_video_idx`. 누락 프레임의 `received_video_idx`는 빈 값이다.
- **JSON:**
  - 식별 정보: `schema_version, scene, variation, camera_id, track_id, condition, kind, seed, rng_key`
  - 범위: `prompt_original_frame_id, original_start_frame, original_end_frame`
  - 누락률: `requested_drop_rate, actual_drop_rate, rate_denominator, rounding`
  - burst: `burst_length, min_gap_between_bursts, requested_number_of_bursts, number_of_bursts, request_fits, bursts[{start,end,length}]`
  - 누락 목록: `dropped_original_frame_ids`
  - 개수: `total_original_frames, post_prompt_frames, total_received_frames, total_dropped_frames`
  - 무결성: `frames_csv, frames_csv_sha256`
- **불러올 때 검사하는 것:** CSV sha256, 구조 조건(아래), 생성기 재실행 결과와의 일치.
- **덮어쓰기 금지:** 같은 이름의 manifest가 이미 있을 때, 내용이 같으면 아무것도 하지 않고 다르면 `ManifestError`를 낸다.
- **구조 조건:**
  - 원본 번호가 빠짐없이 연속하고 오름차순이다.
  - prompt는 첫 행 하나뿐이며 수신된다.
  - 각 프레임은 수신·누락 중 정확히 하나다.
  - 수신 프레임의 video_idx는 0..R−1로 연속이다.
  - 개수와 누락률이 metadata와 일치한다.
  - burst 조건이면 누락 프레임 = burst들의 합집합이고, 각 burst는 길이 4이며 burst 사이 간격 ≥ 1이다.

## 4. 프레임 인덱스 대응과 clip 구성 (`src/baseline/sam2_vos.py`)

- `clip_from_pattern(pattern)`은 **수신 프레임만** 담은 `Clip`을 만든다. 누락 프레임 자리를 검은 화면, 복제, 보간으로 채우지 않는다.
  - 예: 원본 [130, 131, 132, 133, 134, 135] → 수신 [130, 131, 134, 135] → `00000.jpg → rgb_00130.jpg`, `00001 → 00131`, `00002 → 00134`, `00003 → 00135`.
- `Clip`의 변경은 최소한으로 했다.
  - `variant` 필드 추가: 예 `burst_30_seed0`. 비어 있으면 M2와 같다.
  - 원본 번호가 비어 있지 않고 엄격히 증가하는지 검사한다.
  - 폴더 경로는 `clips/<variant>/<scene>_trackNNN/`이다. variant가 없으면 M2 경로와 같으므로 M2 산출물을 덮어쓰지 않는다.
- `write_clip_folder`의 처리:
  - 이전 실행에서 남은 jpg(깨진 symlink 포함)를 지운다.
  - symlink를 만든 뒤 `verify_clip_folder`로 확인한다: 파일 이름이 0..N−1인지, 각 파일이 대응하는 원본 RGB를 가리키는지, mapping.csv가 일치하는지.
- M2 runner와의 호환을 확인했다. 평가 함수를 `src/baseline/evaluation.py`로 옮긴 뒤 저장된 M2 예측을 다시 평가했더니 `per_frame_metrics.csv`·`per_track_metrics.csv`가 **바이트 단위로 같았다**.

## 5. Paired evaluation (`src/baseline/dropout_eval.py`)

- `paired_frames(no_drop, drop)`는 누락 실행에서 **수신된 non-prompt 원본 frame**만 골라 두 실행을 같은 프레임에서 비교한다.
- 비교 지표는 `J, F, JF, F_strict` 각각에 대해 `*_no_drop_matched`, `*_drop`, `delta_*`다. delta는 누락 − 대응 무손실 값이므로 음수가 성능 저하다.
- 다음 경우에는 오류를 낸다: 무손실 실행에 해당 frame이 없을 때, 두 실행의 prompt가 다를 때, GT가 비어 있는지 여부가 서로 다를 때(다른 track이나 데이터라는 뜻).
- `paired_summary`는 track 단위 평균을 낸다. 이후 통계에서는 track이 단위이고, frame별 기록은 진단용이다.
- `check_m2_reference`는 M2에 저장된 예측을 무손실 기준으로 다시 써도 되는지 확인한다.
  - 체크포인트 sha256, SAM 2 commit, model config, postprocessing 설정이 같은가
  - 평가 설정(boundary_th, strict, lost_j)이 같은가
  - M2 clip의 frame 대응이 manifest의 원본 범위와 같은가
  - prompt frame이 같은가
- 따라서 무손실 추론은 다시 실행하지 않는다.
- GT가 빈 프레임은 M2 규칙(DAVIS)을 그대로 따른다.

## 6. Burst 회복 사건 정의

`recovery_events(frames, J_by_frame)`는 누락 구간(연속으로 누락된 원본 프레임 묶음, burst 조건에서는 burst 하나)마다 사건 1개를 만든다.
- **관찰 구간:** 누락 구간 뒤의 수신 프레임들. 다음 누락 구간 직전이나 clip 끝에서 끝난다.
- **회복 판정(계획서 §10.3):** J ≥ 0.5가 연속된 두 수신 프레임에서 처음 확인된 시점이다. 회복이 시작된 프레임 k(1부터 셈)와 확인된 프레임 k+1을 모두 기록한다.
- **시간 좌표:** 수신 프레임 수와 원본 frame 차이(`recovery_onset_original_gap`)만 쓴다. FPS 근거가 없으므로 초 단위로 바꾸지 않는다.

| class | 의미 |
|---|---|
| `intact_at_first_frame` | 누락 직후 1·2번째 수신 프레임이 모두 J ≥ 0.5. **한 번도 부정확해지지 않았으므로 '회복'으로 세지 않는다** |
| `recovered_later` | 회복 판정 쌍이 2번째 이후 수신 프레임에서 시작 |
| `unrecovered_next_burst` | 다음 누락 구간 전까지 회복하지 못함 (다음 누락으로 censoring) |
| `unrecovered_end` | clip이 끝날 때까지 회복하지 못함 (clip 끝으로 censoring) |
| `insufficient_observation` | 관찰 구간이 1프레임뿐이라 2프레임 규칙을 적용할 수 없음 (censoring) |
| `no_observation` | 누락 구간이 clip 끝까지 이어짐 |

- 각 사건의 보조 표시:
  - `degraded_after_drop`: 첫 수신 프레임의 J < 0.5
  - `lost_after_drop`: 첫 수신 프레임의 J < 0.1
  - `pre_drop_J`: 누락 직전 J. 누락 전부터 이미 성능이 나빴던 사건을 구분하는 데 쓴다.
  - `within_n_determinable`: 관찰 구간이 5프레임 이상인가
  - `confirmed_within_n`: 2프레임 규칙이 처음 5개 수신 프레임 안에서 확인되었는가
- `recovery_summary`는 두 비율을 따로 보고한다.
  - **누락 직후 성능이 떨어진 사건 기준 회복률:** 분모는 `degraded_after_drop`이면서 관찰 구간이 5 이상인 사건
  - **계획서 문구 그대로의 비율:** 관찰 구간이 5 이상인 모든 사건. 처음부터 정상이던 사건도 포함한다.

## 7. 실데이터 SAM2 smoke test (MPS float32, 공식 SAM 2.1 Tiny, M2와 같은 체크포인트)

| clip | 수신 / 누락 | SAM2 출력 프레임 | 첫 누락 전 prefix가 M2와 같음 | 비교 frame | J 무손실 → 누락 | ΔJ | ΔJ&F | ΔF_strict | burst 사건 class |
|---|---|---|---|---|---|---|---|---|---|
| Scene01 t86 burst_50 s0 | 43 / 44 | 43 | 1/1 (prompt) | 42 | 0.922 → 0.918 | −0.0035 | −0.0038 | −0.0060 | intact 8, insufficient 3 |
| Scene06 t12 random_30 s0 | 95 / 41 | 95 | 1/1 (prompt) | 94 | 0.862 → 0.862 | −0.0006 | −0.0003 | −0.0024 | (누락 구간 32개) intact 19, insufficient 12, no_obs 1 |
| Scene01 t86 burst_30 s0 | 63 / 24 | 63 | **6/6** | 62 | 0.931 → 0.930 | −0.0007 | −0.0004 | −0.0004 | intact 6 |

확인한 것:
- SAM2 출력 index가 수신 video_idx 0..R−1과 정확히 같다.
- 누락 프레임 이미지는 clip 폴더에 없다.
- prompt mask는 M2와 같은 GT이고, prompt frame의 예측도 M2와 같다.
- 출력 mask는 375×1242이다.
- `run_clip`은 데이터셋을 받지 않으므로 GT가 추론에 들어가지 않는다.
- **인과성:** 첫 누락 전까지는 입력이 M2와 같으므로 예측도 같아야 한다. burst_30에서 6프레임 모두 M2와 픽셀 단위로 같았다. 모델이 미래 프레임을 쓴다면 이 prefix가 달라졌을 것이다.

관찰 (clip 3개, 결론 아님):
- 세 clip 모두 성능 저하가 매우 작다(ΔJ ≥ −0.004). 누락 직후 J < 0.5인 사건은 하나도 없었다.
- Scene01 t86 burst_50은 frame 377–380을 누락했다. 그런데 frame 380은 M2에서 유일하게 추적을 놓친 완전 가림 frame이다. 누락이 오히려 그 실패 frame을 건너뛰게 만든 사례다.

## 8. 테스트

`uv run pytest`: **158개 통과** (M3 관련 92개, 기존 M0–M2 66개 포함).
- **생성:** prompt 보존, 길이 1–135의 clip에서 구조 조건, 반올림 경계(25.8 → 26, 43.5 → 44, burst 6.45 → 6, 10.75 → 11), 결정론성, seed·track이 바뀌면 패턴도 바뀜, 고정된 burst 위치 회귀 테스트, burst가 정확히 길이 4이고 서로 분리됨(20 seed), 들어갈 자리가 부족한 경우와 burst 사이 간격이 정확히 1인 경계.
- **manifest:** 저장·로드가 손실 없음, 같은 내용 재저장은 아무것도 안 함, 다른 패턴은 덮어쓰기 거부, CSV를 고치면 sha256 검사로 거부, 잘못된 패턴 검출, 저장된 실제 manifest 96개가 모두 재생성 결과와 일치.
- **clip:** 수신 프레임만 담김, symlink가 올바른 원본을 가리킴, 누락 이미지가 없음, 이전 실행의 파일이 남지 않음, M2 경로 유지, 잘못된 순서 거부.
- **paired:** 같은 frame id로 비교, prompt 제외, delta 부호, 오류 조건, GT가 빈 frame의 M2 규칙.
- **회복:** 6개 class, censoring(다음 누락 / clip 끝), 관찰 구간 0·1, 5프레임 이내 판정 불가, 누락 전 성능 저하 표시.
- **기존 테스트:** track 선택(Scene01 [90, 86, 63, 54], Scene06 [12, 9, 11, 13])은 생성 스크립트가 직접 확인하고, 바뀌면 중단한다. 공식 SAM2 파일 sha256, M0–M2 테스트도 통과한다.

## 9. 그래프 (`outputs/milestone3/`, git 제외)

| 내용 | 파일 |
|---|---|
| track별 원본 frame 타임라인 (random vs burst, seed 3개) | `figures/<scene>_trackNNN_timeline.png` (8개) |
| 요청 vs 실제 누락률 | `figures/requested_vs_actual_rate.png` |
| 수신 video index vs 원본 frame | `figures/Scene01_track086_index_map.png`, `figures/Scene06_track012_index_map.png` |
| burst 경계와 관찰 구간 | `figures/Scene01_track086_burst_windows.png`, `figures/Scene06_track012_burst_windows.png` |
| smoke: 누락 실행 J + 대응 무손실 J + 관찰 구간 | `smoke/figures/*_recovery.png` |

보고서 파일:
- `reports/manifest_summary.csv`, `generate_report.json`, `burst_events_structure.csv`
- `smoke/reports/smoke_report.json`, `*_paired.csv`, `*_recovery_events.csv`, `*_per_frame.csv`

## 10. 계획서의 모호한 점과 이번 결정 (멘토 확인 필요)

1. **seed 값:** 계획서에 값이 없어 0, 1, 2로 정했다.
2. **burst 누락률:** 계획서는 '전체 누락률을 30%·50%로 맞춘다'고 했다. 하지만 길이 4의 burst만으로는 정확히 맞출 수 없어 반올림한 burst 수를 썼다. 실제 누락률은 27.8–30.5%, 49.5–51.9%다.
3. **burst 사이 최소 간격:** 1 수신 프레임으로 정했다. 이 때문에 burst_50에서는 burst 사이 수신 프레임이 평균 3.7개뿐이다.
4. **5프레임 이내 회복률 정의:** 2프레임 규칙이 5개 수신 프레임 안에서 확인된 경우로 정했다(k+1 ≤ 5). 분모는 계획서 §10.4대로 '관찰 구간이 확보된(≥ 5) 사건'이다. 다만 처음부터 정상이던 사건을 회복으로 셀지는 계획서에 정해져 있지 않아, 두 비율을 모두 보고한다.
5. **설계상 긴장 관계:** burst_50은 burst 사건의 28%(69/249)가 관찰 구간 1프레임 이하라 회복 판정 자체가 불가능하다. 5프레임 이내 판정이 가능한 사건도 31%(76/249)뿐이다. 회복 지표의 통계 검정력은 burst_30 쪽에 크게 의존할 것이다. 회복 분석용으로 burst 간 최소 간격을 따로 두는 방안을 멘토와 검토해야 한다(지금은 변경하지 않았다).
6. **random 조건의 회복 사건:** random 조건도 연속 누락 구간마다 사건을 만들 수 있게 해 두었다. 계획서의 회복 지표는 burst 기준이므로, M4의 주 회복 분석은 burst 조건만 쓰는 것을 제안한다.
7. **GT가 빈 프레임의 회복 판정:** M2와 같은 DAVIS 규칙(둘 다 비면 J = 1)을 썼다. 따라서 완전히 가려진 frame에서 빈 예측을 내면 정상 상태로 본다.

## 11. M4와의 호환

- M4 runner의 흐름: `fd.load(manifest)` → `clip_from_pattern` → `write_clip_folder` → `run_clip` → `evaluate` → `check_m2_reference` + `paired_frames` → `recovery_events`. 모두 이번에 구현했고 smoke test로 끝까지 검증했다.
- 정책 비교(M5)에서는 같은 manifest를 네 정책이 그대로 공유하면 된다.
- 연산량 추정(MPS, 약 1 FPS): 수신 프레임 기준으로 4조건 × 3 seed × 8 track이면 4,821 frame, 즉 약 1.3시간이다(정책 1개 기준, prompt frame 포함).
