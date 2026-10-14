# M7 실행 환경 이전: RTX 2080 (Ubuntu) → RTX 5090 (Vast.ai)

**작성:** 2026-10-14 12:2x KST
**코드:** `exp/m7-algorithm-research` @ `4c63eb1` 이후. 두 서버 모두 같은 커밋으로 동기화했다.

## 1. 역할 분담 (결정)

| 항목 | 위치 | 이유 |
|---|---|---|
| **Git 원본 (코드, 문서, 작은 결과 CSV/JSON, 실행 장부)** | GitHub `origin/exp/m7-algorithm-research`, Ubuntu `/home/sojak/sam2`에서 커밋 | Vast 저장소는 영구 volume이 아니다(`workspace_is_volume = false`). 인스턴스를 삭제하거나 recycle하면 모두 사라진다 |
| Stage 0 기록 원본 (`outputs/milestone7/rec`, 2.3 GB, 64쌍)과 CPU 재생 분석 | Ubuntu (영구 보관) | GPU가 필요 없다. 이전하지 않는다 |
| 이후 GPU 작업 (인과 rollout, T head 학습·평가, 추가 기록, VK) | Vast `/root/sam2-m7-test` | RTX 5090: production 경로 0.14 s/frame. 2080은 0.44 s/frame이었다(약 3.2배) |
| Vast 결과 | 작업이 끝날 때마다 tar로 Ubuntu에 회수한 뒤 커밋(작은 파일만) | 인스턴스 소멸 대비 |
| PUMaVOS C | Ubuntu에만 있다. **Vast에는 전송하지 않았다** | 확인 세트 봉인 |

## 2. 전송한 것과 검증 결과

**전송 방식과 속도:**
- 방식: 단일 `tar` stream over SSH. 수천 개 작은 파일을 rsync로 보내는 방식을 피했다.
- 실측 속도: 4.2 MB/s. 다른 SSH 작업과 병행하면 약 2.7 MB/s로 떨어진다.

| 데이터 | 크기 / 파일 수 | 검증 |
|---|---|---|
| DAVIS 2017 trainval 480p | 1.6 GB | 사용자가 Vast에서 직접 내려받았다(기존 상태) |
| SegTrack v2 `clips` | 250 MB / 1,104 | md5 1,104개 일치 |
| PUMaVOS `clips` 중 **A 9개 + B 6개 영상만** (M5.6 manifest 분할 그대로) | 1.9 GB / 4,530 | md5 4,530개 일치. clip이 raw를 가리키는 symlink라 `tar -h`로 실제 파일을 보냈다. C 영상 9개가 Vast에 없음을 확인 |
| `outputs/milestone55, 56, 6, 61, 5pre` (기준 출력, manifest, M6/M6.1 run) | 약 180 MB | 디렉터리별 파일 수 일치. `--skip-old-files`로 기존 파일을 덮어쓰지 않았다 |
| `outputs/milestone7/tcache` (T head 학습 캐시, RTX 2080에서 생성, 43 REP 객체) | 8.0 GB | 전송 중. 완료 후 파일 수와 크기를 검증한다 |

**옮기지 않은 것과 이유:**
- **Stage 0 기록 2.3 GB:** CPU 재생은 Ubuntu에서 끝났다. GPU 작업에 필요 없다.
- **PUMaVOS raw와 zip, C clip:** 봉인이 이유다. 확인 실행을 승인하면 그때 해당 영상만 보낸다.
- **VKITTI2 Scene01/06 (137 MB):** 사전 등록상 결선 후보가 고정된 뒤에 쓴다. 필요할 때 30초 이내에 보낼 수 있다.
- **원본 zip과 archive 전부.**

## 3. 소프트웨어 차이와 재현성

| | RTX 2080 (Ubuntu) | RTX 5090 (Vast) |
|---|---|---|
| PyTorch / CUDA | 2.14.1+cu130 | 2.11.0+cu128 |
| SAM 2 소스 | upstream `2b90b9f`, 비편집 설치 | upstream `2b90b9f` (`/root/sam2-smoke-test`, 편집 설치). `.py`/`.yaml` md5 전부 일치 |
| checkpoint `sam2.1_hiera_tiny.pt` | md5 `6aa6761c…` | 동일 |
| `sam2._C` | **없음** → hole filling 건너뜀(경고) | **있음** → hole filling(`fill_hole_area = 8`) 실행 |
| TF32 | 꺼짐 | 꺼짐 |

**발견과 조치:**
- `_C` 유무 때문에 기본 설정의 출력이 달랐다. 전체 길이 3객체 중 kite-surf obj2 1 frame에서 J&F 0.0009 차이가 났고, 저장 logit의 부호가 최대 162 px 뒤집혔다.
- M5–M7의 모든 결과는 hole filling 없이 나왔다. 그래서 `scripts/m55.py:build_predictor`에 `pred.fill_hole_area = 0`을 명시했다(커밋 `4c63eb1`).
- Ubuntu에서는 동작이 바뀌지 않는다(원래 0과 같은 동작). 테스트에 `assert pred.fill_hole_area == 0`을 추가했다.

**재현성 검증 (`scripts/m7_repro.py`, `outputs/milestone7/repro/*/compare.csv`; 기능 확인용이며 연구 결과가 아니다):**

| 검사 | 범위 | 결과 |
|---|---|---|
| M7 테스트 | `tests/test_m7.py` 8개 | 양쪽 모두 8 passed. 기존에 건너뛰던 통합 테스트 2개도 M6/M6.1 기준 출력을 옮겨서 통과했다(첫 8 frame J&F가 소수 10자리까지 동일) |
| Stage 0 기록 경로 (SGGL + 모든 probe), 전체 길이 | lab-coat obj1, kite-surf obj2, miami-surf obj5, dog-gooses obj5, drone obj1 (339 frame) | **이 5객체에서는** J&F 모든 frame 동일, crop 결정과 box 동일(전체 세트는 아래 정정 참조). 저장 logit 차이는 fp16 저장 수준(≤ 0.0625)이고, 부호가 뒤집힌 저해상도 픽셀은 객체당 0–2개로 마스크 영향이 없다 |
| 신규 후보 인과 rollout: M3-vel | lab-coat obj1, kite-surf obj2 | 모든 frame 동일 |
| 신규 후보 인과 rollout: F1-blend.7 (연속 logit 가중합) | 같은 2객체 | 객체당 1 frame 차이, 객체 평균 Δ ≤ 6e-5. 가중합이 0 근처인 픽셀이 커널 수준 수치 차이에 민감하다 |
| 속도 / VRAM | 5객체 | production 0.136–0.138 s/frame (2080: 0.437–0.622). 최대 VRAM 1.04–1.14 GB로 2080과 같다(+1 MB) |

**전체 세트 확인 (정정, 13:2x KST):**
- 위 5객체 검사는 모든 frame이 동일했다. 그러나 Vast 인과 pilot 3종의 G 상태(P5, 정책과 무관)를 Ubuntu Stage 0 기록과 비교하니, **DEV + REP 5,082 frame 중 266 frame(5.2%, 41객체)에서 J&F가 달랐다.**
  - frame 최대 차이: 0.004 (soapbox obj1)
  - 객체 평균 최대 차이: 8e-5
- Vast 3개 arm끼리는 G가 완전히 같다. 즉 Vast는 결정적이지만, 두 기계 사이는 긴 영상에서 비트 동일이 아니다. torch 2.14/cu130과 2.11/cu128 커널 수치 차이가 frame을 거치며 누적되는 것으로 보인다.
- 크기는 결정 규칙의 임계값(0.005)보다 약 60배 작다. 그래도 '비트 동일'이라는 앞의 표현은 5객체에만 해당하며 전체 세트에는 성립하지 않는다.

**운용 규칙 (정정 반영):**
1. **모든 비교는 같은 기계의 기준선과 짝지어 계산한다.** Vast 실험의 SGGL 기준선은 Vast에서 다시 실행한다(`outputs/milestone7/runs/SGGL`). 이것은 짝 비교용 별도 결과이고, Stage 0 기록을 대체하지 않는다.
2. Ubuntu Stage 0와 재생 결과(재생 screening, 감사)는 Ubuntu 수치로 그대로 보고한다.
3. 모든 Vast 산출 디렉터리에는 `ENV_vast.json`(기계, torch, 커밋, `fill_hole_area`)을 남긴다.

## 4. Ubuntu 작업 상태 (이전 시점)

| 작업 | 상태 |
|---|---|
| Stage 0 기록 (DEV 21 + REP 43) | 완료. 64쌍 CSV/NPZ, `.tmp` 잔여 없음 |
| Stage 1 재생 (20 정책) + 감사 + 픽셀 회계 | 완료. 64/64 객체 (`replay_report.json`, `fusion_pixels_*.csv`, `taxonomy_frames.csv`) |
| T head 학습 캐시 | 완료 (RTX 2080, 43 객체, 8.0 GB) |
| 대기 중이던 T 학습·평가(chain2)와 인과 pilot(chain3) | 사용자 승인으로 시작 전에 중지했다. Vast에서 실행한다 |
| 시각화 (`scripts/m7_viz.py`) | 사용자 중단 전 2개 그림만 생성. 나머지는 미실행 |

## 5. Vast에서 시작한 작업 (이전 직후)

- 사전 등록된 인과 pilot 3종(F1-blend.7, M3-vel, M1-always; DEV + REP)을 병렬로 시작했다.
- 이 작업들은 `nohup`으로 SSH 연결과 무관하게 실행된다.
- 로그는 `outputs/milestone7/logs/causal_*.log`, 결과는 `outputs/milestone7/runs/<arm>/`에 쌓인다.

## 6. 비용 (시간당 약 $0.571)

**지금까지 (이전 작업, 약 1시간):** 약 $0.6

**남은 M7 GPU 작업 추정 (5090 기준):**

| 작업 | 시간 | 비용 |
|---|---|---|
| 인과 pilot 3종 | 약 0.5시간 | |
| T 양성 대조와 학습 5종, DEV 평가 | 약 1시간 | |
| 개정·조합 2–3종 | 약 1시간 | |
| VK 결선 실행 | 약 0.3시간 | |
| **합계** | **약 3시간** | **약 $1.7** |

**유휴 비용:** 인스턴스를 켜 둔 채 놀리면 하루 약 $13.7이다.

## 7. 인스턴스 종료 전 체크리스트

1. `outputs/milestone7/runs`, `heads`, `teval`, `repro`, `logs`를 Ubuntu로 회수한다. 회수 명령: `ssh … 'cd /root/sam2-m7-test && tar cf - outputs/milestone7/{runs,heads,teval,logs}' | tar xf - --skip-old-files`
2. 작은 결과는 Ubuntu에서 커밋하고 push한다.
3. 그다음에 인스턴스를 중지하거나 삭제한다. 중지(stop)는 데이터를 보존하지만, 삭제(destroy)는 Vast의 모든 것을 지운다.
