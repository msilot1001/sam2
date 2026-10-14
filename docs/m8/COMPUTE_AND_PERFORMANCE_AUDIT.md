# M8 계산 자원과 성능 감사

## 1. 자원 (측정)

| 항목 | Ubuntu (로컬) | Vast RTX 5090 |
|---|---|---|
| GPU | RTX 2080 8 GB, torch 2.14.1+cu130, `sam2._C` 없음 | **M8 동안 연결 불가** (89.221.67.161:16328 연결 거부, 30–60초 간격으로 계속 확인) |
| CPU / RAM | 10 vCPU, RAM 25 GB (사용 가능 20 GB) | — |
| 디스크 | 여유 41 GB. M8 출력은 수백 MB | — |
| 데이터 위치 | DAVIS 2017, PUMaVOS, SegTrack, VKITTI2 모두 로컬 | (M7 때는 C를 뺀 사본이 있었음) |
| 기존 캐시 | M7 `rec` 2.3 GB, `tcache` 7.5 GB (M8에서는 재사용 불필요) | — |
| 활성 작업 | M8 시작 시 없음. M7에서 남은 감시 shell 1개(우리 작업)는 종료 | — |

## 2. 작업 부하 측정 (2080, diag 실행에서 frame별 기록)

| 단계 | 측정값 | 출처 |
|---|---|---|
| SGGL 1 frame (G + L view, 선택) | 0.45 s | `lat_sggl`, shard 0 평균 |
| RAFT-small 광류 (긴 변 640) | 0.028 s / frame | `lat_flow` |
| 생성기 25개 / trigger frame (최적화 전) | 4.1 s | shard 0 `lat_rescue` |
| 생성기 31개 / trigger frame (backbone 공유 후) | 2.9 s | lab-coat 6 frame, GPU 단독 |
| 최대 VRAM (diag, 2080) | ≈ 1.0–1.6 GB | `peak_vram_bytes` |
| GPU 사용률 (diag 1 worker) | 98% | nvidia-smi |
| CPU 부하 (diag 1 worker) | load avg ≈ 2 / 10 | uptime |

**결론:**
- 2080은 GPU-bound다. M7에서도 worker 수와 상관없이 약 2.4 frame/s였다.
- 그래서 2080에서는 worker를 늘리지 않았다(1 worker, 4 thread).
- J/F 계산(CPU)은 frame당 수 ms로 병목이 아니다.
- GPU 사용률이 98%이므로 이 부하에서 GPU 동기화와 Python 루프는 병목으로 잡히지 않는다.

## 3. 최적화 기록

| 최적화 | 전 | 후 | 속도 | 결과 동일성 | 부동소수점 변화 | 유지 |
|---|---|---|---|---|---|---|
| 같은 view 영상의 backbone 1회 공유 (`rescue.probe(feats=...)`) | 25 probe 4.1 s (shard 0, GPU 공유 상태 포함) | 31 probe 2.9 s | probe당 0.164 → 0.094 s (≈1.7×; 측정 조건이 달라 상한 추정) | 비트 동일 (`test_shared_view_backbone_is_bit_identical_to_recomputing`) | 없음 | 유지 |
| 같은 기계 기준선 내장 | 별도 SGGL 실행 필요 | diag 실행이 SGGL 궤적과 지표를 기록해 인과 arm의 같은 기계 기준선으로 씀 | SGGL 1회 실행분 절약 | M7 Vast SGGL과 단위 평균 차이 ≤ 2.4e-4 (5단위) | — | 유지 |
| M7 계승: `OMP_NUM_THREADS` 3–4, `fill_hole_area=0` | — | — | M7에서 측정 | — | — | 유지 |
| 학습형 ranker를 CPU에서 | — | 수 분 | GPU 대기 없음 | — | — | 유지 |

혼합 정밀도(FP16/bf16)는 쓰지 않았다. FP32 기준선의 동작이 바뀌기 때문이다.

## 3.1 RunPod RTX 4090 워커 (17:00 KST부터)

| 항목 | 측정 |
|---|---|
| 자원 | RTX 4090 24 GB. CPU quota 13.6 vCPU (cgroup `cfs_quota_us`; `nproc` 48은 노드 값). RAM 251 GB. `/workspace`는 MooseFS 네트워크 볼륨, 컨테이너 디스크 30 GB |
| 환경 | torch 2.11.0+cu128, torchvision 0.26.0+cu128 (의존성 설치 중 PyPI 빌드로 바뀌어 `torchvision::nms`가 없어졌기 때문에 고정), sam2 @2b90b9f + `_C`, `fill_hole_area=0` |
| 검증 | 체크포인트 sha256 `7402e0d8…` 일치, RAFT 가중치 일치. DAVIS 12,420파일·PUMaVOS A/B + SegTrack 5,634파일 md5 일치. PUMaVOS C는 전송하지 않았다. unit manifest는 C 행을 지운 worker용 사본을 씀 |
| 테스트 | `tests/test_m7.py` + `tests/test_m8.py`: 16 passed, 2 skipped (M6 출력이 필요한 2개) |
| 재현성 | SGGL 8단위: RunPod − Ubuntu 2080, RunPod − M7 Vast 모두 단위 평균 J&F 차이 ≤ 8e-6 (일부 frame에서 부동소수점 수준 차이) |

**처리량 (SGGL arm, 같은 단위 부분집합 733 frame, 프로세스 시작 시간 포함):**

| 구성 | frame/s | 비고 |
|---|---|---|
| 1 worker, 데이터 네트워크 볼륨 | 1.71 | 프로세스가 I/O 대기(D 상태)이고 GPU가 0%인 구간이 많음 |
| 1 worker × 6 thread, 데이터 로컬 디스크 | 2.57 | +50% |
| 2 worker × 6 thread | 3.26 | |
| 4 worker × 3 thread | 3.41 | 채택. 짧은 부분집합에서는 시작 비용(약 40–60 s, venv가 네트워크 볼륨에 있음)의 비중이 큼 |

**최적화 판단:**
- 데이터를 컨테이너 디스크로 옮겼다(`/root/w/data`, 3.2 GB, 6분).
  - 결과는 비트 동일하다. 같은 파일이며 md5를 검증했다.
  - 출력, 코드, 체크포인트는 `/workspace`로 symlink를 걸었다. 출력은 Ubuntu 검증 전까지 영구 볼륨에 남아야 하기 때문이다.
- RTX 2080에서 2 worker를 실측했다(diag).
  - 프로세스당 시간: 1 worker는 0.63 s/frame, 4.9 s/trigger이고, 2 worker에서는 각각 1.1 s/frame, 9.0 s/trigger였다.
  - 합산 처리량이 1 worker의 약 0.82배였다. 그래서 2080은 1 worker로 되돌렸다. shard 2의 남은 단위는 RunPod로 옮겼다.

## 4. 시간과 비용 (2026-10-14)

| 자원 | 사용 | 내용 |
|---|---|---|
| Ubuntu RTX 2080 | 약 15:20–21:40 KST, GPU 거의 연속 사용 (≈ 6 GPU-h) | diag 53단위 + diag1b 22단위, refs, ORACLE_POOL/ADMIT 53단위 ×2, FLOWVEL_U 재현 |
| RunPod RTX 4090 | Pod 가동 07:35–12:25 UTC (4.8 h, 과금 기준). GPU 작업 08:50–12:21 UTC (≈ 3.5 GPU-h) | 설치와 데이터 이전에 약 1.2 h. diag 11단위. arm 14개 × 64단위 + VK 4 arm × 28단위 + ORACLE 11단위 ×2 |
| Vast RTX 5090 | 0 | M8 기간 내내 연결 불가 |
| 비용 | RunPod 4.8 h × 시간당 요금 | 요금은 확인하지 못했다. RTX 4090 일반 요금대(약 $0.3–0.7/h)라면 약 $1.5–3.4. 8시간 예산 안이다. Pod는 audit exit 0을 확인한 뒤 **stop**했다 (terminate 아님) |
| 처리량 개선 | | 데이터 로컬화 1.5×. 4 worker 2.0× (1 worker 대비). backbone 공유 ≈ 1.7× (probe 단위). 2080 2-worker는 0.82×라 되돌림 |

**데이터 전송:**
- DAVIS: Pod에서 직접 다운로드(0.8 GB zip)
- PUMaVOS A/B + SegTrack: 5,634파일
- VKITTI Scene01/06: 1,446파일
- 모두 md5로 검증했다.
- 결과 회수 기록: `outputs/SYNC_LEDGER.tsv` 원장의 NEW/SAME/APPENDED 행. 충돌 9건은 동기화 기능 시험 1건과 실행 중에 복사된 로그 8건이다. 결과 CSV 충돌은 없다.
