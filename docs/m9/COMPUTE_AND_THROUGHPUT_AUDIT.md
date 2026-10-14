# M9 연산 자원과 처리량 감사

## 새 RunPod (2026-10-14 23:0x KST부터)

| 항목 | 측정값 |
|---|---|
| 접속 | 게이트웨이 `ssh.runpod.io`는 PTY 전용이다. Pod의 공개 IP와 TCP 포트로 직접 접속한다(주소는 `.gpu_worker.env`, gitignore). 공개키를 `authorized_keys`에 추가했다 |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, driver 595.91.07 |
| CPU | 실제 quota 13.6 vCPU (`cpu.cfs_quota_us` 1,360,000). `nproc`는 32를 보이지만 이것이 실제 상한이다 |
| RAM | 124 GB |
| 디스크 | 컨테이너 30 GB(빠름, 휘발성). `/workspace`는 MooseFS 네트워크 볼륨(마운트 확인)이며, M8 Pod의 내용이 그대로 이전돼 있었다 |
| 환경 | torch 2.11.0+cu128, torchvision 0.26.0+cu128(`nms` 동작), sam2 `_C` import 정상, nvcc 12.8 |
| 해시 | SAM 2.1 tiny 체크포인트 `7402e0d8…`, RAFT-small `01064c6d…` 모두 Ubuntu와 같다 |
| 이전 상태 | M8용 venv, 데이터(DAVIS, PUMaVOS A/B, SegTrack, VK Scene01/06)가 볼륨에 있었다. C는 없다. 컨테이너 디스크의 데이터 사본(`/root/w`)은 사라져 `localize.sh`로 다시 만들었다(약 1분) |
| 수치 동일성 | M8 FLOWVEL arm 8단위를 새 Pod에서 다시 돌렸다. M8 RunPod 결과와 frame 단위 J&F 최대 차이 **0** |

**worker 수 측정** (FLOWVEL arm, DEV+REP 단위 부분집합 0/8, 733 frame, 프로세스 시작 시간 포함):

| 구성 | frame/s | 단위/분 | GPU 사용률 평균 | VRAM 최대 | CPU throttling |
|---|---|---|---|---|---|
| 1 worker × 6 thread | 3.54 | 2.32 | 38% | 0.8 GB | 1 period |
| 2 worker × 6 thread | 4.16 | 2.73 | 45% | 4.1 GB | 0 |
| **4 worker × 3 thread** | **4.39** | **2.87** | 50% | 7.9 GB | 0 |
| 6 worker × 2 thread | 4.21 | 2.76 | 49% | 9.8 GB | 0 |

4 worker × 3 thread를 채택했다. 이전 M8 Pod의 같은 측정(3.41 frame/s)보다 29% 빠르다.

## Ubuntu RTX 2080

- 1 worker를 쓴다. M8 측정에서 2 worker는 합산 처리량이 0.82배로 떨어졌다.
- **VOST 진단 중단:** VOST Stage 1 진단(생성기 11종)은 첫 단위에서 118 frame 중 114 frame에 trigger가 걸려 단위당 4.6분이 걸렸다. 74단위면 5시간 이상이라 중단했다.
  - 대신 SGGL 기준선 arm과 FLOWVEL arm만 돌린다. 실패 분류는 SGGL 궤적과 GT 형상만으로 계산한다.
  - 중단된 부분 결과는 `outputs/milestone9/_aborted/diag_vost_partial/`에 보존했다.
- **CPU 병행 작업:** 데이터 받기, manifest 고정, 분석을 GPU 작업과 동시에 돌린다.

## 데이터 전송

| 대상 | 방식 | 속도 / 시간 |
|---|---|---|
| MOSEv2 | Pod에서 HF 분할 파일 3개(58 GB)를 한 번 스트리밍하며 선택 영상만 tar 추출 | 약 110 MB/s, 약 9분. 추출 결과 약 3 GB (컨테이너 디스크) |
| VOST | Ubuntu에서 S3 zip에 byte-range 요청 (remotezip, 16–32 thread) | 정답 7,820개 약 10분, JPEG 4,935개 약 8분. 연결 재설정이 한 번 나서 재시도 로직을 넣었다 |
