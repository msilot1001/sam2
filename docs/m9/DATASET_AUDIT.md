# M9 데이터셋 감사

기록 시각: 2026-10-14 23:40 KST (스프린트 시작 23:01 KST)

| 데이터셋 | 상태 | 출처 (확인한 공식 경로) | 라이선스 | 받은 범위 | 무결성 | SAM 2.1 학습 겹침 |
|---|---|---|---|---|---|---|
| **MOSEv2** | **사용** | Hugging Face `FudanCVL/MOSEv2` (MOSE-api README가 지정; 로그인·약관 동의 불필요, gated=False) | CC BY-NC-SA 4.0 (README 본문; HF 메타데이터는 cc-by-sa-4.0로 표기가 엇갈림), 비상업 연구용 | `train.tar.gz.aa/ab/ac`(합계 약 58 GB)를 RunPod에서 한 번 스트리밍하면서, 미리 고정한 180개 영상만 추출 | 분할 파일 3개 sha256이 저장소 `SHA256SUMS`와 일치 (`outputs/milestone9/manifests/mosev2_download_sha256.txt`) | MOSEv2는 2025-08 공개, SAM 2.1은 2024 공개라 학습에 포함될 수 없다. MOSEv1(SAM 2 학습 데이터)과 영상이 겹치는지는 **확인하지 못했다** |
| **VOST** | **사용** | 공식 S3 `tri-ml-public.s3.amazonaws.com/datasets/VOST.zip` (vostdataset.org/data.html이 지정, 등록 불필요) | CC BY-NC-SA 4.0 (zip 안의 LICENSE) | zip의 byte-range 요청으로 val 정답 전부(70영상, 7,820 PNG)와 평가 단위가 있는 44영상의 JPEG 4,935장만 받음 | 받은 파일 12,755개 sha256 목록 `outputs/milestone9/manifests/vost_files_sha256.txt`; zip ETag `656e2cc8…-6439` | 공개 문서상 SAM 2 학습 포함 근거 없음 (원본 영상은 Ego4D·EPIC-KITCHENS) |
| **LVOS v2** | **사용 (늦게, 사전 등록 수정 4)** | GitHub `LingyiHongfd/LVOS` README의 Google Drive 링크 (valid 파일 id `17Hwc__…`) | 주석 CC BY 4.0, 데이터 비상업 연구용 | 1차 재시도 6회 모두 할당량 거부. 2차 재시도 1회차(02:26 KST)에 valid 압축 11.8 GB 수신(zip store). 정답 140영상 전체와 manifest에 필요한 JPEG 7,654개만 Pod 컨테이너 디스크에 추출 | sha256 `beb488046f74…d4e2` 기록(`outputs/milestone9/runpod/manifests/lvosv2_archive_sha256.txt`). 공식 해시가 없어 대조는 못 함 | 압축은 Pod 볼륨에만 있다(Ubuntu에는 manifest와 결과만) |
| MOSEv1 val | 사용 안 함 | HF `FudanCVL/MOSE` | CC BY-NC-SA 4.0 | — | — | valid 정답은 첫 frame만 공개라 평가에 쓸 수 없고, SAM 2 학습에 MOSEv1 train이 들어 있어 독립성도 약함 |
| SA-V | 사용 안 함 | — | — | — | — | SAM 2 학습 데이터 (독립 검증으로 부적합) |
| PUMaVOS C | **봉인** | — | — | 전송·열람·실행 없음 | — | — |

**LVOS v2 처리:**
- Pod에서 20분 간격으로 최대 6번 자동 재시도한다(`/workspace/m9_lvos_retry.sh`, 로그 `m9_lvos_retry.log`).
- 1차는 모두 실패했고, 2차 재시도(30분 간격, `m9_lvos_retry2.sh`) 1회차에서 받았다. 도착을 2시간 반 늦게(04:58) 확인했다.

**MOSEv2 valid를 쓰지 않은 이유:** MOSE 계열 관례대로 valid 정답은 첫 frame만 공개된다(평가 서버 전용). 그래서 frame마다 정답이 있는 train split을 독립 평가 데이터로 쓴다. 이 데이터는 SAM 2.1 이후에 공개되었고, M6–M8에서 한 번도 보지 않았다.
