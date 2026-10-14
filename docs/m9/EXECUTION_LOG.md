# M9 실행 기록 (KST)

| 시각 | 경과 | 일 |
|---|---|---|
| 23:01 | H0.0 | 스프린트 시작. 저장소 HEAD `e0c1b44` 확인, `exp/m9-cross-dataset-validation` 생성 |
| 23:02 | H0.0 | 새 RunPod: PTY 게이트웨이로 직접 TCP 주소 확인, 공개키 등록. RTX 4090 / quota 13.6 vCPU / RAM 124 GB / `/workspace` 이전 확인 |
| 23:03 | H0.0 | Pod 환경 검증(torch, torchvision cu128, `_C`, 체크포인트·RAFT 해시), 데이터 로컬화 |
| 23:04–23:10 | H0.1 | 공식 출처 확인: MOSEv2(HF), VOST(S3 zip), LVOS v2(Google Drive). 라이선스 기록 |
| 23:05 | H0.1 | MOSEv2 추출 표본 고정(meta만, seed 9, 180영상) → 58 GB 스트리밍 추출 시작 |
| 23:06 | H0.1 | 사용자가 Deep Research 보고서 2개를 첨부 → 전체 읽음 |
| 23:06–23:11 | H0.2 | LVOS v2: 처음 받기를 디스크 위험으로 중단 → 재시도 시 Google Drive 할당량 거부(BLOCKED), 자동 재시도 예약 |
| 23:10 | H0.2 | Pod worker 벤치마크(1/2/4, 이어서 6) → 4×3 채택. M8 FLOWVEL 8단위 수치 동일(차이 0) |
| 23:13 | H0.2 | 사전 등록 arm 고정: SCALE, JOINT, PCCARRY (`outputs/milestone9/arms.json`) |
| 23:14 | H0.2 | VOST val 정답 7,820개 수신 → VOST manifest 고정(74단위 / 44영상) |
| ~23:25 | H0.4 | `preregistration.json` 고정. VOST JPEG 4,935개 수신(재시도 로직 추가 후 완료), 어댑터 연기 시험 4단위와 prompt 겹침 그림 확인 |
| ~23:28 | H0.5 | MOSEv2 분할 파일 sha256 공식값 일치 확인, manifest 고정(139단위 / 102영상) |
| ~23:30 | H0.5 | Pod: MOSEv2 Stage 1 진단 시작 (4 worker) |
| ~23:31 | H0.5 | 2080: VOST 진단 시작 → 첫 단위 4.6분(trigger 97%)이라 중단 → SGGL·FLOWVEL arm으로 전환 |
| ~23:45 | H0.7 | 첫 M9 커밋 `550f7c5` push |
| 00:40 | H1.7 | MOSEv2 Stage 1 진단 완료(139단위). 실패 분류: SCALE 44%(38영상), POSITION 19% … 거짓 재획득 495/3,392 GT-빈 frame |
| ~00:45 | H1.7 | Stage 1 관문: 모든 생성기 recall < 0.25, B_att ≈ A_cos(attention 중단), 순위화 held-out 순이득 ≤ 0 → rescue·attention 경로 중단 |
| 00:50 | H1.8 | 사전 등록 수정 1: CONSENSUS arm(존재 합의 기권) 고정 — MOSEv2 dev 재생만 근거, 판정은 conf/VOST |
| ~00:55 | H1.9 | 첫 frame 크기 불일치 메커니즘 확인(대상이 화면에 들어오며 수 배 성장) |
| 01:08 | H2.1 | FLOWVEL MOSEv2 완료: ΔJ&F −0.0142 [−0.0284, −0.0016], −0.05 미만 14단위 → 일반화 실패. 손실은 운동 두 성분 모두에서 → JOINT 건너뜀, PCCARRY 뒤로, VOST arm을 Pod 큐에 추가 |
| 01:30 | H2.5 | SCALE MOSEv2 완료: +0.0116 [−0.0003, +0.0258], 스케일 범주 회복 18%(기준 30%), −0.05 미만 10단위, dev 이득 상위 2영상 99% → 사전 기준 불통과. SCALE − FLOWVEL +0.0257 (CI > 0). 수정 1회는 단일 원인이 없어 쓰지 않음 |
| 01:53 | H2.9 | CONSENSUS MOSEv2 완료: conf −0.0503 [−0.0983, −0.0094], −0.05 미만 15단위 → 사전 반증, 중단. VOST CONSENSUS 취소(STOP_Q3_CONSENSUS). dev 재생 근거가 conf로 옮겨가지 않음(conf에서 기권 frame의 GT 있는 쪽 손실 −392 vs 얻음 +96) |
| 02:00 | H3.0 | 진단: SCALE 범주 놓친 frame(dev)에서 G 단독도 J&F 0.15 → 해상도 문제가 아니라 두 view 모두 커진 대상을 놓침. 새 메커니즘(크기 기반 view 전환)은 근거 없어 만들지 않음 |
| 02:02 | H3.0 | LVOS v2 2차 재시도(30분 간격 6회) 시작. Pod q4 추가: VOST FLOWVEL(Pod). **분석 규칙(VOST 결과 보기 전 고정):** 데이터셋별 주 검정은 RunPod 행(모든 arm이 같은 기계), Ubuntu 2080 행은 재현으로 따로 보고 |
| 02:37 | H3.6 | Pod VOST SGGL·SCALE 완료. SCALE VOST −0.0073 [−0.0389, +0.0199], 스케일 범주 회복 2%, −0.05 미만 10단위 → SCALE 두 데이터셋 모두 불통과. VOST SGGL 기계 간 차이 평균 |Δ| 0.0006 |
| 02:50 | H3.8 | 사전 등록 수정 2(진단만): 스케일 범주 실패가 '부분 마스크 고착'인지 '이탈'인지 측정. `m8.py` 행에 출력 면적·교집합 열 4개 추가(테스트 9개 통과), Pod q5에서 SGGL을 MOSEv2·VOST 전체에 다시 실행(별도 디렉터리 `runs_area`, diag 행과 수치 일치 확인 예정) |
| 03:01 | H4.0 | PCCARRY MOSEv2 완료: −0.0031 [−0.0102, +0.0037], −0.05 미만 6단위, 동시 부재 frame당 +0.004 → 중단, VOST PCCARRY 실행 안 함. Pod q4(VOST FLOWVEL) 시작 |
| 03:11 | H4.2 | Ubuntu 2080 VOST FLOWVEL 완료(재현): +0.0051 [−0.0177, +0.0278], −0.05 미만 8단위. 주 검정(Pod)은 진행 중 |
| 03:25 | H4.4 | 모든 arm이 ±0.05 넘는 단위를 양방향으로 비슷하게 만든다는 관찰(기계 간 잡음은 0개) → 사전 등록 수정 3: NULLJIT 영 대조(L crop 변 × U(0.85, 1.15), 단위별 고정 seed, 크기는 SCALE 변화의 중앙값). VOST는 2080, MOSEv2는 Pod q6 |
| 03:27 | H4.4 | Pod VOST FLOWVEL 완료(주 검정): +0.0046 [−0.0181, +0.0272], −0.05 미만 8단위, 극소 −0.036. 2080 재현과 단위별 상관 0.999 → FLOWVEL 일반화 실패 확정 |
| 03:54 | H4.9 | 면적 진단 MOSEv2: SGGL 재실행이 diag 행과 완전히 일치(최대 |ΔJ&F| 0). 스케일 범주 놓친 frame 중 출력 있는 743개의 precision 중앙값 0.07, recall 0.02 → **H_drift**(사전 기준). 사후(공개): 놓친 손실 전체의 54%가 '출력이 대상 밖'(64영상), 32%가 '출력 없음', 10%만 '대상 일부' |
| 04:16 | H5.3 | 면적 진단 VOST: 일치 0, 스케일 범주 '대상 밖' 73% → H_drift. 사후: 놓친 손실의 68%가 출력이 대상 밖(38영상). NULLJIT MOSEv2(Pod q6) 시작 |
| 04:57 | H5.9 | NULLJIT 완료(MOSEv2 Pod, VOST 2080): MOSEv2 −0.0011 (+0.05 초과 9 / −0.05 미만 8), VOST +0.0089 (9 / 7). 아무 정보 없는 crop 흔들림도 다른 arm과 같은 수준의 단위 변동을 만든다 |
| 04:58 | H5.9 | LVOS v2 valid 압축이 02:26에 2차 재시도 1회차로 도착해 있었음을 확인(sha256 beb48804…, 11.8 GB) |
| 05:08 | H6.1 | 사전 등록 수정 4(LVOS v2): M9 단위 규칙 그대로 + 연산 예산(단위당 첫 300 frame, 영상을 seed 9 순서로 9,000 frame까지). manifest 고정: 32단위 / 27영상(dev 16 / conf 16, 극소 11), 8,854 frame. 필요한 JPEG 7,654개만 추출 |
| 05:12 | H6.2 | Pod q7: LVOS v2 SGGL → FLOWVEL → SCALE → NULLJIT 시작. 첫 단위 정상(299 frame) |
