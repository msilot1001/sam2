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
