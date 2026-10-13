# SAM 2.1 메모리 선택 과제연구

불규칙하게 프레임이 누락되는 주행 영상에서 **촬영 카메라의 자세와 실제 경과시간으로 SAM 2.1의 과거 메모리를 고르면** 객체 분할 정확도와 누락 후 회복이 개선되는지 검증하는 연구다. 현재 저장소는 **연구 계획과 학습·발표 자료**를 담고 있다. SAM 2.1 실험 코드, 데이터셋, 체크포인트, 측정 결과는 아직 없다. 문서의 수치와 성공 기준은 실험 전 설계이며 검증된 성능으로 인용하면 안 된다.

## 먼저 읽을 파일

1. [최신 연구 계획서](guide-build/mentor_plan_revision.md): 연구 질문, 네 가지 메모리 선택 정책, 데이터 분할, 평가 기준의 기준 문서. [PDF](guide-build/mentor-plan-final.pdf)
2. [발표 준비용 해설](guide-build/understanding.md): SAM 2.1 메모리 흐름과 현재 계획을 처음부터 설명한다.
3. [선행연구 정리](SAM2_성능개선_선행연구_정리.md), [초기 주제 탐색](2학기_과제연구_컴퓨터비전_추천주제.md): 배경 자료. 초기 문서의 **신뢰도 보정 기반 선택적 메모리 갱신**은 현재 연구 주제와 다르다.
4. [발표 슬라이드](output/과제연구_계획발표_2장.pptx), [메모리 선택 구조도](output/SAM2_자세시간_메모리선택_구조도.html): 발표 자료.

이전 단계의 완성 문서는 [주제 후보 PDF](sam2_topic_qa/SAM2_topic_candidates_full.pdf), [SAM 2 학습 가이드 PDF](sam2_tutorial_qa/SAM2_full_document.pdf), [연구안 모음 PDF](sam2_plan_qa/sam2_plans_full_v3.pdf)에 보관했다. 현재 계획을 결정할 때는 위의 최신 연구 계획서를 우선한다.

`draft_*.md`와 `build_sam2_tutorial_content.mjs`, `make_sam2_tutorial_diagrams.py`, `sam2_tutorial_assets/`는 이전 학습 가이드의 원고와 제작 자료다. 제작 스크립트에는 원래 PC의 절대 경로와 Windows 글꼴 경로가 남아 있어 다른 컴퓨터에서 그대로 실행할 수 없다. 연구 설계의 기준은 위 최신 계획서다.

## 실험 설계 요약

- 기준 모델: 가중치를 고정한 SAM 2.1 Hiera-Tiny. 첫 프레임의 정답 마스크로 단일 차량을 지정하고 이후 프레임을 분할한다.
- 메모리: 활성 공간 메모리 `K=7`(첫 프롬프트 1개, 최근 수신 메모리 1개, 정책별 5개). 최초 프롬프트는 별도로 고정하고 과거 일반 메모리 후보 `L=32`개를 보관하는 것이 초기안이다.
- 비교 정책: `recent-K`, `time-distributed`, `pose-only`, `pose+Δt`. 모델, 입력 프레임, 누락 위치, 메모리 수와 추론 횟수를 동일하게 유지한다.
- 계획 데이터: Virtual KITTI 2의 clone 변형에서 왼쪽 카메라 RGB·instance mask·카메라 `extrinsic.txt`를 사용한다. 개발 장면은 01·06, 시험 장면은 02·18·20이다.
- 주 지표: 객체별 `J&F` 및 `pose+Δt − recent-K` 차이. 누락 후 회복, 지연 시간, FPS, GPU·CPU 메모리 비용도 기록한다.

## Milestone 0: Virtual KITTI 2 데이터 파이프라인

RGB·instance mask·카메라 extrinsic 로더, 차량 track 선택, 검증 스크립트. SAM 2.1 추론은 아직 포함하지 않는다. 자세한 내용은 [docs/MILESTONE_0.md](docs/MILESTONE_0.md).

```bash
uv sync                                   # Python 3.12 환경 (uv.lock 고정)
python3 scripts/setup_vkitti.py           # VKITTI2 다운로드·MD5 검증·Scene01/clone/Camera_0 추출 (CC BY-NC-SA 3.0)
uv run pytest                             # 단위 테스트 + 실데이터 통합 테스트
uv run python scripts/inspect_vkitti.py   # 실데이터 smoke test → outputs/milestone0/ (--scene Scene06)
```

## Milestone 1: 메모리 선택 정책 분리도 분석

SAM2와 독립된 `MemorySelector`(recent-K, time-distributed, pose-only, pose+Δt)를 구현하고, 개발 장면 01·06에서 네 정책이 실제로 다른 메모리를 고르는지 분석했다. 결과와 연구 판단은 [docs/MILESTONE_1.md](docs/MILESTONE_1.md)에 있다.

```bash
python3 scripts/setup_vkitti.py --scenes Scene01 Scene06 --skip-download
uv run python scripts/analyze_memory_selection.py   # → outputs/milestone1/
```

## Milestone 2: 순정 SAM 2.1 Hiera-Tiny baseline

공식 SAM 2를 upstream commit에 고정해 패키지로 설치하고(소스 복사 없음), 개발 track에서 첫 프레임 GT mask만 주고 순방향 추적한 뒤 J/F/J&F로 평가한다. 결과는 [docs/MILESTONE_2.md](docs/MILESTONE_2.md)에 있다.

```bash
uv sync                                                        # torch + sam-2 (pyproject의 sam2 그룹)
curl -L -o checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
uv run python scripts/run_sam2_baseline.py --device mps --smoke 10   # 1 track, 10 frames
uv run python scripts/run_sam2_baseline.py --device mps              # → outputs/milestone2/
```

## Milestone 3: 결정론적 프레임 누락 생성

`random_30/50`, `burst_30/50`(길이 4) 조건을 seed 0·1·2로 생성하고, track별 manifest(CSV + JSON, sha256 검증)를 만든다. 누락 clip 구성, 무손실 대비 paired 평가, burst 회복 사건 추출도 포함한다. 결과는 [docs/MILESTONE_3.md](docs/MILESTONE_3.md)에 있다.

```bash
uv run python scripts/frame_drops.py generate            # → outputs/milestone3/manifests (96개)
uv run python scripts/frame_drops.py smoke --device mps  # 누락 clip 3개로 SAM 2 smoke test
```

## Milestone 4: 프레임 누락 강건성 기준선

M3 manifest 96개(track 8개 × 4조건 × 3 seed)에 순정 SAM 2.1 Tiny를 실행하고, 같은 원본 frame에서 M2 무손실 예측과 비교했다. 30–50% 누락에서 track 평균 ΔJ&F는 −0.002 ~ +0.0005였다. 결과와 연구 판단은 [docs/MILESTONE_4.md](docs/MILESTONE_4.md)에 있다.

```bash
uv run python scripts/run_dropout_baseline.py --all      # 재개 가능, 순차 실행 (MPS 약 80분)
uv run python scripts/analyze_dropout_baseline.py        # → outputs/milestone4/{reports,figures}
```

## Milestone 4.5: 연구 타당성 검토 (탐색적)

다음 세 가지를 점검해 원래 가설을 검증할 수 있는지 판단했다.
- 실패 시점에 SAM 2가 쓸 수 있던 메모리
- 긴 burst(8·16) 스트레스 실험 96 run
- 정책 분리도·pose 잡음 tolerance·합성 재방문 기하, 시뮬레이터 대안

결론과 다음 단계 권고는 [docs/MILESTONE_4_5.md](docs/MILESTONE_4_5.md)에 있다.

```bash
uv run python scripts/run_dropout_baseline.py --config configs/m45_stress_runner.yaml --all
M4_CONFIG=configs/m45_stress_runner.yaml uv run python scripts/analyze_dropout_baseline.py
uv run python scripts/m45_feasibility.py
```

## 다른 컴퓨터나 에이전트에서 이어서 작업하기

```bash
git clone https://github.com/msilot1001/sam2.git
cd sam2
```

최신 계획서를 먼저 읽고 그 범위에 맞춰 작업한다. 구현을 시작할 때는 [공식 SAM 2 저장소](https://github.com/facebookresearch/sam2)의 SAM 2.1 설치·체크포인트 지침과 [Virtual KITTI 2 자료](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)를 확인해 별도로 준비한다. 모델 코드·가중치·데이터는 이 저장소에 복사하지 않는다.

첫 구현 과제는 공식 기준선 재현, 네 선택 정책을 같은 조건에서 실행하는 파이프라인, 누락 패턴 생성과 평가다. 현재 프레임의 예측이 끝난 후에만 그 프레임을 메모리 후보에 넣고, 이후 프레임의 정답 마스크나 대상 차량의 GT 자세를 메모리 선택 입력에 사용하지 않는다. `extrinsic.txt`의 카메라 자세와 대상 차량의 `pose.txt`를 혼동하지 않는다. 세부 정책과 미확정 사항은 최신 계획서 5~13절을 따른다.

## M5-Pre: 메모리 개입의 인과 효과 검증 (탐색적, CUDA)

메모리 고갈 원인 분리(spatial vs object pointer, backfill), 메모리 하나의 반사실적 효용, 후보 mask oracle 진단을 다룬다. 결과와 권고는 [docs/MILESTONE_5_PRE.md](docs/MILESTONE_5_PRE.md)에 있다. 브랜치는 `exp/m5-pre-memory-intervention`이다.

## M5.5: 공간 디테일 보존 타당성 검토 (탐색적, CUDA)

브랜치 `exp/m55-spatial-detail-feasibility`. 상세 보고서: [docs/MILESTONE_5_5.md](docs/MILESTONE_5_5.md).

- **prompt frame J가 낮았던 원인:** 평가 오류가 아니라 SAM 2 추론 경로의 256×256 재표본화 손실이다. 추론에서는 prompt 메모리를 256 왕복을 거친 마스크로 만들지만, 학습에서는 1024 마스크로 만든다.
- **대상 데이터:** DAVIS val 61개 객체, SegTrack v2 24개 객체(zero-shot) 전수.
  - detail 객체(작은·얇은·여러 조각)의 J는 0.687, plain 객체는 0.915다.
  - 256 출력 해상도가 detail 오차의 42%를 설명한다.
- **P5, full-res prompt 메모리 (5줄 교정):** DAVIS val 전체 frame에서 J&F +0.017(detail +0.055), 악화된 객체는 없다. SegTrack에서는 효과가 0이다.
- **학습 없는 guided 업샘플링:** 출력에만 적용하면 효과 0, 메모리에 넣으면 −0.013으로 해롭다.
- **권고:** NEED_MORE_REAL_VIDEO_DATA. 고해상도 zero-shot 데이터(PUMaVOS 또는 LVOS v2 val)에서 재검증해야 하며, 다운로드는 사용자 승인이 필요하다.

```bash
uv run python scripts/m55.py {audit|manifest|survey|ablate|tune|proto_manifest|proto|analyze}
```

## M5.6: 실제 영상 실패 종합 감사와 연구 방향 (탐색적, CUDA)

브랜치 `exp/m56-comprehensive-failure-audit`. 보고서: [docs/MILESTONE_5_6.md](docs/MILESTONE_5_6.md), 제안서: [docs/FINAL_RESEARCH_DIRECTION_PROPOSAL.md](docs/FINAL_RESEARCH_DIRECTION_PROPOSAL.md).

- **PUMaVOS (CC BY 4.0) 추가:** 영상 단위로 A 탐색 / B 개발 / C 확인으로 분할했다. C는 실행하지 않았다.
- **P5 효과:** PUMaVOS +0.003, DAVIS +0.017 J&F. 두 데이터 모두 악화된 객체는 없다.
- **실패 구간 35개 진단:**
  - 28개는 GT로 다시 prompt해도 회복되지 않는다.
  - 배포 가능한 메모리 개입으로 고쳐지는 것은 2개다.
  - 작은 대상의 결손은 두 데이터셋에서 반복된다.
- **잠정 방향:** 작은 대상을 위한 메모리 일관 확대 추론. 최소 기각 실험이 먼저 필요하다.

```bash
uv run python scripts/m56.py {prepare_pumavos|manifest|run|diagnose|scale_probe|analyze}
```
