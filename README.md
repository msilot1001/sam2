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
uv run python scripts/inspect_vkitti.py   # 실데이터 smoke test → outputs/milestone0/
```

## 다른 컴퓨터나 에이전트에서 이어서 작업하기

```bash
git clone https://github.com/msilot1001/sam2.git
cd sam2
```

최신 계획서를 먼저 읽고 그 범위에 맞춰 작업한다. 구현을 시작할 때는 [공식 SAM 2 저장소](https://github.com/facebookresearch/sam2)의 SAM 2.1 설치·체크포인트 지침과 [Virtual KITTI 2 자료](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)를 확인해 별도로 준비한다. 모델 코드·가중치·데이터는 이 저장소에 복사하지 않는다.

첫 구현 과제는 공식 기준선 재현, 네 선택 정책을 같은 조건에서 실행하는 파이프라인, 누락 패턴 생성과 평가다. 현재 프레임의 예측이 끝난 후에만 그 프레임을 메모리 후보에 넣고, 이후 프레임의 정답 마스크나 대상 차량의 GT 자세를 메모리 선택 입력에 사용하지 않는다. `extrinsic.txt`의 카메라 자세와 대상 차량의 `pose.txt`를 혼동하지 않는다. 세부 정책과 미확정 사항은 최신 계획서 5~13절을 따른다.
