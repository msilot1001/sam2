# 처음부터 배우는 SAM 2 완전 이해 가이드

이 탭의 목적: 컴퓨터비전 지식이 전혀 없는 상태에서 출발해, 옆 탭의 선행연구 문서에 나오는 과제·모델 구조·메모리·실패 원인·평가지표·신뢰도 보정·실험 설계를 스스로 설명할 수 있는 수준까지 올라가는 것이다.

## 이 탭을 다 읽었을 때 할 수 있어야 하는 것

- classification, detection, segmentation, tracking의 차이를 출력 형태로 구분한다.
- SAM 2가 현재 프레임, prompt, 과거 memory를 어떻게 합쳐 mask를 만드는지 단계별로 설명한다.
- occlusion 뒤 drift와 identity switch가 왜 생기며, 틀린 mask의 memory 기록이 왜 위험한지 설명한다.
- SAM2Long, DAM4SAM, SAMURAI, UAMP, SENTRY, Lean-SAM2 같은 선행연구가 각각 어떤 문제를 이미 다뤘는지 구분한다.
- IoU, Dice, J, F, J&F를 직접 계산하고 평균 점수가 숨기는 실패 사건을 찾는다.
- confidence, uncertainty, calibration을 구분하고 ECE, Brier, AUROC, AUPRC가 무엇을 측정하는지 말한다.
- baseline, ablation, data leakage, cross-dataset test, bootstrap confidence interval을 포함한 공정한 실험을 설계한다.
- 최종 제안인 calibration-aware selective memory와 TRACK/SUSPECT/LOST controller가 무엇을 새롭게 검증하려는지 설명한다.

## 가장 효과적인 읽는 순서

1. PART A에서 이미지가 숫자가 되는 과정과 컴퓨터비전 과제를 익힌다.
2. PART B에서 신경망, Transformer, promptable foundation model의 최소 원리를 익힌다.
3. PART C에서 SAM 2의 구조와 memory loop를 한 frame씩 따라간다.
4. PART D에서 가림·방해물·저조도·작은 객체가 어떤 오류를 만드는지 이해한다.
5. PART E에서 평가·통계·재현성의 기준을 익힌다.
6. PART F에서 이 과제연구의 연구 질문, controller, 실험표를 다시 조립한다.
7. 마지막 용어사전과 자가점검 문제로 이해가 빈 곳을 확인한다.

## 처음 읽을 때 지켜야 할 세 가지

- 약어를 외우기 전에 입력, 출력, 정답이 무엇인지 먼저 확인한다.
- 논문의 성능 숫자를 보기 전에 prompt 조건, checkpoint, dataset, metric이 같은지 확인한다.
- 평균 점수가 올랐는지만 보지 말고 어떤 실패가 줄었고 비용이 얼마나 늘었는지 확인한다.

[FIGURE1]

그림 1. 분류에서 영상 객체 분할까지: 모델에게 묻는 질문과 출력의 정밀도가 어떻게 달라지는지 보여 준다.

## 전체 내용을 한 문장으로 먼저 보기

SAM 2는 사용자가 지정한 객체를 현재 영상에서 분할하고 과거 결과를 memory로 저장해 다음 frame을 돕지만, 틀린 결과가 memory에 들어가면 오류가 자기증폭될 수 있으므로, 내부 점수를 실제 미래 실패확률에 맞게 calibration하고 위험도에 따라 memory 기록·검증·복구 행동을 다르게 하자는 것이 옆 탭 연구안의 핵심이다.

