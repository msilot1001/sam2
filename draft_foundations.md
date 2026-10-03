# SAM 2 과제연구를 위한 컴퓨터비전 기초

## 이 글의 목표

이 글은 AI를 처음 배우는 고등학생이 SAM 2 관련 논문과 코드를 스스로 읽을 수 있도록 만든 입문 원고다. 수식을 먼저 외우기보다 다음 질문에 답할 수 있게 되는 것을 목표로 한다.

- 이미지와 영상은 컴퓨터 안에서 어떤 숫자로 표현되는가?
- 분류, 객체 탐지, 분할, 추적은 무엇이 다른가?
- encoder, decoder, feature, attention, loss는 어떤 일을 하는가?
- fine-tuning, adapter, LoRA, distillation은 왜 쓰는가?
- SAM 2의 어느 부분을 바꾸면 정확도·경계·속도·강건성을 개선할 수 있는가?

영문 논문에서는 같은 개념을 영어로 쓰므로, 중요한 용어는 한국어 뒤에 원문 표기를 함께 적었다. 약어만 외우기보다 “입력으로 무엇을 받고, 출력으로 무엇을 내며, 어떤 문제를 해결하는가”를 기준으로 이해하는 편이 좋다.

---

# 1. AI, 머신러닝, 딥러닝, 컴퓨터비전

## 1.1 인공지능 — Artificial Intelligence, AI

**한 줄 정의:** 사람이 지능적으로 한다고 여기는 판단, 계획, 인식, 언어 이해 등을 컴퓨터가 수행하게 만드는 넓은 분야다.

AI는 가장 큰 우산이다. 예를 들어 정해진 규칙만 따라 체스를 두는 프로그램도 AI라고 부를 수 있다. 반드시 데이터를 학습해야 AI인 것은 아니다.

**비유:** “이동수단”이라는 큰 범주 안에 자동차, 자전거, 기차가 들어가는 것과 같다.

**예시**

- 사진을 보고 고양이인지 판단한다.
- 질문을 읽고 답을 생성한다.
- 자율주행차가 보행자를 발견하고 멈춘다.
- 게임에서 다음 행동을 계획한다.

## 1.2 머신러닝 — Machine Learning, ML

**한 줄 정의:** 사람이 모든 규칙을 직접 작성하지 않고, 데이터의 예시에서 규칙이나 패턴을 학습하게 하는 AI 방법이다.

전통적인 규칙 기반 프로그램은 사람이 “귀가 뾰족하고 수염이 있으면 고양이” 같은 규칙을 적는다. 머신러닝은 고양이와 개 사진을 많이 보여 주고 둘을 구분하는 기준을 데이터에서 찾게 한다.

**비유:** 정답 풀이법을 통째로 알려 주는 대신 많은 문제와 답을 주고 학생이 공통 규칙을 발견하게 하는 것과 같다.

## 1.3 딥러닝 — Deep Learning, DL

**한 줄 정의:** 여러 층으로 이루어진 인공신경망을 사용해 데이터에서 복잡한 표현을 단계적으로 학습하는 머신러닝의 한 종류다.

초기 층은 선과 색 변화 같은 단순 특징을 찾고, 뒤쪽 층은 눈, 바퀴, 사람의 형태처럼 더 복잡한 특징을 조합한다. 현대 이미지 분류, 객체 탐지, 영상 분할 대부분이 딥러닝을 사용한다.

**왜 deep인가?** 입력과 출력 사이에 학습되는 층이 많이 쌓여 있기 때문이다. “깊다”는 말이 모델이 인간처럼 깊게 생각한다는 뜻은 아니다.

## 1.4 컴퓨터비전 — Computer Vision, CV

**한 줄 정의:** 컴퓨터가 이미지와 영상에서 의미 있는 정보를 얻도록 만드는 분야다.

컴퓨터비전은 “문제 분야”이고, 딥러닝은 그 문제를 푸는 “방법” 중 하나다. 컴퓨터비전에는 딥러닝 이전의 윤곽선 검출, 색상 분석, optical flow 같은 전통 알고리즘도 포함된다.

**관계 정리**

인공지능 AI ⊃ 머신러닝 ML ⊃ 딥러닝 DL

컴퓨터비전은 이들과 겹치는 응용 분야다. 컴퓨터비전 문제를 딥러닝으로 풀 수도 있고 전통 영상처리로 풀 수도 있다.

**하나의 예로 연결하기:** 재활용 로봇이 컨베이어 벨트 위 물체를 처리한다고 하자.

- AI: 로봇이 상황을 보고 행동을 결정하는 전체 시스템
- ML: 플라스틱과 캔의 차이를 데이터에서 학습
- DL: 여러 층의 신경망으로 모양과 재질 특징을 학습
- CV: 카메라 영상에서 캔의 위치와 윤곽을 찾는 부분

---

# 2. 이미지와 영상은 어떻게 숫자가 되는가

## 2.1 픽셀 — Pixel

**한 줄 정의:** 디지털 이미지를 이루는 가장 작은 격자 칸이다.

사진을 아주 크게 확대하면 작은 정사각형들이 보인다. 각 칸에는 그 위치의 색 또는 밝기를 나타내는 숫자가 들어 있다. 픽셀 하나 자체에는 “사람의 눈”이나 “자동차 바퀴”라는 의미가 없다. 주변 픽셀과 함께 해석될 때 물체의 일부가 된다.

**비유:** 모자이크 벽화의 타일 한 조각이다. 타일 하나만 보면 의미가 없지만 수천 개를 배열하면 그림이 된다.

## 2.2 RGB와 채널 — RGB, Channel

컬러 이미지는 보통 빨강 Red, 초록 Green, 파랑 Blue 밝기를 합쳐 표현한다. 이 세 장의 숫자 지도를 RGB channel이라고 한다.

8-bit 이미지에서는 각 값이 보통 0부터 255까지다.

- 검정: (0, 0, 0)
- 흰색: (255, 255, 255)
- 순수한 빨강: (255, 0, 0)
- 노랑: (255, 255, 0)

모델에 넣기 전에는 0~255를 0~1로 나누거나, 평균과 표준편차를 이용해 정규화 normalization하기도 한다.

**비유:** 투명한 빨강 필름, 초록 필름, 파랑 필름 세 장을 겹쳐 컬러 사진을 만드는 것과 같다.

### 입력 채널과 특징 채널은 다르다

- 입력 RGB 이미지: 3 channels
- 흑백 의료 영상: 흔히 1 channel
- 깊이까지 포함한 RGB-D: RGB 3개와 depth 1개
- 신경망 중간 feature: 64, 256, 1024 channels처럼 많아질 수 있음

중간 feature channel은 실제 색이 아니다. 어떤 채널은 세로 경계에, 다른 채널은 특정 질감이나 물체 부분에 강하게 반응하는 학습된 숫자 지도다.

**자주 하는 오해:** channel 수와 class 수는 같지 않다. 256 feature channels를 쓴다고 256개 종류를 분류하는 것이 아니다.

## 2.3 해상도 — Resolution

**한 줄 정의:** 이미지의 가로와 세로 픽셀 수다.

1920×1080 영상은 가로 1920, 세로 1080이며 한 frame에 약 207만 픽셀이 있다. 해상도가 높으면 작은 물체와 가는 경계를 더 많이 담을 수 있지만 계산량과 메모리 사용량도 커진다.

**주의:** 원본이 저해상도인데 단순히 네 배 확대한다고 새로운 세부 정보가 생기는 것은 아니다. 기존 픽셀 사이를 보간 interpolation한 것일 뿐이다.

### 이미지 해상도와 feature map 해상도

원본이 1024×1024여도 encoder를 지나며 feature가 256×256, 64×64처럼 줄어들 수 있다. 이 과정을 downsampling이라고 한다. 계산은 줄지만 작은 물체나 1~2픽셀짜리 경계가 사라질 수 있다. SAM 2의 경계 개선 논문들이 high-resolution feature와 decoder를 추가하는 이유다.

## 2.4 프레임과 FPS — Frame, Frames Per Second

**frame:** 영상을 이루는 한 장의 정지 이미지다.

**FPS:** 1초에 몇 frame을 보여 주거나 처리하는지를 뜻한다. 30 FPS 영상 10초는 약 300 frames다.

영상에는 공간 정보뿐 아니라 시간 정보가 있다.

- 공간 spatial: 한 frame 안에서 어디에 무엇이 있는가?
- 시간 temporal: 앞뒤 frame 사이에 무엇이 어떻게 움직였는가?

**주의:** 영상 자체가 30 FPS라고 모델도 30 FPS로 처리한다는 뜻은 아니다. 모델 처리 속도가 8 FPS이면 실시간 30 FPS 입력을 따라가지 못한다.

## 2.5 텐서 — Tensor

**한 줄 정의:** 숫자를 여러 축으로 배열한 다차원 상자다. 딥러닝에서는 이미지, 영상, feature, mask를 모두 tensor로 표현한다.

- 숫자 하나: scalar
- 숫자 한 줄: vector
- 행과 열: matrix
- 세 축 이상: tensor

예를 들어 RGB 이미지 한 장은 높이 H × 너비 W × 채널 C 형태로 생각할 수 있다. 라이브러리에 따라 축 순서가 달라진다.

- 보기 쉬운 표기: (H, W, C) = (1080, 1920, 3)
- PyTorch에서 흔한 표기: (B, C, H, W)
- 영상 batch: (B, T, C, H, W)

여기서 B는 한 번에 처리하는 데이터 개수인 batch size, T는 frame 수다.

**비유:** 스프레드시트가 2차원 표라면 tensor는 여러 장의 표를 시간과 batch 방향으로 쌓은 파일함이다.

### Shape를 읽는 습관

논문과 코드를 볼 때 tensor의 shape를 추적하면 오류를 많이 줄일 수 있다. 예를 들어 (2, 30, 3, 512, 512)는 RGB 영상 2개를 한 번에 처리하며, 각 영상은 30 frames이고 frame 해상도는 512×512라는 뜻이다.

## 2.6 Mask, Bounding Box, Ground Truth, Annotation

- **mask:** 각 픽셀이 대상인지 아닌지를 표시한 지도. 이진 mask는 보통 대상 1, 배경 0이다.
- **bounding box:** 대상을 둘러싼 직사각형. 보통 (x_min, y_min, x_max, y_max)로 표현한다.
- **annotation:** 사람이 데이터에 붙인 정답 표시. class label, box, mask 등이 될 수 있다.
- **ground truth, GT:** 평가와 학습에 정답으로 사용하는 annotation.

Box는 만들기 쉽지만 빈 배경까지 포함하고 정확한 모양을 표현하지 못한다. Mask는 정확하지만 사람이 픽셀 단위로 그리는 비용이 크다.

---

# 3. 컴퓨터비전 문제의 종류

한 교실 사진을 예로 들어 보자.

| 문제 | 모델이 답하는 질문 | 대표 출력 |
|---|---|---|
| Image Classification | 이 사진은 무엇을 보여 주는가? | 교실 또는 class probability |
| Object Detection | 어떤 물체가 어디 있는가? | class + box + confidence |
| Semantic Segmentation | 각 픽셀은 어떤 종류인가? | 픽셀별 class map |
| Instance Segmentation | 각 물체 개체의 정확한 모양은? | 개체별 mask와 ID |
| Panoptic Segmentation | 화면의 모든 픽셀은 무엇이며 개체 번호는 무엇인가? | class와 instance가 결합된 전체 지도 |
| Tracking | 시간에 따라 같은 물체가 어디로 이동하는가? | frame별 위치 또는 mask와 지속되는 ID |

## 3.1 이미지 분류 — Image Classification

사진 전체에 하나 이상의 label을 붙인다.

- single-label: 고양이 하나를 선택
- multi-label: 사람, 자전거, 도로를 동시에 선택

분류는 물체가 사진의 어디에 있는지는 알려 주지 않는다. 고양이가 사진 구석에 작게 있어도 출력은 단지 “고양이”일 수 있다.

## 3.2 객체 탐지 — Object Detection

물체의 class와 위치를 직사각형 box로 찾는다. 출력에는 보통 class, confidence, box 좌표가 들어간다.

예: person, confidence 0.93, box (120, 40, 310, 470)

**분류와 비교:** 분류는 사진 전체에 답 하나, detection은 여러 물체마다 답과 위치를 준다.

**segmentation과 비교:** box는 물체를 넉넉히 둘러싸기 때문에 팔 사이의 배경이나 자동차 아래 도로도 포함한다. 정확한 윤곽이 필요하면 segmentation이 필요하다.

## 3.3 의미론적 분할 — Semantic Segmentation

모든 픽셀에 class를 붙인다. 모든 학생 픽셀은 person, 모든 의자 픽셀은 chair가 된다.

같은 class의 서로 다른 개체를 구분하지 않는다. 학생 세 명이 붙어 서 있으면 모두 같은 person 영역으로 표시될 수 있다.

**적합한 예:** 도로, 하늘, 건물, 물, 농경지처럼 “어떤 종류의 영역인가”가 중요한 문제.

## 3.4 개체 분할 — Instance Segmentation

각 물체의 픽셀을 정확히 찾으면서 같은 class의 개체도 따로 구분한다.

- 학생 A: person, instance ID 1
- 학생 B: person, instance ID 2
- 학생 C: person, instance ID 3

**semantic과 비교:** 둘 다 픽셀 mask를 만들지만 semantic은 종류만, instance는 “어느 개체인지”까지 구분한다.

## 3.5 파놉틱 분할 — Panoptic Segmentation

Semantic segmentation과 instance segmentation을 하나로 합친다. 화면의 모든 픽셀에 class를 붙이고, 셀 수 있는 물체는 instance도 나눈다.

- **things:** 사람, 자동차, 컵처럼 개별 개체를 셀 수 있는 대상
- **stuff:** 하늘, 도로, 잔디처럼 덩어리로 보는 배경 영역

교실에서는 학생과 의자는 개체별로 나누고, 벽과 바닥도 빠짐없이 class를 붙인다.

## 3.6 추적 — Tracking

여러 frame에서 같은 물체의 정체성, 즉 identity를 유지하며 위치를 따라간다. 1번 학생이 다른 학생 뒤에 잠깐 가려져도 다시 나타났을 때 계속 ID 1이어야 한다.

Tracking의 출력은 문제에 따라 box, point, keypoint, mask가 될 수 있다. 따라서 “tracking은 항상 box”라고 생각하면 안 되지만, 전통적인 tracking benchmark는 box를 많이 사용한다.

### 어려운 상황

- occlusion: 다른 물체에 가려짐
- disappearance: 화면 밖으로 나가거나 완전히 사라짐
- reappearance: 다시 등장함
- distractor: 목표와 비슷하게 생긴 방해 물체
- fast motion: frame 사이 이동이 너무 큼
- deformation: 사람의 자세처럼 모양이 크게 변함
- camera cut: 장면이 갑자기 바뀜

## 3.7 VOS — Video Object Segmentation

**Video Object Segmentation:** 영상에서 특정 객체의 mask를 frame마다 예측하면서 동일 객체를 유지하는 문제다. “분할과 시간적 추적”이라고 생각하면 된다.

전형적인 설정에서는 첫 frame의 정답 mask 또는 box를 주고 이후 frame을 자동으로 분할한다. 논문에 따라 다음 하위 설정이 있다.

- **semi-supervised VOS:** 첫 frame target annotation을 제공
- **unsupervised 또는 automatic VOS:** 첫 frame 정답 없이 눈에 띄는 객체를 자동 발견
- **interactive VOS:** 사용자가 중간에 click이나 수정 prompt를 제공
- **long-term VOS:** 장시간 가림, 화면 이탈, 재등장을 포함

VOS의 대표 지표는 영역 IoU인 J, 경계 정확도인 F, 두 평균인 J&F다.

## 3.8 PVOS — Promptable Video Object Segmentation

점, box, mask 같은 visual prompt로 사용자가 “이 객체”를 지정하면 모델이 그 mask를 영상 전체에 전파하는 문제다. SAM 2가 대표적인 promptable video segmentation 모델이다.

예를 들어 첫 frame에서 강아지 몸 안을 한 번 클릭하면 이후 frame에서도 강아지를 분할한다. 필요하면 중간 frame에 positive point, negative point, box, correction mask를 추가할 수 있다.

**VOS와 관계:** PVOS는 VOS 중에서도 사용자의 prompt로 어떤 객체를 분할할지 유연하게 지정하고 수정할 수 있는 설정이다.

**약어 주의:** 일부 후속 논문은 PVOS를 Promptable Video Object Segmentation이라는 뜻으로 쓰지만 약어가 완전히 통일된 것은 아니다. SAM 2 원 논문이 image와 video를 함께 포괄하며 공식적으로 강조한 표현은 **PVS, Promptable Visual Segmentation**이다. 논문마다 full name과 task definition을 먼저 확인해야 한다.

## 3.9 VOT — Visual Object Tracking

첫 frame에서 한 target의 box나 mask를 주고 이후 위치를 계속 추정한다. 전통 VOT는 주로 box와 tracking 성공률, precision, AUC, EAO를 평가한다. SAM 2 기반 tracker는 내부적으로 mask를 예측해도 최종 평가는 box로 변환하는 경우가 많다.

**VOS와 비교**

- VOS: 픽셀 단위 mask 품질이 핵심
- VOT: target 위치와 추적 지속성이 핵심이며 box 평가가 흔함

따라서 VOT AUC가 올랐다는 결과만으로 mask 경계가 좋아졌다고 말할 수 없다.

## 3.10 MOT — Multiple Object Tracking

영상 속 여러 객체를 동시에 검출하고 ID를 유지한다. 예를 들어 도로의 차량 20대에 ID를 붙이고 이동 경로를 만든다.

보통 detection과 association 두 단계로 이해할 수 있다.

1. 각 frame에서 사람이나 차량을 검출한다.
2. 이전 frame의 track과 현재 detection이 같은 개체인지 연결한다.
3. 가림 뒤에도 ID가 바뀌지 않게 한다.

**MOT와 multi-object VOS의 차이:** MOT는 box 중심인 경우가 많고 새로 등장하는 객체도 찾는다. Multi-object VOS는 지정된 여러 객체의 mask를 전파하는 경우가 많다. Box 대신 mask까지 추적하는 분야는 MOTS, Multi-Object Tracking and Segmentation이라고 부르기도 한다.

## 3.11 RVOS — Referring Video Object Segmentation

자연어 표현이 가리키는 객체를 영상 전체에서 분할한다.

예: “빨간 모자를 쓰고 왼쪽에서 달리는 사람”

모델은 문장의 색, 위치, 행동 정보를 영상과 연결해 target을 선택한 뒤 mask를 추적해야 한다. RVOS는 “어떤 객체인지”를 점이나 box가 아니라 문장으로 알려 준다.

## 3.12 VRS — Video Reasoning Segmentation

문장에 target이 직접 드러나지 않고, 상식이나 시간적 사건을 추론해야 하는 객체를 영상에서 분할한다.

- RVOS: “빨간 셔츠를 입은 선수” — 눈에 보이는 속성으로 바로 찾을 수 있음
- VRS: “경기에서 마지막 골을 넣은 선수” — 영상을 시간 순서로 이해하고 사건을 추론해야 함

VRS는 language understanding, temporal reasoning, object localization, mask propagation을 모두 요구한다. 최근 reasoning segmentation 문헌에서 VRS는 대체로 Video Reasoning Segmentation을 뜻하지만, 일부 문맥에서는 Video Referring Segmentation처럼 다른 full name으로도 쓰일 수 있다. 처음 등장한 full name과 입력 문장이 명시적인 지시인지 추론이 필요한 지시인지 반드시 확인해야 한다.

## 3.13 한 번에 구분하기

- 사진 전체에 이름 하나 → classification
- 물체마다 직사각형 → detection
- 픽셀마다 종류 → semantic segmentation
- 픽셀마다 종류와 개체 → instance segmentation
- 배경까지 모두 종류와 개체 → panoptic segmentation
- 시간에 따라 같은 ID 유지 → tracking
- 영상에서 지정된 객체의 mask 유지 → VOS
- 점, box, mask prompt로 지정 → PVOS
- SAM 2 원문의 image·video 통합 promptable task → PVS
- 주로 box로 한 target 추적 → VOT
- 여러 target을 detection하고 ID 연결 → MOT
- 명시적 문장으로 target 지정 → RVOS
- 암시적 문장을 추론해 target 지정 → VRS

---

# 4. 신경망 안에서는 무슨 일이 일어나는가

## 4.1 인공신경망 — Neural Network

**한 줄 정의:** 입력 숫자를 여러 층에서 변환해 원하는 출력으로 바꾸는, 학습 가능한 함수다.

신경망의 기본 단위는 입력에 weight를 곱하고 bias를 더한 뒤 activation function을 통과시킨다. 수많은 단위가 층으로 연결되어 복잡한 변환을 만든다.

**비유:** 사진이 여러 검사대를 통과하는 공장과 같다.

1. 앞쪽 검사대는 밝기 변화와 선을 찾는다.
2. 중간 검사대는 모서리, 질감, 물체 부분을 조합한다.
3. 뒤쪽 검사대는 “이 영역이 강아지일 가능성”이나 mask를 만든다.

신경망이 사람의 뇌를 그대로 복제한 것은 아니다. 이름과 초기 아이디어가 신경세포에서 왔지만, 실제로는 큰 행렬 계산을 반복하는 수학 모델이다.

## 4.2 층 — Layer

**한 줄 정의:** 입력 tensor를 다른 tensor로 바꾸는 하나의 계산 단계다.

Convolution layer, linear layer, attention layer, normalization layer처럼 역할이 다른 층이 있다. “깊은 모델”은 이런 층이 많이 연결된 모델이다.

층을 많이 쌓는다고 무조건 좋아지지는 않는다. 계산량, 학습 난도, 과적합 가능성도 커진다. 좋은 구조는 필요한 정보를 잃지 않으면서 효율적으로 변환한다.

## 4.3 파라미터 — Parameter

**한 줄 정의:** 학습 과정에서 데이터에 맞게 값이 바뀌는 숫자다. Weight와 bias가 대표적이다.

처음에는 대체로 임의 값에서 시작하거나 사전학습 checkpoint에서 가져온다. 모델은 loss를 줄이는 방향으로 parameter를 조금씩 수정한다.

**비유:** 사진 보정 앱의 수백만 개 조절 손잡이와 같다. 학습은 정답에 가까워지도록 손잡이 위치를 자동으로 조정하는 과정이다.

### Parameter와 hyperparameter

- **parameter:** 모델이 학습하는 값. 예: convolution weight
- **hyperparameter:** 사람이 실험 전에 정하는 설정. 예: learning rate, batch size, epoch 수, LoRA rank

두 단어를 혼동하지 말아야 한다. “모델이 100M parameters를 가진다”는 것은 학습되는 숫자가 약 1억 개라는 뜻이지, 연구자가 설정 1억 개를 직접 고른다는 뜻이 아니다.

### Parameter 수와 GPU memory

FP32 parameter 하나는 보통 4 bytes, FP16은 2 bytes다. 그러나 학습할 때는 parameter만 저장하는 것이 아니다. Gradient, optimizer state, activation도 필요하다. 따라서 “100M parameters이니 FP16으로 200 MB면 된다”라고 계산하면 실제 학습 VRAM을 크게 과소평가한다.

## 4.4 특징 — Feature

**한 줄 정의:** 모델이 입력에서 뽑아낸 유용한 정보 표현이다.

Feature는 사람이 “이 채널은 귀를 의미한다”라고 직접 정하지 않는다. 학습 과정에서 문제 해결에 유용하도록 만들어진다.

- 낮은 층 feature: 색 변화, 가로선, 세로선, 단순 질감
- 중간 층 feature: 원, 바퀴, 눈, 털무늬 같은 부분
- 높은 층 feature: 물체 전체, 의미, 대상의 정체성

Feature는 보통 feature map 또는 feature vector 형태다.

- **feature map:** 공간 위치가 남아 있는 H×W×C tensor. 분할처럼 “어디인가”가 중요한 작업에 유리하다.
- **feature vector:** 공간 정보를 요약한 긴 숫자 목록. 이미지 검색이나 분류에 자주 쓴다.

**원본 픽셀과 비교:** 픽셀은 센서가 기록한 색이고, feature는 모델이 목적에 맞게 변환한 표현이다.

## 4.5 Encoder, Backbone, Decoder

### Encoder

입력을 받아 중요한 feature로 바꾸는 부분이다. 이미지 encoder는 RGB 픽셀을 여러 해상도의 feature map으로 변환한다.

**비유:** 긴 교과서를 읽고 핵심 내용을 구조화된 필기로 바꾸는 학생.

### Backbone

모델의 주된 feature extractor를 가리킨다. 많은 논문에서 image encoder와 비슷한 뜻으로 쓰지만 완전히 같은 말은 아니다.

- backbone: 전체 시스템에서 핵심 시각 feature를 만드는 주 구조라는 역할 중심 표현
- encoder: 입력을 표현으로 변환한다는 기능 중심 표현

SAM 2에서 Hiera는 image encoder이자 visual backbone이다.

### Decoder

Encoder feature를 사용해 task에 맞는 출력을 만든다. Segmentation decoder는 낮아진 feature 해상도를 결합하고 확대해 픽셀 mask를 만든다.

**비유:** 요약 필기를 보고 질문의 형식에 맞는 답안지를 작성하는 학생.

같은 encoder feature라도 decoder를 바꾸면 분류, detection, segmentation처럼 다른 출력을 만들 수 있다.

### Skip connection과 multi-scale feature

Encoder 뒤쪽 feature는 의미는 강하지만 공간 해상도가 낮을 수 있다. 앞쪽 feature는 경계와 위치가 자세하지만 의미는 약하다. U-Net 같은 구조는 앞쪽과 뒤쪽 feature를 skip connection으로 연결한다. 여러 크기의 feature를 함께 쓰는 것을 multi-scale feature fusion이라고 한다.

작은 물체와 가는 경계를 개선하는 논문에서 자주 등장하는 이유다.

## 4.6 CNN — Convolutional Neural Network

**한 줄 정의:** 작은 필터 kernel를 이미지 위로 이동시키며 주변 패턴을 찾는 신경망이다.

예를 들어 3×3 kernel은 한 번에 주변 9개 픽셀을 보고 계산한다. 같은 kernel를 이미지 여러 위치에 반복 적용하므로 위치가 바뀌어도 비슷한 패턴을 찾기 쉽다.

**비유:** 작은 돋보기를 사진 위에서 한 칸씩 움직이며 “여기에 세로 경계가 있는가?”를 검사하는 것과 같다.

### CNN의 주요 용어

- **kernel 또는 filter:** 지역 패턴을 찾는 작은 weight 묶음
- **stride:** kernel가 한 번에 이동하는 칸 수
- **padding:** 가장자리도 처리하기 위해 주변에 값을 덧붙이는 것
- **pooling:** 가까운 값을 요약해 해상도를 줄이는 것
- **receptive field:** 한 출력 값이 참고할 수 있는 원본 영역
- **dilated convolution:** kernel 원소 사이 간격을 벌려 parameter를 크게 늘리지 않고 넓은 영역을 보는 convolution
- **depthwise convolution:** channel별로 convolution을 수행해 계산량을 줄이는 방식

### 장점과 한계

CNN은 가까운 픽셀의 관계와 질감, 경계를 잘 다루고 구조적 편향 inductive bias가 강하다. 반면 멀리 떨어진 두 영역의 관계를 직접 연결하려면 층을 많이 거쳐야 한다.

## 4.7 Transformer

**한 줄 정의:** 입력을 token들로 나타내고 attention을 이용해 token 사이 관계를 학습하는 구조다.

언어에서는 단어가 token이 될 수 있다. 이미지에서는 이미지를 작은 patch로 나누어 각 patch를 token으로 만들 수 있다. 영상에서는 patch와 시간까지 포함한 spatiotemporal token을 사용할 수 있다.

**비유:** CNN이 가까운 동네부터 차례로 조사한다면 Transformer는 회의실에서 모든 대표가 서로를 보며 “누구의 정보가 지금 중요한가?”를 판단하는 것과 같다.

### Vision Transformer, ViT

이미지를 patch로 나누고 각 patch를 vector로 바꾼 뒤 Transformer에 넣는 구조다. 예를 들어 224×224 이미지를 16×16 patch로 나누면 14×14, 즉 196개의 patch token이 생긴다.

### Hierarchical Transformer

처음에는 고해상도 작은 영역을 보고, 뒤로 갈수록 token 수를 줄이며 더 큰 의미를 보는 계층형 구조다. CNN의 multi-scale 장점과 Transformer의 관계 모델링을 결합하려는 설계다. SAM 2의 Hiera encoder가 여기에 해당한다.

## 4.8 Attention

**한 줄 정의:** 현재 계산에 어떤 정보가 얼마나 중요한지 가중치를 정해 모으는 방법이다.

Attention은 보통 Query, Key, Value 세 요소로 설명한다.

- **Query, Q:** 내가 지금 찾고 싶은 것
- **Key, K:** 각 정보가 무엇을 나타내는지 알려 주는 표지
- **Value, V:** 실제로 가져올 내용

**도서관 비유:** “강아지의 현재 모습”이라는 query로 책의 색인 key들을 비교하고, 관련성이 높은 책의 내용 value를 더 많이 읽는다.

### Self-attention과 cross-attention

- **self-attention:** 같은 집합 안의 token끼리 관계를 계산한다. 한 이미지의 patch들이 서로 참고한다.
- **cross-attention:** 서로 다른 집합 사이 관계를 계산한다. 현재 frame feature가 과거 memory feature를 참고하거나, image token이 text token을 참고한다.

SAM 2의 memory attention은 현재 frame이 과거 frame에 저장된 객체 정보를 선택적으로 읽는 cross-attention으로 이해할 수 있다.

### Attention의 한계

Token 수가 많으면 모든 쌍의 관계를 계산하는 비용이 커질 수 있다. 고해상도 이미지와 긴 영상에서 window attention, sparse attention, token pruning, memory compression 같은 효율화가 등장하는 이유다.

## 4.9 Embedding과 Token

### Embedding

**한 줄 정의:** 이미지, patch, 단어, prompt, 객체 같은 대상을 의미를 담은 숫자 vector로 바꾼 표현이다.

비슷한 의미나 모습은 embedding 공간에서 가까워지도록 학습할 수 있다. 첫 frame의 target crop embedding과 현재 후보 crop embedding을 비교하면 재등장한 객체를 찾는 데 사용할 수 있다.

### Token

**한 줄 정의:** Transformer가 처리하는 하나의 정보 단위다.

Token은 단순한 번호가 아니라 embedding vector로 변환되어 계산된다.

- language token: 단어 또는 부분 단어
- image token: image patch를 나타내는 vector
- prompt token: point, box, text prompt 정보
- object token 또는 object pointer: 특정 객체의 정체성을 요약한 vector
- memory token: 과거 frame에서 저장된 정보 단위

### Feature, embedding, token의 차이

세 용어는 겹쳐 쓰일 수 있지만 초점이 다르다.

- **feature:** 모델이 추출한 유용한 표현이라는 가장 넓은 말
- **embedding:** 대상을 고정 길이 vector 공간에 나타냈다는 점을 강조
- **token:** Transformer가 attention으로 처리하는 한 단위라는 점을 강조

한 image patch의 feature vector가 동시에 patch embedding이자 image token일 수 있다.

## 4.10 활성값 — Activation

신경망이 현재 입력을 처리하며 중간에 만들어 내는 값이다. Parameter는 여러 입력에 공통으로 쓰이는 장기적인 모델 지식이고, activation은 현재 사진이나 batch를 처리할 때만 생기는 중간 계산 결과다.

학습에서는 backward 계산을 위해 activation을 많이 보관하므로 frozen backbone이라도 GPU memory가 필요하다. “학습 parameter가 적다”와 “VRAM이 거의 들지 않는다”는 같은 말이 아니다.

## 4.11 모델의 세 가지 memory를 구분하기

논문에서 memory라는 단어는 서로 다른 뜻으로 쓰인다.

1. **Model parameter:** 학습되어 checkpoint에 저장되는 장기 지식
2. **GPU memory, VRAM:** 계산 중 tensor를 올려 두는 하드웨어 공간
3. **Video memory bank:** 이전 frame의 feature와 mask를 저장해 다음 frame이 참고하는 데이터 구조

SAM 2 연구에서 “memory contamination”은 대개 세 번째 의미다. 잘못 분할한 frame을 memory bank에 저장해 이후 예측까지 연쇄적으로 틀리는 현상을 뜻한다.

---

# 5. 모델은 어떻게 배우고 평가되는가

## 5.1 Dataset, Sample, Label

- **dataset:** 학습과 평가에 사용하는 데이터 전체
- **sample:** 이미지 한 장, 영상 clip 하나처럼 데이터 한 개
- **label:** 정답 class, box, mask, 문장 등
- **annotation:** 사람이 label을 만드는 작업 또는 만들어진 표시

Segmentation은 픽셀마다 정답을 표시해야 하므로 classification보다 annotation 비용이 훨씬 크다. Foundation model과 promptable model이 주목받는 이유 중 하나는 새로운 작업마다 모든 mask를 다시 그리는 비용을 줄일 가능성이 있기 때문이다.

## 5.2 Training, Validation, Test

### Training set

모델 parameter를 실제로 업데이트하는 데이터다. 학생이 문제집을 풀며 실력을 바꾸는 단계에 해당한다.

### Validation set

Learning rate, epoch 수, threshold, adapter 구조 같은 선택을 비교하고 조기 종료 early stopping 시점을 정하는 데이터다. 모의고사에 해당한다. Validation 결과를 보면서 설정을 계속 바꾸므로 최종 성능이라고 보기 어렵다.

### Test set

모든 설계를 확정한 뒤 마지막에 일반화 성능을 확인하는 데이터다. 기말고사에 해당한다. Test 결과를 반복해서 보고 구조를 바꾸면 사실상 test set에 맞춰진 것이므로 공정한 평가가 아니다.

### Data leakage

Test 정보가 training이나 model selection에 흘러 들어가는 문제다. 같은 영상의 연속 frames를 무작위로 train과 test에 나누면 거의 같은 장면이 양쪽에 들어갈 수 있다. 영상 연구에서는 video 단위, 환자 연구에서는 patient 단위로 분리해야 한다.

## 5.3 학습 한 단계

1. **Forward pass:** 입력을 모델에 넣어 prediction을 계산한다.
2. **Loss computation:** prediction과 ground truth의 차이를 loss로 계산한다.
3. **Backward pass, backpropagation:** 각 parameter가 loss에 얼마나 영향을 주었는지 gradient를 계산한다.
4. **Optimizer step:** gradient를 이용해 parameter를 조금 수정한다.
5. 이 과정을 여러 batch와 epoch에 걸쳐 반복한다.

**비유:** 문제를 풀고, 틀린 정도를 확인하고, 어떤 생각 때문에 틀렸는지 되짚고, 풀이 습관을 조금 바꾸는 과정이다.

## 5.4 Loss

**한 줄 정의:** 학습 중 prediction이 정답과 얼마나 다른지 나타내는 미분 가능한 벌점이다.

Loss가 작아지는 방향으로 parameter를 업데이트한다. Loss와 평가 metric은 목적이 다르다.

- loss: 학습을 움직이는 신호
- metric: 사람이 결과를 해석하고 비교하는 점수

### Segmentation에서 자주 쓰는 loss

- **Binary Cross-Entropy, BCE:** 각 픽셀이 target인지 아닌지의 확률을 평가
- **Cross-Entropy, CE:** 여러 class 중 정답 class의 확률을 평가
- **Dice loss:** prediction과 GT 영역의 겹침을 직접 높이는 데 유리
- **IoU loss:** intersection-over-union을 학습 목표로 사용
- **Focal loss:** 쉬운 배경보다 어려운 픽셀과 작은 대상에 더 집중
- **Boundary loss:** 객체 경계 근처의 오차를 더 중요하게 처리

여러 loss를 가중합해 쓰기도 한다. 예를 들어 total loss = BCE + Dice + 0.2 × Boundary loss처럼 구성할 수 있다. 가중치는 hyperparameter다.

**주의:** Training loss가 낮아져도 test 성능이 반드시 좋아지는 것은 아니다. 모델이 training data를 외운 과적합일 수 있다.

## 5.5 Gradient, Optimizer, Learning Rate

- **gradient:** parameter를 어느 방향으로 바꾸면 loss가 커지거나 작아지는지 나타내는 기울기
- **optimizer:** gradient를 이용해 parameter를 업데이트하는 규칙. SGD, AdamW 등이 대표적
- **learning rate:** 한 번에 parameter를 얼마나 크게 바꿀지 정하는 값

Learning rate가 너무 크면 정답 주변을 계속 뛰어넘고, 너무 작으면 학습이 매우 느리거나 나쁜 지점에 머물 수 있다.

## 5.6 Batch, Iteration, Epoch

- **batch:** 한 번의 forward와 backward에 함께 넣는 sample 묶음
- **batch size:** 그 묶음의 sample 수
- **iteration 또는 step:** optimizer가 한 번 업데이트된 횟수
- **epoch:** training set 전체를 한 번 본 것

영상은 frame 수와 해상도 때문에 memory를 많이 사용하므로 batch size 1이나 짧은 clip을 쓰는 경우도 흔하다.

## 5.7 Checkpoint

**한 줄 정의:** 특정 시점의 모델 parameter와 필요하면 optimizer 상태 등을 파일로 저장한 것이다.

Checkpoint는 게임의 저장 지점과 비슷하다. 학습 중 가장 좋은 validation 성능의 checkpoint를 저장하고 나중에 inference하거나 fine-tuning할 수 있다.

**Model architecture와 checkpoint의 차이**

- architecture: 층이 어떻게 연결되는지에 대한 설계도
- checkpoint: 그 설계도의 parameter에 들어 있는 실제 학습 값

같은 Hiera-B+ architecture라도 어떤 데이터로 학습한 checkpoint인지에 따라 성능이 달라질 수 있다.

## 5.8 Pretraining

**한 줄 정의:** 특정 작은 과제를 풀기 전에 크고 다양한 데이터로 일반적인 표현을 먼저 학습하는 것.

**비유:** 의대 전공 공부 전에 생물과 화학의 기초를 폭넓게 배우는 것과 같다.

ImageNet class로 pretraining하거나, mask data로 segmentation pretraining하거나, label 없이 self-supervised pretraining할 수 있다. SAM 2는 대규모 이미지와 영상 segmentation data에서 일반적인 promptable segmentation 능력을 배운 foundation model이다.

## 5.9 Fine-tuning

**한 줄 정의:** pretrained model을 새로운 dataset이나 task에 맞게 추가 학습하는 것.

- **full fine-tuning:** 대부분 또는 모든 parameter를 업데이트
- **partial fine-tuning:** decoder나 일부 stage만 업데이트
- **parameter-efficient fine-tuning:** adapter나 LoRA처럼 적은 parameter만 업데이트

Fine-tuning은 domain 성능을 높일 수 있지만 데이터가 적으면 과적합하고 원래의 범용 능력을 잃을 수 있다. 이를 catastrophic forgetting이라고 부르기도 한다.

## 5.10 Frozen

**한 줄 정의:** 해당 모듈의 parameter를 학습 중 업데이트하지 않도록 고정한 상태.

“Frozen image encoder”는 encoder를 forward 계산에는 사용하지만 gradient로 weight를 바꾸지 않는다는 뜻이다. Encoder 계산 자체가 사라지는 것은 아니다.

### Frozen과 no gradient의 실무적 의미

Parameter update를 막으면 optimizer state와 parameter gradient가 줄어 memory를 아낄 수 있다. 그러나 trainable adapter에 encoder feature가 필요하고 계산 그래프를 유지해야 하면 activation memory는 여전히 들 수 있다. 구현 위치에 따라 절감 폭이 다르다.

## 5.11 Inference

**한 줄 정의:** 학습된 모델을 실제 입력에 사용해 prediction을 만드는 과정이다.

Inference에서는 일반적으로 backward와 optimizer update를 하지 않는다. 그래서 training보다 memory 사용량이 적고 빠르다. 논문의 FPS가 training 속도인지 inference 속도인지 확인해야 한다.

### Offline과 online 또는 streaming

- **offline:** 전체 영상을 미리 보고 앞뒤 정보를 모두 사용할 수 있음
- **online 또는 streaming:** 미래 frame을 보지 않고 도착하는 순서대로 처리

Long video에서 전체 결과를 본 뒤 최적 경로를 고르는 방법은 정확하더라도 실시간 카메라에는 바로 사용할 수 없다.

## 5.12 Overfitting과 Generalization

- **overfitting:** training data에서는 매우 잘하지만 새로운 data에서는 성능이 떨어지는 현상
- **generalization:** 보지 못한 sample, 장면, domain에서도 배운 규칙이 작동하는 능력

Data augmentation, regularization, 더 다양한 data, parameter 수 제한, early stopping 등이 과적합 완화에 쓰인다.

## 5.13 지도학습, 자기지도학습, 약지도학습

- **supervised learning:** 사람이 만든 label을 정답으로 학습
- **self-supervised learning:** data 자체에서 학습 문제를 만들어 label 없이 표현을 학습
- **weakly supervised learning:** 정확한 mask 대신 image label, point, scribble, box처럼 더 약한 annotation으로 학습
- **unsupervised learning:** 명시적 정답 없이 구조를 찾음

“Unsupervised VOS”라는 task 이름과 “모델 pretraining이 완전히 label-free”라는 주장은 같은 뜻이 아니다. Task의 입력 조건과 모델이 과거에 어떻게 학습되었는지를 따로 확인해야 한다.

---

# 6. Prompt, Zero-shot, Foundation Model

## 6.1 Prompt

**한 줄 정의:** 모델에게 무엇을 원하는지 지정해 주는 입력 신호다.

언어 모델에서는 문장이 prompt이고, segmentation model에서는 point, box, mask, text, reference image 등이 prompt가 될 수 있다.

### SAM 계열의 대표 visual prompt

- **positive point:** “이 픽셀은 target 안쪽이다”라고 알려 주는 클릭
- **negative point:** “이 픽셀은 target이 아니다”라고 알려 주는 클릭
- **bounding box:** “이 사각형 안의 대상에 집중하라”는 위치 힌트
- **mask prompt:** 이전 mask 또는 대략 그린 영역을 더 정밀하게 수정하기 위한 힌트

**구체 예시:** 두 마리 강아지가 붙어 있는 사진에서 왼쪽 강아지 몸 안에 positive point를 찍고 오른쪽 강아지에 negative point를 찍으면 어느 개체를 원하는지 더 분명해진다.

### Text prompt

“빨간 우산을 든 사람”처럼 자연어로 target을 지정한다. SAM 2 기본 모델은 본래 class name이나 문장을 직접 이해하는 open-vocabulary detector가 아니다. Text를 사용하려면 Grounding DINO, CLIP, vision-language model 같은 외부 모듈이 box나 feature prompt를 만들어 주거나, 별도의 text adapter를 학습하는 경우가 많다.

### Prompt encoder

Point와 box 같은 prompt를 embedding으로 바꾸는 모듈이다. Image feature와 결합할 수 있는 숫자 표현으로 변환한다.

### Prompt quality가 중요한 이유

Model이 좋아도 target 밖의 point나 너무 큰 box를 주면 잘못된 mask가 나올 수 있다. 논문 비교에서는 다음을 반드시 같게 맞춰야 한다.

- point 개수와 위치를 사람이 정했는지, GT에서 자동 생성했는지
- box가 GT box인지 detector가 예측한 box인지
- 첫 frame만 prompt했는지 중간 frame에도 correction했는지
- 여러 mask 후보 중 어떤 기준으로 하나를 골랐는지

GT mask에서 만든 완벽한 box와 실제 detector box를 비교하면 공정하지 않다.

## 6.2 Prompt-free와 Automatic Prompting

- **prompt-free:** 사용자가 직접 prompt를 주지 않아도 모델이 target mask를 생성하는 설정
- **automatic prompting:** detector, support image, 다른 model, 이전 prediction 등이 point나 box를 자동 생성

두 용어는 같은 듯 보이지만 다르다. 사용자 입력이 없더라도 내부 detector가 target-domain label로 학습되었다면 전체 시스템이 완전히 zero-shot인 것은 아니다.

SAM 2의 Automatic Mask Generator는 격자 형태로 많은 point prompt를 넣고 겹치는 mask를 필터링해 여러 객체 mask를 만든다. 이는 text를 이해해 원하는 class를 찾는 object detector와는 다르다.

## 6.3 Zero-shot

**한 줄 정의:** 평가하려는 새로운 class, dataset 또는 task의 labeled example로 추가 학습하지 않고 수행하는 설정.

Zero-shot은 문맥에 따라 범위가 달라 반드시 “무엇에 대해 zero-shot인가?”를 확인해야 한다.

- **zero-shot class generalization:** 학습 중 보지 않은 class를 처리
- **zero-shot transfer:** target dataset에서 fine-tuning하지 않고 바로 평가
- **zero-shot mask generation:** target-domain mask label 없이 detector box를 SAM에 입력
- **zero-shot tracking:** 첫 frame prompt는 받지만 해당 tracking dataset으로 재학습하지 않음

**중요한 오해**

- Zero-shot이라고 pretraining이 없다는 뜻은 아니다. 대규모 data로 이미 학습된 model을 쓴다.
- Zero-shot이라고 prompt가 없다는 뜻도 아니다. 첫 frame mask나 text prompt를 받을 수 있다.
- System 일부가 target data로 학습되었다면 전체 pipeline의 zero-shot 범위를 정확히 밝혀야 한다.

예를 들어 SAR 영상으로 학습한 detector가 ship box를 찾고 pretrained SAM 2가 mask를 만든다면 “mask annotation에 대해서는 zero-shot”일 수 있지만 “SAR target domain에 대해 완전히 zero-shot”이라고 말하기는 어렵다.

## 6.4 One-shot, Few-shot, Support Set

- **one-shot:** class 또는 task마다 labeled example 하나를 사용
- **few-shot:** 매우 적은 labeled examples를 사용
- **support set:** 새 target을 설명하기 위해 제공되는 소수의 example 묶음
- **query:** support를 참고해 prediction해야 하는 새 입력

의료 영상에서 비슷한 장기의 image-mask 한 쌍을 SAM 2 memory에 넣고 새 환자를 분할한다면 support-guided few-shot 방식으로 볼 수 있다.

“첫 frame mask 한 장을 받는 semi-supervised VOS”를 무조건 one-shot learning이라고 부르지는 않는다. 그 mask는 추적할 instance를 지정하는 task input이고, model parameter를 학습시키는 example이 아닐 수 있기 때문이다.

## 6.5 Foundation Model

**한 줄 정의:** 크고 다양한 data에서 사전학습되어 여러 task와 domain에 재사용할 수 있는 범용 기반 model.

**비유:** 한 메뉴만 만드는 전용 기계가 아니라 다양한 요리의 기초 재료와 조리법을 갖춘 큰 주방이다. 새 요리에 맞는 작은 조정이나 prompt가 여전히 필요할 수 있다.

Foundation model의 일반적 특징은 다음과 같다.

- 매우 크고 다양한 pretraining data
- 여러 downstream task에 transfer 가능
- prompt 또는 작은 fine-tuning으로 활용
- 같은 backbone을 다양한 domain에서 재사용

### Segmentation Foundation Model

점이나 box prompt에 따라 다양한 객체를 분할하도록 대규모 mask data에서 학습된 model이다. SAM과 SAM 2가 대표적이다.

### Foundation이라는 말이 “완벽함”을 뜻하지는 않는다

대규모 pretraining을 해도 다음 상황에서 실패할 수 있다.

- 의료, 수중, SAR처럼 일반 사진과 다른 domain
- 매우 작거나 가는 객체
- 낮은 contrast와 어두운 장면
- 긴 가림 뒤 재등장
- 비슷한 distractor가 많은 영상
- 부정확한 prompt

이 틈을 domain adaptation, memory control, boundary refinement, robustness 연구가 다룬다.

## 6.6 Generalist와 Specialist

- **generalist model:** 다양한 class와 scene을 폭넓게 처리
- **specialist model:** 특정 domain이나 task에서 높은 성능을 목표로 함

SAM 2는 generalist에 가깝고, 의료 병변 전용으로 fine-tune한 모델은 specialist에 가깝다. Specialist가 해당 dataset에서 더 정확하더라도 다른 domain의 범용성이 떨어질 수 있다. 연구에서는 target-domain 성능과 원래 generalization을 함께 보는 것이 좋다.

## 6.7 Open-vocabulary와 Open-world

- **closed-set:** 학습 때 정한 class 목록 안에서만 예측
- **open-vocabulary:** text embedding 등을 이용해 학습 class 목록 밖의 새로운 이름도 query로 사용
- **open-world:** 모르는 class의 존재를 인식하고 새 class를 점차 배울 수 있는 더 넓은 설정

SAM 2는 “아무 물체나 point로 지정해 mask를 만들 수 있다”는 의미에서 category-agnostic이지만, 기본 모델 자체가 class name을 출력하는 open-vocabulary classifier는 아니다.

---

# 7. SAM 2를 부품별로 이해하기

## 7.1 SAM과 SAM 2의 역할

SAM, Segment Anything Model은 이미지에서 point, box, mask prompt로 객체를 분할하는 foundation model이다. SAM 2, Segment Anything in Images and Videos는 이 promptable segmentation 능력을 영상으로 확장하고, image와 video를 하나의 구조에서 처리한다.

SAM 2의 핵심은 과거 frame의 객체 정보를 memory에 저장하고 현재 frame이 이를 읽도록 한 것이다. 단순히 매 frame에 SAM을 독립적으로 실행하는 것과 다르다.

## 7.2 전체 흐름

1. **Frame input:** 현재 RGB frame을 입력한다.
2. **Image encoder 또는 Hiera backbone:** 픽셀을 multi-scale visual feature로 바꾼다.
3. **Prompt encoder:** point, box, mask prompt를 embedding으로 바꾼다.
4. **Memory attention:** 현재 frame feature가 과거 memory feature와 object 정보를 참고한다.
5. **Mask decoder:** prompt와 feature를 결합해 하나 이상의 mask 후보와 품질 관련 score를 만든다.
6. **Memory encoder:** 선택된 mask와 frame feature를 다음 frame에서 사용할 memory 표현으로 만든다.
7. 다음 frame에서 같은 과정을 반복하며 mask를 전파한다.

## 7.3 Image encoder와 Hiera

Hiera는 계층형 Transformer backbone이다. 앞 stage는 비교적 높은 해상도의 지역 정보를, 뒤 stage는 더 낮은 해상도에서 넓은 문맥과 의미를 담는다.

SAM 2 checkpoint 이름의 Tiny, Small, Base Plus, Large는 대체로 backbone 규모 차이를 나타낸다. 큰 모델이 항상 모든 dataset에서 최고인 것은 아니다. 작은 모델이 더 빠르고, 특정 domain에서는 prompt나 memory quality가 모델 크기보다 큰 영향을 줄 수 있다.

## 7.4 Prompt encoder

Point의 위치와 positive 또는 negative label, box의 두 모서리, mask 힌트를 embedding으로 바꾼다. 이 embedding이 mask decoder에 “어느 대상에 집중할지” 알려 준다.

Prompt encoder를 frozen하고 image encoder나 decoder만 fine-tune하는 논문이 많다. 새로운 domain에서도 point와 box라는 prompt 형식 자체는 변하지 않는다고 보기 때문이다.

## 7.5 Memory attention

현재 frame에서 target이 어디인지 판단할 때 과거에 저장된 target feature를 읽는다. 이전 frame과 현재 frame이 완전히 같지는 않으므로 attention이 관련 있는 memory token에 더 높은 가중치를 준다.

Memory가 도움을 주는 예:

- target이 조금 이동해도 이전 모습과 연결
- 자세가 변할 때 최근 appearance 참고
- 사용자가 중간 frame에서 수정한 mask 반영

Memory가 해가 되는 예:

- 잘못된 mask를 저장해 오류가 반복
- target이 사라졌는데 distractor mask를 저장
- 최근 frame만 남겨 처음의 정확한 identity를 잊음
- 오래된 appearance만 믿어 큰 변화를 따라가지 못함

따라서 memory 개선은 다음 네 문제로 나눌 수 있다.

- **admission 또는 write policy:** 어떤 frame을 저장할 것인가?
- **retention 또는 eviction policy:** 무엇을 오래 남기고 무엇을 지울 것인가?
- **retrieval 또는 read policy:** 현재 frame에서 어떤 memory를 읽을 것인가?
- **fusion:** short-term, long-term, motion, semantic memory를 어떻게 결합할 것인가?

## 7.6 Mask decoder

Image feature, prompt embedding, memory 정보를 사용해 pixel mask를 만든다. 모호한 prompt에서는 여러 mask 후보를 만들 수 있고, predicted IoU 같은 품질 score로 후보를 고를 수 있다.

Decoder 출력 해상도가 원본보다 낮으면 마지막에 interpolation으로 확대한다. 이때 머리카락, 혈관, 가는 다리, 작은 물체 경계가 뭉개질 수 있다. High-resolution feature fusion, boundary head, refinement decoder가 이를 개선한다.

## 7.7 Object score, Predicted IoU, Confidence, Uncertainty

이 용어들은 비슷하지만 같은 값이 아니다.

- **predicted IoU:** 모델이 자신의 mask가 좋은 품질일 것이라고 예측한 값
- **object presence 또는 object score:** target이 현재 frame에 존재하는지 나타내는 신호
- **classification confidence:** 특정 class일 확률 또는 score
- **uncertainty:** 모델이 예측을 얼마나 확신하지 못하는지 나타내는 개념

Confidence가 높다고 반드시 맞는 것은 아니다. 처음 보는 domain에서 틀린 mask를 매우 자신 있게 낼 수도 있다. 이를 miscalibration이라고 한다. 따라서 연구에서는 score 하나만 믿기보다 area 변화, motion consistency, appearance similarity, 후보 간 disagreement를 함께 보는 것이 좋다.

## 7.8 Object Pointer

SAM 2가 target 객체의 의미와 정체성을 요약해 다음 frame으로 전달하는 vector 표현이다. Pixel mask 전체와 달리 compact한 object-level token이라고 생각할 수 있다.

연속 frame 사이 object pointer가 갑자기 크게 변하면 target이 다른 객체로 바뀌었거나 appearance가 급격히 변한 신호일 수 있다. 다만 정상적인 자세 변화도 pointer를 바꿀 수 있으므로 단독 오류 판정은 위험하다.

## 7.9 SAM 2가 자동 detector와 다른 점

SAM 2는 기본적으로 “어떤 물체를 분할하라”는 prompt가 있을 때 강하다. 일반 object detector처럼 처음부터 class name과 모든 객체의 box를 반드시 출력하는 모델이 아니다.

전체 장면에서 모든 mask를 얻으려면 grid points를 넣는 Automatic Mask Generator나 외부 detector를 사용할 수 있다. 이때 전체 system 성능은 prompt generator와 mask filtering 품질에도 크게 좌우된다.

## 7.10 SAM 2의 개선 지점 지도

| 개선 위치 | 고칠 수 있는 문제 | 간단한 연구 예 |
|---|---|---|
| 입력 또는 preprocessing | 저조도, 색 왜곡, noise | 밝기 보정 전후 강건성 비교 |
| Image encoder | domain feature 부족 | adapter 또는 LoRA |
| Prompt 생성 | 부정확한 click과 box | uncertainty 기반 자동 재프롬프트 |
| Memory write | 오류 전파 | 신뢰도 낮은 frame 저장 금지 |
| Memory retrieval | 긴 영상 비용과 distractor | top-k salient memory 선택 |
| Motion module | 빠른 이동과 가림 | optical flow 또는 Kalman gate |
| Mask decoder | 작은 객체와 경계 | multi-scale boundary refinement |
| Output control | target 부재 | visible, uncertain, absent 상태기계 |
| 경량화 | 낮은 FPS와 VRAM | quantization, pruning, distillation |

---

# 8. 새 도메인에 모델을 적응시키는 방법

## 8.1 Domain

**한 줄 정의:** 데이터가 만들어지는 환경과 분포의 묶음이다.

사진 속 물체 class가 같아도 촬영 장비, 조명, 배경, 해상도, 색, noise가 달라지면 domain이 달라질 수 있다.

예:

- 일반 스마트폰 사진과 내시경 영상
- 맑은 낮 도로와 야간 안개 도로
- 자연광 RGB 사진과 SAR 위성 영상
- 수면 위 사진과 수중 카메라 영상
- 병원 A의 CT와 병원 B의 CT

## 8.2 Domain Shift

학습 data와 실제 사용 data의 분포가 달라지는 현상이다.

사람은 “둘 다 물고기”라고 쉽게 알아도, 일반 사진에서 학습한 모델은 수중의 푸른 색조, 부유물, 흐린 경계 때문에 feature가 달라져 실패할 수 있다.

Domain shift는 입력 색만 달라지는 단순 문제가 아니다.

- **appearance shift:** 색, 밝기, 질감, noise 변화
- **sensor shift:** RGB, depth, 열화상, SAR 등 장비 차이
- **class 또는 label shift:** 등장 class 비율이나 정의가 다름
- **geometric shift:** 항공 영상처럼 시점과 scale이 다름
- **temporal shift:** frame rate, motion blur, 장면 변화가 다름

## 8.3 Source Domain과 Target Domain

- **source domain:** 원래 학습에 사용한 환경
- **target domain:** 새로 적용하려는 환경

예를 들어 일반 자연 이미지로 사전학습한 SAM 2를 복부 CT에 적용한다면, 일반 이미지가 source에 가깝고 CT가 target domain이다.

## 8.4 Domain Adaptation

**한 줄 정의:** Source에서 학습한 모델이 target domain에서도 잘 작동하도록 차이를 줄이는 방법.

### Target label 사용 정도에 따른 구분

- **supervised domain adaptation:** target의 정답 label을 충분히 사용
- **semi-supervised domain adaptation:** target label 일부와 unlabeled data를 함께 사용
- **unsupervised domain adaptation, UDA:** source label과 unlabeled target data를 사용
- **source-free domain adaptation:** source 원본 data 없이 pretrained source model과 target data만 사용
- **test-time adaptation, TTA:** 실제 test 입력이 들어올 때 일부 parameter나 통계를 즉석에서 조정

### Domain generalization과 비교

- **domain adaptation:** target domain data를 어떤 형태로든 미리 보며 적응
- **domain generalization:** 특정 target을 미리 보지 않고 여러 source에서 일반화 능력을 높임

논문에서 target test image 자체로 online update한다면 일반적인 offline fine-tuning보다 test-time adaptation에 가깝다.

## 8.5 Transfer Learning

**한 줄 정의:** 한 data나 task에서 배운 지식을 다른 data나 task에 재사용하는 큰 개념.

Pretrained encoder를 의료 segmentation에 가져오는 것은 transfer learning이다. Domain adaptation은 transfer learning 중에서도 source와 target의 분포 차이를 명시적으로 다루는 경우가 많다.

## 8.6 Full Fine-tuning과 Partial Fine-tuning

- **full fine-tuning:** backbone, decoder 등 대부분의 parameter를 업데이트
- **partial fine-tuning:** decoder, 마지막 stage 등 일부 기존 parameter만 업데이트

Full fine-tuning은 target 성능을 크게 높일 가능성이 있지만 training VRAM과 data가 많이 필요하고 범용 능력을 잃을 수 있다. Partial fine-tuning은 더 싸고 안정적이지만 domain gap이 매우 크면 부족할 수 있다.

## 8.7 PEFT — Parameter-Efficient Fine-Tuning

**한 줄 정의:** Pretrained model 대부분을 frozen한 채, 적은 수의 parameter만 학습해 새 task나 domain에 적응시키는 방법의 총칭이다.

PEFT는 특정 구조 하나의 이름이 아니다. Adapter, LoRA, prompt tuning, bias-only tuning 등이 PEFT에 포함된다.

### PEFT가 유리한 이유

- Trainable parameter와 optimizer state가 적다.
- 여러 domain마다 작은 추가 weight만 저장할 수 있다.
- 원래 pretrained weight를 덜 훼손한다.
- 작은 dataset에서 full fine-tuning보다 과적합 위험을 줄일 수 있다.

### PEFT가 해결하지 못하는 것

- Frozen backbone의 forward 계산은 그대로 필요하다.
- 고해상도 activation과 video memory 때문에 VRAM이 여전히 클 수 있다.
- Parameter가 적다고 항상 빠른 것은 아니다.
- Domain gap이 매우 크면 작은 모듈만으로 부족할 수 있다.

따라서 논문에서 “efficient”라고 하면 trainable parameter, 저장 용량, training time, inference latency, VRAM 중 무엇이 효율적인지 따로 확인해야 한다.

## 8.8 Adapter

**한 줄 정의:** Pretrained network 사이에 끼워 넣어 새 정보만 학습하는 작은 추가 모듈.

대표적인 bottleneck adapter는 다음처럼 작동한다.

1. 큰 feature dimension을 작은 dimension으로 줄인다.
2. 작은 공간에서 비선형 변환을 한다.
3. 원래 dimension으로 다시 늘린다.
4. 결과를 원래 feature에 residual로 더한다.

**비유:** 기존 대형 공장을 뜯지 않고 생산 라인 사이에 특정 제품용 소형 가공 장치를 붙이는 것과 같다.

### Adapter의 종류

- **MLP adapter:** fully connected layer로 channel 정보를 변환
- **convolutional adapter:** 지역 경계와 질감을 다루기 위해 convolution 사용
- **depthwise-dilated adapter:** 적은 parameter로 여러 scale의 지역 정보를 수집
- **stage-specific adapter:** backbone 해상도 stage마다 별도 adapter 사용
- **shared adapter:** 여러 stage가 같은 weight를 공유
- **temporal adapter:** 여러 frame 관계를 처리
- **deformable convolution adapter:** 물체 모양에 맞게 sampling 위치를 학습

### 어디에 넣는가가 중요하다

앞 stage에 넣으면 색과 texture 같은 낮은 수준 domain shift에 영향을 주기 쉽고, 뒤 stage에 넣으면 의미 feature를 바꾸기 쉽다. 모든 stage에 넣으면 표현력은 커지지만 parameter와 계산도 늘어난다.

## 8.9 LoRA — Low-Rank Adaptation

**한 줄 정의:** 큰 weight 행렬 자체는 frozen하고, 그 변화량을 작은 두 행렬의 곱으로 학습하는 PEFT 방법.

원래 weight를 W라고 할 때 실제 계산에는 W와 작은 변화량 ΔW를 함께 사용한다. LoRA는 ΔW를 B×A처럼 rank가 낮은 두 행렬로 표현한다. Rank r이 원래 차원보다 훨씬 작으므로 trainable parameter가 줄어든다.

**비유:** 거대한 사전을 통째로 다시 쓰지 않고, 필요한 수정사항만 얇은 정정표 두 장의 조합으로 보관하는 것과 같다.

### Rank

LoRA의 rank는 변화량이 표현할 수 있는 방향의 수를 결정한다. Rank가 너무 작으면 적응 능력이 부족하고, 너무 크면 parameter 절감 장점이 줄어든다.

### LoRA를 넣는 위치

Transformer의 query, key, value projection이나 output projection, MLP weight 등에 넣을 수 있다. 어느 층과 projection에 넣었는지에 따라 결과가 다르므로 “LoRA 사용”만으로 실험을 충분히 설명할 수 없다.

### Merge

학습 후 LoRA 변화량을 원래 weight에 합칠 수 있는 구조에서는 inference 때 별도 branch 계산을 줄일 수 있다. 다만 모든 구현과 quantized model에서 항상 완전히 무료인 것은 아니다.

## 8.10 Adapter와 LoRA 비교

| 항목 | Adapter | LoRA |
|---|---|---|
| 바꾸는 방식 | 새 작은 layer를 feature 경로에 추가 | 기존 weight의 저랭크 변화량을 추가 |
| 비선형성 | activation과 convolution 등을 자유롭게 사용 가능 | 기본 형태는 linear weight update |
| 지역 경계 처리 | convolutional adapter로 직접 설계 가능 | 주로 projection weight 적응 |
| inference overhead | 새 layer 때문에 생길 수 있음 | merge 가능하면 매우 작을 수 있음 |
| 설계 선택 | 위치, kernel, bottleneck 등 다양 | rank, 대상 layer, scaling 등이 중요 |
| 공통점 | 원래 backbone 대부분을 frozen하는 PEFT |

어느 쪽이 항상 우월한 것은 아니다. 수중 색과 texture에는 convolutional adapter가, language 또는 attention projection 적응에는 LoRA가 유리할 수 있다. 공정한 비교를 하려면 trainable parameter 수, training data, prompt, learning rate를 최대한 맞춰야 한다.

## 8.11 Prompt Tuning과 Visual Prompt Tuning

Model 본체를 frozen하고 학습 가능한 prompt token을 입력에 추가하는 PEFT 방법이다. 사람이 주는 실제 point prompt와 구분해야 한다.

- **user prompt:** 사용자가 target을 지정하는 point, box, text
- **learnable prompt:** training으로 값이 업데이트되는 가상 token

둘 다 prompt라고 부르지만 하나는 task input이고 다른 하나는 trainable parameter다.

## 8.12 Online Learning과 Test-Time Update

Video를 처리하는 중에 새 frame이나 correction을 이용해 작은 모듈을 업데이트하는 방법이다.

장점은 현재 영상에 빠르게 적응할 수 있다는 것이다. 단점은 잘못된 pseudo-label로 학습하면 drift가 커지고, frame 순서에 따라 결과가 달라지며, inference 중 backward 때문에 지연이 생긴다는 것이다.

공정한 평가를 위해 다음을 밝혀야 한다.

- Test label을 직접 쓰는가?
- Model prediction을 pseudo-label로 쓰는가?
- 영상마다 parameter를 reset하는가?
- Update 시간이 FPS에 포함되는가?

---

# 9. 모델을 가볍게 하거나 세부 정보를 보강하는 방법

## 9.1 Knowledge Distillation

**한 줄 정의:** 크고 강한 teacher model의 지식을 작고 빠른 student model이 따라 배우게 하는 방법.

**비유:** 정답만 보는 것이 아니라 우수한 선배가 문제를 바라보는 방식과 중간 풀이까지 배우는 것.

### 무엇을 따라 배우는가

- **logit distillation:** teacher의 class score 또는 확률
- **feature distillation:** 중간 feature map
- **attention distillation:** 어떤 token이나 위치를 중요하게 본 패턴
- **mask distillation:** teacher의 soft mask 또는 boundary
- **memory distillation:** video memory representation과 attention output

Student는 GT loss와 distillation loss를 함께 사용할 수 있다.

### 한계

- Teacher inference까지 필요한 training 비용이 든다.
- Teacher의 오류와 편향도 전달될 수 있다.
- 서로 feature shape가 다르면 projection module이 필요하다.
- 작은 student가 teacher의 모든 능력을 담을 수는 없다.

Distillation은 training을 싸게 만든다는 뜻이 아니라, 학습이 끝난 student의 inference를 가볍게 만드는 경우가 많다.

## 9.2 Quantization

**한 줄 정의:** Weight와 activation을 더 낮은 정밀도의 숫자로 표현해 memory와 연산을 줄이는 방법.

대표 형식:

- FP32: 32-bit floating point
- FP16 또는 BF16: 16-bit floating point
- INT8: 8-bit integer
- INT4: 4-bit integer

**비유:** 1 mm 단위 자 대신 1 cm 단위 자로 재는 것과 같다. 저장과 계산은 간단해지지만 매우 미세한 차이를 놓칠 수 있다.

### 방법

- **post-training quantization, PTQ:** 학습 완료 후 calibration data로 범위를 정하고 변환
- **quantization-aware training, QAT:** 학습 중 낮은 정밀도 오차를 흉내 내며 적응
- **weight-only quantization:** weight만 낮은 bit로 저장
- **weight-activation quantization:** weight와 activation을 모두 낮은 bit로 계산

### Segmentation에서 주의할 점

전체 IoU는 비슷해도 작은 객체, 얇은 구조, boundary confidence가 먼저 손상될 수 있다. Quantization 연구는 mIoU만이 아니라 boundary metric과 size별 metric을 확인해야 한다.

## 9.3 Pruning

**한 줄 정의:** 중요하지 않다고 판단한 weight, channel, attention head, token 또는 memory entry를 제거하거나 건너뛰는 방법.

### 종류

- **unstructured weight pruning:** weight 개별 원소를 0으로 만듦
- **structured pruning:** channel, layer, attention head처럼 묶음 단위로 제거
- **token pruning:** 중요하지 않은 image 또는 memory token을 계산에서 제외
- **dynamic pruning:** 입력마다 다른 token이나 경로를 선택
- **memory pruning:** 오래되거나 품질 낮은 frame과 token을 memory에서 제거

### Parameter 감소와 실제 속도

Weight를 많이 0으로 만들어도 GPU가 일반 dense 연산을 그대로 하면 실제 FPS가 거의 늘지 않을 수 있다. Structured pruning이나 hardware가 지원하는 sparsity가 실제 가속으로 이어지기 쉽다.

## 9.4 Token Compression과 Sparse Retrieval

Video model은 많은 frame의 고해상도 token을 memory에 저장하면 비용이 커진다. 이를 줄이는 방법은 다음과 같다.

- 평균 또는 attention pooling으로 token 수 축소
- target 주변 token만 유지
- 현재 frame과 관련도 높은 top-k memory만 읽기
- 배경 window는 작은 shortcut으로 처리
- uncertainty가 높을 때만 dense path 사용

속도와 정확도의 trade-off를 보여 주려면 top-k 값에 따른 J&F, latency, VRAM을 함께 그리는 것이 좋다.

## 9.5 Wavelet

**한 줄 정의:** 이미지를 위치 정보가 남아 있는 여러 주파수 대역과 scale로 분해하는 신호처리 방법.

2D wavelet transform은 흔히 다음 네 성분을 만든다.

- **LL:** 가로와 세로 모두 낮은 주파수. 큰 형태와 부드러운 영역
- **LH, HL:** 한 방향의 빠른 변화. 가로 또는 세로 경계
- **HH:** 두 방향의 높은 주파수. 가는 세부, 모서리, noise

**비유:** 사진을 “전체적인 색과 큰 형태” 악보와 “급격한 변화와 세부 장식” 악보로 나누어 보는 것.

### 왜 segmentation에 쓰는가

Encoder downsampling에서 사라지기 쉬운 경계와 texture를 high-frequency branch로 보존하고, semantic feature와 다시 결합할 수 있다.

### 한계

High-frequency에는 유용한 경계뿐 아니라 sensor noise와 압축 artifact도 들어 있다. 단순히 고주파를 모두 강화하면 오히려 거친 false boundary가 생길 수 있다. Attention이나 gate로 필요한 주파수만 선택하는 이유다.

### Fourier transform과 비교

Fourier transform은 전체 신호를 전역적인 주파수 성분으로 보는 데 강하고, wavelet은 위치와 scale 정보를 함께 유지하는 데 유리하다. 경계가 이미지 어디에 있는지 알아야 하는 segmentation에는 wavelet이 편리할 수 있다.

## 9.6 Optical Flow

**한 줄 정의:** 두 frame 사이에서 각 픽셀 또는 위치가 어느 방향으로 얼마나 이동했는지 나타내는 2D motion vector field.

각 픽셀에 (dx, dy)가 있다고 생각하면 된다. 이전 mask를 flow에 따라 이동시키는 것을 warping이라고 한다.

**비유:** 첫 사진의 작은 종이 조각마다 화살표를 붙여 다음 사진에서 어디로 움직였는지 표시한 지도.

### 활용

- 이전 mask를 현재 frame 위치로 warp해 prompt 생성
- motion consistency로 잘못된 mask 검사
- 움직이는 target과 정적 배경 구분
- camera motion 보정

### 한계

- 완전히 가려진 부분의 실제 motion은 관측할 수 없음
- 빠른 motion과 blur에서 matching 실패
- texture가 없는 영역과 반복 무늬에서 모호함
- 물체 경계에서 foreground와 background flow가 섞임
- 조명 변화가 실제 motion처럼 보일 수 있음

### Optical flow와 tracking 비교

Optical flow는 frame 사이의 지역적 motion field를 계산한다. Tracking은 장시간 같은 객체 identity를 유지하는 더 큰 문제다. Flow만으로 긴 가림 뒤 같은 객체를 다시 찾기는 어렵다.

## 9.7 Kalman Filter

**한 줄 정의:** 이전 위치와 속도를 바탕으로 다음 상태를 예측하고, 새 관측이 오면 예측을 수정하는 확률적 추정 방법.

잠깐 target mask가 사라졌을 때 box 중심과 크기를 예측하는 데 사용할 수 있다. 계산이 매우 가볍다는 장점이 있다.

**한계:** 일정한 속도 같은 단순 motion 가정이 흔하므로 급격한 방향 전환, camera cut, 긴 가림에서는 예측이 빗나갈 수 있다. Kalman box는 identity를 이해하지 못하므로 비슷한 distractor를 구분하려면 appearance feature가 필요하다.

## 9.8 Tiling

초고해상도 항공 영상을 작은 tile로 나누어 처리하는 방법이다. GPU memory를 줄이고 작은 객체가 resize 과정에서 사라지는 것을 막을 수 있다.

하지만 tile 경계에 걸친 객체가 두 조각으로 분리되거나, 넓은 문맥을 잃어 비슷한 구조를 혼동할 수 있다. Overlap, contextual padding, seam merging, adaptive tile size가 이를 보완한다.

## 9.9 Mixed Precision

FP32와 FP16 또는 BF16 계산을 섞어 속도와 memory를 줄이는 방법이다. Modern GPU의 tensor core를 활용할 수 있다. 아주 작은 gradient가 사라지는 underflow를 막기 위해 loss scaling을 쓰기도 한다.

Mixed precision은 model 구조를 바꾸는 compression과 다르다. 같은 model을 더 낮은 계산 정밀도로 실행하는 최적화다.

## 9.10 Gradient Accumulation과 Gradient Checkpointing

- **gradient accumulation:** 작은 batch를 여러 번 처리한 gradient를 모아 큰 effective batch처럼 update
- **gradient checkpointing 또는 activation checkpointing:** 일부 activation을 저장하지 않고 backward 때 다시 계산해 VRAM 절약

둘 다 8 GB GPU에서 학습 가능성을 높이지만 training time은 늘 수 있다. 이름에 checkpoint가 들어가도 “모델 저장 파일”과는 다른 개념이다.

## 9.11 경량화 기법 한눈에 비교

| 방법 | 줄이는 대상 | 보통 필요한 것 | 주의점 |
|---|---|---|---|
| PEFT | 학습되는 parameter | target data 학습 | inference가 반드시 빨라지지는 않음 |
| Distillation | 최종 student 크기 | teacher와 재학습 | teacher 오류 전달 |
| Quantization | 숫자 bit 수 | calibration 또는 QAT | 작은 객체와 경계 손상 가능 |
| Structured pruning | channel, layer, head | pruning과 보통 재학습 | 무엇을 지울지 어려움 |
| Token pruning | 입력별 token 계산 | saliency 또는 gate | target token을 잘못 버릴 위험 |
| Memory pruning | 저장하고 읽는 과거 정보 | quality와 relevance score | 재등장 단서까지 지울 위험 |
| Mixed precision | 연산 정밀도 | 지원 GPU와 안정화 | 일부 연산은 FP32 유지 필요 |
| Tiling | 한 번에 처리할 공간 크기 | merge 규칙 | seam과 global context 문제 |

---

# 10. 성능은 무엇으로 측정하는가

“성능 performance”은 정확도 하나만 뜻하지 않는다. 연구 질문에 따라 영역 정확도, 경계, 시간 일관성, 강건성, 속도, memory, 사용자 prompt 비용을 함께 봐야 한다.

## 10.1 IoU 또는 Jaccard Index, J

Prediction mask와 GT mask가 겹친 영역을 두 mask의 합집합 영역으로 나눈 값이다.

IoU = 겹친 픽셀 수 ÷ 둘 중 하나라도 포함된 픽셀 수

- 완전히 일치: 1
- 전혀 겹치지 않음: 0

VOS 논문에서는 region similarity를 J라고 표기한다.

**예시:** Prediction 100픽셀, GT 120픽셀, 겹침 80픽셀이면 합집합은 100+120−80=140이고 IoU는 약 0.571이다.

## 10.2 Dice 또는 DSC

Prediction과 GT가 겹친 영역을 두 mask 크기의 평균과 비교한다.

Dice = 2 × 겹침 ÷ (prediction 크기 + GT 크기)

같은 prediction에 대해 Dice는 보통 IoU보다 숫자가 크게 나온다. 따라서 “Dice 0.85”와 “IoU 0.85”를 같은 수준으로 보면 안 된다. 이진 mask에서는 다음 관계가 있다.

Dice = 2 × IoU ÷ (1 + IoU)

의료 segmentation 논문에서 DSC, Dice Similarity Coefficient를 많이 사용한다.

## 10.3 Boundary F-measure, F

Prediction boundary와 GT boundary가 허용 거리 안에서 얼마나 잘 맞는지 precision과 recall로 평가한다.

영역 대부분이 맞아도 털, 혈관, 가는 다리 경계가 뭉개지면 F가 낮을 수 있다. Boundary refinement 연구에는 IoU만보다 F, Boundary IoU, HD95, NSD 같은 경계 지표가 더 민감하다.

## 10.4 J&F

VOS에서 많이 쓰는 종합 지표로, region J와 boundary F의 평균이다.

J&F = (J + F) ÷ 2

J&F가 높아졌을 때 J와 F 중 어느 쪽이 개선되었는지도 따로 확인해야 한다.

## 10.5 mIoU

Mean Intersection over Union의 약자다. 여러 class, 객체, 영상 또는 dataset에서 IoU를 평균한 값이다.

“mean”의 대상이 class인지 frame인지 video인지 논문마다 다를 수 있다. Averaging protocol을 확인해야 한다. 큰 객체 픽셀을 모두 합쳐 계산하는 global IoU와 객체별 IoU 평균은 결과가 다르다.

## 10.6 Precision, Recall, F1

- **precision:** model이 positive라고 한 것 중 실제 positive 비율
- **recall:** 실제 positive 중 model이 찾아낸 비율
- **F1:** precision과 recall의 조화평균

작은 병변을 하나도 놓치지 않는 것이 중요하면 recall을, 잘못된 경보를 줄이는 것이 중요하면 precision을 중시할 수 있다.

Segmentation에서는 pixel 단위 precision과 boundary point 단위 precision이 모두 가능하므로 정의를 확인해야 한다.

## 10.7 AP와 mAP

Average Precision은 confidence threshold를 바꾸며 얻는 precision-recall 곡선을 요약한 값이다. Detection과 instance segmentation에 많이 쓴다.

mAP는 여러 class 또는 IoU threshold의 AP를 평균한다. COCO의 mAP와 AP50은 다르다.

- **AP50:** IoU 0.5 이상이면 match로 보는 비교적 느슨한 기준
- **COCO AP:** IoU 0.50부터 0.95까지 여러 threshold 평균, 경계와 위치에 더 엄격

Underwater 논문의 box-prompt mAP와 VOS 논문의 J&F는 다른 task와 protocol이므로 숫자 크기를 직접 비교할 수 없다.

## 10.8 Temporal Consistency

각 frame IoU가 좋아도 mask가 매 frame 흔들리면 영상으로 볼 때 품질이 나쁘다.

측정 예:

- optical-flow로 이전 mask를 warp한 뒤 현재 mask와 IoU
- boundary flicker
- mask area와 centroid의 비정상적인 jump
- 동일 객체의 identity switch 수
- frame별 성능의 표준편차

## 10.9 가림과 재등장 지표

장기 tracking 개선 연구라면 전체 J&F만으로 부족하다.

- **time-to-recovery:** target 재등장 후 IoU 0.5 등을 회복하는 데 걸린 frame 수
- **false reacquisition rate:** 재등장 때 distractor를 잘못 target으로 선택한 비율
- **target-absent false positive:** target이 없는데 mask를 출력한 비율
- **memory contamination rate:** 품질 낮은 prediction이 memory에 기록된 비율
- **failure detection AUROC 또는 AUPRC:** 실제 실패 frame을 risk score가 얼마나 잘 구분하는지

## 10.10 사용자 상호작용 비용

Promptable model에서는 accuracy뿐 아니라 사람이 얼마나 개입했는지도 성능이다.

- 영상당 point 또는 click 수
- correction이 필요한 frame 수
- 목표 품질에 도달할 때까지의 clicks
- annotation time
- 자동 correction 횟수와 false alarm

다섯 clicks 성능과 한 click 성능은 같은 조건으로 비교하면 안 된다.

## 10.11 Parameter, FLOPs, FPS, Latency, VRAM

- **Params:** model에 저장된 parameter 수. 저장 크기와 관련
- **FLOPs:** 한 번 처리할 때 필요한 부동소수점 연산량의 근사치
- **MACs:** multiply-accumulate 연산 수. FLOPs와 환산 기준이 논문마다 다름
- **FPS:** 1초에 처리하는 frame 수. throughput 관점
- **latency:** 입력 하나가 출력되기까지 걸리는 시간
- **peak VRAM:** 실행 중 가장 많이 사용한 GPU memory

작은 Params가 항상 높은 FPS를 뜻하지 않는다. Memory access, attention token 수, image resolution, kernel 최적화가 속도에 큰 영향을 준다.

### 속도 수치를 볼 때 확인할 조건

- GPU 또는 mobile device 종류
- 입력 해상도
- batch size
- FP32, FP16, INT8 여부
- image encoder 시간을 포함했는지
- prompt 수와 객체 수
- preprocessing과 postprocessing 포함 여부
- warm-up 후 측정인지
- online update와 re-detection 시간을 포함했는지

## 10.12 Absolute Improvement와 Relative Improvement

Baseline 80에서 proposed 84가 되었다고 하자.

- absolute improvement: 4 points 또는 4 percentage points
- relative improvement: (84−80)÷80 = 5%

논문이 “5% improvement”라고 썼을 때 어느 의미인지 확인해야 한다. 한국어 보고서에서는 혼동을 막기 위해 “80→84, +4점”처럼 원래 값과 새 값을 함께 적는 것이 가장 안전하다.

---

# 11. 논문에서 반복되는 원문 용어 사전

## 11.1 연구 설계와 주장

### Baseline

새 방법과 비교하는 기준 방법. Vanilla SAM 2, FIFO memory, decoder-only fine-tuning 등이 baseline이 될 수 있다. Baseline의 checkpoint, prompt, resolution, training data가 proposed method와 같아야 공정하다.

### Vanilla 또는 Off-the-shelf

특별한 개조 없이 공개된 기본 model과 checkpoint를 그대로 사용한다는 뜻. 다만 preprocessing과 prompt 설정이 무엇인지 여전히 확인해야 한다.

### Benchmark

여러 방법을 같은 dataset, metric, protocol로 비교하기 위한 평가 체계. Dataset만이 아니라 split과 metric, prompt 규칙까지 포함할 수 있다.

### State of the Art, SOTA

특정 benchmark와 protocol에서 당시 가장 높은 공개 성능. 모든 실제 환경에서 가장 좋은 모델이라는 뜻은 아니다.

### Ablation Study

제안한 구성 요소를 하나씩 빼거나 바꾸어 무엇이 성능을 만들었는지 확인하는 실험.

예:

- FIFO baseline
- confidence gate만
- diversity selection만
- confidence와 diversity 결합

제안 전체와 baseline만 비교하면 어느 부품이 효과적인지 알기 어렵다.

### Controlled Experiment

한 번에 한 조건만 바꾸고 나머지를 고정하는 실험. Adapter 구조를 비교하면서 training epoch나 prompt까지 바꾸면 원인을 분리하기 어렵다.

### Reproducibility

다른 연구자가 같은 방법을 구현해 비슷한 결과를 얻을 수 있는 정도. Code, seed, dependency version, checkpoint, split, hyperparameter, prompt protocol이 중요하다.

## 11.2 구조와 정보 결합

### Module 또는 Component

전체 model 안에서 특정 기능을 맡는 부품. Adapter, memory selector, boundary head 등이 module이다.

### Pipeline

입력부터 출력까지 여러 model과 처리 단계를 연결한 전체 흐름. Detector → SAM 2 → tracker → postprocessing처럼 구성할 수 있다.

### End-to-End

입력에서 최종 출력까지 연결된 구성 요소를 하나의 loss로 함께 학습한다는 뜻으로 주로 쓰인다. 모든 논문에서 범위가 같지 않으므로 어느 모듈까지 joint training했는지 확인해야 한다.

### Residual Connection

새 변환 결과를 원래 feature에 더하는 연결. Adapter가 원래 pretrained 표현을 보존하면서 작은 변화만 추가하기 좋다.

### Concatenation

두 feature를 channel 방향 등으로 이어 붙이는 것. Addition은 같은 위치끼리 더하지만 concatenation은 정보를 나란히 보존해 channel 수가 늘어난다.

### Fusion

여러 정보원을 결합하는 넓은 표현. 단순 addition, concatenation, attention, gate, weighted sum 모두 fusion이 될 수 있다.

### Gate 또는 Gating

정보를 얼마나 통과시킬지 0~1 weight 등으로 조절한다. 신뢰도가 낮으면 memory update를 닫고, 높으면 여는 방식이다.

### Multi-scale

서로 다른 공간 해상도나 receptive field의 feature를 함께 사용한다. 작은 객체의 세부와 큰 객체의 문맥을 동시에 다루려는 목적이다.

### Hierarchical

낮은 수준에서 높은 수준, 짧은 시간에서 긴 시간처럼 여러 단계 또는 scale로 구성한 구조.

### Coarse-to-Fine

먼저 낮은 해상도나 넓은 범위에서 대략 찾고, 그 주변을 높은 해상도로 정밀하게 다듬는다.

### Local과 Global

- local feature: 가까운 영역의 texture와 boundary
- global feature: 이미지 전체 문맥과 멀리 떨어진 관계

둘은 경쟁 관계가 아니라 보완 관계다.

### Spatial과 Temporal

- spatial: 한 frame 안의 위치와 형태
- temporal: frame 사이 motion, 변화, 순서

Spatiotemporal은 둘을 함께 다룬다는 뜻이다.

## 11.3 예측과 선택

### Logit

Sigmoid나 softmax를 적용하기 전의 model 원시 score. 확률 범위 0~1로 제한되지 않는다.

### Sigmoid와 Softmax

- sigmoid: 각 항목을 독립적으로 0~1 값으로 변환. 이진 또는 multi-label 문제에 흔함
- softmax: 여러 항목의 합이 1이 되도록 변환. 서로 배타적인 class 선택에 흔함

### Threshold

Score가 특정 값 이상일 때만 positive 또는 memory write로 결정하는 경계값. 0.5가 언제나 최적인 것은 아니며 validation set에서 정해야 한다.

### Top-k

Score가 높은 항목 k개만 선택. Memory token 10,000개 중 관련도 높은 256개만 읽는 방식 등에 쓰인다.

### Similarity

두 feature 또는 embedding이 얼마나 비슷한지 나타내는 값.

### Cosine Similarity

두 vector의 크기보다 방향이 얼마나 같은지 측정한다. −1부터 1 범위를 갖지만 실제 normalized visual embedding에서는 0 근처 이상이 흔할 수 있다. Domain과 model마다 적절한 threshold가 다르다.

### Prototype

Target을 대표하는 feature vector. 첫 frame embedding 하나를 쓰거나 여러 고품질 frame feature의 평균을 사용할 수 있다.

### Saliency

Model이 중요하다고 보는 위치 또는 시각적으로 두드러지는 정도. Object-aware sparse routing에서는 target mask 주변 saliency로 계산을 집중할 수 있다.

### Calibration

Confidence와 실제 정답 확률이 맞도록 조정하는 것. Confidence 0.8인 prediction 100개 중 약 80개가 맞는 상태가 잘 calibrated된 예다.

### Disagreement

여러 mask 후보, model, augmentation 결과가 서로 다른 정도. Disagreement가 크면 uncertainty 신호로 사용할 수 있다.

### Entropy

확률 분포가 얼마나 퍼져 있는지 나타낸다. 여러 선택지가 비슷한 확률이면 entropy가 높아 불확실하다고 볼 수 있다.

## 11.4 학습 data를 만드는 용어

### Pseudo-label 또는 Pseudo-mask

사람 GT 대신 model prediction을 임시 정답으로 사용하는 것. Label 비용을 줄이지만 오답을 다시 학습해 confirmation bias가 생길 수 있다.

### Self-training

Model이 unlabeled data에 pseudo-label을 만들고 그 결과로 다시 학습하는 방식.

### Data Augmentation

Crop, flip, color jitter, blur, noise 등을 적용해 다양한 training example을 만드는 것. Mask와 image에 같은 geometric transform을 적용해야 위치가 맞는다.

### Synthetic Corruption

밝기 감소, fog, snow, motion blur, compression noise 등을 인위적으로 만들어 robustness를 평가하거나 학습하는 것. 실제 환경의 복잡한 분포를 완전히 대신하지는 못한다.

### Class Imbalance

배경 픽셀이 대부분이고 작은 target 픽셀이 매우 적은 것처럼 class 비율이 크게 다른 상황. Accuracy만 쓰면 전부 배경이라고 해도 높게 나올 수 있어 Dice, focal loss, class weight가 필요하다.

## 11.5 영상과 memory

### Memory Bank

과거 frame의 feature, mask, object token을 모아 두는 저장 구조. Model weight와 다른 실행 중 memory다.

### Short-term Memory

최근 appearance와 motion 변화에 빠르게 적응하기 위한 memory. 오래된 identity는 잊기 쉽다.

### Long-term Memory

첫 frame이나 고품질 keyframe처럼 오래 유지할 단서를 저장한다. 재등장과 distractor 구분에 도움을 줄 수 있다.

### Memory Admission, Retention, Eviction, Retrieval

- admission: 새 정보를 쓸지 말지
- retention: 무엇을 계속 보존할지
- eviction: 공간이 찼을 때 무엇을 지울지
- retrieval: 현재 prediction에 무엇을 읽을지

### FIFO — First In, First Out

가장 먼저 들어온 항목을 먼저 지우는 queue 정책. 단순하지만 오래된 고품질 identity frame도 시간만 지났다는 이유로 삭제한다.

### Keyframe

영상 전체를 대표하거나 target이 선명하게 보이는 중요한 frame. 고정 간격, confidence, diversity, scene change 등으로 선택할 수 있다.

### Memory Contamination

잘못된 mask 또는 distractor feature가 memory에 들어가 이후 prediction까지 오염시키는 현상.

### Drift

Prediction이 조금씩 target에서 벗어나 결국 다른 부분이나 다른 객체를 따라가는 현상.

### Occlusion, Disappearance, Reappearance

- occlusion: target이 다른 물체에 일부 또는 전부 가려짐
- disappearance 또는 absence: target이 frame에 실제로 없음
- reappearance: target이 다시 보임

완전 가림과 화면 밖 부재를 GT만 보고도 구분하기 어려울 수 있다. 연구에서는 상태 정의를 명확히 해야 한다.

### Distractor

Target과 비슷해 model을 혼동시키는 다른 객체. Positive target prototype뿐 아니라 negative distractor prototype을 저장하면 구분에 도움을 줄 수 있다.

### Re-detection과 Re-prompting

- re-detection: target을 넓은 영역 또는 전체 frame에서 다시 찾음
- re-prompting: 찾은 point, box, mask를 segmentation model에 새 prompt로 입력

둘은 연결될 수 있지만 같은 단계는 아니다.

### Propagation

첫 frame 또는 correction frame의 target 정보를 이후 또는 이전 frame으로 퍼뜨려 mask를 만드는 과정.

### Bidirectional Propagation

Keyframe에서 앞으로와 뒤로 모두 mask를 전파한다. 전체 영상을 볼 수 있는 offline 설정에서 사용하기 쉽다.

## 11.6 강건성과 일반화

### Robustness

Noise, blur, 저조도, 가림, prompt 오차처럼 입력 조건이 나빠져도 성능을 유지하는 능력.

### Corruption

원본 data를 손상시키는 변화. Gaussian noise, fog, brightness, compression 등이 있다.

### In-distribution, ID

Training data와 비슷한 분포의 입력.

### Out-of-distribution, OOD

Training 분포와 다른 입력. 새 병원, 새 sensor, 새로운 날씨 등이 될 수 있다.

### Cross-domain Evaluation

Domain A에서 학습하고 B에서 평가해 일반화 또는 adaptation 정도를 확인하는 실험.

### Failure Case

Model이 반복적으로 실패하는 대표 상황. 좋은 논문은 평균 score뿐 아니라 어떤 조건에서 왜 실패하는지 보여 준다.

## 11.7 효율과 배포

### Lightweight

작은 parameter, 낮은 FLOPs, 적은 VRAM, 높은 FPS 중 하나 이상을 목표로 한다는 넓은 말. 구체 수치 없이 “lightweight”라는 말만 믿어서는 안 된다.

### Real-time

사용 환경의 입력 속도를 지연 없이 따라갈 수 있다는 뜻. 30 FPS camera에서는 전처리와 후처리까지 포함해 약 33 ms 안에 frame을 처리해야 한다. 논문마다 real-time 기준이 다르다.

### Throughput과 Latency

- throughput: 단위 시간에 총 몇 sample을 처리하는가
- latency: sample 하나의 응답이 끝날 때까지 얼마나 걸리는가

큰 batch는 throughput을 높여도 개별 latency는 늘 수 있다.

### Training-free

제안 모듈을 위해 parameter 학습을 추가로 하지 않는다는 뜻. Pretrained model과 threshold, search, optical flow를 사용할 수 있으며 계산 비용이 무료라는 뜻은 아니다.

### Plug-and-Play

기존 model에 큰 재학습이나 구조 변경 없이 붙일 수 있다고 주장하는 모듈. 실제로 checkpoint compatibility와 code 수정 범위를 확인해야 한다.

### Post-processing

Model 출력 뒤에 적용하는 정리 단계. 작은 component 제거, hole 채우기, morphology, NMS, temporal smoothing 등이 있다.

### NMS — Non-Maximum Suppression

서로 많이 겹치는 detection이나 mask 중 score가 높은 것만 남겨 중복을 줄이는 방법.

---

# 12. SAM 2 개선 논문에서 자주 만나는 구조 용어

## 12.1 Head

Backbone feature를 받아 특정 출력을 만드는 비교적 작은 마지막 모듈이다.

- classification head: class score 출력
- detection head: box와 class 출력
- segmentation head 또는 mask head: pixel mask 출력
- boundary head: 경계 map 출력
- uncertainty head: 예측 불확실성 출력

Decoder 전체를 head라고 부르는 논문도 있으므로 구조 그림을 함께 봐야 한다.

## 12.2 U-Net

Encoder에서 해상도를 줄였다가 decoder에서 다시 늘리며, 같은 scale의 encoder feature를 skip connection으로 전달하는 U자형 segmentation 구조다.

뒤쪽 semantic 정보와 앞쪽 boundary 정보를 결합하기 좋아 의료 영상과 자연 영상 segmentation에 널리 쓰인다. “SAM2-UNet”처럼 SAM 2 encoder를 frozen feature extractor로 쓰고 새로운 U-Net decoder를 붙이는 모델도 있다. 이 경우 원래 SAM 2 memory와 prompt 기능을 제거했을 수 있으므로 이름만 보고 video SAM 2 개선이라고 판단하면 안 된다.

## 12.3 ASPP — Atrous Spatial Pyramid Pooling

서로 다른 dilation rate의 dilated convolution을 병렬로 사용해 여러 크기의 context를 모으는 구조다. 작은 건물과 큰 건물을 함께 분할하는 원격탐사 등에 유용하다.

**비유:** 같은 사진을 작은 돋보기, 중간 돋보기, 큰 돋보기로 동시에 보고 결과를 합치는 것.

## 12.4 RFB — Receptive Field Block

서로 다른 kernel 또는 dilation branch를 결합해 다양한 receptive field를 만드는 모듈이다. ASPP와 목적이 비슷하지만 구체 branch 구조와 결합 방식이 다를 수 있다.

## 12.5 Deformable Convolution

일반 convolution은 고정된 격자 위치를 본다. Deformable convolution은 입력에 따라 sampling 위치의 offset을 학습해 구부러진 장기, 불규칙한 해양 생물처럼 형태가 일정하지 않은 객체를 따라 볼 수 있다.

자유도가 늘어 경계 적응에는 유리할 수 있지만 연산과 구현이 복잡하고 작은 dataset에서 과적합할 수 있다.

## 12.6 Positional Encoding

Transformer token에 위치 정보를 더하는 방법이다. Attention만으로는 token 순서를 자동으로 알기 어렵기 때문에 image patch의 x, y 위치나 frame 시간 t를 알려 준다.

- absolute positional encoding: 각 위치 자체를 표현
- relative positional encoding: 두 token 사이 거리와 방향을 표현
- temporal positional encoding: frame 순서를 표현

새로운 aspect ratio나 해상도로 바꿀 때 positional encoding을 interpolation하거나 convolutional 방식으로 적응시키는 논문이 있다.

## 12.7 Normalization

Feature 값의 범위를 안정화해 학습을 돕는 연산이다. Batch Normalization, Layer Normalization 등이 있다.

입력 image를 평균과 표준편차로 정규화하는 preprocessing과 신경망 내부 normalization layer는 관련은 있지만 같은 단계가 아니다.

## 12.8 MoE — Mixture of Experts

여러 expert branch 중 입력에 맞는 branch를 router 또는 gate가 선택하거나 가중 결합하는 구조다.

RGB, depth, LiDAR처럼 modality별 특징이 다를 때 각 expert가 전문화할 수 있다. 그러나 expert 수가 늘면 구조가 복잡해지고 router가 잘못 선택할 위험이 있다.

## 12.9 Routing

입력에 따라 어떤 계산 경로, window, expert, memory token을 사용할지 선택하는 것. Dynamic routing은 쉬운 배경에는 작은 경로를, target 주변에는 큰 경로를 사용해 계산을 줄일 수 있다.

## 12.10 Re-parameterization

Training 때는 표현력이 좋은 여러 branch를 쓰고, inference 전에 하나의 단순한 연산으로 수학적으로 합치는 기법을 가리키는 경우가 많다. Adapter가 training 중 여러 convolution을 쓰더라도 배포 시 한 convolution으로 합쳐 latency를 줄일 수 있다.

PEFT의 parameter-efficient와 re-parameterization의 의미는 다르다. 전자는 학습 parameter 수, 후자는 구조를 등가 변환해 배포 계산을 단순화하는 데 초점이 있다.

## 12.11 Multi-modal과 Modality

**modality:** 정보를 측정하는 방식.

- RGB camera
- depth sensor
- thermal image
- event camera
- LiDAR point cloud
- SAR
- text와 audio

**multi-modal model:** 두 종류 이상의 modality를 함께 사용한다.

같은 장면의 RGB와 depth가 pixel 단위로 정확히 정렬되지 않으면 fusion 오차가 생긴다. 한 sensor가 고장나거나 누락되는 missing-modality robustness도 중요하다.

## 12.12 RGB-D

RGB color와 depth를 함께 사용한 입력. Depth는 카메라에서 물체까지의 거리 정보를 담는다. 색이 비슷한 foreground와 background를 거리로 분리할 수 있지만 depth sensor noise와 정렬 문제가 있다.

## 12.13 SAR

Synthetic Aperture Radar의 약자다. 전파를 이용한 원격탐사 영상으로 구름과 야간의 영향이 적지만, 일반 RGB 사진과 모양과 noise 특성이 크게 다르다. Speckle noise와 낮은 시각적 직관 때문에 SAM 2의 큰 domain shift 사례다.

## 12.14 Event Camera

일반 camera처럼 일정 시간마다 전체 frame을 찍지 않고, 각 pixel의 밝기 변화가 일정 기준을 넘을 때 event를 기록한다. 매우 빠른 motion과 넓은 밝기 범위에 유리하지만 RGB frame과 표현이 완전히 다르다.

## 12.15 LiDAR

Laser를 쏘고 돌아오는 시간을 이용해 3D point cloud를 측정한다. Camera의 풍부한 texture와 LiDAR의 정확한 거리 정보를 결합할 수 있다.

## 12.16 Semantic Feature와 Appearance Feature

- **appearance feature:** 색, texture, 모양처럼 어떻게 생겼는지
- **semantic feature:** 어떤 종류 또는 의미인지

같은 종류의 검은 신발 두 짝은 semantic은 같지만 instance appearance와 위치가 다를 수 있다. Category semantics만으로 tracking하면 target을 다른 같은 class 객체로 바꿀 수 있다.

## 12.17 Motion Residual

Motion model이 예측한 위치와 실제 관측 mask 위치의 차이. 갑자기 residual이 커지면 빠른 motion, camera cut, tracking failure 가능성이 있다.

## 12.18 Forward-Backward 또는 Cycle Consistency

Frame t의 mask를 t+1로 추적한 뒤 다시 t로 되돌렸을 때 처음 mask와 비슷해야 한다는 검사다.

앞으로는 우연히 distractor를 따라가도 뒤로 되돌아왔을 때 불일치가 커질 수 있어 오류 검증에 유용하다. 그러나 distractor 경로도 일관되면 통과할 수 있고 계산량이 늘어난다.

## 12.19 EMA — Exponential Moving Average

최근 값에 더 큰 weight를 주면서 과거 값을 부드럽게 누적하는 평균이다.

Risk score나 pointer distance가 한 frame noise 때문에 급변하지 않도록 smoothing할 수 있다. 반대로 너무 강하게 smoothing하면 실제 failure 감지가 늦어진다.

## 12.20 Beam Search 또는 Memory Tree

매 frame 최고 mask 하나만 확정하지 않고 여러 후보 경로를 잠시 유지한다. 문장 생성에서 여러 단어 후보를 보존하는 beam search와 비슷하다.

장점은 초기에 두 번째였던 mask가 나중에 올바른 경로로 밝혀질 가능성을 살리는 것이다. 단점은 beam width만큼 계산과 memory가 늘고, 전체 영상을 본 뒤 선택하면 offline이 된다는 것이다.

---

# 13. 특히 헷갈리는 개념 비교

## 13.1 Training-free, Zero-shot, Frozen, Prompt-free

| 용어 | 실제 뜻 | 있어도 되는 것 |
|---|---|---|
| Training-free method | 제안 방법을 위해 추가 parameter 학습을 하지 않음 | pretrained model, search, threshold, optical flow |
| Zero-shot transfer | target dataset label로 추가 학습하지 않음 | 대규모 pretraining, test-time prompt |
| Frozen module | 해당 module weight를 업데이트하지 않음 | forward 계산, trainable adapter |
| Prompt-free inference | 사람이 prompt를 직접 입력하지 않음 | automatic detector나 learned prompt generator |

한 model이 동시에 frozen이고 zero-shot일 수도 있지만 네 용어는 서로 바꿔 쓸 수 없다.

## 13.2 Accuracy, Robustness, Efficiency

- **accuracy:** 정상 조건에서 prediction이 GT와 얼마나 맞는가
- **robustness:** corruption, domain shift, prompt noise, occlusion에서도 얼마나 유지되는가
- **efficiency:** time, memory, energy, parameter, 사용자 개입을 얼마나 아끼는가

“더 나은 성능”을 주장하려면 무엇이 더 나아졌고 무엇을 희생했는지 밝혀야 한다. J&F +1점과 FPS 2배를 얻었지만 저조도 성능이 크게 떨어졌다면 단일 숫자로 우열을 정하기 어렵다.

## 13.3 Confidence, Uncertainty, Correctness

- confidence가 높음: model이 스스로 확신
- uncertainty가 높음: model이 선택에 확신하지 못함
- correctness: 실제 GT와 맞음

Model은 틀렸는데 confidence가 높을 수 있다. 따라서 confidence threshold는 validation에서 calibration하고, 가능하면 여러 독립 신호로 검증한다.

## 13.4 Parameter 수, 파일 크기, VRAM, 속도

- Parameter 수가 줄면 checkpoint 저장 크기는 대체로 줄어든다.
- Training VRAM은 activation과 optimizer 때문에 별개다.
- FPS는 hardware와 kernel 최적화 영향을 크게 받는다.
- Token 수를 줄이면 parameter는 그대로여도 attention 속도가 빨라질 수 있다.

따라서 “parameter 90% 감소”만으로 “10배 빨라졌다”고 결론 내리면 안 된다.

## 13.5 Class-agnostic과 Open-vocabulary

- **class-agnostic segmentation:** class 이름과 무관하게 prompt된 foreground mask를 만듦
- **open-vocabulary segmentation:** text로 임의 class 이름을 지정하고 class와 mask를 연결

SAM 2 기본 능력은 class-agnostic promptable segmentation에 가깝다.

## 13.6 Fine-tuning과 Domain Adaptation

Fine-tuning은 pretrained weight를 추가 학습하는 행위다. Domain adaptation은 source-target 분포 차이를 줄이려는 문제 설정과 방법론이다. Target labeled data로 단순 fine-tuning해 domain 성능을 올리는 것도 넓게는 adaptation이지만, UDA처럼 별도 목표를 가진 연구와 구분해야 한다.

## 13.7 Detector box와 GT box

GT box는 정답 mask에서 만든 거의 완벽한 prompt다. Detector box는 class 누락, 위치 오차, 중복이 생긴다. SAM 2가 GT box에서 높은 성능을 냈다고 실제 자동 pipeline도 같은 성능을 낸다고 볼 수 없다.

## 13.8 Frame-wise Segmentation과 Video Propagation

- frame-wise: 각 frame을 독립 이미지로 처리
- video propagation: 과거 memory와 temporal relation을 사용

Frame-wise 방식은 memory contamination이 없지만 매 frame prompt가 필요할 수 있고 시간 일관성이 약하다. Propagation은 효율적이지만 한 번의 오류가 이어질 수 있다.

## 13.9 Semantic Segmentation과 SAM 2

Semantic segmentation은 모든 pixel에 class를 붙인다. SAM 2는 prompt가 가리키는 object mask를 만드는 category-agnostic model이다.

SAM 2 encoder에 U-Net decoder를 붙여 organ class를 자동 출력하면 그것은 SAM 2 backbone을 이용한 semantic model이지, 원래 promptable SAM 2 전체 기능을 개선한 것과는 다르다.

## 13.10 Image-only Improvement와 Video Improvement

Image encoder adapter가 static image mask를 개선해도 memory propagation, occlusion recovery, temporal consistency가 좋아졌다고 자동으로 말할 수 없다. Video 개선을 주장하려면 실제 영상 dataset과 temporal metric이 필요하다.

---

# 14. 논문 문장을 해석하는 연습

다음과 같은 문장을 보자.

> We freeze the Hiera image encoder and insert stage-specific depthwise-dilated adapters, while fine-tuning a lightweight boundary-aware decoder with Dice and BCE losses.

이를 부품별로 번역하면 다음과 같다.

1. **freeze the Hiera image encoder:** SAM 2 backbone weight는 업데이트하지 않는다.
2. **insert adapters:** 기존 층 사이에 작은 trainable module을 추가한다.
3. **stage-specific:** 서로 다른 해상도 stage가 같은 adapter를 공유하지 않는다.
4. **depthwise-dilated:** channel별 가벼운 convolution과 간격 있는 kernel로 여러 크기의 지역 정보를 본다.
5. **lightweight decoder:** 최종 mask를 만드는 작은 모듈을 학습한다.
6. **boundary-aware:** 객체 내부뿐 아니라 경계를 더 잘 맞추려는 구조 또는 loss가 있다.
7. **Dice and BCE losses:** 영역 겹침과 픽셀별 foreground 확률을 함께 학습한다.

이 문장만으로는 아직 다음 정보를 알 수 없다.

- Trainable parameter 수가 얼마인가?
- Prompt는 point, box, none 중 무엇인가?
- Static image인지 video인지?
- Memory encoder와 attention을 실제로 사용했는가?
- Input resolution과 GPU는 무엇인가?
- Baseline도 같은 training data와 prompt를 사용했는가?
- Boundary F가 실제로 개선되었는가?

논문 제목과 한 문장만 읽고 “SAM 2 전체를 개선했다”고 결론 내리지 말고, 구조 그림, training 설정, 평가 protocol을 함께 확인해야 한다.

---

# 15. 과제연구 논문을 읽는 순서

## 15.1 첫 번째 읽기 — 문제만 파악

1. 무엇이 실패한다고 주장하는가?
2. 입력과 출력은 무엇인가?
3. Image task인가 video task인가?
4. 실제 SAM 2의 어느 module을 사용하는가?

## 15.2 두 번째 읽기 — 방법을 부품으로 나누기

1. Backbone은 frozen인가?
2. 새 trainable module은 어디에 들어가는가?
3. Prompt를 누가 어떻게 만드는가?
4. Memory에 무엇을 쓰고 무엇을 읽는가?
5. Decoder와 output resolution이 바뀌는가?
6. Training-free인가, fine-tuning인가, test-time update인가?

## 15.3 세 번째 읽기 — 실험의 공정성 확인

1. Baseline과 같은 checkpoint인가?
2. 같은 dataset split과 resolution인가?
3. 같은 prompt 개수와 품질인가?
4. 같은 training data와 epoch인가?
5. 수치가 mean인지 best run인지?
6. 여러 random seed를 사용했는가?
7. 속도 측정 hardware가 같은가?

## 15.4 네 번째 읽기 — 한계와 확장 찾기

1. 어느 조건에서 실패하는가?
2. 새 module이 정말 필요한가, 단순 threshold로도 되는가?
3. Accuracy를 얻으며 speed나 generalization을 잃지 않았는가?
4. Component별 ablation이 충분한가?
5. 다른 domain과 긴 영상에서도 작동하는가?

---

# 16. SAM 2 과제연구를 설계할 때의 기본 원칙

## 16.1 “정확도를 측정한다”에서 “원인을 검증한다”로

약한 연구 질문:

> SAM 2는 이 dataset에서 정확도가 몇 점인가?

더 좋은 연구 질문:

> 저조도에서 첫 mask confidence가 떨어질 때 잘못된 frame이 memory에 기록되는 것이 장기 성능 하락의 주원인인가?

더 좋은 질문은 원인, 개입 방법, 비교군, 측정 지표가 함께 들어 있다.

## 16.2 한 번에 하나의 핵심 주장

예를 들어 memory 개선 연구라면 먼저 다음 최소 비교를 만든다.

1. Vanilla FIFO
2. Confidence-only gate
3. Diversity-only selection
4. Confidence와 diversity 결합

동시에 detector, LoRA, optical flow, boundary decoder를 모두 추가하면 성능은 오를 수 있지만 무엇이 원인인지 설명하기 어렵다.

## 16.3 Accuracy 외에 반드시 기록할 것

- J, F, J&F 또는 task에 맞는 정확도
- 객체 size와 corruption severity별 성능
- Prompt 수와 correction 수
- FPS 또는 latency
- Peak VRAM
- Trainable Params
- 실패 사례 영상
- Random seed와 평균·표준편차

## 16.4 8 GB급 GPU에서 현실적인 방향

- 작은 SAM 2 checkpoint부터 시작
- Backbone frozen
- Decoder-only 또는 adapter, LoRA
- 짧은 clip과 작은 batch
- Mixed precision
- Gradient accumulation과 필요 시 activation checkpointing
- 공개 checkpoint의 training-free memory policy 먼저 실험
- Full SA-1B 또는 SA-V 재학습은 피하기

Training-free memory selection, automatic re-prompting, lightweight boundary head, adapter 비교는 대규모 foundation model을 처음부터 학습하는 것보다 과제연구에 현실적이다.

---

# 17. 빠른 복습용 핵심 용어표

| 용어 | 가장 짧은 뜻 |
|---|---|
| Pixel | 이미지의 한 격자 칸 |
| RGB | 빨강, 초록, 파랑 세 입력 channel |
| Resolution | 가로×세로 pixel 수 |
| Frame | 영상의 한 장 |
| Tensor | 다차원 숫자 배열 |
| Channel | 같은 위치에서 서로 다른 정보 축 |
| Mask | pixel별 target 영역 지도 |
| Feature | 모델이 입력에서 추출한 유용한 표현 |
| Parameter | 학습으로 바뀌는 weight와 bias |
| Hyperparameter | 사람이 정하는 learning rate 등의 설정 |
| Encoder | 입력을 feature로 변환 |
| Backbone | 주된 feature extractor |
| Decoder | Feature를 task output으로 변환 |
| CNN | 지역 kernel로 패턴을 찾는 신경망 |
| Transformer | Token 관계를 attention으로 처리 |
| Attention | 중요한 정보를 가중 선택 |
| Embedding | 대상을 vector 공간에 나타낸 표현 |
| Token | Transformer가 처리하는 한 정보 단위 |
| Loss | Training을 위한 오차 신호 |
| Metric | 사람이 결과를 평가하는 점수 |
| Checkpoint | 학습된 model 상태 저장 파일 |
| Pretraining | 크고 일반적인 data로 먼저 학습 |
| Fine-tuning | 새 task나 domain에 추가 학습 |
| Frozen | Parameter update를 막음 |
| Inference | 학습 없이 prediction을 생성 |
| Prompt | 원하는 target을 지정하는 입력 |
| Zero-shot | Target label로 추가 학습하지 않은 설정 |
| Foundation Model | 여러 task에 재사용할 범용 pretrained model |
| Domain Shift | 학습 환경과 사용 환경의 분포 차이 |
| Domain Adaptation | Target domain에 맞게 차이를 줄임 |
| PEFT | 적은 parameter만 학습하는 방법들의 총칭 |
| Adapter | 기존 model 사이에 넣는 작은 trainable module |
| LoRA | 기존 weight 변화량을 저랭크 행렬로 학습 |
| Distillation | Teacher 지식을 작은 student가 학습 |
| Quantization | 숫자 bit 수를 낮춤 |
| Pruning | 덜 중요한 weight, channel, token을 제거 |
| Wavelet | 위치를 유지하며 주파수와 scale로 분해 |
| Optical Flow | 두 frame 사이 pixel motion vector |
| Memory Bank | 과거 frame 정보를 저장하는 구조 |
| Memory Contamination | 잘못된 frame 저장으로 오류가 전파됨 |
| Re-prompting | 새 point나 box로 모델을 다시 안내 |
| Boundary Refinement | Mask 경계를 후속 모듈로 정밀화 |
| VOS | 영상에서 지정 객체의 mask를 유지 |
| PVOS | Visual prompt로 지정하는 VOS |
| PVS | SAM 2 원문의 Promptable Visual Segmentation |
| VOT | 주로 한 target 위치를 추적 |
| MOT | 여러 target을 검출하고 ID를 유지 |
| RVOS | 명시적 문장이 가리키는 객체를 영상 분할 |
| VRS | 암시적 문장을 추론해 객체를 영상 분할 |

---

# 18. 마지막 점검 질문

아래 질문에 답할 수 있다면 SAM 2 선행연구의 기본 문장을 읽을 준비가 된 것이다.

1. RGB channel과 feature channel은 왜 다른가?
2. Semantic segmentation과 instance segmentation의 출력은 어떻게 다른가?
3. VOS와 VOT의 평가 metric을 왜 직접 비교하면 안 되는가?
4. Frozen backbone도 왜 GPU 계산과 activation memory가 필요한가?
5. Feature, embedding, token은 각각 무엇을 강조하는 표현인가?
6. Zero-shot, prompt-free, training-free는 왜 같은 말이 아닌가?
7. Adapter와 LoRA는 pretrained model을 어떤 방식으로 바꾸는가?
8. Parameter가 줄어도 FPS가 오르지 않을 수 있는 이유는 무엇인가?
9. Optical flow가 긴 가림 뒤 identity를 혼자 복구하기 어려운 이유는 무엇인가?
10. SAM 2 memory에 잘못된 mask가 들어가면 왜 다음 frame 오류까지 커지는가?
11. IoU만으로 가는 혈관이나 머리카락 경계 개선을 충분히 평가할 수 있는가?
12. “성능이 5% 향상되었다”는 문장에서 무엇을 추가로 확인해야 하는가?
