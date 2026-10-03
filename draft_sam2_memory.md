# SAM, SAM2, SAM2.1의 구조와 비디오 메모리 완전 이해

> 대상 독자: 컴퓨터비전을 처음 배우는 고등학생·학부 초급자  
> 목표: SAM2를 단순히 실행하는 수준을 넘어, 각 모듈이 무슨 정보를 만들고 그 정보가 어떻게 오류 누적·드리프트·재등장 실패로 이어지는지 설명할 수 있게 한다.

---

## 0. 이 글을 다 읽으면 답할 수 있어야 하는 질문

1. 분류, 검출, 분할, 추적은 출력이 어떻게 다른가?
2. SAM은 점 하나를 받고 어떻게 픽셀 마스크를 만드는가?
3. SAM2에서 Hiera image encoder, prompt encoder, memory attention, mask decoder, memory encoder는 각각 무엇을 하는가?
4. candidate mask, predicted IoU, object score, object pointer는 서로 무엇이 다른가?
5. 정지 영상과 비디오를 넣었을 때 실제 계산 흐름은 어떻게 달라지는가?
6. autoregression과 FIFO memory 때문에 왜 한 번의 실수가 여러 프레임 뒤 큰 실패가 되는가?
7. occlusion, disappearance, reappearance, distractor, drift, identity switch, memory contamination은 어떤 인과관계로 연결되는가?
8. SAM2Long, DAM4SAM, SAMURAI, UAMP, OASAMT, SENTRY, Lean-SAM2, SAM2Dual, StreamDAM, RUAC은 각각 무엇을 바꾸며 무엇을 해결하지 못했는가?

---

## 1. 먼저 큰 지도: 컴퓨터비전 모델은 무엇을 출력하는가?

같은 사진을 보더라도 문제에 따라 정답 형식이 다르다. 사진 속에 사람 두 명과 강아지 한 마리가 있다고 하자.

| 문제 | 모델이 답하는 질문 | 출력 예시 | 쉬운 비유 | SAM2와의 관계 |
|---|---|---|---|---|
| Image classification | 이 사진 전체는 무엇인가? | `공원`, `강아지 사진` 같은 한 개 이상의 label | 사진에 제목 붙이기 | SAM2의 주목적이 아니다. SAM2는 class 이름보다 영역을 찾는다. |
| Object detection | 무엇이 어디에 있는가? | class + bounding box | 물체마다 네모 상자 치기 | detector가 찾은 box를 SAM2 prompt로 줄 수 있다. |
| Semantic segmentation | 각 픽셀은 어느 class인가? | 모든 사람 픽셀은 `person=1` | 같은 종류는 같은 색으로 칠하기 | 같은 class의 개체를 구분하지 않는 점이 SAM2의 instance-like mask와 다르다. |
| Instance segmentation | 개별 물체마다 픽셀 영역은 어디인가? | 사람 1 mask, 사람 2 mask, 강아지 mask | 물체마다 다른 색으로 오려내기 | SAM/SAM2가 prompt를 받았을 때 만드는 출력과 가장 가깝다. |
| Panoptic segmentation | 모든 픽셀의 class와 개체 ID는 무엇인가? | 하늘·도로 같은 stuff + 사람 1·2 같은 thing | 장면 전체를 빠짐없이 색칠하고 번호 붙이기 | SAM2 단독은 class label과 완전한 장면 구성을 자동 보장하지 않는다. |
| Visual tracking | 처음 지정한 물체가 다음 프레임 어디로 갔는가? | 프레임별 box 또는 위치 | 한 선수를 계속 따라가는 카메라 | SAM2 mask를 box로 바꾸면 VOT 방식 평가도 가능하다. |
| Video Object Segmentation, VOS | 처음 지정한 물체의 매 프레임 픽셀 영역은? | 시간에 따른 mask 묶음, 즉 masklet | 움직이는 물체를 매 장면 가위로 오려내기 | SAM2의 대표 사용법이다. |
| Promptable Visual Segmentation, PVS | 어느 프레임에서든 점·box·mask로 지정·수정한 대상을 영상 전체에서 분할할 수 있는가? | 상호작용 가능한 masklet | 중간 장면에서 다시 지시할 수 있는 추적 도구 | SAM2 논문이 공식적으로 정의한 더 넓은 과제다. |
| Visual Object Tracking, VOT | 첫 프레임 box/mask로 지정한 한 물체를 계속 찾을 수 있는가? | 주로 프레임별 box, 일부 benchmark는 mask | 한 명의 선수를 끝까지 따라가기 | SAMURAI·DAM4SAM 등은 SAM2를 VOT tracker로 변형한다. |
| Multi-Object Tracking, MOT | 여러 물체의 위치와 ID를 동시에 유지할 수 있는가? | box + track ID + trajectory | 경기장의 모든 선수를 번호별로 따라가기 | 기본 SAM2는 object별 memory를 따로 돌리고 object 간 통신은 하지 않는다. |
| Referring Video Object Segmentation, RVOS | 문장으로 가리킨 물체를 영상 전체에서 분할할 수 있는가? | `빨간 모자를 쓴 사람`의 masklet | 말로 지시한 대상을 오려내기 | text grounding/VLM이 SAM2 prompt를 만들거나 feature를 보강한다. |

주의할 점이 있다. 논문마다 `PVOS`를 Promptable Video Object Segmentation 등의 뜻으로 쓰기도 하지만 약어가 완전히 통일된 것은 아니다. SAM2 원문이 명시적으로 쓰는 용어는 **PVS, Promptable Visual Segmentation**이다. `VRS`도 Video Referring Segmentation 또는 Video Reasoning Segmentation 등 문맥에 따라 달라질 수 있으므로, 약어만 보지 말고 해당 논문의 task definition과 입력·출력을 확인해야 한다.

### SAM과 SAM2는 class 이름을 맞히는 모델인가?

핵심적으로는 아니다. SAM 계열은 `이 점/상자/마스크가 가리키는 유효한 영역을 분할하라`는 **class-agnostic promptable segmentation** 모델이다. 강아지라는 단어를 몰라도 강아지 몸의 경계를 찾을 수 있다. 반대로 출력 마스크에 자동으로 `강아지`라는 이름을 붙이는 것은 별도 classifier, detector, VLM이 필요할 수 있다.

---

## 2. SAM: 한 장의 이미지에서 prompt를 mask로 바꾸는 모델

[Segment Anything — ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)의 구조는 크게 세 부분이다.

```text
이미지 ──> Image Encoder ──> image embedding ─────────────┐
                                                         ├─> Mask Decoder ─> masks + predicted IoU
점/box/mask ──> Prompt Encoder ──> prompt embedding ─────┘
```

### 2.1 Image encoder: 사진을 특징 지도(feature map)로 바꾸기

원본 SAM은 큰 Vision Transformer를 사용한다. 픽셀 RGB 값을 그대로 두고 mask를 그리는 것이 아니라, 이미지 전체를 여러 토큰과 feature map으로 바꾼다.

- 얕은 시각 특징: 선, 모서리, 색 변화, 작은 무늬
- 깊은 시각 특징: 물체의 부분, 형태, 주변 문맥

Image encoder는 무거운 부분이지만 한 이미지에 대해 한 번만 실행할 수 있다. 같은 사진에 사용자가 점을 여러 번 추가하더라도 image embedding을 재사용하고, 비교적 가벼운 prompt encoder와 mask decoder만 다시 돌릴 수 있다.

### 2.2 Prompt encoder: 사용자의 지시를 벡터로 번역하기

Prompt는 모델에게 `무엇을 분할해야 하는지` 알려 주는 입력이다.

- Positive point: 이 픽셀은 목표 안쪽이다.
- Negative point: 이 픽셀은 목표 바깥쪽이다.
- Box: 이 사각형 안의 물체를 찾으라는 힌트다.
- Mask: 이전 예측이나 사람이 그린 영역을 더 정밀한 dense prompt로 준다.

점과 box 같은 sparse prompt는 위치 encoding과 prompt 종류 embedding을 결합한다. Mask 같은 dense prompt는 convolution으로 낮은 해상도의 feature로 바꿔 image embedding과 결합한다.

### 2.3 Mask decoder: image와 prompt를 서로 대화시키기

SAM의 mask decoder는 two-way transformer를 사용한다.

1. Prompt token이 image token을 보며 `지시와 관련된 픽셀이 어디인가?`를 찾는다.
2. Image token도 prompt 정보를 받아 `현재 사용자가 원하는 영역`에 맞게 갱신된다.
3. Mask token이 동적 분류기처럼 작동해 각 위치가 foreground일 가능성을 계산한다.
4. 낮은 해상도의 mask logits를 원본 크기에 가깝게 올려 최종 mask를 만든다.

`two-way`는 prompt가 image만 일방적으로 읽는 것이 아니라 prompt와 image 표현이 서로 갱신된다는 뜻이다.

### 2.4 왜 candidate mask가 여러 개인가?

자전거 바퀴에 점 하나를 찍으면 정답은 하나로 고정되지 않는다.

- 타이어만 선택했을 수 있다.
- 바퀴 전체를 선택했을 수 있다.
- 자전거 전체를 선택했을 수 있다.

SAM은 이런 **구조적 모호성**을 평균 내어 흐릿한 한 mask로 만들지 않고, 보통 여러 candidate mask를 낸다. 원 논문에서는 흔한 part–subpart–whole 모호성을 다루기 위해 세 후보가 충분하다고 보았다.

### 2.5 Predicted IoU는 실제 IoU가 아니다

실제 IoU는 정답 mask (G)가 있을 때만 계산할 수 있다.

\[
IoU(M,G)=\frac{|M\cap G|}{|M\cup G|}
\]

실제 서비스 추론에서는 (G)가 없으므로 IoU를 알 수 없다. SAM이 내는 `predicted IoU`는 **자기 mask가 실제 정답과 얼마나 겹칠 것인지 모델이 추정한 품질 점수**다. 따라서 다음을 구분해야 한다.

- Actual IoU: 정답으로 사후 계산한 평가값
- Predicted IoU: 정답 없이 모델이 예측한 자기평가값

높은 predicted IoU가 항상 올바른 identity를 뜻하지는 않는다. 다른 사람을 아주 깔끔하게 분할하면 mask 모양 품질은 좋아 보여도 목표 identity는 틀릴 수 있다.

### 2.6 한 장 이미지의 실제 추론 흐름

1. 이미지를 resize·normalize한다.
2. Image encoder가 image embedding을 만든다.
3. 사용자의 점·box·mask를 prompt encoder가 embedding으로 만든다.
4. Mask decoder가 candidate masks와 predicted IoU를 출력한다.
5. 보통 predicted IoU가 가장 높은 후보를 기본 mask로 표시한다.
6. 사용자가 positive/negative point를 더 주면 같은 image embedding으로 다시 decoding한다.

SAM에는 시간축 기억이 없다. 다음 이미지가 들어오면 이전 mask는 자동으로 이어지지 않는다.

---

## 3. SAM2: SAM에 시간축과 기억을 추가하다

[SAM 2 — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html)는 이미지를 `프레임이 하나뿐인 비디오`로 보고 image와 video를 하나의 구조로 처리한다.

```text
현재 프레임
   │
   ▼
Hiera Image Encoder ──> 현재 frame features ────────────┐
                                                        │ query
과거 spatial memories + object pointers ───────────────> Memory Attention
                                                        │
현재 frame의 점/box/mask ──> Prompt Encoder ────────────┤
                                                        ▼
                                                   Mask Decoder
                                                        │
                  ┌──────────────┬──────────────┬────────┴─────────┐
                  ▼              ▼              ▼                  ▼
            candidate masks  predicted IoU  object score      object pointer
                  │                                              │
                  └── 선택된 mask + 원래 frame feature ─┐         │
                                                        ▼         │
                                                  Memory Encoder  │
                                                        │         │
                                                        └──> Streaming Memory Bank
```

### 3.1 Hiera image encoder: 여러 크기의 시각 정보를 동시에 만들기

SAM2는 MAE로 사전학습된 **Hiera** 계층형 encoder를 사용한다. `hierarchical`은 네트워크가 진행될수록 해상도는 낮추고 표현은 더 추상적으로 만든다는 뜻이다.

- Stage 1, stride 4: 해상도가 높아 얇은 경계와 작은 구조에 유리하다.
- Stage 2, stride 8: 중간 크기 세부 정보를 담는다.
- Stage 3, stride 16: 더 넓은 문맥과 물체 수준 정보를 담는다.
- Stage 4, stride 32: 가장 넓고 추상적인 정보를 담는다.

SAM2는 FPN으로 stride 16·32 feature를 융합해 주 image embedding을 만든다. Stride 4·8의 고해상도 feature는 memory attention을 통과하지 않고 mask decoder의 upsampling 단계로 skip connection되어 경계 복원에 도움을 준다.

쉬운 비유로는 다음과 같다.

- 고해상도 feature: 확대경으로 보는 털·끈·가장자리
- 저해상도 deep feature: 멀리서 보는 전체 형태와 문맥

### 3.2 Prompt encoder: SAM과 거의 같은 입력 언어

SAM2의 prompt encoder는 SAM 설계를 따른다.

- positive/negative click
- bounding box
- mask prompt

비디오에서는 첫 프레임뿐 아니라 **어느 프레임에서든** 새 prompt를 줄 수 있다. 중간 프레임에서 오류를 수정하면 그 정보가 새로운 conditioning memory가 되어 다른 프레임 예측을 고칠 수 있다.

### 3.3 Memory attention: 현재 장면을 과거 기억으로 조건화하기

현재 프레임 encoder feature만 보면 `지금 보이는 물체가 처음 지정한 바로 그 물체인가?`를 알기 어렵다. Memory attention은 현재 feature를 query로 하고 과거의 spatial memory와 object pointer를 key/value처럼 참고한다.

기본적으로 각 memory-attention block은 다음 순서를 따른다.

1. 현재 frame token끼리 self-attention한다.
2. 과거 spatial memory와 object pointer에 cross-attention한다.
3. MLP로 feature를 갱신한다.

공개 기본 구조는 이 block을 4층 사용한다. 결과는 단순한 현재 image feature가 아니라 `과거에 지정된 목표를 고려한 현재 feature`다.

주의: attention은 정답을 논리적으로 검색하는 데이터베이스 조회가 아니다. 현재 feature와 닮은 memory를 강하게 참고하는 학습된 유사도 연산이므로, memory가 잘못되면 오히려 오답을 증폭할 수 있다.

### 3.4 Mask decoder가 내는 네 종류의 핵심 출력

#### 3.4.1 Candidate masks

현재 prompt와 memory에 맞는 여러 mask 후보다. 비디오에서도 part/whole 모호성과 distractor 모호성이 있으므로 여러 후보가 생성될 수 있다.

#### 3.4.2 Predicted IoU

각 candidate mask의 예상 분할 품질이다. 모델이 추정하는 점수이지 실제 정답 IoU가 아니다. 기본 SAM2는 추가 prompt가 모호성을 해결하지 않으면 현재 프레임에서 predicted IoU가 가장 높은 mask 하나를 다음 전파에 사용한다.

#### 3.4.3 Object score 또는 occlusion/presence score

현재 프레임에 목표가 보이는지 추정하는 별도 head의 출력이다. SAM에서는 positive prompt가 주어진 정지 이미지에 유효한 영역이 항상 있다고 가정하기 쉽지만, 비디오에서는 목표가 완전히 가려지거나 화면 밖으로 나갈 수 있다.

이 점수도 완벽한 확률이 아니다. 가려진 목표를 `보임`으로 잘못 판단하거나 비슷한 distractor를 목표로 판단할 수 있다.

#### 3.4.4 Object pointer

선택된 mask에 해당하는 mask token을 변환한 가벼운 벡터다. Spatial memory가 `어느 픽셀에 무엇이 있었는가`를 담는 지도라면, object pointer는 `이 프레임에서 모델이 목표라고 요약한 것`을 담는 압축된 명함에 가깝다.

원문 구현은 256차원 pointer를 64차원 네 token으로 나누어 memory attention에서 사용한다. Pointer는 identity에 도움이 되지만, 잘못된 mask token에서 만들어지면 잘못된 대상의 명함이 된다.

### 3.5 Memory encoder: 현재 예측을 미래가 읽을 기억으로 바꾸기

Memory encoder는 별도의 무거운 image encoder를 다시 돌리지 않는다.

1. 선택된 mask logits를 convolution으로 downsample한다.
2. 이를 현재 프레임의 **unconditioned Hiera feature**와 위치별로 결합한다.
3. 가벼운 convolution block으로 융합한다.
4. 압축된 spatial memory feature를 만든다.

즉, memory에는 이미지 자체만 들어가는 것도 아니고 mask만 들어가는 것도 아니다. **그 프레임의 시각 특징과 모델이 목표라고 칠한 위치가 결합된 표현**이 들어간다. 그래서 wrong mask를 encode하면 잘못된 위치의 시각 특징이 목표 정보로 묶인다.

### 3.6 Streaming memory bank: 모든 과거를 저장하지 않는 이유

수천 프레임을 모두 attention에 넣으면 시간과 메모리가 계속 증가한다. SAM2는 bounded streaming memory를 사용한다.

- Prompted/conditioning memory: 사용자가 직접 prompt를 준 프레임의 기억
- Non-conditioning memory: 모델이 자동으로 예측한 최근 프레임의 기억
- Object pointers: 각 프레임 mask token에서 나온 고수준 벡터

원 논문은 최근 memory를 FIFO로 관리한다. 공개 기본 코드의 `num_maskmem=7`은 전형적인 first-frame-only VOS에서 **입력/조건 프레임 1개 + 최근 프레임 6개**를 뜻한다. 추가 prompt를 여러 프레임에 주는 interactive setting은 conditioning memory 선택이 더 복잡해질 수 있다.

### 3.7 SAM2.1은 SAM2와 완전히 다른 구조인가?

아니다. SAM2.1은 같은 큰 구조를 유지하면서 학습과 일부 positional/memory 설정을 강화한 업데이트다. Meta의 공식 설명에 따르면 다음이 개선되었다.

- 비슷하게 생긴 물체와 작은 물체를 더 잘 다루도록 data augmentation 강화
- 더 긴 frame sequence로 학습해 occlusion 처리 개선
- spatial memory와 object pointer memory의 positional encoding 조정
- 더 강한 updated checkpoint 제공

ICLR 2025 최종 SAM2 논문의 결과는 사실상 개선된 SAM2.1 버전을 사용하지만, 논문에서는 간단히 SAM2라고 부른다. 따라서 논문·코드 비교 시 `SAM2`, `SAM2.1`, `SAM2.1Long`이 정확히 어떤 checkpoint와 config를 뜻하는지 확인해야 한다.

---

## 4. 정지 이미지와 비디오에서 실제 추론 흐름

### 4.1 단일 이미지 추론

1. Hiera encoder가 한 장의 multiscale feature를 만든다.
2. 과거 memory가 없으므로 memory attention에는 실질적인 과거 정보가 없다. 모델은 no-memory 상태로 동작한다.
3. Prompt encoder가 점·box·mask를 변환한다.
4. Mask decoder가 후보 mask와 predicted IoU를 낸다.
5. 기본 후보를 표시하고 추가 prompt가 들어오면 같은 image feature로 다시 decoding한다.

이 경우 memory encoder와 streaming bank는 다음 프레임 전파를 위해 꼭 사용할 필요가 없다. 그래서 SAM2는 image에 적용될 때 SAM처럼 행동한다.

### 4.2 첫 비디오 프레임

사용자가 프레임 1의 강아지에 positive click 또는 mask를 준다고 하자.

1. Frame 1을 Hiera encoder로 변환한다.
2. 과거 memory 없이 prompt를 이용해 mask 후보를 만든다.
3. 후보 중 하나를 선택한다.
4. 선택 mask에서 object pointer를 만든다.
5. Frame 1 feature와 mask를 memory encoder에 넣는다.
6. 결과를 **conditioning memory**, 즉 강한 기준 기억으로 저장한다.

### 4.3 두 번째 이후 프레임

프레임 `t`의 전형적인 흐름은 다음과 같다.

1. 현재 frame `I_t`를 Hiera encoder로 변환해 `F_t`를 얻는다.
2. Memory attention이 `F_t`와 과거 memories·pointers를 융합해 memory-conditioned feature `F_t_tilde`를 만든다.
3. 현재 프레임에 새 prompt가 없다면, 과거 memory가 사실상 target prompt 역할을 한다.
4. Mask decoder가 후보 `M_t^1`, `M_t^2`, `M_t^3`, predicted IoU, object score, mask token을 출력한다.
5. 한 후보 `M_hat_t`를 선택한다.
6. `F_t`와 `M_hat_t`를 memory encoder에 넣어 새 memory `E_t`를 만든다.
7. `E_t`와 pointer를 bank에 넣고, recent queue가 가득 찼다면 가장 오래된 recent memory를 제거한다.

### 4.4 중간 프레임에서 사용자가 수정하는 경우

프레임 20에서 모델이 강아지를 놓쳤다고 하자. 사용자가 강아지 위에 positive click을 추가하면 다음이 가능하다.

1. Frame 20 mask를 다시 계산한다.
2. Frame 20이 새 conditioning frame이 된다.
3. 이 수정 정보를 사용해 이후 방향뿐 아니라 도구 설정에 따라 이전 방향으로도 다시 전파할 수 있다.

SAM2의 장점은 correction을 받을 때 매번 처음부터 새 tracker를 시작하지 않고 기존 object memory 문맥을 이용한다는 점이다.

### 4.5 여러 객체를 동시에 분할하는 경우

기본 SAM2는 모든 객체가 Hiera image feature는 공유하지만, 그 이후의 memory bank와 mask decoding은 객체별로 따로 수행한다. 객체 A와 B가 서로 겹친다는 사실을 명시적으로 토론하는 공동 memory가 기본 구조에는 없다.

따라서 다음 문제가 생길 수 있다.

- 두 object mask가 같은 distractor를 동시에 선택한다.
- A와 B가 교차할 때 ID를 맞바꾼다.
- 각 object score는 높지만 전체적으로 mask가 겹친다.

이는 multi-object association이나 inter-object communication이 별도 연구 주제가 되는 이유다.

---

## 5. 메모리 관리 용어를 정확히 이해하기

### 5.1 Autoregression: 오늘의 답이 내일의 입력이 되는 구조

SAM2의 비디오 추론은 넓은 의미의 autoregressive process다.

\[
\hat M_t=f(I_t,\;P_t,\;\mathcal{B}_{t-1})
\]

\[
\mathcal{B}_{t}=Update(\mathcal{B}_{t-1},\;Encode(I_t,\hat M_t))
\]

- `I_t`: 현재 영상 프레임
- `P_t`: 현재 프레임 prompt, 없을 수도 있음
- `B_(t-1)`: 이전까지의 memory bank
- `M_hat_t`: 모델이 고른 현재 mask

현재 예측 `M_hat_t`가 memory를 바꾸고, 바뀐 memory가 다음 예측 `M_hat_(t+1)`에 영향을 준다. 언어 모델이 앞에서 만든 token을 뒤 token 생성에 쓰는 것과 비슷하지만, 여기서는 mask-conditioned memory가 전달된다.

### 5.2 FIFO: 먼저 들어온 recent memory가 먼저 나간다

FIFO는 First-In, First-Out이다. Recent queue가 `[F2, F3, F4]`로 가득 찬 상태에서 `F5`가 들어오면 가장 오래된 `F2`가 빠져 `[F3, F4, F5]`가 된다.

장점:

- 구현이 단순하다.
- 계산량과 VRAM이 영상 길이와 함께 끝없이 증가하지 않는다.
- 최근 외형·최근 위치 변화에 잘 적응한다.

단점:

- 최근 프레임이 가림·blur·오답으로 가득하면 좋은 기억이 빠르게 밀려난다.
- 오래전의 선명한 target 모습이 첫 prompt frame 하나를 제외하고 사라질 수 있다.
- `최근`과 `좋은 품질`은 같은 개념이 아니다.

### 5.3 Anchor memory

Anchor는 target identity를 잃지 않게 붙잡아 두는 기준 기억이다. 논문마다 이름과 구현이 다르며, 기본 SAM2의 공식 모듈명은 아니다.

- 첫 prompt frame을 항상 남기는 가장 단순한 anchor
- 가림 전 고품질 frame을 추가 anchor로 보존
- 여러 시점·각도·크기의 대표 frame을 anchor set으로 보존
- 이미지 feature 대신 object prototype이나 text concept를 anchor로 사용

Anchor가 너무 오래되면 현재 외형과 달라지는 **anchor staleness**가 생긴다. 반대로 자동으로 anchor를 자주 갱신하면 오답이 anchor가 되는 위험이 있다.

### 5.4 Short-term memory와 long-term memory

| 종류 | 주목적 | 보통 담는 것 | 장점 | 위험 |
|---|---|---|---|---|
| Short-term | 최근 변화와 운동에 빠르게 적응 | 직전 몇 프레임 | 자세·크기·조명 변화 반영 | 가림이 이어지면 오답으로 모두 오염될 수 있음 |
| Long-term | 오래전의 안정적인 identity 보존 | 첫 frame, 고품질·다양한 keyframe | 재등장과 장기 외형 비교에 유리 | 오래되어 현재 모습과 달라질 수 있고 selection이 어려움 |

기본 SAM2에는 prompted conditioning memory와 recent FIFO memory가 있다. 많은 후속 연구가 이를 더 명시적인 short/long, recent/anchor, appearance/distractor memory로 재구성한다.

### 5.5 Memory admission, write, read, retention, eviction, pruning

이 용어들을 섞지 말아야 연구 질문이 선명해진다.

| 용어 | 질문 | 예시 |
|---|---|---|
| Write | 현재 정보를 memory 형태로 만들고 기록한다 | Frame 8 mask와 image feature를 encode해 bank에 넣음 |
| Admission | 이 후보를 아예 쓸 자격이 있는가? | predicted IoU가 높고 motion과 일치할 때만 write |
| Read/Retrieval | 현재 frame이 어느 memory를 얼마나 참고하는가? | 최근 3개와 anchor 2개만 attention에 사용 |
| Retention | 이미 쓴 memory 중 무엇을 계속 보존하는가? | 첫 frame과 다양한 고품질 frame 유지 |
| Eviction | 공간이 부족할 때 무엇을 완전히 제거하는가? | 가장 오래되거나 중복된 frame 삭제 |
| Pruning | memory 내부의 frame/token 일부를 줄인다 | target과 관련 없는 background token 제거 |
| Fusion | 여러 memory의 정보를 어떻게 합치는가? | short/long 값을 uncertainty gate로 가중합 |
| Refresh/Re-anchor | 낡은 기준을 새 기준으로 갱신하는가? | 재등장 후 검증된 모습으로 새 anchor 추가 |

예를 들어 `memory 개선`이라는 말만으로는 부족하다. 다음처럼 구체화해야 한다.

> 본 연구는 mask 품질과 가림 상태를 이용해 **admission**을 제어하고, short/long memory의 **retention**을 분리하며, 재등장 때 prototype 기반 **retrieval**을 수행한다.

### 5.6 Memory 정책을 설계할 때의 네 가지 상충 관계

1. **적응성 대 안정성**: 자주 쓰면 최신 외형에는 적응하지만 오답도 빨리 배운다.
2. **최근성 대 대표성**: 최근 frame은 현재와 가깝지만 선명하고 정체성을 잘 보여 준다는 보장은 없다.
3. **정확도 대 계산량**: 많은 memory와 후보를 읽으면 강건할 수 있지만 느려진다.
4. **빠른 복구 대 오탐 억제**: 재탐색 threshold를 낮추면 빨리 찾지만 distractor를 target으로 채택할 수 있다.

---

## 6. 실패 용어와 인과 사슬

### 6.1 용어 정의

| 용어 | 정확한 뜻 | 예시 |
|---|---|---|
| Occlusion, 가림 | 목표가 장면에는 있지만 다른 물체 뒤에 일부 또는 전부 가려짐 | 사람이 기둥 뒤를 지나감 |
| Partial occlusion | 일부 픽셀만 보임 | 사람 상체만 보임 |
| Full occlusion | 장면에 있지만 현재 RGB에서 목표 픽셀이 전혀 보이지 않음 | 공이 손 뒤에 완전히 숨음 |
| Disappearance / out-of-view | 목표가 화면 밖으로 나가 현재 프레임에 존재하지 않음 | 자동차가 화면 오른쪽 밖으로 나감 |
| Reappearance | 가림 또는 화면 이탈 후 다시 보임 | 다른 쪽에서 자동차가 다시 등장 |
| Distractor | 목표와 헷갈릴 수 있는 비슷한 물체·부분·배경 | 같은 유니폼의 다른 선수, 셔츠의 로고 |
| Drift | 예측이 시간에 따라 목표에서 서서히 벗어남 | 사람 전체→셔츠→옆 사람으로 이동 |
| Identity switch | 추적 대상 ID가 다른 개체로 바뀜 | 선수 7번 대신 선수 9번을 계속 추적 |
| Memory contamination | 틀린/부분/부재 frame을 목표 기억으로 저장해 bank가 오염됨 | 다른 선수의 feature를 target memory로 기록 |
| Error accumulation | 앞 프레임 오류가 다음 프레임 조건이 되어 연쇄 확대됨 | 한 번의 잘못된 mask가 여러 bad memory를 생성 |
| Target loss | 모델이 더 이상 원래 target을 안정적으로 지역화하지 못함 | 재등장해도 찾지 못함 |

### 6.2 가장 중요한 인과 사슬

```text
가림·빠른 운동·blur·유사 객체 출현
        ↓
현재 프레임의 target 시각 증거가 약해짐
        ↓
여러 candidate mask가 모호해지거나 모두 같은 오답으로 치우침
        ↓
predicted IoU / object score가 오답을 충분히 경고하지 못함
        ↓
틀린 mask가 선택됨
        ↓
Memory encoder가 '틀린 위치의 image feature + 틀린 mask'를 target memory로 결합
        ↓
Memory contamination
        ↓
다음 프레임 memory attention이 틀린 영역을 더 강하게 참고
        ↓
다음 mask도 틀릴 확률 증가
        ↓
새로운 bad memory가 다시 기록됨
        ↓
FIFO에서 오래된 good recent memory가 밀려남
        ↓
Drift가 커지고 distractor로 identity switch
        ↓
원래 target이 reappear해도 오염된 memory 때문에 무시하거나 다른 개체를 계속 추적
```

### 6.3 왜 predicted IoU가 높은데 identity switch가 가능한가?

Predicted IoU head가 주로 답하려는 질문은 `이 mask의 모양과 경계가 그럴듯한가?`에 가깝다. 하지만 tracking에서 더 중요한 질문은 `이 mask가 처음 지정한 바로 그 개체인가?`이다.

같은 유니폼을 입은 다른 선수를 아주 정확한 경계로 분할하면 다음이 동시에 가능하다.

- mask boundary quality: 높음
- predicted IoU: 높을 수 있음
- target identity correctness: 틀림

따라서 identity를 지키려면 predicted IoU 외에도 appearance prototype, object pointer consistency, motion, neighbor/distractor 관계, forward-backward consistency 같은 신호가 필요하다.

### 6.4 Object score가 있는데도 왜 부재 처리가 어려운가?

Object score도 학습된 추정값이다. 다음 두 상황의 영상 픽셀이 비슷할 수 있다.

- target이 거의 보이지 않지만 작은 일부는 남아 있음
- target은 사라졌고 비슷한 distractor만 있음

너무 민감하게 absent를 선언하면 어려운 target을 놓치고, 너무 둔감하면 distractor를 target으로 memory에 쓴다. 그래서 최신 연구는 `보임/부재` 두 상태보다 `visible/uncertain/absent` 세 상태 또는 uncertainty-aware soft gate를 사용하려 한다.

### 6.5 Failure와 memory contamination 중 무엇이 원인인가?

항상 한 방향은 아니다.

- 첫 오답 mask가 contamination을 만든다.
- contamination이 이후 오답 확률을 높인다.
- 이후 오답이 더 많은 contamination을 만든다.

즉, feedback loop다. 연구에서는 이를 끊는 위치를 명시해야 한다.

- Decoder 단계: 더 좋은 후보 생성·선택
- Admission 단계: 의심 frame을 쓰지 않기
- Retention 단계: 오래된 좋은 기억 지키기
- Read 단계: 오염 가능 memory의 가중치 줄이기
- Recovery 단계: target을 전역 재탐색하고 재프롬프트하기

---

## 7. 12프레임 장난감 예시: 한 번의 가림이 identity switch가 되는 과정

### 7.1 상황과 표기

영상에는 처음 지정한 **빨간 공 A**와 매우 비슷한 **빨간 공 B**가 있다. 설명을 단순화하기 위해 recent memory를 3칸만 둔다. 실제 공개 SAM2 기본 설정은 전형적으로 first input frame과 최근 6개 frame을 사용한다.

- `A1`, `A2`: 해당 frame의 올바른 공 A memory
- `P4`: 부분 가림된 불완전한 A memory
- `B5`: distractor B를 target이라고 잘못 저장한 memory
- `Anchor A1`: 사용자가 지정한 첫 frame memory, recent FIFO와 별도 보존
- `[x,y,z]`: 오래된 것에서 최신 순서의 toy recent FIFO

### 7.2 프레임별 진행

| Frame | 실제 장면 | 기본 SAM2식 탐욕·FIFO에서 가능한 진행 | Toy recent memory | 상태 기반 개선 정책 |
|---:|---|---|---|---|
| 1 | 사용자가 공 A를 클릭. A가 선명함 | A mask 생성, 첫 conditioning memory와 pointer 저장 | `[]`, Anchor=`A1` | 동일. 장기 identity prototype도 A1에서 생성 |
| 2 | A가 오른쪽으로 조금 이동 | A2를 정확히 선택하고 write | `[A2]` | reliable visible이므로 short memory에 write |
| 3 | A가 더 이동, 여전히 선명 | A3를 write | `[A2,A3]` | A2와 다른 각도라면 장기 keyframe 후보로 보존 가능 |
| 4 | 손이 A의 70%를 가림 | 작은 부분 mask P4의 predicted IoU가 가장 높아 선택·write될 수 있음 | `[A2,A3,P4]` | area 급감·후보 disagreement·motion residual로 `uncertain`; 출력할 수는 있지만 write는 보류 |
| 5 | A가 손 뒤에 완전히 가려짐. B는 화면에 보임 | object score가 B를 target으로 오판하면 B5 선택·write, 가장 오래된 A2가 빠짐 | `[A3,P4,B5]` | `absent`로 판단해 empty output 또는 출력 보류. Memory와 motion update를 freeze |
| 6 | A는 여전히 안 보이고 B만 움직임 | 오염 memory B5가 B6 선택을 강화 | `[P4,B5,B6]` | absent 유지, memory write 없음. 전역 재탐색은 하되 채택하지 않음 |
| 7 | A는 아직 가려지고 B가 anchor A1과 비슷한 자세 | B7 선택. 마지막 좋은 recent A3가 FIFO에서 사라짐 | `[B5,B6,B7]` | 장기 A prototype은 보존. B 후보는 appearance는 비슷해도 가림 전 trajectory와 불일치하여 보류 |
| 8 | A가 멀리 떨어진 오른쪽에 재등장, B도 왼쪽에 존재 | 세 recent memory가 모두 B라 memory attention이 B8을 고름. Identity switch 성립 | `[B6,B7,B8]` | presence 변화가 reappearance search를 trigger. Full-frame 후보 중 A1 prototype·motion·cycle consistency가 가장 맞는 A8 선택. 아직 즉시 memory write하지 않음 |
| 9 | A와 B 모두 선명 | 기본 모델은 B9를 계속 추적 | `[B7,B8,B9]` | A8→A9가 2프레임 일관되면 rescue를 검증하고 A9를 short memory에 admission |
| 10 | 사용자가 기본 모델 오류를 보고 A에 새 click | 새 conditioning prompt 덕분에 기본 모델도 A10으로 회복 가능 | B recent는 남아 있지만 새 Anchor=`A10` | 자동 복구 모델은 이미 A를 추적. 새 user prompt가 있다면 가장 강한 anchor로 반영 |
| 11 | A가 빠르게 이동 | 새 conditioning memory로 A11 가능 | `[B8,B9,A11]` | motion residual은 크지만 appearance·cycle이 맞으면 reliable write |
| 12 | A가 안정적으로 보임 | B memory가 FIFO에서 점차 빠지며 정상화 | `[B9,A11,A12]` | A short/long memory가 모두 정상. 자동 복구 성공 |

### 7.3 이 예시에서 측정해야 할 것은 평균 IoU만이 아니다

- Frame 4: bad-write를 사전에 탐지했는가?
- Frame 5–7: target 부재 중 false-positive mask를 얼마나 출력했는가?
- Frame 8: reappearance를 몇 frame 만에 알아냈는가?
- Frame 8–9: 공 B로 identity switch 했는가?
- Frame 10: 사람의 correction이 없으면 회복 가능한가?
- Recent memory 중 실제 target이 아닌 entry 비율은 얼마인가?

이런 사건 기반 지표가 있어야 `평균 J&F가 0.5 올랐다`보다 왜 개선되었는지 설명할 수 있다.

---

## 8. 주요 후속 연구: 문제–아이디어–남은 한계

### 8.1 한눈에 보는 지도

| 방법 | 주로 건드리는 지점 | 초보자용 한 줄 비유 |
|---|---|---|
| SAM2Long | 후보 선택 + 각 경로별 memory | 하나의 추적 일기 대신 가능성 높은 여러 일기를 병렬로 써 둔다. |
| DAM4SAM | Admission + recent/anchor memory | 최근 사진첩과 범인 식별용 기준 사진첩을 따로 둔다. |
| SAMURAI | Motion + memory selection | 시각 정보에 속도·방향을 알려 주는 GPS를 붙인다. |
| UAMP | Appearance/motion uncertainty + short/long memory | 자신 없는 부분은 과거와 현재를 다르게 섞고 장·단기 기억을 나눈다. |
| OASAMT | Explicit occlusion state + dual memory | 보임·가림 상태를 먼저 판정한 뒤 상태별 서랍에 기억을 넣는다. |
| SENTRY | Candidate verification before write | 일기에 쓰기 전에 앞뒤 장면과 맞는지 역검산한다. |
| Lean-SAM2 | Target-anchored pruning + insurance + sparse routing | 목표 기준 사진으로 쓸모없는 기억·계산을 줄이되 비상용 기억을 남긴다. |
| SAM2Dual | Short/long + text-aware memory | 작업 기억과 장기 기억을 나누고 단어 힌트로 잘못된 기억을 약화한다. |
| StreamDAM | Presence-controlled streaming policy | 목표가 있는지에 따라 쓰기·읽기·침묵·재탐색을 한 신호로 제어한다. |
| RUAC | Pixel-wise uncertainty calibration | 모델의 자신감 온도계를 실제 오류와 맞게 교정한다. 영상 memory 연구는 아니다. |

### 8.2 SAM2Long

출처: [SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html)

**문제**  
기본 SAM2는 현재 predicted IoU가 가장 높은 mask 하나를 즉시 선택한다. 그 선택이 틀리면 다른 후보는 버려져 되돌아가기 어렵다.

**아이디어**  
고정된 수의 segmentation path를 유지한다. 각 path가 현재 프레임의 여러 mask 후보로 분기하고, 누적 log predicted-IoU가 좋은 path만 beam search처럼 남긴다. Occlusion score가 불확실할 때 비슷한 path만 남지 않도록 mask diversity도 고려하고, 품질 좋은 object memory를 선택한다.

**SAM2보다 나아지는 이유**  
프레임 하나의 판단을 즉시 확정하지 않고 몇 frame 뒤 증거를 보고 올바른 경로가 살아남을 수 있다. 가림 뒤 target이 재등장했을 때 과거에 남겨 둔 다른 가설로 복구할 가능성이 생긴다.

**남은 한계**  
모든 path가 같은 distractor로 치우치면 도움이 되지 않는다. 점수 자체가 SAM2 predicted IoU에 많이 의존하며, 저자도 동적 배경·유사 옷에서 semantic identity 부족을 failure로 제시한다. 여러 path 계산과 영상 종료 후 최종 경로 선택은 엄격한 online streaming에 부담이다.

### 8.3 DAM4SAM

출처: [A Distractor-Aware Memory for Visual Object Tracking with SAM2 — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html)

**문제**  
기본 recent FIFO는 target이 없는 frame도 밀어 넣을 수 있고, 유사 distractor가 나타났을 때 최근 오답이 bank를 차지한다.

**아이디어**  
RAM, Recent Appearance Memory와 DRM, Distractor Resolving Memory를 분리한다. RAM은 최근 정상 외형 적응용이고, DRM은 첫 frame과 신뢰도 높은 anchor를 보존한다. 선택 mask와 alternative mask의 차이를 tracker introspection 신호로 사용해 distractor 상황을 판단하고, object presence·predicted IoU·면적 안정성을 통과한 frame만 갱신한다.

**SAM2보다 나아지는 이유**  
`가장 최근`이 아니라 `최근이면서 믿을 수 있음`과 `distractor 해결에 유용함`을 구분한다. Bad frame이 들어오는 것을 줄이고 좋은 기준 frame을 FIFO에서 보호한다.

**남은 한계**  
여러 threshold가 수작업 값이다. SAM2의 모든 alternative mask가 같은 잘못된 개체를 가리키면 disagreement 신호가 약해진다. 같은 물체의 일부와 전체를 혼동하는 internal distractor도 완전히 해결되지 않는다.

### 8.4 SAMURAI

출처: [SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory — arXiv 2024](https://arxiv.org/abs/2411.11922), [공식 코드](https://github.com/yangchris11/samurai)

**문제**  
SAM2는 위치·속도라는 명시적인 물리 운동 모델이 없고, fixed-window memory가 품질보다 최근성에 치우친다.

**아이디어**  
Kalman filter가 target box의 위치와 크기를 예측한다. SAM2 후보 mask가 예측 운동과 얼마나 맞는지 motion score를 만들고, mask affinity·object score와 함께 후보와 memory를 선택한다. 추가 학습이 필요 없다.

**SAM2보다 나아지는 이유**  
비슷하게 생긴 객체가 있어도 갑자기 먼 곳에 나타난 distractor보다 이전 속도·방향에서 자연스럽게 이어지는 후보를 선호한다.

**남은 한계**  
Kalman filter의 단순 운동 가정은 급회전, 카메라 컷, 심한 변형, 긴 완전 가림에 약하다. 주요 결과가 box 기반 VOT benchmark이고 peer-reviewed venue가 확인되지 않은 preprint다.

### 8.5 UAMP

출처: [UAMP: Consistent Video Object Segmentation with Uncertainty-Aware Memory Propagation — PLOS One 2026](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353156)

**문제**  
Occlusion·blur·빠른 운동에서는 frame의 일부 영역만 크게 변한다. 기본 SAM2는 어디가 불확실한지 세밀하게 표현하지 못하고 short recent memory만으로 장기 일관성을 잃을 수 있다.

**아이디어**  
작은 convolution module이 appearance uncertainty map을 예측한다. 불확실한 영역과 안정된 영역에서 현재·memory feature를 다른 비율로 섞는다. Motion uncertainty, score-based short-term selection, recurrent linear-attention long-term memory도 결합한다.

**SAM2보다 나아지는 이유**  
프레임 전체를 동일하게 믿지 않고 위치별 불확실성을 사용하며, 최근 적응과 장기 identity 보존을 분리한다.

**남은 한계**  
Uncertainty module과 long-term memory를 학습해야 한다. 여러 모듈이 동시에 들어가므로 어느 요소가 향상을 만들었는지 엄격한 ablation이 필요하다. 어려운 조건의 정밀 경계 문제도 남는다.

### 8.6 OASAMT

출처: [Occlusion-aware Visual Object Tracking with Explicit Temporal State Modeling and Dual-memory Mechanism — Pattern Recognition 2026](https://www.sciencedirect.com/science/article/pii/S0031320326011209)

**문제**  
SAM2는 object score 한 번으로 상태를 판단하지만, 가림은 시간에 걸쳐 시작·유지·종료된다. Occluded frame을 정상 frame과 같은 bank에 넣으면 identity가 오염된다.

**아이디어**  
Temporal Occlusion Classifier가 confidence, mask IoU, area ratio의 시계열을 보고 occlusion state를 판정한다. Temporal Occlusion Predictor는 가림 중에도 box trajectory를 예측해 재등장 위치 prior를 만든다. Non-Occlusion Memory Bank와 Occlusion Memory Bank를 분리한다.

**SAM2보다 나아지는 이유**  
현재 frame의 한 점수 대신 여러 frame의 변화 패턴으로 상태를 판단하고, 정상 identity memory를 가림 memory와 분리한다.

**남은 한계**  
TCN 학습과 occlusion label이 필요하다. 주요 목적과 평가는 box VOT에 가깝고, 새로운 domain에서는 학습된 상태 분류기가 틀릴 수 있다.

### 8.7 SENTRY

출처: [SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking — arXiv 2026](https://arxiv.org/abs/2606.24449), [ECCV 2026 채택·공식 코드](https://github.com/HamadYA/SENTRY)

**문제**  
Confidence가 높은 현재 mask라고 해서 시간적으로 올바른 target이라는 보장은 없다. 오답을 memory에 쓴 뒤에는 되돌리기 어렵다.

**아이디어**  
`Refine before write`를 적용한다. SAM2 decoder 후보, automatic mask 후보, Kalman fallback으로 candidate pool을 만든다. 각 후보를 짧은 시간 뒤에서 앞으로가 아니라 현재에서 과거로 backtrack해 최근 target trajectory와 cycle-consistent한지 검사한다. 주변 distractor trajectory와 더 잘 맞는 후보는 벌점을 주고, 엄격히 검증된 severe rescue만 memory에 쓴다.

**SAM2보다 나아지는 이유**  
현재 frame confidence만 보지 않고 여러 frame에 걸쳐 `이 후보를 과거로 되돌렸을 때 원래 target과 연결되는가?`를 검산한다.

**남은 한계**  
후보 생성·역추적으로 계산량이 늘고 공식 결과에서 대형 모델 FPS가 약 23–26% 감소한다. 매우 긴 disappearance로 연결할 short tracklet 자체가 없거나 candidate generator가 target을 전혀 제안하지 못하면 구조상 복구가 어렵다.

### 8.8 Lean-SAM2

출처: [Lean-SAM2: Target-Anchored Memory and Encoder Acceleration for SAM2 — arXiv v1, 2026-07-22](https://arxiv.org/abs/2607.19811), [공식 코드](https://github.com/DeawhaleQwQ/Lean-SAM2)

**문제**  
SAM2는 full-frame Hiera encoding과 memory cross-attention이 무겁다. 단순 attention-based pruning이나 window skipping은 occlusion·distractor 상황에서 중요한 target token을 버려 정확도가 크게 떨어질 수 있다.

**아이디어**

- TAMP, Target-Anchored Memory Pruning: 첫 prompt foreground에서 만든 anchor와 semantic consistency가 높은 memory token을 보호한다.
- TCIM, Temporal Condensation with Insurance Memory: 과거를 visibility gate로 압축하면서 고품질 entry는 별도의 insurance bank에 보관한다.
- TARR, Target-Anchored Risk-Aware Routing: target과 관련된 image window에 heavy encoder 계산을 집중하고, 위험한 변화가 감지되면 full-frame refresh로 되돌아간다.

**SAM2보다 나아지는 이유**  
단순히 계산을 줄이는 것이 아니라 target anchor와 비상 memory를 사용해 `잘못 줄여서 target을 잃는 위험`을 같이 관리한다.

**남은 한계**  
조사일 기준 매우 최근의 arXiv preprint다. 첫 prompt anchor가 시간이 지나 외형 변화로 낡으면 관련 token을 잘못 제거하거나 full-frame fallback이 자주 발생해 속도 이득이 줄 수 있다. 보고된 속도·정확도는 독립 재현이 필요하다.

### 8.9 SAM2Dual

출처: [SAM2Dual: Training-Free, Dual Memory for Long-Term Video Object Segmentation — arXiv v1, 2026-08-19](https://arxiv.org/abs/2608.18640)

**문제**  
최근 unreliable prediction이 streaming memory를 지배하면 긴 occlusion·scene change 뒤 identity가 무너진다.

**아이디어**  
빠른 local adaptation용 short-term memory와 간격 sampling으로 identity를 남기는 long-term memory를 분리해 gated fusion한다. 초반 frame에서 compact word cue를 뽑고 text embedding으로 memory 기여도를 재가중하는 Text-Aware Memory도 제안한다. Training-free plug-in 방식이다.

**SAM2보다 나아지는 이유**  
최근 오답이 모든 기억을 덮는 것을 줄이고, 시각 증거가 약할 때 semantic cue로 관련 memory를 강조한다.

**남은 한계**  
조사 시점 기준 공개 이틀 된 초기 v1 preprint다. `공`, `사람` 같은 category-level word는 같은 class의 서로 다른 instance를 구분하지 못한다. 논문 failure에서도 `신발` target이 사람 전체로 퍼지는 part–whole drift가 나타난다.

### 8.10 StreamDAM

출처: [StreamDAM: Presence-Aware Memory for Real-Time Streaming Video Object Segmentation — arXiv v1, 2026-08-04](https://arxiv.org/abs/2608.03912)

**문제**  
Offline benchmark는 frame 하나를 처리하는 데 시간이 얼마나 걸리든 기다린 뒤 결과를 평가한다. 실제 30 FPS stream에서는 deadline을 놓치면 과거 mask를 다시 내보내야 하고, target이 없는데도 mask를 출력하는 문제가 중요해진다.

**아이디어**  
학습된 presence signal 하나가 다음을 frame별로 제어한다.

- 어떤 frame을 memory에 쓸지
- memory를 얼마나 멀리 읽을지
- 현재 출력을 보류할지
- 언제 global re-detection할지

**SAM2보다 나아지는 이유**  
Fixed policy 대신 `지금 target이 실제로 사라진 것인지, 단지 보기 어려운 것인지`에 따라 계산과 출력을 바꾼다. Accuracy와 real-time deadline을 함께 본다.

**남은 한계**  
최근 단독 저자의 v1 preprint로 peer review와 외부 재현이 없다. Presence model이 틀리면 write·read·output·re-detection 네 동작이 동시에 잘못될 수 있다.

### 8.11 RUAC

출처: [Segment Anything with Robust Uncertainty-Accuracy Correlation — ICML 2026 프로젝트](https://hongyouzhou.github.io/ruac/), [arXiv](https://arxiv.org/abs/2605.10603)

**먼저 바로잡을 점**  
RUAC은 SAM2 장기 비디오 memory tracker가 아니다. 주 대상은 domain shift가 있는 **이미지 segmentation의 pixel-wise uncertainty calibration**이다.

**문제**  
SAM2의 mask-level predicted IoU 한 숫자는 경계 일부의 오류와 pixel별 불확실성을 충분히 나타내지 못한다. 새로운 촬영 환경·texture·변형에서는 confidence와 실제 accuracy의 관계가 무너질 수 있다.

**아이디어**  
Lightweight uncertainty head와 Bayesian mask decoder를 추가한다. Style attack과 deformation attack으로 appearance·geometry shift를 만들고, uncertainty가 실제 error pixel에 높아지도록 Uncertainty–Accuracy Alignment로 학습한다.

**SAM2 연구에 쓸 수 있는 이유**  
RUAC uncertainty map을 그대로 쓰거나 더 단순한 대체 신호를 만들어 다음처럼 사용할 수 있다.

- 불확실 pixel 비율이 높으면 memory admission 금지
- 경계 uncertainty만 높으면 mask refinement 실행
- uncertainty가 급증하면 SENTRY식 검증이나 global re-detection 호출

**남은 한계**  
추가 학습이 필요하고, 시간적 identity·occlusion state·reappearance를 직접 모델링하지 않는다. Pixel uncertainty가 높다는 사실만으로 target이 사라졌는지 distractor로 바뀌었는지는 알 수 없다.

### 8.12 추가로 연결되는 연구

| 연구 | 핵심 연결점 | 남은 한계 |
|---|---|---|
| [HiM2SAM](https://arxiv.org/abs/2507.07603), PRCV 2025 | Kalman coarse motion과 불확실 시 CoTracker3 fine motion, short/long memory | Offline point tracker와 여러 threshold, box tracking 중심 |
| [SAMITE](https://arxiv.org/abs/2507.21732), arXiv 2025 | Feature·position prototype으로 best memory 선택, positional pseudo-prompt와 cycle check | Preprint, 초기 prototype이 틀리면 오류 강화 가능 |
| [SAM2RL](https://arxiv.org/abs/2507.08548), RLC 2025 workshop | Admission/eviction을 RL 순차 의사결정으로 최적화 | Video마다 agent를 따로 overfit한 upper-bound라 일반화 tracker가 아님 |
| [IMTrack](https://www.sciencedirect.com/science/article/pii/S0952197626018804), EAAI 2026 | Keyframe으로 re-detection memory를 미리 warm-up하고 loss 후 global search | Warm-up·전역 탐색 비용, box recovery 중심 |
| [SeC](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d3f48777432f64bc96b1203713f20352-Abstract-Conference.html), ICLR 2026 | 새 scene에서 LVLM이 object-centric concept를 만들고 semantic identity를 보강 | 4B급 모델과 학습이 무거우며 same-class instance 구분은 여전히 어려움 |

### 8.13 선행연구를 관통하는 결론

한 방법이 모든 문제를 해결하지는 않는다.

- SAM2Long: 잘못된 결정을 늦춘다.
- DAM4SAM·SENTRY: 잘못된 기억이 들어가는 것을 막는다.
- SAMURAI·HiM2SAM: 위치·운동 정보를 보강한다.
- UAMP·RUAC: 불확실성을 더 잘 표현한다.
- OASAMT·StreamDAM: 대상 상태를 명시적으로 제어한다.
- IMTrack: 잃은 뒤 다시 찾는 과정을 따로 둔다.
- SAM2Dual·SeC: 의미와 장기 identity를 보강한다.
- Lean-SAM2: 정확도를 지키며 memory·encoder 계산을 줄이려 한다.

가장 큰 공백은 **bad write 예방, 명시적 absent 상태, 재등장 후 identity 검증을 하나의 가벼운 구조에서 함께 다루는 것**이다.

---

## 9. 기본 SAM2 흐름의 교육용 의사코드

아래 코드는 실제 Meta 코드의 모든 예외와 tensor 연산을 재현한 것이 아니라, 정보 흐름을 이해하기 위한 의사코드다.

```python
def run_sam2_video(frames, prompts):
    conditioning_memory = {}       # 사용자가 prompt를 준 frame
    recent_memory = FIFO(maxlen=6) # 자동 예측한 최근 frame
    object_pointers = []

    for t, frame in enumerate(frames):
        # 1. 현재 영상 자체의 시각 특징. 같은 frame에서는 재사용 가능
        image_features = hiera_image_encoder(frame)

        # 2. 과거 spatial memory와 object pointer로 현재 특징을 조건화
        conditioned_features = memory_attention(
            current=image_features,
            spatial_memories=list(conditioning_memory.values()) + list(recent_memory.values()),
            object_pointers=object_pointers,
        )

        # 3. 현재 frame에 user prompt가 있으면 encoding
        prompt_features = prompt_encoder(prompts.get(t))

        # 4. 여러 mask, 예상 품질, 존재 점수, mask token 출력
        masks, predicted_ious, object_score, mask_tokens = mask_decoder(
            conditioned_features,
            prompt_features,
            high_resolution_skip=image_features.high_res,
        )

        # 5. 기본적으로 predicted IoU가 가장 높은 후보 선택
        k = argmax(predicted_ious)
        selected_mask = masks[k]
        pointer = project_to_object_pointer(mask_tokens[k])

        # 6. 현재 frame + 선택 mask를 미래용 spatial memory로 encoding
        spatial_memory = memory_encoder(
            unconditioned_image_features=image_features,
            selected_mask=selected_mask,
            object_score=object_score,
        )

        # 7. Prompt frame과 자동 추적 frame을 구분해 저장
        if t in prompts:
            conditioning_memory[t] = spatial_memory
        else:
            recent_memory.push(t, spatial_memory)  # 가득 차면 가장 오래된 recent 제거

        object_pointers.append(pointer)

        # 8. Object가 없다고 판단하면 빈 mask를 표시할 수 있음
        if object_score < PRESENCE_THRESHOLD:
            yield t, empty_mask_like(selected_mask)
        else:
            yield t, selected_mask
```

핵심 위험은 5–7단계다. 5단계에서 오답이 선택되면 6단계가 오답을 target memory로 바꾸고, 7단계가 이를 다음 frame의 조건으로 만든다.

---

## 10. 불확실성·상태 기반 개선안의 의사코드

다음은 여러 선행연구를 베끼는 완성 알고리즘이 아니라, 연구 질문을 분명하게 보여 주는 최소 통합 예시다.

```python
def robust_update(frame, sam2_output, tracker_state):
    masks = sam2_output.masks
    iou_score = sam2_output.predicted_iou
    presence = sam2_output.object_score
    pointer = sam2_output.object_pointer

    best = masks[argmax(iou_score)]

    # SAM2 기본 출력 외의 검증 신호
    area_jump = relative_area_change(best, tracker_state.last_reliable_mask)
    motion_error = distance_to_motion_prediction(best, tracker_state.motion_model)
    pointer_shift = cosine_distance(pointer, tracker_state.identity_pointer)
    disagreement = diversity_between_candidate_masks(masks)
    cycle_error = optional_forward_backward_error(best, tracker_state.history)

    # 예시용 reliability. 계수와 threshold는 실험으로 정해야 함
    reliability = (
        + 1.0 * iou_score.max()
        + 0.8 * sigmoid(presence)
        - 0.7 * area_jump
        - 0.8 * motion_error
        - 0.9 * pointer_shift
        - 0.6 * cycle_error
    )

    state = classify_state(
        presence=presence,
        reliability=reliability,
        disagreement=disagreement,
        recent_states=tracker_state.recent_states,
    )  # VISIBLE, UNCERTAIN, ABSENT

    if state == VISIBLE and reliability > WRITE_THRESHOLD:
        output = best
        write_short_term_memory(best)
        update_motion_model(best)

        if is_high_quality(best) and is_diverse_from_long_term(best):
            retain_as_long_term_anchor(best)

    elif state == UNCERTAIN:
        # 화면에는 출력할 수 있지만, 미래를 오염시키지 않도록 쓰지 않음
        output = best
        freeze_memory_write()
        keep_motion_covariance_growing()

    else:  # ABSENT
        output = EMPTY_MASK
        freeze_memory_write()
        do_not_learn_from_distractors()

        proposals = global_candidate_search(frame)
        rescue = select_by(
            proposals,
            long_term_appearance_similarity=True,
            motion_gate=True,
            cycle_consistency=True,
        )

        if rescue.is_consistent_for_multiple_frames():
            output = reprompt_sam2_with(rescue)
            admit_verified_rescue_to_memory()

    tracker_state.recent_states.append(state)
    return output, tracker_state
```

### 이 의사코드의 연구 가설

> `신뢰도가 낮은 frame은 출력과 memory write를 분리하고, 부재 상태에서는 memory를 동결하며, 재등장 후보를 장기 identity와 시간 일관성으로 검증한 뒤 쓰면 SAM2의 memory contamination과 recovery delay를 동시에 줄일 수 있다.`

---

## 11. 과제연구에서 모듈별로 검증하는 법

한 번에 모든 기능을 넣으면 무엇이 효과가 있었는지 알 수 없다. 다음 순서로 ablation한다.

| 실험 | 추가되는 요소 | 확인할 질문 |
|---:|---|---|
| A0 | SAM2.1 baseline | 기본 오류와 처리 비용은? |
| A1 | Confidence admission | 낮은 predicted IoU frame을 쓰지 않는 것만으로 충분한가? |
| A2 | + area/motion gate | 기하학적 이상을 더하면 bad-write 검출이 좋아지는가? |
| A3 | + pointer consistency | Identity 변화 감지에 도움이 되는가? |
| A4 | + visible/uncertain/absent state | 어려움과 실제 부재를 구분할 수 있는가? |
| A5 | + short/long retention | 재등장 뒤 recovery가 빨라지는가? |
| A6 | + global re-detection | 사람의 correction 없이 회복하는가? |
| A7 | + cycle verification | 빠른 복구의 오탐·identity switch를 줄이는가? |

### 표준 mask 품질 외에 필요한 지표

- J&F와 시간에 따른 J&F decay
- Drift onset frame
- Memory contamination rate
- Bad-write detection AUROC, precision, recall
- Target absence 중 false-positive mask area
- Identity switch count
- Reappearance 후 recovery delay
- Post-reappearance J&F
- User correction 횟수
- FPS, VRAM, frame당 추가 decoder/encoder 호출 수
- 30 FPS deadline-aware J&F

VOS의 J&F와 box VOT의 AUC/AO는 서로 다른 task protocol이므로 한 숫자 순위로 직접 섞지 않는다. 같은 SAM2.1 checkpoint, 같은 prompt, 같은 memory budget 아래에서 비교해야 memory policy 효과를 공정하게 볼 수 있다.

---

## 12. 자주 생기는 오해 바로잡기

### 오해 1. Predicted IoU가 0.95면 95% 확률로 정답이다

아니다. 실제 정답과의 IoU를 추정하도록 학습된 score이지, 모든 domain에서 완벽히 calibrated된 정답 확률이 아니다. Identity correctness와도 다르다.

### 오해 2. Object pointer에는 항상 올바른 target identity가 들어 있다

아니다. 선택된 mask token에서 나오므로 mask가 틀리면 pointer도 오염될 수 있다.

### 오해 3. Memory가 많을수록 항상 좋다

좋은 memory라면 도움이 되지만, 오염되거나 중복된 memory를 많이 읽으면 느려지고 오답 attention도 커질 수 있다.

### 오해 4. Object score가 있으므로 full occlusion 문제는 해결되었다

아니다. Object score 자체가 어려운 장면에서 오판할 수 있고, absent를 알아도 재등장 위치·identity를 자동으로 찾는 별도 과정이 필요하다.

### 오해 5. SAM2.1은 SAM2와 완전히 다른 세대의 구조다

아니다. 큰 구조는 같고 checkpoint, augmentation, longer-sequence training, positional/memory 설정이 강화된 업데이트다.

### 오해 6. SAMEO는 SAM2의 occlusion tracker다

아니다. [SAMEO](https://openaccess.thecvf.com/content/CVPR2025/html/Tai_Segment_Anything_Even_Occluded_CVPR_2025_paper.html)는 정지 이미지에서 보이지 않는 부분까지 추론하는 amodal segmentation 연구다. Temporal drift·reappearance tracker가 아니다.

### 오해 7. RUAC만 붙이면 장기 추적 문제가 해결된다

아니다. RUAC은 불확실성 측정을 개선하지만 temporal state와 re-detection은 별도로 설계해야 한다.

### 오해 8. Training-free는 아무 데이터도 사용하지 않았다는 뜻이다

아니다. 이미 사전학습된 SAM2 weight를 그대로 두고 추가 fine-tuning을 하지 않는다는 뜻이다. Threshold tuning, external pretrained model, benchmark 선택은 여전히 결과에 영향을 준다.

### 오해 9. Zero-shot VOS는 첫 frame 정보도 전혀 안 준다

문맥에 따라 다르다. Semi-supervised VOS는 첫 frame mask를 inference-time supervision으로 준다. `Zero-shot`은 보통 target dataset에 추가 학습하지 않았다는 뜻이지 prompt가 없다는 뜻이 아니다.

---

## 13. 핵심 용어 사전

| 용어 | 한 문장 정의 |
|---|---|
| Mask | 각 픽셀이 목표 영역인지 나타내는 2차원 지도 |
| Mask logits | sigmoid/threshold 전의 실수형 mask 점수 |
| Embedding | 이미지·prompt·객체 정보를 신경망이 계산하기 좋은 벡터로 표현한 것 |
| Token | Transformer가 attention하는 벡터 단위 |
| Feature map | 위치별 feature vector가 놓인 공간 지도 |
| Prompt | 분할할 목표를 지정하거나 수정하는 점·box·mask 등의 입력 |
| Candidate mask | 모호성을 표현하기 위해 동시에 생성한 여러 가능한 mask |
| Predicted IoU | 모델이 예측한 mask 품질 추정값 |
| Object score | 현재 frame에서 목표가 보인다고 판단하는 정도 |
| Object pointer | 선택 mask token에서 만든 고수준 target 요약 벡터 |
| Memory attention | 현재 frame feature가 과거 spatial memory와 pointer를 참고하는 모듈 |
| Memory encoder | 현재 image feature와 선택 mask를 미래용 spatial memory로 바꾸는 모듈 |
| Conditioning frame | 사용자가 직접 prompt를 준 기준 frame |
| Non-conditioning frame | 모델이 자동 전파해 예측한 frame |
| Masklet | 한 객체의 시간에 따른 mask sequence |
| Calibration | confidence 또는 uncertainty 값이 실제 정확도·오류 빈도와 맞게 대응하는 성질 |
| Re-detection | target을 잃은 뒤 local 영역이 아니라 넓은 화면에서 다시 찾는 과정 |
| Cycle consistency | 앞으로 추적한 후보를 다시 뒤로 추적했을 때 출발 target으로 돌아오는지 보는 검증 |

---

## 14. 최종 요약

SAM2의 핵심은 단순히 `SAM + 이전 mask`가 아니다.

1. Hiera가 현재 frame의 multiscale 시각 특징을 만든다.
2. Memory attention이 이를 과거 spatial memory와 object pointer로 조건화한다.
3. Prompt encoder와 mask decoder가 여러 mask, predicted IoU, object score, pointer를 만든다.
4. 선택된 mask와 현재 image feature를 memory encoder가 새 target memory로 바꾼다.
5. Streaming bank가 이 memory를 다음 frame에 제공한다.

강점과 약점은 같은 autoregressive loop에서 나온다. 올바른 기억은 target을 오래 유지하지만, 잘못된 mask를 memory에 쓰면 오답도 스스로 강화된다. 따라서 개선 연구의 핵심 질문은 `decoder를 더 정확하게 만들자` 하나가 아니다.

- 어떤 frame을 쓸 것인가?
- 어떤 기억을 오래 남길 것인가?
- 어떤 기억을 얼마나 읽을 것인가?
- target이 실제로 없는 동안 무엇을 하지 말아야 하는가?
- 다시 나타났을 때 어느 후보가 같은 identity인지 어떻게 검증할 것인가?
- 이 모든 과정의 confidence와 uncertainty를 어떻게 믿을 만하게 만들 것인가?

이 질문들을 admission, retention, read, state control, recovery로 나누면 SAM2Long부터 StreamDAM·RUAC까지의 연구가 하나의 지도 위에 정리된다.

---

## 15. 원문과 공식 자료

### 기본 모델

- [Segment Anything — ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)
- [SAM 2: Segment Anything in Images and Videos — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html)
- [SAM2 공식 코드와 SAM2.1 checkpoint/config](https://github.com/facebookresearch/sam2)
- [Meta 공식 SAM2.1 업데이트 설명](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-sona/)

### 장기 추적·메모리·가림·재등장

- [SAM2Long — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html)
- [DAM4SAM — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html)
- [SAMURAI — arXiv 2024](https://arxiv.org/abs/2411.11922)
- [UAMP — PLOS One 2026](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0353156)
- [OASAMT — Pattern Recognition 2026](https://www.sciencedirect.com/science/article/pii/S0031320326011209)
- [SENTRY — arXiv/ECCV 2026](https://arxiv.org/abs/2606.24449)
- [HiM2SAM — PRCV 2025](https://arxiv.org/abs/2507.07603)
- [SAMITE — arXiv 2025](https://arxiv.org/abs/2507.21732)
- [SAM2RL — RLC workshop 2025](https://arxiv.org/abs/2507.08548)
- [IMTrack — EAAI 2026](https://www.sciencedirect.com/science/article/pii/S0952197626018804)

### 최신 효율·장기·불확실성 방향

- [Lean-SAM2 — arXiv 2026](https://arxiv.org/abs/2607.19811)
- [SAM2Dual — arXiv 2026](https://arxiv.org/abs/2608.18640)
- [StreamDAM — arXiv 2026](https://arxiv.org/abs/2608.03912)
- [RUAC — ICML 2026](https://hongyouzhou.github.io/ruac/)
- [SeC — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d3f48777432f64bc96b1203713f20352-Abstract-Conference.html)

> 출판 상태 주의: Lean-SAM2, SAM2Dual, StreamDAM은 조사 시점에 매우 최근 공개된 arXiv preprint다. 아이디어 동향으로는 유용하지만, 핵심 결론은 peer-reviewed 연구와 독립 재현 결과를 중심으로 세우는 것이 안전하다.
