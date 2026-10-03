# SAM 2 과제연구를 위한 평가·신뢰도 보정·연구설계 입문

## 이 글의 목표

이 글은 컴퓨터비전 평가를 처음 배우는 학생이 다음 세 단계를 차례로 이해하도록 만든 학습 원고다.

1. 한 장의 이진 마스크를 픽셀 단위로 비교해 TP, FP, FN, TN과 IoU, Dice를 직접 계산한다.
2. 영상 분할에서 쓰는 경계 점수와 DAVIS의 \(\mathcal J\), \(\mathcal F\), \(\mathcal J\&\mathcal F\)를 이해한다.
3. 모델의 점수가 실제 오류 가능성을 얼마나 잘 나타내는지 보정하고, 그 결과를 SAM 2의 메모리 기록·재탐색 정책에 연결한다.

마지막에는 이 개념을 이용해 다음 연구를 설계한다.

> **미래 실패위험을 보정해 TRACK–SUSPECT–LOST 상태를 전환하고, 선택적으로 메모리를 기록하며, 장기 anchor로 가림 후 대상을 복구하는 SAM 2 시스템**

이 글에서 가장 중요한 원칙은 “숫자가 높아졌다”와 “연구 가설이 지지되었다”를 구분하는 것이다. 한 번의 평균값만으로는 연구 결론을 내릴 수 없다. 무엇을 같은 조건으로 비교했는지, 데이터가 독립적인지, 향상이 여러 시퀀스에서 반복되는지, 계산비용은 얼마인지까지 확인해야 한다.

---

# 1. 한 장의 마스크에서 시작하기

## 1.1 Ground truth와 prediction

**Ground truth, GT**는 정답 마스크다. **Prediction**은 모델이 예측한 마스크다. 이진 객체 분할에서는 각 픽셀을 다음 두 값으로 나타낼 수 있다.

- `1`: 대상 객체, foreground
- `0`: 배경, background

다음은 5×5 이미지의 정답이다. 가운데 3×3 영역이 객체다.

```text
GT
. . . . .
. 1 1 1 .
. 1 1 1 .
. 1 1 1 .
. . . . .
```

모델은 같은 크기의 객체를 오른쪽으로 한 칸 밀려 예측했다고 하자.

```text
Prediction
. . . . .
. . 1 1 1
. . 1 1 1
. . 1 1 1
. . . . .
```

전체 픽셀은 25개다. 두 마스크를 겹쳐 보면 다음 네 종류가 생긴다.

## 1.2 TP, FP, FN, TN

| 이름 | 뜻 | 이 예시의 수 |
|---|---|---:|
| TP, True Positive | 실제 객체이고 모델도 객체라고 한 픽셀 | 6 |
| FP, False Positive | 실제 배경인데 모델이 객체라고 잘못 칠한 픽셀 | 3 |
| FN, False Negative | 실제 객체인데 모델이 놓친 픽셀 | 3 |
| TN, True Negative | 실제 배경이고 모델도 배경이라고 한 픽셀 | 13 |

검산하면 \(6+3+3+13=25\)다.

**자주 하는 혼동**

- FP는 “예측한 것 중 틀린 부분”이다.
- FN은 “정답 객체 중 놓친 부분”이다.
- positive가 무엇인지는 연구자가 먼저 정한다. 여기서는 foreground를 positive로 정했다.
- 다중 클래스에서는 각 클래스를 한 번씩 positive로 놓는 one-vs-rest 방식으로 계산할 수 있다.

## 1.3 Accuracy가 분할에서 위험한 이유

정확도는 다음과 같다.

\[
\mathrm{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}
=\frac{6+13}{25}=0.76
\]

76%라고 하면 꽤 좋아 보인다. 하지만 객체 영역의 3분의 1을 놓쳤고, 객체 크기의 3분의 1만큼 배경도 잘못 포함했다. 배경이 아주 큰 이미지에서는 아무것도 분할하지 않아도 TN이 많아 accuracy가 높아질 수 있다. 그래서 객체 분할은 IoU, Dice, boundary metric을 주로 사용한다.

---

# 2. Precision과 Recall

## 2.1 Precision — 예측한 픽셀 중 얼마나 맞았는가

\[
\mathrm{Precision}=\frac{TP}{TP+FP}
=\frac{6}{6+3}=\frac{2}{3}\approx0.667
\]

모델이 객체라고 칠한 9픽셀 중 6픽셀이 진짜 객체다.

Precision이 낮다는 것은 mask가 배경으로 많이 번졌다는 뜻이다. 예를 들어 그림자, 비슷한 옷을 입은 다른 사람, 주변 물체까지 포함한 경우 FP가 증가한다.

## 2.2 Recall — 실제 객체 중 얼마나 찾았는가

\[
\mathrm{Recall}=\frac{TP}{TP+FN}
=\frac{6}{6+3}=\frac{2}{3}\approx0.667
\]

실제 객체 9픽셀 중 6픽셀을 찾았다.

Recall이 낮다는 것은 객체의 일부를 놓쳤다는 뜻이다. 가느다란 다리, 자전거 바퀴살, 가려진 부분처럼 어려운 영역에서 FN이 증가한다.

## 2.3 Precision–recall trade-off

마스크 threshold를 낮추면 더 많은 픽셀을 객체로 선택한다. 보통 recall은 높아지지만 FP도 늘어 precision이 낮아진다. threshold를 높이면 정반대가 된다. 따라서 threshold 하나의 결과만 보고 모델 전체가 우수하다고 말하면 안 된다.

---

# 3. IoU/Jaccard와 Dice

## 3.1 IoU — Intersection over Union

IoU는 정답과 예측의 **교집합을 합집합으로 나눈 값**이다. Jaccard index와 같은 뜻이다.

\[
\mathrm{IoU}
=\frac{|G\cap P|}{|G\cup P|}
=\frac{TP}{TP+FP+FN}
\]

예시에서는 다음과 같다.

\[
\mathrm{IoU}=\frac{6}{6+3+3}=\frac{6}{12}=0.5
\]

TN이 공식에 들어가지 않는 점이 중요하다. 배경이 아무리 넓어도 객체의 겹침 정도만 평가한다.

## 3.2 Dice coefficient

Dice는 겹치는 영역을 두 배 한 뒤 두 마스크의 면적 합으로 나눈다.

\[
\mathrm{Dice}
=\frac{2|G\cap P|}{|G|+|P|}
=\frac{2TP}{2TP+FP+FN}
\]

예시에서는 다음과 같다.

\[
\mathrm{Dice}=\frac{2\times6}{12+3+3}
=\frac{12}{18}\approx0.667
\]

## 3.3 IoU와 Dice의 관계

같은 마스크 쌍에 대해 두 값은 다음처럼 변환된다.

\[
\mathrm{Dice}=\frac{2\mathrm{IoU}}{1+\mathrm{IoU}},\qquad
\mathrm{IoU}=\frac{\mathrm{Dice}}{2-\mathrm{Dice}}
\]

IoU가 0.5이면 Dice는 \(2\times0.5/1.5=0.667\)이다. Dice가 숫자상 더 크게 보이지만 같은 정보를 다른 척도로 표시했을 뿐이다. Dice 0.8과 IoU 0.8을 직접 같은 수준이라고 해석하면 안 된다.

## 3.4 무엇을 평균하는가

한 영상에 여러 객체와 여러 프레임이 있으면 평균 방법을 명시해야 한다.

- 모든 픽셀을 한꺼번에 합쳐 계산하는 micro average
- 객체별 점수를 계산한 뒤 평균하는 macro average
- 프레임별로 계산한 뒤 평균하는 방식
- 시퀀스별 평균을 다시 평균하는 방식

작은 객체를 큰 객체와 동등하게 중요하게 보려면 객체별 또는 시퀀스별 macro average가 유용하다. 반대로 전체 픽셀 정확도가 목적이면 micro average가 적합할 수 있다. 논문 비교에서는 benchmark의 공식 평가 코드를 그대로 사용해야 한다.

---

# 4. Boundary F와 Boundary IoU

영역 IoU가 같아도 윤곽선 품질은 다를 수 있다. 머리카락, 자전거 바퀴, 얇은 도구, 글자 획에서는 경계 평가가 특히 중요하다.

## 4.1 경계 픽셀 구하기

입문 계산에서는 foreground 픽셀 중 상하좌우 4-neighbor에 배경이 하나라도 있는 픽셀을 boundary라고 하자. 3×3 정사각형에서는 가운데 한 픽셀만 내부이고 나머지 8픽셀이 경계다.

앞의 GT와 prediction은 각각 경계 픽셀이 8개다. tolerance를 0으로 놓은 단순 예시에서는 정확히 같은 위치의 경계 픽셀이 4개다.

\[
P_{boundary}=\frac{4}{8}=0.5,\qquad
R_{boundary}=\frac{4}{8}=0.5
\]

Boundary F-measure는 두 값의 조화평균이다.

\[
\mathcal F
=\frac{2P_{boundary}R_{boundary}}
{P_{boundary}+R_{boundary}}
=0.5
\]

실제 DAVIS 평가는 경계 위치가 몇 픽셀 정도 어긋난 것을 허용하는 tolerance를 사용한다. 따라서 위의 정확 일치 계산과 실제 공식 \(\mathcal F\) 값은 달라질 수 있다. 해상도에 따라 같은 2픽셀 오차의 의미가 다르기 때문에 tolerance 정의와 resize 방식도 고정해야 한다.

## 4.2 Boundary IoU

Boundary IoU는 정답과 예측의 경계 주변 띠 영역을 만든 뒤 그 영역의 IoU를 계산한다. 개념을 보여 주기 위해 경계 폭이 0인 집합으로 단순화하면 다음과 같다.

\[
\mathrm{BIoU}=\frac{|B_G\cap B_P|}{|B_G\cup B_P|}
=\frac{4}{8+8-4}=\frac{4}{12}\approx0.333
\]

실제 구현에서는 contour 주변에 폭 \(d\)인 영역을 만들며, \(d\)를 픽셀 수로 둘지 영상 대각선 비율로 둘지 명시해야 한다.

## 4.3 Boundary F와 Boundary IoU 차이

- Boundary F는 예측 경계와 GT 경계가 tolerance 안에서 대응되는지를 precision과 recall로 평가한다.
- Boundary IoU는 경계 띠 영역의 교집합과 합집합을 평가한다.
- 둘 다 경계를 보지만 계산법과 tolerance 처리 방식이 다르므로 값을 섞어 쓰면 안 된다.

---

# 5. DAVIS의 J, F, J&F와 mIoU, mBIoU

## 5.1 \(\mathcal J\): region similarity

DAVIS에서 \(\mathcal J\)는 foreground mask의 Jaccard, 즉 IoU다.

앞의 예시에서는 다음과 같다.

\[
\mathcal J=0.5
\]

## 5.2 \(\mathcal F\): contour accuracy

\(\mathcal F\)는 경계 precision과 boundary recall의 조화평균이다. 단순 tolerance 0 예시에서는 0.5다.

## 5.3 \(\mathcal J\&\mathcal F\)

보통 두 값의 산술평균으로 보고한다.

\[
\mathcal J\&\mathcal F
=\frac{\mathcal J+\mathcal F}{2}
\]

예시에서는 \((0.5+0.5)/2=0.5\)다. 논문 표의 `J&F`는 논리 연산 AND가 아니라 두 지표의 평균이라는 점에 주의한다.

DAVIS 공식 평가는 객체와 프레임에 대한 평균뿐 아니라 recall과 temporal decay를 함께 보고하는 설정도 있다. 본 연구에서는 공식 코드가 출력하는 필드를 그대로 보존하고, 대표값만 임의로 재계산하지 않는다.

## 5.4 mIoU — mean IoU

다중 클래스 semantic segmentation에서는 클래스마다 IoU를 계산한 뒤 평균한다.

앞의 이진 예시에서 foreground IoU는 0.5다. 이번에는 background를 positive로 바꾸면 다음과 같다.

- background TP = 기존 TN = 13
- background FP = 기존 FN = 3
- background FN = 기존 FP = 3

\[
\mathrm{IoU}_{bg}=\frac{13}{13+3+3}=\frac{13}{19}\approx0.684
\]

foreground와 background를 모두 클래스 평균에 넣는다면 다음과 같다.

\[
\mathrm{mIoU}=\frac{0.5+0.684}{2}\approx0.592
\]

그러나 VOS의 \(\mathcal J\)는 보통 지정 객체의 foreground IoU를 뜻한다. 따라서 `mIoU 59.2%`와 `J 50.0%`가 같은 예시에서도 나올 수 있다. 배경 포함 여부와 평균 단위를 반드시 적어야 한다.

## 5.5 mBIoU — mean Boundary IoU

각 클래스의 Boundary IoU를 계산한 뒤 평균한 값이다.

\[
\mathrm{mBIoU}=\frac{1}{C}\sum_{c=1}^{C}\mathrm{BIoU}_c
\]

예를 들어 두 대상 클래스의 BIoU가 0.333과 0.750이라면 다음과 같다.

\[
\mathrm{mBIoU}=\frac{0.333+0.750}{2}=0.5415
\]

클래스가 하나뿐인 binary foreground 평가라면 mBIoU와 그 foreground BIoU가 같다. 배경도 독립 클래스로 포함하는 구현이라면 결과가 달라질 수 있다.

## 5.6 지표 선택 요약

| 연구 질문 | 우선 지표 |
|---|---|
| 객체 면적이 얼마나 정확히 겹치는가 | IoU, \(\mathcal J\), Dice |
| 얇은 구조와 윤곽이 정확한가 | \(\mathcal F\), Boundary IoU |
| 다중 semantic class 성능은 어떤가 | mIoU, mBIoU |
| 영상 전체에서 객체 분할 품질은 어떤가 | \(\mathcal J\&\mathcal F\)와 시퀀스별 분포 |
| 가림 후 다시 찾는가 | recovery success, time-to-recovery, false reacquisition |

---

# 6. Confidence, probability, uncertainty는 다르다

## 6.1 Confidence — 모델이 내놓은 자신감 점수

Confidence는 모델이 “이 결과가 좋다”고 보는 정도를 나타내는 숫자다. softmax 최대값, sigmoid 값, object score, predicted IoU 등이 confidence로 쓰일 수 있다.

하지만 0.9라는 점수가 곧 “90% 확률로 맞는다”는 뜻은 아니다. 0.9를 출력한 사례 100개 중 실제로 60개만 맞을 수도 있다. 이 모델은 **과신, overconfidence** 상태다.

## 6.2 Probability — 반복했을 때의 빈도 의미를 갖는 값

잘 보정된 확률 0.8은 비슷한 조건의 사례를 많이 모았을 때 약 80%가 정의한 사건을 만족해야 한다. 여기서 사건을 정확히 정해야 한다.

- 픽셀 사건: 이 픽셀이 foreground인가?
- mask 사건: 이 mask의 \(\mathcal J\)가 0.7 이상인가?
- 미래 실패 사건: 앞으로 10프레임 안에 \(\mathcal J<0.3\)이 되는가?
- 메모리 사건: 이 frame을 memory에 기록해도 이후 추적을 오염시키지 않는가?

서로 다른 사건의 확률을 같은 `confidence` 열에 섞으면 해석할 수 없다.

## 6.3 Uncertainty — 무엇을 모르는가

Uncertainty는 예측이 불확실한 정도다. 단순히 \(1-\text{confidence}\)로 만들 수 있지만, 항상 충분하지는 않다.

- **Aleatoric uncertainty**: blur, occlusion, 작은 객체처럼 입력 자체가 애매해서 생긴다.
- **Epistemic uncertainty**: 학습 데이터에 없던 장면이나 도메인이라 모델 지식이 부족해서 생긴다.
- **Temporal uncertainty**: 현재 frame 하나는 그럴듯하지만 과거 identity와 연결이 불안정한 경우다.

SAM 2 연구에서는 한 숫자보다 여러 신호가 유용하다.

- mask predicted IoU 또는 quality score
- object presence score
- object pointer의 프레임 간 변화
- mask 면적·중심·모양 변화
- 초기 대상 anchor와 현재 crop의 feature similarity
- motion model residual
- forward–backward cycle inconsistency

## 6.4 Predicted IoU는 실제 IoU가 아니다

SAM 계열의 predicted IoU는 mask 품질을 추정하도록 학습된 출력이다. 현재 frame의 GT를 보지 않고 “실제 IoU가 어느 정도일 것”이라고 추정한다.

- 실제 IoU: GT가 있어야 계산할 수 있는 평가값
- predicted IoU: 추론 중 모델이 내놓는 품질 추정값

predicted IoU 0.85가 실제 IoU 0.85를 보장하지 않는다. 특히 긴 가림, 비슷한 distractor, domain shift에서는 높은 점수를 유지한 채 틀린 객체를 따라갈 수 있다. 따라서 predicted IoU 자체를 baseline failure score로 삼되, 보정 전후를 평가해야 한다.

---

# 7. Calibration curve와 reliability diagram

## 7.1 무엇을 그리는가

Reliability diagram은 confidence를 여러 구간 bin으로 나누고 다음 두 값을 비교한다.

- x축: bin 안의 평균 예측 확률
- y축: bin 안의 실제 성공 비율

완벽히 보정되었다면 점들이 \(y=x\) 대각선에 가깝다.

- 점이 대각선 아래: 예측 확률보다 실제 성공률이 낮다 → 과신
- 점이 대각선 위: 실제 성공률이 더 높다 → 과소신

## 7.2 숫자 예시

mask가 좋다는 사건을 `실제 J ≥ 0.7`로 정의하자. 8개 사례에 대한 예측 확률과 실제 정답은 다음과 같다.

| 사례 | 예측 성공확률 \(p\) | 실제 성공 \(y\) |
|---:|---:|---:|
| 1 | 0.95 | 1 |
| 2 | 0.90 | 0 |
| 3 | 0.85 | 1 |
| 4 | 0.80 | 1 |
| 5 | 0.75 | 0 |
| 6 | 0.70 | 1 |
| 7 | 0.65 | 0 |
| 8 | 0.60 | 0 |

2개씩 bin을 묶으면 다음과 같다.

| bin | 평균 confidence | 실제 accuracy | 절대 차이 |
|---|---:|---:|---:|
| 0.60–0.69 | 0.625 | 0.000 | 0.625 |
| 0.70–0.79 | 0.725 | 0.500 | 0.225 |
| 0.80–0.89 | 0.825 | 1.000 | 0.175 |
| 0.90–1.00 | 0.925 | 0.500 | 0.425 |

표본이 8개뿐이라 diagram이 거칠다. 실제 연구에서는 bin 수를 고정해 보고하고, equal-width와 equal-frequency binning 중 무엇을 썼는지 적는다. Adaptive ECE나 calibration plot도 함께 보면 bin 선택 의존성을 줄일 수 있다.

## 7.3 영상에서의 주의점

인접 프레임은 서로 매우 비슷하므로 1,000프레임이 1,000개의 독립 표본은 아니다. reliability diagram은 frame-level로 그릴 수 있지만, confidence interval과 calibration split은 sequence 단위로 처리해야 한다. 한 긴 영상이 결과를 지배하는지도 함께 점검한다.

---

# 8. ECE, Brier score, NLL

Calibration은 하나의 지표만으로 판단하지 않는다. ECE는 직관적이지만 bin에 의존하고, Brier와 NLL은 개별 확률을 직접 평가한다.

## 8.1 ECE — Expected Calibration Error

\[
\mathrm{ECE}=\sum_{b=1}^{B}\frac{n_b}{N}
\left|\mathrm{acc}(b)-\mathrm{conf}(b)\right|
\]

앞 예시에서는 모든 bin 크기가 2로 같다.

\[
\mathrm{ECE}=\frac{0.625+0.225+0.175+0.425}{4}=0.3625
\]

0에 가까울수록 좋다. 그러나 모든 예측을 단순히 0.5 근처로 모으면 ECE가 좋아 보이면서 discrimination이 나쁠 수 있다. AUROC·AUPRC와 함께 봐야 한다.

## 8.2 Brier score

이진 사건에서 다음 평균 제곱오차를 계산한다.

\[
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

앞의 8개 예시에서는 제곱오차 합이 2.31이므로 다음과 같다.

\[
\mathrm{Brier}=2.31/8=0.28875
\]

0에 가까울수록 좋다. Brier는 calibration과 discrimination이 모두 반영된 proper scoring rule이다.

## 8.3 NLL — Negative Log-Likelihood

\[
\mathrm{NLL}=-\frac{1}{N}\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
\]

앞 예시의 NLL은 약 0.806이다. 틀린 사례에 0.99처럼 극단적으로 자신하면 매우 큰 벌점을 받는다. 계산에서는 \(p=0\)이나 1 때문에 로그가 무한대가 되지 않도록 작은 \(\epsilon\)으로 clipping한다.

## 8.4 무엇을 함께 보고할까

권장 조합은 다음과 같다.

- reliability diagram
- ECE
- Brier score 또는 NLL
- AUROC와 AUPRC
- 선택 시스템이라면 risk–coverage curve

---

# 9. AUROC와 AUPRC: 순위를 얼마나 잘 매기는가

Calibration은 숫자의 절대 의미를 보고, AUROC·AUPRC는 위험한 사례가 안전한 사례보다 높은 점수를 받는지 본다.

## 9.1 작은 순위 예시

미래 실패를 positive로 놓고 5개의 위험 점수가 다음과 같다고 하자.

| 위험 점수 | 실제 실패 |
|---:|---:|
| 0.90 | 1 |
| 0.80 | 0 |
| 0.70 | 1 |
| 0.60 | 0 |
| 0.10 | 0 |

positive 2개와 negative 3개의 모든 쌍은 6개다. 위험 점수 0.90인 positive는 세 negative보다 모두 높다. 0.70인 positive는 0.60과 0.10보다 높지만 0.80보다 낮다. 따라서 6쌍 중 5쌍의 순서가 맞다.

\[
\mathrm{AUROC}=5/6\approx0.833
\]

## 9.2 AUPRC가 희귀 실패에 중요한 이유

실패가 전체의 2%뿐이면 모든 frame을 정상이라고 해도 accuracy가 98%다. AUROC도 많은 easy negative 때문에 낙관적으로 보일 수 있다. Precision–recall curve는 높은 위험으로 선택한 사례 중 진짜 실패가 얼마나 되는지 직접 보여 준다.

AUPRC의 무작위 기준선은 대략 positive prevalence다. 실패율이 2%라면 기준선도 약 0.02다. 반드시 실패 prevalence와 함께 보고한다.

`AUPRC`라는 이름 아래 trapezoidal PR-AUC와 average precision이 혼용되므로 구현을 명시한다. scikit-learn의 `average_precision_score`를 썼다면 그대로 적는다.

## 9.3 높은 AUROC가 확률을 보장하지 않는다

모든 위험 점수에 10을 곱해도 순서는 같아 AUROC는 변하지 않는다. 그러나 0~1 확률 해석은 완전히 깨진다. 그래서 다음 두 질문을 분리한다.

1. 위험한 frame을 먼저 찾는가? → AUROC, AUPRC
2. 위험도 0.1이 실제 10% 실패를 뜻하는가? → ECE, Brier, NLL, reliability diagram

---

# 10. 사후 보정 방법

모델과 위험 점수 계산기를 학습한 뒤, 별도의 calibration set에서 score를 확률로 변환한다. Calibration set을 test set과 공유하면 안 된다.

## 10.1 Temperature scaling

logit \(z\)를 온도 \(T>0\)로 나눈 뒤 sigmoid 또는 softmax를 적용한다.

\[
p=\sigma(z/T)
\]

- \(T>1\): 확률을 0.5 쪽으로 부드럽게 만들어 과신을 줄인다.
- \(T<1\): 확률을 더 극단적으로 만든다.
- 순위를 바꾸지 않으므로 AUROC는 일반적으로 그대로다.
- parameter가 적어 과적합 위험이 낮지만 복잡한 calibration 오류는 고치지 못한다.

## 10.2 Logistic/Platt scaling

원래 score \(s\)에 대해 다음을 적합한다.

\[
p=\sigma(as+b)
\]

기울기와 절편을 함께 학습하므로 temperature scaling보다 유연하다. 위험 feature 여러 개를 logistic regression에 넣는 것은 단순 score 보정을 넘어 작은 failure predictor를 학습하는 것으로 구분해 써야 한다.

## 10.3 Isotonic regression

score가 커질수록 확률이 감소하지 않는다는 단조성만 가정하고 계단형 함수를 학습한다.

- 장점: 비선형 calibration 오류를 고칠 수 있다.
- 단점: calibration 표본이 적으면 계단이 거칠고 과적합하기 쉽다.
- sequence 수가 적은 연구에서는 bootstrap으로 불안정성을 확인한다.

## 10.4 보정법 비교 원칙

- 동일한 raw score와 동일한 calibration sequences를 사용한다.
- 보정 방법과 threshold를 calibration set에서만 선택한다.
- test에서는 ECE, Brier, NLL, AUROC/AUPRC를 한 번 평가한다.
- 보정 전후 AUROC가 같아도 calibration 개선은 의미가 있다.
- domain shift가 크면 source에서 잘 보정된 확률이 target domain에서 다시 틀어질 수 있다. dataset별 reliability diagram을 따로 그린다.

---

# 11. Conformal risk control

## 11.1 직관

일반 calibration은 “0.1이라고 한 사건이 약 10% 발생하는가”를 본다. Conformal risk control은 한 단계 더 나아가 **허용 위험 이하가 되도록 행동 threshold를 calibration data로 선택**하는 방법이다.

예를 들어 연구자가 다음 위험 예산을 정했다고 하자.

> memory에 기록하도록 허용한 frame 중, 실제로 나쁜 frame의 비율을 평균 5% 이하로 제한하고 싶다.

각 sequence \(i\)와 threshold \(\lambda\)에 대해 bounded loss \(L_i(\lambda)\in[0,1]\)를 정의한다. 예를 들면 선택된 memory write 중 이후 실패와 연결된 비율이다. Calibration sequences에서 여러 threshold를 시험하고, 위험의 one-sided upper confidence bound가 목표 \(\alpha=0.05\) 이하인 범위에서 coverage가 가장 큰 threshold를 선택한다.

개념적으로 다음 조건을 만족하는 \(\hat\lambda\)를 찾는다.

\[
U_\delta\!\left(\frac{1}{n}\sum_{i=1}^{n}L_i(\lambda)\right)
\le \alpha
\]

여기서 \(U_\delta\)는 신뢰수준 \(1-\delta\)의 상한이다.

## 11.2 보장 범위

Conformal 방법의 보장은 calibration과 test 사례가 exchangeable하다는 가정에 의존한다. 쉽게 말해 순서를 바꿔도 같은 분포에서 나온 사례처럼 취급할 수 있어야 한다.

다음은 자동으로 보장되지 않는다.

- 전혀 다른 카메라·날씨·도메인
- 모든 개별 sequence에서 각각 5% 이하
- 매우 작은 subgroup의 성능
- frame을 독립 표본으로 잘못 간주한 경우

따라서 본 연구에서는 frame이 아니라 sequence를 calibration 단위로 삼고, cross-dataset 성능은 별도로 보고한다. Conformal이라는 이름만 붙였다고 arbitrary domain shift까지 안전해지는 것은 아니다.

## 11.3 초보자용 구현 순서

1. 실패 사건과 loss를 먼저 정의한다.
2. train sequence로 raw failure score를 만든다.
3. 별도 calibration sequence에서 threshold 후보마다 sequence-level loss를 계산한다.
4. 목표 risk budget과 신뢰수준을 정한다.
5. 조건을 만족하는 threshold 중 coverage가 가장 큰 것을 고른다.
6. threshold를 고정한 채 test와 cross-dataset test를 수행한다.

---

# 12. Abstention, selective prediction, risk budget

## 12.1 Abstention — 모르면 강제로 답하지 않기

일반 모델은 모든 frame에서 mask를 출력한다. Selective system은 위험할 때 다음 행동을 할 수 있다.

- mask를 확정하지 않고 `SUSPECT`로 표시
- memory write를 생략
- 빈 mask 또는 `target absent` 출력
- 추가 detector·retriever를 호출
- 사용자 correction을 요청

이것이 abstention이다. 영상 추적에서는 틀린 mask를 memory에 넣는 것보다 잠시 보류하는 편이 장기 성능에 유리할 수 있다.

## 12.2 Coverage와 selective risk

\[
\mathrm{Coverage}=\frac{\text{자동으로 수용한 사례 수}}{\text{전체 사례 수}}
\]

\[
\mathrm{Selective\ Risk}
=\frac{\text{수용한 사례 중 실패 수}}{\text{수용한 사례 수}}
\]

threshold를 엄격하게 하면 selective risk는 낮아지지만 coverage도 낮아져 detector 호출이나 사용자 개입이 많아진다. 따라서 risk–coverage curve 전체를 보고한다.

## 12.3 Risk budget 예시

한 가지 budget만 정하는 것보다 실제 사용 비용을 나눠 적는 편이 좋다.

- 잘못된 memory write 비율 ≤ 5%
- false recovery ≤ 3%
- 추가 검색 호출 ≤ 20회/1,000 frames
- 사용자 correction ≤ 2회/1,000 frames
- 평균 처리속도 ≥ vanilla SAM 2의 70%

이 수치는 예시다. 소규모 pilot 뒤 test를 보기 전에 고정해야 한다. 결과를 본 뒤 달성 가능한 기준으로 바꾸면 성공 기준의 의미가 사라진다.

---

# 13. 좋은 연구 질문과 가설 만들기

## 13.1 연구 질문, RQ

연구 질문은 측정 가능한 형태여야 한다.

- **RQ1:** predicted IoU 하나보다 다중 신호로 만든 보정 위험도가 앞으로 10프레임 내 추적 실패를 더 잘 예측하는가?
- **RQ2:** 위험도가 높은 frame의 memory write를 막으면 장기 가림 이후의 비가역적 drift가 줄어드는가?
- **RQ3:** TRACK–SUSPECT–LOST 상태와 anchor recovery가 재등장 복구율을 높이면서 false reacquisition을 허용 범위 안에 유지하는가?
- **RQ4:** 개선 효과가 DAVIS뿐 아니라 MOSE, LVOS, RobustPVOS에서도 유지되는가?
- **RQ5:** 정확도 향상에 필요한 latency·VRAM·추가 호출 비용은 얼마이며 Pareto-efficient한가?

## 13.2 가설, hypothesis

가설은 방향과 비교 대상을 포함한다.

- **H1:** 다중 신호 calibrated risk는 predicted IoU baseline보다 미래 실패 AUPRC가 높다.
- **H2:** selective memory write는 전체 frame의 memory write보다 LVOS의 sequence-level \(\mathcal J\&\mathcal F\)와 recovery success를 높인다.
- **H3:** anchor recovery는 motion-only recovery보다 유사 distractor 상황의 false reacquisition을 줄인다.
- **H4:** 전체 시스템은 vanilla SAM 2 대비 LVOS와 MOSE에서 양의 paired improvement를 보이며, 평균 FPS는 baseline의 70% 이상을 유지한다.

“성능이 좋아질 것이다”만 쓰면 무엇을 측정할지 알 수 없다. 실패 horizon, dataset, baseline, metric을 포함한다.

---

# 14. Baseline과 공정한 비교

## 14.1 Baseline의 역할

Baseline은 새 방법이 없을 때의 기준이다. 최소한 다음을 포함한다.

- 공식 vanilla SAM 2 또는 SAM 2.1
- predicted IoU 또는 object score 하나로 trigger하는 단순 방법
- 고정 간격 every-N-frame 재프롬프트
- 가능하면 관련 선행연구의 공개 구현

복잡한 새 방법을 약하게 구현한 baseline과 비교하면 공정하지 않다. 공식 checkpoint와 공식 preprocessing을 먼저 재현한다.

## 14.2 Fair comparison 체크리스트

비교 방법 사이에서 다음을 같게 유지한다.

- SAM 2 모델 크기와 checkpoint
- 입력 해상도와 resize/interpolation
- 첫 prompt 형태: point, box, mask
- 같은 GT first-frame prompt
- propagation 방향과 처리 frame 수
- memory 최대 크기
- 외부 detector와 feature encoder 사용 여부
- threshold를 고른 validation data
- mixed precision과 batch size
- hardware, CUDA/PyTorch 환경
- metric 공식 코드와 absent-frame 처리

외부 Grounding DINO를 새 방법만 사용한다면, “SAM 2 내부 개선”이 아니라 “추가 모델을 포함한 시스템 개선”으로 써야 한다. parameter와 FLOPs도 전체 시스템 기준으로 보고한다.

---

# 15. Data split과 video leakage

## 15.1 프레임 무작위 분할이 왜 누수인가

한 영상의 100번째 frame을 train에, 101번째 frame을 test에 넣으면 두 장은 거의 같다. 배경, 대상 identity, 카메라가 모두 공유된다. 높은 test 성능은 일반화가 아니라 같은 영상을 기억한 결과일 수 있다.

따라서 split의 최소 단위는 frame이 아니라 **원본 video sequence**다.

## 15.2 자주 발생하는 video leakage

- 같은 원본 YouTube 영상에서 잘라낸 clip이 train과 test에 나뉨
- 같은 촬영 세션·카메라·장소의 연속 clip이 양쪽에 존재
- 한 sequence의 augmentation이 다른 split에 들어감
- 같은 객체 identity가 거의 동일한 배경에서 반복 등장
- calibration threshold를 test sequence를 보며 조절
- test GT로 실패 시점을 정의한 뒤, 그 정보를 online trigger 입력에 실수로 사용
- SAM 2가 학습한 SA-V를 독립 zero-shot test라고 주장

가능하면 source video ID, 촬영 세션, near-duplicate perceptual hash를 이용해 누수를 검사한다.

## 15.3 권장 분할

1. **Train:** failure predictor 또는 추가 모듈 학습
2. **Validation:** hyperparameter와 모델 선택
3. **Calibration:** 확률 보정 및 conformal threshold 결정
4. **Test:** 마지막 한 번의 성능 평가
5. **Cross-dataset test:** domain shift 평가

데이터가 작다면 validation 안에서 calibration subset을 sequence 단위로 분리한다. Test를 calibration에 사용해서는 안 된다.

---

# 16. Domain shift와 cross-dataset 평가

## 16.1 Domain shift 예시

- 맑은 야외 → 안개·야간·눈
- 짧고 중심 객체가 뚜렷한 DAVIS → crowded MOSE
- 수 초 영상 → 수 분 LVOS
- 일반 객체 → camouflage 또는 작은 대상
- 영상 mask benchmark → box 중심 tracking benchmark

source dataset에서 calibration이 좋아도 target dataset에서 과신할 수 있다. 전체 ECE 하나만 보고하지 말고 dataset별, 조건별 reliability diagram을 그린다.

## 16.2 Cross-dataset protocol

예를 들어 MOSE train으로 failure predictor를 학습하고 MOSE의 held-out sequence에서 calibrate한 뒤 다음을 수행한다.

- in-domain: MOSE test 또는 frozen validation split
- short clean shift: DAVIS 2017
- long-term shift: LVOS
- corruption shift: RobustPVOS
- distractor stress: DiDi

Target test에 맞춰 threshold를 다시 조절하면 strict cross-dataset 결과가 아니다. 재보정을 허용한 결과는 `target-calibrated`로 별도 표기한다.

---

# 17. Ablation과 controlled variable

## 17.1 Ablation study

전체 방법에서 구성요소를 하나씩 제거하거나 추가해 어느 부분이 효과를 냈는지 확인한다.

권장 순서는 다음과 같다.

1. E0: vanilla SAM 2
2. E1: predicted IoU threshold trigger
3. E2: 다중 신호 raw failure score
4. E3: E2 + calibration
5. E4: E3 + selective memory write
6. E5: E4 + TRACK/SUSPECT/LOST
7. E6: E5 + positive anchor recovery
8. E7: E6 + negative distractor memory

## 17.2 Controlled variable

한 번에 하나의 원인만 비교한다.

| 비교 목적 | 바꿀 것 | 고정할 것 |
|---|---|---|
| calibration 효과 | raw vs calibrated risk | feature, classifier, threshold 선택 규칙 외 전부 |
| memory gate 효과 | write gate on/off | 같은 mask prediction과 recovery |
| anchor 효과 | anchor scoring on/off | 같은 candidate proposals |
| motion vs appearance | candidate score 항 | candidate 수, 호출 시점, SAM 2 |
| event trigger 효과 | fixed interval vs risk trigger | 총 허용 호출 수 또는 계산 budget |

호출 횟수가 다른 두 시스템을 그대로 비교하면 정확도 향상이 단순히 계산을 더 많이 쓴 결과일 수 있다. 같은 호출 budget 비교와 unconstrained 비교를 둘 다 제시하면 좋다.

---

# 18. Hyperparameter와 재현성

## 18.1 주요 hyperparameter

- 미래 실패 horizon \(H\)
- 실패 IoU threshold \(\tau_{fail}\)
- TRACK→SUSPECT, SUSPECT→LOST threshold
- hysteresis 연속 frame 수
- memory write risk threshold
- anchor similarity와 distractor margin
- LOST 상태 검색 간격
- candidate 수와 NMS threshold
- calibration bin 수와 방법

## 18.2 선택 절차

- coarse grid를 validation에서 비교한다.
- 가장 좋은 값 하나만 남기지 말고 sensitivity plot을 제공한다.
- 모든 dataset에 같은 threshold를 적용하는 결과와 dataset-specific threshold를 구분한다.
- test 성능을 본 뒤 값을 수정하지 않는다.

## 18.3 재현성 기록

- Git commit hash
- SAM 2 checkpoint 파일명과 checksum
- Python, PyTorch, CUDA, driver 버전
- 모든 package lock 또는 environment 파일
- random seed와 deterministic 설정
- 데이터 다운로드 날짜·버전·split 목록
- 실행 command와 config
- 전처리 해상도·색공간·interpolation
- metric code version
- raw per-frame, per-object, per-sequence 결과
- 실패 로그와 예외 처리 규칙

평균 점수 표만 저장하면 나중에 통계 검정을 다시 할 수 없다. 최소한 sequence별 결과를 CSV/JSON으로 남긴다.

---

# 19. Sequence-level paired bootstrap 95% CI

## 19.1 왜 frame bootstrap이 아닌가

영상의 인접 frame은 강하게 상관되어 있다. frame 10,000개를 독립 표본처럼 bootstrap하면 표본 수가 과장되어 CI가 지나치게 좁아진다.

독립 단위에 더 가까운 sequence를 cluster로 사용한다. 한 sequence 안의 여러 객체와 frame은 함께 움직인다.

## 19.2 Paired bootstrap 절차

baseline A와 새 방법 B를 같은 \(N\)개 sequence에서 평가했다고 하자.

1. 각 sequence에서 두 방법의 metric을 계산한다.
2. sequence별 차이 \(d_i=M_{B,i}-M_{A,i}\)를 만든다.
3. \(N\)개 sequence ID를 복원추출로 다시 뽑는다.
4. 선택된 sequence들의 평균 차이를 계산한다.
5. 이 과정을 10,000번 반복한다.
6. bootstrap 차이 분포의 2.5 percentile과 97.5 percentile을 95% CI로 사용한다.

예를 들어 평균 향상이 +2.4 J&F point이고 95% CI가 [+0.8, +4.0]이면 대부분의 bootstrap 표본에서 양수다. 반면 [+2.4]라도 CI가 [-0.7, +5.1]이면 평균 향상은 관찰됐지만 불확실성이 커서 일관된 개선이라고 강하게 결론내리기 어렵다.

## 19.3 Pairing의 의미

두 방법은 반드시 같은 sequence 쌍으로 비교한다. 어려운 영상과 쉬운 영상 구성이 동일하므로 between-sequence 변동을 제거할 수 있다.

한 sequence에 객체가 여러 개라면 두 가지 estimand가 가능하다.

- sequence를 동등 가중: sequence별 객체 평균을 만든 뒤 bootstrap
- 객체를 동등 가중하되 sequence cluster를 함께 resample

어느 것을 주요 분석으로 쓸지 사전에 정하고 공식 benchmark aggregation과의 차이를 설명한다.

---

# 20. Wilcoxon signed-rank test

## 20.1 무엇을 검정하는가

같은 sequence에서 얻은 차이 \(d_i\)를 사용한다. 차이의 절댓값에 순위를 매기고 부호를 반영해, 차이 분포가 0을 중심으로 하는지 검정한다.

- 귀무가설: 두 방법의 paired difference가 0을 중심으로 한다.
- 대립가설: 차이가 0이 아니다. 방향 가설을 미리 정했다면 one-sided도 가능하다.

## 20.2 주의점

- frame별 점수에 Wilcoxon을 적용하지 않는다.
- 0 차이는 통상 제외되며 사용한 구현을 적는다.
- 차이 분포가 매우 비대칭이면 sign test도 보조적으로 고려한다.
- p-value는 효과 크기가 아니다. 평균·중앙값 차이와 95% CI를 함께 보고한다.
- 여러 dataset, metric, ablation에 검정을 반복하면 Holm 등의 multiple-comparison correction을 고려한다.
- “p>0.05이므로 두 방법이 같다”라고 결론내리면 안 된다. 증거가 부족한 것과 동등함은 다르다.

권장 보고 예시는 다음과 같다.

> LVOS 50개 validation sequence에서 새 방법은 SAM 2.1 대비 sequence-level J&F를 평균 2.4 point 향상시켰다. Paired bootstrap 95% CI는 [0.8, 4.0]이었으며, 양측 Wilcoxon signed-rank test의 Holm-adjusted p는 0.012였다.

---

# 21. 희귀 실패와 event-based recovery 평가

평균 J&F는 정상 frame이 많으면 가림·재등장 실패를 숨긴다. 연구 주제가 복구라면 event를 별도로 정의한다.

## 21.1 Failure event 정의 예시

대상이 보이는 frame에서는 다음 중 하나를 failure로 정할 수 있다.

- \(\mathcal J_t<0.3\)이 2프레임 이상 지속
- target 대신 distractor와 더 크게 겹치는 ID switch
- mask area가 GT target과 무관하게 급증하며 IoU가 붕괴

대상이 GT상 완전히 보이지 않는 frame에서는 빈 mask가 정답일 수 있다. 따라서 `target absent인데 mask가 없음`을 실패로 세면 안 된다. Presence annotation 또는 visibility rule을 명시한다.

## 21.2 미래 실패 label

현재 시점 \(t\)의 위험 모델을 학습하려면 미래 horizon \(H\)를 정한다.

\[
y_t^{(H)}=
\mathbf1\!\left[\text{frame }t+1\ldots t+H\text{ 안에 failure 발생}\right]
\]

예를 들어 \(H=10\)이면 현재까지 관찰한 정보로 향후 10프레임의 실패를 예측한다. 입력 feature에 미래 frame이나 GT를 넣으면 leakage다.

## 21.3 Detection metrics

- event precision, recall, F1
- frame-level AUROC와 AUPRC
- false alarms per 1,000 frames
- event당 중복 trigger 수
- detection lead time: 실패 몇 frame 전에 경고했는가
- missed-event rate

Frame-level AUPRC와 event-level recall을 둘 다 보고한다. 한 failure 주변에서 경고가 20번 발생해도 event 하나를 잡은 것으로만 세는 matching window가 필요하다.

## 21.4 Recovery metrics

- **Recovery success@H:** 재등장 후 \(H\)프레임 안에 IoU threshold를 회복한 event 비율
- **Time-to-recovery, TTR:** 재등장부터 처음 \(\mathcal J\ge0.5\)를 연속 \(k\)프레임 달성할 때까지의 frame 수
- **Post-recovery IoU:** 복구 후 일정 window의 평균 IoU
- **Recovery AUC:** 재등장 후 시간에 따른 IoU curve의 면적
- **False reacquisition:** target이 아닌 distractor를 target으로 확정한 비율
- **Relapse rate:** 복구 뒤 짧은 시간 안에 다시 실패한 비율
- **Never-recovered rate:** 영상 종료까지 복구하지 못한 event 비율

영상 종료 때문에 관찰 window가 부족한 event는 censored case가 될 수 있다. 단순히 제외하면 짧은 영상에 유리해지므로 별도 표시하거나 survival analysis를 고려한다.

## 21.5 통계 단위

Event는 같은 sequence 안에서 상관될 수 있다. CI를 구할 때 event만 독립 resampling하지 말고 sequence cluster를 resample한 뒤 그 안의 event를 함께 포함한다. Event 수와 이를 포함한 sequence 수도 함께 보고한다.

---

# 22. 효율성: FPS, latency, VRAM, FLOPs, parameter, Pareto

## 22.1 FPS

\[
\mathrm{FPS}=\frac{\text{처리한 frame 수}}{\text{걸린 초}}
\]

다음 조건을 함께 적는다.

- GPU와 CPU 모델
- 입력 해상도
- batch size, 보통 streaming은 1
- FP32, FP16, BF16 여부
- warm-up frame 수
- disk I/O와 video decoding 포함 여부
- CUDA synchronization 여부
- normal path와 recovery path 호출 비율

## 22.2 Latency

평균 latency만으로 event-driven 시스템을 설명하기 어렵다.

- p50: 평소 frame의 체감 지연
- p95, p99: recovery가 호출된 느린 frame 포함
- normal-path latency
- trigger-path latency
- end-to-end latency: decoding과 후처리 포함

실시간 시스템에서는 평균 30 FPS여도 p99가 1초라면 사용이 불편할 수 있다.

## 22.3 VRAM

- peak allocated VRAM
- peak reserved VRAM
- sequence 길이에 따른 증가
- 객체 수에 따른 증가
- memory bank와 anchor 저장량

측정 직전 peak statistic을 reset하고 동일 sequence로 비교한다. 다른 process가 GPU를 쓰지 않는 상태를 기록한다.

## 22.4 FLOPs와 parameters

- total parameters와 trainable parameters를 구분한다.
- 외부 DINO, detector, optical-flow model도 전체 시스템 parameter에 포함한다.
- 동적 시스템은 normal FLOPs, triggered FLOPs, 실제 trigger rate를 이용한 expected FLOPs를 모두 보고한다.

개념적으로 평균 frame 비용은 다음과 같다.

\[
C_{avg}=C_{normal}+r_{trigger}C_{extra}
\]

## 22.5 Pareto frontier

방법 A가 방법 B보다 정확도도 높고 속도도 빠르면 B는 dominated다. 어느 한쪽만 좋은 방법들이 Pareto frontier를 이룬다.

다음 plot을 권장한다.

- x축: average latency 또는 expected FLOPs
- y축: J&F 또는 recovery success
- 점 크기: peak VRAM
- 점 색: false reacquisition 또는 selective risk

단일 “최고 모델”보다 accuracy–cost trade-off를 명확히 보여 준다.

---

# 23. 데이터셋 이해와 역할 분담

## 23.1 DAVIS 2017

DAVIS 2017은 고품질 dense mask가 있는 대표적인 다중 객체 VOS benchmark다. Semi-supervised, 더 정확히는 human-guided 설정에서는 첫 frame의 대상 mask가 제공된다. 공식 평가는 region \(\mathcal J\)와 contour \(\mathcal F\)를 사용한다.

**장점**

- 선명한 mask와 널리 쓰이는 공식 평가 코드
- 빠른 sanity check와 선행연구 비교에 좋음
- 경계 품질을 확인하기 좋음

**한계**

- 비교적 짧고 잘 촬영된 영상이 많다.
- 중심 대상이 뚜렷해 실제 crowded·장기 재등장 문제를 충분히 대표하지 못한다.
- DAVIS만 향상되면 장기 recovery 개선이라고 주장하기 어렵다.

[DAVIS 공식 사이트](https://davischallenge.org/)

## 23.2 MOSE

MOSEv1은 2,149 clips, 5,200 objects, 431,725개 고품질 mask를 포함하며 crowded scene, heavy occlusion, 유사 객체, disappearance–reappearance를 의도적으로 많이 담았다.

**적합한 질문**

- distractor 때문에 target identity가 바뀌는가?
- 가림 중 memory가 오염되는가?
- 작은 객체와 복잡한 움직임에서 trigger가 작동하는가?

MOSEv2를 사용할 경우 버전을 명시한다. MOSEv2는 adverse weather, low light, multi-shot, camouflage, non-physical target과 knowledge-dependent 장면까지 확장한다. v1과 v2 결과를 같은 split처럼 섞지 않는다.

[MOSE 공식 프로젝트](https://mose.video/), [MOSEv1 ICCV 원문](https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_MOSE_A_New_Dataset_for_Video_Object_Segmentation_in_Complex_ICCV_2023_paper.pdf)

## 23.3 LVOS

LVOS는 220개 장기 영상, 총 약 421분으로 구성되며 평균 영상 길이는 약 1.59분, 평균 574 frames다. Long-term reappearance는 적어도 100 frames 동안 사라졌다 다시 나타나는 속성으로 정의된다. 시간상 동시에 등장하지 않는 유사 객체로 인한 cross-temporal confusion도 포함한다.

**적합한 질문**

- 짧은 FIFO memory가 오래된 target identity를 잊는가?
- target이 화면 밖으로 나간 뒤 다시 찾는가?
- immutable anchor가 장기 drift를 줄이는가?
- recovery time이 얼마나 줄어드는가?

[LVOS 공식 프로젝트](https://lingyihongfd.github.io/lvos.github.io/), [ICCV 원문](https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_LVOS_A_Benchmark_for_Long-term_Video_Object_Segmentation_ICCV_2023_paper.pdf)

## 23.4 DiDi

DiDi는 기존 tracking benchmark에서 distractor가 뚜렷한 sequence를 반자동으로 추출한 distractor-distilled dataset이다. visually similar object가 있을 때 target을 유지하는 능력을 집중 평가한다.

**중요한 한계**

- 본래 visual object tracking 평가가 중심이며 bounding box와 VOT toolkit 지표를 사용한다.
- DiDi의 quality/accuracy/robustness를 VOS의 J&F와 동일하게 해석하면 안 된다.
- mask 품질 주장을 하려면 별도의 pixel annotation 또는 VOS 평가가 필요하다.

본 연구에서는 negative distractor memory와 false reacquisition stress test에 사용하는 것이 적절하다.

[DAM4SAM·DiDi 공식 저장소](https://github.com/jovanavidenovic/dam4sam)

## 23.5 RobustPVOS

RobustPVOS는 adverse-condition promptable VOS를 위한 CVPR 2026 benchmark다. 두 실제 평가셋에 총 351 clips와 2,500개 이상의 object mask가 있다.

- ACDC-Video: 149 clips, 3,259 frames, 613 objects
- MVSeg 계열: 202 clips, 13,581 frames, 1,930 objects
- 조건: 안개, 비, 눈, 야간, 저조도, motion blur 등
- synthetic train/evaluation: MOSE-C, DAVIS-C, YouTube-VOS-C와 같이 시간에 따라 변하는 corruption을 적용

**적합한 질문**

- source에서 보정한 risk가 날씨·저조도 shift에서도 calibration을 유지하는가?
- blur에서 predicted IoU가 과신하는가?
- recovery 호출 증가가 latency tail을 얼마나 키우는가?

[CVPR 원문](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Robust_Promptable_Video_Object_Segmentation_CVPR_2026_paper.html), [공식 프로젝트](https://sohyun-l.github.io/RobustPVOS_project_page/)

## 23.6 SA-V

Meta의 SA-V는 약 51K 영상과 643K class-agnostic masklet을 포함한다. 약 191K masklet은 SAM 2-assisted manual annotation이고, 약 452K는 SAM 2가 자동 생성한 뒤 annotator가 검증했다. 평균 영상당 12.61 masklet이며 class label은 제공하지 않는다.

**장점**

- 다양하고 대규모인 open-world video segmentation 자료
- failure predictor의 추가 학습 또는 self-supervised 분석에 활용 가능

**주의점**

- SAM 2 학습에 사용된 데이터이므로 SA-V 성능을 완전히 독립적인 zero-shot generalization으로 주장하면 안 된다.
- annotation의 상당 부분이 SAM 2 생성·인간 검증 방식이므로 모델 고유 bias가 반영될 수 있다.
- 규모가 매우 커 학기 연구에서는 curated subset과 명확한 sampling rule이 필요하다.

[Meta SA-V 데이터 카드](https://ai.meta.com/datasets/segment-anything-video/), [SAM 2 ICLR 원문](https://proceedings.iclr.cc/paper_files/paper/2025/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf)

## 23.7 데이터셋 역할 요약

| 데이터셋 | 주 역할 | 강한 실패 유형 | 주의점 |
|---|---|---|---|
| DAVIS 2017 | 기본 mask·경계 성능 | 변형, 일부 가림, 다중 객체 | 짧고 비교적 깨끗함 |
| MOSE | 복잡 장면 | crowd, occlusion, distractor, 재등장 | v1/v2 구분 |
| LVOS | 장기 평가 | 긴 부재, out-of-view, cross-temporal confusion | 긴 sequence 비용 |
| DiDi | distractor stress | 유사 객체, identity drift | tracking 지표 중심 |
| RobustPVOS | domain/corruption shift | 날씨, 저조도, blur | calibration shift를 별도 분석 |
| SA-V | 대규모 학습·분석 | open-world 다양성 | SAM 2 train data와 순환성 |

---

# 24. 제안 연구: 미래 실패위험 보정 기반 SAM 2

## 24.1 핵심 문제

SAM 2는 현재 mask를 memory에 저장하고 다음 frame 예측에 사용한다. 잘못된 mask가 memory에 들어가면 다음 예측도 틀리고, 그 결과가 다시 memory에 들어가는 self-reinforcing drift가 생길 수 있다.

따라서 현재 mask가 얼마나 좋아 보이는지만 판단하지 않고 다음을 물어야 한다.

> **“현재 상태를 그대로 믿고 memory에 기록하면 앞으로 몇 프레임 안에 추적이 무너질 위험은 얼마인가?”**

## 24.2 예측 대상

미래 horizon을 \(H=10\) frames로 둔 예를 생각하자.

\[
r_t=P\!\left(y_t^{(H)}=1\mid x_{1:t},\text{SAM2 outputs}_{1:t}\right)
\]

여기서 \(r_t\)는 미래 failure probability이고 추론 시점에는 현재와 과거 정보만 사용한다.

Failure label은 visibility를 고려해 정의한다.

- target visible: 향후 \(H\) 안에 \(\mathcal J<\tau_{fail}\)이 연속 \(k\) frames 발생
- target absent: 큰 false-positive mask를 출력하거나 distractor를 target으로 확정
- identity annotation 가능: 다른 객체와의 overlap이 target보다 커지는 ID switch

## 24.3 입력 feature

GT 없이 추론 중 얻을 수 있는 값만 사용한다.

| 종류 | feature 예시 | 의미 |
|---|---|---|
| SAM 2 내부 | predicted IoU, object score, object pointer distance | 모델 자체 품질·존재 신호 |
| mask geometry | area ratio, centroid velocity, aspect ratio, fragmentation | 갑작스러운 모양 변화 |
| appearance | DINOv2 anchor similarity, positive–negative margin | 대상 identity 유지 여부 |
| motion | Kalman residual, optical-flow consistency | 위치 예측과 관측 불일치 |
| temporal | forward–backward cycle error, mask overlap history | 시간 일관성 |
| context | blur, brightness, corruption estimate | 위험 장면 여부 |

처음에는 logistic regression이나 작은 MLP로 시작한다. 큰 모델보다 feature와 evaluation protocol의 타당성을 먼저 입증한다.

## 24.4 보정

1. Train sequences로 failure predictor를 학습한다.
2. 별도 calibration sequences에서 temperature, logistic, isotonic을 비교한다.
3. ECE, Brier, NLL로 최종 calibrator를 선택한다.
4. Memory write와 LOST transition threshold는 validation/calibration에서 고정한다.
5. Test와 cross-dataset에서는 수정하지 않는다.

Predicted IoU 단독 score를 반드시 baseline으로 둔다. 그래야 다중 신호와 calibration의 실제 기여를 알 수 있다.

---

# 25. TRACK–SUSPECT–LOST 상태기계

## 25.1 TRACK

대상을 정상적으로 추적한다고 판단한 상태다.

- mask를 정상 출력
- risk가 충분히 낮을 때만 short-term memory write
- 신뢰도 높은 일부 frame만 long-term anchor 후보로 승격
- 외부 detector/retriever는 호출하지 않음

## 25.2 SUSPECT

오류 가능성은 있지만 즉시 LOST로 확정하기 어려운 상태다.

- memory write를 일시 중지
- 현재 mask를 provisional output으로 표시
- motion box, appearance proposal, negative point 등 소수 후보 생성
- 후보 간 consistency 확인
- 위험이 낮아지면 TRACK으로 복귀
- 모든 후보가 약하거나 위험이 더 높아지면 LOST로 전환

SUSPECT 상태가 없으면 점수 한 번의 흔들림만으로 비싼 재탐색을 호출하거나 target을 잘못 초기화할 수 있다.

## 25.3 LOST

현재 target identity를 신뢰할 수 없거나 target이 화면에 없다고 보는 상태다.

- 잘못된 mask를 강제로 memory에 기록하지 않음
- 필요하면 empty mask와 abstention 상태를 출력
- 일정 간격 또는 target-presence 신호가 생길 때 candidate search
- immutable anchor와 negative distractor memory로 후보 검증
- 여러 조건을 연속 \(m\) frames 통과할 때만 TRACK으로 복귀

## 25.4 Hysteresis

한 threshold만 사용하면 TRACK과 SUSPECT가 매 frame 진동할 수 있다. 진입·복귀 threshold를 다르게 둔다.

- TRACK→SUSPECT: \(r_t\ge\tau_s\)가 2 frames 지속
- SUSPECT→TRACK: \(r_t\le\tau_{back}<\tau_s\)이고 anchor 검증 통과
- SUSPECT→LOST: \(r_t\ge\tau_l\) 또는 후보 검증 연속 실패
- LOST→TRACK: recovery score가 \(\tau_{recover}\) 이상으로 2 frames 지속

이 숫자는 예시이며 validation에서 정한다.

---

# 26. Selective memory write

## 26.1 왜 필요한가

최근 frame을 무조건 memory에 넣으면 최신 appearance를 잘 따라갈 수 있지만, 가림·distractor frame도 함께 저장된다. 반대로 너무 적게 저장하면 appearance change에 적응하지 못한다.

## 26.2 Write gate 예시

다음 조건을 모두 만족할 때만 현재 frame을 memory에 쓴다.

\[
\mathrm{WRITE}_t=
\mathbf1[
S_t=\mathrm{TRACK},
r_t\le\tau_w,
s^{anchor}_t\ge\theta_a,
m^{distractor}_t\ge\theta_m]
\]

- \(S_t\): 현재 상태
- \(r_t\): calibrated future failure risk
- \(s^{anchor}_t\): positive anchor similarity
- \(m^{distractor}_t\): target score와 가장 유사한 distractor score의 margin

## 26.3 메모리 종류

- **Short-term memory:** 최근 appearance 적응용. FIFO로 교체 가능.
- **Long-term positive anchors:** 최초 prompt frame과 매우 신뢰도 높은 corrected frame. 쉽게 삭제하지 않음.
- **Negative distractor memory:** 잘못 선택되기 쉬운 유사 객체 특징. 후보 감점에 사용.

첫 prompt anchor는 immutable하게 유지한다. 다만 대상이 장기간 변형되면 최초 모습만으로 부족하므로, 검증된 새로운 anchor를 소수 추가한다.

## 26.4 무엇을 측정할까

- 전체 memory write 수
- 나중에 실제 저품질로 판명된 contaminated write 비율
- contaminated write 후 첫 failure까지의 시간
- gate로 막은 frame 중 사실 안전했던 비율
- coverage와 selective risk

---

# 27. Anchor recovery

## 27.1 Candidate 생성

SUSPECT 또는 LOST에서 다음 후보를 만든다.

- motion prediction box
- DINO similarity heatmap의 높은 영역 box
- SAM 2 automatic mask proposals
- 선택적으로 open-vocabulary detector의 text box
- 이전 mask 주변의 positive/negative point 조합

모든 frame에서 모든 후보를 만들지 않고 event-trigger 방식으로 호출한다.

## 27.2 Candidate 점수

예시 점수는 다음과 같다.

\[
S(c)=w_a\,\mathrm{sim}_{pos}(c)
-w_n\,\mathrm{sim}_{neg}(c)
+w_m\,\mathrm{motion}(c)
+w_q\,\mathrm{quality}(c)
-w_r\,\mathrm{risk}(c)
\]

- positive anchor와 닮을수록 가점
- known distractor와 닮을수록 감점
- 가능한 motion 범위와 맞으면 가점
- SAM predicted quality가 높으면 가점
- 미래 failure risk가 높으면 감점

최고 후보가 threshold를 넘지 못하면 억지로 선택하지 않고 LOST를 유지한다.

## 27.3 Recovery 확인

한 frame에서만 좋은 후보는 우연일 수 있다. 다음 중 일부를 요구한다.

- 2개 연속 frame에서 높은 anchor similarity
- forward로 추적한 뒤 backward 했을 때 원래 후보와 일치
- negative distractor margin 확보
- 예상 object size와 큰 모순이 없음
- calibrated recovery confidence가 threshold 이상

검증을 통과하면 box 또는 mask prompt로 SAM 2를 재초기화하고 TRACK으로 돌아간다.

---

# 28. 전체 알고리즘 의사코드

```text
입력: video frames, initial prompt, SAM 2
초기화:
    state = TRACK
    immutable positive anchor 저장
    short-term memory 초기화
    negative distractor memory = empty

각 frame t에 대해:
    SAM 2 mask와 내부 quality 신호 계산
    geometry, appearance, motion, cycle feature 계산
    calibrated future failure risk r_t 계산

    if state == TRACK:
        if r_t가 낮고 anchor 검증 통과:
            mask 출력
            조건부로 memory write
        else if suspect 조건 지속:
            state = SUSPECT
            memory write 중지

    if state == SUSPECT:
        소수 candidate prompt 생성
        candidate를 anchor/distractor/motion/quality로 평가
        if 안전한 후보가 연속 검증 통과:
            SAM 2 재프롬프트
            state = TRACK
        else if lost 조건 충족:
            state = LOST
        else:
            provisional mask 또는 abstention 출력

    if state == LOST:
        memory write 금지
        empty/abstain 출력
        검색 간격이 되었으면 candidate search
        if anchor recovery 검증 통과:
            SAM 2 재초기화
            state = TRACK
```

---

# 29. 권장 실험 계획

## 29.1 단계별 개발

### 단계 A — 평가 파이프라인 검증

- vanilla SAM 2.1을 DAVIS와 LVOS 일부에서 재현
- per-frame J, F, object presence, predicted IoU 저장
- 공식 점수와 직접 집계 점수가 일치하는지 확인

### 단계 B — failure predictor

- future failure label 생성
- predicted IoU 단독 baseline
- geometry, appearance, temporal feature를 순서대로 추가
- AUROC/AUPRC와 lead time 평가

### 단계 C — calibration과 selective write

- temperature/logistic/isotonic 비교
- reliability, ECE, Brier, NLL 평가
- risk–coverage curve와 memory contamination 측정

### 단계 D — 상태기계와 anchor recovery

- TRACK/SUSPECT/LOST 전환
- motion-only, positive-anchor, positive+negative anchor 비교
- event-based recovery metric 추가

### 단계 E — cross-dataset·효율성

- DAVIS, MOSE, LVOS, RobustPVOS, DiDi 역할별 평가
- FPS, p95 latency, peak VRAM, expected FLOPs
- sequence paired bootstrap와 Wilcoxon

## 29.2 Main result 표 양식

| 방법 | DAVIS J&F | MOSE J&F | LVOS J&F | RobustPVOS J&F | Recovery@30 | Median TTR↓ | False reacq.↓ | FPS↑ | VRAM↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM 2.1 |  |  |  |  |  |  |  |  |  |
| Fixed-N re-prompt |  |  |  |  |  |  |  |  |  |
| Pred-IoU trigger |  |  |  |  |  |  |  |  |  |
| Proposed full |  |  |  |  |  |  |  |  |  |

## 29.3 Failure prediction·calibration 표

| 위험 모델 | AUROC | AUPRC | ECE↓ | Brier↓ | NLL↓ | Lead time↑ | False alarms/1K↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Predicted IoU |  |  |  |  |  |  |  |
| + geometry |  |  |  |  |  |  |  |
| + appearance |  |  |  |  |  |  |  |
| + temporal |  |  |  |  |  |  |  |
| + calibration |  |  |  |  |  |  |  |

## 29.4 Ablation 표

| ID | Calibrated risk | Selective write | State machine | Positive anchor | Negative memory | J&F | Recovery@30 | Contam. write↓ | FPS |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E0 |  |  |  |  |  |  |  |  |  |
| E1 | ✓ |  |  |  |  |  |  |  |  |
| E2 | ✓ | ✓ |  |  |  |  |  |  |  |
| E3 | ✓ | ✓ | ✓ |  |  |  |  |  |  |
| E4 | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |
| E5 | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |

## 29.5 통계 보고 표

| 데이터셋 | Sequence 수 | 평균 ΔJ&F | 95% paired bootstrap CI | Median Δ | Wilcoxon p | 보정 p |
|---|---:|---:|---|---:|---:|---:|
| DAVIS |  |  | [ , ] |  |  |  |
| MOSE |  |  | [ , ] |  |  |  |
| LVOS |  |  | [ , ] |  |  |  |
| RobustPVOS |  |  | [ , ] |  |  |  |

---

# 30. 성공 기준 예시

성공 기준은 pilot을 마친 뒤 test 결과를 보기 전에 고정한다. 다음은 과제연구용 예시다.

## 30.1 Primary success criterion

- 주 데이터셋 LVOS에서 vanilla SAM 2.1 대비 sequence-level J&F가 최소 +2.0 point 향상
- paired bootstrap 95% CI의 하한이 0보다 큼

둘 중 하나만 만족했다고 표현하지 말고 primary 기준을 사전에 `AND` 또는 `OR`로 명시한다. 엄격한 예시는 둘 다 만족하는 것이다.

## 30.2 Failure prediction

- predicted IoU 단독 대비 미래 실패 AUPRC 상대 향상 ≥ 20%
- 보정 후 ECE ≤ 0.05 또는 raw score 대비 30% 이상 감소
- Brier와 NLL 중 적어도 하나가 악화되지 않음
- median detection lead time ≥ 3 frames

## 30.3 Recovery

- Recovery success@30이 baseline보다 ≥ 10 percentage points 향상
- median TTR이 ≥ 20% 감소
- false reacquisition ≤ 3% 또는 baseline보다 증가하지 않음
- never-recovered rate 감소

## 30.4 Selective memory

- contaminated memory write 비율 ≤ 5%
- memory write coverage ≥ 70%
- 같은 recovery 호출 budget에서 gate-off보다 J&F 향상

## 30.5 효율성

- 평균 FPS가 vanilla baseline의 70% 이상
- peak VRAM 증가 ≤ 25%
- 추가 검색 호출 ≤ 전체 frames의 10%
- p95 latency와 worst-case recovery latency를 별도 공개

## 30.6 일반화

- LVOS 외 MOSE 또는 RobustPVOS 중 적어도 한 곳에서 양의 평균 ΔJ&F
- dataset별 calibration plot을 제시
- target-specific 재보정 없이 얻은 strict cross-dataset 결과를 포함

위 숫자는 연구를 시작하기 위한 예시이지 보편적 기준이 아니다. 데이터 규모와 응용 위험에 따라 조정한다.

---

# 31. 결과 해석에서 피해야 할 문장

## 잘못된 예

> 우리 방법은 J&F가 1.2 높으므로 모든 상황에서 우수하다.

문제점: CI, sequence 분포, 계산량, 특정 dataset 의존성을 모른다.

## 개선된 예

> LVOS validation sequence에서 평균 J&F는 1.2 point 향상했으나 paired bootstrap 95% CI가 -0.3에서 2.7이어서 일관된 개선이라고 단정하기 어렵다. 이득은 long-term reappearance subset에 집중됐고 DAVIS에서는 차이가 작았다.

## 잘못된 예

> 위험 예측 AUROC가 0.9이므로 0.9 score는 90% 실패 확률이다.

문제점: AUROC는 순위 지표이지 calibration 지표가 아니다.

## 개선된 예

> 위험 score의 AUROC는 0.9였지만 high-risk bin에서 과신이 관찰됐다. Logistic calibration 후 AUROC는 유지됐고 ECE가 0.12에서 0.04로 감소했다.

---

# 32. 연구 실행 전 최종 체크리스트

## 문제·지표

- [ ] foreground와 absent frame 정의가 명확한가?
- [ ] primary metric 하나를 test 전에 정했는가?
- [ ] J, F, J&F, mIoU의 평균 단위를 기록했는가?
- [ ] failure와 recovery event matching window가 정해졌는가?
- [ ] AUPRC와 failure prevalence를 함께 보고하는가?

## 데이터

- [ ] frame이 아니라 source sequence 단위로 split했는가?
- [ ] 원본 영상이 같은 clip의 누수를 확인했는가?
- [ ] validation, calibration, test 역할이 분리됐는가?
- [ ] SA-V가 SAM 2 학습 데이터라는 점을 해석에 반영했는가?
- [ ] MOSE v1/v2, LVOS 버전과 split을 명시했는가?

## 비교

- [ ] 동일 checkpoint, prompt, 해상도, memory budget인가?
- [ ] 추가 detector와 encoder 비용을 포함했는가?
- [ ] fixed compute/call budget 비교가 있는가?
- [ ] 공식 metric code를 사용했는가?

## Calibration·선택

- [ ] score가 어떤 사건의 확률인지 정의했는가?
- [ ] reliability diagram, ECE, Brier/NLL을 함께 확인했는가?
- [ ] conformal 가정과 risk 단위를 sequence 수준으로 설명했는가?
- [ ] coverage와 selective risk를 함께 보고했는가?

## 통계·효율성

- [ ] sequence-level paired bootstrap CI가 있는가?
- [ ] Wilcoxon을 frame이 아니라 sequence에 적용했는가?
- [ ] multiple testing 보정을 고려했는가?
- [ ] FPS 측정 조건과 p95 latency를 공개했는가?
- [ ] peak VRAM, total/trainable parameters, 외부 모델 FLOPs를 포함했는가?

## 재현성

- [ ] commit, 환경, seed, checkpoint checksum을 기록했는가?
- [ ] sequence별 raw 결과를 저장했는가?
- [ ] 실패 사례와 성공 사례를 함께 시각화했는가?
- [ ] test를 본 뒤 threshold를 바꾸지 않았는가?

---

# 33. 자가점검 문제

## 문제 1

100픽셀 이미지에서 GT 객체가 20픽셀이다. 모델은 25픽셀을 객체로 예측했고, 그중 15픽셀이 GT와 겹쳤다. TP, FP, FN, TN을 구하라.

## 문제 2

문제 1의 precision, recall, IoU, Dice를 계산하라.

## 문제 3

문제 1의 pixel accuracy가 높더라도 모델이 좋다고 단정할 수 없는 이유를 설명하라.

## 문제 4

IoU가 0.6인 mask의 Dice를 계산하라.

## 문제 5

두 모델의 region IoU가 모두 0.8인데 모델 A의 boundary F가 0.9, 모델 B가 0.6이다. 어떤 차이를 의심할 수 있는가?

## 문제 6

foreground IoU만 평균한 값과 background까지 포함한 binary mIoU가 왜 다를 수 있는가?

## 문제 7

모델이 confidence 0.9를 출력한 frame 100개 중 65개만 실제 성공했다. 이 점수는 잘 보정되었는가? 어떤 방향의 오류인가?

## 문제 8

높은 AUROC와 나쁜 ECE가 동시에 가능하다는 것을 설명하라.

## 문제 9

실패율이 1%인 데이터에서 accuracy보다 AUPRC가 중요한 이유와 무작위 AUPRC 기준선을 설명하라.

## 문제 10

Temperature scaling, logistic scaling, isotonic regression의 유연성과 데이터 요구량을 비교하라.

## 문제 11

Conformal risk control로 calibration sequence의 memory-write 위험을 5% 이하로 맞췄다. 전혀 다른 열화상 카메라 영상에서도 반드시 5% 이하라고 말할 수 있는가?

## 문제 12

같은 영상의 홀수 frame을 train, 짝수 frame을 test로 나누면 왜 안 되는가?

## 문제 13

10개 영상에서 각 1,000 frame을 평가했다. 10,000 frame을 독립 표본으로 bootstrap하는 대신 10개 sequence를 resample해야 하는 이유를 설명하라.

## 문제 14

새 방법의 평균 ΔJ&F가 +2.0이고 95% paired bootstrap CI가 [-0.5, 4.4]다. 어떻게 해석해야 하는가?

## 문제 15

Wilcoxon p-value가 0.03이면 새 방법이 실용적으로 크게 향상되었다는 뜻인가?

## 문제 16

TRACK 상태에서 failure risk가 높아졌지만 anchor similarity는 아직 높다. 바로 LOST로 가지 않고 SUSPECT를 거치는 이유는 무엇인가?

## 문제 17

대상이 완전히 가려진 동안 예측 mask를 memory에 계속 기록하면 어떤 문제가 생길 수 있는가?

## 문제 18

Anchor recovery가 motion-only recovery보다 유사 distractor에 강할 수 있는 이유를 설명하라.

## 문제 19

방법 A는 J&F 75, 30 FPS이고 방법 B는 J&F 74, 20 FPS다. 정확도와 속도만 고려할 때 어느 방법이 Pareto dominated인가?

## 문제 20

Recovery@30은 높아졌지만 false reacquisition도 크게 증가했다. 이 시스템을 성공으로 볼 수 있는가? 어떤 추가 기준이 필요한가?

---

# 34. 자가점검 정답과 해설

## 정답 1

- TP = 겹친 픽셀 = 15
- FP = 예측 객체 25 중 틀린 것 = \(25-15=10\)
- FN = GT 객체 20 중 놓친 것 = \(20-15=5\)
- TN = 전체 100에서 나머지 = \(100-15-10-5=70\)

## 정답 2

\[
\mathrm{Precision}=15/(15+10)=0.6
\]

\[
\mathrm{Recall}=15/(15+5)=0.75
\]

\[
\mathrm{IoU}=15/(15+10+5)=0.5
\]

\[
\mathrm{Dice}=30/(30+10+5)=30/45\approx0.667
\]

Precision보다 recall이 높으므로 실제 객체를 비교적 많이 찾았지만 배경으로 번진 FP가 더 큰 문제다.

## 정답 3

Accuracy는 \((15+70)/100=0.85\)다. 배경 TN 70개가 점수를 높였기 때문에 객체 영역 오류가 가려진다. 분할 대상이 작을수록 이 현상이 심하다. IoU와 Dice처럼 TN을 제외한 지표가 객체 품질을 더 직접적으로 보여 준다.

## 정답 4

\[
\mathrm{Dice}=\frac{2\times0.6}{1+0.6}=1.2/1.6=0.75
\]

## 정답 5

두 모델의 전체 겹침 면적은 비슷하지만 B는 윤곽선이 거칠거나 위치가 더 어긋났을 가능성이 크다. 얇은 구조를 놓치거나 object 주변으로 부풀린 mask일 수도 있다. Region metric 하나로 fine detail을 판단하면 안 된다.

## 정답 6

Foreground IoU는 지정 객체만 계산한다. Binary mIoU에 background를 포함하면 넓고 쉬운 배경 IoU가 평균에 더해진다. 따라서 foreground IoU가 낮아도 mIoU가 더 높게 나올 수 있다. 어떤 class를 평균했는지 밝혀야 한다.

## 정답 7

잘 보정되지 않았다. 0.9라고 예측했지만 실제 성공률은 0.65이므로 0.25만큼 과신했다. Reliability diagram에서 해당 점은 대각선 아래에 놓인다.

## 정답 8

AUROC는 positive가 negative보다 높은 score를 받는 순서만 본다. 모든 score를 과도하게 높은 값으로 바꾸어도 순서가 유지되면 AUROC는 같다. 그러나 score의 절대값과 실제 빈도가 맞지 않아 ECE는 나쁠 수 있다.

## 정답 9

모두 정상이라고 예측해도 accuracy가 99%이므로 실패 탐지 성능을 알 수 없다. AUPRC는 높은 위험으로 선택한 사례 중 실제 실패 비율과 전체 실패를 찾은 비율을 본다. 무작위 ranking의 기대 AUPRC 기준은 대략 positive prevalence인 0.01이다.

## 정답 10

- Temperature scaling: parameter가 매우 적고 안정적이지만 단순한 sharpness 오류만 주로 고친다.
- Logistic/Platt scaling: 기울기와 절편을 학습해 더 유연하다.
- Isotonic: 단조 비선형 함수를 학습해 가장 유연하지만 calibration sequence가 적으면 과적합하기 쉽다.

보정법은 calibration data에서 선택하고 test에서는 고정해야 한다.

## 정답 11

말할 수 없다. Conformal 보장은 calibration과 test의 exchangeability에 의존한다. 열화상이라는 큰 domain shift가 있으면 가정이 깨질 수 있다. 열화상 test 결과와 calibration curve를 별도로 평가하거나, target calibration을 허용한 프로토콜을 분리해야 한다.

## 정답 12

홀수와 짝수 frame은 배경, 대상 identity, motion이 거의 동일하다. Test가 새로운 영상이 아니라 train frame의 근접 복사본이 되어 성능이 과대평가된다. 원본 sequence 또는 촬영 세션 단위로 나눠야 한다.

## 정답 13

같은 sequence의 frame들은 독립이 아니다. 10,000개로 취급하면 effective sample size를 크게 과장해 CI가 지나치게 좁아진다. Sequence를 cluster로 resample하면 영상 내부 상관을 보존한다. 단, sequence가 10개뿐이면 CI 자체가 불안정하다는 한계도 보고해야 한다.

## 정답 14

관찰 평균은 +2.0이지만 CI가 0을 포함한다. 새 방법이 평균적으로 좋아 보인다는 신호는 있으나, sequence sampling 변동을 고려하면 일관된 양의 개선이라고 강하게 결론내리기 어렵다. 더 많은 sequence나 failure-specific 분석이 필요하다.

## 정답 15

아니다. p=0.03은 귀무가설 아래 현재와 같은 순위 차이가 드물다는 통계적 증거다. 향상 크기를 뜻하지 않는다. 평균/중앙값 Δ, 95% CI, recovery benefit, 계산비용을 함께 봐야 한다. 여러 검정을 했다면 보정 p-value도 확인한다.

## 정답 16

Risk spike가 blur 한 frame이나 일시적 motion 때문에 생겼을 수 있다. Anchor similarity가 높다면 identity는 아직 유지됐을 가능성이 있다. SUSPECT에서 memory write만 막고 소수 후보를 검증하면 불필요한 LOST 전환과 비싼 전체 재검색을 줄일 수 있다.

## 정답 17

가림 물체나 배경, distractor의 특징이 target memory로 들어간다. 이후 SAM 2가 잘못된 특징에 attention하고, 그 예측이 다시 기록되는 memory poisoning과 self-reinforcing drift가 생긴다. Selective write와 absent 상태가 필요한 이유다.

## 정답 18

Motion-only 방식은 target과 distractor가 비슷한 경로를 지나거나 카메라가 급격히 움직이면 잘못된 후보를 선택할 수 있다. Anchor는 초기 target의 appearance와 identity 정보를 제공하며, negative distractor memory를 함께 쓰면 비슷한 후보도 margin으로 구분할 수 있다. 반대로 appearance가 크게 변하면 motion cue가 필요하므로 둘을 결합하는 것이 좋다.

## 정답 19

B는 A보다 J&F가 낮고 FPS도 낮으므로, 두 축만 보면 A가 B를 지배한다. 따라서 B가 Pareto dominated다. 다만 B가 VRAM이나 false recovery에서 더 좋다면 그 축까지 포함한 Pareto 판단은 달라질 수 있다.

## 정답 20

Recovery@30만으로 성공이라고 할 수 없다. 잘못된 객체를 빠르게 선택해도 recovery로 잘못 집계될 수 있다. False reacquisition 상한, anchor identity 검증, post-recovery IoU, relapse rate, 사용자·검색 호출 비용을 공동 성공 기준으로 둬야 한다.

---

# 35. 한 페이지 요약

1. TP·FP·FN·TN은 모든 분할 지표의 출발점이다.
2. IoU와 Dice는 영역, Boundary F와 BIoU는 윤곽 품질을 본다.
3. DAVIS의 J는 IoU, F는 contour F, J&F는 두 값의 평균이다.
4. mIoU와 mBIoU는 어떤 클래스를 평균했는지 반드시 밝혀야 한다.
5. Confidence는 아직 probability가 아니며 calibration으로 실제 빈도와 맞춰야 한다.
6. ECE·Brier·NLL은 확률 품질, AUROC·AUPRC는 위험 순위를 본다.
7. 실패가 희귀하면 AUPRC와 event metric이 중요하다.
8. Selective prediction은 위험할 때 강제로 답하지 않고 memory write나 mask 확정을 보류한다.
9. Train·validation·calibration·test는 sequence 단위로 분리한다.
10. 통계는 frame이 아니라 sequence-level paired bootstrap과 Wilcoxon을 사용한다.
11. 평균 J&F만 보지 말고 recovery time, false reacquisition, contamination, latency를 함께 본다.
12. 제안 시스템의 핵심은 미래 실패위험을 보정해 TRACK–SUSPECT–LOST를 전환하고, 안전한 frame만 memory에 기록하며, 검증된 anchor로 target을 다시 찾는 것이다.

