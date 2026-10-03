# PART G — 원문을 읽는 법과 핵심 용어사전

## 원문 구간별로 이 가이드에서 찾아볼 곳

- 원문 1장 ‘연구 범위’가 헷갈리면 PART A의 task taxonomy와 VOS/PVOS/VOT/MOT/RVOS/VRS 구분을 다시 본다.
- 원문 2장 ‘구조적 실패 이유’가 헷갈리면 PART C의 SAM 2 inference loop와 memory contamination을 다시 본다.
- 원문 3.1 ‘장기 memory, 가림, 재등장’은 PART C의 admission·retention·read·pruning과 PART D의 drift 사례에 대응한다.
- 원문 3.2 ‘prompt, 언어, 의미 기반 재집중’은 prompt, visual anchor, text embedding, VLM, open-vocabulary 항목에 대응한다.
- 원문 3.3 ‘다중 객체 identity’는 identity switch, association, mask ownership, bipartite matching 항목에 대응한다.
- 원문 3.4 ‘경계와 작은 객체’는 downsampling, high-resolution feature, boundary F, mBIoU, temporal flicker 항목에 대응한다.
- 원문 3.5 ‘도메인 적응과 강건성’은 domain shift, corruption, fine-tuning, PEFT, adapter, LoRA 항목에 대응한다.
- 원문 3.6 ‘속도·memory 효율’은 FPS, latency, VRAM, parameter, FLOPs, token pruning, quantization, distillation, Pareto frontier 항목에 대응한다.
- 원문 4장 ‘연구 공백’은 PART E의 calibration과 PART F의 연구가설을 연결해서 읽는다.
- 원문 5장 ‘최우선 연구안’은 미래 실패 label, calibrator, selective write, 3상태 controller, anchor recovery 순으로 읽는다.
- 원문 6장 ‘실험 설계’는 baseline, data split, ablation, event metric, bootstrap, Wilcoxon을 확인하며 읽는다.

## 논문 표를 읽는 7단계

1. task를 확인한다. VOS, VOT, MOT, RVOS는 숫자를 바로 비교할 수 없는 서로 다른 문제다.
2. input prompt를 확인한다. 첫 frame GT mask, box, point, 자동 prompt는 난이도가 다르다.
3. backbone과 checkpoint를 확인한다. SAM 2와 SAM 2.1, Tiny·Base+·Large를 섞으면 안 된다.
4. dataset과 split을 확인한다. validation을 조정에 쓰고 같은 set을 test라고 부르면 안 된다.
5. metric과 단위를 확인한다. J&F 0.80과 80은 같은 값일 수 있고, AUC와 J&F는 의미가 다르다.
6. 추가 계산과 supervision을 확인한다. detector, optical flow, VLM, GT box, test-time training이 있으면 비용과 정보 조건이 달라진다.
7. 평균과 함께 실패 유형별 결과, confidence interval, 속도, VRAM을 확인한다.

## 핵심 용어사전

### A–C

**Ablation study:** 제안 방법의 부품을 하나씩 빼거나 바꾸어 어느 요소가 실제 향상을 만들었는지 확인하는 실험.

**Anchor memory:** 대상이 확실히 보였던 신뢰 frame이나 feature를 오래 보존해 소실 뒤 복구에 사용하는 기준 기억.

**Annotation:** 사람이 데이터에 붙인 정답 표시. class, box, point, mask 등이 있다.

**Appearance:** 물체의 색, 질감, 모양처럼 보이는 특성. motion과 대비되는 단서다.

**Association:** 현재 frame의 후보와 과거 track 중 무엇이 같은 객체인지 연결하는 과정.

**Attention:** query가 여러 key와의 관련성을 계산해 필요한 value를 가중합하는 정보 선택 연산.

**AUROC:** 실패와 성공을 구분하는 score의 임계값을 전부 움직였을 때 true-positive rate와 false-positive rate의 관계를 요약한 면적.

**AUPRC:** precision–recall 곡선 아래 면적. 실패가 드문 불균형 문제에서는 AUROC보다 실용적일 수 있다.

**Backbone:** 입력에서 일반적인 feature를 뽑는 모델의 중심 encoder.

**Baseline:** 제안 방법의 효과를 판단하기 위해 같은 조건에서 비교하는 기준 방법.

**Batch:** 한 번의 forward/backward에서 함께 처리하는 데이터 묶음.

**Bipartite matching:** 두 집합의 후보를 비용이 최소가 되도록 일대일 연결하는 방법. 다중 객체 ID association에 쓰인다.

**Bounding box:** 객체를 둘러싼 직사각형 위치 표현.

**Brier score:** 예측확률과 실제 0/1 결과의 제곱오차 평균. 낮을수록 확률이 실제 결과와 잘 맞는다.

**Calibration:** 0.8이라고 예측한 사건이 실제로도 약 80% 일어나도록 score의 확률적 의미를 맞추는 것.

**Candidate mask:** decoder가 한 prompt에 대해 내놓는 여러 가능한 mask 중 하나.

**Checkpoint:** 학습된 parameter와 설정을 저장한 파일 또는 상태.

**Class:** 모델이 구분하려는 의미 범주. instance ID와는 다르다.

**CLIP:** image와 text를 같은 embedding 공간에 정렬해 의미 유사도를 비교하도록 학습한 모델 계열.

**Confidence:** 모델이 자기 예측을 얼마나 믿는지를 나타내는 score. 자동으로 정확한 probability가 되는 것은 아니다.

**Conformal risk control:** calibration data를 이용해 제한된 가정 아래 선택 결과의 위험이나 오류율을 통제하려는 방법 계열.

**Confounder:** 원인으로 생각한 변수와 결과 사이를 함께 바꾸어 잘못된 결론을 만들 수 있는 숨은 요인.

**Convolution:** 작은 kernel을 영상 위로 이동시키며 주변 패턴을 검출하는 연산.

**Corruption:** blur, noise, rain, fog, low light처럼 입력을 손상시키는 변화.

**Cross-dataset evaluation:** 한 dataset에서 정한 방법이나 calibrator를 다른 dataset에 적용해 일반화를 보는 평가.

### D–H

**Data leakage:** test 정보가 학습, threshold 선택, feature 생성 과정에 흘러 들어가 성능이 실제보다 높아지는 문제.

**Decoder:** 압축된 feature와 prompt를 최종 mask나 label로 변환하는 부분.

**Deep supervision:** 중간 decoder 단계에도 loss를 주어 여러 해상도에서 학습을 돕는 방법.

**DINO/DINOv2:** label 없이도 강한 visual feature를 학습하는 self-supervised vision model 계열.

**Dice coefficient:** 두 mask의 겹침을 2|교집합|/(|정답|+|예측|)으로 측정하는 값.

**Disappearance:** target이 화면 밖으로 나가거나 완전히 가려져 관측되지 않는 상태.

**Distractor:** target과 비슷해 모델을 헷갈리게 하는 다른 객체.

**Distillation:** 큰 teacher의 출력이나 feature를 작은 student가 따라 배우게 하는 압축 학습.

**Domain:** 데이터가 만들어지는 환경과 분포. 일반 사진, 의료, 수중, 항공, 야간 영상은 서로 다른 domain이다.

**Domain adaptation:** source domain에서 학습한 모델을 target domain에 맞게 조정하는 방법.

**Domain shift:** 학습 환경과 실제 평가 환경의 밝기, 센서, 대상, 배경 분포가 달라지는 현상.

**Downsampling:** feature의 가로·세로 해상도를 줄여 계산을 아끼는 연산. 작은 객체와 가는 경계가 사라질 수 있다.

**Drift:** 작은 추적 오류가 시간에 따라 누적되어 mask가 target에서 벗어나는 현상.

**ECE:** confidence 구간별 평균 confidence와 실제 accuracy의 차이를 가중 평균한 calibration 오차.

**Embedding:** 이미지, prompt, text, object를 의미나 관계가 보존되도록 숫자 vector로 표현한 것.

**Encoder:** 원시 픽셀이나 prompt를 task에 유용한 feature로 바꾸는 부분.

**Epoch:** 학습 dataset 전체를 한 번 순회한 단위.

**Event-triggered:** 매 N frame처럼 고정 주기로 동작하지 않고 위험·가림·score 변화 같은 사건이 생길 때 동작하는 방식.

**Feature:** 입력에서 추출한 유용한 숫자 표현. edge, texture, object part, semantics 등이 암묵적으로 담긴다.

**Fine-tuning:** pretrained parameter의 일부 또는 전체를 새 데이터로 추가 학습하는 것.

**Flicker:** 영상 mask의 경계나 영역이 frame마다 불안정하게 떨리는 현상.

**FLOPs:** 모델이 수행하는 부동소수점 연산량의 근사치. 실제 latency와 항상 일치하지는 않는다.

**Foundation model:** 매우 넓은 데이터로 학습해 여러 task와 domain으로 확장할 수 있도록 만든 대형 기반 모델.

**FPS:** 초당 처리 frame 수. batch, 해상도, 장치, 측정 구간을 함께 밝혀야 비교할 수 있다.

**Frozen:** parameter를 학습 중 업데이트하지 않는 상태.

**Ground truth, GT:** 학습과 평가에서 정답으로 취급하는 annotation.

**Hiera:** SAM 2 image encoder에 쓰인 계층형 vision Transformer 계열. 여러 해상도의 visual feature를 만든다.

**Hyperparameter:** 학습률, threshold, memory size처럼 학습 전에 사람이 정하거나 validation으로 선택하는 설정.

### I–P

**Identity:** 여러 frame에서 같은 실제 객체임을 나타내는 지속적인 구분.

**Identity switch:** tracker가 target의 ID를 다른 객체와 바꾸는 오류.

**Image encoder:** RGB pixels를 reusable visual feature로 변환하는 encoder.

**Inference:** 학습이 끝난 모델로 새로운 입력의 결과를 계산하는 과정.

**Instance:** 같은 class 안에서도 서로 구별되는 개별 객체.

**IoU/Jaccard:** 정답과 예측 mask의 교집합 크기를 합집합 크기로 나눈 영역 겹침 지표.

**Isotonic regression:** score의 순서를 유지하면서 단조로운 비선형 함수로 probability를 보정하는 방법.

**Latency:** 입력 하나 또는 frame 하나의 결과를 얻는 데 걸리는 시간.

**LoRA:** 큰 weight는 고정하고 낮은 rank의 작은 행렬만 학습해 parameter와 memory를 줄이는 PEFT 방법.

**Logistic regression:** 입력 feature의 가중합을 sigmoid에 넣어 0–1 probability를 만드는 단순한 분류·calibration 모델.

**Long-term memory:** 오래전의 중요 정보를 보존해 긴 가림이나 재등장에 사용하는 memory.

**Loss:** 예측이 정답과 얼마나 다른지 나타내며 학습이 줄이려는 목적함수.

**Mask:** target인 pixel과 background pixel을 구분하는 지도.

**mBIoU:** 여러 sample이나 class의 boundary IoU 평균.

**Memory admission:** 현재 정보를 memory에 새로 기록할지 결정하는 단계.

**Memory attention:** 현재 feature가 과거 memory의 어느 부분을 참고할지 attention으로 선택하는 부분.

**Memory bank:** 과거 frame의 feature, mask, pointer 등을 저장하는 제한된 공간.

**Memory contamination:** 틀린 mask나 잘못된 identity 정보가 memory에 저장되어 이후 예측을 해치는 현상.

**Memory encoder:** mask와 visual feature를 다음 frame에서 사용할 memory representation으로 변환하는 부분.

**Memory pruning:** 덜 중요하거나 중복된 memory item을 제거하는 것.

**Memory read:** 현재 추론에 사용할 memory를 선택하고 가져오는 단계.

**Memory retention:** 기록된 항목을 얼마나 오래 보존할지 정하는 정책.

**mIoU:** 여러 sample 또는 class IoU의 평균.

**Motion:** frame 사이 위치와 모양의 시간적 변화.

**Multi-hypothesis:** 한 순간에 후보 경로 하나만 남기지 않고 여러 가능성을 잠시 유지하는 방식.

**NLL:** 실제 결과에 부여한 probability의 negative log likelihood. 확신하고 틀린 예측을 크게 벌준다.

**Object pointer:** SAM 2에서 추적 객체의 고수준 identity와 의미를 압축해 전달하는 token 성격의 표현.

**Object score/presence score:** 현재 frame에 target이 존재하는지를 나타내려는 score.

**Occlusion:** target의 일부 또는 전부가 다른 물체에 가려진 상태.

**Online learning:** test 또는 사용 도중 새 data·수정 신호로 모델 일부를 업데이트하는 학습.

**Open-vocabulary:** 학습 때 고정한 class 목록 밖의 개념도 text나 exemplar로 지정할 수 있는 능력.

**Optical flow:** 인접 frame 사이 각 pixel의 이동을 추정한 vector field.

**Overfitting:** 학습 data에는 잘 맞지만 새로운 data에는 성능이 떨어지는 현상.

**Parameter:** 학습으로 값이 바뀌는 weight와 bias.

**Pareto frontier:** 한 목표를 더 좋게 하려면 다른 목표를 희생해야 하는 최적 trade-off 경계. 예: J&F–FPS–VRAM.

**PEFT:** 전체 모델 대신 작은 일부 parameter만 학습하는 parameter-efficient fine-tuning 방법군.

**Post-hoc calibration:** 원 모델의 학습을 바꾸지 않고 완성된 score를 validation data로 뒤에서 보정하는 방법.

**Precision:** positive라고 예측한 것 중 실제 positive의 비율.

**Predicted IoU:** mask decoder가 자기 mask와 GT의 IoU를 추정하려고 출력하는 품질 score. 실제 IoU나 보정된 확률과 동일하지 않다.

**Pretraining:** 넓은 데이터로 일반 feature를 먼저 학습하는 단계.

**Prompt:** 모델이 무엇을 처리해야 하는지 알려 주는 입력 지시. point, box, mask, text, exemplar가 될 수 있다.

**Prompt encoder:** point·box·mask prompt를 model이 사용할 embedding으로 바꾸는 부분.

### Q–Z

**Quantization:** float weight와 activation을 int8/int4 같은 낮은 bit 표현으로 바꾸어 memory·연산을 줄이는 기술.

**Query–Key–Value:** attention에서 query가 질문, key가 검색표, value가 실제 가져올 내용에 해당하는 세 표현.

**Recall:** 실제 positive 중 모델이 positive로 찾아낸 비율.

**Reappearance:** 가려지거나 화면 밖으로 나간 target이 다시 보이는 사건.

**Reliability diagram:** confidence bin마다 평균 confidence와 실제 accuracy를 비교하는 calibration 그래프.

**Re-prompt:** 모델이 실패했다고 판단했을 때 point, box, mask 등의 prompt를 자동 또는 사용자 입력으로 다시 주는 것.

**Reproducibility:** 다른 사람이 같은 code, data, seed, configuration으로 비슷한 결과를 얻을 수 있는 성질.

**Risk budget:** 허용할 평균 오류, 실패율, 계산량, 사용자 개입 횟수 같은 자원·위험 한도.

**Robustness:** noise, blur, low light, domain shift, prompt error가 있어도 성능이 크게 무너지지 않는 성질.

**Selective prediction/abstention:** 불확실할 때 무리하게 답하지 않고 보류, 추가 계산, 사용자 요청을 선택하는 방식.

**Short-term memory:** 최근 appearance와 motion을 빠르게 반영하는 memory.

**SOTA:** 특정 공개 조건에서 당시 보고된 최고 수준 결과. task와 조건이 다르면 직접 비교할 수 없다.

**Streaming:** 미래 frame을 보지 않고 들어오는 순서대로 제한된 memory와 시간 안에 처리하는 설정.

**Temperature scaling:** logit을 하나의 양수 temperature로 나눈 뒤 probability를 다시 계산하는 단순 calibration 방법.

**Temporal consistency:** 같은 객체의 mask와 identity가 시간에 따라 자연스럽고 안정적으로 이어지는 성질.

**Tensor:** 여러 축을 가진 숫자 배열. image, video, feature, mask의 기본 자료구조.

**Test-time adaptation:** 배포 또는 test data를 보면서 parameter나 statistic을 현장에서 조정하는 방법.

**Token:** Transformer가 처리하는 기본 단위. image patch, prompt, object pointer, memory item 등이 token이 될 수 있다.

**Token pruning:** 중요도가 낮은 token을 attention과 memory 계산에서 제거하는 효율화 방법.

**Tracking-by-detection:** 매 frame detection을 만든 뒤 시간에 따라 같은 객체를 연결하는 tracking 방식.

**Transformer:** token 사이 관계를 attention으로 학습하는 신경망 구조.

**Uncertainty:** 예측이 틀릴 가능성이나 결과의 모호함에 대한 정보. 하나의 confidence 숫자와 동일하지 않다.

**Upsampling:** 낮은 해상도 feature나 mask를 높은 해상도로 키우는 과정.

**Validation set:** 학습 자체가 아니라 model·threshold·hyperparameter 선택에 쓰는 data.

**Visual prompt:** point, box, mask, exemplar image처럼 시각적 위치나 대상을 지정하는 prompt.

**VLM/LVLM/MLLM:** image/video와 language를 함께 이해·생성하는 vision-language model 계열. large 또는 multimodal을 강조할 때 약어가 달라진다.

**VRAM:** GPU가 model weight, activation, memory feature를 저장하는 전용 memory.

**Wavelet:** image를 여러 주파수·scale 성분으로 분해하는 수학적 표현. 미세 경계와 저주파 구조를 따로 다루는 데 쓰인다.

**Zero-shot:** target task의 labeled example로 추가 학습하지 않고 pretrained 능력과 prompt만으로 수행하는 설정.

## 마지막 10문장 요약

- Image는 pixel tensor이고 video는 시간축으로 image가 이어진 tensor다.
- Segmentation은 pixel마다 target 여부를 맞추고 VOS는 같은 target mask를 시간축에서 유지한다.
- SAM 2는 image encoder, prompt encoder, memory attention, mask decoder, memory encoder가 순환한다.
- Memory는 과거를 돕지만 틀린 결과를 저장하면 drift를 증폭시키는 통로가 된다.
- 가림과 비슷한 distractor는 target 존재 판단과 identity 유지 문제를 동시에 만든다.
- Predicted IoU와 object score는 유용한 내부 신호지만 실제 probability라고 가정하면 안 된다.
- Calibration은 score를 실제 실패 빈도와 맞추고, policy는 그 risk에 따라 write·보류·복구를 결정한다.
- 평균 J&F만으로는 memory 오염, 실패 시점, 가림 후 recovery delay를 충분히 설명할 수 없다.
- 공정한 연구는 같은 checkpoint·prompt·dataset split에서 baseline과 ablation을 비교하고 통계적 uncertainty를 보고한다.
- 좋은 결론은 ‘점수가 올랐다’가 아니라 어떤 failure mechanism이 줄었고 어떤 비용을 지불했는지를 말한다.

