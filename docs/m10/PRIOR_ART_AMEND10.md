# M10 amend10 선행 연구: SAM 2 가설 선택기 / predicted IoU 재보정

작성: 2026-10-15. 범위: 30분 웹 조사 + 기존 노트(`docs/m7/PRIOR_ART_raw*.md`, `docs/m61/PRIOR_ART_raw.md`) 재확인.
대상 실험: frozen SAM 2.1 Hiera-T SGGL(G 전체 frame + L 확대 crop)이 매 frame 내는 6개 가설(2 view × 3 multimask) 중 하나를 고르는 **외부 소형 head**.
입력은 decoder 내부값(mask token 256-d, IoU token, object pointer cosine, predicted IoU, stability, 면적, 이전 출력과의 시간 IoU/중심 거리, cross-view IoU, appearance cosine, view object score, crop 배율).
H1 = view 간(G vs L, 배율이 다름) predicted IoU 보정 + view 선택, H2 = view 안의 3가설 선택. 모델은 view별 affine 보정(4 파라미터) → 스칼라 MLP → MLP + token. 라벨은 가설별 GT J.

읽기 수준 표기: READ = 본문(HTML) 확인, ABS = arXiv 초록 페이지만, NOTE = 이전 M6.1/M7 노트에서 가져옴(이번 세션에서 링크 재확인 안 함).

## 1. 표

| 논문 | 무엇을 학습/보정하는가 | 배율(scale/crop) 간 점수 비교 여부 | 본 실험과의 차이 |
|---|---|---|---|
| **UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model** (ICML 2025) https://arxiv.org/abs/2505.05049 — READ | SAM의 **mask token + IoU token(256+256 = 512-d)을 입력으로 받는 3층 MLP**가 기대 IoU와 설정 간 IoU 차(Δ)를 MSE로 회귀. post-hoc, SAM 재학습 없음. "task gap"은 SamScore로 고른 mask와 3개 multimask 중 최선의 차이를 예측하므로 multimask 선택 문제를 직접 다룬다. | 없음. flip/JPEG/blur/noise 섭동만 쓰며 scale/crop은 입력도 목표도 아니다. | 가장 가깝다. 차이: 이미지 단위(SA-V/MOSE/DAVIS frame을 이미지로 사용), 단일 view, 시간 특징·cross-view 특징·object pointer 없음, 추적 중 선택기로 평가하지 않음. 본 실험의 "MLP + token" 모델은 USAM의 입력 구성과 사실상 같으므로 **반드시 인용하고 baseline으로 재현**해야 한다. |
| **SAMRefiner / SAMRefiner++: Taming Segment Anything Model for Universal Mask Refinement** (ICLR 2025) https://arxiv.org/abs/2502.06756 — READ | SAM 가중치를 고정하고 **IoU head 마지막 층에 LoRA(rank 4) adaptor**만 학습. GT 없이 coarse mask와의 IoU를 pseudo-label로 쓰는 pairwise ranking loss(margin 0.02). 추론 시 adapted IoU가 최대인 multimask 후보를 고른다. DAVIS-585에서 top-1 정확도 44.1 → 63.8. | 없음. 단일 이미지, 단일 배율. | "IoU head를 다시 보정해 multimask 선택을 고친다"는 아이디어는 선행. 차이: 이미지, pseudo-label, in-decoder LoRA(본 실험은 decoder 밖의 외부 head), view/배율 비교 없음, 추적 없음. |
| **Segment Anything with Robust Uncertainty-Accuracy Correlation (RUAC)** (ICML 2026) https://arxiv.org/abs/2605.10603 — ABS | SAM의 단일 IoU mask score가 경계부 신뢰도를 못 잡는 "Mask-level Confidence Confusion"을 지적하고 **경량 pixel-wise uncertainty head**를 style/deformation 공격으로 학습. 23개 zero-shot 도메인. | 없음(초록 기준). | 픽셀 수준 불확실성이지 가설 선택이 아니다. 배율·view·영상 없음. "SAM의 predicted IoU는 그대로 믿을 수 없다"는 동기 인용용. |
| **Uncertainty Measures and Prediction Quality Rating for the Semantic Segmentation of Nested Multi Resolution Street Scene Images** (Rottmann & Schubert, CVPRW 2019) https://arxiv.org/abs/1904.04516 — ABS | 고정된 DeepLabv3+ 위에 **nested center crop(여러 배율)을 같은 크기로 리스케일해 돌린 뒤 crop 간 softmax 평균/분산을 입력으로 segment IoU를 메타 회귀(R² 0.85)·IoU = 0 메타 분류(AUROC 0.90)**. | **있음.** 배율 간 분산이 핵심 입력이다. | 가장 가까운 "배율 간 신호로 segment 품질을 학습" 사례. 차이: semantic segmentation, 영상·추적 없음, 후보 중 선택이 아니라 품질 추정, SAM 아님. 본 실험의 cross-view IoU 특징은 이 논문의 crop 간 분산과 같은 계열이므로 인용 필수. |
| **Multivariate Confidence Calibration for Object Detection** (Küppers et al., CVPRW 2020) https://arxiv.org/pdf/2004.13546v1.pdf — NOTE(M7에서 READ) | 검출기 confidence를 **box 위치·크기에 조건부로** 보정. confidence만 쓴 보정보다 좋다. | 간접. 입력 배율이 아니라 객체 크기에 조건부. | H1의 "배율별 affine 보정"을 뒷받침하는 가장 가까운 보정 문헌. 차이: 검출기, 이미지, 후보 선택 없음. |
| **Mask Scoring R-CNN** (CVPR 2019) https://arxiv.org/abs/1903.00241 — ABS | 예측 mask와 GT의 IoU를 회귀하는 **MaskIoU head**를 추가해 분류 점수와 mask 품질의 불일치를 고친다. | 없음. | 학습된 mask 품질 점수의 고전 선례(SAM의 IoU head 자체가 이 계열). 본 실험은 frozen 모델 **밖**에서 2 view × 3 가설을 비교한다는 점이 다르다. |
| **SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree** (ICCV 2025) https://arxiv.org/abs/2410.16268 — NOTE(M7 READ) | 학습 없음. P개 경로 × 3 multimask, 누적 점수 Σ log(predicted IoU + ε)로 top-P 유지. IoU > 0.3, object score > 0으로 기억 선택. | 없음. 단일 view. | raw predicted IoU만 쓰고 선택기를 학습하지 않는다(확인됨). 본 실험의 선택기는 SAM2Long의 경로 점수에 대체 투입할 수 있다. |
| **SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory** (arXiv 2411.11922; TIP 2026은 snippet) https://arxiv.org/abs/2411.11922 — NOTE(M7 READ) | 학습 없음. argmax[α·IoU(KF box, mask) + (1−α)·predicted IoU], α ≈ 0.15–0.2 수작업. | 없음. | raw 점수의 수작업 가중합. 선택기 미학습(확인됨). |
| **HiM2SAM: Enhancing SAM2 with Hierarchical Motion Estimation and Memory Optimization towards Long-term Tracking** (arXiv 2507.07603) https://arxiv.org/abs/2507.07603 — 검색 요약 수준 | 학습 없음. Kalman 기반 motion 점수와 SAM2 confidence를 고정 α로 선형 결합해 proposal 재순위. 어려운 frame에서만 CoTracker3로 전환. | 없음. | 수작업 선형 결합. 선택기 미학습(검색 요약으로만 확인). |
| **SatSAM2: Motion-Constrained Video Object Tracking in Satellite Imagery using Promptable SAM2 and Kalman Priors** (arXiv 2511.18264) https://arxiv.org/abs/2511.18264 — 검색 요약 수준 | 학습 없음. top-3 mask 중 Kalman prior와의 일관성으로 선택. | 없음. | 규칙 기반 선택. 위성 영상. |
| **A Distractor-Aware Memory for Visual Object Tracking with SAM2 (DAM4SAM / SAM2.1++)** (CVPR 2025) https://arxiv.org/abs/2411.17576 — NOTE(M7 READ) | 학습 없음. 선택된 mask와 대안 mask(largest CC)의 bbox 면적비 < 0.7로 distractor 판단, predicted IoU > 0.8로 기억 admission. | 없음. 단일 view, "introspection" 규칙. | multimask 대안 가설을 **선택이 아니라 기억 게이트의 신호**로 쓴다. 본 실험은 그 가설들 중 출력을 고른다. |
| **Prompt Self-Correction for SAM2 Zero-Shot VOS** (*Electronics* 14(18):3602, 2025) https://www.mdpi.com/2079-9292/14/18/3602 — NOTE(M6.1, 페이지 403으로 snippet만; **미검증**) | 학습 없음. particle filter box와 object pointer 거리가 임계를 넘으면 box 재프롬프트. | 없음. | 규칙 기반 재프롬프트. 선택기 미학습(snippet 기준). |
| **Live Interactive Training (LIT-LoRA)** (arXiv 2603.26929) https://arxiv.org/abs/2603.26929 — ABS(부록 B 내용은 M6.1 노트) | 사용자 교정으로 LoRA를 온라인 갱신. 부록 B: SAM2 predicted IoU를 오류 검출기로 쓰면 VOST에서 최적 임계 ≈ 0.8, 정확도 0.74에 그치고 데이터셋마다 다르며 품질을 **과소 추정**. | 없음. | predicted IoU가 그대로는 신뢰 불가하다는 **VOST 수치 근거**. 보정은 시도하지 않았다. |
| **Calibrating Promptable Concept Segmentation via Paraphrase Consistency (EPC)** (ICML 2026) https://icml.cc/virtual/2026/74287 — 학회 페이지 초록만(arXiv ID 미확인) | 학습 없음. SAM 3의 텍스트 프롬프트를 바꿔 말한 뒤 일치하는 검출만 유지해 ECE 0.140 → 0.114. | 없음(프롬프트 간 일관성). | "같은 대상을 다른 조건으로 다시 물어 일치로 신뢰도를 보정"한다는 점에서 cross-view IoU 특징과 정신이 같다. 차이: 텍스트, 이미지, 무학습, 후보 선택 없음. |
| **SAM 2++: Tracking Anything at Any Granularity** (arXiv 2510.18822) https://arxiv.org/abs/2510.18822 — ABS | 통합 decoder가 mask/box/point 후보를 IoU token으로 점수화해 선택(검색 요약). 모델 전체를 학습. | 없음. | frozen SAM 2 + 외부 head가 아니라 재학습된 통합 모델. 후보 선택을 IoU token에 맡긴다는 점만 공통. |
| **Rethinking Memory Design in SAM-Based Visual Object Tracking** (arXiv 2512.22624) https://arxiv.org/abs/2512.22624 — ABS | SAM2/SAM3에 기존 기억 정책들을 재구현하고 단기/장기(distractor) 기억을 나눈 하이브리드 제안. 초록에는 학습된 선택기 없음. | 없음. | 기억 정책 비교 논문. 가설 선택 head 아님(초록 기준, **본문 미확인**). |
| **StreamDAM** (arXiv 2608.03912) — NOTE(M7 READ; 이번 세션 링크 미재확인) | 런타임 스칼라(logits, overlap 통계, 면적비, 부재 run-length)로 **GRU presence head**를 GT presence에 BCE로 학습. raw logit 임계보다 false-absent/false-present 모두 낮음. | 없음. 단일 view. | "SAM 2 런타임 스칼라 위의 소형 학습 게이트"라는 틀이 가장 비슷한 추적 논문. 차이: 목표가 presence(있냐/없냐)이지 가설 간 선택이 아니고, 배율 조건 없음. |
| **CTU 학사 논문 "Tracking in the Age of SAM, Dino, Monodepth and CLIP"** https://dspace.cvut.cz/handle/10467/123552 — NOTE(M7 READ) | 학습 없음. crop 결과와 full-frame 결과의 IoU > γ, bbox/window ≤ β 규칙으로 crop 채택. | **있음(규칙).** crop vs full 두 결과를 IoU로 비교. | SGGL의 뿌리. G/L 선택을 raw 규칙으로 한다. 본 실험은 같은 선택을 학습하고 배율 보정을 더한다. |

이번 세션에서 검색했으나 **찾지 못한 것**: "DecoupledSAM2"라는 제목의 논문(검색 결과 없음; 가장 가까운 것은 arXiv 2506.22880 "Decoupled Seg Tokens…"로 LLM 기반 reasoning 분할이라 무관), SAM 2 predicted IoU에 temperature/Platt scaling을 적용한 논문, SAM/SAM 2에서 **crop 배율에 따른 predicted IoU 편향**을 측정하거나 보정한 논문, 추적 중 **GT IoU로 학습한 reranker**로 SAM 2 가설을 고르는 논문.

## 2. 직접 겹치는 것

- **UncertainSAM(ICML 2025)**: frozen SAM의 mask token + IoU token을 MLP에 넣어 기대 IoU를 회귀하고 multimask 선택 차이까지 예측한다. 본 실험의 "MLP + token" 모델(H2 부분)과 입력·목표가 거의 같다. 남는 차이는 영상 추적 맥락(시간 특징, object pointer, 기억에 쓰이는 선택), 두 view/배율의 비교, 그리고 선택기로서의 J&F 평가다. **이것을 baseline으로 넣지 않으면 H2는 신규 기여로 주장하기 어렵다.**
- **SAMRefiner++(ICLR 2025)**: frozen SAM의 IoU head를 ranking loss로 재보정해 multimask top-1을 고친다. H2의 "스칼라 보정" 수준과 겹친다. 차이는 이미지·pseudo-label·단일 배율.
- H1(배율/view 간 predicted IoU 보정 + G/L 선택)을 **학습**으로 푼 논문은 찾지 못했다. 가장 가까운 것은 Rottmann & Schubert(배율 간 분산으로 segment IoU 메타 회귀, 비추적)와 Küppers(크기 조건부 보정, 검출기), 규칙으로는 CTU 논문이다.

## 3. 결론

SAM 2 추적기 계열(SAM2Long, SAMURAI, HiM2SAM, DAM4SAM, SatSAM2, Prompt Self-Correction)은 모두 raw predicted IoU에 수작업 가중/임계를 얹을 뿐 선택기를 학습하지 않는다는 점이 다시 확인됐다.
그러나 **이미지 영역에서는 frozen SAM의 decoder token 위에 소형 MLP를 얹어 IoU를 회귀하는 것(UncertainSAM)과 IoU head를 재보정해 multimask 선택을 고치는 것(SAMRefiner++)이 이미 있다.** 따라서 H2 단독은 선행 연구의 영상 적용에 가깝고, 신규성은 H1 쪽, 즉 **다른 배율의 두 view가 낸 가설을 한 척도로 보정해 비교하고 그 선택을 기억에까지 쓰는 것**에 있다. 보고서에는 UncertainSAM·SAMRefiner++를 baseline, Rottmann & Schubert·Küppers를 H1 근거로 인용해야 한다.
