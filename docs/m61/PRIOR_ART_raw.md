# M6.1 prior-art / novelty notes (raw), Agent R

Date: 2026-10-13. Sources are web only. This extends `docs/m6/PRIOR_ART_NOTES_raw.md` (CTU thesis, AdaSAM2, SAMITE, Lean-SAM2, DAM4SAM, StreamDAM, SAM2Refiner, HQ-SAM 2, MGD-SAM2) and does not repeat those entries.

Labels:
- **READ**: I read the method section (arXiv HTML or ar5iv) through a fetch-and-summarise tool. The numbers are copied from that output; none were recomputed.
- **ABSTRACT**: I saw only the abstract or landing page.
- **SNIPPET/UNVERIFIED**: I saw only a search-engine summary.

Caveat: every "READ" went through an LLM page summariser, not a raw PDF read. Re-check any number before citing it in a paper.

---

## Mechanism A: tracker-prompted local re-segmentation (crop/zoom) and global–local fusion

### A-1. HQTrack: "Tracking Anything in High Quality"
- Zhu et al., arXiv 2307.13974. https://arxiv.org/abs/2307.13974
- **Status:** VOTS2023 challenge technical report (2nd place). Not peer-reviewed. **READ.**
- **Mechanism:**
  - A video multi-object segmenter (VMOS, DeAOT-style) predicts a mask.
  - The enclosing box of that mask prompts a pre-trained HQ-SAM (ViT-H) on the **original full image**. The paper describes no crop or zoom.
  - **Selection rule:** the HQ-SAM mask is used if IoU(VMOS, HQ-SAM) > τ, with τ = 0.1; otherwise the VMOS mask is kept.
  - The paper does not say whether refined masks are written back to memory.
- **Training:** the mask refiner (MR) is not trained. VMOS is trained.
- **Numbers:**
  - VOTS2023 test AUC: VMOS 0.596, VMOS + SAM_H 0.610, HQTrack 0.615.
  - Validation: VMOS alone 0.650; with τ = 0.1, 0.708 (τ = 0 gives 0.702).
  - No DAVIS evaluation.
- **Overlap:** **A high.** This is the closest published "tracker box → SAM image-mode re-segmentation, gated by agreement with the tracker". The difference is that it runs on the full frame (no zoom), uses a different tracker (not SAM2 itself), and has no memory write-back.

### A-2. Track Anything (TAM)
- Yang et al., arXiv 2304.11968. https://arxiv.org/abs/2304.11968
- **Status:** technical report. **READ.**
- **Mechanism:**
  - XMem propagates the mask. When XMem's quality is "not satisfactory" (metric not defined), the XMem mask plus point prompts derived from XMem probes/affinities go to SAM for refinement.
  - **Refined masks are added back into XMem memory.**
  - Crop vs full frame: not stated.
- **Training:** none.
- **Numbers:** DAVIS-2016 val J&F 88.4 and DAVIS-2017 test-dev 73.1, both with click init. XMem with mask init gets 92.0 and 81.2. Click init vs mask init is a different protocol, so these are not comparable.
- **Overlap:** **A moderate.** It has tracker → SAM image refinement → memory write-back, plus a quality trigger (which is **D**). It has no zoom.

### A-3. SAM-PD: "How Far Can SAM Take Us in Tracking and Segmenting Anything in Videos by Prompt Denoising"
- arXiv 2403.04194. https://arxiv.org/abs/2403.04194
- **Status:** preprint. **READ.**
- **Mechanism:**
  - The previous-frame mask box becomes the next-frame box prompt.
  - It adds jittered and scaled copies (9 boxes × 2 scales) and picks the candidate whose feature best matches the first-frame template.
  - A point refinement step uses the positive point farthest from the boundary and keeps the largest of the 3 outputs.
  - The SAM encoder runs on the **full frame**; crops are taken only on feature maps.
- **Training:** none (SAM ViT-H).
- **Numbers:** DAVIS-17 val J&F 71.3 (SAM-PT 76.6). YouTube-VOS-18 G 63.0.
- **Overlap:** **A moderate** (box from the previous prediction re-prompts SAM image mode). It has no zoom and no SAM2 memory.

### A-4. Alpha-Refine
- Yan et al., arXiv 2012.06815. Accepted to CVPR 2021 per the arXiv note. https://arxiv.org/abs/2012.06815
- **Status:** peer-reviewed. **READ** (ar5iv).
- **Mechanism:**
  - A plug-in refiner for any box tracker. It crops a **concentric search region of about 2× the predicted box**, resizes it to 256×256, and predicts a refined box and a mask (pixel-wise correlation, corner head, U-Net mask head).
  - The mask head is disabled by default at inference.
- **Training:** yes (LaSOT, GOT-10k, ImageNet VID/DET, COCO, YouTube-VOS). Loss is box MSE plus 1000 × BCE.
- **Numbers:**
  - VOT2020 EAO: DiMPsuper 0.314 → ARDiMPsuper 0.471. It is unclear whether this was computed on masks or boxes.
  - LaSOT AUC: SiamRPN++ 47.6 → 55.9; DiMP50 55.9 → 60.2.
- **Overlap:** **A high in geometry** (tracker box → fixed object-relative zoom crop → mask). However, it is a trained, dedicated refiner, not SAM2's own decoder.
- **Context:** the VOTS2024 results report (SNIPPET only, https://link.springer.com/chapter/10.1007/978-3-031-91767-7_24) says over 2/3 of submissions used SAM and 1/5 used AlphaRef "for target segmentation or for refining the mask". One entry passed DMAOT masks and boxes to HQ-SAM. "Tracker box → SAM-family refinement" is therefore standard practice in VOTS. Whether those entries crop: UNVERIFIED.

### A-5. f-BRS "Zoom-In"
- Sofiiuk et al., CVPR 2020, arXiv 2001.10331.
- **Status:** peer-reviewed. **SNIPPET only.**
- **Mechanism:** interactive segmentation. From the 3rd click on, it crops to the current mask's bbox padded by 40% per side, resizes the long side to 400 px, and re-segments. The authors say this helps small objects.
- **Overlap:** **A** (fixed object-relative zoom re-segmentation from the current mask). It is image-only and interactive. This crop is widely reused in later interactive segmentation work (RITM, SimpleClick, etc.; UNVERIFIED here).

### A-6. FocSAM
- Huang et al., CVPR 2024, arXiv 2405.18706. https://arxiv.org/abs/2405.18706
- **Status:** peer-reviewed. **READ.**
- **Mechanism:**
  - The paper argues that SAM's fixed preprocessing **prevents image-level zoom-in**.
  - Rather than cropping the image, it restricts window attention (Dwin-MSA) to embedding windows that intersect the previous-mask box, adds P-DyReLU, and replaces the image embedding.
- **Training:** yes (decoder fine-tuned, then a refiner trained). The encoder is frozen.
- **Numbers:** NoC@90 on DAVIS: SAM 5.19 → FocSAM 4.77. Mean: 6.85 → 5.38.
- **Overlap:** **A low to moderate.** It confirms the motivation (SAM benefits from focusing on the object), but it solves it in feature space and needs training.

### A-7. CascadePSP
- Cheng et al., CVPR 2020, arXiv 2005.02551. https://arxiv.org/abs/2005.02551
- **Status:** peer-reviewed. **READ.**
- **Mechanism:**
  - Class-agnostic refiner taking image plus coarse mask. A **global step** runs on the image downsampled to long side 900, with a 3-level cascade (stride 8/4/1).
  - A **local step** runs on overlapping 900×900 crops (stride L/2 − 32, 16 px edges trimmed) with a 2-level cascade. Overlapping outputs are **averaged**.
- **Training:** yes (36,572 images; masks perturbed from GT). Losses:
  - stride 8: CE
  - stride 4: CE + L1/L2
  - stride 1: L1/L2 plus a **Sobel-gradient boundary loss** with α = 5
- **Numbers:** BIG, DeepLabV3+ input: IoU 89.65 → 92.01, mBA 60.94 → 75.59. No video.
- **Overlap:**
  - **A:** the global-then-local probability fusion idea.
  - **B high:** image + logits input, boundary-aware high-resolution loss.

### A-8. GLNet: "Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images"
- Chen et al., CVPR 2019 (oral), arXiv 1905.06368.
- **Status:** peer-reviewed. **ABSTRACT.**
- **Mechanism:** a global branch on the downsampled image and a local branch on patches, with deep feature fusion between them. Trained.
- **Overlap:** **A** conceptually (global–local fusion), but learned fusion at the feature level and image-only.

### A-9. Remote SAMsing: "From Segment Anything to Segment Everything"
- arXiv 2605.00256. Submitted to *JAG*; a ScienceDirect listing exists (search result).
- **Status:** preprint / journal (unclear). **READ.**
- **Mechanism:**
  - Training-free SAM2.1-L run on padded T×T tiles. Tile size acts as an implicit zoom.
  - Multi-pass grid prompting, with predicted-IoU and stability thresholds decaying from 0.93 to 0.60.
  - Best-match merging across tile borders.
  - It uses SAM2's predicted IoU **raw**; there is no scale-dependent calibration.
- **Numbers:**
  - BSB-1 Det@0.5: 56.0% → 85.1% with T 1000 → 250 (full pipeline).
  - SAM2's `crop_n_layers` gives about +6 pp.
- **Overlap:** **A** (training-free zoom improves small objects for SAM2), but image "segment everything", not VOS. Indirect evidence for **D**: the paper uses IoU thresholds across scales without calibrating them.

### A-10. CamoSAM2 (arXiv 2504.00375) and SAM2-for-VCOS (arXiv 2409.18653)
- **CamoSAM2:** **SNIPPET.** Iteratively expands the SAM2 box prompt outward and re-queries SAM2.
- **SAM2-for-VCOS:** **READ.**
  - Feeds another model's mask to SAM2 as a mask prompt for refinement. Crop vs full frame: not stated.
  - SAM2-T mIoU 0.388 → 0.392, while mDice *dropped* (0.458 → 0.442).
  - **Weak evidence** that naive mask-prompt re-segmentation with Hiera-T helps little.
- **Overlap:** **A low to moderate.**

### A-11. SAM 2 paper (arXiv 2408.00714): **SNIPPET**
- The SA-V data engine prompts grids on **zoomed crops** (4 and 16 crops) to surface small objects. This is a data-annotation use, not tracking-time re-segmentation.

### Verdict A
- **Already known:**
  1. Tracker box → SAM-family image-mode refinement, gated by agreement IoU with the tracker. HQTrack, VOTS practice, and TAM (TAM also does memory write-back).
  2. Fixed object-relative zoom crop around the tracker box. Alpha-Refine (2×, trained); f-BRS Zoom-In (+40%, interactive).
  3. Crop-window SAM2 with full-frame selection. CTU thesis, AdaSAM2 (m6 notes).
  4. Global + local averaging / fusion. CascadePSP, GLNet (trained).
- **Not found (searches broad, not exhaustive):**
  1. Using **SAM2's own video-tracker output** as the prompt for **SAM2's own image decoder on a zoomed crop at a fixed object-relative scale**, evaluated on VOS J&F with Hiera-T. HQTrack/TAM/SAM-PD all run on the full frame; Alpha-Refine uses a separate trained net.
  2. **Probability-level fusion** of global and crop SAM2 logits after geometric alignment, in VOS.
  3. Writing a crop-refined mask back into SAM2 memory, and measuring the drift and feedback effects.
- **Framing:** A is an incremental combination of known parts (HQTrack + Alpha-Refine/f-BRS zoom + CTU). Novelty claims should be limited to "SAM2-self-prompted zoom, VOS evaluation, fusion and memory-write analysis".

---

## Mechanism B: lightweight learned refinement head on frozen features (output-only)

### B-1. CascadePSP: see A-7
- **Overlap: B high.** Trained class-agnostic refiner (image + coarse mask), boundary gradient loss, global and local high-resolution passes.

### B-2. SegRefiner: "Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process"
- Wang M., Ding H. et al., NeurIPS 2023, arXiv 2312.12425.
- **Status:** peer-reviewed. **READ.**
- **Mechanism:**
  - Image + coarse mask go into a U-Net, and discrete diffusion runs T = 6 steps.
  - High-resolution handling: global steps first, then a final local step on patches centred at low-confidence pixels (NMS 0.3).
  - Loss: BCE + gradient "texture" L1 (α = 5).
- **Training:** yes (LVIS; HR model fine-tuned on DIS5K + ThinObject-5K).
- **Numbers:**
  - BIG with DeepLabV3+: IoU 89.42 → 94.86, mBA 60.25 → 77.64.
  - COCO Mask R-CNN R50: AP 39.8 → 41.9, boundary AP 27.3 → 32.6.
  - 0.52 s/img for 1 step, 2.94 s for 6 steps.
  - No video.
- **Overlap:** **B high** (model-agnostic output-only refiner, boundary loss). Slow.

### B-3. BPR: "Look Closer to Segment Better: Boundary Patch Refinement"
- Tang et al., CVPR 2021, arXiv 2104.05239.
- **Status:** peer-reviewed. **ABSTRACT/SNIPPET.**
- **Mechanism:** extracts small patches along predicted boundaries and refines each at higher resolution with a dedicated net (image + mask patch). Post-processing for any instance-segmentation model.
- **Numbers:** Cityscapes test AP 42.7 (PolyTransform + SegFix + BPR).
- **Overlap:** **B high** (output-only, high-resolution boundary refinement).

### B-4. HQ-SAM: "Segment Anything in High Quality"
- Ke et al., NeurIPS 2023, arXiv 2306.01567.
- **Status:** peer-reviewed. **ABSTRACT.**
- **Mechanism:** a learnable HQ output token plus fusion of early/final ViT features with decoder features. SAM weights are preserved. Trained on 44K fine masks in about 4 h on 8 GPUs.
- **Overlap:** **B high**, but it sits inside the decoder rather than output-only.
- Its SAM2 successor, HQ-SAM 2, and SAM2Refiner (ICCV'25) are in the m6 notes. Both have video numbers.

### B-5. Video Mask Transfiner (VMT)
- Ke et al., ECCV 2022, arXiv 2207.14012.
- **Status:** peer-reviewed. **ABSTRACT.**
- **Mechanism:** finds sparse error-prone spatio-temporal regions per tracklet and refines them with high-resolution features using an efficient transformer. Trained. Introduces HQ-YTVIS and Tube Boundary AP.
- **Overlap:** **B + C** (video, high-resolution, error-region refinement). The task is VIS, not SAM2.

### B-6. MoSAM
- arXiv 2505.00739. https://arxiv.org/abs/2505.00739
- **Status:** preprint. **READ.**
- **Mechanism:** frozen SAM2 with trained motion encoders (sparse key points and dense optical flow → warped mask → box prompt), plus frame-level memory selection (IoU > 0.7, occlusion ≥ 0) and pixel-level selection (confidence threshold not given).
- **Training:** the motion encoders, on DAVIS-17.
- **Numbers (J&F):**
  - LVOS v1 L: 80.2 → 84.6
  - SA-V val L: 78.6 → 81.6
  - SA-V test L: 79.6 → 81.5
  - Tiny on LVOS v1: +3.6
  - No DAVIS or MOSE evaluation, since DAVIS was used for training.
- **Overlap:** **B** (trained add-on on frozen SAM2), **C** (flow-warped mask as a prompt), **D** (IoU-score memory gating).

### Verdict B
- **Already known:** output-only class-agnostic refiners with boundary losses and global/local passes (CascadePSP, SegRefiner, BPR), in-decoder HQ tokens (HQ-SAM, HQ-SAM 2), and a frozen SAM2 with a 1024 refinement head and LVOS numbers (SAM2Refiner).
- **Not found:**
  1. A head that consumes **SAM2's internal stride-4/8 features plus decoder logits plus RGB** (rather than RGB + binary mask), is trained cheaply, and is evaluated **output-only on VOS** without writing to memory, stratified by object size. This differs only marginally from SAM2Refiner and HQ-SAM 2.
  2. To claim anything, the head must beat or match HQ-SAM 2 (DAVIS +1.2 on L) and CascadePSP as an off-the-shelf post-processor. Running CascadePSP on SAM2 VOS output is an obvious, cheap baseline that I did not see reported for SAM2.

---

## Mechanism C: temporal correction (flow warping, consistency checks, component retention)

### C-1. ViperSAM: "Inference-Time Temporal Probability Smoothing for Stable Video Segmentation with SAM2 under Weak Prompts"
- Deka, arXiv 2604.17115. Listed for ICCCV 2026; acceptance not confirmed.
- **Status:** preprint. **READ.**
- **Mechanism:**
  - Training-free. Warps the previous refined probability map with Farnebäck flow.
  - A forward–backward flow residual gives the motion uncertainty Q_t; the entropy of the current SAM2 prediction gives R_t.
  - Per-pixel Kalman-like gain K = Q/(Q+R), clipped to [0.05, 0.95]. The output P̂ = K·P_t + (1−K)·warp(P̂_{t−1}) is thresholded at 0.5 and used recursively.
- **Numbers:** only 4 self-selected 1080p videos and GT-free stability metrics.
  - Warped IoU: 0.9620 → 0.9768
  - Boundary F: 0.4834 → 0.5533
  - **No J&F and no DAVIS.**
  - Internal inconsistencies: +14.45% vs +17.12%.
- **Overlap:** **C very high** (flow warp, fb-consistency, entropy gating, frozen SAM2). The difference is that it is not evaluated on accuracy benchmarks. An accuracy evaluation on DAVIS/MOSE is still open.

### C-2. SAM-PT: "Segment Anything Meets Point Tracking"
- Rajič et al., arXiv 2307.01197. WACV 2025 per mlanthology.
- **Status:** peer-reviewed. **ABSTRACT/SNIPPET.**
- **Mechanism:** point trackers (e.g. CoTracker/PIPS) propagate positive and negative points (K-Medoids initialised), which prompt SAM per frame. It uses iterative mask refinement passes and point re-initialisation.
- **Training:** none.
- **Numbers:** DAVIS-17 val 76.6 (as quoted in the SAM-PD paper).
- **Overlap:** **C moderate** (motion-propagated prompts rather than warped masks).

### C-3. MoSAM: see B-6
- **Overlap:** **C** (dense optical flow warps the mask to produce a box prompt for SAM2). Trained.

### C-4. FlowI-SAM / FlowP-SAM
- arXiv 2404.12389.
- **Status:** **SNIPPET.**
- **Mechanism:** optical flow as SAM input or prompt for moving-object segmentation, plus frame linking. Training is likely (UNVERIFIED).
- **Overlap:** **C low to moderate.**

### C-5. SAM2Plus (Kalman, in the m6 notes) and SAMURAI (C/D, below)
- Motion priors applied at the box level, not the mask level.

### C-6. Prompt Self-Correction for SAM2 Zero-Shot VOS
- *Electronics* (MDPI) 14(18):3602, 2025. https://www.mdpi.com/2079-9292/14/18/3602
- **Status:** peer-reviewed journal, but the page returned 403. **SNIPPET only.**
- **Mechanism:** a particle filter tracks the box. Re-inference with the corrected box prompt is triggered when the object-pointer distance exceeds a threshold while the object score is > κ.
- **Numbers (snippet):** the DAVIS mIoU is insensitive to κ (0.9035 to 0.9030).
- **Overlap:** **C/D moderate** (temporal consistency check triggers a box re-prompt).

### Verdict C
- **Already known:** flow-warped prior fusion with fb-consistency and confidence gating on frozen SAM2 (ViperSAM, training-free, stability-only evaluation), flow-warped mask → prompt (MoSAM, trained), point propagation (SAM-PT), and temporal-trigger re-prompting (Prompt Self-Correction).
- **Not found:**
  1. **Accuracy** (J&F, size-stratified) evaluation of training-free flow-warp fusion on DAVIS/MOSE with SAM2.1-T.
  2. Connected-component retention rules for thin or small parts. No paper found.

---

## Mechanism D: calibrated reliability to choose between global and crop predictions

### D-1. SAMURAI
- Yang et al., arXiv 2411.11922.
- **Status:** preprint (venue not shown). **READ.**
- **Mechanism:**
  - Mask selection: M* = argmax[α_kf · IoU(KF box, mask) + (1−α_kf) · predicted IoU]. α = 0.15 is best in Table 4; the text says 0.2.
  - Memory admission requires predicted IoU, object score and motion score each above a threshold. The threshold values are not reported.
  - Uses the **raw** predicted IoU (trained with MAE); there is **no calibration**.
- **Numbers:**
  - LaSOT AUC: SAM2.1-L 68.54 → SAMURAI-L 74.2
  - SAMURAI-T 69.3
  - Ablation: neither component 68.32, both 74.23
  - No VOS numbers.
- **Overlap:** **D high** for "fuse SAM2's quality score with an external consistency score to select among candidates". It is hand-weighted, not calibrated, and does not select between scales.

### D-2. Live Interactive Training (LIT)
- Yang X. et al. (Cornell), arXiv 2603.26929.
- **Status:** preprint. **READ** (Appendix B).
- **Finding:** used as an error detector, SAM2's predicted IoU is "**unreliable in practice**".
  - On VOST, the predicted-IoU threshold that best matches GT IoU = 0.5 is about 0.8, and accuracy peaks at only 0.74.
  - The best threshold varies by dataset.
  - Predicted IoU **underestimates** quality, so low thresholds give a high false-negative rate.
- **Overlap:** **D: direct motivation.** The raw IoU needs calibration and is dataset-dependent. LIT did not attempt to calibrate it.

### D-3. RUAC: "Segment Anything with Robust Uncertainty-Accuracy Correlation"
- arXiv 2605.10603. ICML 2026 per the arXiv comment.
- **Status:** peer-reviewed. **ABSTRACT.**
- **Mechanism:** names SAM's IoU mask score as a source of "Mask-level Confidence Confusion". It trains a lightweight pixel-wise uncertainty head and a Bayesian decoder with style and deformation attacks, across 23 zero-shot domains. The abstract gives no numbers.
- **Overlap:** **D moderate** (SAM's confidence is unreliable, so learn a separate one). Image-only.

### D-4. Conformal prediction for SAM instance segmentation
- arXiv 2602.10045.
- **Status:** **SNIPPET.**
- **Mechanism:** argues that SAM's predicted IoU has no accuracy guarantee and proposes prediction sets with coverage guarantees.
- **Overlap:** **D low to moderate.**

### D-5. Selection rules as reliability gates (already in the m6 notes)
- CTU thesis: IoU(crop, full) > γ and bbox/window ≤ β.
- HQTrack: IoU > 0.1.
- DAM4SAM DRM admission: IoU score > 0.8.
- MoSAM: IoU > 0.7 for memory.
- StreamDAM: a learned GRU presence head, which is the closest "small learned gate on SAM2 runtime scalars".
- Remote SAMsing: raw IoU thresholds across tile scales.

### Verdict D
- **Already known:**
  1. SAM2's predicted IoU is used everywhere as a hand-thresholded gate.
  2. It is reported unreliable and dataset-dependent (LIT).
  3. Learned gates on SAM2 runtime scalars exist for presence (StreamDAM).
  4. Agreement-IoU gating between tracker and refiner exists (HQTrack, CTU).
- **Not found:**
  1. **Scale-dependent calibration** of SAM2's predicted IoU and object score: the same decoder on a zoomed crop vs the full frame, with reliability diagrams or ECE per crop scale.
  2. A small logistic gate on (calibrated IoU at both scales, agreement IoU, area ratio, ...) for **choosing between global and crop** predictions, evaluated as a selector (oracle gap, AUROC) on VOS. This is the most clearly uncovered sliver, though it is narrow and incremental relative to the CTU and HQTrack rules.

---

## Not read / caveats
- **ABSTRACT or SNIPPET only:** HQ-SAM v1 internals, BPR, VMT, GLNet, SAM-PT, FlowI-SAM, the conformal SAM paper, CamoSAM2, f-BRS Zoom-In details, the VOTS2024 report, and Prompt Self-Correction (MDPI 403).
- Whether any VOTS2023–2025 entry crops before SAM refinement is UNVERIFIED. The individual tracker appendices were not read. This is the most likely place for an unknown direct precedent for A.
- Alpha-Refine: it is unclear whether its VOT EAO uses masks or boxes.
- SAMURAI: the τ thresholds are not reported.
