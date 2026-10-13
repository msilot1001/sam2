# M6 prior-art / novelty notes (raw)

Reviewer pass on 2026-10-13. Sources are web only. **READ** means I read the method section in full text (arXiv HTML, EuropePMC full-text XML, or thesis PDF). **ABSTRACT** means I saw only the abstract or landing page. **UNVERIFIED** means I could not read the claim myself. All numbers below are copied from the source. None are recomputed.

---

## Track A: crop / zoom / search-region SAM2

### A1. AdaSAM2: "Learning Adaptive Search with Reinforcement Learning for Small and Fast Object Tracking"
- Liu B., Bo X., Luo W., Li H., Wang J., Zheng G., Li S. (Guilin Univ. of Technology). *Sensors* 26(18):5819, Sep 2026. doi:10.3390/s26185819. https://www.mdpi.com/1424-8220/26/18/5819
- **Status:** peer-reviewed journal (MDPI). **READ.** mdpi.com returned 403, so I read the full text from EuropePMC (PMC13611299).
- **Mechanism:**
  - Built on **SAMITE**, not raw SAM2. The task is **VOT** with box metrics (Success, Precision@20, normalized precision).
  - Each frame is a **square crop centred on the predicted target location**. Its side comes from a discrete action set {full image, L/1.5, L/2, L/3, L/4}, where L = max(W,H). The scale is a **fraction of the image**, not a multiple of object size.
  - A **Double-DQN** policy picks the scale. It is a 2-layer MLP (hidden 128) on a 4-D geometric state: log w/W, log h/H, log aspect ratio, and area ratio.
  - The policy is **event-driven**. It runs at initialization and again only when tracking is "unreliable". Otherwise the previous scale is reused.
  - **Recovery** ("lost-aware") works in two steps:
    - It **progressively enlarges** the crop with the number of consecutive unreliable events.
    - After a fixed interval (10 frames), it re-invokes the RL policy. A **constrained re-decision** step rejects a proposed scale that changes too much from the last reliable one.
  - It adds multi-score candidate selection: α·cos-appearance + β·exp(−centre distance/σ) + λ·scale consistency.
- **Memory under crop:** **not specified.** The paper says only that the SAMITE memory is "inherited" and that "outdated memory entries are pruned under a fixed budget". It does not say whether memory is kept in crop coordinates or reset when the scale changes. It also does not say how "unreliable" is detected.
- **Training:** yes. The RL policy is trained offline with IoU-based rewards; the paper does not name the training data. SAMITE/SAM2 is not retrained.
- **Datasets/numbers (Success / Precision vs SAMITE):**
  - TSFMO: 42.6/74.0 → 43.9/75.3
  - LaTOT: Success +0.8, Precision +2.5, ending at 38.6 / 74.3
  - UAV123: 69.6/92.7 → 71.1/94.8
  - UAV123@10fps: 70.5/92.9 → 71.3/94.2
  - UAVDT: 71.0/95.8 → 72.1/96.7
  - The RL policy runs on 0.74% of frames.
  - Fixed heuristic crop policies (1/2 or 1/3, area thresholds 0.0005 / 0.001) helped on UAV123 but hurt on TSFMO.
  - I did not find which SAM2 backbone size was used.
- **Overlap:**
  - **A:** high for the concept (causal crop centred on the previous prediction, scale picked at init and reused).
  - **B:** moderate (enlarge-on-loss recovery).
  - **C:** none.
  - **Differences from our A:** box-level VOT rather than mask J&F; an RL-learned, image-relative scale; SAMITE as the base; no stated memory-coordinate handling; no global–local fusion.

### A2. CTU bachelor thesis: Ruslan Rozumnyi, "Sledování objektu s pomocí SAM, Dino, Monodepth nebo CLIP reprezentace" (Tracking in the Age of SAM, Dino, Monodepth and CLIP)
- Czech Technical University, FEE. Supervisor Jiří Matas; a grading report by A. Lukežič is attached. Issued 2025-01-07. https://dspace.cvut.cz/handle/10467/123552
- **Status:** defended bachelor thesis. It is not a peer-reviewed publication, but it is public. **READ** (thesis PDF, ch. 4–6).
- **Mechanism, variant 1 (crop window):**
  - Take the previous-frame mask's bbox and enlarge each side by a factor N. Enforce a minimum window side m (256 or 512 px), clipped to the image.
  - Crop the next frame to this window and feed it to SAM2.1-L. The window is variable-size and per-frame.
  - Optional occlusion factor k: if the previous mask is empty, enlarge the last non-empty bbox by N·k.
  - Combined with memory stride 5/7 and "do not write empty masks to memory".
- **Mechanism, variant 2 (§5.3.4, parallel fixed-size windows):**
  - Run three SAM2.1 trackers in parallel: a fixed **H/4×W/4** window, a fixed **H/2×W/2** window (each centred on the previous bbox and shifted inside the image), and the **full frame**.
  - **Causal selection rule:** try M/4, then M/2, else M/1. Accept a crop mask only if IoU(crop mask, full-frame mask) > γ (this check is skipped when the full-frame mask is empty) **and** bbox-area / window-area ≤ β (β = 0.1).
  - Oracle selection over the three trackers gives 78.10 vs 73.04 mIoU on VOT2022ST.
- **Memory under crop:** not discussed explicitly. Each tracker seems to just consume the cropped sequence. The author argues that changing crop aspect ratios hurt Accuracy, and this motivated the fixed-size windows.
- **Training:** none (training-free).
- **Numbers (SAM2.1-L):**
  - VOT2022ST: EAO 0.691 → 0.729 and mIoU 0.730 → 0.760 with crop N=3, m=512, ms=5, no empty masks in memory. Accuracy dropped (0.779 → 0.772).
  - Parallel fusion (γ=0.5, β=0.1, ms=5, no empty masks):
    - VOT2022ST: EAO 0.720, A 0.790, R 0.950, mIoU 0.749
    - VOT2020ST: EAO 0.710, A 0.790, R 0.950, mIoU 0.736 (baseline 0.681 / 0.778 / 0.941 / 0.711)
  - VOT24STval: Q 0.371 → 0.380 at best. SAM2.1++ (DAM4SAM) still leads on EAO.
  - Per-sequence gains on tiny targets: drone_flip 1.5 → 66.8 IoU, marathon +22.9, hand2 +19.5.
- **Overlap:**
  - **A:** very high. This is a training-free crop-window SAM2 centred on the previous mask, with fixed-scale windows **and** a full-frame + crop **global–local fusion with a causal per-frame selection rule**.
  - **B:** low to moderate (k-enlargement on loss, no empty masks in memory).
  - **C:** indirect. Zoom raises the effective resolution.
  - **Differences from our A:** a VOT-only evaluation with SAM2.1-L. The crop scale is a fraction of the image or bbox×N, not a single scale fixed from the first-frame mask. Memory in crop coordinates and pasting back are not discussed. There is no DAVIS/PUMaVOS J&F and no Hiera-T.

### A3. SAMITE: "Position Prompted SAM2 with Calibrated Memory for Visual Object Tracking"
- Xu Q. et al. arXiv 2507.21732 (Jul 2025). https://arxiv.org/abs/2507.21732
- **Status:** preprint. No venue found. **READ** (HTML).
- **Mechanism:**
  - A prototypical memory bank scores the last ~30 frames by cosine similarity of FG prototypes to a feature anchor (frame 1) and a position anchor (frame t−1), with α = 0.3. It keeps the top 5 plus the 2 anchors.
  - A positional prompt generator builds a mask prompt from FG−BG prototype similarity × positional prior. The prompt is used only if a cycle-consistency mIoU exceeds 0.7.
  - It uses **full-frame** 1024 input; I saw no crop.
- **Training:** none (zero-shot).
- **Numbers (SAM2.1-T):**
  - LaSOT AUC: SAM2.1 66.7, SAMURAI 69.3, SAMITE 72.8
  - LaSOText AUC: SAM2.1 52.3, SAMITE 57.5
- **Overlap:**
  - **A:** none, except as AdaSAM2's base.
  - **B:** moderate (reliability-calibrated memory with a first-frame anchor).
  - **C:** none.

### A4. Lean-SAM2: "Target-Anchored Memory and Encoder Acceleration for SAM2"
- Ouyang X. et al. arXiv 2607.19811 (Jul 2026). https://arxiv.org/abs/2607.19811
- **Status:** preprint. **READ** (HTML).
- **Mechanism:** an efficiency method in three parts:
  - TAMP: anchor-weighted memory token pruning.
  - TCIM: visibility-gated memory condensation plus an "insurance" bank.
  - TARR: per-window routing. Windows near the previous mask or anchors go to the heavy encoder; the others go to a light bypass network. It falls back to the full frame when the occlusion score is below 0.99 or the area changes by more than 50%.
  - This is **window routing at native scale for speed, not zoom.** Window size and crop/resize behaviour are not specified.
- **Training:** partly. The bypass branch is trained on 30 SA-V videos; TAMP and TCIM are training-free.
- **Numbers:** mostly at or slightly below SAM2.1 for about a 1.41× speedup. For example, the Large model on SA-V val goes 78.5 → 79.8 and on MOSEv1 74.6 → 74.3. No DAVIS results and no small-object results.
- **Overlap:**
  - **A:** low (spatial focusing for efficiency, not resolution).
  - **B:** low.
  - **C:** none.

### A5. Other searched items
- **SAMURAI** (arXiv 2411.11922; venue not shown on arXiv; **ABSTRACT**): training-free. Uses motion-aware memory selection. LaSOT_ext +7.1% AUC and GOT-10k +3.5% AO "over existing trackers". From the abstract alone I could not confirm whether it uses the full frame (UNVERIFIED), although SAMITE's description fits a full-frame design.
- **SatSAM2** (arXiv 2511.18264, preprint, **ABSTRACT**): zero-shot SAM2 satellite tracker with a Kalman-constrained motion module and a motion-constrained state machine. OOTB +5.84% AUC. Cropping is not stated.
- **FocusTrack** (arXiv 2504.13604; **search-result summary only**, I did not open the paper, UNVERIFIED): reportedly an OSTrack-based tracker, not SAM2. A target-presence score expands the search crop when confidence is low. This would be the classic search-region precedent for "enlarge when lost".
- I did not find any other SAM2 paper with a "zoom" or "tiny-object VOS" crop evaluated on DAVIS/MOSE J&F. The searches were broad but not exhaustive.

---

## Track B: occlusion, disappearance and re-acquisition

### B1. DAM4SAM / SAM2.1++: "A Distractor-Aware Memory for Visual Object Tracking with SAM2"
- Videnovic, Lukežič, Kristan. arXiv 2411.17576. https://arxiv.org/abs/2411.17576
- **Status:** the arXiv page says "under review". AdaSAM2's reference list cites it as CVPR 2025, which is secondary evidence. **READ** (HTML).
- **Mechanism:**
  - Memory is split into a Recent Appearance Memory (RAM) and a Distractor-Resolving Memory (DRM).
  - **RAM** is updated every 5 frames and is **not updated when the predicted mask is empty**.
  - **DRM** keeps the init frame plus anchor frames. A frame is added when the chosen and alternative masks diverge (bbox-area ratio < 0.7), the IoU score is > 0.8, and the area is within 20% of the median of the last 10 frames.
- **Training:** none.
- **Numbers (SAM2.1 → SAM2.1++):**
  - DiDi Q: 0.649 → 0.694
  - VOT2020 EAO: 0.681 → 0.729
  - VOT2022 EAO: 0.692 → 0.753
  - LaSOT AUC: 70.0 → 75.1
  - LaSOText AUC: 56.9 → 60.9
- **Overlap:** **B** high (absence-gated memory writes, re-detection memory).

### B2. SAM2Long
- Ding S. et al. arXiv 2410.16268. ICCV 2025 per the arXiv comment. https://arxiv.org/abs/2410.16268
- **Status:** peer-reviewed. **ABSTRACT** only, plus search snippets.
- **Mechanism:**
  - A constrained tree search over multiple segmentation pathways, keeping the paths with the highest cumulative score.
  - Memory entries are weighted by occlusion score. This comes from a secondary snippet and is UNVERIFIED in the paper.
- **Training:** none.
- **Numbers:** an average gain of 3.0 J&F across 24 comparisons, and up to 5.3 on SA-V and LVOS.
- **Overlap:** **B** moderate (robustness to reappearance through path hypotheses). It has no explicit absence classifier.

### B3. SENTRY
- Alansari M. et al. arXiv 2606.24449. "Accepted to ECCV 2026" per the arXiv comment. https://arxiv.org/abs/2606.24449
- **Status:** ABSTRACT only.
- **Mechanism:** "refine-before-write". It builds multiple mask hypotheses per frame, links them into short tracklets, and uses neighbour-aware, cycle-consistent matching to validate memory writes. It plugs into 5 SAM2 trackers, SAMITE among them.
- **Training:** none.
- **Benchmarks:** VOT-style (LaSOT, LaSOT_ext, GOT-10k, VOT20, VOT22, DiDi, and others). It claims zero-shot SOTA. The abstract gives no accuracy numbers, only 32.8 FPS for SAM2-L on an A100.
- **Overlap:** **B** moderate (memory-write validation, not re-acquisition).

### B4. StreamDAM: "Presence-Aware Memory for Real-Time Streaming VOS"
- Xiang Chen. arXiv 2608.03912 (Aug 2026). https://arxiv.org/abs/2608.03912
- **Status:** preprint. **READ** (HTML).
- **Mechanism:**
  - A **learned presence head**: a small causal GRU over runtime scalars (logits, mask-overlap statistics, area ratios, absence run-length). It is trained with class-balanced BCE on GT presence labels from undisclosed held-out sequences.
  - This one presence signal gates memory writes, the read window, **output suppression**, and re-detection.
  - Re-detection relaxes the admission threshold after 5 suppressed frames. **There is no global or crop search.**
  - The paper argues that a fixed policy fails, because true absence and hard visibility need opposite controls.
- **Training:** yes (the presence head). Built on DAM4SAM-L.
- **Numbers (streamed J&F, StreamDAM vs DAM4SAM-L offline):**
  - DAVIS-17 val: .914 vs .914
  - LVOS: .825 vs .838
  - VOST: .544 vs .570
  - MOSE-hard (60-seq subset): .626 vs .610
  - Vanilla SAM2.1-L streamed scores .884 on DAVIS-17 val and .570 on MOSE-hard.
- **Overlap:** **B** high on "distinguish true absence from failure".

### B5. SAM2Dual
- Kim J., Lim C. arXiv 2608.18640 (Aug 2026). https://arxiv.org/abs/2608.18640
- **Status:** preprint, ABSTRACT only.
- **Mechanism:** short-term memory plus interval-sampled long-term memory with gated fusion, and text-aware memory reweighting from early-frame word cues.
- **Training:** none.
- **Numbers:** MOSEv2 J&F 49.33 → 50.65. LVOSv2 "consistent gains", no numbers given.
- **Overlap:** **B** low to moderate.

### B6. Other Track B items
- **SeC (Segment Concept)**, arXiv 2507.15852. Under the title "Advancing Complex VOS via Progressive Concept Construction" it is listed in the ICLR 2026 proceedings (search result). **ABSTRACT.**
  - Builds LVLM concept priors, invoked on scene change.
  - SeCVOS: +11.8 over SAM2.1.
  - Which components are trained: UNVERIFIED. MOSE and DAVIS numbers: UNVERIFIED.
  - **Overlap:** **B** moderate (identity re-acquisition after scene changes).
- **Re-Prompting SAM 3 via Object Retrieval** (arXiv 2603.23788, PVUW 2026 MOSEv2 3rd-place report, **ABSTRACT**).
  - The SAM3 detector proposes same-category candidates. DINOv3 matching against a "transformation-aware target feature pool" selects anchors, which are re-injected as prompts.
  - MOSEv2 test J&F 51.17.
  - **Overlap:** **B** high for "global search + appearance revalidation + re-prompt", but on SAM3 with an external detector.
- **SAMOSA** (arXiv 2605.22538, preprint, alphaXiv summary page).
  - An error detection–recovery module flags drift from aspect, area, or semantic similarity to a **frozen target prototype**. It then re-checks the 3 decoder candidates against the prototype until one matches.
  - It also has an LSTM motion predictor (trained on box trajectories) and a target-aware memory.
  - Anti-UAV300: 71.39%.
  - **Overlap:** **B** moderate. Revalidation is limited to SAM2's own candidates; there is no global search.
- **SAM2Plus (Kalman)**, *Sensors* 25(13):4199, 2025. **READ** (PMC).
  - KF-IoU candidate fusion plus dynamic memory selection. No global re-detection.
  - Training-free.
  - LVOS v2 val (B+) 80.2 → 82.9.
  - **Overlap:** **B** low.
- **AdaSAM2 recovery** (A1) and **CTU k-enlargement** (A2) are crop-enlargement recoveries.

---

## Track C: small and thin detail representation

### C1. SAM2Refiner: "Towards Fine-grained Interactive Segmentation in Images and Videos"
- Yao Y. et al. (Alibaba). arXiv 2502.09660. ICCV 2025. https://arxiv.org/abs/2502.09660
- **Status:** peer-reviewed. **READ** (HTML).
- **Mechanism:**
  - The **SAM2 weights are frozen.**
  - It trains three modules: a Localization Augment module (local cross-attention), Prompt Retargeting, and a multi-scale cascaded **mask refinement that fuses the stride-4/8/16 encoder features to a 1024×1024 mask** (versus SAM2's 256×256 upsampled).
- **Training:** yes, on images only (HQSeg-44K, which includes ThinObject-5K and DIS).
- **Numbers:**
  - ThinObject: 95.9 mIoU / 91.0 mBIoU (HQ-SAM ViT-L: 87.0 / 76.6)
  - LVOS v1 J&F: SAM2 77.2, HQ-SAM2 79.6, SAM2Refiner 80.9
  - LVOS v2 J&F: SAM2 78.3, HQ-SAM2 79.1, SAM2Refiner 79.9
  - No DAVIS results.
  - The authors note that LA causes **boundary flicker in video**.
- **Overlap:** **C** high (frozen backbone with a trained high-resolution refinement head).

### C2. HQ-SAM 2 (SysCV sam-hq, `sam-hq2` folder; beta release 2024-11-17)
- https://github.com/SysCV/sam-hq/tree/main/sam-hq2
- **Status:** code release only. I found no paper. **README READ.**
- **Mechanism:** "extends the HQ-SAM training strategy" to SAM2.1. For HQ-SAM v1 that strategy is a learned HQ output token fused with early/late features while SAM stays frozen. Whether SAM2 is frozen in HQ-SAM 2: UNVERIFIED.
- **Numbers (SAM2.1-L → HQ):** DAVIS val J&F 89.8 → 91.0. MOSE 74.6 → 74.7. Parameters 224.4M → 224.7M.
- **Overlap:** **C** high (lightweight trained add-on for detail, with video numbers).

### C3. MGD-SAM2
- Shen H. et al. arXiv 2503.23786. https://arxiv.org/abs/2503.23786
- **Status:** preprint. **READ** (HTML).
- **Mechanism:** frozen SAM2-B encoder plus MPAdapters. It splits the image into a global view and 4 local 512² patches (MCEM/HMIM fusion), then uses a progressive detail-restoration decoder.
- **Training:** yes (DIS5K, DUTS, HRSOD).
- **Scope:** image-only. No video evaluation.
- **Numbers:** DIS5K max-F 0.933.
- **Overlap:** **C** moderate. It is image-only, but local-patch zoom fusion is conceptually close to A+C.

### C4. Other Track C items
- No SAM2 paper found that targets **small-object VOS** with a frozen-backbone head. The searches were not exhaustive.
- Image-only relatives: arXiv 2601.12147 (segment+matte, 1024 decoder) and arXiv 2607.02708 (frozen Hiera with LoRA and a high-frequency side-stem). Both are ABSTRACT or snippet only.

---

## Per-track verdicts

**A (crop / virtual camera, global–local fusion): mostly covered.**
- The CTU thesis already implements a training-free SAM2.1 crop window centred on the previous mask, with fixed-size windows run in parallel with the full frame and a causal selection rule (IoU with the full frame > γ, and bbox/window ≤ β). It reports VOT gains that concentrate on tiny targets.
- AdaSAM2 (peer-reviewed) adds learned scale selection and enlarge-on-loss recovery.
- **What remains uncovered (narrow):**
  1. A single **object-relative** scale fixed once from the first-frame mask. Prior work uses image-fraction windows or a per-frame bbox×N.
  2. **Explicit** handling of SAM2 memory consistency in crop coordinates (constant scale, so memory features stay scale-consistent), plus paste-back. Neither paper describes this.
  3. Mask-quality evaluation on semi-supervised **VOS** (DAVIS / PUMaVOS J&F, boundary F, size-stratified) with **Hiera-T**. Prior work is VOT-only, with box metrics or VOT EAO, on L or SAMITE.
  4. Failure analysis of when the crop path hurts. The CTU thesis already notes Accuracy drops and aspect-ratio sensitivity.
- **Framing:** a VOS-protocol replication and extension, citing CTU and AdaSAM2. Not a new mechanism.

**B (absence vs failure, re-acquisition): heavily covered.**
- Absence-gated memory: DAM4SAM, CTU.
- Learned presence driving suppression and re-detection: StreamDAM.
- Memory-write validation: SENTRY, SAMITE.
- Prototype revalidation: SAMOSA.
- Global detector search plus appearance revalidation and re-prompt: the PVUW SAM3 report.
- Enlarge-on-loss: AdaSAM2, FocusTrack (unverified).
- **Uncovered sliver:** a **training-free** absence-vs-failure discriminator built only from frozen SAM2 signals and evaluated *as a classifier* on VOS (presence AUROC on MOSE/PUMaVOS). StreamDAM trains a GRU for this.
- **Also uncovered:** periodic full-frame search with first-frame revalidation **inside SAM2 itself** (no external detector) for mask-prompt VOS. This is close to DAM4SAM DRM and SAMOSA in spirit, so novelty is weak.

**C (stride-4 limit, thin components): covered for trained heads.**
- SAM2Refiner (frozen SAM2, 1024 mask, ICCV'25) and HQ-SAM 2 (DAVIS +1.2) already provide frozen-backbone or light-weight detail heads with video numbers.
- **Uncovered:**
  1. Per-component thin/small-structure recall on VOS. Prior work reports global J&F or image mBIoU.
  2. A **training-free** route to detail (crop zoom, i.e. A as a C-solution) quantified on component loss. Prior work shows only VOT IoU.
- A new trained head would need to beat SAM2Refiner and HQ-SAM 2 to be novel.

## Not read / caveats
- SAMURAI, SAM2Long, SENTRY, SAM2Dual and SeC were read at abstract or landing-page level only.
- DAM4SAM's CVPR venue is secondary (cited in AdaSAM2's references).
- HQ-SAM 2 training details are UNVERIFIED.
- FocusTrack is known only from a search-result summary.
- AdaSAM2's backbone size and RL training data are not stated in its text.
