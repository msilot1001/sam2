# M8 prior-art / novelty audit (raw)

Date: 2026-10-14. Scope: co-absence rescue mechanisms (both SGGL views report the target absent) on frozen SAM 2.1 Hiera-T.
I compared at the level of individual operations. I did not survey every topic.

READ levels:
- **READ**: I read the method section (full text pulled through arXiv HTML/PDF or the CVF PDF and searched for the relevant passages).
- **READ-sum**: I read a tool summary of the method section on the arXiv HTML page. This is method-level, but not every equation was checked.
- **ABSTRACT**: only the abstract, arXiv page or search snippet.
- **PRIOR**: cited from background knowledge. I did not re-open it this session.

A venue marked UNVERIFIED means I could not confirm it this session. I invented no numbers. Every number below was quoted from the source.

---

## 0. Context: SGGL's own closest work (re-checked only briefly)

| Work | Venue / id | Overlapping op | Read |
|---|---|---|---|
| R. Rozumnyi, "Tracking in the Age of SAM, Dino, Monodepth and CLIP" | CTU FEE bachelor thesis, Jan 2025 (dspace.cvut.cz/handle/10467/123552) | Keeps a "window around the object" (crop) through the video for SAM2. Also optical flow in SAM2 mask selection, a larger memory stride, and no empty masks in memory | ABSTRACT (search snippet; the DSpace page did not render) |
| AdaSAM2, "Learning Adaptive Search with Reinforcement Learning for Small and Fast Object Tracking" | Sensors (MDPI) 26(18):5819, 2026 | Built on SAMITE. An RL policy chooses the search region (the "resolution-context trade-off") | ABSTRACT (MDPI returned 403; snippet only) |
| Alpha-Refine | arXiv 2012.06815 (CVPR 2021 per common citation, UNVERIFIED this session) | Refinement on a concentric search region "two times the size" of the base tracker's box | READ (grep of PDF) |
| SAM2Refiner, "Towards Fine-grained Interactive Segmentation in Images and Videos" | ICCV 2025 (poster page), arXiv 2502.09660 | See Section 3. Its Localization Augment uses **4 fixed quadrant sub-images** upsampled and cross-attended to global features. It is not an object-centred crop | READ-sum |

---

## 1. VOTS2025 challenge report (READ: full PDF text, Sec. 3 + Appendix A/B/C)

Source: Kristan et al., "The Third Visual Object Tracking Segmentation VOTS2025 Challenge Results", ICCVW 2025 (CVF open access).
- 32 submissions across 3 sub-challenges. VOTS2025 had 15 valid entries. "80% of submissions were based on the SAM2 tracker and 33% on its recent extension DAM4SAM".
- "33% of the trackers localized the target in a constrained search region, while 67% applied image-wide target localization." The report does not say which trackers do which.
- Winner: S3-DAM4SAM (Q 0.751). Next: DAM4SAM 0.711, DAMTrack 0.706, SAMPT 0.704. Baseline SAM2: 0.642.
- VOTS-RT2025: SV-DAM4SAM is top on the committee hardware run. VOTSt2025: of 5 trackers, 3 SAM2-based, and all of them "localized the target in the entire image".

| Tracker (VOTS2025) | SAM2-based | Crop/local + full-frame/global? | Re-detection / recovery mechanism | Relevance to M8 |
|---|---|---|---|---|
| S3-DAM4SAM (A.8, winner) | yes (DAM4SAM + Cutie, shared encoder) | no | none stated. Fuses two trackers by "IoU consistency, mask quality assessment, temporal stability" | output-level tracker fusion (cf. SGGL gate / rankers) |
| DAM4SAM (A.1) | yes | no | implicit via distractor-resolving memory | memory admission (#11) |
| DAMTrack (A.6) | yes (DAM4SAM + DMAOT) | no | Kalman-box IoU picks between two trackers when their scores differ by > 0.4 | motion-based arbitration (#7, #10) |
| SAMPT (A.14) | yes (DAM4SAM + CoTracker) | no | CoTracker points pick the SAM2 mask and serve as **point prompts** when masks score low | point-track -> SAM prompts (#8, #1) |
| DQASOT / DQASOTg / DQASOTb (A.12/A.4/A.2) | partly (SAM2 or DMAOT chosen per sequence) | no | "Reinitialization via language-grounded detection" with Grounding DINO + text prompt after prolonged occlusion | global re-detection by external detector (#1) |
| DAM4SAM2-SAM1 (A.5) | yes | no | none (SAM1 post-refinement) | - |
| MIEM-HSE ALPHACHIP Qwen VL (A.15) | SAM2 only for masks | Qwen2.5-VL searches "close to the given" previous position | VLM returns boxes or "absence". Heuristic candidate weighting (distance, area ratio, IoU) | candidate ranking heuristic (#10) |
| MMF Tracker (A.13, withdrawn) | yes (+Depth-Anything-V2) | no | none | - |
| HQ-DMAOT, DMAOT TTA, DeSAMTrack, EVPTrack, edgesam | no / SAM only as refiner | DMAOT TTA: **multi-scale resized views**, each with an isolated memory snapshot, averaged | none | multi-view + per-view state isolation (cf. SGGL's two states) |
| MSAOT (B.4, RT baseline) | no (AOT) | "For tiny objects, we use MixFormer ... to coarsely locate the objects before applying MS-AOT" | - | coarse-locate-then-segment for tiny objects (cf. H_s8, F_oclr) |
| DAM-RMemAOT (C.2, VOTSt) | no (RMemAOT) | no | "When all trackers fail simultaneously ... recovery protocols using historical mask analysis, morphological expansion" | co-failure trigger in a parallel multi-tracker (closest in spirit to "co-absence" trigger) |

**Finding:** no VOTS2025 SAM2-based entry combines a cropped/local SAM2 view with a full-frame SAM2 view. Re-detection in VOTS2025 SAM2 entries is either implicit (memory), done by an external detector (Grounding DINO, Qwen-VL), or done by point tracks (CoTracker). DAM-RMemAOT has a "fire when all parallel trackers fail" trigger, but it uses AOT and recovers by morphology. VOTS2024 (earlier session text): "All submissions applied the same network for frame-to-frame localization and target re-detection."

---

## 2. Per-candidate prior art

### 2.1 A_cos: frame-0 mask-pooled prototype -> cosine heatmap on SAM 2 backbone -> top-k peaks -> SAM 2 point prompts (into the tracker's own memory)

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| PerSAM, "Personalize Segment Anything Model with One Shot" | ICLR 2024 (arXiv 2305.03048) | SAM encoder features of reference masked; per-pixel cosine "location confidence map" (averaged over fg features); **highest = positive point, lowest = negative point**; mask-pooled target embedding T_R added to decoder tokens; evaluated on DAVIS17 (PerSAM 66.9, PerSAM-F 76.1 J&F) frame-by-frame vs reference | READ |
| Matcher | ICLR 2024 (arXiv 2305.13310) | DINOv2 patch cosine correspondence, bidirectional matching, **k-means++ (K=8) clusters of matched points -> SAM point prompts**, instance-level filtering (emd, purity, coverage); VOS mode keeps the reference frame fixed in memory; DAVIS17 79.5 J&F | READ-sum |
| SAMITE | arXiv 2507.21732 (venue none listed) | On SAM2: **FG and BG prototypes by GAP of mask-pooled features**, cosine of current-frame features vs FG and BG prototypes, Disc = ReLU(FG−BG), re-weighted by a positional prior near the previous mask, then used as **pseudo mask prompt**; cycle-consistency gate decides whether to prompt; training-free | READ (PDF Sec 3.2) |
| Lean-SAM2 | arXiv 2607.19811 | Prompt-derived **foreground anchor tokens** (8-64, diversity-greedy), cosine similarity of memory tokens to anchors, used for memory pruning and window routing. Has **no point prompts and no re-detection** | READ-sum |
| Track Anything (TAM) | arXiv 2304.11968 (tech report) | XMem "probes and affinities" **projected to point prompts for SAM** when quality is low; refined mask written back to XMem memory | READ |
| SAM-Track (Segment and Track Anything) | arXiv 2305.06558 (tech report) | Detects **new** objects every n-th frame (SAM everything / Grounding DINO) and compares with tracked masks. No lost-object re-detection | READ-sum (ar5iv) |
| SAM-PT | WACV 2025 (arXiv 2307.01197) | Prompts come from **point tracks** (8 K-Medoids positives + 1 negative), not from feature matching; optional re-init every 8 frames | READ-sum |
| DAM4SAM | CVPR 2025 | Re-detection is implicit through memory attention (DRM). No explicit heatmap or prompts | READ-sum |
| SAM2Long | ICCV 2025 (CVF) | Recovery through a memory tree (P=3). "does not describe a separate re-detection module" | READ-sum |
| Re-Prompting SAM 3 via Object Retrieval (Gao, Li, Han) | arXiv 2603.23788 (PVUW 2026 MOSE 3rd) | Detector candidates -> **DINOv3 matching to a target feature pool** -> selected masks fed back to the SAM 3 tracker "as initial memory at their frame indices" with the first-frame mask | ABSTRACT + search snippet |
| SAM-H / WOFTSAM (Šerých, Matas) | arXiv 2602.19624 | SAM2.1-tiny with DAM4SAM-style memory. Re-detection is credited to SAM2 itself; DINOv2 used only to resolve corner ambiguity | READ-sum |
| StreamDAM | arXiv 2608.03912 | Re-detection = **relaxing memory admission** after K=5 suppressed frames; no feature matching / prompting | READ-sum |
| VOTS2025 DQASOT | ICCVW 2025 report | Grounding-DINO text re-init after long occlusion | READ |
| Classic: SiamRCNN | CVPR 2020 (CVF) | Re-detection over the full image against the first-frame template, plus tracklet DP | READ (grep) |
| Classic: GlobalTrack | AAAI 2020 (arXiv comment) | Full-image query-guided search | ABSTRACT |

Residual difference: the mechanism (a) fires **only on co-absence of two SAM 2 views**, (b) uses SAM 2's **own Hiera features** with no extra backbone, and (c) injects the peaks as **point prompts into the running video state**, so they are written to memory. Among the works checked, it is closest to PerSAM's location prior and Matcher's matched-point prompts applied inside a SAM2 video state. SAMITE already does FG/BG-prototype cosine maps as prompts on SAM2 (mask prompt, every frame, local-biased).
Risk: **HIGH** for the operation itself. Novelty can only be claimed for the trigger and its placement in the dual-view pipeline.

### 2.2 B_att: contrastive multi-token attention (cell softmax over k-means fg tokens + bg tokens of frame 0; fg mass = posterior)

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| VideoMatch | ECCV 2018 (arXiv comment) | Soft matching layer: cosine of each pixel vs **first-frame FG set and BG set**, mean of top-K → FG/BG scores → softmax | READ (PDF Sec 3.2) |
| FEELVOS | CVPR 2019 | **Global matching** of current pixels to first-frame embeddings (nearest neighbour distance maps per object/bg) + local matching to previous frame | READ (grep) |
| STM / XMem / Cutie memory readout | ICCV 2019 (PRIOR) / ECCV 2022 (verified) / CVPR 2024 (verified) | Softmax attention of query pixels over memory keys whose values encode the mask. B_att with frame-0-only memory is a **non-learned, label-valued special case** of this readout | XMem READ (grep); Cutie ABSTRACT; STM PRIOR |
| Cutie | CVPR 2024 | Object queries + **foreground-background masked attention** separating fg and bg semantics | ABSTRACT |
| SAMITE PPG | arXiv 2507.21732 | FG prototype − BG prototype contrast map (single prototype each, not multi-token softmax) | READ |
| PerSAM target-guided attention | ICLR 2024 | Adds α·softmax(S) bias to decoder cross-attention | READ |
| DiMP | arXiv 1904.07220 (ICCV 2019, UNVERIFIED this session) | Discriminative target model learned from **target and background** | ABSTRACT |
| KeepTrack | ICCV 2021 | Distractor-aware candidate association, discriminative base tracker | READ |
| Lean-SAM2 anchors | arXiv 2607.19811 | Multi-token fg anchors (diversity-greedy, max-cos), **fg only, no bg contrast** | READ-sum |

Residual difference: only the parameter-free use on frozen SAM 2 Hiera features as a co-absence re-detector. The k-means fg tokens + bg tokens + softmax mass is VideoMatch/FEELVOS-style label propagation.
Risk: **HIGH**.

### 2.3 H_s8: prototype matching on stride-8 FPN features for tiny targets

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| SAM2Refiner | ICCV 2025 | Multi-scale cascade using hierarchical encoder features for **mask refinement** (not matching) | READ-sum |
| FEELVOS / VideoMatch | CVPR 2019 / ECCV 2018 | Pixel-level matching on embedding maps (resolution choice is a design parameter) | READ (grep) |
| MSAOT (VOTS-RT baseline) | ICCVW 2025 report | Tiny objects handled by a separate coarse locator (MixFormer) | READ |
| SAM 2 itself | ICLR 2025 | Stride-4/8 high-res features feed the mask decoder | PRIOR |

Residual difference: choosing the higher-resolution SAM 2 FPN level for matching is an engineering variant of A_cos.
Risk: **LOW as a contribution, because there is nothing novel to claim**. Treat it as an ablation of A_cos.

### 2.4 G_neg: automatic negative point prompts at distractor peaks

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| PerSAM | ICLR 2024 | Automatic negative point at the **lowest**-similarity location (background, not distractor) | READ |
| SAM-PT | WACV 2025 | 1 negative point per mask sampled from the **background**, tracked; used in pass 2 | READ-sum |
| Matcher | ICLR 2024 | Uses no negative points. Distractors are rejected by instance-level filtering (emd/purity/coverage) | READ-sum |
| "Prompt Self-Correction for SAM2 Zero-Shot Video Object Segmentation" | Electronics 14(18):3602, 2025 (MDPI) | Particle-filter box + current box re-prompt SAM2 with **negatives sampled uniformly outside both boxes**. Best result with 3 negatives | ABSTRACT/snippet only (MDPI 403); details UNVERIFIED |
| Bridge the Points | arXiv 2410.06964 | Positive-Negative Alignment using background references to choose point prompts (image few-shot) | ABSTRACT/snippet |
| Few-Shot Semantic Segmentation Meets SAM3 | arXiv 2604.05433 | Auto negatives from low-confidence candidates to suppress look-alike distractors. Reports that negatives **consistently degrade** performance | ABSTRACT/snippet |
| SAM-PD | arXiv 2403.04194 | Notes negative point prompts "may occasionally lead to unstable mask predictions" | ABSTRACT/snippet |
| DAM4SAM / SAMITE | CVPR 2025 / arXiv | Distractor handling by memory (DRM) or by a positional mask prompt, not by negative clicks | READ-sum / READ |

Residual difference: placing negatives specifically at **secondary prototype peaks (distractors)** in a SAM 2 *video* state. I found no exact match. Several works do automatic negatives (background-based or random), and one 2026 SAM3 study reports harm.
Risk: **MEDIUM**.

### 2.5 D_ref: re-prompt / decode with only the frame-0 memory

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| SAM 2 | ICLR 2025 | Always conditions on the prompted (frame-0) memory plus 6 recent memories | PRIOR (also restated in SAMITE text, READ) |
| DAM4SAM | CVPR 2025 | Init frame "always kept in the memory" (reserved DRM slot) | READ-sum |
| SAMITE PMB | arXiv 2507.21732 | Frame-1 and frame-t−1 prototypes are anchors; memory subset selection | READ |
| SAM2Refiner | ICCV 2025 | Video mode "re-uses the prompt input from the first frame" for every later frame | READ-sum |
| PerSAM / Matcher (VOS) | ICLR 2024 | Per-frame segmentation from the reference only. Matcher keeps the reference fixed | READ / READ-sum |
| Re-Prompting SAM 3 via Object Retrieval | arXiv 2603.23788 | Anchors fed back "with the first-frame mask" | ABSTRACT |
| SAM2Long / SAMURAI / MoSAM / HiM2SAM | ICCV 2025 / arXiv | Memory **subset selection** policies (which entries are attended) | READ-sum |

Residual difference: none at the operation level. Restricting attention to the conditioning frame is a memory-selection setting.
Risk: **HIGH** (non-novel). Keep it only as a diagnostic or ablation.

### 2.6 A_crop / B_crop: heatmap- or attention-guided crop placement

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| AdaSAM2 | Sensors 2026 | Learned (RL) search-region selection on SAMITE/SAM2 | ABSTRACT |
| DaSiamRPN | arXiv 1808.06048 (ECCV 2018, UNVERIFIED this session) | On failure: **local-to-global search** (search region 255 → 767; enter/leave thresholds 0.8/0.95) | READ (grep) |
| SPLT (Skimming-Perusal) | arXiv 1909.01840 (ICCV 2019, UNVERIFIED) | When the target is absent, a skimming module selects the "most possible regions" among sliding windows for global search | ABSTRACT |
| LTMU | CVPR 2020 | Local tracker + verifier + SiamRPN-based **global re-detector**; meta-updater | READ (grep) |
| FOVEA / SALISA | ICCV 2021 / ECCV 2022 | Saliency from previous detections steers *where resolution goes* (warp, not crop) | READ / ABSTRACT |
| SAMITE | arXiv | Prior-mask heatmap as prompt (not crop placement) | READ |
| MIEM Qwen-VL (VOTS2025) | ICCVW 2025 | VLM search near the previous position, then a SAM2 mask | READ |

Residual difference: centring SAM 2's object-relative crop view on a training-free feature heatmap peak, inside a dual-state SAM 2. The concept "global coarse localisation, then a local high-res window" is classic.
Risk: **MEDIUM-HIGH**.

### 2.7 E_beam: velocity-extrapolated multi-hypothesis crop centres

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| SAMURAI | arXiv 2411.11922 (venue none listed) | Kalman box state [x,y,w,h, velocities]. Mask choice by α_kf·KF-IoU + (1−α_kf)·affinity (paper inconsistent: 0.15 in table vs 0.2 in text). Motion-aware memory selection | READ-sum |
| SAM2Long | ICCV 2025 | **P=3 pathways** of mask hypotheses with cumulative log-IoU scores; best path output | READ-sum |
| SENTRY | ECCV 2026 (arXiv 2606.24449) | Multiple mask hypotheses backtracked into short tracklets; cycle-consistent matching before memory write | ABSTRACT |
| MoSAM | arXiv 2505.00739 | Sparse motion prompt: 5 object points **linearly extrapolated** to the next frame → positive points. Dense: flow-warped mask → box prompt | READ-sum |
| HiM2SAM | arXiv 2507.07603 | Kalman coarse + CoTracker3 fine motion scoring of SAM2 proposals | READ-sum |
| SAM-PD | arXiv 2403.04194 | **Multiple jittered/scaled box prompts** from the previous box; picks the most template-similar mask | ABSTRACT |
| DaSiamRPN | ECCV 2018 (UNVERIFIED) | Local-to-global search expansion on failure | READ (grep) |
| DAMTrack (VOTS2025) | ICCVW 2025 | Kalman prediction arbitrates between two trackers | READ |
| Prompt Self-Correction (Electronics 2025) | MDPI | Particle filter → corrected box re-prompt | ABSTRACT/snippet, UNVERIFIED |

Residual difference: the hypotheses here are **input view placements** (crop centres) for a frozen SAM 2 crop state, not masks or memory paths. Velocity extrapolation, Kalman filtering and multi-hypothesis tracking are all standard.
Risk: **MEDIUM**. The beam over view placements is a small step from SAM-PD (jittered boxes) and SAM2Long (beam over masks).

### 2.8 C_flow: RAFT flow-carried box -> crop centre / SAM 2 box prompt

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| MoSAM | arXiv 2505.00739 | Optical flow (network unnamed) masked by the prediction. Mask warped forward, and the **warped region's box becomes the SAM2 box prompt** | READ-sum |
| SAM-PT | WACV 2025 | Point tracks (PIPS/CoTracker) → SAM point prompts per frame | READ-sum |
| SAMPT (VOTS2025) / HiM2SAM | ICCVW 2025 / arXiv | CoTracker points for SAM2 mask selection and point prompting | READ / READ-sum |
| FlowSAM, "Moving Object Segmentation: All You Need Is SAM (and Flow)" | ACCV 2024 (CVF) | Flow as SAM input or **flow as prompt**. RAFT used; previous masks warped with flow | ABSTRACT + snippet |
| Segment Any Motion in Videos | CVPR 2025 | Long-range point tracks + DINO features; SAM2 iterative prompting | ABSTRACT |
| MaskTrack (Perazzi et al.) | arXiv 1612.02646 (CVPR 2017, UNVERIFIED) | Previous-mask guidance + "MaskTrack+Flow" (flow-magnitude stream) | READ (grep) |
| Rozumnyi thesis | CTU 2025 | Optical flow in SAM2 mask selection | ABSTRACT |
| Turbulence-Robust Dynamic Object Segmentation | arXiv 2605.29292 | RAFT motion map → connected components → **SAM2 box prompts** (training-free) | ABSTRACT/snippet |

Residual difference: using flow only during co-absence to move the crop view, with no per-frame flow prompting.
Risk: **HIGH** for the operation (flow→box→SAM2 prompt is published). Novelty would be limited to the trigger.

### 2.9 F_oclr (occupancy-controlled crop) vs F_cpfr (context-preserving foveated warp + inverse mask warp). CHECKED CAREFULLY

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| Recasens et al., Learning to Zoom: a Saliency-Based Sampling Layer | ECCV 2018 | Saliency → non-uniform sampling grid (learned), "degrades gracefully to uniform"; gaze + fine-grained classification | ABSTRACT (+ PDF downloaded, not inspected in detail) |
| Thavamani et al., FOVEA | ICCV 2021 | Saliency = **KDE of the previous frame's boxes** (temporal prior). **Separable** warp (keeps boxes axis-aligned, and the paper says it improves AP). **Anti-cropping** constraint so the whole frame stays visible. Boxes mapped back by the backward map. Explicitly notes that unwarping a **segmentation mask** needs an inversion of T⁻¹. Works with and without fine-tuning | READ (PDF Sec 3) |
| Thavamani et al., LZU (Learning to Zoom and Unzoom) | CVPR 2023 | Zoom, compute features, then **unzoom features** via a piecewise-linear invertible approximation. **Separable warps** (non-separable "no better"). Semantic segmentation on Cityscapes. "Adaptive" variant uses previous-frame detections as the temporal prior | READ (PDF Sec 4) |
| Jin et al., Learning to Downsample (LDS) | ICLR 2022 | Learned non-uniform downsampling for ultra-high-res **segmentation**, edge-concentrated. LZU says LDS "does not unzoom" (the network learns the mapping) | ABSTRACT + LZU text |
| SALISA | ECCV 2022 | Saliency-based TPS resampling for video object detection; small-object gain | ABSTRACT |
| **ZoomTrack**, Kou et al., "Target-aware Non-uniform Resizing for Efficient Visual Tracking" | NeurIPS 2023 spotlight (arXiv 2310.10071) | **Tracking**. Previous-frame box → Gaussian importance. **Axis-aligned (separable) grid** with zoom energy and **explicit magnification factor γ** (=1.5, measured 1.342× on target size). Rigid energy limits aspect distortion. Linear constraint **"avoid cropping ... so that the original visual field is also retained"**. **Reverse box mapping** back to the image. Compared against FOVEA (FOVEA hurt tracking). Works **test-time-only on off-the-shelf trackers** (LaSOT/TNL2K gains without retraining, Table 4 ②) | READ (PDF Sec 3-5) |
| AdaSAM2 | Sensors 2026 | RL-chosen search region for SAM2 (resolution-context trade-off) | ABSTRACT |
| DMAOT TTA (VOTS2025) | ICCVW 2025 | Multi-scale resized views, with outputs mapped back and averaged | READ |
| Siamese crop normalisation (e.g. ZoomTrack Eq. 1 context factor f) | standard | Crop side = f × target size, so the target occupies a fixed share of the input. **This is F_oclr's principle** | READ (ZoomTrack Sec 3.1) |

Notes:
- ZoomTrack warps the **search crop** (context factor 5), not the full frame. FOVEA and LZU warp the **full frame** but are detection/semantic-segmentation methods, not VOS.
- I found **no** work that applies a context-preserving separable foveation to a **memory-based VOS model (SAM 2)** with **inverse warping of the predicted mask** and consistent memory across varying warps. Searches for "foveated / non-uniform SAM 2 VOS" returned nothing relevant (search-level evidence only).
- F_oclr (choosing zoom so the target spans N stride-16 cells) is target-size-normalised cropping. That is standard in Siamese and transformer trackers (context factor), and AdaSAM2 learns the same choice.

Residual difference for F_cpfr: (i) the full frame is warped inside a **memory-propagating** segmenter, so memory features are written in warped coordinates that change every frame; (ii) mask-level inverse warp (FOVEA flags this as harder); (iii) frozen SAM 2, used only as a fallback view.
Risk: F_cpfr **HIGH** (ZoomTrack + FOVEA + LZU cover the operation; only the VOS/memory setting differs). F_oclr **HIGH** (standard).

### 2.10 Rankers: set-attention over competing proposals with explicit none/abstain token, conditioned on frame-0 fg/bg tokens

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| KeepTrack | ICCV 2021 | Candidates = local maxima of the score map (score ≥ τ). Each is encoded as (score, position, appearance). **SuperGlue-style self/cross-attention GNN** across candidates. Optimal matching with **dustbins** for "disappeared / newly appeared". Target-confidence for memory | READ (PDF Sec 3) |
| DETR | arXiv 2005.12872 (ECCV 2020, UNVERIFIED this session) | Set prediction with bipartite matching. The "no object" ∅ class is a PRIOR recollection (abstract does not mention it) | ABSTRACT + PRIOR |
| LTMU | CVPR 2020 | Offline-trained meta-updater (cascaded LSTM) gives a binary update decision; online verifier | READ (grep) |
| SiamRCNN | CVPR 2020 | Re-detection of first-frame template + tracklet dynamic programming over candidates | READ (grep) |
| Cutie | CVPR 2024 | Object queries attending fg/bg masked pixels | ABSTRACT |
| Matcher | ICLR 2024 | Hand-designed proposal scoring (emd, purity, coverage) + top-k selection | READ-sum |
| StreamDAM | arXiv 2608.03912 | Learned GRU presence estimator over runtime scalars (logits, overlap, area ratios, absence run-length) that gates output and memory | READ-sum |
| S3-DAM4SAM / DAMTrack / MIEM (VOTS2025) | ICCVW 2025 | Output arbitration between trackers or candidates (heuristic) | READ |

Residual difference: a small trained set-attention head over **SAM 2 rescue proposals plus SGGL views**, with an abstain token, conditioned on frame-0 tokens. KeepTrack's association net with dustbins is the same idea family (learned set attention with a "no match" slot).
Risk: **MEDIUM-HIGH**. The architecture is not novel; only the proposal source and the abstain-vs-SGGL decision differ.

### 2.11 Memory admission of a rescued candidate

| Work | Venue | Overlapping operation | Read |
|---|---|---|---|
| DAM4SAM | CVPR 2025 | RAM every Δ=5 frames, skipped when the mask is empty. DRM admits on alt-mask divergence (bbox area ratio < θ_anc=0.7) + predicted IoU > 0.8 + area within 20% of the 10-frame median | READ-sum |
| SAMURAI | arXiv 2411.11922 | Memory only if mask-affinity, object score and motion score pass thresholds | READ-sum |
| SAM2Long | ICCV 2025 | Admit if IoU > 0.3 and occlusion score > 0; attention weights [0.95, 1.05] by rank | READ-sum |
| MoSAM ST-MS | arXiv 2505.00739 | Frames with IoU > 0.7 and occlusion > 0, ranked; pixel-level confidence masking | READ-sum |
| HiM2SAM | arXiv 2507.07603 | Short-term by IoU + motion score; long-term when proposals are mutually distinct (Hausdorff) | READ-sum |
| SENTRY | ECCV 2026 | "Refine-before-write": verify the memory write by short-horizon consistency | ABSTRACT |
| StreamDAM | arXiv 2608.03912 | Presence-gated admission p_j > τ; relaxation after K=5 suppressed frames | READ-sum |
| Lean-SAM2 | arXiv 2607.19811 | Insurance bank of high-occlusion-score entries as "recovery anchors" | READ-sum |
| SAM2Dual | arXiv 2608.18640 | Short-term + interval-sampled long-term memory, gated fusion, text-aware reweighting | ABSTRACT |
| Track Anything | arXiv 2304.11968 | **SAM-refined mask (from point prompts) added to XMem memory** | READ |
| Re-Prompting SAM 3 via Object Retrieval | arXiv 2603.23788 | **Retrieved candidate masks encoded as memory at their frame indices** | ABSTRACT + snippet |

Residual difference: writing an **externally rescued** mask (from A_cos / C_flow etc.) into a frozen SAM 2 state after a co-absence episode. TAM and the SAM 3 re-prompting report already write externally obtained masks into tracker memory, and gated admission is crowded.
Risk: **HIGH**.

---

## 3. Specific details requested

**DAM4SAM (CVPR 2025; arXiv 2411.17576; READ-sum)**
- RAM: FIFO of 3, temporal encoding, update every 5 frames, never when the mask is empty. DRM: reserved init-frame slot + 3-slot FIFO of anchor frames, no temporal encoding.
- Distractor detection: alternative SAM2 masks diverge from the output (bbox area ratio < 0.7), with a reliability gate (IoU > 0.8, area within 20% of median). The paper says it is "insensitive to the exact value".
- No crop/search region. Re-detection only through memory. About 20% slower than SAM2.1 (11 vs 13.3 fps).
- VOTS2025: DAM4SAM is 2nd (Q 0.711). The top 4 extend DAM4SAM.

**SAM2Refiner (ICCV 2025; arXiv 2502.09660; READ-sum)**
- SAM2 frozen; new modules trained on image data only.
- Localization Augment: 4 non-overlapping fixed quadrants, upsampled, encoded, then cross-attention fusion with global features (G_ref = CrossAttn(G_t, L_t, L_t)). It is **not object-centred** and not a tracking crop.
- Prompt Retargeting: dense prompt map (pos clicks, neg clicks, mask). Mask Refinement: multi-scale cascade on hierarchical encoder features.
- Video: SAM2 streaming memory; "re-uses the prompt input from the first frame" every frame; no re-detection.
- Overlap with M8: weak. Relevant only to H_s8 (hierarchical features) and as a local+global fusion precedent for SGGL.

---

## 4. Not found / unverified

- **AdaSAM2 full method**: MDPI returned 403. Only abstract-level claims (RL search region on SAMITE; TSFMO 42.6→43.9 Success; UAV123 69.6→71.1). Whether it keeps a full-frame view is UNVERIFIED.
- **Rozumnyi thesis**: abstract only this session; crop details UNVERIFIED here.
- **"Prompt Self-Correction for SAM2 ..." (Electronics 2025)**: snippet only; all details UNVERIFIED.
- **VOTS2025 tracker source code**: not inspected. Descriptions are 1-paragraph author summaries.
- I found no paper triggering re-detection on **agreement of absence across two simultaneous SAM 2 views** (search-level negative evidence only).
- I did not locate the SAM2-based VOTS2025 entries that use a "constrained search region" (the 33%). The report gives no per-tracker breakdown, and none of the appendix descriptions mention a crop.
- Venues UNVERIFIED this session (standard citations): DiMP (ICCV 2019), DETR (ECCV 2020), DaSiamRPN (ECCV 2018), Alpha-Refine (CVPR 2021), MaskTrack (CVPR 2017), SPLT (ICCV 2019), STM (ICCV 2019).

---

## 5. Summary table

| Candidate | Closest work | Overlapping operation | Remaining difference | Novelty risk |
|---|---|---|---|---|
| A_cos | PerSAM (ICLR'24); Matcher (ICLR'24); SAMITE (arXiv'25); TAM | Mask-pooled prototype → cosine map → peak points → SAM prompts (PerSAM, Matcher); FG/BG prototype cosine maps as SAM2 prompts (SAMITE); memory affinities → SAM points (TAM) | Fires only on dual-view co-absence. Uses SAM 2's own Hiera features. Points enter the running video state | HIGH |
| B_att | VideoMatch (ECCV'18); FEELVOS (CVPR'19); STM/XMem/Cutie readout | Per-pixel matching vs first-frame FG and BG sets → FG/BG posterior; attention readout over memory | Parameter-free on frozen SAM 2 features; used as a re-detector only | HIGH |
| H_s8 | SAM2Refiner (hierarchical features); FEELVOS | Matching or refinement on higher-res feature levels | Resolution choice only (an ablation of A_cos) | LOW (nothing to claim) |
| G_neg | PerSAM; SAM-PT; Electronics'25 (UNVERIFIED); FSS-meets-SAM3 (arXiv'26) | Automatic negative points (lowest-sim / background / random outside boxes / low-confidence candidates) | Negatives placed on **secondary prototype peaks (distractors)** inside a SAM 2 video state | MEDIUM |
| D_ref | DAM4SAM (init frame always kept); SAM2Refiner (re-uses frame-0 prompt); Matcher VOS | Restrict or anchor memory to the conditioning frame | None at the operation level | HIGH (diagnostic only) |
| A/B_crop | DaSiamRPN local-to-global; SPLT; LTMU; AdaSAM2 | Global coarse localisation → place a local search window | Heatmap from frozen SAM 2 features places SGGL's crop state | MEDIUM-HIGH |
| E_beam | SAM2Long (P=3 beam); SAM-PD (jittered boxes); SAMURAI / MoSAM (Kalman / linear extrapolation) | Multi-hypothesis tracking; velocity extrapolation; multiple box prompts | Beam over **input crop placements** of a frozen SAM 2 view, not over masks or memories | MEDIUM |
| C_flow | MoSAM (flow-warped mask → box prompt); SAM-PT; FlowSAM (ACCV'24); turbulence paper (RAFT → box prompt) | Flow/point-track → SAM 2 box/point prompts | Used only on co-absence to move the crop view | HIGH |
| F_oclr | ZoomTrack Eq. 1 / Siamese context factor; AdaSAM2 | Crop size tied to target size so the target occupies fixed input cells | Target expressed in SAM 2 stride-16 cells | HIGH |
| F_cpfr | **ZoomTrack (NeurIPS'23)**; FOVEA (ICCV'21); LZU (CVPR'23) | Previous-box-centred **separable** non-uniform resize with controlled magnification, **no cropping (full field kept)**, reverse mapping; test-time on off-the-shelf trackers (ZoomTrack). Full-frame anti-crop separable warp + inverse (FOVEA/LZU) | Applied to a **memory-based VOS model** with **mask-level inverse warp** and memory written under per-frame-varying warps; frozen SAM 2 | HIGH (op), MEDIUM only if the memory/warp-consistency issue is the claim |
| Rankers | KeepTrack (ICCV'21) | Learned set attention over score-map candidates with **dustbin** (no-match) slots | Proposals are SAM 2 rescue masks + SGGL views; frame-0 fg/bg-token conditioning | MEDIUM-HIGH |
| Memory admission | DAM4SAM; SAMURAI; SAM2Long; SENTRY; StreamDAM; TAM; Re-Prompting SAM 3 | Gated memory writes; external masks written into tracker memory | Admission of a co-absence rescue into a frozen dual-view SAM 2 | HIGH |

Bottom line: none of the M8 candidate operations is novel on its own. The only plausibly unclaimed element across all the checked work is the **co-absence trigger in a dual-view (object-relative crop + full-frame) frozen SAM 2 tracker**, together with whatever empirical behaviour it shows. F_cpfr would be new only as "foveation inside a memory-propagating VOS model", and ZoomTrack must be cited as the direct precedent.
