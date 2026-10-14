# M7 prior-art / novelty notes (raw)

Date: 2026-10-14. Web sources only. This extends `docs/m6/PRIOR_ART_NOTES_raw.md` and `docs/m61/PRIOR_ART_raw.md`. Entries already READ there are summarised only where M7 needs a new detail.

This review is scoped to the M7 candidates:
- **C**: state-aware reacquisition gate
- **D**: observability-aware global veto
- **E**: identity verification
- **G-probe**: resolution-sensitivity probe
- **MA**: memory admission
- **F**: D + E combined

## Reading-level labels
- **READ**: I read the method section this session (arXiv HTML or ar5iv) through a fetch-and-summarise tool. Thresholds are copied from that output. Re-check any number against the PDF before citing it in a paper.
- **READ (M6/M6.1)**: the method was read in an earlier pass. I did not re-read it this session.
- **ABSTRACT**: I saw only the abstract or landing page.
- **SNIPPET**: I saw only a search-engine or secondary summary.
- **KNOWN**: a classic paper that I did not re-read this session. The definitions come from memory, so verify them before citing.

## Status labels
- **PR**: peer-reviewed
- **PP**: preprint
- **TH**: thesis
- **REP**: challenge report

---

## 1. Crop-based SAM2 with full frame (the base of our SGGL)

### 1.1 CTU bachelor thesis
- R. Rozumnyi, "Sledování objektu s pomocí SAM, Dino, Monodepth nebo CLIP reprezentace", CTU FEE, 2025. Supervisor J. Matas; referee A. Lukežič. Issued 2025-01-07, accepted 2025-06-19 (DSpace API metadata).
- https://dspace.cvut.cz/handle/10467/123552
- **Status:** TH. **Reading level:** READ (M6). I did not re-download the 21 MB PDF this session.
- **Mechanism relevant to M7 (§5.3.4):**
  - It runs three SAM2.1-L trackers in parallel: an H/4×W/4 window, an H/2×W/2 window, and the full frame.
  - Selection is causal: try the H/4 crop first, then H/2, then the full frame.
  - A crop mask is accepted only if:
    - IoU(crop, full) > γ, **and this check is skipped when the full-frame mask is empty**, and
    - bbox/window ≤ β = 0.1.
  - "Don't write empty masks to memory." On loss, the window is enlarged by N·k.
- **Relation to us:**
  - The thesis treats an empty global mask as *no evidence*, so it always trusts the crop when the global view sees nothing. That is SGGL.
  - Our SGGL-V treats an empty global view as *veto evidence*.
  - **D sits between the two**: the global view's absence is trusted only when the global view could have seen the target.
  - The thesis has no reacquisition state, no identity check, and no failure analysis of crops grabbing a distractor. The reading notes mention none.

### 1.2 AdaSAM2
- Liu et al., *Sensors* 26(18):5819, 2026. doi:10.3390/s26185819
- **Status:** PR (MDPI). **Reading level:** READ (M6; EuropePMC full text).
- **Mechanism:**
  - Built on SAMITE. A Double-DQN policy picks an image-relative crop scale; it is triggered only on "unreliable" events.
  - Lost-aware recovery progressively enlarges the crop and re-decides after 10 frames, with a constrained re-decision step.
  - It adds a candidate score: α·cos-appearance + β·exp(−dist/σ) + λ·scale-consistency.
  - How "unreliable" is detected is **not specified**.
  - It has no full-frame co-tracker and no veto.
- **M7 relevance:**
  - The appearance cosine in its candidate selection is a weak precedent for E.
  - The enlarge-on-loss state is a weak precedent for C (a loss state changes the *search*, not the *acceptance* rule).

---

## 2. SAM2 memory, reliability and distractor methods

### 2.1 DAM4SAM / SAM2.1++
- Videnovic, Lukežič, Kristan. arXiv 2411.17576. CVPR 2025: the CVF open-access supplemental exists at openaccess.thecvf.com/content/CVPR2025/...Videnovic_A_Distractor-Aware_Memory...
- Journal-style extension: arXiv 2509.13864 (PP; it says the "preliminary version" was at CVPR 2025).
- **Status:** PR (CVPR'25), extension PP. **Reading level:** READ (both).
- **Distractor detection:**
  - Distractors are found *intra-view*, from SAM2's own multimask output.
  - The method fits a bbox to the chosen mask, and another to the union of the chosen mask with the largest connected component of an alternative mask.
  - Divergence is flagged if area(bbox_out) / area(bbox_union) < θ_anc = 0.7.
  - It does not use appearance features and does not check cross-view agreement.
- **Memory gating:**
  - **RAM**: FIFO, updated every Δ = 5 frames (the most recent frame is always included). It is **not updated when the predicted mask is empty** ("target absent ⇔ mask all zeros").
  - **DRM** (init frame plus anchors) admits a frame only if all of these hold:
    - the target is present,
    - at least 5 frames have passed since the last DRM update,
    - the divergence ratio is < 0.7,
    - the predicted IoU is > 0.8,
    - the area is within 20% of the median over the last 10 frames.
  - DRM entries carry no temporal encoding.
- **Re-detection / identity:**
  - It has **no explicit re-detection or verification step**.
  - Recovery relies on memory attention, with the init frame kept in DRM.
  - The extension calls the rules "introspection-based", meaning they use only SAM2's own outputs.
- **Overlap:**
  - MA: high (absence-gated writes, quality-gated admission).
  - E: none.
  - C: none (no state machine).
  - D: none.

### 2.2 SAMURAI
- Yang et al., arXiv 2411.11922. A search snippet gives IEEE TIP 2026 (doi:10.1109/TIP.2026.3651835). Venue: SNIPPET only.
- **Status:** PR (per snippet). **Reading level:** READ.
- **Mechanism:**
  - A Kalman filter on (x, y, w, h) plus velocities.
  - Mask selection: argmax[α_kf · IoU(KF box, mask) + (1−α_kf) · predicted IoU]. α = 0.15 in Table 4; the text says 0.2.
  - Motion is used only if the target was updated within the last τ_kf frames.
  - Memory selection: walk backward from the current frame and keep frames whose s_mask, s_obj and s_kf all exceed their thresholds. **The threshold values are not given** ("e.g.").
- **Re-detection / identity:** none. It is a full-frame tracker; cropping is not stated.
- **Overlap:**
  - MA: high (multi-score admission).
  - C: weak (the τ_kf "recently updated" condition is a crude loss state).

### 2.3 SAM2Long
- Ding et al., arXiv 2410.16268. ICCV 2025 per the arXiv comment noted in M6; the HTML page itself states no venue.
- **Status:** PR (secondary). **Reading level:** READ.
- **Mechanism:**
  - P pathways × 3 multimask candidates. The cumulative score is Σ log(predicted IoU + ε); the top-P pathways are kept.
  - Memory selection: walk backward and keep frames with IoU > δ_IoU = 0.3 **and** object score o > 0.
  - Memory keys are weighted 0.95 to 1.05 by sorted occlusion score.
  - Under uncertainty (max |o| < δ_conf = 2), it enforces distinct-IoU candidates to keep diversity.
- **Re-detection / identity:** none explicit. Recovery is implicit through the tree search.
- **Overlap:**
  - MA: high (o > 0 and IoU gate).
  - C: weak. The tree search is a "delayed decision" alternative to a reacquisition state, but SAM2Long has no verification.
- Note: δ_conf = 2 on |object score| is an existing precedent for treating a small-magnitude object score as uncertain.

### 2.4 SENTRY
- Alansari et al. (Khalifa Univ. / UWA), "SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking". arXiv 2606.24449.
- M6 recorded "Accepted to ECCV 2026" from the arXiv comment. The v2 HTML I read this session states no venue. **Venue: UNVERIFIED.**
- **Status:** PP (or PR). **Reading level:** READ (method; the appendix with α, β, θ_rel, θ_min values was not reached).
- **Mechanism:**
  - **Candidate generation:**
    - Decoder hypotheses with predicted IoU > α·max and o > 0.
    - Plus AMG proposals scored by cosine similarity of region features to a **first-frame template**, kept if s ≥ β·max.
    - Plus a Kalman (constant-acceleration) fallback mask.
    - Candidates are merged by Soft-NMS.
  - **Validation:** each candidate is re-segmented **backward over τ = 10 frames** to form a tracklet. Similarity to the rolling trajectory of previously selected masks is mean box IoU (geometry only, explicitly "instead of appearance distance").
  - **Neighbours:** non-selected candidates become neighbour tracklets. Hungarian matching acts as a consistency filter.
  - **Decision:** select if max s ≥ θ_rel; otherwise take the top candidate if ≥ θ_min; otherwise use the Kalman mask.
  - The selected mask is written on SAM2's default schedule ("refine-before-write").
- **Disappearance / identity:**
  - No explicit absence or re-identification module.
  - Their Fig. 6 shows every method, SENTRY included, failing after about 100 frames out of view.
- **Overlap:**
  - MA: moderate to high. Writes are validated by candidate consistency, but it is per-frame selection, not post-loss probation.
  - E: moderate. It uses a first-frame template, but only for proposal scoring, not as an acceptance test.
  - C, D: none. It has no crop.

### 2.5 StreamDAM (presence head)
- Xiang Chen, arXiv 2608.03912.
- **Status:** PP. **Reading level:** READ (also M6).
- **Mechanism:**
  - A learned GRU presence head over runtime scalars ("logits, mask-overlap statistics, area ratios, absence run-length"), trained with class-balanced BCE on GT presence.
  - The head drives memory read eligibility (τ = 0.34), the dense/sparse read stride (hysteresis p_hi/p_lo/R), and output suppression.
  - **After K = 5 suppressed frames the admission threshold is *relaxed*** to break an "absent latch".
  - No identity check, consecutive-frame confirmation or verification of reacquisition is described.
  - It does not analyse false reacquisition of distractors or small targets misjudged as absent.
- **Results:**
  - The learned head beats the raw-logit threshold: false-absent 0.067 vs 0.099 at matched hallucination; false-present 0.080 vs 0.234 at Youden.
  - It argues that fixed policies fail because "true absence and hard visibility need opposite controls".
- **Overlap:**
  - C: moderate, in the **opposite direction**. StreamDAM loosens acceptance after a loss; C tightens it. That is a useful contrast to cite.
  - D: moderate in spirit (presence ≠ the raw object score, presence learned from runtime scalars). It is single-view and trained, with no scale conditioning.

### 2.6 SAM 3 presence head
- arXiv 2511.16719. **Status:** PP (Meta). **Reading level:** SNIPPET (paper appendix and secondary explainers).
- **Mechanism:** a global presence token predicts whether the concept is present at all. The final object score = presence × per-query localisation score. Rationale: recognition needs global context, while localisation needs local focus.
- **Overlap with D:** conceptual only. SAM 3 argues *for* global evidence for presence. D argues that global evidence is untrustworthy at low pixel size. Image-concept setting, trained.

### 2.7 SatSAM2
- arXiv 2511.18264 (v3, Apr 2026). **Status:** PP. **Reading level:** READ.
- **Mechanism:**
  - A **five-state machine**: Uninitialized → Initialized → Stabilizing → Stable ↔ ResetStable.
  - **Stabilizing:** requires SAM2 affinity > τ_h = 0.3 for T_f = 12 consecutive frames (the text says 2 to 10, which is inconsistent) before reaching Stable.
  - **Stable**, by SAM2 score:
    - high: accept and update the KF;
    - mid: accept without a KF update;
    - low but with KF consistency: take the candidate with max IoU against the KF prediction;
    - otherwise: go to ResetStable.
  - **ResetStable:** output the KF prediction only, with no updates, until a "high-confidence" mask returns. Then go back through Stabilizing.
  - Memory: s_mask > 0.5, s_obj > 0, s_kf > τ_kf; N_max = 16.
  - Reacquisition is gated only by motion: area ratio in [0.5, 2] and centre distance < √S. There is no appearance verification.
- **Overlap with C:** **HIGH.** It is a SAM2-based, training-free state machine in which re-entry after loss must pass a K-consecutive-frame stabilisation. This is the closest SAM2 precedent for C.
  - **Differences:** single view; motion-only gate; no cross-view agreement; satellite domain; no crop.

### 2.8 SAMOSA
- Zhu, Wang, Liu, Tang, Yu, Lu, Zhou (Tsinghua). "Segment Anything with Motion, Geometry, and Semantic Adaptation for Complex Nonlinear VOT". arXiv 2605.22538.
- **Status:** PP. **Reading level:** READ.
- **Error detection (EDRM):**
  - Three similarities against a prototype:
    - aspect ratio,
    - area,
    - **semantic cosine of SAM2 Hiera encoder embeddings, mask-gated average pooled.**
  - The prototype is the average over the latest T frames. It is **frozen when an error is flagged**.
  - Defaults: σ_s = 0.10, τ_ar = τ_a = 0.40, τ_s = 0.60.
- **Recovery:** check all 3 decoder candidates against the frozen prototype. If one passes all thresholds, accept it, override the motion predictor, and resume prototype updates.
- **Memory (TAMB):** walk backward and keep frames with IoU ≥ 0.5, obj ≥ 0.5, motion ≥ 0 (M = 30). Rank them by a weighted sum and keep the top N_m − 1 = 5.
- **Training:** only the LSTM motion predictor is trained.
- **Distractor-relative margin:** **none.** Every check is an absolute threshold.
- **Overlap with E:** **HIGH for the feature choice.** SAMOSA uses exactly mask-pooled frozen SAM2 encoder features and cosine to a target prototype, with the prototype frozen on suspected error.
  - **Differences:** its reference is recent-average, not frame-0; it has no distractor margin; it only re-ranks SAM2's own 3 candidates; it has no crop and no global view.
- **Overlap with C:** moderate (a detection mode vs a recovery mode).

### 2.9 SAMITE
- arXiv 2507.21732. **Status:** PP. **Reading level:** READ (M6).
- Prototype memory: FG-prototype cosine against a **frame-1 feature anchor** and a t−1 position anchor (α = 0.3). It keeps the top 5 plus the anchors. The prompt is gated on cycle-consistency mIoU > 0.7.
- **Overlap:** E moderate (frame-1 anchor similarity), MA moderate.

### 2.10 Other SAM2-family items (brief)
- **Lean-SAM2** (arXiv 2607.19811; PP; READ in M6): efficiency; visibility-gated memory condensation; window routing with full-frame fallback when the occlusion score is < 0.99. **Low overlap.**
- **MoSAM** (arXiv 2505.00739; PP; READ in M6.1): memory gated on IoU > 0.7 and occlusion ≥ 0. Overlap: MA.
- **SAM2Plus** (*Sensors* 2025; READ in M6): KF-IoU fusion plus memory selection. Overlap: MA, low.
- **Seg2Track-SAM2** (arXiv 2509.11772v2; PP; READ):
  - MOTS. Each mask is labelled High / Uncertain / Low with τ_h = 0.70 and τ_l = 0.10.
  - **Only High masks update SAM2 memory; Uncertain masks do not.** A track is removed after 5 consecutive Low frames.
  - No ReID. Overlap: MA moderate.
- **FreeVPS** (arXiv 2508.19705; PP; READ):
  - Polyp VPS with an external detector plus SAM2.
  - IAF: a detection survives only if matched (IoU > 0.5) across all frames of a T = 3 window.
  - IAR: a new instance must appear in λ1 consecutive windows (default 1); a lost one is deleted after λ2 = 3 windows. Only the refined reference-frame mask is written to memory.
  - **Overlap: C** (K-consecutive confirmation before admitting a new object into SAM2 memory) **and MA.**
- **"Rethinking Memory Design in SAM-Based VOT"** (arXiv 2512.22624; PP; ABSTRACT): reimplements SAM2-tracker memory policies in SAM3; hybrid short-term plus distractor-resolving memory. Says nothing on reacquisition verification in the abstract.
- **Re-Prompting SAM 3 via Object Retrieval** (arXiv 2603.23788; REP; ABSTRACT, M6): SAM3 detector candidates plus DINOv3 matching against a target feature pool, then re-prompting. **E/F high in spirit** (global search plus appearance revalidation), but it uses an external detector and DINOv3.

---

## 3. Long-term tracking: local tracker + global re-detection + verification

### 3.1 LTMU
- Dai, Zhang, Wang, Li, Lu, Yang. "High-Performance Long-Term Tracking with Meta-Updater". arXiv 2004.00305. CVPR 2020 (KNOWN; the arXiv page does not state it).
- **Status:** PR. **Reading level:** READ.
- **Mechanism:**
  - Components:
    - local tracker: ATOM* plus SiamMask;
    - **verifier:** RTMDNet, threshold 0;
    - global re-detection: when the verifier score is below threshold, Faster R-CNN proposes all candidates in the next frame, SiamRPN regresses around each, and the verifier picks the target, then the local tracker is reset.
  - The **meta-updater** decides "update or not" each frame. It is a cascaded LSTM (t_s = 20) over:
    - geometric cues (box),
    - discriminative cues (response map via CNN, max score),
    - appearance cues: **distance between embeddings of the current result and the first-frame template** (ResNet-50, triplet loss).
  - It is trained on LaSOT-train rollouts (positive IoU > 0.5, negative IoU = 0).
- **Overlap:**
  - **MA: very high** ("should this frame update the model?" with first-frame appearance as one cue). It is learned; ours is rule-based.
  - E: high (first-frame embedding distance).
  - F: high (local + verifier + global re-detector).

### 3.2 SPLT ("Skimming-Perusal" tracking)
- Yan et al., arXiv 1909.01840, ICCV 2019 (KNOWN).
- **Status:** PR. **Reading level:** READ.
- **Mechanism:**
  - Local search over a 4× target-size region. SiamRPN generates candidates; the SiamRPN scores are distrusted.
  - A **verifier** computes c = max(cos(f(Z₀), f(b)), 0) against a **fixed first-frame template** (triplet embedding, margin 0.2).
  - If c* > θ = 0.65 the target is present and the next frame stays local; otherwise the next frame does a **global search**. A "skimming" net scores sliding windows for presence and the top K = 3 are perused.
  - No template update is described.
- **Overlap:**
  - E: high (first-frame cosine verifier as the present/absent decision).
  - C: moderate (the loss state switches the search regime).
  - It has the same 4× local region as ours, but SPLT's local region is a search window, not a separate memory state.

### 3.3 GlobalTrack
- Huang, Zhao, Huang. arXiv 1912.08531, AAAI 2020 (KNOWN).
- **Status:** PR. **Reading level:** READ.
- **Mechanism:**
  - Full-image, multi-scale, query-guided RPN + RCNN with a **fixed first-frame query**.
  - It makes no temporal assumption and does no update; the top-1 output is the result.
  - Absence (OxUvA only): threshold the top-1 score at τ = 0.84.
- **Overlap:** D, as a contrast. It is a pure global view with no local branch; absence is a single score threshold, not conditioned on scale.

### 3.4 Siam R-CNN
- Voigtlaender, Luiten, Torr, Leibe. arXiv 1911.12836, CVPR 2020 (KNOWN).
- **Status:** PR. **Reading level:** READ.
- **Mechanism:**
  - Whole-image re-detection: RPN proposals plus the previous frame's detections, each scored against the first-frame template.
  - **TDPA:** tracklets are kept for *all* candidates, so distractors are tracked rather than discarded.
  - Track score = Σ unary + w_loc·loc_score, where:
    - unary = w_ff·ff_score + (1−w_ff)·ff_tracklet_score, i.e. similarity to the first-frame GT **and** to the latest box of the first-frame tracklet: a frozen reference plus a "recent but trusted" reference;
    - loc_score = −L1 jump between tracklets.
  - Gaps of up to 1500 frames are allowed. The output is "absent" when the selected tracklet has no current detection.
  - Hard-example mining on distractors from other videos.
- **Overlap:**
  - **E: high.** Dual reference (frame-0 plus a trusted tracklet reference) for identity is exactly E's "frozen frame-0 + conservatively updated recent reference".
  - C: moderate (delayed tracklet-level decision instead of per-frame acceptance).

### 3.5 KeepTrack
- Mayer, Danelljan, Paudel, Van Gool. arXiv 2103.16556, ICCV 2021.
- **Status:** PR. **Reading level:** READ.
- **Mechanism:**
  - Candidates are local maxima of the score map (5×5 max-pool, τ = 0.05), encoded as appearance + MLP(score, position).
  - A SuperGlue-style GNN with Sinkhorn (10 iterations) and dustbins associates candidates across frames. A new object is created if the assignment probability is < ω = 0.75.
  - **Target selection:**
    - Keep the target object ID while it is visible. Switch only if another object's score exceeds the target's *historical maximum*.
    - **After loss, the best candidate becomes the target only if its score > η = 0.25.** Otherwise a reappearing candidate stays a "distractor" until its score rises enough.
    - The search area is enlarged after occlusion.
  - **Memory:**
    - sample weight √σ if the selected object is the original target, σ otherwise;
    - samples with confidence < 0.5 are ignored (LaSOT AUC 64.6 → 65.0);
    - lowest age × confidence is evicted.
- **Overlap:**
  - **C: high.** A stricter rule for a candidate that appears after the target is lost, plus explicit distractor identities.
  - MA: moderate (confidence-weighted samples).
  - E: the distractor-aware part; the identity comes from association, not from a margin.

### 3.6 MBMD, LT_DSE, OxUvA
- **MBMD** ("Learning regression and verification networks for long-term visual tracking", arXiv 1809.04320; VOT2018-LT winner).
  - **Reading level:** SNIPPET.
  - Regression net plus a verification net (MDNet-style); global search when the score is low.
- **LT_DSE** (VOT2019-LT winner).
  - **Reading level:** SNIPPET via arXiv 2008.04722.
  - ATOM localisation, SiamMask refinement, an RT-MDNet verifier, and global re-detection.
  - F 0.695, Pr 0.715, Re 0.677.
- **OxUvA** (Valmadre et al., ECCV 2018, arXiv 1803.09502).
  - **Reading level:** ABSTRACT.
  - The benchmark requires present/absent predictions and evaluates both location and presence.
- **Pattern:** "local tracker → verifier → global re-detector" with a **first-frame verifier** is the standard LT recipe since 2018.
  - Our L/G pair is a variant of it, with *G as a second tracker rather than a detector*.
  - "Now you see me" (Lukežič et al., arXiv 1804.07056; SNIPPET) formalises LT0/LT1 re-detection capability.

### 3.7 VOTS challenge reports
- **VOTS2024** (Kristan et al.; REP; READ, main text plus selected appendix entries via a local text extraction of data.vicos.si/publications/kristan2024the.pdf).
  - 67% of entries used SAM (v1 or HQ-SAM). I found no SAM2 entries in the text. "All submissions applied the same network for frame-to-frame localization and target re-detection."
  - **dmaot_sam** (2nd): the box from DMAOT prompts SAM; it selects the SAM mask with the largest IoU against the DMAOT mask (cross-model agreement selection).
  - **LY-SAM:** chooses between the DMAOT and HQ-SAM masks by **cosine similarity to the initial template mask**, and replaces the template with an "optimal template" in long sequences. **E precedent** (template similarity used to arbitrate between two candidate masks from two models).
  - **RMemAOT** (VOTSt winner): "independently run 3 trackers initialized from the first three frames and fuse their predictions to filter out unstable or anomalous tracking outcomes". This gave the best drift and absence prediction among the top trackers. **Precedent for cross-tracker consensus as an absence or anomaly filter (≈ SGGL-V veto).**
  - S3_Track (winner): 63% of its failures were false "absent" reports and 36% were drift. Absence prediction is a dominant failure mode.
- **VOTS2025** (ICCVW 2025, openaccess.thecvf.com ...Kristan_The_Third_VOTS2025...).
  - **Reading level:** SNIPPET only. I did not read the per-tracker descriptions.
  - It is the most likely place for an unknown SAM2 crop + full-frame + verification entry. **Gap.**
- **VOTS2023 / HQTrack:** READ in M6.1 (box → HQ-SAM on the full frame, IoU > 0.1 gate).

---

## 4. Identity verification and distractor-relative margins (classic)

- **TLD** (Kalal, Mikolajczyk, Matas, TPAMI 34(7), 2012). **Reading level:** KNOWN.
  - Its nearest-neighbour object model defines:
    - **relative similarity** S_r = S⁺/(S⁺+S⁻), i.e. similarity to target patches *relative to* background/distractor patches;
    - **conservative similarity** S_c, which uses only the earliest half of the positive patches.
  - The detector re-initialises the tracker; P-N learning constrains updates.
  - This is the direct ancestor of E's "frozen early reference + target-vs-distractor relative score" *and* of the L/G "tracker + detector cross-check".
  - Verify the definitions in Kalal's 2011 thesis before citing.
- **Forward-backward error** (Kalal et al., ICPR 2010). **Reading level:** SNIPPET (confirmed abstract).
  - Self-consistency of a tracker under time reversal is used as a failure detector. It is the classic "re-run under a transformation and compare" reliability signal, and the conceptual ancestor of G-probe.
- **Peak-to-sidelobe ratio** (MOSSE, Bolme et al., CVPR 2010) and **APCE** (LMCF, Wang, Liu, Huang, CVPR 2017). **Reading level:** KNOWN.
  - Both measure response peak vs the rest of the map. They are used to detect failure/occlusion (PSR) and to update only on high-confidence frames (APCE).
  - **E's "distractor-relative margin" (similarity to target minus the max similarity of other regions) is a PSR/APCE-style peak-vs-rest statistic.** It is also a Lowe ratio test (IJCV 2004) in matching.
- **Track confirmation** (SORT/DeepSORT, Wojke et al. 2017). **Reading level:** KNOWN.
  - New tracks are "tentative" until n_init = 3 consecutive matched detections; then "confirmed"; then deleted after max_age.
  - **This is the standard K-consecutive confirmation used by C.**
- **XMem** (ECCV 2022), **Cutie** (CVPR 2024). **Reading level:** KNOWN.
  - Object-level / long-term memory for VOS, with no explicit post-loss identity test.
- **QDMN** (Liu et al., ECCV 2022, arXiv 2207.07922). **Reading level:** ABSTRACT.
  - A quality-assessment module scores each frame's segmentation; only good frames enter memory. **MA precedent in VOS** (learned).

---

## 5. Perturbation / resolution sensitivity as a reliability signal (for G-probe)

- I found **no tracking or VOS paper that re-runs the tracker on a blurred or down-up-sampled copy of the search region from a cloned state** and uses the change as a reliability or identity signal. The searches were broad but not exhaustive.
- Related general principles:
  - **TTA uncertainty** (e.g. Wang et al., Neurocomputing 2019, medical segmentation; KNOWN): the variance across augmented inputs serves as uncertainty.
  - **ODIN** (Liang et al., ICLR 2018; KNOWN): input perturbation separates in- vs out-of-distribution confidence.
  - **Augmentation-sensitivity risk scoring** (arXiv 2510.01683; SNIPPET): high sensitivity to plausible perturbations flags unreliable "confident" predictions.
  - **Detection consistency** (arXiv 2207.13890; SNIPPET): detectors are inconsistent under small input changes.
  - **FB error** (above).
- Related negative evidence:
  - StreamDAM's resolution-scaling negative result: at reduced resolution, box initialisation picked a distractor.
  - The VOTS2024 and anti-UAV literature: few-pixel targets give "weak texture … appearance-based recovery may … incorrectly reactivate disappeared trajectories" (Remote Sensing 18(13):2084; SNIPPET). **This bears on E: appearance verification is weakest exactly in the small-target regime where D matters.**

## 6. Scale-aware trust in absence evidence (for D)

- **Küppers et al.**, "Multivariate Confidence Calibration for Object Detection" (arXiv 2004.13546; CVPRW 2020 (KNOWN); READ).
  - Detector miscalibration depends on box position **and width/height**. Calibration conditioned on (confidence, size, position) beats confidence-only calibration.
  - This is the closest published support for "confidence or absence reliability depends on target scale", but it addresses positives, not absence, for detectors, not trackers.
- **SAHI / slicing inference:** zooming raises small-object recall a lot, which implies full-frame misses are size-driven. It is not used to judge absence evidence.
- **CTU thesis:** skips the agreement check whenever the full-frame mask is empty, an implicit "global absence is uninformative" assumption.
- **SAM 3:** the opposite assumption (global presence is the right place to decide).
- I found **no tracking or VOS paper that decides whether to trust a full-frame model's absence evidence as a function of the target's effective pixel size in that model's input, or of the two views' recent agreement history.** Searches: SAM2 small objects / object score, tiny-object tracking absence, detectability. This is the least covered sliver.

---

## 7. Per-candidate assessment

### C: state-aware reacquisition gate (STABLE / SUSPECTED_LOSS / REACQ_CANDIDATE)
1. **Problem prior work solves:** preventing a tracker from snapping to a distractor right after loss (KeepTrack, SatSAM2, SPLT, LTMU), and confirming new tracks (DeepSORT tentative tracks; FreeVPS λ1).
2. **Component changed:** output selection / track-state logic, not the model.
3. **Already known?** Yes, in substance.
   - SatSAM2: a SAM2 state machine with K = 12 consecutive confirmations to re-enter Stable.
   - KeepTrack: a stricter threshold (η = 0.25 plus beating the historical max) for a candidate after loss.
   - DeepSORT: n_init = 3.
   - StreamDAM: hysteresis states, though it *relaxes* after loss.
4. **Material difference:**
   - The confirmation source is a **second, concurrently running SAM2 state on a different view** (global agreement), rather than temporal persistence or motion.
   - It applies only to the *crop* state's reacquisitions, not to the whole tracker.
5. **Experimentally relevant?** Yes, but narrowly.
   - It decides whether the catastrophic false-reacquisition case is fixed without SGGL-V's tiny-target recall loss.
   - K-consecutive alone will not stop a *persistent* confident distractor (score up to 7.8 over many frames). Global agreement is what does that, which is the existing veto, so C ≈ "SGGL-V only in REACQ state".
6. **Engineering vs research:** engineering integration. The defensible part is only the empirical finding of *where* the veto is needed (post-loss vs steady state).
- **Novelty risk: HIGH.**

### D: global observability-aware veto
1. **Problem prior work solves:** calibrating detector confidence by box size (Küppers); learned presence from runtime scalars (StreamDAM); cross-model consensus as an anomaly filter (RMemAOT, HQTrack, dmaot_sam).
2. **Component changed:** the decision rule combining two views' presence evidence.
3. **Already known?** Partially.
   - The ingredients are known: size-conditioned reliability, agreement history, consensus vetoes.
   - CTU always ignores global absence; SGGL-V always trusts it.
   - I did not find a rule that trusts global absence *conditioned on effective target size in the global input and on recent cross-view agreement*.
4. **Material difference:** yes. It is a cross-view, scale-conditioned trust of *absence* (negative evidence), training-free, inside SAM2.
5. **Experimentally relevant?** Yes. This is exactly the knob that trades the catastrophic case against tiny-target recall.
   - A measurement such as "P(G false-absent | effective size, agreement history)" on DAVIS/MOSE/PUMaVOS would be new evidence even if the rule itself is simple.
6. **Engineering vs research:**
   - The rule alone is a heuristic: engineering.
   - The **reliability analysis** (G's absence reliability vs pixel size, with a reliability curve and AUROC as a veto signal) is a defensible small research contribution.
- **Novelty risk: MEDIUM** (low-medium if framed as the analysis; medium-high if framed as a "new veto").

### E: original-target identity verification (mask-pooled frozen SAM2 features, frame-0 + conservative recent reference, distractor margin, colour baseline)
1. **Problem prior work solves:** verifying that a candidate is the original target against a fixed or slowly updated template, and rejecting distractors.
2. **Component changed:** the acceptance test for a candidate mask.
3. **Already known?** Yes, closely.
   - **SAMOSA**: mask-pooled SAM2 Hiera features, cosine to a prototype, prototype frozen on error.
   - **SAMITE**: frame-1 anchor.
   - **SPLT / LTMU / GlobalTrack**: first-frame verifier.
   - **Siam R-CNN**: frame-0 plus trusted-tracklet dual reference.
   - **TLD**: conservative similarity plus relative (target vs background) similarity.
   - **PSR/APCE/Lowe ratio**: peak-vs-rest margin.
   - **LY-SAM (VOTS2024)**: template cosine to arbitrate between two models' masks.
4. **Material difference:** small.
   - The distractor margin is computed over *other regions of the full frame* using SAM2 encoder features, and is applied to crop-state reacquisitions.
   - SAMOSA has no margin and no frame-0 reference; TLD/PSR have both ideas, but with older features.
5. **Experimentally relevant?** Uncertain.
   - The literature warns that appearance is weak for few-pixel targets. The distractor in our failure case was a *similar* object.
   - E's value must be shown on that case **and** must not reject true tiny reacquisitions. The colour-histogram baseline is essential.
6. **Engineering vs research:** engineering (a re-application of known verification to a new pipeline).
- **Novelty risk: HIGH.**

### G-probe: resolution-sensitivity probe
1. **Problem prior work solves:**
   - TTA/ODIN: uncertainty from perturbation.
   - FB error: failure detection by consistency.
   - Not specifically "would a lower-resolution view see this?".
2. **Component changed:** adds a diagnostic forward pass, from a cloned pre-frame state, on a degraded crop.
3. **Already known?** The principle (perturbation-consistency reliability) is well known. **I found no tracking/VOS use of blur or down-up resolution sensitivity from a cloned memory state.**
4. **Material difference:** yes. It simulates the global view's effective resolution to estimate *observability*, which links it to D.
5. **Experimentally relevant?** Questionable.
   - It measures **observability, not identity**. A small distractor and a small true target are both resolution-sensitive, so G-probe cannot separate the catastrophic case (a confident, similar distractor) from a correct tiny reacquisition.
   - It mainly offers a data-driven replacement for D's analytic size threshold.
   - It costs about one extra local predictor pass per gated frame.
6. **Engineering vs research:** research-flavoured (a novel diagnostic), but a weak hypothesis for the target failure.
- **Novelty risk: LOW-MEDIUM. Utility risk: HIGH.**

### MA: memory admission (suppress or delay writing unverified reacquisitions into L)
1. **Problem prior work solves:** keeping bad frames out of memory.
   - DAM4SAM: no write when empty; DRM IoU > 0.8 plus area gates.
   - SAMURAI and SAM2Long: IoU and o > 0 thresholds.
   - SAMOSA, SatSAM2, MoSAM: threshold-gated memory.
   - Seg2Track: Uncertain frames are not written.
   - SENTRY: refine-before-write.
   - StreamDAM: admission τ.
   - QDMN: learned quality gate.
   - LTMU: learned "update or not".
   - KeepTrack: confidence < 0.5 ignored.
   - CTU: no empty masks in memory.
2. **Component changed:** the memory-encoder write path.
3. **Already known?** **Yes, very.** Probation-style delayed admission is implied by FreeVPS (confirm, then write) and LTMU.
4. **Material difference:** only the trigger (post-loss reacquisition in the *crop* state) and the fact that L's memory is in crop coordinates.
5. **Experimentally relevant?** Possibly. It prevents a false reacquisition from self-reinforcing (score up to 7.8 suggests a memory-reinforced distractor).
   - Expect interaction with C/E. It must be ablated jointly.
   - Note the SAM2 behaviour: frames are written with a no-object embedding even when absent. DAM4SAM showed that skipping those writes matters (bus example).
6. **Engineering vs research:** engineering.
- **Novelty risk: HIGH.**

### F: D + E (observability-gated trust of global absence, plus identity evidence when G is uninformative)
1. **Problem prior work solves:** LTMU, SPLT, LT_DSE and TLD combine local tracking, verification and global re-detection. Re-Prompting SAM3 combines global retrieval with appearance revalidation.
2. **Component changed:** the decision policy across two views plus a verifier.
3. **Already known?** The architecture pattern is known. **The specific logic was not found**: "accept the local view against global absence only when (a) the global view is unlikely to observe the target at its pixel size *and* (b) identity evidence supports the original target".
4. **Material difference:** moderate. The verifier is consulted *only in the regime where the cross-view veto is uninformative*, rather than always.
5. **Experimentally relevant?** Yes, if both arms are needed. It directly targets the two observed failure modes (the catastrophic distractor when G is reliable; tiny-target recall when G is blind).
   - Risk: E is weakest exactly in D's "G is blind" regime (small targets).
6. **Engineering vs research:**
   - The combination is engineering.
   - The defensible contribution is the **failure-mode taxonomy plus the evidence** that absence reliability is scale-dependent and that identity evidence is (or is not) informative there.
- **Novelty risk: MEDIUM.**

---

## 8. Not read / caveats
- **CTU thesis:** not re-read this session (relying on the M6 READ).
- **VOTS2025:** per-tracker descriptions were not read (SNIPPET). This is the most likely place for an unknown direct precedent (SAM2 crop + full frame + verification).
- **Read at SNIPPET or ABSTRACT level only:** MBMD, LT_DSE, OxUvA metrics, QDMN, SAM 3 presence details, Rethinking-Memory (2512.22624).
- **Read from memory, not re-read (KNOWN):** TLD, PSR/APCE, DeepSORT, XMem/Cutie, ODIN and TTA uncertainty. Verify definitions before citing.
- **Venues:**
  - SENTRY: UNVERIFIED (M6 saw "ECCV 2026" in the arXiv comment; v2 HTML shows none).
  - SAMURAI: TIP 2026 (snippet only).
  - SAM2Long: ICCV 2025 (secondary).
- **Missing values:** SENTRY's α/β/θ values are in its appendix, which I did not read. SAMURAI's τ values are not reported.
- **Method:** every READ went through an LLM page summariser. Numbers must be re-checked against the PDFs.

## 9. Summary table

| Candidate | Closest prior art | Already known? | Material difference | Exp. relevant | Nature | Novelty risk |
|---|---|---|---|---|---|---|
| C: reacq. state gate | SatSAM2 (K = 12 re-stabilise), KeepTrack (η = 0.25 + beat hist. max after loss), DeepSORT n_init, FreeVPS λ1; StreamDAM (opposite: relax) | Yes | Confirmation by a concurrent second SAM2 view; crop-state only | Narrow (≈ SGGL-V restricted to the REACQ state) | Engineering | **High** |
| D: observability-aware veto | Küppers (size-conditioned calibration), CTU (ignores G-absence), RMemAOT/HQTrack consensus, StreamDAM presence | Partially | Trust of *absence* conditioned on pixel size in G's input + agreement history | Yes (the key trade-off knob) | Rule = engineering; reliability-vs-scale analysis = research | **Medium** |
| E: identity verification | SAMOSA (mask-pooled Hiera cosine, frozen prototype), SAMITE, SPLT/LTMU/GlobalTrack (frame-0 verifier), Siam R-CNN (dual ref), TLD (conservative + relative sim.), PSR/APCE, LY-SAM | Yes | Margin over full-frame regions with SAM2 features; crop reacq. only | Uncertain (appearance weak for tiny targets) | Engineering | **High** |
| G-probe | TTA uncertainty, ODIN, FB error (principle only) | Principle known; tracking use not found | Simulates G's resolution from a cloned state | Questionable (measures observability, not identity) | Novel diagnostic, weak hypothesis | **Low-Med** (utility risk high) |
| MA: memory admission | DAM4SAM, SAMURAI, SAM2Long, Seg2Track, SENTRY, QDMN, LTMU, KeepTrack, CTU | Yes, very | Trigger = post-loss crop reacq.; crop-coordinate memory | Possibly (stops self-reinforcement) | Engineering | **High** |
| F: D + E | LTMU / SPLT / LT_DSE / TLD pattern; Re-Prompting SAM3 | Pattern known; exact logic not found | Verifier consulted only where G's absence is uninformative | Yes, if both arms matter | Combination = engineering; failure-mode + scale-reliability evidence = research | **Medium** |

**Bottom line:**
- All candidate *mechanisms* have close precedents.
- The defensible research content is empirical. Show, on VOS with Hiera-T, that (i) full-frame absence evidence is scale-dependent and quantify when it is trustworthy, and (ii) which evidence (agreement history, identity, persistence) separates true tiny reacquisitions from confident distractor reacquisitions.
- Cite CTU, SatSAM2, KeepTrack, SAMOSA, LTMU, DAM4SAM and TLD explicitly.
