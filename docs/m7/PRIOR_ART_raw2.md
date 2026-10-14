# M7 prior-art notes, part 2 (raw): candidate families F / M / T / R / O1

Date: 2026-10-14. Web sources only. This file extends `docs/m7/PRIOR_ART_raw.md` (part 1) and `docs/m61/PRIOR_ART_AND_NOVELTY_REVIEW.md`. It does not repeat works covered there: CTU thesis, AdaSAM2, DAM4SAM, SAMURAI, SAM2Long, SENTRY, StreamDAM, SatSAM2, SAMOSA, KeepTrack, LTMU, SPLT, GlobalTrack, Siam R-CNN, HQTrack, Track Anything, TLD, VOTS reports. Those are cited by name only, and their details are in part 1.

## Labels
Same conventions as part 1.

**Reading level**
- **READ**: I read the method section this session through a fetch-and-summarise tool. The one exception is Matthews et al., which I read from extracted PDF text. Re-check every number against the PDF before citing it.
- **ABSTRACT**: I read only the abstract or landing page.
- **SNIPPET**: I saw only a search-engine or secondary summary.
- **KNOWN**: I did not re-read the paper this session and describe it from prior knowledge. Verify before citing.
- **CODE**: I checked the installed `sam2` package source in `.venv`.

**Status**
- **PR**: peer-reviewed
- **PP**: preprint
- **TH**: thesis
- **REP**: challenge or technical report

**Our system, for reference.** There are two frozen SAM 2.1-T video states:
- **G**: the full frame.
- **L**: an object-relative 4× crop that follows the previous output.

SGGL outputs L's pasted mask if L's object score is > 0, its mask is non-empty, and the mask does not touch an interior crop edge. Otherwise it outputs G's mask.

New update to part 1: the DAM4SAM journal extension (arXiv 2509.13864) appears to be published in IJCV 2026 (Springer s11263-026-02790-7; SNIPPET). Treat it as PR, with the venue not verified.

---

## F-family: fusion or correction of two predictions of the same target (different scale or view)

### F.1 Pixel-wise logit or probability fusion across scales

**Hierarchical Multi-Scale Attention (Tao, Sapra, Catanzaro, arXiv 2005.10821, 2020)**
- Status: PP. It is widely cited, and I could not confirm a peer-reviewed venue. Reading level: READ.
- Operation: `L_1.0 = U(α_0.5 · L_0.5) + (1 − U(α_0.5)) · L_1.0`.
  - L are **logits**.
  - α is a dense per-pixel attention map predicted by a head on the *lower-scale* features. Using features from both scales made no difference.
  - The fusion is chained hierarchically over the inference scales {0.5, 1, 2}, plus flip, and a new scale can be added at test time.
- Overlap: **this is a learned per-pixel gate between a low-resolution and a high-resolution prediction of the same image.** A learned F-family "logit fusion with a gate" is this paper applied to two SAM2 states.

**Attention to Scale (Chen et al., CVPR 2016)**
- Status: PR. Reading level: SNIPPET (via the Tao comparison).
- Operation: one learned dense weight map per scale, then a pixel-wise weighted sum. It is the "explicit" version of the above.

**Multi-scale and flip TTA voting**
- Reading level: SNIPPET. This is standard practice.
- Example: a VSPW challenge entry (arXiv 2406.04979) uses scales 0.9 to 1.2 plus flip, a per-pixel vote, and ties broken by the higher resolution.

**CGFSeg (1st place, MOSEv1, LSVOS 2025; arXiv 2509.25738)**
- Status: REP/PP. Reading level: READ.
- Operation:
  - It runs 5 VOS models (SAM2Long, SAM2, Cutie, LiVOS, XMem) plus flip TTA.
  - It makes a **pixel-wise** decision: foreground if the cumulative confidence across models is above a threshold. The threshold and the definition of confidence are not reported.
  - Object-ID voting resolves ID conflicts across models.
  - The paper does not say whether the result is written back to memory.
- Overlap: pixel-wise fusion of several VOS predictions **with an ID-consistency step**. It is the closest VOS-challenge precedent for "instance-consistent fusion". It differs from ours: the members are different models at the same scale, there is no crop and no asymmetry, and fusion is output-only.

**Already in part 1 or M6.1**
- CTU thesis: whole-mask IoU agreement gate between crop and full frame. This is mask-level selection, not pixel fusion.
- HQTrack: IoU > 0.1 gate.
- dmaot_sam and LY-SAM (VOTS2024): mask-level arbitration between two models.

**Verdict for F.1 (pixel-wise fusion of G and pasted L): novelty risk HIGH.**
- Learned per-pixel scale gating (Tao) and threshold voting (CGFSeg, TTA) are standard.
- The only distinction left is that the two members are *two independently-memoried SAM2 states*, not two passes of one stateless network.

### F.2 Refinement restricted to boundary, disagreement or uncertain regions

**PointRend (Kirillov et al., CVPR 2020)**
- Status: PR. Reading level: SNIPPET plus KNOWN.
- Operation: coarse-to-fine refinement. At each step it picks the most uncertain points (probability near 0.5) and re-predicts them with an MLP on fine features.

**Mask Transfiner (Ke et al., CVPR 2022, arXiv 2111.13673)**
- Status: PR. Reading level: SNIPPET plus KNOWN. The PDF was too large for the fetch tool.
- Operation:
  - "Incoherent regions" are pixels whose mask value is lost when the mask is downsampled ×2 and then upsampled back. This is exactly a *resolution-sensitivity* mask, computed on the mask itself.
  - Only those regions are refined, using a quadtree transformer on high-resolution features.
  - Note: this is also relevant to the R-family.

**MagNet / "Progressive Semantic Segmentation" (Huynh et al., CVPR 2021)**
- Status: PR. Reading level: SNIPPET.
- Operation: multi-scale patch refinement. At each scale it refines the prediction only at uncertain locations, using local patch processing combined with downsampled global context.

**BPR "Look Closer to Segment Better" (Tang et al., CVPR 2021, arXiv 2104.05239)**
- Status: PR. Reading level: ABSTRACT/SNIPPET.
- Operation:
  - Extract small patches along the predicted boundary.
  - Refine each patch at higher resolution with a network.
  - Paste the patches back. It is model-agnostic post-processing.

**CascadePSP (CVPR 2020) and SegRefiner (NeurIPS 2023)**
- Read in M6.1 (READ there). Both are model-agnostic RGB plus mask refiners: CascadePSP is global-to-local, SegRefiner uses discrete diffusion.

**AFB-URR (Liang et al., NeurIPS 2020)**
- Status: PR. Reading level: SNIPPET plus KNOWN.
- Operation:
  - It is VOS with an adaptive feature bank plus *uncertain-region refinement*.
  - Uncertainty is computed from the ratio of the top-2 class probabilities. Uncertain pixels are refined with local similarity.
  - **This is a VOS precedent for "refine where uncertain".**

**SAM2Refiner (Yao, Yang, Cui, Bo, "Towards Fine-grained Interactive Segmentation in Images and Videos", ICCV 2025, arXiv 2502.09660)**
- Status: PR. Venue per mlanthology; the arXiv page does not state it. Reading level: READ. It was READ in M6; this session I re-read the method for the T-family.
- **Localization Augment**:
  - The image is cut into **4 fixed non-overlapping quadrants**. Each quadrant is upsampled to full input resolution and passed through the **same SAM2 encoder**.
  - The local maps are reassembled into a single map L at H/8, average-pooled at 3 granularities (2, 4, 8) and tokenised.
  - Then `G_ref = CrossAttn(Q = G_tokens, K = V = L_tokens)`, and G_ref replaces G as the decoder input.
  - **No gate** and no described residual.
- **Prompt Retargeting**: rebuilds the object embedding from click maps and the previous mask.
- **Mask Refinement**: a UNet-style head that fuses the object embedding with Hiera features at 256, 128 and 64.
- SAM2 is frozen and the new modules are trained on images only (HQSeg-44K). For video, the first-frame prompt is reused and SAM2's own memory is used.

**HQ-SAM / HQ-SAM 2**
- Status: HQ-SAM is PR (NeurIPS 2023); HQ-SAM 2 is code plus README. Reading level: KNOWN, plus README in M6.1.
- Operation: a learnable HQ output token plus a fusion of early and late encoder features, with SAM frozen.

**Alpha-Refine (CVPR 2021)**
- Read in M6.1. A fixed object-relative crop of about 2× is refined by a learned refiner.

**Verdict for F.2: novelty risk HIGH.**
- "Refine only where G and L disagree" is a *disagreement-defined* uncertainty region instead of a probability-defined one (PointRend, AFB-URR) or a downsampling-defined one (Transfiner).
- That substitution is small. Transfiner's incoherence mask is itself a "would a lower resolution see this pixel differently?" test, which is the conceptual twin of G-versus-L disagreement.

### F.3 Asymmetric add/delete edits (adding foreground needs identity evidence; deleting needs stronger negative evidence)

I searched specifically for this ("asymmetric fusion add remove foreground", "positive negative mask change", "conservative removal", "union with component overlap"). I found **no VOS, tracking or segmentation-fusion paper that applies different evidence requirements to added versus removed pixels when fusing two predictions.** The closest analogues:

**MiVOS, Difference-Aware Fusion (Cheng, Tai, Tang, CVPR 2021, arXiv 2103.07941)**
- Status: PR. Reading level: READ.
- Operation:
  - It computes **D⁺ = (M_after − M_before)₊ (added region) and D⁻ = (M_before − M_after)₊ (removed region) separately**.
  - Each is transported to the target frame through the space-time-memory affinity: A± = W·D±.
  - The aligned maps are concatenated with the image, both masks and temporal-distance weights. A 5-layer residual CNN with a sigmoid outputs the fused mask.
- Overlap: **the closest precedent for treating additions and deletions as separate signals in VOS.**
- Differences:
  - The asymmetry is *learned implicitly*, not imposed as different evidence rules.
  - The two masks come from before and after a user interaction, not from two scales.
  - It is interactive VOS.

**TLD P-N learning (Kalal, Mikolajczyk, Matas, TPAMI 2012)**
- Status: PR. Reading level: KNOWN.
- Operation: two experts with different constraints.
  - The P-expert *adds* missed positives, using temporal-trajectory structure.
  - The N-expert *removes* false positives, using spatial uniqueness (only one target).
- This is asymmetric add/remove logic, but for detector training samples, not for mask pixels.

**Hysteresis thresholding / seeded region growing (Canny 1986, classic)**
- Reading level: KNOWN.
- Operation: weak responses are *added* only if they connect to strong seeds.
- This is exactly the "add only if connected to identity-confirmed evidence" pattern, at pixel level.

**STAPLE (Warfield et al., TMI 2004)**
- Status: PR. Reading level: SNIPPET.
- Operation: label fusion with **separate sensitivity and specificity per source**, estimated by EM. False-positive and false-negative evidence are weighted differently per rater, which is a statistical form of asymmetric add/delete trust.

**KeepTrack (part 1)**
- An identity *switch* requires beating the target's historical maximum score, while keeping the identity needs only to stay visible. This is asymmetric acceptance at object level.

**SAM2's own post-processing**
- Reading level: CODE. Source: `sam2/utils/misc.py: fill_holes_in_mask_scores`.
- Operation: it only *adds* small enclosed background components (area ≤ max_area, score set to 0.1).
- It is a size-gated, add-only edit with no identity evidence.

**Verdict for F.3: novelty risk MEDIUM.**
- The specific rule "a pixel added from L must belong to a component with identity evidence; a pixel deleted relative to G or L needs stronger negative evidence" was not found as a published fusion rule for VOS or cross-scale SAM2.
- The idea is a direct analogue of hysteresis or seeded growth (adding) and of TLD's asymmetric P/N experts. MiVOS already separates D⁺ and D⁻ in VOS.
- A reviewer would likely call it "hysteresis-style asymmetric fusion". It is defensible only with an ablation showing that the asymmetry, not just the gating, matters.

### F.4 Instance-consistent fusion (component-level identity check before merging)
- **DAM4SAM** (part 1): uses the largest connected component of an alternative mask in its divergence test. This is component-level reasoning inside one view.
- **CGFSeg**: object-ID voting across models before pixel fusion.
- **CTU / HQTrack**: whole-mask IoU agreement gate.
- **Panoptic merging heuristics** (Panoptic FPN etc.; KNOWN): instance masks are pasted in score order and a segment is dropped if its visible fraction is below a threshold. These are component-level overlap checks before merging.
- **SENTRY** (part 1): Hungarian matching of candidate tracklets.

**Verdict for F.4: novelty risk MEDIUM-HIGH.**
- Connected-component identity checks (overlap with G's component, or with the previous target component) are standard post-processing.
- Cross-view per-component checks are only a small step beyond CTU's whole-mask IoU and DAM4SAM's component test.

---

## M-family: memory

### M.1 Counterfactual / alternative-memory re-inference ("re-run the frame with a protected frame-0-only memory and compare")

**Matthews, Ishikawa, Baker, "The Template Update Problem" (BMVC 2003; TPAMI 26(6), 2004)**
- Status: PR. Reading level: **READ** (BMVC PDF text extracted locally).
- Operation, "Strategy 3: Template Update with Drift Correction":
  1. Track frame n with the *updated* template T_n, which gives parameters p_n.
  2. **Re-track the same frame with the first template T₁, initialised at p_n**, which gives p*_n.
  3. "If ‖p*_n − p_n‖ ≤ ε then T_{n+1} = I_n(W(x; p*_n)), else T_{n+1} = T_n." In the paper's words: "if it does [diverge], there must be a problem and so we act conservatively by not updating the template."
  - Second interpretation in the paper: track with the frame-1 template, but initialise from the updated-template estimate.
- Overlap: **this is structurally the M-family idea.**
  - The same frame is run under the normal (updated) model and under a protected first-frame-only model.
  - Their disagreement is a drift test.
  - On disagreement, the model takes the conservative action (no update).
- Differences:
  - It is Lucas–Kanade template alignment, not a deep memory.
  - Its output is the first-template refinement, not a binary verification of a re-acquisition.
  - It has no crop or global view.

**STARK (Yan et al., ICCV 2021) and MixFormer (Cui et al., CVPR 2022)**
- Status: PR. Reading level: SNIPPET.
- Operation:
  - Both feed a **static initial template plus dynamic template(s)** into one transformer.
  - STARK: an MLP score head with a threshold τ decides whether the dynamic template is updated.
  - MixFormer's Score Prediction Module: **a learnable score token attends to the search-region tokens, then to the *initial-target* tokens**, then MLP plus sigmoid. Templates scoring < 0.5 are rejected.
- Overlap: comparison against a protected initial template gates memory admission. This is learned, single-pass and joint, not a counterfactual second pass.

**GlobalTrack, SPLT, LTMU, Siam R-CNN (part 1)**
- These are verifiers or re-detectors that use **only the first-frame template**, always or on failure.
- In effect each is a "frame-0-only memory" branch running next to the main tracker. A re-detection with the initial template is the standard long-term recipe.

**SAM2Dual (arXiv 2608.18640, Aug 2026)**
- Status: PP. Reading level: READ.
- Operation:
  - SAM2 memory attention is queried twice, against a short-term bank (K = 7 most recent) and a long-term bank (every m = 10 frames, cap 7). **Both banks are seeded with the frame-1 slot.**
  - The two readouts are fused as `Z = α·Z_ST + (1 − α)·Z_LT`, with a constant α = 0.63. The decoder runs once.
- **Branch disagreement is not used.** The paper lists consistency-aware updates as future work.
- Overlap: two memory readouts of the same frame in SAM2. It differs from ours: there is no first-frame-only bank, the readouts are averaged rather than compared, and there is no verification.

**SAM3Dual (3rd place, MOSEv2, LSVOS 2026; arXiv 2608.22193)**
- Status: REP/PP. Reading level: ABSTRACT.
- Operation: a frozen SAM 3 with short- and long-term branches. Their responses are merged by a fixed schedule, modulated by the previous frame's object confidence.
- A search summary says the first-frame representation stays fixed in both banks. No comparison or verification is described.

**UAMP (PLOS ONE 2026), DTMNet (ACM MM 2020), RPCMVOS (AAAI 2022, arXiv 2112.02853)**
- Status: PR. Reading level: SNIPPET/ABSTRACT.
- Operation: these are learned long/short or propagation/correction dual paths. RPCM's "correction modulator" uses reliable reference cues to correct the propagated embedding.
- None of them run a counterfactual second pass and compare its output.

**Protected memories (no comparison)**
- SAM2: conditioning frames are always attended (CODE/KNOWN).
- XMem++: a *permanent memory* for annotated frames, never evicted (ICCV 2023; ABSTRACT).
- DAM4SAM: the DRM keeps the init frame (part 1).
- Cutie: the object memory is left unchanged when the pooling weights are 0 (READ below).
- These provide the "protected frame-0 memory" ingredient, but none of them uses it in a separate inference pass whose disagreement acts as a signal.

**SAM2-specific search**
- Searches: "SAM2 first-frame-only memory verify", "memory-conditioned vs memoryless prediction compare", "memory ablation at test time", "dual memory disagreement".
- **I found no SAM2, XMem or Cutie paper that re-runs a frame with a frame-0-only memory and uses disagreement with the normal-memory output to accept or reject a candidate.** The practice of re-prompting or resetting exists only informally (GitHub issue #625, blog posts).

**Verdict for M.1: novelty risk MEDIUM.**
- The *concept* has a direct, peer-reviewed precedent: Matthews et al. Strategy 3, which is "re-run with the first template, compare, act conservatively".
- First-template verifiers (SPLT, GlobalTrack, LTMU, MixFormer SPM) are the deep-tracker norm.
- The SAM2 instantiation, a cloned state whose memory bank is restricted to the conditioning frame and whose re-acquisition is checked, was not found.
- It must be cited as "Matthews-style drift correction applied to SAM2 memory", not as a new idea.

### M.2 Independent memory admission (which frames enter L's or G's memory)
**Novelty risk HIGH.** Part 1 §2 and §4 cover this in detail:
- DAM4SAM (no write when the mask is empty; DRM gates).
- SAMURAI, SAM2Long, SAMOSA, SatSAM2, MoSAM, Seg2Track (only High masks are written), SENTRY (refine-before-write), StreamDAM, QDMN, LTMU meta-updater, KeepTrack.

Additions this session:
- STARK and MixFormer score-gated template admission.
- Cutie's zero-weight object-memory freeze.
- Matthews' conservative no-update rule.

### M.3 Crop-centre control in crop-based trackers (where L's crop is centred; behaviour on loss)
- **Centring the search window on the previous estimate.** SiamFC/SiamRPN/DiMP-style trackers do this by design (KNOWN). The search area is about 4–5× the target size; SPLT uses 4×, the same as ours.
- **DaSiamRPN (Zhu et al., ECCV 2018, arXiv 1808.06048)**
  - Status: PR. Reading level: ABSTRACT/SNIPPET.
  - Operation: a switch between the short-term phase and a *failure phase*, triggered by the distractor-aware detection score. In failure, the **search region grows by a constant step each failed frame** ("local-to-global"), and the count of not-found frames is tracked.
- **KeepTrack** enlarges the search area after occlusion. **AdaSAM2** progressively enlarges its crop on loss and re-decides after 10 frames. **CTU** enlarges the window by N·k on loss. **SAMURAI** and **SatSAM2** use Kalman-predicted motion. All are in part 1.
- **Zoom Consistency (arXiv 2604.15376, 2026)**
  - Status: PP. Reading level: READ.
  - Operation: VLM GUI grounding.
    1. Predict a point on the full image.
    2. Crop at r = 0.5 *centred on that prediction* and predict again.
    3. The distance of the second prediction from the crop centre serves as a confidence signal. AUROC is about 0.60, only modest.
  - It is relevant to M.3 and R as "re-centred crop re-prediction consistency".

**Verdict for M.3: novelty risk HIGH.** Re-centring, freezing or enlarging a crop after a low-confidence event is standard (DaSiamRPN, KeepTrack, AdaSAM2, CTU).

---

## T-family: lightweight learned global–local fusion on frozen foundation features; target-token conditioning

### T.1 Global–local feature fusion for high-resolution or small objects

**GLNet (Chen et al., CVPR 2019, arXiv 1905.06368)**
- Status: PR. Reading level: READ.
- Global branch: the downsampled whole image. Local branch: full-resolution tiles (500², 50 px overlap).
- **G→L**: global features are *cropped at the patch location*, upsampled and concatenated to the local features at each layer.
- **L→G**: local features are downsampled, stitched and concatenated back. Bidirectional sharing scored 71.6 mIoU vs 70.9 for G→L only on DeepGlobe.
- Aggregation is a 3×3 conv on the concatenated last-layer features.
- Regulariser: λ = 0.15 · ‖X_loc − X_glb‖₂ on the last-layer features, updating only the local branch.
- Coarse-to-fine variant: the global branch produces a coarse mask, a box is placed around it, and the local branch processes only that box.
- Overlap:
  - **The architectural ancestor of "G and L features fused per layer".**
  - **The coarse-to-fine variant is an object-relative crop chosen from the global prediction.**
  - It is trained end to end with no frozen backbone, works on images, and uses no target tokens.

**FCtL (Li et al., ICCV 2021, "From Contexts to Locality")**
- Status: PR. Reading level: ABSTRACT.
- Operation: a local patch is fused with *several* context crops of different sizes via locality-aware contextual correlation (attention), followed by contextual-semantics refinement.

**MagNet (CVPR 2021)**
- See F.2. Multi-scale progressive refinement combining a global downsampled context with local patches.

**SAM2Refiner Localization Augment (ICCV 2025)**
- See F.2 for the operation.
- **The closest SAM2-specific precedent for T.1**: frozen SAM2 encoder features from zoomed sub-images are fused into the global features by cross-attention (global queries, local keys and values), with a lightweight trained module on frozen SAM2.
- Differences from T:
  - The crops are fixed quadrants, not object-relative.
  - There is no target token; the fusion is dense feature-to-feature.
  - There is no gate.
  - It is trained on images only, and there is no second memory state.

**SAHI, f-BRS zoom-in, ZoomTrack**
- SAHI: slicing for small-object *detection* (KNOWN; part 1).
- f-BRS zoom-in: interactive segmentation crop (KNOWN; M6.1).
- ZoomTrack: non-uniform resizing of the search region (NeurIPS 2023; KNOWN).
- They show that "zoom for small objects" is standard. None of them fuses learned features across views.

**HQ-SAM / HQ-SAM 2**
- A lightweight trained token plus feature fusion on a frozen SAM (KNOWN).
- **This is the standard precedent for "small trainable module on frozen SAM features".**

### T.2 Target-token / object-query cross-attention conditioning

**SAM2 object pointers**
- Reading level: CODE plus KNOWN.
- Operation:
  - The mask decoder's output token is projected (`obj_ptr_proj`) into an object pointer.
  - Up to `max_obj_ptrs_in_encoder = 16` past pointers, with temporal position encoding, are concatenated into the memory-attention keys and values, so current-frame features cross-attend to target tokens.
  - When the object is judged absent, a learned `no_obj_ptr` is substituted. This is a built-in "abstain" token.

**Cutie object transformer (Cheng et al., CVPR 2024, arXiv 2310.12982)**
- Status: PR. Reading level: READ.
- Object memory:
  - `S_q = Σ U(i) W_q(i) / Σ W_q(i)`.
  - Queries 1..N/2 pool only foreground (mask ≥ 0.5) and queries N/2+1..N pool only background, with learned sigmoid weights.
  - It is a streaming average. A slot is **left unchanged when its weights sum to 0**.
- Object queries: `X₀ = X_learned + S`.
- Each block runs:
  1. **foreground/background-masked cross-attention** from queries to pixel readout;
  2. self-attention and FFN on the queries;
  3. reverse cross-attention from pixels to queries;
  4. pixel FFN.
- **No gate.**
- Overlap: **"target tokens built from the prompt/mask that read pixel features and write back a residual" is exactly Cutie's object transformer**, with fg/bg masked attention in place of a gate.

**MixFormer Score Prediction Module (CVPR 2022)**
- Status: PR. Reading level: SNIPPET.
- Operation: a score token cross-attends to the search tokens, *then* to the initial-target tokens. The model reads current features, then compares them with the frame-0 target.

**OSTrack (ECCV 2022, arXiv 2203.11991), early candidate elimination**
- Status: PR. Reading level: SNIPPET.
- Operation: search tokens are ranked by attention to the **template centre token** (best among 4 choices), and low-ranked background tokens are dropped.
- This is target-token-driven selection of which features to trust.

**TransT (CVPR 2021) and DETR object queries**
- Reading level: KNOWN.
- Template–search cross-feature attention, and learned queries with a "no object" class.
- I did not re-read TransT this session.

**Flamingo gated cross-attention (Alayrac et al., NeurIPS 2022)**
- Status: PR. Reading level: KNOWN. The abstract page does not show the detail.
- Operation: `y = x + tanh(α)·XAttn(x, visual)`, with α initialised at 0, inserted into a frozen LM.
- **This is the standard "gate the new residual, start at identity" mechanism.**

**Uncertainty-aware or abstaining attention**
- Reading level: KNOWN.
- Examples: SuperGlue/KeepTrack dustbins, SAM2 `no_obj_ptr` and no-memory embeddings, DETR ∅ class, Heo et al. "Uncertainty-Aware Attention" (NeurIPS 2018).
- Learned "abstain" slots exist widely.

**Direct-precedent search**
- Searches: "target query token cross-attention to both full-frame global features and zoomed local crop features gated fusion", "frozen SAM2 zero-initialized gated cross-attention adapter target tokens".
- Adapter papers found: SAMWISE (CVPR 2025), DD-SAM2, SAM2Matting and CamoSAM2. None matches.
- **Not found**: "target tokens initialised from the frame-0 prompt read G's and L's frozen SAM2 features in separate cross-attentions, and a learned gate scales L's residual."

**Verdict for T: novelty risk MEDIUM (component risk HIGH).**
- Every operation has a close precedent:
  - target tokens reading pixel features (Cutie; SAM2 object pointers);
  - comparison with frame-0 target tokens (MixFormer SPM);
  - global–local cross-attention on frozen SAM2 features (SAM2Refiner LA);
  - per-layer global–local fusion (GLNet);
  - per-pixel learned scale gate (Tao et al.);
  - zero-init tanh gate (Flamingo).
- The remaining distinctions:
  - (i) The local view is an *object-relative, memory-carrying second SAM2 state*, not a fixed tile or quadrant.
  - (ii) The gate is conditioned on the target tokens, so it can choose L or G per target and per frame, rather than per pixel from features.
  - (iii) It runs in a streaming VOS setting with two memories.
- A reviewer will read it as "SAM2Refiner-LA + Cutie object queries + Flamingo gate, applied to an object-relative crop". It is publishable only if the gating demonstrably beats the rule-based SGGL selection and a Tao-style per-pixel gate baseline.

---

## R-family: resolution sensitivity as an uncertainty or observability signal; cross-scale consistency

**Rottmann & Schubert, "Uncertainty Measures and Prediction Quality Rating for the Semantic Segmentation of Nested Multi Resolution Street Scene Images" (CVPRW 2019, arXiv 1904.04516)**
- Status: PR (workshop). Reading level: ABSTRACT/SNIPPET.
- Operation:
  1. Run a segmentation net on **a hierarchy of nested centre crops, each rescaled to the same input size** (that is, several effective resolutions).
  2. Compute the mean and **variance across crops** of the probabilities and of the pixel-wise uncertainty maps.
  3. Aggregate per predicted segment and meta-classify IoU = 0 vs IoU > 0 (AUROC 89.89%), or meta-regress IoU (R² 0.85).
- Overlap: **the closest precedent for R.** Cross-zoom prediction variance is used to rate a *segment's* correctness. This is per-segment false-positive detection, which is our "is this re-acquired component real?" question in spirit.
- Differences: semantic segmentation of single images, no tracking state, fixed centre crops, no object identity.

**Mask Transfiner incoherent regions (F.2)**
- The ×2 down-and-up information loss of the *mask* is used as an error-proneness map.
- This is resolution sensitivity applied to the output rather than the input.

**Zoom Consistency (arXiv 2604.15376)**
- See M.3. Re-predicting at a centred zoom and using the displacement as a confidence signal, with weak AUROC of about 0.60.

**TTA uncertainty**
- Wang et al., Neurocomputing 2019 (arXiv 1807.07356; READ at SNIPPET level this session).
- A multi-organ CT TTA study (PMC, 2026; SNIPPET) averages the variance or entropy across augmentations within a dilated organ mask.
- This is the general principle.

**Historical-snapshot trajectory consistency (Li, Hong, Zhao, "SME", ICCVW 2015)**
- Status: PR (workshop). Reading level: SNIPPET.
- An ensemble of the current tracker and its past snapshots, with the expert chosen by accumulated trajectory consistency.
- This is a precedent for "consistency of trajectories under different model states" as identity or failure evidence. The variation is temporal, not across scales.

**FB error (Kalal 2010), ODIN, augmentation-sensitivity risk scoring**
- Already in part 1 §5.

**Search result**
- Searches: "scale consistency failure detection tracking", "SAM uncertainty resolution zoom".
- **I found no VOS or tracking work that uses the *cross-scale prediction trajectory* (G at native scale vs L at 4× zoom, or L under down-up degradation) as identity or observability evidence for a re-acquisition.**
- SAM-specific uncertainty work (UncertainSAM, arXiv 2505.05049; RUAC, arXiv 2605.10603; SNIPPET) uses learned heads or Bayesian decoders, not resolution probes.

**Verdict: novelty risk LOW-MEDIUM for the tracking/VOS use; MEDIUM if framed generally.**
- Rottmann & Schubert already establishes "cross-zoom variance → segment-level false-positive detection" in segmentation, and must be cited.
- The utility caveat from part 1 still holds. Resolution sensitivity measures *observability*, not *identity*: a small distractor and a small true target are both resolution-sensitive.

---

## O1: event-triggered verification (run expensive checks only on disagreement or low confidence)

**Classic cascades**
- **Viola–Jones (CVPR 2001; KNOWN)**: a cheap-to-expensive rejection cascade.
- **Clockwork convnets (ECCVW 2016; KNOWN)**: update deep layers only when the shallow features change.

**DVSNet (Xu et al., CVPR 2018, arXiv 1804.00931)**
- Status: PR. Reading level: ABSTRACT/SNIPPET.
- Operation: a decision network predicts an "expected confidence score" per frame region.
  - High-confidence regions are flow-warped (cheap).
  - Low-confidence regions are re-segmented (expensive). Up to 95% of the computation is saved.

**Tracking**
- DaSiamRPN switches to a global search only on failure (M.3).
- SPLT does a global search only when the verifier score is < 0.65.
- LTMU runs global re-detection only when the verifier score is < 0.
- MBMD / LT_DSE follow the same pattern (part 1).

**SAM2-specific**
- Lean-SAM2: window routing with a full-frame fallback when the occlusion score is < 0.99 (part 1).
- AdaSAM2: the RL crop policy fires only on "unreliable" events (part 1).
- StreamDAM: a presence head gates dense or sparse memory reads (part 1).

**Verdict for O1: novelty risk HIGH.**
- Disagreement- or confidence-triggered expensive checks are a standard efficiency and robustness pattern.
- The only content we can claim is *which* trigger (G/L disagreement) and the measured cost/benefit.

---

## Summary table

| Candidate family | Closest work(s) | Overlapping operation | Remaining distinction | Novelty risk |
|---|---|---|---|---|
| F.1 pixel-wise logit/prob fusion of G and pasted L | Tao et al. 2020 (hier. multi-scale attention, READ); Chen et al. CVPR'16; CGFSeg (LSVOS'25, READ); multi-scale/flip TTA | Per-pixel (learned or voted) fusion of predictions of the same image at different scales or from different models | Members are two separately-memoried SAM2 states (full frame vs object-relative 4× crop) | **HIGH** |
| F.2 refine only the disagreement/uncertain region | PointRend; Mask Transfiner (down-up incoherence); MagNet; BPR; CascadePSP; AFB-URR (VOS); SAM2Refiner | Refinement restricted to uncertain or boundary pixels, often with high-resolution patches | Uncertainty is defined by G-vs-L disagreement instead of probability or mask down-up loss | **HIGH** |
| F.3 asymmetric add/delete edits | MiVOS DAF (separate D⁺/D⁻, READ); TLD P/N experts; hysteresis/seeded growth; STAPLE (per-source sens./spec.); KeepTrack (switch must beat hist. max); SAM2 fill-holes (add-only, CODE) | Separate handling or evidence for added vs removed foreground | No published cross-scale VOS fusion found that requires identity evidence to add and stronger negative evidence to delete; it must be shown that the asymmetry itself matters | **MEDIUM** |
| F.4 component-level identity check before merging | CTU IoU gate; HQTrack; DAM4SAM (largest CC of alt. mask); CGFSeg ID voting; panoptic merge heuristics; SENTRY Hungarian | Component or mask-level overlap/ID test before accepting or merging | Per-component cross-view test (small step beyond CTU's whole-mask IoU) | **MEDIUM-HIGH** |
| M.1 counterfactual frame-0-only memory re-inference | **Matthews et al. 2003/04 Strategy 3 (READ)**; SPLT/GlobalTrack/LTMU first-frame verifiers; STARK/MixFormer SPM (static template + score); SAM2Dual (two readouts, averaged, READ); XMem++ permanent memory | Re-run the same frame with only the protected first template or memory, compare with the normal result, act conservatively on disagreement | SAM2 instantiation (cloned state with a conditioning-frame-only bank, used to accept or reject a crop re-acquisition) not found | **MEDIUM** |
| M.2 independent memory admission | DAM4SAM, SAMURAI, SAM2Long, Seg2Track, SENTRY, QDMN, LTMU, KeepTrack, STARK/MixFormer, Cutie | Gate which frames are written | Only the trigger (crop-state re-acquisition) and crop-coordinate memory | **HIGH** |
| M.3 crop-centre control / loss behaviour | DaSiamRPN local-to-global; KeepTrack; AdaSAM2; CTU; SAMURAI/SatSAM2 KF; Zoom Consistency | Re-centre, freeze or enlarge the search window on low confidence | None material | **HIGH** |
| T. target tokens read G and L frozen SAM2 features, gated local residual | Cutie object transformer (READ); SAM2 object pointers (CODE); SAM2Refiner Localization Augment (global ← local cross-attn on frozen SAM2, READ); GLNet (READ); Tao per-pixel scale gate; MixFormer SPM; OSTrack CE; Flamingo tanh gate; HQ-SAM | Target or object tokens cross-attending to pixel features; global-local cross-attention on frozen SAM2; learned gates between scales or into frozen models | Exact combination not found: object-relative memory-carrying local state, target-conditioned gate between two SAM2 states, streaming VOS | **MEDIUM** (components HIGH) |
| R. resolution sensitivity / cross-scale consistency as evidence | Rottmann & Schubert CVPRW'19 (nested-crop variance → segment IoU = 0 meta-classification); Mask Transfiner incoherence; Zoom Consistency (2026 PP); TTA uncertainty; SME snapshot consistency; FB error | Prediction variance across zoom levels or perturbations used as reliability or false-positive evidence | Not found in VOS/tracking as identity or observability evidence for re-acquisition; utility caveat (observability ≠ identity) | **LOW-MEDIUM** (MEDIUM if framed generally) |
| O1. event-triggered verification | Viola–Jones; Clockwork; DVSNet; DaSiamRPN; SPLT; LTMU; Lean-SAM2; AdaSAM2; StreamDAM | Run the expensive path only on low confidence or disagreement | Only the specific trigger and the measured cost/benefit | **HIGH** |

**Bottom line**
- No M7 family is clean.
- The two least-covered slivers:
  - **F.3**: an explicitly asymmetric add/delete fusion rule between two SAM2 views. Its analogues are hysteresis, TLD P/N and MiVOS D±.
  - **R**: cross-scale sensitivity as re-acquisition evidence in VOS. Its analogue is Rottmann & Schubert.
- **M.1 must cite Matthews et al. 2004.** It is the same "re-run with the first template, compare, be conservative" logic, and presenting the SAM2 version as a new idea would be an overclaim.
- **T must cite SAM2Refiner (LA), Cutie, GLNet, Tao et al. and Flamingo.** Novelty there rests on the combination and on the evidence, not on any single operation.

## Caveats
- Every READ went through an LLM page summariser, except Matthews et al. (raw PDF text). Re-check numbers before citing.
- Read at SNIPPET or ABSTRACT level only: STARK, MixFormer, OSTrack, FCtL, BPR, MagNet, DaSiamRPN, DVSNet, STAPLE, Rottmann & Schubert (abstract), AFB-URR, SME, UAMP, RPCMVOS, SAM3Dual, XMem++.
- KNOWN, not re-read: PointRend, Mask Transfiner details, Flamingo gate, TLD P-N, hysteresis, Viola–Jones, Clockwork, TransT, DETR, HQ-SAM, Heo et al. Verify before citing.
- Not exhaustive:
  - VOTS2025 per-tracker descriptions are still unread (gap carried over from part 1).
  - The LSVOS 2025/2026 reports were only skimmed. CGFSeg was read; SAMSON and SAM3Dual were seen at abstract level only.

## Sources (URLs)
- Tao et al.: https://ar5iv.labs.arxiv.org/html/2005.10821
- CGFSeg: https://arxiv.org/html/2509.25738
- MiVOS: https://ar5iv.labs.arxiv.org/html/2103.07941
- SAM2Refiner: https://arxiv.org/html/2502.09660v1 ; https://mlanthology.org/iccv/2025/yao2025iccv-finegrained/
- Mask Transfiner: https://arxiv.org/abs/2111.13673
- BPR: https://arxiv.org/abs/2104.05239
- MagNet: https://openaccess.thecvf.com/content/CVPR2021/supplemental/Huynh_Progressive_Semantic_Segmentation_CVPR_2021_supplemental.pdf
- Matthews et al.: https://bmva-archive.org.uk/bmvc/2003/papers/paper-143.html
- STARK: https://arxiv.org/abs/2103.17154
- MixFormer: https://arxiv.org/abs/2203.11082
- OSTrack: https://arxiv.org/abs/2203.11991
- SAM2Dual: https://arxiv.org/html/2608.18640
- SAM3Dual: https://arxiv.org/abs/2608.22193
- RPCMVOS: https://arxiv.org/abs/2112.02853
- UAMP: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0353156
- XMem++: https://arxiv.org/abs/2307.15958
- RMem: https://arxiv.org/abs/2406.08476
- DAM4SAM IJCV: https://link.springer.com/article/10.1007/s11263-026-02790-7
- GLNet: https://ar5iv.labs.arxiv.org/html/1905.06368
- FCtL: https://openaccess.thecvf.com/content/ICCV2021/html/Li_From_Contexts_to_Locality_Ultra-High_Resolution_Image_Segmentation_via_Locality-Aware_ICCV_2021_paper.html
- Cutie: https://ar5iv.labs.arxiv.org/html/2310.12982
- DaSiamRPN: https://arxiv.org/abs/1808.06048
- Zoom Consistency: https://arxiv.org/html/2604.15376
- Rottmann & Schubert: https://arxiv.org/abs/1904.04516
- TTA (Wang et al.): https://arxiv.org/abs/1807.07356
- SME: https://openaccess.thecvf.com/content_iccv_2015_workshops/w14/html/Li_Robust_Visual_Tracking_ICCV_2015_paper.html
- DVSNet: https://arxiv.org/abs/1804.00931
- STAPLE (secondary): https://arxiv.org/abs/1910.12077
- SAMWISE: https://arxiv.org/abs/2411.17646
- UncertainSAM: https://arxiv.org/abs/2505.05049
- RUAC: https://arxiv.org/html/2605.10603v1
