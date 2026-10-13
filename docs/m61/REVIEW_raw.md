# M6.1 independent review (raw), Agent V

Date: 2026-10-14 KST. Scope: the `KEEP_GLOBAL_LOCAL` / SGGL-W (PROVISIONAL) recommendation in
`FINAL_RESEARCH_DIRECTION.md` and the documents, pre-registrations, results and code it rests on. CPU only. I modified no
existing file. The DAVIS-train replication (r4) is still running and is not assessed here.

Severity: **critical** = a stated conclusion is wrong or unsupported. **major** = a conclusion is overstated, or a
decision-relevant fact is missing or biased. **minor** = presentation, consistency or robustness.

## 0. Verdict

- **KEEP_GLOBAL_LOCAL (SGGL = A2x) as a PROVISIONAL direction: supported, relative to what was tested.**
  - A2x is the only consistently positive arm.
  - It beats the flip control F2 and the CTU-style gate A2c on small objects.
  - Its absolute effect is small (+0.004 overall, +0.018 small).
  - It has missed its own +0.03 gate five times, on 10 videos that have been looked at roughly 15 times.
- **The "-W" (write-back) extension in the title and the reframed hypothesis ("더 나은 후보 만들기") are not supported
  (critical, §1.2).** W should be an ablation of SGGL, not its headline.
- **No leakage found** (§2).
- **The statistics are thinner than the documents suggest** (§3).
- **The selection procedure is asymmetric in favour of the incumbent and of W** (§6).

## 1. Are the numbers right, and do they justify the topic?

### 1.1 Recomputation from `comparison_per_unit.csv`

Gate set = DAVIS_val + PUMaVOS_B: 67 objects / 36 videos. "Small" = area < 1%: 15 objects / **10 videos**. Each
bootstrap below is my own 4,000-sample, video-clustered, unit-weighted bootstrap.

| quantity | docs | recomputed | CI (mine) |
|---|---|---|---|
| A2x overall | +0.0042 [+0.0015, +0.0071] | +0.0042 | [+0.0015, +0.0073] |
| A2x small (15) | +0.018 | +0.0177 | [+0.0098, +0.0251] |
| A3 overall | +0.0047 | +0.0047 | [+0.0015, +0.0082] |
| A3 small (15) | +0.020 | +0.0199 | [+0.0103, +0.0301] |
| F2 overall / small | +0.001 / +0.000 | +0.0010 / +0.0001 | [−0.0005, +0.0023] / [−0.0013, +0.0015] |
| HEAD overall | −0.0035 | −0.0035 | [−0.0050, −0.0018] |
| A2x − F2, small | +0.018 [+0.009, +0.026] | +0.0176 (14 up / 1 down) | [+0.0088, +0.0255] |
| A2x − F2, all 67 | +0.003 [+0.0001, +0.007] | +0.0032 (26 up / 21 down) | [−0.0000, +0.0067] |
| A3 − A2x, small | +0.0022 [+0.0002, +0.0054] | +0.0022 (5 up / 0 down) | [+0.0002, +0.0054] |
| A2x − A2c, small | +0.008 [+0.001, +0.014] | +0.0079 | [+0.0014, +0.0137] |

- Every headline number in `comparison_report.json` and the documents reproduces.
- The Track D gate numbers (`gates_report.json`) reproduce:
  - self +0.0193, calibrated +0.0177, oracle +0.0216
  - self / oracle = 0.89
- The complementarity numbers (`complementarity_frames.csv`) reproduce:
  - selection oracle +0.0276
  - combination +0.0306
  - both wrong 2.9%

### 1.2 [critical] W ("one-way write-back") is promoted to the headline although it failed its own pre-registration, and its stated rationale is contradicted by its own oracle test

`preregistration_r2.json` set three success conditions for A3. All three had to hold.

| condition | result | met? |
|---|---|---|
| small ≥ A2x + 0.01 | +0.0022 | ✗ |
| no unit below A2x − 0.01 | none | ✓ |
| per-frame oracle over {G′, L′} ≥ oracle over {B1, A1} + 0.01 | **+0.0002** (`a3_report.json: oracle_gain_new_vs_old_small_over15`) | ✗ |

- The oracle condition was the direct test of the round-2 hypothesis that write-back "creates NEW candidates". It
  returned essentially zero.
- `FINAL_RESEARCH_DIRECTION.md` nonetheless re-targets the next hypothesis to "'더 나은 후보 만들기' (단방향 메모리
  되쓰기)".
  - The oracle failure appears only in `OVERNIGHT_DEEP_RESEARCH_REPORT.md` §2.3.
  - It is absent from FINAL, EXPERIMENT_COMPARISON and FINAL_ALGORITHM_SPEC.
- G′ improves (+0.0086), but the report itself says this happens mostly on frames where L was already the output.
  W makes G copy L. It does not add information.
- **The +0.0022 is not robust:**
  - 90% of the summed A3 − A2x gain comes from 3 units: lab-coat obj2 +0.020, mbike-trick obj2 +0.006, lab-coat obj1
    +0.004.
  - Dropping the single video lab-coat gives **+0.0007** (13 units, 9 videos).
  - The "A3 tiny +0.044 vs A2x +0.036" difference is entirely lab-coat obj2.
  - lab-coat is also the video where A3s collapsed and where Track C found wrong-object capture (frames 1–25). Its
    behaviour is idiosyncratic.
- **Fix:**
  - Title: "Self-Gated Global–Local SAM 2 for tiny-target VOS".
  - Treat W as a pre-registered ablation in M7 with its own kill rule (FINAL kill rule 4 is fine).
  - State in FINAL that A3 failed 2 of 3 pre-registered conditions, including the information test.

### 1.3 [major] "Independent evidence 12 ↑ / 0 ↓, 36 videos" overstates breadth

- The 12 improved units come from **7 videos**:
  - soapbox ×3
  - lab-coat ×2
  - mbike-trick ×2
  - scooter-black ×2
  - drift-chicane, kite-surf, queen_car
- 51 of the 67 gate units are uncropped and are 0 by construction (16 cropped units / 11 videos). They cannot go down,
  so "0 ↓" is mostly structural.
- The tiny stratum (+0.036, +0.044) is 3 units in **2 videos**.
- The "36 videos" in the FINAL evidence table should read "effect carried by 7 videos; 16 objects were cropped".

### 1.4 [major] Omitted out-of-gate negative and null results (repeat of an M6 review finding)

- SegTrack birdfall obj1 (small, 0.4%) is **−0.023** for both A2x and A3. It is the only unit < −0.01 across all
  100 dev units.
- On sets outside the gate:
  - SegTrack: A2x −0.0002 [−0.003, +0.002]
  - PUMaVOS_A: +0.0015 [0.0, +0.0035]
- `docs/m6/REVIEW_raw.md` §2 flagged birdfall. FINAL still reports only "12 ↑ / 0 ↓". Only
  `OVERNIGHT_DEEP_RESEARCH_REPORT.md` §1 mentions "13 ↑ / 1 ↓" (all dev).
- FINAL's evidence table should include the all-dev row (+0.003 [+0.001, +0.005], 13 ↑ / 1 ↓) and name birdfall.

## 2. Causality and leakage (checked in `scripts/m61.py`)

| path | finding | severity |
|---|---|---|
| TPLR (`tplr`) | The box comes from the B1 mask at frame t (`G[t-1]` = video_idx t, logged online run). The obj score comes from the B1 log at t. The crop side comes from GT0 (the prompt). GT is used only in `m55.jf`. Causal. | — |
| A3 / A3s (`run_dual`) | The crop centre uses only outputs up to t−1. Write-back at t re-encodes G's (or L's) frame-t memory from the frame-t fused output, after both states predicted t. It affects only frames > t. `tests/test_m61.py` checks the n=5 vs n=8 prefix and A2d = logged A2x. Causal. | — |
| F2 (`run_flip`) | The flipped state is prompted with flipped GT0 only. Output = mean of the two probabilities. Causal. | — |
| Track D replay (A2o) | Pure replay of the logged B1/A1. Gate features are online scalars and mask geometry. GT is used only as the fitting target on train and for evaluation. | — |
| Calibrator fit | `gates()` fits only on `trackD/runs` = `davis_units("train")`. Dev logs are used for prediction and for the scale-bias diagnostic. **DAVIS train only: confirmed.** | — |
| Head training | `_samples()` uses `davis_units("train")` only, for both positive controls and main training. Image eval uses dev GT boxes and is labelled retrospective. **DAVIS train only: confirmed.** | — |
| W uses G's object score | `run_dual` passes `oG["object_score_logits"]` to `_run_memory_encoder`. When G's score ≤ 0 but L is accepted, SAM 2 adds `no_obj_embed_spatial` on top of the written mask (`sam2_base._encode_new_memory`), and `obj_ptr` stays G's. On dev this happened in 1 of 2,007 write frames (kite-surf obj2). On a tiny-target set where G loses the target, this is exactly the case W is meant for. Decide the score source (L's, or max) before M7. | minor (design) |
| DAVIS train in SAM 2 training | DAVIS 2017 train is in SAM 2's training mix (SAM 2 paper; also noted in `preregistration_r4.json`). The replication is not clean independent evidence, and its bar (small > 0, CI > 0) is low. | minor (already disclosed) |

No GT or future-frame leakage was found in any deployable arm.

## 3. Statistical units

- **[major] Few clusters.**
  - Every small-object claim rests on **10 videos**. Every tiny-object claim rests on **2 videos**.
  - Percentile cluster bootstrap with 10 clusters gives intervals that are too narrow. M6 review §3 already said
    this, and it was not addressed.
  - Lower bounds such as A3 − A2x +0.0002 and A2x − F2 (all) +0.0001 sit at the edge of zero. They should not be read
    as "CI > 0, therefore real".
  - Report the number of videos next to every CI, and add a video-level sign or t test.
- **[major] Dilution by structural zeros.**
  - "Overall" CIs (67 units) average 51 forced zeros with 16 cropped units.
  - The tiny paired "all" CIs (for example A3 − A2x all [+0.00003, +0.0012]) are the same 5 nonzero units rescaled.
  - They carry no independent information beyond the small-unit test.
- **[major] Multiple looks.**
  - The same 15 small units (10 videos) have been used for:
    - M5.5/M5.6 exploration
    - M6 rounds 1–5 (A2x was itself chosen **post hoc** in M6 round 2, commit `6af5464`)
    - M6.1 rounds 1–3
  - There has been no multiplicity correction and no held-out split.
  - The incumbent's +0.018 is therefore a winner's-curse estimate. "A2x − F2 = +0.018" is essentially A2x's own
    selected gain, because F2 ≈ 0.
  - FINAL acknowledges this under "데이터 재사용". It should also say that this inflates the positive claims, not only
    that it protects against false passes.
- **Robustness of the incumbent itself (positive):**
  - A2x small without lab-coat: +0.014 [+0.008, +0.019].
  - A2x − F2 small without lab-coat: +0.014 [+0.008, +0.019].
  - Leave-one-video-out range: +0.014 to +0.019.
  - The direction of the A2x effect is robust. Its size is not.

## 4. Strongest counterarguments; is another candidate better supported?

1. **Effect size versus target.**
   - A2x is +0.018 on small objects, against a +0.03 gate missed five times.
   - The M6 review measured the all-arm per-frame oracle at +0.037 on these units, so the gate may be unreachable on
     this data.
   - The direction is being kept because nothing else worked. That does not show that it works.
2. **The second state buys safety, not accuracy.**
   - A1 (crop only, 1× compute) already gives +0.0167 small (M6).
   - A2x adds about +0.001 and removes A1's ordinary-object regressions (motocross −0.114).
   - The "beats a 2× generic ensemble" argument is therefore partly a comparison of two ways to spend the second
     forward pass. Only A2x's spends it on a fallback.
3. **W is near-null** (§1.2).
4. **Alternative candidates:**
   - **SGGL without W** is equally supported, simpler and cheaper (no extra memory encoder call, no feedback risk). It
     should be the main method.
   - **The instructed fallback (reliability/selection)** is not better supported on dev data (§7). It is not
     clearly worse supported than W either.
   - **TASK-M7-010 ("upper-bound analysis of training-free inference-time improvements to SAM 2.1")** may be the best
     supported contribution. The documents already contain a coherent negative-results body:
     - selection saturated: 89% of oracle
     - candidate complementarity +0.003
     - TPLR, temporal reuse, head, calibration and flip-TTA all null or negative
     - feedback collapse
   - I would carry this as a co-primary fallback deliverable, not only a "기각 시" branch.

## 5. Novelty versus prior work

- **CTU thesis:**
  - The comparison is gate-only. A2c transplants the CTU acceptance rule onto our object-relative crop policy, with
    γ = 0.5 chosen by us; the thesis's γ is unknown according to `PRIOR_ART_raw.md`.
  - The original CTU H/4 and H/2 windows with SAM2.1-L were never run.
  - "beats the nearest prior-art gate" is a fair claim. "beats CTU" would be an overclaim. FINAL wording is mostly
    careful. Keep it that way, and run the CTU original in M7-005 as planned.
- **Write-back:**
  - Re-encoding a better frame-t mask into SAM 2 memory is SAM 2's own interactive-correction pathway (`add_new_mask`
    and the re-encode used by P5). Here the crop tracker plays the role of the user.
  - Track Anything writes SAM refinements into XMem memory.
  - The novelty left is "the source is the same SAM 2 on a zoomed crop", which is narrow. Calling the directional
    finding an "analysis" is an overclaim (§6.3).
- **[major] Missing prior-art family.**
  - Long-term trackers have long combined local search with global re-detection and a verifier or switch: LTMU, SPLT
    (skimming-perusal), GlobalTrack, Siam R-CNN.
  - In small-object detection, slicing inference (SAHI) is standard.
  - None of this appears in `PRIOR_ART_*.md`. This comes from my own knowledge and was not verified against the repo.
  - A reviewer would raise these against "global–local with self-gated acceptance".
- **HQTrack, AdaSAM2:** the differences are stated correctly (full-frame / different model; RL scale choice).
- VOTS appendices unread: correctly flagged as the main risk (TASK-M7-001).

## 6. Bias in the selection procedure

1. **[major] Asymmetric bars.**
   - The incumbent (post hoc, never passed its gate) faced no margin requirement.
   - Challengers had to beat it by +0.01 (TPLR, A3) or +0.005 (D).
   - D's bar was **unpassable by construction**: oracle − self headroom on A2o = +0.0216 − 0.0193 = **+0.0023 < 0.005**.
   - W failed its bar and was adopted anyway.
   - Rules were enforced to reject challengers and waived to adopt W.
2. **[major] Track B was under-budgeted and mis-described.**
   - The main training loss was flat: first 500 steps 0.2235, last 500 steps 0.2149 (`trainlog.csv`; pc2 reached
     0.055).
   - Training-set J for the main run was never measured.
   - The run therefore shows no fit to the training distribution, so "일반화 실패 / 기억은 가능하지만 일반화 안 됨"
     is the wrong diagnosis. It is under-fitting or inconclusive.
   - There are also two mismatches between training and evaluation:
     - training used ±10% jittered GT-box image-mode outputs
     - evaluation applied the head to memory-conditioned video decoder outputs
   - Unreported protocol deviation:
     - pre-registration says "AdamW 1e-3 cosine"
     - `_train` with `steps=None` makes the `LambdaLR` factor 1.0, so the learning rate was **constant**
   - The decision impact is small, since B's prior-art risk is high anyway. The label should be "inconclusive at a
     30-minute budget", not "rejected".
3. **[minor/major] TPLR was a weak instantiation.**
   - Box-only prompt, with no mask_input or point prompt from the tracker.
   - No gate.
   - The box is truncated to the crop (`vcam.bbox(gm[y0:y1, x0:x1])`).
   - The information-oracle criterion (+0.001) does not depend on the gate, so the rejection stands.
   - The mechanistic reading ("메모리가 부분 대상 정의를 갖는다") is confounded by the lack of a mask prompt, the
     standard way to pass the part definition to SAM.
4. **[minor] F2 is one generic control.**
   - Flip TTA produces highly correlated predictions.
   - A higher-resolution or tiled full-frame 2× control would test "is it just more pixels?" more directly.
   - F2 being run only on the gate set is fine, since all comparisons are on the gate set.
5. **[minor] Track D was evaluated on the A2o replay, not on A2x.**
   - In A2x the gate also steers the crop trajectory, so the "+0.002 headroom" is A2o's, not A2x's.

## 7. Deviation from the instructed fallback ("scale-aware reliability estimation")

**[major] Partly honest. It overstates how decisively Track D refuted the fallback.**

- **What is correct:** on these candidates and this data, per-frame selection is saturated. The A2o oracle is only
  +0.0023 above the self gate. No reliability estimator can add more than that to selection here, and FINAL says so.
- **What is overstated:**
  1. The D success bar was unreachable (§6.1), so "not supported" is a headroom statement, not evidence against
     calibration.
  2. Reliability has uses beyond output selection, and the documents ignore them:
     - **W's own write admission.** W writes on every self-accepted frame (write rate 0.89). The wrong-adopt rate is
       about 2%, and those frames are where feedback collapse would start.
     - **Confident-wrong failure detection.** In billie_shoes the obj score stays above 5.9 and iou_pred is 0.8–0.97
       while J = 0.
  3. The D1 scale-bias finding (crop iou_pred +0.111 optimistic below 500 px) supports the fallback's premise exactly
     in the tiny regime that M7 targets. It rests on only 6 objects.
  4. The evidence for W (+0.0007 without lab-coat) is no stronger than the evidence against the fallback. Choosing W
     over the instructed hypothesis is a judgement call, not a data-driven win.
- **Suggested honest framing:**
  - The instructed fallback is "not testable on current dev data (no selection headroom)".
  - M7 should test reliability where it can matter: write-back admission, and selection on a tiny-target set with
    more headroom.
  - This keeps SGGL and absorbs both W and the instructed hypothesis as pre-registered ablations.

## 8. Inconsistencies and gaps across documents

| item | where | severity |
|---|---|---|
| A3 oracle criterion failure (+0.0002) missing | FINAL, EXPERIMENT_COMPARISON, SPEC (present only in OVERNIGHT §2.3) | major (§1.2) |
| Birdfall −0.023 and SegTrack/PUMaVOS_A nulls missing from "12 ↑ / 0 ↓" | FINAL | major (§1.4) |
| "방향성은 측정된 사실이다" / "반대 방향이 붕괴한다는 방향성 분석" rests on **one** collapsed unit (A3s small vs A2x CI [−0.127, +0.001] includes 0) | OVERNIGHT §2.3, FINAL novelty (2), SPEC | major |
| Two different CIs for A3 − A2x: [+0.0002, +0.0054] over 15 units vs [+0.0003, +0.006] over 13 applied units. SPEC and the COMPARISON §1 table use different ones | FINAL, SPEC, COMPARISON | minor |
| "Small" used for both 15 (< 1%) and 12 (0.1–1%) units in adjacent tables | COMPARISON §1 vs §2 | minor |
| Unfilled placeholders: §4.3, §6 "[실행 후 기입]", §8 "[결정 후 기입]". The pre-registration table lists r1, pc2, r2 but omits r3 (F2, `0bb3a78`) and r4 (replication, `f992b6f`). §0 points to §4.3, but the A3 content is in §2.3 | OVERNIGHT_DEEP_RESEARCH_REPORT | minor |
| TASK-M7-003 cites "FINAL_RESEARCH_DIRECTION §J", which does not exist. TASK-M7-006 expects "극소 객체에서 **A2x** ≥ +0.03", but FINAL's gate is "**A3** ≥ +0.03". TASK-M7-008 "§J 4 판정" (Hiera-L) does not match FINAL kill rule 4 (write-back). These must be unambiguous before the M7 pre-registration | IMPLEMENTATION_ROADMAP | major (for M7) |
| Track B labelled "일반화 실패" despite a flat training loss; cosine-vs-constant LR deviation unreported | COMPARISON, FINAL §1 table, OVERNIGHT §5.3 | major (§6.2) |
| Head train-loss quote 0.224 → 0.216 | OVERNIGHT §5.3 | OK (first/last 500 steps: 0.2235 → 0.2149) |

## 9. Recommended changes before committing the decision

1. Retitle and re-scope to SGGL (A2x). Make W an M7 ablation and state its 2-of-3 pre-registration failure in FINAL.
2. Add to FINAL:
   - the number of videos behind each CI
   - leave-one-video-out results (A3 − A2x without lab-coat +0.0007)
   - all-dev 13 ↑ / 1 ↓, naming birdfall
3. Reword the Track B verdict to "inconclusive (under-fit, 30 min; constant LR deviation)".
4. Reword the D / fallback deviation:
   - "no selection headroom on this data; bar was unreachable"
   - carry reliability-gated write admission as a pre-registered M7 arm
5. Downgrade the bidirectional-collapse claim to "observed once (lab-coat obj1)".
6. Fix the roadmap cross-references (§J, A2x vs A3 in TASK-M7-006) before writing `outputs/milestone7/preregistration.json`.
7. Add long-term-tracking global/local re-detection (LTMU, SPLT, GlobalTrack, Siam R-CNN) and SAHI to TASK-M7-001.
8. Decide the object-score source for W (G vs L) before M7.

## Reproduction

The scripts are in the session scratchpad and are not committed. They read only these files:
- `outputs/milestone61/comparison_per_unit.csv`
- `trackA/{a3_report.json,complementarity_frames.csv,A3/*.csv}`
- `trackD/gates_report.json`
- `trackB/{positive_control*.json,image_eval.json,trainlog.csv}`

Video-clustered bootstrap: resample videos with replacement, take the mean over the pooled units, 4,000 draws,
seed 0.
