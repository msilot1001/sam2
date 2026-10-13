# M6 independent review (raw)

Reviewer: independent skeptical pass, 2026-10-13. Read-only, except this file. CPU-only pandas checks on
`outputs/milestone6/{runs,reports}`. No GPU used. A2c (round 5) was still running during the review.

Severity: **critical** = conclusion wrong or unsupported. **major** = conclusion overstated, or a decision-relevant
gap. **minor** = wording, bookkeeping or reproducibility.

## Verdict in one paragraph

The engineering is clean. I found no GT or future-frame leakage in any online path. The gate numbers reproduce
exactly from the per-unit CSVs, and "Track A failed its pre-registered gate" holds. Several interpretive claims
in the summary go further than the data supports:

- "Even a perfect selector stays below +0.03" holds only for pairwise B1-vs-one-arm oracles. A per-frame oracle
  over all arms already run reaches **+0.037**.
- The claim that per-frame representation is the limit (video J ≈ image oracle) is an aggregation artifact.
- "Absence detection is solved" rests on pooled AUROC over about 7 videos.
- "H3 refuted" hinges on one unit.
- The head-pilot null comes from a head that barely trained.

None of these flips the decision to **not** pursue zoom as the thesis topic. They do change what can honestly be
written, and what the next step should be.

## 1. Causality and leakage

| Check | Finding | Severity |
|---|---|---|
| Online GT use | `run_unit` reads GT only for the frame-0 prompt/crop centre (`vcam.center_of(gt(u,0))`). Everything else that reads GT (`gt_in_crop`, `jf`, `other_obj_frac`, `J_cand_best`) is logged after `out` is fixed and never feeds back. The next centre comes from `out`. OK. | — |
| Future frames | `LazyCropFrames.__getitem__` is materialised only when SAM 2 requests frame t, after `frames.box` was set from t−1 output. `start_dir` (init, P5 memory encode, preflight) all run with the frame-0 box, so frame 0 is never re-cropped later. SAM 2's `cached_features` holds only the latest frame. OK. | — |
| A2x reading logged B1 | `gm[t-1]` is the B1 mask of video_idx t (B1 npz stores frames 1..n−1) and `gs[t]` is indexed by video_idx. B1 is an independent deterministic online run (reproduces M5.5/5.6 bit-exactly, 9,899 frames, max dJ 0.0), so using its frame-t output at frame t equals running both trackers in parallel. Causal. A2c is the same. | — |
| S1 replay | Arms never exchange state, and the crop arms' centres follow their own outputs, so "B1 on 1..K, then the picked arm's logged trajectory" is exactly what a parallel run with a one-time switch produces. Valid. | — |
| A1v output resolution | Fix `2715ce1` (11:56:36) sets `state["video_height/width"]=side` before every step. It was committed after prereg r2 (11:56:09) and before the A1v run (finished 12:30). For A1 the value is constant and equals the init value, so A1 is unaffected. The memory encoder works on SAM 2's internal 1024 grid, so no other resolution-dependent state needs changing. OK. | — |
| Test coverage | The causality test (`tests/test_m6.py:69`) compares n=5 vs n=8 for **A1 only**. That shows no dependence on later frames being *processed*, but it does not perturb future frame *content*. There is no test for A1v, A2x or A2c indexing (`gm[t-1]`, `gs[t]`), which I verified only by reading the code. | minor |

## 2. Gate recomputation (DAVIS_val + PUMaVOS_B, uncropped units counted as dJF=0)

I recomputed these from `trackA_*_per_unit.csv`. All match `m6_report.json` and the report.

| Arm | n small (applied) / videos | small dJF | frac < −0.01 | non-small dJF (n=52) | non-small dJF on the 3 cropped units |
|---|---|---|---|---|---|
| A1 | 15 (13) / 10 | **+0.0167** | 6.7% | −0.0037 | **−0.064** |
| A1z | 15 (13) / 10 | −0.0445 | 66.7% | −0.0135 | −0.032 (22 cropped) |
| A1v | 15 (13) / 10 | +0.0238 | 0% | −0.0023 | −0.040 |
| A2x | 15 (13) / 10 | **+0.0177** | 0% | +0.0003 | +0.005 |

- The sets, the small definition and dJF=0 for uncropped units all follow `preregistration.json`. The code
  `gate()` implements it literally.
- **[major] The non-small criterion has almost no teeth.** 49 of 52 non-small units are uncropped zeros, so A1's
  −0.114 (motocross) and −0.089 (drift-straight) regressions dilute to −0.0037 and pass ≥ −0.005. The criterion
  was pre-registered, so the verdict stands. The report should still state the applied-unit mean (−0.064), since
  that is what a user of A1 experiences.
- **[minor]** "A2x 11/0, regression 0" is true on the gate set only. On SegTrack, birdfall regresses −0.023.
- Robustness of the A1 gain: leave-one-video-out mean ranges 0.012–0.018. Dropping lab-coat (1 video, 2 units,
  37% of the summed gain) gives +0.012. The video-level t-test against 0 gives p = 0.008. The gain is real but
  small.

## 3. Statistics

- **[major] Too few clusters for the CIs.** The video-clustered percentile bootstrap uses only **10 videos**, and
  13 of the 15 small units are actually cropped. With so few clusters, percentile CIs run too narrow. The report
  never states n_videos next to the CIs. Bootstrap CIs and video-level t-tests should be reported side by side.
  - A1v small CI [+0.007, +0.045] contains +0.03.
  - A video-level one-sample t-test of A1v against 0.03 gives **p = 0.22**.
  - So "A1v < gate" is a point-estimate statement, not a demonstrated one.
- **[major] Multiple looks, and a gate set that was not fresh.** The same development data was used for 5 rounds
  (r1–r5), and DAVIS_val was already explored in M5.5/M5.6.
  - This is acknowledged in §8, but only as protection against false passes.
  - It equally weakens the positive mechanism claims, all of which are post-hoc or one-look on the same 15 units
    with no correction: "H3 refuted", "dose not supported", "iou_pred inflation +0.026 / −0.033", "Spearman −0.59".
- **[minor] Spearman −0.59 not reproduced.** I get −0.30 (median GT area, gate small, p=0.27) and −0.41
  (frame-1 area, all applied small, n=16, p=0.11). The definition is not in committed code. Either way it is not
  significant.
- **[minor] Several headline numbers come from no committed code.** These include video 0.776 vs oracle 0.763,
  the frame-oracle ceilings, M5.6 18/35, the §7 error shares, the multimask gap, iou-inflation and the Spearman.
  I re-derived 0.776/0.763 (419 frames), the frame oracles (A1 0.0216, A1z 0.0235, A1v 0.0285) and 18/35 / 15-of-28.
  The rest is unverifiable. These analyses should be added to `analyze()`.

## 4. Overstated causal claims

**4a. [major] "Video B1 J 0.776 ≈ image oracle 0.763, so temporal propagation is not the bottleneck and
single-frame representation is the limit."**
- The comparison pools frames. At unit level, video averages **0.703** and the image oracle **0.718**, so the sign
  flips. The image oracle beats video on 9/19 small units: kite-surf obj2 by **0.37**, pimples by 0.15, lab-coat by
  0.04–0.06. Video beats the oracle on others (eye_area by 0.22).
- The near-equality is two large opposite effects cancelling, not evidence that propagation is fine. For
  kite-surf obj2, propagation and reacquisition clearly *are* the bottleneck.
- The oracle uses a **GT box** prompt (+ inner point), which is a weaker prompt than the mask memory video gets,
  and box prompts are known to be poor for thin and small targets. It bounds "box-prompted decoding", not the
  representation.
- Acceptable wording: "on average, frozen per-frame decoding with a GT box is not better than video tracking".

**4b. [major] "Head pilot null, so no learnable headroom for a light head on frozen features."**
- The training loss barely moves: means over successive blocks of 500 steps go 0.29 → 0.27 → 0.23 → 0.27 → 0.25.
  That looks like a head that did not fit, not a head that fitted and found nothing.
- The budget was small: 8 min, 2 epochs, 4,196 steps, supervision at 256 against area-downsampled GT. The 256
  supervision means it cannot learn sub-256 detail, which is exactly the regime of interest.
- There was no positive control (does it improve on DAVIS *train*?) and no learning-rate check.
- The base it refines is the box-prompted decoder (small J_frozen 0.696), not the video output.
- The §6 caveat is good, but the §0 verdict "경량 head 모두 여지 없음" should read "an untuned 8-minute pilot found
  nothing; inconclusive".

**4c. [minor] "M5.6: 18/35 transient; 15 of the 28 'unrecoverable' had nothing to recover."**
- The arithmetic is verified from `outputs/milestone56/reports/diagnose_episodes.csv`: J_I0_native ≥ 0.5 in 18/35
  episodes, 15 of them among the 28.
- The correction itself is right. With I0 already ≥ 0.5, the "+0.15 over control" recovery rule could not fire.
  That was a design flaw in M5.6.
- Two caveats:
  - 5 of the 18 sit at 0.53–0.60, right at the threshold.
  - Many are 1–3-frame episodes, which are transient by construction.
- The "persistent failures are mostly small or thin" claim is roughly 10/17. The persistent set also contains
  billie_shoes and paragliding obj2. Both are memory-propagation failures that GT reprompt fixes (0.94, 0.84).

**4d. [major] "Presence AUROC 0.994–0.9999, so absence detection is (nearly) solved."**
- **Class balance and clustering.** GT-empty frames come from only **7 videos / 9 units**:
  - DAVIS 55 frames: india 43 and kite-surf 12.
  - PUMaVOS_A 238 frames: **206 from pimples**, which the report itself calls an annotation artifact.
  - PUMaVOS_B: **6 frames from ice_cream alone**, with 2/6 false presence (33%).
  - SegTrack: 39/40 from monkeydog.
- **Pooling inflates the AUROC.** Positives from easy units dominate. Within-unit AUROCs are lower: pimples
  0.878, india obj2 0.903, kite-surf obj2 0.935.
- **The real failure is reacquisition.** For fully occluded objects, first-frame J after reappearance is 0.25
  (7/10 fail). At kite-surf obj2's reappearances the object score is negative (−0.2 to −2.2) while the object is
  visible.
- **The "online opportunity = 0" argument is weaker than it looks.** It counts only failures where one of the 3
  multimask candidates is already correct. Re-detection failures, where no candidate overlaps the target, are
  excluded by construction.
- Acceptable wording: "frame-level presence is well separated on most units, and Track B is underpowered on this
  data (failures in 3 videos, one of them an annotation artifact)". "Solved" is not justified.

**4e. [major] "H3 refuted, so the memory-scale-consistency novelty is retracted."**
- The pre-registered rule did return "refuted" (A1−A1v = −0.007).
- But the result hinges on **one unit**: kite-surf obj2, where A1v scores +0.179 vs A1's +0.024. Without it,
  A1−A1v = **+0.0035**.
- Across all small applied units (including PUMaVOS_A), A1−A1v = **+0.024 in favour of fixed scale**, driven by
  A1v's catastrophic **smile −0.496**. That collapse is direct evidence that variable-scale memory can fail
  badly.
- The honest reading is "inconclusive: one large win and one catastrophic loss for variable scale". The
  retraction is premature in the opposite direction. The question is still open, though small.

## 5. "Zoom family capped below +0.03 on this data"

- **[major] Not justified as stated.**
  - The report's ceilings (+0.018 per-unit over {B1, A1, A1z}; frame-level +0.022 / +0.024 / +0.029) are
    **pairwise or partial**.
  - Recomputed on the gate set: the per-unit oracle over all arms run (B1, A1, A1z, A1v, A2x) is **+0.0297**, and
    the **per-frame oracle over all arms is +0.037**, which is above the gate.
  - Per-frame switching is not deployable, but it is the stated kind of ceiling ("완벽한 선택기").
- **[major] The cap is partly a property of the crop rule, not of zoom.** The two small units with the most
  headroom are never cropped (dJF ≡ 0) because their prompt box is too large for the rule:
  - kite-surf obj1, B1 0.66
  - paragliding obj3, B1 0.43, a thin line
- **[major] The data has little headroom.** Mean B1 JF on the 15 gate small units is 0.817, but 11 of the 13
  cropped units already sit at 0.87–0.95. A +0.03 mean asks for removing about 40% of their residual error.
- Defensible statement: "box-centred single-crop zoom with the tested scales gives about +0.02 on DAVIS/PUMaVOS
  small objects. The headroom on this data is about +0.03–0.04 even with oracle switching, because these
  'small' objects mostly do not fail." §7's error decomposition (small < 500 px = 11% of J error) is the
  strongest argument, and it is the one to lead with.

## 6. Novelty

- **[minor] The prior-art assessment is honest.** CTU covers crop-around-previous-mask and global+local parallel
  with selection. AdaSAM2 covers scale selection.
- **[major] `PRIOR_ART_COMPARISON.md` is stale and inconsistent with the overnight report.**
  - §2/§3 still name the final candidate as "SAM 2 메모리 배율 일관성 유지 객체 상대 확대 추론" and point to
    `FINAL_RESEARCH_TOPIC_DECISION.md`, **which does not exist**.
  - Its "남는 틈 (1)(2)" (object-relative fixed scale; memory-scale consistency) are exactly what M6 retracted
    or found inconclusive.
- **[minor] Overlap with CTU asserted before A2c finished.** §4.5 asserts A2x "mechanism is the same as CTU", but
  round 5 (A2c, CTU's acceptance rule) was pre-registered to test exactly that distinction and is unreported
  (still running).
- **[minor] One possibly reportable observation.** CTU reports variable windows hurting, while M6 finds A1v ≈ A1
  on average, with a catastrophic smile case. That is a small, possibly reportable observation, not a method.
- Nothing in M6 is presented as novel that the notes show is covered, apart from the stale document above.

## 7. Missing for a final research-topic decision

1. **[major] A2c result.** Round 5 is pre-registered but not yet in the report, and §4.5 depends on it.
2. **[major] Data with real small-target failure.** All Track A/B/C conclusions are "on this data", where
   small = < 1% at the prompt but mostly easy, and the extreme-small regime exists in one video (lab-coat).
   - A decision against zoom needs either a pre-registered check on a dataset with many failing tiny targets
     (e.g. small-object subsets of MOSE/LVOS/SA-V, or UAV VOS), or an explicit statement that the topic is
     rejected **for DAVIS-like data**.
   - PUMaVOS C has only 2 small objects and cannot do this.
3. **[major] Backbone dependence.** Everything is Hiera-T. The CTU result (EAO +0.03–0.04) used SAM2.1-L. The
   claims "per-frame representation is the limit" and "zoom capped" may not transfer. One B+/L run of B1 vs A1
   on the 15 gate units is cheap.
4. **[major] Where the remaining error actually is.** §7 says 59% of J error is in frames with J ≥ 0.5 (boundary
   and extent), with the top 12 objects holding 58%. That points to boundary/detail refinement, which is
   SAM2Refiner/HQ-SAM2 territory and prior-art heavy, or to per-object failure modes.
   - The report does not draw the decision consequence: no listed track addresses this error mass in a novel
     way.
   - A decision document should say explicitly whether the project pivots: to a diagnostic/benchmark
     contribution, to a different failure mode (e.g. memory-propagation drift such as billie_shoes), or to a new
     dataset regime.
5. **[minor] Bookkeeping.** The report's prereg table lists rounds 1–4 with the placeholder "3회차 커밋 / 4회차
   커밋". The hashes are c102f6d (r3), f5aa2d4 (r4) and 371c0de (r5), and r5 is missing from the table. The
   prior-art doc gives AUROC "0.994–0.9998" against the report's 0.9999.
6. **[minor] Effect sizes and power.** For transparency, add a minimum-detectable-effect statement for 15 units
   / 10 videos (unit SD of A1 dJF ≈ 0.016–0.02, so CI half-width ≈ ±0.01). Gate decisions near +0.02–0.03 are not
   separable on this set.

## Checks I ran (reproducible, CPU)

- Gate numbers: `trackA_{A1,A1z,A1v,A2x}_per_unit.csv` filtered to set ∈ {DAVIS_val, PUMaVOS_B}, `small`.
- Oracles: per-frame max over `runs/{B1,A1,A1z,A1v,A2x}/*.csv` JF on gate small units. Missing frames (no crop)
  were filled with B1.
- Video vs image oracle: `cscale_frames.csv` ⨝ `runs/B1` on (unit, video_idx), giving 419 frames.
- Presence: `runs/B1`, guitar excluded, within-unit Mann-Whitney AUROC.
- M5.6: `outputs/milestone56/reports/diagnose_episodes.csv`, `J_I0_native ≥ 0.5`.
