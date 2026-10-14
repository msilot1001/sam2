# M7 independent review (raw)

Reviewer scope: docs/m7/*.md, scripts/m7.py, scripts/m7_replay.py, src/spatial/{sggl,policies,evidence,mask_fusion,feature_fusion,target_attention}.py, tests/test_m7.py, outputs/milestone7/** (raw per-frame CSVs, prereg JSONs, reports, heads, viz). CPU only. All headline numbers were recomputed from the per-frame CSVs (rec/, rec_replay/, rec_replay_combos/, rec_vk/, runs/<arm>/, teval/). Report JSONs were used only for comparison. Recompute scripts are in the session scratchpad and were not added to the repo.

**Verdict in one line:** no causality or leakage problem was found, and the numbers reproduce. KEEP_SGGL is a defensible outcome of the frozen rule. Several claims are worded more strongly than the evidence allows: VK "replication / new videos", the M3-vel margin, R1 "cause found", and "false re-acquisition = appearance ambiguity". There are also undisclosed or under-disclosed protocol deviations and reproducibility gaps. No HIGH-severity defect.

---

## 1. Recomputation from raw per-frame CSVs

Method: per-unit mean J&F over frames 1..n-1. Small = `area0 < 0.01` from `replay_per_unit.csv`. Video-clustered percentile bootstrap, 2000 resamples, seed 0. VK uses a unit-clustered bootstrap, as in the preregistration.

### 1.1 Counts

| Item | Recomputed | Docs | Match |
|---|---|---|---|
| Crop units, DEV+REP | 64 units / 39 videos; frames DEV 2182, REP 2900 | same | OK |
| Small units | 57 / 34 videos (DEV 16 / 12, REP 41 / 22) | same | OK |
| Tiny units | 7 | same | OK |
| VK | 28 units, 2 scenes, 1736 frames. area0 is 0.10–0.34 %, so all small and **0 tiny** | same | OK |
| teval DEV small | 16 units / 12 videos | same | OK |
| M3-vel-r PROVENANCE | 40 ubuntu-rtx2080 + 24 vast-rtx5090. Matches `m3r_per_unit.csv` `vast` flags (0 mismatches) | | OK |

### 1.2 Headline deltas (small units)

| Comparison | Recomputed mean [video CI] | LOVO min / max | Without miami | Docs |
|---|---|---|---|---|
| SGGL − P5, DEV+REP (Ubuntu rec) | **+0.0112 [−0.009, +0.027]**; 23 videos up / 2 down | +0.0089 / +0.0218 | +0.0218 | +0.011 [−0.010, +0.027] OK |
| SGGL − P5, Vast runs/SGGL | +0.0112 [−0.009, +0.027] | | | (consistent) |
| M3-vel − SGGL (Vast, matched) | **+0.00456 [−0.0001, +0.0110]**; 2 / 1 videos | +0.0017 / +0.0050 | +0.0045 | OK |
| M3-vel-r − SGGL (machine-matched per PROVENANCE) | **+0.00444 [+0.0000, +0.0111]**; 2 / 1 | +0.0015 / +0.0048 | +0.0046 | OK |
| F1-blend.7 causal − SGGL | +0.0043 [−0.007, +0.020] | | −0.0046 | OK |
| M1-always causal − SGGL | +0.0003 [−0.002, +0.003] | | −0.0009 | OK |
| VK SGGL − P5 | **+0.0173**, unit CI [+0.009, +0.025]; Scene01 +0.0165 / Scene06 +0.0204; 22 up / 1 down | | | OK |
| VK M3-vel − SGGL | +0.0013 [−0.0001, +0.0029]; 1 up / 0 down | | | OK |
| VK SGGL-V − SGGL | −0.00002 | | | OK |
| VK J: SGGL − B0 | +0.034 | | | OK |
| VK J: P5 − B0 | −0.0011 | | | OK |
| T0 − SGGL (DEV teval) | −0.0127 | | | OK |
| T1 − SGGL | −0.0134 | | | OK |
| T2 − SGGL | −0.0013 | | | OK |
| T3 − SGGL | −0.0008 | | | OK |
| T4 − SGGL | −0.0081 | | | OK |
| T contrasts | T1−T0 −0.0007; T2−T1 +0.0121 [+0.0025, +0.029]; T3−T2 +0.0005; T4−T3 −0.0073 | | | OK |

Additional checks on the same data:
- **T3 − T1** = +0.0126 [+0.003, +0.031], which supports the docs' "T2 and T3 meet the attention condition".
- **T4 − T1** = +0.0054 [−0.0005, +0.014], so T4 does not meet it (also as the docs state).
- **Replay table (STAGE1 §2):** every point estimate, up/down count, LOVO range and worst-unit value reproduces from `rec_replay/*.csv`.
- **Combos table (K1–K5):** reproduces from `rec_replay_combos/*.csv`.
- **Oracles:** sel +0.018, sel5 +0.033, fuse +0.020, comp +0.024 all reproduce.
- **M1 2×2:** 3932 / 361 / 26 / 211 reproduces.
- **R1 rows:** 46 / 4.3 %, 130 / 2.3 %, 390 / 0 % reproduce.
- **Event fraction:** 4.76 % reproduces.

### 1.3 Machine matching and P5 checks

- The P5 (G) column of every causal arm equals the P5 of its paired baseline exactly (max diff 0).
- For M3-vel-r, pairing each unit against the opposite machine's P5 gives a max diff of 8e-5. This confirms the PROVENANCE assignment is real, not nominal.
- Cross-machine SGGL difference: max 0.0017 (queen_car obj1); 3 units exceed 1e-4. This matches `MIGRATION_RTX5090.md`.
- M3-vel against the Ubuntu rec SGGL instead of the Vast SGGL gives +0.00458. The conclusion does not depend on the baseline machine.

### 1.4 Bootstrap fragility of the decisive bounds (MEDIUM, see §4)

| Comparison | Seed-0 lower bound | Lower bound over 10 seeds (n = 2000) | Seeds with LB > 0 | n = 20000 LB | P(boot mean ≤ 0) |
|---|---|---|---|---|---|
| M3-vel | −0.00008 | −0.00025 … +0.00002 | 1 / 10 | −0.00012 | 0.032 |
| M3-vel-r | −0.00005 | −0.00015 … +0.00003 | 2 / 10 | −0.00008 | 0.031 |

- The pass/fail of the final rule ("video-CI lower bound > 0") for M3-vel and M3-vel-r is decided at Monte Carlo noise level.
- The docs print this bound as "−0.000" (FINAL_ALGORITHM_DECISION.md:8 and :29; STAGE1_EXPERIMENTS.md:65; STAGE2_REVISIONS.md:47).
- M3-vel without its two best videos (skate-park, sheep): +0.0006. Unit sign count at ±0.001: 19 up / 14 down. The effect is concentrated, not broad.
- SGGL vs P5 on DEV+REP, by contrast, is broad (50 units up / 6 down at ±0.001). Its CI includes 0 solely because of miami-surf obj5 (−0.477). P(boot mean ≤ 0) = 0.13.

---

## 2. Causality and leakage

**No leakage found.**

- **Policies** (`src/spatial/policies.py`, `evidence.py`) read only the observation dict `o` (model logits, scores, crop box, frame-0-reference similarity) and their own `Hist`. `Hist` is seeded only with the frame-0 prompt box (policies.py:34-37).
- **GT handling:** in `run_policy`, `teval`, `record_unit` and `replay_unit`, GT is loaded only after the policy decision and is used only for metrics (m7.py:358-367, 614-626, 270; m7_replay.py:73-87). Oracles, taxonomy, the component oracle and pixel accounting use GT but are labelled bounds or diagnostics.
- **M3 / M3-r centre** (policies.py:244-282) uses only the current output and earlier outputs or acceptances.
- **T training:**
  - `tcache` asserts REP only (m7.py:440).
  - Fit and val splits are video-disjoint: 18 / 5 videos with overlap 0 in every `heads/T*.json`.
  - Val videos (car-turn, hockey, sheep, stunt, surf) are DAVIS train. No DEV, VK or C labels are used.
  - The final-step checkpoint is used with no val-based selection, so val does not leak into the reported head.
  - Positive-control items are drawn from fit.
- **Probe isolation:**
  - `Dual.probe` is non-storing and restores the feature cache (sggl.py:119-138).
  - `test_recording_rollout_reproduces_sggl_probes_are_isolated_and_causal` compares probes on vs off and checks the n = 5 vs n = 9 prefix.
  - Caveat (LOW): both arms of that test run with the L2 state on. The "L2 state does not perturb G/L" property is covered only by the 8-frame match against the M6 A2x reference on one unit (tests/test_m7.py:58-60).
  - Indirect evidence: on VK, runs/SGGL-V (run path, no L2 or probes) equals the rec-path SGGL on every frame of 27/28 units. The remaining unit differs only on the 5 veto frames (§5).
- **Unit selection** uses only frame-0 GT (`crop_side(gt(u, 0))`), which is the allowed prompt.

---

## 3. Pre-registration compliance

| # | Item | Finding | Severity |
|---|---|---|---|
| P1 | S2 → M3-vel in the causal pilots | `preregistration_r2_amend.json` is frozen at 11:45. The first causal output appears at 12:19. Procedurally OK. But the narrative docs never state that the pre-registered S-family causal pilot (S2-observe) was dropped and replaced by a new, post-audit candidate: STAGE1_EXPERIMENTS.md:5 only says "…과 amend", and §3 lists the three arms silently. Result: **the S family never received a causal check**. The amendment's rationale ("replay-identical to SGGL-V on every unit") is also not exactly true (P4). | MEDIUM |
| P2 | M3-vel-r machine split | `preregistration_r3.json` states "run: Ubuntu RTX 2080". The actual run was 40 Ubuntu units (13:50–14:18) plus 24 Vast units (14:12–14:15). The Vast tail started after Ubuntu outputs existed. STAGE2_REVISIONS.md:42 reports the split as fact, not as a deviation, with no amendment or rationale. Impact on numbers is nil (matched baselines, P5 diff 0). | LOW–MEDIUM |
| P3 | VK run composition and timing | r4 frozen 13:56:11; first VK output 13:57:39. Arms match r4: M3-vel, SGGL-V, and SGGL+P5+B0 via `rec VK`. M3-vel-r was excluded before its own result existed, which is consistent with "only if it passes". Compliant. | OK |
| P4 | S2 == SGGL-V claim | STAGE1_EXPERIMENTS.md:56 says "모든 객체에서 같은 값". In fact they differ on 1 frame (pumavos smile obj1, t = 222, unit Δ 0.0021). The doc's own table shows +0.0070 vs +0.0069. | LOW |
| P5 | Replay outputs predating the freeze | 11 `rec_replay` CSVs (DEV units) predate the policy-freeze commit 274e4d7 (10:14:35); they come from the pipeline test mentioned in the commit message. `replay_unit` skips existing files, so they were reused. **I re-ran these 11 units with the frozen code: bit-identical (max diff 0).** No impact; disclose in STAGE1. | LOW |
| P6 | Stage 3 combinations | No preregistration file exists. The code is uncommitted (`git diff src/spatial/policies.py`, `scripts/m7_replay.py`). The doc admits the stage was entered without justification. All four are REJECT, so harmless, but they must be labelled post-hoc exploratory and committed. | LOW |
| P7 | Final rule application | The frozen rule was applied literally. No candidate has video-CI LB > 0. Given §1.4, M3-vel's fail is a bootstrap-noise call. That still counts as a fail under the rule as written, but must be disclosed. | see §4 |
| P8 | `evidence.py:4` | Says thresholds are "pre-registered in scripts/m7.py PREREG_S1". No such symbol exists. The thresholds are frozen by commit 274e4d7, not by a prereg entry. | LOW |
| P9 | Pilot vs final status | F1-blend (0.5) and M1-trusted are INCONCLUSIVE under the pilot rule (STAGE1 §2) but appear as REJECT in the FINAL 18-row table (F1 and M1 rows). This reclassification is reviewer judgment and should be stated. | LOW |

---

## 4. Is KEEP_SGGL supported? Are claims worded at the right strength?

### Decision

ADVANCE_OTHER_CANDIDATE for M3-vel is **not** supported:
- It fails the pilot rule (2 videos up, fewer than 3).
- It fails the final rule's CI lower bound, albeit marginally.
- Its gain is concentrated in skate-park and sheep.

Between KEEP_SGGL and INSUFFICIENT_EVIDENCE, the prereg gives no criterion to choose (preregistration.json `final_rule`: "otherwise KEEP_SGGL or INSUFFICIENT_EVIDENCE"). My recommendation:

- **Keep KEEP_SGGL as the operational decision.** No replacement met the rule. M3-vel costs zero extra calls, so nothing is lost by deferring it.
- **State explicitly that the M3-vel vs SGGL question is unresolved** (INSUFFICIENT_EVIDENCE for that contrast). M3-vel:
  - is positive on both DEV+REP (LOVO min > 0, without-miami > 0) and VK (1 up / 0 down);
  - fails the decisive bound by about 1e-4, inside bootstrap noise;
  - should be named as the single pre-specified challenger for the next confirmation (NEXT_MILESTONE_ROADMAP M71-03 already does this implicitly).
- **Do not present KEEP_SGGL as "SGGL validated".** On the development data, SGGL − P5 has CI [−0.009, +0.027].

### Claim wording

| Claim | Location | Issue | Suggested wording | Severity |
|---|---|---|---|---|
| "SGGL replicated on VK, +0.017 [+0.009, +0.025]" (bold) | FINAL_ALGORITHM_DECISION.md:9, STAGE2_REVISIONS.md:78 | (a) The CI is clustered by unit, but all units of a scene share the same video. The effective n is 2 scenes, so the bracket overstates precision; r4 itself calls it "descriptive". (b) **VK Scene01/06 are not new videos for the project.** They have been the project's development scenes since Milestone 2 (docs/MILESTONE_2.md:5, 66-73), and several M7 VK units are the same tracks (Scene01 54/63/86/90, Scene06 12/13). "SGGL에는 처음" is true. "새 영상" (FINAL:16) is false. Risk to SGGL is low (SGGL was never tuned on VK), but the claim must be qualified. (c) Synthetic, cars only, 0 tiny units. | "On VK (2 synthetic scenes previously used in M2–M4 with official SAM 2, never with SGGL), SGGL − P5 = +0.017; both scenes positive (+0.017 / +0.020), 22 / 1 objects; unit-level CI shown for description only." | MEDIUM |
| "M3-vel … +0.0046 [−0.000, +0.012]" | FINAL:8, :29; STAGE1:65; STAGE2:47; TRACK_B:75 | Hides that the lower bound is −0.0001 and that its sign flips with the bootstrap seed (§1.4). | Print 4 decimals and add "fails the final-rule bound at Monte Carlo noise level (P(boot ≤ 0) ≈ 0.03)". | MEDIUM |
| "G misses small targets because of feature-grid occupancy, not pixel detail" / "원인 발견" | PRIOR_ART_AND_NOVELTY.md:55; FINAL:54 (R1 row) | R1 degrades only L's input and measures whether L **keeps** its candidate (n = 46 correct candidates at event frames, 16 units). It never intervenes on G, for example by increasing G's cell occupancy. It also leaves L's field of view and memory unchanged; the audit itself lists these as confounds (OPPORTUNITY_AND_FAILURE_AUDIT.md:104, "…시야·메모리 차이일 가능성"). The probe supports "pixel detail at G's density is *sufficient* for L (with L's framing and memory)". It does not support "grid occupancy is *the cause* of G's misses". | "Consistent with grid occupancy / field-of-view / memory rather than pixel detail; G-side intervention not done." | MEDIUM |
| "False re-acquisition is appearance ambiguity, not memory corruption" | PRIOR_ART_AND_NOVELTY.md:54; COMBINATION_EXPERIMENTS.md:32; TRACK_B:47 | Rests on 2 episodes in **1 video** (miami-surf obj5; `reacq_episodes.csv`). As a general mechanism claim this is n = 1. | "In the single observed case (miami-surf obj5)…" | MEDIUM |
| "SGGL-V veto did not fire on VK" | FINAL:11 | It fired on 5 frames of 1 unit (Scene01 obj91; `rec_vk` a2x & ~a2v; unit Δ −0.0005). STAGE2:79 ("사실상") is correct. | "fired on 5 frames / 1 object (Δ −0.0005)" | LOW |
| "32종" prototypes | FINAL:6 (20+3+1+5+4 = **33**), STAGE1:18 (= 32) | The two docs disagree with each other and FINAL's own sum. The "20 replay" count includes SGGL itself and SGGL-V (controls). F1-blend.7 and M1-always are counted twice (replay and causal). T0 is a control. Distinct new candidates ≈ 18 replay + M3-vel + M3-vel-r + T0–T4 + 4 K = 29. | Give one consistent count with the definition stated. | LOW |
| STAGE2 §1 vs §3 | STAGE2_REVISIONS.md §1 attributes losses to "G/L 출처 전환"; §3 concludes losses are not explained by source switching | Internally contradictory. §3 is supported by the data (switch counts: lab-coat obj2 M3-vel 6 vs M3-vel-r 4, same loss). Revise §1 to "hypothesis (refuted in §3)". | | LOW |

---

## 5. Cross-document consistency and cross-machine handling

- **Cross-machine handling is good.** Every causal comparison is paired with a same-machine baseline, and the P5 identity checks pass exactly (§1.3). One weak spot: the M3-vel-r vs M3-vel target-regression comparison (STAGE2 table) is cross-machine for lab-coat obj2 and lady-running obj2. The differences (≈0.003) are only about 2× the per-unit cross-machine SGGL noise (≤ 0.0017). The double-difference I computed, (M3r − SGGL_matched) − (M3 − SGGL_vast), gives the same conclusion (mean −0.00012), so the outcome is unaffected. Note the caveat in the table. LOW.
- **Mixed modes in FINAL §1.** The §1 comparison table mixes **replay** numbers (SGGL-V +0.0069, and the S/F/M1 rows in §2) with **causal** numbers (M3-vel, M3-vel-r, F1-blend.7 causal). SGGL-V changes the selection, so in a live run it also changes the crop trajectory. Its replay value is an approximation; replay ≈ causal was verified only for F1-blend.7 and M1-always. Add a "mode" column. LOW–MEDIUM.
- **CI rounding** differs between STAGE1 and TRACK_A for the same report values:
  - F1-blend.7 upper: report 0.0235; STAGE1 +0.023, TRACK_A +0.024.
  - F-inter lower: report −0.0135; STAGE1 −0.013, TRACK_A −0.014.
  - F1-blend lower: report −0.0115; STAGE1 −0.011, TRACK_A −0.012.
  - LOW.
- **Stale statements:**
  - MIGRATION_RTX5090.md:81 ("viz: only 2 figures"); 8 figures exist.
  - MIGRATION_RTX5090.md:28 (tcache "전송 중").
  - MIGRATION_RTX5090.md:55 ("8 tests"); there are now 9, 7 pass on CPU and 2 skip.
  - `EXECUTION_STATUS.json`: CAUSAL-PILOTS "running", `next_scheduled` T PC, no VK / M3-vel-r / combos entries.
  - LOW.
- **Provenance gap:** MIGRATION rule 3 says every Vast output directory gets an `ENV_vast.json`. Only `runs/ENV_vast.json` and `heads/ENV_vast.json` exist, both recording commit **4c63eb1**. That commit predates `M3VelocityR` (added in 6077884), yet the Vast M3-vel-r tail and the VK runs were produced on Vast. Their code commit is unrecorded. `rec_vk/` and `teval/` have no ENV file. LOW–MEDIUM.

---

## 6. Completeness against the M7 task

**18-candidate table (FINAL §2).**
- All 18 IDs are present, along with K and O1.
- Status vocabulary does not match the requested labels. "REJECT" is used instead of **REJECTED**. **NOT_IMPLEMENTED** is never used. **ADVANCE** has 0 uses, which is legitimate.
- **M2 is labelled IMPLEMENTED_NOT_EVALUATED, but no M2 policy exists.** `Dual.excluded` and `planners.backfill` are plumbing; no policy or arm ever sets `excluded` (grep across scripts/m7.py and src/spatial/policies.py). It should be **NOT_IMPLEMENTED** (infrastructure only). MEDIUM (label accuracy in the required table).
- **Test column overstates coverage.** It cites "재생 동등성" for S1–S4 and "기계 간 동일" for M3, but these are report checks, not unit tests. There is **no unit test for the M3-vel / M3-vel-r centre logic (step cap, source-consistency), the S-family `Hist` / `identity`, or the K combos.** Add a small test for M3VelocityR (cap and source conditions). LOW–MEDIUM.

**Visual evidence.**
- `outputs/milestone7/viz/` has 8 stage-0 figures (miami-surf, dog-gooses, drone, kite-surf, skate-park, lab-coat). They are informative (G / L / L2 / Lp / SGGL / SGGL-V / F1 / GT panels).
- Missing: figures for the **decisive** evidence:
  - M3-vel causal gain (skate-park obj2, sheep obj4) and loss (lab-coat obj2);
  - any VK example;
  - any T-head output.
- Only TRACK_A references viz; FINAL does not. MEDIUM.

**Tests.** `tests/test_m7.py`: 7 passed, 2 skipped (CUDA) on CPU in this review.

**Reproducibility.**
- The code that produced `causal_pilots_report.json`, `teval_report.json`, `m3r_report.json` / `m3r_per_unit.csv`, `vk_report.json` / `vk_per_unit.csv` and `runs/M3-vel-r/PROVENANCE.json` is **not in any script** (grep finds no producer).
- Key evidence is git-ignored and uncommitted, while other runs were force-added:
  - `runs/M3-vel-r/`, `runs/SGGL-V/`, the 28 VK files in `runs/M3-vel/`, `rec_vk/`;
  - `vk_report.json`, `m3r_report.json`, `combos_report.json`;
  - the combos code and the 8 new docs.
- The final decision's VK and stage-2 evidence therefore exists only on the Ubuntu disk. I reproduced everything independently, so this is a durability and auditability gap, not a correctness one. MEDIUM.

**Missed opportunity (advisory).** VKITTI test scenes 02 / 18 / 20 were never opened (MILESTONE_2.md:5). They are a ready, approval-free (already on disk) confirmation set for SGGL and M3-vel. The roadmap names only new external data plus PUMaVOS C. Consider adding them, subject to the user's data rules.

---

## Verdict

- **Numbers:** all headline values reproduce from raw CSVs.
- **Causality / leakage:** clean.
- **Machine matching:** done correctly.
- **KEEP_SGGL:** acceptable as the outcome of the frozen rule, and ADVANCE_OTHER_CANDIDATE is correctly not taken. The write-up should present it as "incumbent retained; M3-vel unresolved (fails the bound at noise level)", not as SGGL being validated.
- **Wording:** the VK "replication on new videos" framing and two mechanism claims (R1 cause, appearance ambiguity) need softening.
- **Protocol / reproducibility:** protocol deviations (S2 substitution, M3-vel-r machine split) and reproducibility gaps (uncommitted analysis code and evidence) need disclosure and fixing.

## Prioritized required corrections

1. **(MEDIUM)** FINAL:9, :16 and STAGE2:78: drop "새 영상" and "재현됐다 [+0.009, +0.025]" as stated. Say VK Scene01/06 were M2–M4 development scenes (fresh only for SGGL), effective n = 2 scenes, both positive, CI descriptive.
2. **(MEDIUM)** Everywhere M3-vel and M3-vel-r CIs appear (FINAL:8, :29; STAGE1:65; STAGE2:47; TRACK_B:75): print the lower bound to 4 decimals (−0.0001 / −0.0001), state that it fails the final rule at bootstrap-noise level, and add an explicit "M3-vel vs SGGL: INSUFFICIENT_EVIDENCE; designated challenger for confirmation" line under FINAL_DECISION.
3. **(MEDIUM)** Commit a script that regenerates `causal_pilots_report`, `teval_report`, `m3r_report` / `m3r_per_unit`, `vk_report` / `vk_per_unit` and `PROVENANCE.json` from raw CSVs. Force-add the decision-relevant outputs: runs/M3-vel-r, runs/SGGL-V, the VK files in runs/M3-vel, rec_vk CSVs, and the reports. Commit the combos code.
4. **(MEDIUM)** STAGE1 §3: disclose that the pre-registered S2-observe causal pilot was replaced by the post-audit M3-vel (cite `preregistration_r2_amend.json`, 11:45 < first output 12:19), and that the S family has no causal confirmation. Correct the "identical on every unit" claim (STAGE1:56 and the amendment text: 1 frame differs).
5. **(MEDIUM)** Soften PRIOR_ART_AND_NOVELTY.md:54-55 and the FINAL R1 row: "consistent with feature-grid occupancy / FOV / memory; no G-side intervention", and "in the single observed case (1 video)".
6. **(MEDIUM)** 18-candidate table: M2 → NOT_IMPLEMENTED (only the `Dual.excluded` hook exists); use REJECTED; note that the F1 and M1 rows override the pilot verdict INCONCLUSIVE for F1-blend and M1-trusted; add a replay / causal mode column to FINAL §1–§2.
7. **(MEDIUM)** Add visual evidence for M3-vel (skate-park obj2 / sheep obj4 gain, lab-coat obj2 loss) and one VK example, and reference viz from FINAL.
8. **(LOW–MEDIUM)** STAGE2 §3: record the M3-vel-r Ubuntu/Vast split as a deviation from `preregistration_r3.json` with a rationale. Write ENV files with the actual commit for the Vast M3-vel-r tail, VK, rec_vk and teval.
9. **(LOW)** Fix the prototype count (FINAL:6 says 32, but its own terms sum to 33; STAGE1:18 says 32; define what is counted). Fix "VK에서는 발동하지 않았다" (FINAL:11). Harmonize CI rounding between STAGE1 and TRACK_A. Resolve the STAGE2 §1 vs §3 source-switch contradiction. Fix `evidence.py:4` (PREREG_S1 does not exist). Disclose the 11 pre-freeze replay files (verified identical). Update the stale MIGRATION §2/§4 and EXECUTION_STATUS.json.
10. **(LOW)** Add unit tests for `M3VelocityR.center` (cap and both-accepted condition) and one for `evidence.Hist` / `identity`. Optionally run the isolation test with `scale2=None` vs `2.0` to cover the L2 state.
