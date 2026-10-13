# M6.1 second independent review (SGGL-V / A2v)

Scope: the recommendation as it changed after REVIEW_raw.md (SGGL-V = A2x self gate + global score > 0 veto, KEEP_GLOBAL_LOCAL, PROVISIONAL). CPU only. No existing files modified.
Recomputed from `trackA/a2v_dev_per_unit.csv`, `replication/a2v_per_unit.csv` and the per-frame `replication/A2v/*.csv` and `trackA/A2v/*.csv`. Bootstrap: video-clustered, 5000 resamples, seed 0.

## 1. Numbers recomputed

| Quantity | Docs | Recomputed | OK? |
|---|---|---|---|
| Dev A2v small (sum/15) | +0.0168 | +0.0168 | yes |
| Dev A2v − A2x | −0.0009 | −0.0009 (kite-surf obj2 −0.0132, smile −0.0017; every other unit 0) | yes |
| Dev NI CI (13 applied) | [−0.004, 0.0] | [−0.0036, 0.0] from json | yes. Upper bound = 0: **the veto never helps on dev** |
| Rep A2v small (41 / 22) | +0.020 [+0.011, +0.029] | +0.0195 [+0.0116, +0.0289] | yes |
| Rep A2v − A2d | +0.010 [−0.005, +0.032] | +0.0102 [−0.0046, +0.0320] | yes |
| Rep up/down (objects, ±0.01) | 26 / 2 | 26 / 2 | yes |
| Rep videos up/down | "17 ↑ / 1 ↓" | 17 / 1 at ±0.005; 21 / 1 at >0; 15 / 0 at ±0.01 | yes, but FINAL does not state the threshold |
| Rep leave-one-video-out | +0.018 to +0.022 | +0.0175 to +0.0224 | yes |
| miami-surf obj5 | −0.477 → +0.001 | −0.4773 → +0.0010 | yes |
| Hurt by veto (< −0.01 vs A2x) | dog-gooses obj5, lady-running obj2, scooter-board obj2, kite-surf obj2 | same 4 | complete at that threshold |

**Not reported anywhere (new finding):** with the motivating unit (miami-surf obj5) removed, the veto is net slightly negative on replication:

- A2v − A2d on the other 40 small units: **−0.0015 [−0.0048, +0.0011]**. Helped 2 (miami-surf obj4 +0.015, sheep obj4 +0.017), hurt 3 (sum −0.092 vs +0.032).
- A2d alone on those 40 units: +0.0215. A2v on those 40: +0.0200.
- So the +0.020 [+0.011, +0.029] headline is "A2x without its one catastrophe". The only evidence that the veto helps is n = 1, and that unit is the one it was designed on. Everywhere else (dev, and replication minus one unit) its point estimate is ≤ 0.

Veto trigger frequency (frames with score_G ≤ 0 and score_L > 0): only 8 of 64 crop units ever trigger it, mostly 1–6 frames; miami-surf obj5 has 31. **None of the 7 tiny (< 0.1%) units, dev or replication, triggers it.** So there is no evidence either way on the target population. The hurt units are 0.15–0.7% area ("small", not "극소" as the report's wording says).

## 2. Causality / leakage: PASS

`scripts/m61.py:run_dual`: `sG` comes from `engine.predict(pred, stG, odG, t)` on the online global state at frame t. G and L share no memory in A2v (write-back only for A3/A3s). GT (`m6.gt(u,t)`) is used only for metrics after `out` is decided. The crop center for t+1 follows `out`, which is causal. A side effect: the veto also moves the crop center, because the next center follows `mG` when vetoed. SPEC pseudocode shows this (`c = box_center(out)`), but the docs never describe it as part of the mechanism. Severity: info.

## 3. Post-hoc promotion and labelling

- **[HIGH] Contradictory labelling in FINAL.** The evidence-table row in FINAL_RESEARCH_DIRECTION.md reads "**A2v 복제 (DAVIS train, 설계에 미사용)**" ("not used in design"). That is false for A2v: the veto was designed from that set. The same document's "Evidence contradicting" section, SPEC, REPORT §10 and COMPARISON §5 all say correctly that it is post-hoc. Fix the row label to something like "사후, 설계에 사용됨" (post-hoc, used in design).
- **[HIGH] Missing disclosure.** Report the result excluding miami-surf obj5 (−0.0015 [−0.005, +0.001]) next to the +0.020. Without it a reader will read +0.020 vs +0.009 as a general veto gain.
- **[MEDIUM] Decision.** KEEP_GLOBAL_LOCAL is defensible as a *direction* decision: A2x − F2 and A2x − CTU are positive on dev, and replication is positive at the video level except for one failure mode. But two things fail together: A2x's own pre-registered replication rule failed, and the veto's benefit rests on n = 1 post-hoc. Promoting SGGL-V over SGGL as *the* main algorithm is therefore not supported by data. It is a reasonable safety prior. Recommendation: keep KEEP_GLOBAL_LOCAL + PROVISIONAL. In M7, state SGGL and SGGL-V as co-primary arms, with the main-algorithm choice decided by the M7 gate rather than now. The roadmap already runs both arms, so this is a wording change. PROVISIONAL_ONLY would also be defensible. I would not call KEEP_GLOBAL_LOCAL wrong.
- **[MEDIUM] Non-inferiority margin.** NI was pre-registered as a point estimate (≥ −0.003). It passes. The applied-unit CI lower bound (−0.0036) crosses the margin, so a CI-based NI test would not pass. Say so in one clause.

## 4. Costs: complete at the stated threshold, two gaps

- The 4 listed units are complete at < −0.01. Smile obj1 (−0.0017) is the only other negative and is negligible.
- **[MEDIUM] No cost-side kill condition in M7.** FINAL §2 condition 4 tests only the benefit side (first 5 frames after reappearance). The premise of the work is that the global view is unreliable on tiny targets (B0 is −0.24 on tiny objects), and the veto hands presence authority to exactly that view. Add a pre-registered cost check: frames with GT present, score_G ≤ 0 and an accepted-by-self-gate crop, plus SGGL-V − SGGL on tiny objects ≥ −0.005. Without it the veto could silently remove the tiny-target gain on the new set.
- [LOW] The FINAL finalist table's "최악 객체 −0.056 (vs A2x)" mixes reference points with A2x's −0.477 (vs B1). The worst A2v unit vs B1 is motocross-bumps obj1 −0.044 (replication) and birdfall −0.023 (dev).

## 5. Cross-document consistency

- [LOW] ROADMAP "출발점": `run_dual` arms are listed as "A2d / A3 / A3s"; A2v is missing (SPEC lists it).
- [LOW] ROADMAP says "개발 세트에 극소 객체 4개, 영상 3개". Dev tiny is 3 objects / 2 videos (COMPARISON §1, FINAL); 4 / 3 is the replication tiny count.
- [LOW] `outputs/milestone61/EXECUTION_STATUS.json` is stale. It has `strongest_candidate: "SGGL (A2x); reliability veto under test"`, and its `running` list holds items that are already in `completed`.
- [LOW] REPORT §10 / COMPARISON §5 call the hurt cases "극소 대상" (tiny targets), but they are 0.15–0.7% area objects.
- [INFO] FINAL video up/down "17 ↑ / 1 ↓" should name the ±0.005 threshold (REPORT §9 does).
- Consistent: the SPEC pseudocode gate matches the code. TASK-M7-004 (b) requires reproducing the logged A2v. M7-006 includes both SGGL and SGGL-V.

## Verdict

The numbers are correct and the rule is leak-free. Honest-labelling is mostly good, with one contradictory label in FINAL. The key omission is that the veto's whole measured benefit is the single unit it was built on, while elsewhere it costs about 0.001–0.002 and is untested on tiny targets. KEEP_GLOBAL_LOCAL / PROVISIONAL can stand if SGGL-V is framed as a co-primary hypothesis, not a validated improvement, and M7 adds a cost-side veto condition.
