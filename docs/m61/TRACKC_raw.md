# M6.1 Track C (raw): temporal error analysis of B1 (retrospective, GT-based, diagnosis only)

Code: `outputs/milestone61/trackC/trackc.py` (events) and `summarize.py`. CPU only. Outputs: `events.csv` (one row per event, with `frames` list), `carryover_oracle.csv`, `drift_per_unit.csv`, `frames.csv`, `billie_shoes_switch.png`.

**Denominator:** B1 J-error on GT-visible frames in DAVIS_val + PUMaVOS_A + PUMaVOS_B is sum(1-J) = **1112.3** (8040 frames). Shares below use this denominator.

**Artifact exclusion:** pimples t >= 74 is excluded everywhere. Guitar GT-empty frames and the guitar `annotation_defined` reappearance are also excluded. Guitar visible frames are kept.

**Transient vs persistent:** an event is transient if it resolves within 3 frames or fewer.

**Error mass:** the default is the frame-level sum(1-J) over the event's frames. It is an upper bound because it charges the whole frame's error to the event. For (a), the component-attributable mass is also reported: the J gain from adding back only the lost GT component.

SegTrack contributed no events in categories a–e.

## Definitions
- **a. Component loss:** a GT component of at least 50 px at t overlaps a component of at least 50 px at t-1. That t-1 component was at least 50% covered by the prediction, and the t component is less than 10% covered. The event follows the component until it is at least 50% covered again or leaves the GT.
- **b. Collapse:** pred area at t is less than 0.5x pred area at t-1, |GT area ratio - 1| < 0.2, and J_t < J_{t-1}. The event lasts until J returns to within 0.1 of J_{t-1}.
- **c. Wrong object:** other_obj_frac >= 0.3. This only sees *annotated* other objects.
- **c2 (added):** at least 30% of predicted pixels lie in connected components that do not touch the GT. This catches capture of *unannotated* regions, which c misses.
- **d. False presence:** GT is empty and the prediction is non-empty.
- **e. Failed reappearance:** taken from `reappear_events.csv`. Failed means J_first < 0.5; it is transient if the delay to J >= 0.5 is 3 frames or fewer.
- **f. Drift:** for units with at least 10 visible frames in each window, mean J over frames 1–20 minus mean J over frames 21+ is greater than 0.1.

## 1. Events (all from the 3 main sets; SegTrack adds only one drift unit)

| cat | events (transient/persistent) | units | videos | frames | err share (frame-level) | notes / examples |
|---|---|---|---|---|---|---|
| a component loss | 125 (108/17) | 29 | 20 (18 DAVIS) | 390 | 14.4%; **component-attributable 4.4%** | median lost component 110 px; 57/125 end because the GT fragment disappears. billie_shoes 189–299 is 11.6% frame-level and 3.7% attributable. Excluding billie: 124 events, 19 videos, 4.4% frame-level, **0.7% attributable**. Examples: kite-surf obj3 29–49, bike-packing obj2 25–43, mbike-trick obj2 63–78 |
| b collapse | 2 (1/1) | 1 | 1 | 113 | 10.1% | billie_shoes only: 185–186, then 189–299 |
| c wrong object (annotated) | 7 (5/2) | 3 | 2 | 20 | 1.3% | lab-coat obj1 1–14, obj2 1–4 and 18–25; dogs-jump obj1 8–10 |
| c2 disjoint region | 9 (6/3) | 5 | 4 | 157 | 13.4% (2.9% excl. billie) | billie_shoes 170–178 and 188–299; pimples 49–73 (overlaps e); lab-coat 12–14 and 24; bmx-trees obj2 68 |
| d false presence | 4 (4/0) | 3 | 3 | 4 | 0% (visible); 4 empty frames | ice_cream 246 and 256, pimples 48, smile 119; all single-frame |
| e failed reappearance | 8 (3/5) | 4 | 3 | 76 | 6.6% (union 4.3%) | pimples 47–73 (two events); kite-surf obj2 15, 21–30, 39–42, 45–49; bmx-trees obj1 67–69, obj2 63 |
| f drift | 12 units | 12 | 9 | 1012 | late-frame error 32.6%; **excess below the early mean 21.7%** | billie_shoes (drop 0.42), india obj2 (0.38), kite-surf obj2 (0.51), paragliding-launch obj2 (0.26), guitar (0.15, which may be artifact-related), and others. Excluding guitar, billie and pimples: 6 videos |

**billie_shoes (checked visually, `billie_shoes_switch.png`):**
- The target is both white shoes.
- From about frame 170 the prediction also includes the orange shorts, a disjoint unannotated region.
- At frame 189 the prediction drops the shoes and stays on the shorts until the end, with J = 0 for 111 frames.
- Over that stretch obj_score stays above 5.9 and iou_pred is 0.80–0.97.
- This single unit accounts for 10.5% of all visible error.
- It is not an annotation artifact.

## 2. Previous-mask carry-over oracle (J_t < J_{t-1} - 0.2, zero-motion reuse of pred_{t-1})

| scope | events | mean gain | median | gain >= 0.15 | videos with gain >= 0.15 | gain < 0 |
|---|---|---|---|---|---|---|
| main sets, all | 38 | **-0.108** | -0.019 | 4 | 2 (billie_shoes x2, smile x2) | 19 |
| main sets, GT visible | 35 | -0.145 | -0.121 | 3 | 2 | 19 |
| all 4 sets | 47 | -0.112 | -0.086 | 4 | 2 | 26 |

- The largest gains are smile 119 (GT empty, +1.0), billie_shoes 189 (+0.41) and billie_shoes 185 (+0.33).
- Reusing the previous mask does not correct J drops in general. Most drops are real changes to the target, where the old mask is also wrong.

## 3. Conclusions per category (criterion: at least 4 independent non-annotation videos AND plausibly correctable online)
- **a:**
  - Frequent: 19–20 videos.
  - But the lost components are small fragments (median 110 px) and 86% are transient.
  - Error that can be attributed to them is 0.7% excluding billie (4.4% including it).
  - The carry-over oracle does not help.
  - Verdict: **common but negligible mass**.
- **b:** 1 video. **Not common.**
- **c:**
  - c (annotated other objects): 2 videos. **Not common.**
  - c2 (disjoint unannotated region): 4 videos, so it meets the threshold, but 2 of the 4 are single-frame or short lab-coat/bmx events. Mass outside billie and pimples is about 0.4% (4.6 of 1112).
  - Verdict: **borderline at best.** The billie_shoes switch is the one large online-detectable case: the shorts were included about 19 frames before the switch. It is one video.
- **d:** 4 single frames, 0 visible error. **Not worth it.**
- **e:** 3 videos (pimples, kite-surf, bmx-trees). **Below threshold.** This matches the M6 B3 decision.
- **f:**
  - Drift is a symptom and overlaps a, c2 and e (billie, pimples, kite-surf). It is not a mechanism.
  - The excess is 21.7% of error, or 8.4% excluding billie and guitar.
  - It cannot be judged correctable from these data.
- **Share of B1 visible J-error in temporal events (union of a, b, c, c2, e; main sets):**
  - **20.7%** (frame-level upper bound), from 22 videos.
  - **10.2% excluding billie_shoes.**
  - Transient events alone: 4.6%. Persistent events excluding billie: 6.2%.
  - Union of the two categories that meet the frequency criterion (a, c2): 17.8%, or **7.3% excluding billie**.
  - Component-attributable mass for a is much lower (0.7% excluding billie).
- No category meets both criteria with material mass beyond the single billie_shoes identity drift.
