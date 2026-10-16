"""Paper figure 6: example frames B0 / SGGL / GT, units and frames chosen by rule (never by hand).

    uv run python scripts/paper_fig6.py select     # -> docs/paper/fig6_selection.md + outputs/milestone10/fig6/rerun.txt
    M10_SAVE_MASKS=outputs/milestone10/fig6 uv run python scripts/m10.py rerun outputs/milestone10/fig6/rerun.txt   # 2080
    uv run python scripts/paper_fig6.py plot       # -> docs/paper/fig6_examples.png

Rules
- MOSEv2: tiny unit (area0 < 0.001) with the largest unit-mean SGGL - B0 J&F (paired frames, 4090 rows).
- VOST: unit with the largest (SGGL - B0) per-frame loss contribution of wrong_object frames
  (sum of 1 - J&F over wrong_object frames / frames, as amend9 Table 3).
- Frames: in the WORSE arm of that unit (MOSEv2: B0, VOST: SGGL), loss episodes = maximal runs of GT-present frames with
  J < 0.5 (m10_analysis.episodes); take the episode with the largest summed J&F gap (better - worse arm). Show
  the last GT-present frame with J >= 0.5 before the onset (the onset itself if none: the arm fails from frame 1),
  the episode's middle frame, and the first GT-present frame with J >= 0.5 after it (recovery) or the unit's last frame.
  (Revised once after the first draft picked a GT-absent frame as 'before'; disclosed in fig6_selection.md.)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("m10_analysis", Path(__file__).with_name("m10_analysis.py")))
s.loader.exec_module(A)

OUT = Path("outputs/milestone10/fig6")
DOC = Path("docs/paper")
PICK = {"mosev2": "B0", "vost": "SGGL"}                    # worse arm in which the episode is taken


def wrong_object_loss(g):
    st = A.state(g)
    return ((1 - g.JF) * (st == "wrong_object")).groupby(g.unit).sum() / g.groupby("unit").size()


def episode_frames(w, b):
    """w, b = worse / better arm rows of one unit -> (frames, episode dict)."""
    m = w[["video_idx", "J", "JF", "gt_empty"]].merge(b[["video_idx", "JF"]], on="video_idx", suffixes=("", "_b")).sort_values("video_idx")
    f = ((~m.gt_empty) & (m.J < A.TAU)).to_numpy()
    eps, i = [], 0
    while i < len(f):
        if f[i]:
            k = i
            while k < len(f) and f[k]:
                k += 1
            eps.append({"i": i, "k": k, "gap": float((m.JF_b.iloc[i:k] - m.JF.iloc[i:k]).sum())})
            i = k
        else:
            i += 1
    e = max(eps, key=lambda x: x["gap"])
    idx = m.video_idx.to_numpy()
    good = ((~m.gt_empty) & (m.J >= A.TAU)).to_numpy()            # GT present and tracked by the worse arm
    before = [i for i in range(e["i"]) if good[i]]
    after = [i for i in range(e["k"], len(idx)) if good[i]]
    frames = [int(idx[before[-1] if before else e["i"]]), int(idx[(e["i"] + e["k"] - 1) // 2]),
              int(idx[after[0] if after else len(idx) - 1])]
    return frames, {"onset": int(idx[e["i"]]), "dur": e["k"] - e["i"], "recovered": e["k"] < len(idx),
                    "gap_sum": e["gap"], "n_episodes": len(eps)}


def select():
    b0, sg = A.load("B0"), A.load("OR0")
    sg = sg[sg.unit.isin(set(b0.unit))]
    pu = A.paired(sg, b0)                                   # d = SGGL - B0 per unit
    tiny = pu[(pu.ds == "mosev2") & (pu.area0 < 0.001)].sort_values("d", ascending=False)
    wo = (wrong_object_loss(sg[sg.ds == "vost"]) - wrong_object_loss(b0[b0.ds == "vost"])).sort_values(ascending=False)
    units = {"mosev2": tiny.unit.iloc[0], "vost": wo.index[0]}
    sel = {"rules": __doc__.split("Rules")[1].strip(), "units": {}, "ranking": {
        "mosev2_tiny_SGGL_minus_B0": tiny[["unit", "d", "area0"]].head(5).to_dict("records"),
        "vost_wrong_object_loss_SGGL_minus_B0": [{"unit": u, "d": float(v)} for u, v in wo.head(5).items()]}}
    for ds, u in units.items():
        arms = {"B0": b0[b0.unit == u], "SGGL": sg[sg.unit == u]}
        w = PICK[ds]
        frames, ep = episode_frames(arms[w], arms["SGGL" if w == "B0" else "B0"])
        sel["units"][ds] = {"unit": u, "worse_arm": w, "frames_video_idx": frames, "episode": ep,
                            "JF_unit_mean": {k: float(v.JF.mean()) for k, v in arms.items()},
                            "area0": float(arms["SGGL"].area0.iloc[0])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    (OUT / "rerun.txt").write_text("".join(f"{a} {x['unit']}\n" for x in sel["units"].values() for a in ("B0", "OR0")))
    md(sel)
    print(json.dumps(sel["units"], indent=1))


def md(sel):
    f = lambda x: f"{x:+.3f}"
    L = ["# 그림 6 예시 단위 선택 근거", "", "`scripts/paper_fig6.py select`가 계산했다(손으로 고르지 않음). 원자료: `outputs/milestone10/runpod/runs/{B0,OR0}` (RunPod 4090 dev 행).", "",
         "## 규칙", "", "```", sel["rules"], "```", "", "## 선택", "",
         "| 데이터셋 | 단위 | area0 | B0 J&F | SGGL J&F | 에피소드를 잡은 arm | 에피소드 (시작, 길이, 회복) | 그린 frame (video_idx) |", "|---|---|---|---|---|---|---|---|"]
    for ds, x in sel["units"].items():
        e = x["episode"]
        L.append(f"| {ds} | `{x['unit']}` | {x['area0']:.5f} | {x['JF_unit_mean']['B0']:.3f} | {x['JF_unit_mean']['SGGL']:.3f} | {x['worse_arm']} "
                 f"| {e['onset']}, {e['dur']} frame, {'구간 끊김' if e['recovered'] else '끝까지'} | {', '.join(map(str, x['frames_video_idx']))} |")
    L += ["", "## 후보 순위 (상위 5)", "", "MOSEv2 극소 단위, SGGL − B0 (단위 평균 J&F):", "", "| 단위 | SGGL − B0 | area0 |", "|---|---|---|"]
    L += [f"| `{r['unit']}` | {f(r['d'])} | {r['area0']:.5f} |" for r in sel["ranking"]["mosev2_tiny_SGGL_minus_B0"]]
    L += ["", "VOST, wrong_object frame 손실 기여 (SGGL − B0):", "", "| 단위 | 차이 |", "|---|---|"]
    L += [f"| `{r['unit']}` | {f(r['d'])} |" for r in sel["ranking"]["vost_wrong_object_loss_SGGL_minus_B0"]]
    if "rerun" in sel:
        L += ["", "## 마스크 재실행 (Ubuntu RTX 2080)", "", "그림의 마스크는 2080에서 다시 돌린 것이다. 표의 수치는 4090 행이다. 두 기계의 단위 평균 J&F:", "",
              "| 단위 | arm | 4090 | 2080 | 차이 |", "|---|---|---|---|---|"]
        L += [f"| `{r['unit']}` | {r['arm']} | {r['JF_4090']:.4f} | {r['JF_2080']:.4f} | {r['JF_2080'] - r['JF_4090']:+.4f} |" for r in sel["rerun"]]
        L += ["", "## 공개", "",
              "- frame 규칙은 한 번 고쳤다. 첫 초안이 '직전' frame으로 GT 없는 frame(VOST t=54)을 골라, 'GT가 있고 J ≥ 0.5인 가장 가까운 frame'으로 바꿨다. 단위 선택 규칙은 바꾸지 않았다.",
              "- MOSEv2는 B0가 frame 1부터 끝까지 놓친다(에피소드 1개, 회복 없음). 그래서 '직전' 자리에 시작 frame을, '회복' 자리에 마지막 frame을 그렸다.",
              "- VOST의 손실 구간은 GT 없는 frame에서 끊기지만(표의 '구간 끊김'), 그 뒤 SGGL이 GT 있고 J ≥ 0.5로 돌아온 frame이 없어 세 번째 자리에 마지막 frame을 그렸다. 즉 회복하지 못했다.",
              "- MOSEv2 영상 `2c713560`의 frame은 Ubuntu에 없어, 공개 HF `FudanCVL/MOSEv2` train 분할 3개를 한 번 스트리밍해 이 영상만 추출했다. 분할 sha256은 M9 기록(`outputs/milestone9/manifests/mosev2_download_sha256.txt`)과 일치한다(`outputs/milestone10/fig6/dl/`).",
              "- 패널 위 J&F는 2080 재실행 값이다. 극소 객체가 보이도록 행마다 확대했다(마스크·GT·crop 사각형을 모두 담는 영역)."]
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "fig6_selection.md").write_text("\n".join(L) + "\n")


def zoom(masks, box, hw, pad=0.3):
    """Row view window: union bbox of the masks and the SGGL crop square, padded, at least 15% of the frame side."""
    ys, xs = [box[0], box[0] + box[2]], [box[1], box[1] + box[2]]
    for m in masks:
        if m.any():
            r, c = np.nonzero(m)
            ys += [r.min(), r.max()]
            xs += [c.min(), c.max()]
    cy, cx = (min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2
    h = max(max(ys) - min(ys), max(xs) - min(xs), 0.15 * max(hw)) * (1 + pad)
    return max(0, cy - h / 2), min(hw[0], cy + h / 2), max(0, cx - h / 2), min(hw[1], cx + h / 2)


def plot(only=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    sys.path.insert(0, "src")
    from rfeas import xdata as m7
    sel = json.loads((OUT / "selection.json").read_text())
    sel["rerun"] = []
    if only:
        sel["units"] = {k: v for k, v in sel["units"].items() if k in only.split(",")}
    rows = len(sel["units"]) * 3
    fig, ax = plt.subplots(rows, 3, figsize=(12, 3.2 * rows))
    r = 0
    for ds, x in sel["units"].items():
        u = next(v for _, v in m7.units(ds, "dev") if v.name == x["unit"])
        M, R = {}, {}
        for arm in ("B0", "OR0"):
            z = np.load(OUT / arm / f"{m7_stem(u)}.npz")
            hw = tuple(z["hw"])
            M[arm] = np.unpackbits(z["out"], axis=1)[:, : hw[0] * hw[1]].reshape(-1, *hw).astype(bool)
            R[arm] = pd.read_csv(OUT / arm / f"{m7_stem(u)}.csv")
            ref = A.load(arm)
            sel["rerun"].append({"unit": u.name, "arm": "SGGL" if arm == "OR0" else arm, "JF_2080": float(R[arm].JF.mean()),
                                 "JF_4090": float(ref[ref.unit == u.name].JF.mean())})
        paths = m7.paths(u)
        for t in x["frames_video_idx"]:
            img = np.asarray(Image.open(paths[t]).convert("RGB"))
            g = m7.gt(u, t)
            crop = R["OR0"].set_index("video_idx").loc[t, ["crop_y0", "crop_x0", "crop_side"]]
            y0, y1, x0, x1 = zoom((M["B0"][t - 1], M["OR0"][t - 1], g), crop.to_numpy(), g.shape)
            for c, (title, m, col) in enumerate((("B0 (official SAM 2.1)", M["B0"][t - 1], (1, .2, .2)),
                                                 ("SGGL", M["OR0"][t - 1], (.2, .5, 1)), ("GT", g, (1, 0, 1)))):
                a = ax[r, c]
                over = img.astype(float) / 255
                over[m] = 0.35 * over[m] + 0.65 * np.array(col)
                a.imshow(over)
                if c == 1:
                    a.add_patch(Rectangle((crop.crop_x0, crop.crop_y0), crop.crop_side, crop.crop_side, fill=False, ec="yellow", lw=1.5))
                jf = "" if c == 2 else f"  J&F {(R['B0'] if c == 0 else R['OR0']).set_index('video_idx').loc[t, 'JF']:.2f}"
                a.set_xlim(x0, x1)
                a.set_ylim(y1, y0)
                a.set_title(f"{ds} t={t}  {title}{jf}", fontsize=9)
                a.axis("off")
            r += 1
    fig.tight_layout()
    DOC.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOC / "fig6_examples.png", dpi=110)
    md(sel)
    print(sel["rerun"])


def m7_stem(u):
    sys.path.insert(0, "scripts")
    import m7
    return m7.stem(u)


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
