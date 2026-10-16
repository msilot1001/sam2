"""Paper figures 3/4/5 from the data CSVs (docs/paper/fig{3,4,5}_*.csv, made by scripts/paper_tables.py).

    uv run python scripts/paper_figs.py      # -> docs/paper/fig{3,4,5}.{png,svg} + docs/paper/captions.md

Korean font: NanumGothic (SIL OFL) from github.com/google/fonts ofl/nanumgothic, kept at data/fonts/ (sha256 below).
SVG text is stored as paths, so the SVG renders without the font. Colours: Okabe-Ito (colour-blind safe); every
colour distinction is doubled by hatching (fig 3), marker shape (fig 4, 5) or fill (fig 4 control).
"""
import csv
import importlib.util
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

T = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location("paper_tables", Path(__file__).with_name("paper_tables.py")))
s.loader.exec_module(T)

DOC = Path("docs/paper")
FONTS = {"data/fonts/NanumGothic-Regular.ttf": "76f45ef4a6bcff344c837c95a7dcc26e017e38b5846d5ae0cdcb5b86be2e2d31",
         "data/fonts/NanumGothic-Bold.ttf": "f96298f9fb18e364d2370f4c3ce948ac67a2b61af992d7234bc15c42b033c674"}
OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
         "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999", "black": "#000000"}
NAMES = {"mosev2": "MOSEv2", "vost": "VOST", "lvosv2": "LVOS v2"}
STATES = (("correct", "정상 (J ≥ 0.5) 잔여", OKABE["sky"], ""), ("partial", "부분", OKABE["yellow"], ".."),
          ("empty", "빈 출력", OKABE["grey"], "xx"), ("wrong_region", "다른 영역 (배경)", OKABE["orange"], "//"),
          ("wrong_object", "다른 주석 객체", OKABE["vermillion"], "\\\\"), ("false_present", "없는데 출력", OKABE["purple"], "oo"))
OFFSET = {("OR0", "mosev2"): (-12, -18), ("OR0", "vost"): (-14, -14), ("OR8", "mosev2"): (4, 16), ("OR8", "vost"): (6, 14),
          ("C58", "mosev2"): (16, -4), ("C58", "vost"): (6, 18)}            # label placement only
DS_STYLE = {"mosev2": (OKABE["blue"], "o"), "vost": (OKABE["orange"], "s")}


def setup():
    import hashlib
    for f, h in FONTS.items():
        assert Path(f).exists(), f"missing {f}: download NanumGothic from github.com/google/fonts/tree/main/ofl/nanumgothic"
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f"sha256 mismatch: {f}"
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False, "svg.fonttype": "path",
                         "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "hatch.linewidth": 0.6})


def rows(name):
    with open(DOC / name) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def save(fig, stem):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # a missing glyph raises instead of drawing a box
        for ext in ("png", "svg"):
            fig.savefig(DOC / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig3():
    R = rows("fig3_states.csv")
    loss = {(r["dataset"], r["arm"], r["state"]): float(r["loss"]) for r in R}
    jf = {(r["dataset"], r["arm"]): float(r["frame_JF"]) for r in R}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, ds in zip(axs, NAMES):
        for i, arm in enumerate(("B0", "SGGL")):
            y = 0.0
            for st, label, col, hatch in STATES:
                h = loss.get((ds, arm, st), 0.0)
                ax.bar(i, h, 0.6, bottom=y, color=col, hatch=hatch, edgecolor="black", linewidth=0.5, label=label if ds == "mosev2" and i == 0 else None)
                y += h
            ax.text(i, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
            assert abs(y - (1 - jf[(ds, arm)])) < 0.002, (ds, arm, y, jf[(ds, arm)])   # states sum to 1 - frame J&F
        ax.set_xticks([0, 1], ["B0\n(공식 SAM 2.1)", "SGGL"])
        ax.set_title(NAMES[ds])
        ax.set_xlim(-0.6, 1.6)
    axs[0].set_ylabel("frame당 손실 기여 (합 = 1 − frame 평균 J&F)")
    axs[0].set_ylim(0, max(1 - v for v in jf.values()) * 1.15)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("상태별 손실 분해: B0 대 SGGL (dev)", y=1.02)
    save(fig, "fig3")
    vost = {a: {st: loss[("vost", a, st)] for st, *_ in STATES} for a in ("B0", "SGGL")}
    return (f"그림 3. 데이터셋별 frame당 손실을 상태로 나눈 것(B0 대 SGGL, dev). 막대 위 숫자는 1 − frame 평균 J&F다. "
            f"VOST에서 SGGL의 '다른 주석 객체' 손실은 {vost['B0']['wrong_object']:.3f}에서 {vost['SGGL']['wrong_object']:.3f}로 늘어난다.")


def fig4():
    R = {(r["arm"], r["dataset"]): r for r in rows("fig4_oracle.csv")}
    arms = [a for a, _ in T.ORACLES]
    label = dict(T.ORACLES)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.4), sharey=True)
    for ax, ds in zip(axs, ("mosev2", "vost")):
        for i, arm in enumerate(arms):
            r = R.get((arm, ds))
            if not r:
                continue
            y = len(arms) - 1 - i
            d, lo, hi, rec = (float(r[k]) for k in ("dJF", "ci_lo", "ci_hi", "recovery"))
            ctrl, off = arm == "OR7R", r["kind"] == "offline-framewise"
            col = OKABE["grey"] if ctrl else OKABE["vermillion"] if off else OKABE["blue"]
            ax.hlines(y, lo, hi, color=col, linewidth=1.4, linestyles="--" if ctrl else "-")
            ax.plot(d, y, marker="D" if ctrl else "s" if off else "o", markersize=5, color=col,
                    markerfacecolor="white" if ctrl else col, markeredgewidth=1.2)
            ax.text(max(hi, d) + 0.012, y, f"{rec * 100:.0f}%", va="center", fontsize=7)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(NAMES[ds])
        ax.set_xlabel("ΔJ&F (oracle − SGGL) [95% CI]")
        ax.set_xlim(min(-0.08, min(float(r["ci_lo"]) for (a, d), r in R.items() if d in ("mosev2", "vost") and a in label) - 0.01), 0.55)
        ax.grid(axis="x", linewidth=0.3, alpha=0.6)
    axs[0].set_yticks(range(len(arms)), [f"{a} {label[a]}" for a in reversed(arms)], fontsize=7.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=OKABE["blue"], marker="o", label="온라인 oracle"),
         Line2D([], [], color=OKABE["vermillion"], marker="s", label="오프라인 (frame별)"),
         Line2D([], [], color=OKABE["grey"], marker="D", markerfacecolor="white", linestyle="--", label="대조 (무작위 차단)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.06), frameon=False)
    fig.suptitle("Oracle 상한 (dev, 영상 동일 가중; 점 옆 숫자 = 회복 비율)", y=1.0)
    save(fig, "fig4")
    c, o = R[("C58", "vost")], R[("OR7R", "vost")]
    return (f"그림 4. GT로 병목 하나씩 고친 oracle의 SGGL 대비 ΔJ&F와 95% CI(MOSEv2·VOST dev). 숫자는 SGGL 손실 중 회복한 비율이다. "
            f"선택 + 완벽한 기억(C58)이 VOST에서 {float(c['dJF']):+.3f}({float(c['recovery']) * 100:.0f}%)로 가장 크고, "
            f"같은 수 무작위 기록 차단 대조(OR7R)는 {float(o['dJF']):+.3f}이다.")


def fig5():
    R = [r for r in rows("fig5_episodes.csv") if r["kind"] == "drift" and r["dataset"] in DS_STYLE and r["episodes"] not in ("", "0")]
    arms = {"OR0"} | {a for a, _ in T.ORACLES}
    pts = [(r["arm"], r["dataset"], int(r["episodes"]), float(r["mean_len"])) for r in R if r["arm"] in arms and r["mean_len"]]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for ds, (col, mk) in DS_STYLE.items():
        P = [p for p in pts if p[1] == ds]
        ax.scatter([p[2] for p in P], [p[3] for p in P], color=col, marker=mk, s=28, label=NAMES[ds], edgecolor="black", linewidth=0.4)
        for arm, _, n, L in P:
            if arm in ("OR0", "OR8", "C58"):
                ax.annotate(f"{'OR0 (SGGL)' if arm == 'OR0' else arm} · {NAMES[ds]}", (n, L), xytext=OFFSET[(arm, ds)],
                            textcoords="offset points", fontsize=7.5, ha="left" if OFFSET[(arm, ds)][0] > 0 else "right",
                            arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "black"})
    ax.set_ylim(0.5, 10.6)
    ax.set_xlabel("이탈 에피소드 수")
    ax.set_ylabel("평균 길이 (frame)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(linewidth=0.3, alpha=0.6)
    ax.set_title("이탈 에피소드: 시작 횟수 대 지속 길이 (dev)")
    save(fig, "fig5")
    g = {(a, d): (n, L) for a, d, n, L in pts}
    return (f"그림 5. arm별 이탈(배경·다른 객체) 에피소드 수와 평균 길이(MOSEv2·VOST dev). 완벽한 기억(OR8)은 VOST 평균 길이를 "
            f"{g[('OR0', 'vost')][1]:.1f}에서 {g[('OR8', 'vost')][1]:.1f} frame으로 줄이지만 횟수는 {g[('OR0', 'vost')][0]}에서 {g[('OR8', 'vost')][0]}로 "
            f"조금만 줄인다. 선택을 더하면(C58) {g[('C58', 'vost')][0]}회로 준다.")


if __name__ == "__main__":
    setup()
    caps = [fig3(), fig4(), fig5()]
    (DOC / "captions.md").write_text("# 그림 캡션 초안 (그림 3·4·5)\n\n`scripts/paper_figs.py`가 CSV에서 수치를 읽어 만든다.\n\n"
                                     + "\n\n".join(caps) + "\n")
    print("\n".join(caps))
