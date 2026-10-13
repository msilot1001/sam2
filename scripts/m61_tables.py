"""Per-object markdown tables for docs/m61/EXPERIMENT_COMPARISON.md (from saved CSVs only)."""
import pandas as pd

W = pd.read_csv("outputs/milestone61/comparison_per_unit.csv")
T = pd.read_csv("outputs/milestone61/trackA/tplr_per_unit.csv").set_index("unit")
A = pd.read_csv("outputs/milestone61/trackA/a3_per_unit.csv")
S = A[A.arm == "A3s"].set_index("unit")
f = lambda x: "" if pd.isna(x) else f"{x:+.3f}"
rows = ["| 세트 | 객체 | 면적% | B1 J&F | ΔB0 | ΔA2x | ΔA2c | ΔA3 | ΔA3s | ΔT1 | ΔF2 | Δhead |", "|" + "---|" * 12]
W = W.sort_values(["set", "area_frac"])
for r in W.itertuples():
    u = r.unit
    show = (r.area_frac < 0.01) or u in T.index or (abs(getattr(r, "dJF_A2x", 0) or 0) > 0.005)
    if not show:
        continue
    rows.append(f"| {r.set} | {u.split('__', 1)[1].replace('__', '/')} | {100 * r.area_frac:.3f} | {r.JF_B1:.3f} | {f(r.dJF_B0)} | "
                f"{f(r.dJF_A2x)} | {f(r.dJF_A2c)} | {f(r.dJF_A3)} | {f(S.d[u] if u in S.index else float('nan'))} | "
                f"{f(T.d_JF_T[u] if u in T.index else float('nan'))} | {f(getattr(r, 'dJF_F2', float('nan')))} | {f(getattr(r, 'dJF_HEAD', float('nan')))} |")
V = W.groupby(["set", "video"]).mean(numeric_only=True).reset_index()
vrows = ["| 세트 | 영상 | B1 | ΔA2x | ΔA3 | ΔF2 | Δhead |", "|---|---|---|---|---|---|---|"]
for r in V.sort_values(["set", "video"]).itertuples():
    vrows.append(f"| {r.set} | {r.video} | {r.JF_B1:.3f} | {f(r.dJF_A2x)} | {f(r.dJF_A3)} | {f(getattr(r, 'dJF_F2', float('nan')))} | {f(getattr(r, 'dJF_HEAD', float('nan')))} |")
open("docs/m61/per_object_tables.md", "w").write(
    "# M6.1 객체별·영상별 표 (J&F, P5 = B1 대비 Δ)\n\n"
    "- **포함 기준:** 작은 객체(< 1%), TPLR 적용 객체, 또는 A2x Δ가 0.005를 넘는 객체만 객체 표에 실었다.\n"
    "- **빈 칸:** 해당 arm을 그 세트에서 실행하지 않았다는 뜻이다.\n"
    "  - F2와 head는 DAVIS val + PUMaVOS B만 실행했다.\n"
    "  - A3s와 T1은 crop 적용 21개 객체만 실행했다.\n"
    "- **crop 미적용 객체:** A2x, A2c, A3는 규칙상 B1과 같아 Δ = 0이다.\n"
    "- **원자료:** `outputs/milestone61/comparison_per_unit.csv`, `trackA/*.csv`\n\n"
    "## 객체별\n\n" + "\n".join(rows) + "\n\n## 영상별 (객체 평균)\n\n" + "\n".join(vrows) + "\n")
print(len(rows) - 2, len(vrows) - 2)
