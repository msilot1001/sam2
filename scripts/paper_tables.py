"""Paper tables/figure data from frozen report JSONs (no numbers typed by hand).

    uv run python scripts/paper_tables.py

Writes docs/paper/{table5_main,table6_tiny,table8_states,table9_oracle,table10_selection}.md and
fig{3_states,4_oracle,5_episodes}.csv, then cross-checks against docs/m10/{BASELINES,CORE_CONCLUSIONS,FINAL_REPORT}.md
(mismatches are printed, never forced).
"""
import csv
import json
import math
import re
from pathlib import Path

REP = Path("outputs/milestone10/reports")
M10 = Path("outputs/milestone10")
M61 = Path("outputs/milestone61/comparison_report.json")
M7R = Path("outputs/milestone7/replay_report.json")
M7V = Path("outputs/milestone7/vk_report.json")
OUT = Path("docs/paper")
NAMES = {"mosev2": "MOSEv2", "vost": "VOST", "lvosv2": "LVOS v2"}
DS = tuple(NAMES)
PAIRS = ("SGGL-B0", "SGGL-P5", "P5-B0")
HEAD = ("| 데이터셋 | 단위/영상 | B0 | P5 | SGGL | SGGL − B0 [CI] | ↑/↓ | LOVO min | SGGL − P5 [CI] | ↑/↓ | LOVO min "
        "| P5 − B0 [CI] | ↑/↓ | LOVO min |")
SEP = "|" + "---|" * 14


def J(p):
    return json.loads(Path(p).read_text())


def bad(x):
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def s(x):                                              # signed, ASCII minus (BASELINES.md style)
    return "—" if bad(x) else f"{x:+.3f}"


def a(x):
    return "—" if bad(x) else f"{x:.3f}"


def ci(d, lo, hi):
    return f"{s(d)} [{s(lo)}, {s(hi)}]"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")


def baseline_rows(tag):
    res = J(REP / "baselines.json")
    rows = []
    for ds, r in res["tables"][tag].items():
        row = [NAMES[ds], f"{r['units']}/{r['videos']}" + ("" if r["units"] == r["units_total"] else f" (전체 {r['units_total']})")]
        row += [a(r["abs"][k]) for k in ("B0", "P5", "SGGL")]
        for p in PAIRS:
            d = r["delta"][p]
            row += [ci(d["dJF"], *d["ci"]), f"{d['videos_up']}/{d['videos_down']}", s(d["lovo_min"])]
        rows.append(row)
    return res, rows


def md_row(row):
    return "| " + " | ".join(row) + " |"


# ----------------------------------------------------------------------------- table 5
def table5():
    res, rows = baseline_rows("all")
    g = J(M61)["gate_set(DAVIS_val+PUMaVOS_B)"]
    b0 = g["B0"]
    thr = next(k for k in b0 if k.startswith("up_")).split("_", 1)[1]
    sg = g["A2x"]                                       # A2x = SGGL (docs/m7 FINAL_ALGORITHM_DECISION §1)
    m61 = ["DAVIS val + PUMaVOS B (M6.1 관문 세트)ᵃ", f"{g['units']}/{g['videos']}", "", a(g["JF_B1"]), "",
           "", "", "",
           ci(sg["dJF"], *sg["CI"]), f"{sg[f'up_{thr}']}/{sg[f'down_{thr}']}", "",
           ci(-b0["dJF"], -b0["CI"][1], -b0["CI"][0]), f"{b0[f'down_{thr}']}/{b0[f'up_{thr}']}", ""]
    rp = J(M7R)
    pil = rp["pilots"]["SGGL-V"]                        # every M7 replay pilot shares this small-unit set
    sp = rp["SGGL_vs_P5_small"]
    m7 = ["DAVIS DEV + REP (M7, 작은 객체)ᵇ", f"{pil['n_units']}/{pil['n_videos']}", "", "", "", "", "", "",
          ci(sp["mean"], *sp["CI"]), "", "", "", "", ""]
    vk = J(M7V)
    v1, vb, vp = vk["SGGL-P5"], vk["J:SGGL-B0"], vk["J:P5-B0"]
    ud = lambda d: f"{d[f'up_{thr}']}/{d[f'down_{thr}']}"
    vkr = ["VKITTI2 (M7)ᶜ", f"{vk['units']}/{vk['scenes']} scene", "", "", "",
           ci(vb["mean"], *vb["CI_by_unit_descriptive"]) + " (J)", ud(vb), "",
           ci(v1["mean"], *v1["CI_by_unit_descriptive"]), ud(v1), "",
           ci(vp["mean"], *vp["CI_by_unit_descriptive"]) + " (J)", ud(vp), ""]
    scenes = ", ".join(f"{k.split('/')[-1]} {s(v)}" for k, v in v1["by_scene"].items())
    L = ["# 표 5. 주 결과: 공식 SAM 2.1 (B0) / P5 / SGGL", "",
         "MOSEv2·VOST·LVOS v2는 M10 dev split(같은 frame 짝, 영상 동일 가중, 영상 bootstrap CI). "
         "↑/↓ = Δ > +0.01 / < −0.01인 영상 수, LOVO min = 영상 하나씩 뺀 평균의 최솟값. 빈 칸 = 그 milestone에서 측정하지 않음.", "",
         HEAD, SEP] + [md_row(r) for r in rows] + [md_row(r) for r in (m61, m7, vkr)] + [
        "",
        "출처: `outputs/milestone10/reports/baselines.json` (MOSEv2·VOST·LVOS v2), `outputs/milestone61/comparison_report.json` (ᵃ), "
        "`outputs/milestone7/replay_report.json` (ᵇ), `outputs/milestone7/vk_report.json` (ᶜ). "
        "해설 문서: `docs/m10/BASELINES.md`, `docs/m61/EXPERIMENT_COMPARISON.md`, `docs/m7/FINAL_ALGORITHM_DECISION.md`.",
        "",
        f"- ᵃ J&F, 단위 = 객체(DAVIS val + PUMaVOS B 전체 객체), P5 = `JF_B1`. SGGL = A2x. ↑/↓는 객체 수(±{thr}), LOVO 없음. "
        f"M6.1에서 B0는 P5 대비 {ci(b0['dJF'], *b0['CI'])}만 측정했다(B0·SGGL 절대값과 SGGL − B0는 기록 없음). "
        "P5 − B0 칸은 이 값의 부호를 바꾼 것이다(CI 양 끝도 바꿈).",
        "- ᵇ J&F, M7 DEV(M6 dev crop 객체) + REP(DAVIS 2017 train crop 객체) 중 작은 crop 객체, 영상 CI. "
        "M7은 이 세트에서 B0를 다시 돌리지 않았다. 여러 번 본 개발 데이터라 탐색적이다.",
        f"- ᶜ VKITTI2 {', '.join(k.split('/')[-1] for k in v1['by_scene'])} (scene별 SGGL − P5: {scenes}). SGGL − P5는 J&F, "
        "SGGL − B0와 P5 − B0는 **J만** 기록되어 있다(표시 (J)). CI는 객체 단위 서술용이고 실효 표본은 scene "
        f"{vk['scenes']}개다. ↑/↓는 객체 수. M2–M4 개발 scene이다.",
        "- 행마다 실행 기계가 다르다. 기계가 다른 행의 절대값은 서로 비교하지 않는다(같은 행 안의 차이만 짝지은 비교).",
    ]
    write("table5_main.md", L)
    return rows, b0


# ----------------------------------------------------------------------------- table 6, 8, fig 3
def table6():
    _, rows = baseline_rows("tiny")
    write("table6_tiny.md", ["# 표 6. 극소 단위 (area0 < 0.001): B0 / P5 / SGGL", "",
                             "M10 dev split, 표 5와 같은 집계. 단위/영상 옆 괄호는 SGGL(OR0)의 전체 극소 단위 수.", "",
                             HEAD, SEP] + [md_row(r) for r in rows] +
          ["", "출처: `outputs/milestone10/reports/baselines.json` (`tables.tiny`). 해설: `docs/m10/BASELINES.md` 표 2."])
    return rows


def table8():
    st = J(REP / "baselines.json")["states"]
    L = ["# 표 8. 상태별 손실 분해 (frame당 손실 기여, 합 = 1 − frame 평균 J&F)", "",
         "M10 dev split. P5는 G와 다른 객체의 교집합이 기록되지 않아 wrong_region과 wrong_object를 나눌 수 없다(합친 열만).", "",
         "| 데이터셋 | arm | frame J&F | correct 잔여 | partial | empty | wrong_region | wrong_object | 이탈 합 (region+object) | false_present |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in st.items():
        for k, x in r.items():
            L.append(f"| {NAMES[ds]} | {k} | {a(x['JF_frame_mean'])} | {a(x['correct'])} | {a(x['partial'])} | {a(x['empty'])} | "
                     f"{a(x['wrong_region']) if k != 'P5' else '—'} | {a(x['wrong_object'])} | {a(x['wrong_region+wrong_object'])} | "
                     f"{a(x['false_present'])} |")
    L += ["", "출처: `outputs/milestone10/reports/baselines.json` (`states`). 해설: `docs/m10/BASELINES.md` 표 3."]
    write("table8_states.md", L)
    with open(OUT / "fig3_states.csv", "w", newline="") as f:
        f.write("# source: outputs/milestone10/reports/baselines.json (states)\n")
        w = csv.writer(f)
        w.writerow(["dataset", "arm", "state", "loss", "frame_JF", "frames"])
        for ds, r in st.items():
            for arm in ("B0", "SGGL"):
                x = r[arm]
                for state in ("correct", "partial", "empty", "wrong_region", "wrong_object", "false_present", "absent_ok"):
                    w.writerow([ds, arm, state, a(x[state]), a(x["JF_frame_mean"]), x["frames"]])


# ----------------------------------------------------------------------------- table 9, fig 4
ORACLES = (("C58", "선택 + 완벽한 기억"), ("OR5M", "선택 + 선택 가설로 기억"), ("OR8", "완벽한 기억 (GT mask 재부호화)"),
           ("C75", "선택 + 기록 허용 (GT mask 없음)"), ("OR5S", "선택 (출력만)"), ("OR8K10", "완벽한 기억, 10 frame마다"),
           ("OR7", "기억 기록 허용 (맞는 예측만)"), ("OR7R", "대조: 같은 수 무작위 차단"), ("OR7W", "대상 밖만 차단"),
           ("OR4", "완벽한 기하 (위치 + 크기)"), ("OR5off", "best-of-7"), ("OR1off", "존재"), ("OR6off", "대상 영역 판별"))
KIND = {"online-rollout": "온라인", "offline-framewise": "오프라인"}


def table9():
    ot = J(REP / "oracle_table.json")
    base, arms = ot["baseline"], ot["arms"]
    flags, L = [], ["# 표 9. Oracle 상한 (dev, OR0 = SGGL 대비 ΔJ&F)", "",
                    "같은 단위·같은 frame 짝, 영상 동일 가중, 영상 bootstrap CI. 회복 = ΔJ&F / (1 − OR0 J&F) "
                    "(`scripts/m10_analysis.py` `vstats`의 `recovered_frac`, 같은 단위의 OR0 기준).", "",
                    "| Oracle | 종류 | MOSEv2 ΔJ&F [CI] | VOST ΔJ&F [CI] | LVOS v2 ΔJ&F [CI] | 회복 MOSEv2 | 회복 VOST | 회복 LVOS v2 |",
                    "|---|---|---|---|---|---|---|---|",
                    "| OR0 SGGL | 기준 | J&F " + " | ".join(a(base[d]["JF_video_equal"]) for d in DS) + " | — | — | — |"]
    for arm, label in ORACLES:
        A = arms[arm]
        cells, rec = [], []
        for d in DS:
            x = A["per_dataset"].get(d)
            if x is None:
                cells.append("NOT_RUN"), rec.append("—")
                continue
            if not x["complete"]:                       # partial run: no estimate shown
                cells.append(f"미완료 ({x['n_units']}/{base[d]['units']}단위)"), rec.append("—")
                continue
            mark = ""
            if x["top2_share"] is not None and x["top2_share"] >= 0.5:
                flags.append((arm, d, x))
                mark = "†"
            cells.append(ci(x["dJF"], *x["ci"]) + mark)
            rec.append(a(x["recovered_frac"]))
        L.append(f"| {arm} {label} | {KIND[A['kind']]} | " + " | ".join(cells + rec) + " |")
    shown = {k for k, _ in ORACLES}
    incomplete = sorted(k for k, v in arms.items() if any(not x.get("complete", True) for x in v["per_dataset"].values()))
    L += ["", "출처: `outputs/milestone10/reports/oracle_table.json`. 해설: `docs/m10/CORE_CONCLUSIONS.md` 2절, "
              "`docs/m10/FINAL_REPORT.md` 2절.", ""]
    for arm, d, x in flags:
        L.append(f"- † {arm} {NAMES[d]}: 상위 2영상이 순이득의 {a(x['top2_share'])}을 차지한다(≥ 0.5). "
                 f"LOVO {s(x['lovo_min'])} ~ {s(x['lovo_max'])}, 영상 ↑/↓ {x['videos_up']}/{x['videos_down']} "
                 f"({x['n_videos']}영상).")
    L += [f"- 이 표에 없는 arm: {', '.join(k for k in arms if k not in shown)}. "
          f"데이터셋 일부만 실행된 arm(미완료): {', '.join(incomplete)}. B0는 표 5, VXO·VXM·N1M·N2X는 표 10, "
          "OR0F·OR0K·OR0T는 OR0 재실행 대조다.",
          "- CI는 2026-10-16 스냅샷의 `oracle_table.json`(모든 run 완료 뒤 재생성)에서 읽었다. bootstrap 난수 흐름이 arm 목록에 따라 달라져, "
          "`FINAL_REPORT.md`·`CORE_CONCLUSIONS.md`의 이전 값과 CI 끝이 최대 0.003, OR7W MOSEv2 점추정이 0.001 다르다(원인 미확인, 이전 값 0.042). 판정은 바뀌지 않는다."]
    write("table9_oracle.md", L)
    with open(OUT / "fig4_oracle.csv", "w", newline="") as f:
        f.write("# source: outputs/milestone10/reports/oracle_table.json\n")
        w = csv.writer(f)
        w.writerow(["arm", "kind", "dataset", "n_units", "n_videos", "complete", "dJF", "ci_lo", "ci_hi", "recovery", "top2_share"])
        for arm, A in arms.items():
            for d, x in A["per_dataset"].items():
                w.writerow([arm, A["kind"], d, x["n_units"], x["n_videos"], x["complete"], a(x["dJF"]), a(x["ci"][0]),
                            a(x["ci"][1]), a(x["recovered_frac"]), "" if x["top2_share"] is None else a(x["top2_share"])])
    o8 = arms["OR8"]["per_dataset"]["lvosv2"]
    print(f"LVOS v2 OR8: dJF {s(o8['dJF'])}, top2_share {a(o8['top2_share'])}, LOVO min {s(o8['lovo_min'])} max {s(o8['lovo_max'])}, "
          f"videos up/down {o8['videos_up']}/{o8['videos_down']} of {o8['n_videos']}; footnote fired: {o8['top2_share'] >= 0.5}")
    return ot


# ----------------------------------------------------------------------------- table 10
def maxp(curve, key, min_sw):
    ok = [x for x in curve if x[key] is not None and x["switches"] >= min_sw]
    return max(ok, key=lambda x: x[key]) if ok else None


def pc(m, key="precision"):
    return "—" if m is None else f"{a(m[key])} ({m['switches']})"


def table10():
    min_sw = int(re.search(r">= (\d+) switches", (M10 / "preregistration_amend10.json").read_text()).group(1))
    L = ["# 표 10. GT 없는 선택 신호 검정 (dev)", "",
         "모든 정밀도 = 스위치 중 고른 가설의 J가 SAM 2 선택보다 높은 비율. 합동 = MOSEv2 + VOST.", "",
         "## (a) 손규칙 오프라인 관문 (수정 5)", "",
         "| 규칙 | 합동 정밀도 (τ 전 범위) | 스위치 수 범위 | τ* | τ=0 ΔJ (GT 있음) MOSEv2 [CI] | VOST [CI] | LVOS v2 [CI] | 판정 |",
         "|---|---|---|---|---|---|---|---|"]
    sg = J(REP / "selector_gate.json")
    for rule, r in sg["rules"].items():
        p = [x["precision_pooled"] for x in r["curve"] if x["precision_pooled"] is not None]
        n = [x["switches"] for x in r["curve"]]
        e = r["at_tau0"]
        verdict = "PASS" if r["pass"] else ("FAIL (τ* 없음)" if r["tau_star"] is None else "FAIL")
        L.append(f"| {rule} | {a(min(p))}–{a(max(p))} | {min(n)}–{max(n)} | {'—' if r['tau_star'] is None else a(r['tau_star'])} | "
                 + " | ".join(ci(e[d]["dJ_present"], *e[d]["ci"]) if d in e else "—" for d in DS) + f" | {verdict} |")
    so = J(REP / "selector_offline.json")
    rules = [k[:-4] for k in next(iter(so.values())) if k.endswith("_hit")]
    L += ["", f"통과 규칙: {', '.join(sg['passing']) or '없음'}. 선택: {sg['chosen'] or '없음'}.", "",
          "오프라인 적중률 (GT 있는 frame 중 J ≥ 0.5 가설이 하나라도 있는 frame에서 맞는 가설을 고른 비율):", "",
          "| 데이터셋 | 맞는 가설이 있는 frame / GT 있는 frame | " + " | ".join(rules) + " | 평균 J: SGGL / oracle |",
          "|---|---|" + "---|" * len(rules) + "---|"]
    for d in DS:
        x = so[d]
        L.append(f"| {NAMES[d]} | {x['frames_with_a_correct_hypothesis']}/{x['present_frames']} | "
                 + " | ".join(a(x[f'{k}_hit']) for k in rules) + f" | {a(x['R_sggl_meanJ_present'])} / {a(x['oracle_meanJ_present'])} |")

    # (b) amend6/8 exploratory online arms
    dg = J(REP / "degraded.json")
    ar = dg["arms"]
    L += ["", "## (b) 불일치 조건부 view 전환, 온라인 (수정 6·8, 탐색)", "",
          "| arm | MOSEv2 ΔJ&F [CI] | VOST ΔJ&F [CI] | LVOS v2 ΔJ&F [CI] | 실현 정밀도 (M / V / L) | 스위치 (M / V / L) | 판정 (수정 8 기준) |",
          "|---|---|---|---|---|---|---|"]
    for arm in ("VXO", "VXM", "N1M", "N2X"):
        x = ar[arm]
        if arm in ("VXO", "VXM"):
            crit = {1: all(x[d]["ci"][0] > 0 for d in ("vost", "lvosv2")),
                    2: x["mosev2"]["dJF"] >= -0.005,
                    4: all(x[d]["units_below_m005"] <= ar["N2X"][d]["units_below_m005"] for d in ("vost", "lvosv2"))}
            if arm == "VXM":
                crit[3] = all(x[d]["dJF"] > ar[n][d]["dJF"] for d in ("vost", "lvosv2") for n in ("N1M", "N2X"))
            failed = [str(k) for k in sorted(crit) if not crit[k]]
            verdict = "PASS" if not failed else f"FAIL (기준 {'·'.join(failed)} 미충족)"
        else:
            verdict = "대조"
        L.append(f"| {arm} | " + " | ".join(ci(x[d]["dJF"], *x[d]["ci"]) for d in DS) + " | "
                 + " / ".join(a(x[d]["realized_precision"]) for d in DS) + " | "
                 + " / ".join(str(x[d]["switches"]) for d in DS) + f" | {verdict} |")
    op = dg["offline_disagree_precision"]
    ot = J(REP / "oracle_table.json")["arms"]
    diff = max(abs(ar[k][d]["dJF"] - ot[k]["per_dataset"][d]["dJF"]) for k in ("VXO", "VXM", "N1M", "N2X") for d in DS)
    L += ["", "- 오프라인 불일치 frame 정밀도 (OR0F, 전환 전): " + ", ".join(f"{NAMES[d]} {a(op[d]['precision'])} ({op[d]['switches']})" for d in DS) + ".",
          "- 수정 8 기준: (1) VOST·LVOS v2 CI 하한 > 0, (2) MOSEv2 점추정 ≥ −0.005, (3) VXM > N1M·N2X (VOST·LVOS v2, VXM만), "
          "(4) −0.05 미만 단위 수 ≤ N2X. 전환 조건(교차 IoU < 0.5)은 dev를 본 뒤 정했다(공개).",
          f"- N1M = 같은 frame에서 자기 mask로 기억만 재부호화, N2X = 같은 수 무작위 전환. "
          f"`oracle_table.json`의 같은 arm 점추정과 최대 차이 {a(diff)} (CI는 bootstrap 순서가 달라 소수 셋째 자리에서 다를 수 있음)."]

    # (c) amend10 learned heads (two rounds) + amend11 pooled diagnostic
    r1, r2 = J(REP / "head_offline.json"), J(REP / "head_offline_a01.json")
    L += ["", "## (c) 학습 head (수정 10 두 회차, 수정 11 진단)", "",
          f"정밀도 = 전환 ≥ {min_sw}회인 τ 중 최댓값 (전환 수). τ*는 학습 OOF에서 정밀도 ≥ 0.70 (전환 ≥ {min_sw})인 최소 τ. "
          f"학습: 1회차 {r1['train']['units']}단위/{r1['train']['videos']}영상, 2회차 {r2['train']['units']}단위/{r2['train']['videos']}영상 "
          f"(데이터셋: {', '.join(NAMES[k] for k in r2['train']['per_ds'])}).", "",
          "| 모델 | 가설 | 1회차 학습 OOF | 1회차 dev 합동 | 1회차 τ* | 2회차 학습 OOF | 2회차 dev 합동 | 2회차 τ* | 통과 (1 / 2회차) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for hyp in ("H1", "H2"):
        for m in r1["models"]:
            x1, x2 = r1["models"][m][hyp], r2["models"][m][hyp]
            c = [maxp(x[k], "precision", min_sw) for x in (x1, x2) for k in ("curve_train_oof", "curve_dev")]
            if not any(c):
                continue
            t = lambda x: "—" if x["tau_star"] is None else a(x["tau_star"])
            L.append(f"| {m} | {hyp} | {pc(c[0])} | {pc(c[1])} | {t(x1)} | {pc(c[2])} | {pc(c[3])} | {t(x2)} | "
                     f"{'PASS' if x1['pass'] else 'FAIL'} / {'PASS' if x2['pass'] else 'FAIL'} |")
    b1 = r1["models"]["b"]["H1"]
    if b1["tau_star"] is not None:
        L += ["", f"1회차 b(H1)만 τ*가 있었다: 학습 OOF {a(b1['train_oof_precision'])}, dev 합동 {a(b1['dev_precision_pooled'])} "
                  f"({b1['dev_switches_pooled']}회). 선택된 head: "
                  + ", ".join(f"{n} {h} {r['chosen'][h] or '없음'}" for n, r in (("1회차", r1), ("2회차", r2)) for h in r["chosen"]) + "."]
    hu = J(REP / "head_upper.json")
    L += ["", "수정 11 (진단 전용): dev 109단위를 데이터셋마다 영상 5-fold로 나눈 OOF 점수로 평가 (τ를 같은 점수로 골라 낙관적).", "",
          "| 모델 | 가설 | 합동 | MOSEv2 | VOST | LVOS v2 | τ* |", "|---|---|---|---|---|---|---|"]
    for hyp in ("H1", "H2"):
        for m, v in hu["models"].items():
            x = v[hyp]["max_precision"]
            if not any(x.values()):
                continue
            L.append(f"| {m} | {hyp} | {pc(x['pooled_mosev2_vost'])} | " + " | ".join(pc(x[d]) for d in DS)
                     + f" | {'—' if v[hyp]['tau_star'] is None else a(v[hyp]['tau_star'])} |")

    # (d) amend12 within-dataset diagnostic
    hw = J(REP / "head_within.json")["datasets"]
    models = list(next(iter(hw.values()))["models"])
    L += ["", "## (d) 같은 데이터셋 안 상한 (수정 12, 진단 전용)", "",
          "데이터셋마다 그 데이터셋의 dev 영상만으로 영상 5-fold OOF. 칸 = OOF 최대 정밀도 (전환 수). 판정은 사전 등록 문장(최고 학습 모델 a·b·c 기준).", "",
          "| 데이터셋 (단위/영상) | 가설 | " + " | ".join(models) + " | 판정 (최고 모델) |", "|---|---|" + "---|" * len(models) + "---|"]
    for d, r in hw.items():
        for hyp in ("H1", "H2"):
            v = r["verdict"][hyp]
            L.append(f"| {NAMES[d]} ({r['units']}/{r['videos']}) | {hyp} | " + " | ".join(pc(r["models"][m][hyp]["max_precision"]) for m in models)
                     + f" | {v['verdict'] or '—'} ({v['best_model']}, {a(v['max_precision'])}) |")
    L += ["", "출처: `outputs/milestone10/reports/selector_gate.json`, `selector_offline.json` (a); `degraded.json`, "
              "`oracle_table.json` (b); `head_offline.json`, `head_offline_a01.json`, `head_upper.json` (c); `head_within.json` (d). "
              "사전 등록: `outputs/milestone10/preregistration_amend5.json`–`amend12.json`. 해설: `docs/m10/FINAL_REPORT.md` 4절, "
              "`docs/m10/HEAD_RESULTS.md`."]
    write("table10_selection.md", L)


# ----------------------------------------------------------------------------- fig 5
def fig5():
    ep = J(REP / "episodes.json")
    with open(OUT / "fig5_episodes.csv", "w", newline="") as f:
        f.write("# source: outputs/milestone10/reports/episodes.json\n")
        w = csv.writer(f)
        w.writerow(["arm", "dataset", "kind", "units", "episodes", "mean_len", "or0_episodes_same_units", "or0_mean_len_same_units"])
        for d, r in ep["OR0_all"].items():
            for k, x in r.items():
                w.writerow(["OR0", d, k, "", x["episodes"], a(x["mean_dur"]) if x["mean_dur"] is not None else "", "", ""])
        for arm, v in ep.items():
            if arm == "OR0_all":
                continue
            for d, r in v.items():
                for k in ("loss", "drift"):
                    x, o = r["arm"][k], r["OR0"][k]
                    w.writerow([arm, d, k, r["units"], x["episodes"], "" if x["mean_dur"] is None else a(x["mean_dur"]),
                                o["episodes"], "" if o["mean_dur"] is None else a(o["mean_dur"])])


# ----------------------------------------------------------------------------- cross-checks (print only)
def crosscheck(rows5, rows6, b0, ot):
    norm = lambda t: t.replace("−", "-").strip()
    doc = Path("docs/m10/BASELINES.md").read_text().splitlines()
    for title, rows in (("표 1", rows5), ("표 2", rows6)):
        i = next(k for k, l in enumerate(doc) if l.startswith(f"## {title}"))
        got = {r[0]: r for r in rows}
        for l in doc[i + 4:i + 7]:
            cells = [c.strip() for c in l.strip("|").split("|")]
            mine = got.get(cells[0])
            print(f"BASELINES {title} {cells[0]}:", "OK" if mine and [norm(c) for c in mine] == [norm(c) for c in cells] else f"MISMATCH {cells} vs {mine}")
    m61 = Path("docs/m61/EXPERIMENT_COMPARISON.md").read_text()
    want = ci(b0["dJF"], *b0["CI"]).replace("-", "−")
    print("M6.1 B0 footnote value", want, "found in EXPERIMENT_COMPARISON.md:", want in m61)
    arms = ot["arms"]
    pat = re.compile(r"([+−-]\d\.\d{3})(?: \[([+−-]\d\.\d{3}), ([+−-]\d\.\d{3})\])?")
    for name, path, cols in (("CORE_CONCLUSIONS", "docs/m10/CORE_CONCLUSIONS.md", (2, 3, 4)),
                             ("FINAL_REPORT", "docs/m10/FINAL_REPORT.md", (2, 4, 6))):
        text = Path(path).read_text()
        sec = text[text.index("| Oracle", text.index("## 2. Oracle")):]
        sec = sec[:sec.index("\n\n")]                     # first table of section 2 only
        bad_ = []
        for l in sec.splitlines():
            if not l.startswith("| ") or l.startswith("| Oracle") or l.startswith("|---"):
                continue
            cells = [c.strip() for c in l.strip("|").split("|")]
            arm = cells[0].replace("**", "").split()[0]
            if arm not in arms:
                continue
            for d, c in zip(DS, cols):
                m = pat.search(cells[c].replace("**", "")) if c < len(cells) else None
                x = arms[arm]["per_dataset"].get(d)
                if m is None or x is None:
                    if (m is None) != (x is None):
                        bad_.append(f"{arm} {d}: doc '{cells[c] if c < len(cells) else ''}' vs json {'absent' if x is None else s(x['dJF'])}")
                    continue
                mine = [s(x["dJF"])] + ([s(x["ci"][0]), s(x["ci"][1])] if m.group(2) else [])
                doc_ = [norm(g) for g in m.groups() if g]
                if mine != doc_:
                    bad_.append(f"{arm} {d}: doc {doc_} vs json {mine}")
            if name == "CORE_CONCLUSIONS" and "%" in cells[-1]:
                pct = [p.strip() for p in cells[-1].split("/")]
                for d, p in zip(DS, pct):
                    x = arms[arm]["per_dataset"].get(d)
                    if p.endswith("%") and x is not None and f"{round(x['recovered_frac'] * 100)}%" != norm(p):
                        bad_.append(f"{arm} {d} recovery: doc {p} vs json {x['recovered_frac']:.4f}")
        print(f"{name} §2 vs oracle_table.json:", "OK" if not bad_ else f"{len(bad_)} mismatch(es)")
        for b in bad_:
            print("   ", b)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows5, b0 = table5()
    rows6 = table6()
    table8()
    ot = table9()
    table10()
    fig5()
    crosscheck(rows5, rows6, b0, ot)
