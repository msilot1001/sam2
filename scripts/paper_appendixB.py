"""Appendix B: pre-registration and amendment history, M6 to M10 amend12.

    uv run python scripts/paper_appendixB.py      # writes docs/paper/appendixB_amendments.md

Rows come from
  - outputs/milestone*/**/preregistration*.json   (time, content, motive, "seen before?" evidence),
  - outputs/milestone{8,9}/arms.json              (arm freezes, grouped by frozen_kst),
  - docs/m{9,10}/EXECUTION_LOG.md                 (log time of each amendment; events without a JSON),
  - git: time (KST) and hash of the commit that first added the file / arm.
"관련 결과" is a curated map (RESULTS) with a source per entry; it is not in the pre-registration files.
Rows already in the output file whose source cell starts with "수동 추가" are kept on regeneration.
"""
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper/appendixB_amendments.md"
MANUAL = "수동 추가"
COLS = ["마일스톤", "번호", "시각(KST)", "내용", "동기", "결과를 본 뒤인가", "관련 결과", "출처"]
TOL = timedelta(minutes=5)  # "a few minutes": below this JSON and commit time count as equal
LOGS = {"M9": ("docs/m9/EXECUTION_LOG.md", "2026-10-14"), "M10": ("docs/m10/EXECUTION_LOG.md", "2026-10-15")}
MS_ORDER = ["M6", "M6.1", "M7", "M8", "M9", "M10"]

# Hand-checked "seen before?" labels where the JSON keywords are silent or misleading.
OVERRIDES = {
    "outputs/milestone10/preregistration_amend11.json": (
        "일부 (사후 진단)", "수정 10 두 회차 불통과(커밋 606191d)를 본 뒤 등록. 자체 출력(head_upper.json)은 등록 전에 없었다"),
    "outputs/milestone61/trackB/preregistration_pc2.json": (
        "일부", "1차 양성 대조(head_pc) 실패를 본 뒤 같은 frame으로 다시 설계 (docs/m61/OVERNIGHT_DEEP_RESEARCH_REPORT.md 5.1–5.2절)"),
    "outputs/milestone7/preregistration_r3.json": (
        "일부", "M3-vel 단위별 손실을 본 뒤 그 단위를 성공 조건에 넣은 개정. 자체 출력 전 (커밋 제목 'before any output', docs/m7/STAGE2_REVISIONS.md 2절)"),
    "outputs/milestone7/preregistration_r4_vk.json": (
        "아니오", "VK 결과 커밋 6368503(10-14 14:27)보다 먼저 커밋됨"),
    "outputs/milestone8/arms.json@2026-10-14 16:14": (
        "일부", "M8 사전 등록과 Stage 1 diag 7단위 기록 뒤, 첫 인과 출력(18:41) 전 (docs/m8/STAGE2_CAUSAL_RESULTS.md, REVIEW_RESPONSE.md 9번)"),
    "outputs/milestone8/arms.json@2026-10-14 17:06": (
        "아니오", "기준선. 첫 인과 출력 18:41 전 (docs/m8/STAGE2_CAUSAL_RESULTS.md)"),
    "outputs/milestone8/arms.json@2026-10-14 18:13": (
        "일부", "Stage 1 후보 31개를 본 뒤 만든 상한(배포 불가). 첫 인과 출력 18:41 전"),
    "outputs/milestone8/arms.json@2026-10-14 18:34": (
        "일부", "진출 계열이 없어 Stage 1 재생 순이득이 양수인 조합을 진단용으로 고름 (docs/m8/STAGE2_CAUSAL_RESULTS.md '사전 등록과 다른 점')"),
    "outputs/milestone8/arms.json@2026-10-14 18:38": (
        "일부", "구성요소 인과 결과 전, M3vel 부분 결과만 있었음 (docs/m8/COMBINATION_RESULTS.md 1절)"),
    "outputs/milestone9/arms.json@2026-10-14 23:13": (
        "아니오", "M9 사전 등록(23:18, created_before_any_m9_sam2_output_on_new_data)보다 앞섬. 결과 뒤 변경 없음 (docs/m9/SCALE_CONTROL_RESULTS.md)"),
    "outputs/milestone9/arms.json@2026-10-15 03:13": (
        "아니오", "수정 3과 같음 (docs/m9/REGRESSION_AND_FAILURE_AUDIT.md)"),
    "outputs/milestone9/preregistration_amend2.json": (
        "아니오", "docs/m9/REGRESSION_AND_FAILURE_AUDIT.md: 수정 1–4는 해당 출력 전에 고정"),
    "outputs/milestone9/preregistration_amend3.json": (
        "아니오", "docs/m9/REGRESSION_AND_FAILURE_AUDIT.md: 수정 1–4는 해당 출력 전에 고정"),
}

FR = "docs/m10/FINAL_REPORT.md"
# Related result per row: (text, source). Keys: JSON path, "<arms.json>@<frozen_kst>", or "log:<ms>:<HH:MM>[:arm]".
RESULTS = {
    "outputs/milestone6/preregistration.json": ("A1 관문 불통과(작은 객체 ΔJ&F +0.017 < 0.03). A2·B3 trigger, 트랙 C 규칙 미충족", "preregistration_r2.json created_after"),
    "outputs/milestone6/preregistration_r2.json": ("A1z 해로움(−0.044), H3 반증. A2x +0.018 회귀 0, 관문 미달(사후 탐색이라 통과 자격 없음)", "docs/m6/OVERNIGHT_EXPERIMENT_REPORT.md"),
    "outputs/milestone6/preregistration_r3.json": ("S1 불통과. 객체별 배율 oracle 상한 약 +0.018", "preregistration_r4.json created_after"),
    "outputs/milestone6/preregistration_r4.json": ("학습 head 파일럿 포함 모든 트랙 관문 불통과", "preregistration_r5.json created_after"),
    "outputs/milestone6/preregistration_r5.json": ("A2x − A2c +0.008 [+0.001, +0.014]. 탐색 비교, C 봉인 유지", "docs/m6/OVERNIGHT_EXPERIMENT_REPORT.md"),
    "outputs/milestone61/preregistration.json": ("TPLR 기각, 보정 gate 이득 없음(선택 포화), head v2 이미지 수준 음성", "preregistration_r2.json created_after"),
    "outputs/milestone61/trackB/preregistration_pc2.json": ("진단 후 양성 대조 통과(+0.108)", "docs/m61/OVERNIGHT_DEEP_RESEARCH_REPORT.md"),
    "outputs/milestone61/preregistration_r2.json": ("A3 − A2x +0.0022 [+0.0003, +0.006], 기준 +0.01 미달", "preregistration_r3.json created_after"),
    "outputs/milestone61/preregistration_r3.json": ("F2 전체 +0.001. A2x − F2 작은 객체 +0.018 [+0.009, +0.026]", "preregistration_r4.json created_after"),
    "outputs/milestone61/preregistration_r4.json": ("A2x 복제 실패 +0.009 [−0.019, +0.032]. miami-surf obj5 −0.477", "preregistration_r5.json created_after"),
    "outputs/milestone61/preregistration_r5.json": ("A2v 개발 A2x 대비 −0.0009(CI 하한 −0.0036). 복제 +0.020은 사후 근거", "docs/m61/FINAL_RESEARCH_DIRECTION.md"),
    "outputs/milestone7/preregistration.json": ("Stage 1 재생: S/F/M/R 모두 REJECT 또는 INCONCLUSIVE", "preregistration_r2.json created_after"),
    "outputs/milestone7/preregistration_r2.json": ("T0–T4 DEV 모두 REJECT(양성 대조 통과). F1-blend.7·M1-always REJECT", "preregistration_r4_vk.json created_after"),
    "outputs/milestone7/preregistration_r2_amend.json": ("M3-vel +0.0046 [−0.0001, +0.0115] INCONCLUSIVE. S 계열은 인과 확인 없음", "docs/m7/FINAL_ALGORITHM_DECISION.md"),
    "outputs/milestone7/preregistration_r3.json": ("M3-vel-r +0.0044, 표적 회귀 못 줄임 → 개정 실패", "docs/m7/STAGE2_REVISIONS.md 3절"),
    "outputs/milestone7/preregistration_r4_vk.json": ("VK: SGGL − P5 +0.0173, M3-vel − SGGL +0.0013. 결정 KEEP_SGGL", "docs/m7/STAGE2_REVISIONS.md 4절"),
    "outputs/milestone8/preregistration.json": ("Stage 1 진출 조건 통과 계열 없음. Stage 2 FLOWVEL +0.0130 [+0.0010, +0.0285] PROMISING", "docs/m8/STAGE1_DIAGNOSTIC_RESULTS.md, STAGE2_CAUSAL_RESULTS.md"),
    "outputs/milestone8/preregistration_amend1.json": ("_pt 변형 포함, Stage 1 진출 계열 없음", "docs/m8/STAGE1_DIAGNOSTIC_RESULTS.md"),
    "outputs/milestone8/preregistration_amend2.json": ("H_s8 수정 후에도 Stage 1 진출 계열 없음", "docs/m8/STAGE1_DIAGNOSTIC_RESULTS.md"),
    "outputs/milestone8/preregistration_vk.json": ("VK FLOWVEL +0.0002 [−0.0006, +0.0008]. 비회귀 통과, 이득 없음", "docs/m8/STAGE2_CAUSAL_RESULTS.md"),
    "outputs/milestone8/preregistration_vk_amend1.json": ("VK FLOWCARRY −0.00005 [−0.0001, +0.0000]", "docs/m8/STAGE2_CAUSAL_RESULTS.md"),
    "outputs/milestone8/arms.json@2026-10-14 16:14": ("FLOWVEL +0.0130 [+0.0010, +0.0285], M3vel +0.0046. FLOWVEL은 M9 새 데이터에서 일반화 실패", "docs/m8/STAGE2_CAUSAL_RESULTS.md, docs/m9/FLOWVEL_GENERALIZATION.md"),
    "outputs/milestone8/arms.json@2026-10-14 17:06": ("기계 짝 SGGL 기준선", "outputs/milestone8/arms.json"),
    "outputs/milestone8/arms.json@2026-10-14 18:13": ("ORACLE_POOL +0.0136, _ADMIT +0.0169 (배포 불가 상한)", "docs/m8/STAGE2_CAUSAL_RESULTS.md"),
    "outputs/milestone8/arms.json@2026-10-14 18:34": ("RESC_OCLR −0.0001, _ADMIT +0.0046, RESC_CPFR −0.0070. 모두 REJECTED", "docs/m8/STAGE2_CAUSAL_RESULTS.md"),
    "outputs/milestone8/arms.json@2026-10-14 18:38": ("K_OCLR_M3 +0.0071, K_OCLR_FLOWVEL +0.0121. 결합 이득 없음", "docs/m8/COMBINATION_RESULTS.md"),
    "outputs/milestone8/arms.json@2026-10-14 19:08": ("FLOWCARRY +0.0141, CAMCARRY +0.0137, FV_PRESENT +0.0016", "docs/m8/STAGE2_CAUSAL_RESULTS.md"),
    "outputs/milestone8/arms.json@2026-10-14 19:43": ("+0.0103, FLOWVEL 대비 −0.0027. 이득 없음, 위험 증가", "docs/m8/COMBINATION_RESULTS.md"),
    "outputs/milestone8/arms.json@2026-10-14 20:50": ("+0.0161 [+0.0014, +0.0345], RunPod FLOWVEL과 상관 0.999998 (계산 재현)", "docs/m8/STAGE2_CAUSAL_RESULTS.md 5절"),
    "outputs/milestone9/preregistration.json": ("FLOWVEL 일반화 실패(MOSEv2 −0.0142, VOST +0.0046, LVOS −0.0021). SCALE 불통과. JOINT 미실행", "docs/m9/EXECUTION_LOG.md, FINAL_ASSESSMENT.md"),
    "outputs/milestone9/arms.json@2026-10-14 23:13": ("SCALE 불통과, PCCARRY MOSEv2 −0.0031 중단, JOINT 미실행", "docs/m9/EXECUTION_LOG.md, FINAL_ASSESSMENT.md 6절"),
    "outputs/milestone9/arms.json@2026-10-15 00:42": ("수정 1 행 참조", "docs/m9/CONSENSUS_RESULTS.md"),
    "outputs/milestone9/preregistration_amend1.json": ("CONSENSUS conf −0.0503 [−0.0983, −0.0094], −0.05 미만 15단위 → 반증", "docs/m9/EXECUTION_LOG.md 01:53"),
    "outputs/milestone8/arms.json@2026-10-14 16:14": (
        "일부", "M8 사전 등록과 Stage 1 diag 7단위 기록 뒤, 첫 인과 출력(18:41) 전 (docs/m8/STAGE2_CAUSAL_RESULTS.md, REVIEW_RESPONSE.md 9번)"),
    "outputs/milestone8/arms.json@2026-10-14 17:06": (
        "아니오", "기준선. 첫 인과 출력 18:41 전 (docs/m8/STAGE2_CAUSAL_RESULTS.md)"),
    "outputs/milestone8/arms.json@2026-10-14 18:13": (
        "일부", "Stage 1 후보 31개를 본 뒤 만든 상한(배포 불가). 첫 인과 출력 18:41 전"),
    "outputs/milestone8/arms.json@2026-10-14 18:34": (
        "일부", "진출 계열이 없어 Stage 1 재생 순이득이 양수인 조합을 진단용으로 고름 (docs/m8/STAGE2_CAUSAL_RESULTS.md '사전 등록과 다른 점')"),
    "outputs/milestone8/arms.json@2026-10-14 18:38": (
        "일부", "구성요소 인과 결과 전, M3vel 부분 결과만 있었음 (docs/m8/COMBINATION_RESULTS.md 1절)"),
    "outputs/milestone9/arms.json@2026-10-14 23:13": (
        "아니오", "M9 사전 등록(23:18, created_before_any_m9_sam2_output_on_new_data)보다 앞섬. 결과 뒤 변경 없음 (docs/m9/SCALE_CONTROL_RESULTS.md)"),
    "outputs/milestone9/arms.json@2026-10-15 03:13": (
        "아니오", "수정 3과 같음 (docs/m9/REGRESSION_AND_FAILURE_AUDIT.md)"),
    "outputs/milestone9/preregistration_amend2.json": ("H_drift. MOSEv2 precision 중앙값 0.07, VOST '대상 밖' 73%", "docs/m9/EXECUTION_LOG.md 03:54, 04:16"),
    "outputs/milestone9/arms.json@2026-10-15 03:13": ("수정 3 행 참조", "docs/m9/REGRESSION_AND_FAILURE_AUDIT.md"),
    "outputs/milestone9/preregistration_amend3.json": ("NULLJIT MOSEv2 −0.0011 (9↑/8↓), VOST +0.0089 (9/7). '−0.05 단위' 규칙은 기제를 못 가름", "docs/m9/EXECUTION_LOG.md 04:57"),
    "outputs/milestone9/preregistration_amend4.json": ("LVOS FLOWVEL −0.0021 [−0.0110, +0.0071], SCALE −0.0015, NULLJIT −0.0073", "docs/m9/EXECUTION_LOG.md 05:47, 06:20"),
    "log:M9:02:02": ("VOST 주 검정(Pod) FLOWVEL +0.0046, 2080 재현 +0.0051, 상관 0.999", "docs/m9/EXECUTION_LOG.md 03:27"),
    "outputs/milestone10/preregistration.json": ("OR8 +0.149 / +0.330 최대, OR7 +0.054 / +0.049, OR4 +0.020 / +0.062 (MOSEv2 / VOST)", FR + " 2절"),
    "outputs/milestone10/preregistration_amend1.json": ("VOST: small +0.037, base+ +0.021, large −0.005 (모두 CI가 0 포함)", FR + " 2절"),
    "outputs/milestone10/preregistration_amend2.json": ("OR9 J&F 0.852 / 0.744 / 0.921. VOST에서 OR8보다 0.09 낮음 → '용량 상한' 해석 철회", FR + " 2절, 5절 3번"),
    "outputs/milestone10/preregistration_amend3.json": ("C58 +0.207 / +0.398, C75 +0.152 / +0.221, OR8K10 +0.069 / +0.090. C47·C48 보류", FR + " 2절"),
    "outputs/milestone10/preregistration_amend4.json": ("OR0F = OR0, 12,989 frame |Δ| = 0. 선택기 개발 자료", "docs/m10/EXECUTION_LOG.md 12:41"),
    "outputs/milestone10/preregistration_amend5.json": ("다섯 규칙 모두 불통과, 합동 정밀도 0.22–0.40. 온라인 arm 미실행", FR + " 4절"),
    "outputs/milestone10/preregistration_amend7.json": ("OR0T 109단위 실행(runs/OR0T). 계획한 오프라인 분석 결과는 문서에 없음. 수정 10의 OR0K가 OR0T 기록을 확장", "preregistration_amend10.json arms_doc"),
    "outputs/milestone10/preregistration_amend9.json": ("SGGL − B0: MOSEv2 +0.054 [+0.009, +0.109], VOST −0.036 [−0.081, −0.002], LVOS +0.012 [−0.069, +0.126]. P5 몫 +0.002–0.013", "docs/m10/BASELINES.md"),
    "outputs/milestone10/preregistration_amend10.json": ("H1·H2 모두 불통과(두 회차). 온라인 arm 미실행", "docs/m10/HEAD_RESULTS.md"),
    "outputs/milestone10/preregistration_amend11.json": ("τ* 도달 모델 없음. 합동 최대 b(H1) 0.414, R_piou 0.391", "reports/head_upper.json, docs/m10/HEAD_RESULTS.md"),
    "outputs/milestone10/preregistration_amend12.json": ("MOSEv2 ≤ 0.45(두 가설), VOST 구분 불가, LVOS H1 ≥ 0.70이나 라벨 섞은 영 대조도 비슷", "docs/m10/HEAD_RESULTS.md '같은 데이터셋 안 상한'"),
    "log:M10:11:19": ("우선 arm 12:40 완료. OR2·OR3·OR1M·OR6·OR8C0·C47·C48 미측정(OR6 18단위·C48 10단위 부분 결과는 판정에 안 씀)", FR + " 2절, 5절 8번"),
    "log:M10:11:19:OR1M": ("미측정. 존재 정보 누수를 oracle 이득에서 분리하지 못함", FR + " 5절 8번"),
}
# Log events without a JSON: (milestone, HH:MM) -> (번호, seen label, evidence). Matched by keyword below.
EVENT_KEYS = re.compile(r"우선순위 변경|분석 규칙")
EVENTS = {
    ("M10", "11:19"): ("우선순위 변경 (사용자 지시)", "일부", "단일 arm 결과(OR7·OR8·OR4, 10:40경 완료)를 본 뒤 결정. 외부 검토가 '결론 쪽으로 기울었다'고 지적 (FINAL_REPORT 5절 8번)"),
    ("M9", "02:02"): ("분석 규칙 (기계)", "아니오", "로그: 'VOST 결과 보기 전 고정'. MOSEv2 결과는 이미 봄"),
}
DEFERRED_ROWS = ("OR1M",)  # deferred arms that get their own row (named in FINAL_REPORT 5절 8번)


def rel(p):
    return str(Path(p).relative_to(ROOT))


def git(*args):
    env = dict(os.environ, TZ="Asia/Seoul")
    out = subprocess.run(["git", *args], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def first_commit(path, pickaxe=None):
    """(hash, datetime KST, subject) of the earliest commit that added the file (or the string)."""
    sel = ["-S", pickaxe] if pickaxe else ["--diff-filter=A"]
    lines = git("log", *sel, "--format=%h|%ad|%s", "--date=format-local:%Y-%m-%d %H:%M", "--", path)
    if not lines:
        return None, None, ""
    h, ad, subj = lines[-1].split("|", 2)
    return h, datetime.strptime(ad, "%Y-%m-%d %H:%M"), subj


def parse_dt(s):
    m = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})", s or "")
    return datetime.strptime(" ".join(m.groups()), "%Y-%m-%d %H:%M") if m else None


def hm(dt, ref=None):
    return dt.strftime("%H:%M") if ref and ref.date() == dt.date() else dt.strftime("%m-%d %H:%M")


def is_utc(jdt, cdt):
    return bool(jdt and cdt) and timedelta(hours=8, minutes=55) <= cdt - jdt <= timedelta(hours=9, minutes=30)


def time_cell(jdt, cdt, ldt=None):
    """One time if JSON, commit (and log) agree within TOL; otherwise all of them."""
    if jdt is None:
        return f"커밋 {hm(cdt)}" if cdt else "?"
    parts = [f"JSON {hm(jdt)}"]
    if cdt and abs(cdt - jdt) > TOL:
        parts[0] += " (UTC로 보임)" if is_utc(jdt, cdt) else ""
        parts.append(f"커밋 {hm(cdt, jdt)}" + (" (JSON보다 앞섬)" if cdt < jdt - TOL else ""))
    if ldt and abs(ldt - jdt) > TOL:
        parts.append(f"로그 {hm(ldt, jdt)}")
    return hm(jdt) if len(parts) == 1 else " / ".join(parts)


def short(s, n):
    s = re.sub(r"\s+", " ", s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def milestone(path):
    ms = re.search(r"milestone(\d+)", path).group(1)
    return "M6.1" if ms == "61" else f"M{ms}"


def label_of(path):
    ms = milestone(path)
    name = Path(path).stem.replace("preregistration", "")
    if name == "":
        return ms, "사전 등록"
    num = {"_r2_amend": "r2 수정", "_vk": "VK 사전 등록", "_vk_amend1": "VK 수정 1", "_pc2": "pc2 (트랙 B)", "_r4_vk": "r4 (VK)"}.get(name)
    if num:
        return ms, num
    m = re.fullmatch(r"_(r|amend)(\d+)", name)
    return ms, (f"r{m.group(2)}" if m.group(1) == "r" else f"수정 {m.group(2)}")


def seen(d, extra=""):
    """예 / 아니오 / 일부 from the file's own words. Returns (label, evidence)."""
    t = json.dumps(d, ensure_ascii=False) + " " + extra
    m = re.search(r"post hoc for (\w+)", t)
    if "post_hoc_disclosure" in d or re.search(r"chosen AFTER|frozen AFTER seeing", t):
        ev = d.get("post_hoc_disclosure") or re.search(r"[^\"]*(?:chosen|frozen) AFTER[^\"]*", t).group(0)
        return "예", short(ev, 150)
    if m:
        return f"일부 ({m.group(1)}: 예)", short(d.get("seen_before_this_file", ""), 150)
    for k in ("seen_before_this_file", "also_seen_before_this_amendment", "disclosure", "design_data"):
        if isinstance(d, dict) and k in d:
            return "일부", f"{k}: " + short(d[k], 140)
    hit = re.search(r"[^.\"]*(POST-HOC|Post-hoc addition|EXPLORATORY|after diag|frozen after|Designed from)[^.\"]*", t)
    if hit:
        return "일부", short(hit.group(0), 150)
    if isinstance(d, dict) and "before_any_output_of" in d:
        return "아니오", "before_any_output_of: " + short(d["before_any_output_of"], 80)
    hit = re.search(r"(created_before_any\w*)\": true", t)
    if hit:
        return "아니오", f"{hit.group(1)}: true"
    hit = re.search(r"[^.\"]*(?:before any|registered before|[Ff]rozen before|[Dd]ecided before)[^.\"]*", t)
    if hit:
        return "아니오", short(hit.group(0), 100)
    return "불명", ""


def content(d):
    out = []
    if "arms" in d:
        a = d["arms"]
        out.append("arm: " + (", ".join(a) if isinstance(a, (list, dict)) else short(a, 120)))
    for k in ("arm", "candidate", "finalists", "addition", "amendment", "fix", "what", "question", "purpose", "measurement", "manifest", "T_heads"):
        if k in d and not out and k != motive_key(d):
            out.append(short(d[k], 120))
    if "before_any_output_of" in d:
        out.append("출력 전: " + short(d["before_any_output_of"], 80))
    if "status" in d:
        out.append("상태: " + short(d["status"], 70))
    for k in d:
        if "(replaces" in k:
            out.append("교체: " + k)
        m = re.search(r"\(added (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", k)
        if m:
            out.append(f"추가 필드 {m.group(1)[5:]}: {k.split(' (')[0]}")
    if "created_kst_note" in d:
        out.append("시각 정정 공개: " + short(d["created_kst_note"], 70))
    return " · ".join(out) or short(d, 100)


MOTIVE_KEYS = ("why", "created_after", "observation", "context", "purpose", "question", "source", "bug", "text",
               "post_hoc_disclosure", "incumbent", "doc")


def motive_key(d):
    return next((k for k in MOTIVE_KEYS if k in d), None)


def motive(d):
    k = motive_key(d)
    if k is None:
        return short(d.get("status", ""), 170)
    return short(d[k].split("\n")[0] if k == "doc" else d[k], 170)


def result(key):
    r = RESULTS.get(key)
    return f"{r[0]} ({r[1]})" if r else ""


def degraded_result(arms):
    """VXO / VXM / N1M / N2X dev results, read from reports/degraded.json."""
    p = ROOT / "outputs/milestone10/reports/degraded.json"
    d = json.loads(p.read_text())["arms"]
    f = lambda a: f"{a} " + " / ".join(f"{d[a][ds]['dJF']:+.3f}" for ds in ("mosev2", "vost", "lvosv2"))
    return "; ".join(f(a) for a in arms) + " (MOSEv2 / VOST / LVOS ΔJ&F vs OR0; " + rel(p) + ", " + FR + " 4.1절)"


def read_log(ms):
    """[(datetime, text)] from an EXECUTION_LOG table; empty time cells continue the previous row."""
    path, day = LOGS[ms]
    day = datetime.strptime(day, "%Y-%m-%d")
    rows, prev = [], None
    for ln in (ROOT / path).read_text().splitlines():
        if not ln.startswith("|") or ln.startswith("|---"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        t, text = cells[0], cells[-1]
        if t == "" and rows:
            rows[-1] = (rows[-1][0], rows[-1][1] + " " + text)
            continue
        m = re.search(r"(\d{2}):(\d)(\d|x)", t)
        if not m:
            continue
        md = re.search(r"(\d{2})-(\d{2}) ", t)
        if md:
            day = day.replace(month=int(md.group(1)), day=int(md.group(2)))
        dt = day.replace(hour=int(m.group(1)), minute=int(m.group(2) + m.group(3).replace("x", "0")))
        if prev and dt < prev - timedelta(hours=6) and not md:
            day += timedelta(days=1)
            dt += timedelta(days=1)
        rows.append((dt, text))
        prev = dt
    return path, rows


def auto_rows():
    rows = []
    logs = {ms: read_log(ms) for ms in LOGS}
    amend_log = {}  # (ms, n) -> log datetime
    for ms, (_, lrows) in logs.items():
        for dt, text in lrows:
            for m in re.finditer(r"사전 등록 수정 (\d+)|수정 (\d+)(?: \(사용자 지시\))?(?::| 사전 등록)", text):
                amend_log.setdefault((ms, int(m.group(1) or m.group(2))), dt)
            if re.search(r"사전 등록 고정|`preregistration\.json` 고정", text):
                amend_log.setdefault((ms, 0), dt)  # the base pre-registration

    files = sorted(p for p in ROOT.glob("outputs/milestone*/**/preregistration*.json") if ".conflict" not in p.name)
    for p in files:
        path = rel(p)
        d = json.loads(p.read_text())
        ms, num = label_of(path)
        h, cdt, subj = first_commit(path)
        jdt = parse_dt(d.get("created_kst") or d.get("created"))
        n = re.search(r"amend(\d+)\.json$", path)
        ldt = amend_log.get((ms, int(n.group(1)) if n else 0)) if n or num == "사전 등록" else None
        lab, ev = OVERRIDES.get(path) or seen(d)
        res = result(path)
        if path.endswith("amend6.json"):
            res = degraded_result(["VXO", "VXM"]) + ". 수정 6 기준은 수정 8로 대체"
        if path.endswith("amend8.json"):
            res = "수정 8 기준 불통과(VXO·VXM). 대조 " + degraded_result(["N1M", "N2X"])
        rows.append(dict(ms=ms, num=num, dt=(jdt + timedelta(hours=9) if is_utc(jdt, cdt) else jdt or cdt), time=time_cell(jdt, cdt, ldt), content=content(d),
                         motive=motive(d), seen=f"{lab} — {ev}" if ev else lab, result=res,
                         src=f"`{path}` ({h})" if h else f"`{path}`"))

    for arms in ("outputs/milestone8/arms.json", "outputs/milestone9/arms.json"):
        d = json.loads((ROOT / arms).read_text())
        groups = {}
        for name, a in d.items():
            groups.setdefault(a["frozen_kst"], []).append((name, a))
        ms = milestone(arms)
        for fk, members in groups.items():
            jdt = parse_dt(fk)
            h, cdt, _ = first_commit(arms, pickaxe=f'"name": "{members[0][0]}"')
            whys = {n: a.get("why", "") for n, a in members}
            lab, ev = OVERRIDES.get(f"{arms}@{fk}") or seen(whys)
            rows.append(dict(ms=ms, num="arm 고정", dt=jdt, time=time_cell(jdt, cdt),
                             content="arm: " + ", ".join(whys), motive=short(members[0][1].get("why", ""), 170),
                             seen=f"{lab} — {ev}" if ev else lab, result=result(f"{arms}@{fk}"),
                             src=f"`{arms}` frozen_kst ({h})"))

    for ms, (lpath, lrows) in logs.items():
        for dt, text in lrows:
            if not EVENT_KEYS.search(text):
                continue
            key = (ms, dt.strftime("%H:%M"))
            num, lab, ev = EVENTS.get(key, ("로그 사건", "불명", ""))
            base = dict(ms=ms, dt=dt, time=f"로그 {hm(dt)}", seen=f"{lab} — {ev}", src=f"`{lpath}` {key[1]} (JSON 없음)")
            rows.append(dict(base, num=num, content=short(text.replace("**", ""), 260), motive="사용자 지시" if "사용자" in text else "",
                             result=result(f"log:{ms}:{key[1]}")))
            if "DEFERRED" in text:
                for arm in DEFERRED_ROWS:
                    if arm in text:
                        rows.append(dict(base, num=f"{arm} 보류", content=f"{arm} 실행 안 함 (DEFERRED, 사용자 결정)",
                                         motive="11:19 우선순위 변경", result=result(f"log:{ms}:{key[1]}:{arm}")))
    return rows


def manual_rows():
    """Rows marked '수동 추가' in the current output file, kept verbatim."""
    if not OUT.exists():
        return []
    rows = []
    for ln in OUT.read_text().splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]
        if ln.startswith("|") and len(cells) == len(COLS) and cells[-1].startswith(MANUAL):
            m = re.search(r"(\d{2})-(\d{2}) (\d{2}):(\d{2})", cells[2])
            dt = datetime(2026, *map(int, m.groups())) if m else None
            rows.append(dict(ms=cells[0], line=ln, dt=dt))
    return rows


def sort_key(r):
    return (MS_ORDER.index(r["ms"]), r["dt"] or datetime.max)


def render(rows):
    head = """# 부록 B. 사전 등록과 수정 이력 (M6 – M10 수정 12)

`scripts/paper_appendixB.py`가 만든 표다. 다시 만들 때 '수동 추가' 행은 그대로 남는다.

**열 설명**
- **번호:** 파일 이름에서 온다. r = 회차, 수정 = amend. 'arm 고정'은 `arms.json`의 `frozen_kst` 묶음이다.
- **시각(KST):** JSON의 `created_kst`(또는 `created`)와, 그 파일을 처음 추가한 커밋 시각이다. 5분 넘게 다르면 둘 다 적는다. 실행 기록(EXECUTION_LOG)에서 그 수정이 처음 나오는 행의 시각이 다르면 '로그'로 덧붙인다. 'UTC로 보임'은 JSON 시각 + 9시간이 커밋 시각과 맞는 경우다.
- **내용·동기:** JSON 원문 발췌다(대부분 영어). 동기는 `why`, `created_after`, `observation` 같은 필드다.
- **결과를 본 뒤인가:**
  - 아니오 = 그 파일이 다루는 출력이 나오기 전에 고정했다.
  - 일부 = 관련 결과(같은 데이터의 일부, 이전 회차, 같은 규칙의 재생)를 본 뒤였고 공개했다.
  - 예 = 그 결과를 보고 정했다(사후).
  - 근거 문구는 '—' 뒤에 붙인다. JSON 필드(`seen_before_this_file`, `disclosure`, `post_hoc_disclosure`, `status`의 EXPLORATORY 등)에서 자동으로 뽑는다. 자동 판정이 맞지 않는 파일은 스크립트의 `OVERRIDES`에 근거와 함께 고정했다.
- **관련 결과:** 사전 등록 파일에는 없다. 스크립트의 `RESULTS` 표에 출처와 함께 적었다.
- **출처:** 파일과 처음 추가한 커밋. '수동 추가'는 JSON이 없어 문서에서 옮긴 행이다.

"""
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in sorted(rows, key=sort_key):
        if "line" in r:
            lines.append(r["line"])
        else:
            lines.append("| " + " | ".join(cell(r[k]) for k in ("ms", "num", "time", "content", "motive", "seen", "result", "src")) + " |")
    foot = f"""

**출처**
- 사전 등록 JSON: `outputs/milestone{{6,61,7,8,9,10}}/**/preregistration*.json` ({sum(1 for r in rows if 'line' not in r and 'preregistration' in r['src'])}개)
- arm 고정: `outputs/milestone8/arms.json`, `outputs/milestone9/arms.json`
- 실행 기록: `docs/m9/EXECUTION_LOG.md`, `docs/m10/EXECUTION_LOG.md` (M6–M8에는 실행 기록 파일이 없다)
- 결과: `RESULTS` 표의 각 출처, `outputs/milestone10/reports/degraded.json`, `{FR}`
- 커밋 시각: `TZ=Asia/Seoul git log --diff-filter=A --format=%ad --date=format-local:'%Y-%m-%d %H:%M' -- <file>` (arm은 `git log -S`)
- 생성: `uv run python scripts/paper_appendixB.py` → `docs/paper/appendixB_amendments.md`
"""
    return head + "\n".join(lines) + foot


def main():
    rows = auto_rows()
    unknown = [r["src"] for r in rows if r["seen"].startswith("불명")]
    rows += manual_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows))
    print(f"wrote {rel(OUT)}: {len(rows)} rows ({sum('line' in r for r in rows)} manual)")
    for ms in MS_ORDER:
        print(f"  {ms}: {sum(r['ms'] == ms for r in rows)}")
    if unknown:
        print("seen-label 불명:", *unknown, sep="\n  ")


if __name__ == "__main__":
    main()
