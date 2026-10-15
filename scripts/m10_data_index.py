"""Write docs/m10/RESEARCH_DATA_INDEX.md: one page of direct raw-GitHub links to every research artifact (no auth needed;
the repository is public). Re-run after each results commit.   uv run python scripts/m10_data_index.py"""
import json
from pathlib import Path

BR = "exp/m10-oracle-bottleneck-decomposition"
RAW = f"https://raw.githubusercontent.com/msilot1001/sam2/{BR}/"
TREE = f"https://github.com/msilot1001/sam2/tree/{BR}/"
OUT = Path("docs/m10/RESEARCH_DATA_INDEX.md")

COLUMNS = """| 열 | 뜻 |
|---|---|
| set, split, unit, video, arm, video_idx | 데이터셋:split, dev/conf, 단위(영상/객체), 영상, arm, frame 번호 (1부터, 0은 prompt) |
| J, F, JF | 출력 mask의 J (IoU), F (경계), J&F |
| JF_G, JF_L, J_G, J_L | G view(전체 frame)와 L view(crop을 frame에 붙인 것) 각각의 J&F / J |
| Jv_L | L 출력 vs crop 영역 안 GT의 J |
| acc | SGGL 자기 게이트 통과 여부 (True면 출력 = L, 아니면 G) |
| src | 실제 출력 출처: L, G, E(빈 mask) |
| score_G, score_L, iou_G, iou_L | SAM 2 객체 점수 logit, 디코더 예측 IoU (선택된 가설) |
| cJ0–cJ2, cJ3–cJ5, cJ6 | G 원시 가설 3개, L 원시 가설 3개, 빈 mask의 GT J (존재 점수로 지우기 전) |
| best, JF_best | 7개 중 J 최대인 후보 번호와 그 J&F |
| ok_G, ok_L | view 예측이 그 view 영역 안 GT에 대해 맞음 (J ≥ 0.5, 둘 다 비면 맞음) |
| adm_G, adm_L | 그 frame이 view 기억에 기록됐는지 (OR7류에서 False 가능) |
| rew_G, rew_L | 그 frame 기억이 다시 부호화됐는지 (OR8: GT mask, OR5M: 최선 가설) |
| crop_y0, crop_x0, crop_side | L crop 위치·크기 (frame 좌표) |
| gt_empty, gt_area, gt_in_crop, area0 | GT가 비었는지, GT 면적, GT 중 crop 안 비율, prompt 면적 비율 |
| out_area, out_inter_gt, out_inter_other, other_area | 출력 면적, 출력∩GT, 출력∩주석된 다른 객체, 다른 객체 면적 |
| G_area, G_inter_gt, L_area, L_inter_gt | view별 면적과 GT 교집합 |
| latency_s, peak_vram_bytes, wall_s, machine | 지연, 최대 VRAM, 단위 실행 시간, 기계 |
| OR9 행 | J_full/F_full/JF_full (GT box, 전체 frame), *_crop (GT box + 완벽 crop), score_* |
"""


def raw(p):
    return f"[{p}]({RAW}{p})"


def main():
    L = ["# 연구 데이터 색인 (Claude Research / 외부 검토용)", "",
         "인증 없이 열리는 링크다(공개 GitHub 저장소, raw 파일).",
         f"",
         f"- **브랜치:** `{BR}`",
         f"- **기반 이력:** M7·M8·M9 브랜치를 모두 조상으로 포함한다.",
         f"- **원본 저장소:** Ubuntu 서버. 결과는 sha256으로 검증한 뒤 커밋한다.",
         f"- **대시보드:** https://research.sojak.io/#m10 (로그인 필요)", "",
         "## 1. 먼저 읽을 것", "",
         f"- {raw('docs/m10/DEEP_RESEARCH_BRIEF.txt')}: M10 중간 브리핑 (모든 수치, 잠정 결론, 질문)",
         f"- {raw('docs/m10/ORACLE_DEFINITIONS.md')}: oracle 정의, 추적 상태, 통계",
         f"- {raw('docs/m10/EXECUTION_LOG.md')}: 진행 기록, 사고, 우선순위 변경",
         f"- {raw('docs/m9/FINAL_ASSESSMENT.md')}, {raw('docs/m9/NEXT_DECISION_TREE.md')}: 직전 마일스톤 결론", ""]
    L += ["## 2. M10 사전 등록", ""]
    L += [f"- {raw(str(p))}" for p in sorted(Path("outputs/milestone10").glob("preregistration*.json"))]
    L += ["", "## 3. M10 분석 리포트 (JSON, `scripts/m10_analysis.py`)", ""]
    L += [f"- {raw(str(p))}" for p in sorted(Path("outputs/milestone10/reports").glob("*.json"))]
    L += ["", "## 4. M10 frame별 원자료 (단위 1개 = CSV 1개)", "",
          "| arm | 단위 수 (MOSEv2 / VOST / LVOS v2) | 폴더 | 예시 CSV |", "|---|---|---|---|"]
    for root in (Path("outputs/milestone10/runpod/runs"), *sorted(Path("outputs/milestone10/ubuntu").glob("runs_*"))):
        dirs = [root / "OR0"] if root.name.startswith("runs_") else sorted(d for d in root.iterdir() if d.is_dir())
        for d in dirs:
            fs = sorted(f for f in d.glob("*.csv"))
            if not fs:
                continue
            n = [sum(f.name.startswith(k + "__") for f in fs) for k in ("mosev2", "vost", "lvosv2")]
            name = d.name if root.name == "runs" else f"{d.name} (모델 {root.name[5:]}, RTX 2080)"
            L.append(f"| {name} | {n[0]} / {n[1]} / {n[2]} | [폴더]({TREE}{d}) | [{fs[0].name}]({RAW}{fs[0]}) |")
    L += ["", "※ OR6, C48은 사용자 결정으로 보류(부분 결과). 진행 중인 arm은 단위 수가 늘어난다.", "",
          "### CSV 열 사전", "", COLUMNS, "",
          "## 5. 데이터셋 manifest (GT만으로 추론 전에 고정, M9)", ""]
    L += [f"- {raw(str(p))}" for p in sorted(Path("outputs/milestone9/manifests").glob("*_units.json"))]
    L += ["", "## 6. M9 리포트", ""]
    L += [f"- {raw(str(p))}" for p in sorted(Path("outputs/milestone9/reports").glob("*.json"))]
    L += ["", "## 7. 문서 (마일스톤별)", ""]
    for m in ("m10", "m9", "m8", "m7"):
        L += [f"- **{m}:** " + ", ".join(raw(str(p)) for p in sorted(Path(f"docs/{m}").glob("*.md")) + sorted(Path(f"docs/{m}").glob("*.txt")))]
    L += ["", "## 8. 코드", "",
          f"- {raw('scripts/m10.py')}: oracle runner (`run_unit`, `rewrite_memory`, `box_unit`)",
          f"- {raw('scripts/m10_analysis.py')}: 통계 (영상 동일 가중, 영상 bootstrap)",
          f"- {raw('src/spatial/sggl.py')}: SGGL (G·L 이중 view, 자기 게이트)",
          f"- {raw('scripts/m8.py')}: M8/M9 arm runner", ""]
    OUT.write_text("\n".join(L) + "\n")
    print(OUT, len(L), "lines")


if __name__ == "__main__":
    main()
