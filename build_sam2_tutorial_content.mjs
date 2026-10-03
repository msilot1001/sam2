import fs from "node:fs";
import path from "node:path";

const root = String.raw`C:\Users\msilo\OneDrive\과제연구2`;
const outDir = path.join(root, "sam2_tutorial_build");
fs.mkdirSync(outDir, { recursive: true });

const read = (name) => fs.readFileSync(path.join(root, name), "utf8").replace(/\r\n/g, "\n");
const intro = read("draft_intro.md");
const foundation = read("draft_foundations.md");
const memory = read("draft_sam2_memory.md");
const evaluation = read("draft_eval_research.md");
const appendix = read("draft_appendix.md");

function sliceByHeading(text, startPattern, endPattern = null) {
  const lines = text.split("\n");
  const start = lines.findIndex((line) => startPattern.test(line));
  if (start < 0) throw new Error(`Could not find ${startPattern}`);
  let end = lines.length;
  if (endPattern) {
    const relative = lines.slice(start + 1).findIndex((line) => endPattern.test(line));
    if (relative >= 0) end = start + 1 + relative;
  }
  return lines.slice(start, end);
}

function injectBefore(lines, pattern, inserted) {
  const at = lines.findIndex((line) => pattern.test(line));
  if (at < 0) throw new Error(`Could not inject before ${pattern}`);
  return [...lines.slice(0, at), ...inserted, ...lines.slice(at)];
}

function injectAfter(lines, pattern, inserted) {
  const at = lines.findIndex((line) => pattern.test(line));
  if (at < 0) throw new Error(`Could not inject after ${pattern}`);
  return [...lines.slice(0, at + 1), ...inserted, ...lines.slice(at + 1)];
}

const parts = [];
parts.push({ lines: intro.split("\n"), headingOffset: -1, titleLine: true });
parts.push({ lines: ["# PART A — 이미지와 컴퓨터비전 과제의 기초"], headingOffset: 0 });
parts.push({ lines: sliceByHeading(foundation, /^# 1\./, /^# 4\./), headingOffset: 1 });
parts.push({ lines: ["# PART B — 신경망·학습·프롬프트와 모델 적응"], headingOffset: 0 });
parts.push({ lines: sliceByHeading(foundation, /^# 4\./, /^# 7\./), headingOffset: 1 });
parts.push({ lines: sliceByHeading(foundation, /^# 8\./, /^# 10\./), headingOffset: 1 });
parts.push({ lines: sliceByHeading(foundation, /^# 13\./, /^# 17\./), headingOffset: 1 });
parts.push({ lines: ["# PART C — SAM, SAM 2, SAM 2.1과 비디오 메모리"], headingOffset: 0 });
let mem = sliceByHeading(memory, /^## 2\./, /^## 13\./);
mem = injectBefore(mem, /^## 4\./, [
  "[FIGURE2]",
  "그림 2. SAM 2의 한 frame 처리 흐름: 현재 feature와 prompt가 과거 memory를 읽고, 결과 mask가 다시 미래 memory가 되는 순환 구조다.",
]);
mem = injectBefore(mem, /^## 7\./, [
  "[FIGURE3]",
  "그림 3. 같은 부분 가림에서도 무조건 memory를 갱신하면 drift가 커지지만, 위험 기반 선택적 기록과 anchor recovery를 쓰면 잘못된 identity의 자기증폭을 막을 수 있다.",
]);
parts.push({ lines: mem, headingOffset: 0 });
parts.push({ lines: ["# PART D — 분할 평가와 신뢰도·불확실성 보정"], headingOffset: 0 });
let evalMetrics = sliceByHeading(evaluation, /^# 1\./, /^# 13\./);
evalMetrics = injectAfter(evalMetrics, /^# 3\./, [
  "[FIGURE5]",
  "그림 5. 정답과 예측의 교집합·합집합을 직접 세어 IoU와 Dice를 계산하는 예시다.",
]);
parts.push({ lines: evalMetrics, headingOffset: 1 });
parts.push({ lines: ["# PART E — 공정한 실험, 통계, 데이터셋과 계산 비용"], headingOffset: 0 });
parts.push({ lines: sliceByHeading(evaluation, /^# 13\./, /^# 24\./), headingOffset: 1 });
parts.push({ lines: ["# PART F — 제안 연구를 처음부터 다시 조립하기"], headingOffset: 0 });
let evalProposal = sliceByHeading(evaluation, /^# 24\./, null);
evalProposal = injectBefore(evalProposal, /^# 25\./, [
  "[FIGURE4]",
  "그림 4. 내부 score를 바로 threshold하지 않고 실제 미래 실패위험으로 calibration한 뒤 TRACK·SUSPECT·LOST 상태별 행동으로 연결하는 구조다.",
]);
parts.push({ lines: evalProposal, headingOffset: 1 });
parts.push({ lines: appendix.split("\n"), headingOffset: 0 });

function splitTableRow(line) {
  const sentinel = "\uE000";
  const safe = line.replace(/\\\|/g, sentinel).trim().replace(/^\|/, "").replace(/\|$/, "");
  return safe.split("|").map((cell) => cell.replaceAll(sentinel, "|").trim());
}

function isSeparatorRow(cells) {
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, "")));
}

function cleanMath(value) {
  return value
    .replace(/<br\s*\/?\s*>/gi, " / ")
    .replace(/<[^>]+>/g, "")
    .replace(/\\\(|\\\)|\\\[|\\\]/g, "")
    .replace(/\$+/g, "")
    .replace(/\\mathcal\s*\{([A-Za-z])\}/g, "$1")
    .replace(/\\mathcal\s+([A-Za-z])/g, "$1")
    .replace(/\\text\{([^{}]*)\}/g, "$1")
    .replace(/\\mathrm\{([^{}]*)\}/g, "$1")
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "($1)/($2)")
    .replace(/\\cap/g, "∩")
    .replace(/\\cup/g, "∪")
    .replace(/\\times/g, "×")
    .replace(/\\cdot/g, "·")
    .replace(/\\left|\\right/g, "")
    .replace(/\\leq?(?![A-Za-z])|≤/g, "≤")
    .replace(/\\geq?(?![A-Za-z])|≥/g, "≥")
    .replace(/\\rightarrow|\\to/g, "→")
    .replace(/\\approx/g, "≈")
    .replace(/\\sum/g, "Σ")
    .replace(/\\sigma/g, "σ")
    .replace(/\\tau/g, "τ")
    .replace(/\\lambda/g, "λ")
    .replace(/\\alpha/g, "α")
    .replace(/\\delta/g, "δ")
    .replace(/\\epsilon/g, "ε")
    .replace(/\\theta/g, "θ")
    .replace(/\\hat\s*/g, "")
    .replace(/\\mathbf\s*\{?1\}?/g, "1")
    .replace(/\\ldots/g, "…")
    .replace(/\\mid/g, "|")
    .replace(/\\log/g, "log")
    .replace(/\\infty/g, "∞")
    .replace(/\\in/g, "∈")
    .replace(/\\begin\{[^}]+\}|\\end\{[^}]+\}/g, "")
    .replace(/\\quad|\\qquad/g, " ")
    .replace(/\\!/g, "")
    .replace(/\\[,;]/g, " ")
    .replace(/\\(?=\s)/g, "")
    .replace(/\\&/g, "&")
    .replace(/\\_/g, "_")
    .replace(/\\\*/g, "*")
    .replace(/\\\\/g, "; ")
    .replace(/\\frac/g, "/")
    .replace(/[{}]/g, "")
    .replace(/\s{2,}/g, " ")
    .trimEnd();
}

function parseInline(raw) {
  const source = cleanMath(raw);
  let text = "";
  const spans = [];
  const opens = { bold: null, italic: null, code: null };
  for (let i = 0; i < source.length;) {
    if (source.startsWith("**", i)) {
      if (opens.bold === null) opens.bold = text.length;
      else { if (text.length > opens.bold) spans.push({ type: "bold", start: opens.bold, end: text.length }); opens.bold = null; }
      i += 2;
      continue;
    }
    if (source[i] === "`") {
      if (opens.code === null) opens.code = text.length;
      else { if (text.length > opens.code) spans.push({ type: "code", start: opens.code, end: text.length }); opens.code = null; }
      i += 1;
      continue;
    }
    if (source[i] === "*" && source[i + 1] !== "*") {
      if (opens.italic === null) opens.italic = text.length;
      else { if (text.length > opens.italic) spans.push({ type: "italic", start: opens.italic, end: text.length }); opens.italic = null; }
      i += 1;
      continue;
    }
    if (source[i] === "[") {
      const mid = source.indexOf("](", i + 1);
      const close = mid >= 0 ? source.indexOf(")", mid + 2) : -1;
      if (mid >= 0 && close >= 0) {
        const label = source.slice(i + 1, mid).replace(/\*\*|`/g, "");
        const url = source.slice(mid + 2, close);
        const start = text.length;
        text += label;
        if (label && /^https?:\/\//.test(url)) spans.push({ type: "link", start, end: text.length, url });
        i = close + 1;
        continue;
      }
    }
    text += source[i];
    i += 1;
  }
  for (const [type, start] of Object.entries(opens)) {
    if (start !== null && text.length > start) spans.push({ type, start, end: text.length });
  }
  return { text: text.replace(/~~/g, ""), spans };
}

let content = "";
let cursor = 1;
const blocks = [];
const inlineSpans = [];

function addParagraph(raw, role, headingLevel = null) {
  const parsed = parseInline(raw);
  const text = parsed.text.trimEnd();
  if (!text) return;
  const start = cursor;
  content += `${text}\n`;
  cursor += text.length + 1;
  const block = { start, endText: start + text.length, end: start + text.length + 1, role, headingLevel, text };
  blocks.push(block);
  for (const span of parsed.spans) inlineSpans.push({ ...span, start: start + span.start, end: start + span.end, blockRole: role, headingLevel });
}

function parsePart(part, partIndex) {
  const lines = part.lines;
  let inCode = false;
  for (let i = 0; i < lines.length;) {
    const line = lines[i].replace(/\t/g, "    ");
    if (/^```/.test(line.trim())) { inCode = !inCode; i += 1; continue; }
    if (inCode) { if (line.trim()) addParagraph(line, "CODE"); i += 1; continue; }
    if (!line.trim() || /^---+$/.test(line.trim())) { i += 1; continue; }
    if (/^\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\|.*\|\s*$/.test(lines[i + 1])) {
      const headers = splitTableRow(line);
      const separators = splitTableRow(lines[i + 1]);
      if (isSeparatorRow(separators)) {
        i += 2;
        while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) {
          const cells = splitTableRow(lines[i]);
          if (!isSeparatorRow(cells) && cells.some(Boolean)) {
            const first = cells[0] || "항목";
            const details = cells.slice(1).map((cell, idx) => cell ? `${headers[idx + 1] || `항목 ${idx + 2}`}: ${cell}` : "").filter(Boolean);
            addParagraph(`**${first}**${details.length ? ` — ${details.join(" · ")}` : ""}`, "BULLET");
          }
          i += 1;
        }
        continue;
      }
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) {
      if (part.titleLine && partIndex === 0 && i === 0) addParagraph(heading[2], "TITLE");
      else {
        const level = Math.max(1, Math.min(6, heading[1].length + (part.headingOffset || 0)));
        addParagraph(heading[2], "HEADING", level);
      }
      i += 1;
      continue;
    }
    const bullet = /^\s*[-+*]\s+(.+)$/.exec(line);
    if (bullet) { addParagraph(bullet[1], "BULLET"); i += 1; continue; }
    const numbered = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (numbered) { addParagraph(numbered[1], "NUMBERED"); i += 1; continue; }
    const quote = /^>\s?(.*)$/.exec(line);
    if (quote) { addParagraph(quote[1], "CALLOUT"); i += 1; continue; }
    if (/^\[FIGURE\d+\]$/.test(line.trim())) { addParagraph(line.trim(), "IMAGE_PLACEHOLDER"); i += 1; continue; }
    if (/^그림\s+\d+\./.test(line.trim())) { addParagraph(line.trim(), "CAPTION"); i += 1; continue; }
    addParagraph(line.trim(), "BODY");
    i += 1;
  }
}

parts.forEach(parsePart);

const manifest = {
  version: 1,
  contentLength: content.length,
  endIndex: cursor,
  blocks,
  inlineSpans,
  images: blocks.filter((b) => b.role === "IMAGE_PLACEHOLDER").map((b) => ({ marker: b.text, start: b.start, endText: b.endText, end: b.end })),
  stats: {
    paragraphs: blocks.length,
    headings: blocks.filter((b) => b.role === "HEADING").length,
    bullets: blocks.filter((b) => b.role === "BULLET").length,
    numbered: blocks.filter((b) => b.role === "NUMBERED").length,
    codeParagraphs: blocks.filter((b) => b.role === "CODE").length,
    inlineSpans: inlineSpans.length,
    images: blocks.filter((b) => b.role === "IMAGE_PLACEHOLDER").length,
  },
};

fs.writeFileSync(path.join(outDir, "content.txt"), content, "utf8");
fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest), "utf8");

const tabId = "t.o5ehjizdlxvq";
const ink = { color: { rgbColor: { red: 0.1254902, green: 0.12941177, blue: 0.14117648 } } };
const muted = { color: { rgbColor: { red: 0.29411766, green: 0.33333334, blue: 0.3882353 } } };
const codeInk = { color: { rgbColor: { red: 0.20392157, green: 0.2509804, blue: 0.32941177 } } };
const linkInk = ink;
const pt = (magnitude) => ({ magnitude, unit: "PT" });
const range = (startIndex, endIndex) => ({ startIndex, endIndex, tabId });
const requests = [];

requests.push({
  updateTextStyle: {
    range: range(1, manifest.endIndex),
    textStyle: {
      weightedFontFamily: { fontFamily: "Arial", weight: 400 },
      fontSize: pt(11),
      foregroundColor: ink,
      bold: false,
      italic: false,
      underline: false,
    },
    fields: "weightedFontFamily,fontSize,foregroundColor,bold,italic,underline",
  },
});
requests.push({
  updateParagraphStyle: {
    range: range(1, manifest.endIndex),
    paragraphStyle: { namedStyleType: "NORMAL_TEXT", lineSpacing: 115, spaceBelow: pt(6) },
    fields: "namedStyleType,lineSpacing,spaceBelow",
  },
});

const headingSizes = { 1: 20, 2: 16, 3: 14, 4: 12, 5: 11, 6: 10.5 };
for (const block of blocks) {
  if (block.role === "TITLE") {
    requests.push({ updateParagraphStyle: { range: range(block.start, block.end), paragraphStyle: { namedStyleType: "NORMAL_TEXT", spaceBelow: pt(14), keepWithNext: true }, fields: "namedStyleType,spaceBelow,keepWithNext" } });
    requests.push({ updateTextStyle: { range: range(block.start, block.endText), textStyle: { weightedFontFamily: { fontFamily: "Arial", weight: 400 }, fontSize: pt(26), foregroundColor: ink, bold: false }, fields: "weightedFontFamily,fontSize,foregroundColor,bold" } });
  } else if (block.role === "HEADING") {
    const level = block.headingLevel;
    requests.push({ updateParagraphStyle: { range: range(block.start, block.end), paragraphStyle: { namedStyleType: `HEADING_${level}`, spaceAbove: pt(level === 1 ? 18 : 11), spaceBelow: pt(level <= 2 ? 8 : 5), keepWithNext: true }, fields: "namedStyleType,spaceAbove,spaceBelow,keepWithNext" } });
    requests.push({ updateTextStyle: { range: range(block.start, block.endText), textStyle: { weightedFontFamily: { fontFamily: "Arial", weight: 400 }, fontSize: pt(headingSizes[level]), foregroundColor: level <= 2 ? ink : muted, bold: true }, fields: "weightedFontFamily,fontSize,foregroundColor,bold" } });
  } else if (block.role === "CAPTION") {
    requests.push({ updateParagraphStyle: { range: range(block.start, block.end), paragraphStyle: { alignment: "CENTER", spaceAbove: pt(4), spaceBelow: pt(12), keepWithNext: false }, fields: "alignment,spaceAbove,spaceBelow,keepWithNext" } });
    requests.push({ updateTextStyle: { range: range(block.start, block.endText), textStyle: { weightedFontFamily: { fontFamily: "Arial", weight: 400 }, fontSize: pt(9.5), foregroundColor: muted, italic: true }, fields: "weightedFontFamily,fontSize,foregroundColor,italic" } });
  } else if (block.role === "IMAGE_PLACEHOLDER") {
    requests.push({ updateParagraphStyle: { range: range(block.start, block.end), paragraphStyle: { alignment: "CENTER", spaceAbove: pt(8), spaceBelow: pt(6) }, fields: "alignment,spaceAbove,spaceBelow" } });
  }
}

const introCallout = blocks.find((block) => block.text.startsWith("이 탭의 목적:"));
if (introCallout) {
  requests.push({ updateParagraphStyle: { range: range(introCallout.start, introCallout.end), paragraphStyle: { shading: { backgroundColor: { color: { rgbColor: { red: 0.95686275, green: 0.96862745, blue: 0.98039216 } } } }, indentStart: pt(12), indentEnd: pt(12), spaceAbove: pt(4), spaceBelow: pt(10) }, fields: "shading,indentStart,indentEnd,spaceAbove,spaceBelow" } });
  requests.push({ updateTextStyle: { range: range(introCallout.start, introCallout.endText), textStyle: { fontSize: pt(10.5), foregroundColor: codeInk }, fields: "fontSize,foregroundColor" } });
}

function contiguousGroups(role) {
  const selected = blocks.filter((b) => b.role === role);
  const groups = [];
  for (const block of selected) {
    const last = groups.at(-1);
    if (last && last.end === block.start) { last.end = block.end; last.endText = block.endText; }
    else groups.push({ start: block.start, end: block.end, endText: block.endText });
  }
  return groups;
}

for (const group of contiguousGroups("CODE")) {
  requests.push({ updateParagraphStyle: { range: range(group.start, group.end), paragraphStyle: { shading: { backgroundColor: { color: { rgbColor: { red: 0.96862745, green: 0.97647059, blue: 0.98823529 } } } }, indentStart: pt(10), indentEnd: pt(10), lineSpacing: 100, spaceBelow: pt(1) }, fields: "shading,indentStart,indentEnd,lineSpacing,spaceBelow" } });
  requests.push({ updateTextStyle: { range: range(group.start, group.endText), textStyle: { weightedFontFamily: { fontFamily: "Courier New", weight: 400 }, fontSize: pt(9.5), foregroundColor: codeInk }, fields: "weightedFontFamily,fontSize,foregroundColor" } });
}

for (const group of contiguousGroups("CALLOUT")) {
  requests.push({ updateParagraphStyle: { range: range(group.start, group.end), paragraphStyle: { shading: { backgroundColor: { color: { rgbColor: { red: 0.94509804, green: 0.96078431, blue: 1 } } } }, indentStart: pt(12), indentEnd: pt(12), spaceAbove: pt(4), spaceBelow: pt(8) }, fields: "shading,indentStart,indentEnd,spaceAbove,spaceBelow" } });
}

for (const group of contiguousGroups("BULLET")) requests.push({ createParagraphBullets: { range: range(group.start, group.end), bulletPreset: "BULLET_DISC_CIRCLE_SQUARE" } });
for (const group of contiguousGroups("NUMBERED")) requests.push({ createParagraphBullets: { range: range(group.start, group.end), bulletPreset: "NUMBERED_DECIMAL_NESTED" } });

const nonLinks = inlineSpans.filter((span) => span.type !== "link");
for (const span of nonLinks) {
  if (span.type === "bold") requests.push({ updateTextStyle: { range: range(span.start, span.end), textStyle: { bold: true }, fields: "bold" } });
  if (span.type === "italic") requests.push({ updateTextStyle: { range: range(span.start, span.end), textStyle: { italic: true }, fields: "italic" } });
  if (span.type === "code") requests.push({ updateTextStyle: { range: range(span.start, span.end), textStyle: { weightedFontFamily: { fontFamily: "Courier New", weight: 400 }, fontSize: pt(10), foregroundColor: codeInk }, fields: "weightedFontFamily,fontSize,foregroundColor" } });
}

function baseForSpan(span) {
  const block = blocks.find((b) => span.start >= b.start && span.end <= b.endText);
  const level = block?.headingLevel;
  const boldOverlap = inlineSpans.some((other) => other.type === "bold" && other.start <= span.start && other.end >= span.end);
  const italicOverlap = inlineSpans.some((other) => other.type === "italic" && other.start <= span.start && other.end >= span.end);
  if (block?.role === "TITLE") return { size: 26, bold: false, italic: false };
  if (block?.role === "HEADING") return { size: headingSizes[level], bold: true, italic: false };
  if (block?.role === "CAPTION") return { size: 9.5, bold: boldOverlap, italic: true };
  return { size: 11, bold: boldOverlap, italic: italicOverlap };
}

for (const span of inlineSpans.filter((item) => item.type === "link")) {
  const base = baseForSpan(span);
  requests.push({ updateTextStyle: { range: range(span.start, span.end), textStyle: { link: { url: span.url }, weightedFontFamily: { fontFamily: "Arial", weight: 400 }, fontSize: pt(base.size), foregroundColor: linkInk, underline: false, bold: base.bold, italic: base.italic }, fields: "link,weightedFontFamily,fontSize,foregroundColor,underline,bold,italic" } });
}

const contentChunkDir = path.join(outDir, "content_chunks");
const formatChunkDir = path.join(outDir, "format_chunks");
fs.mkdirSync(contentChunkDir, { recursive: true });
fs.mkdirSync(formatChunkDir, { recursive: true });
for (const file of fs.readdirSync(contentChunkDir)) fs.rmSync(path.join(contentChunkDir, file));
for (const file of fs.readdirSync(formatChunkDir)) fs.rmSync(path.join(formatChunkDir, file));

const contentChunks = [];
let pending = "";
for (const line of content.split(/(?<=\n)/)) {
  if (pending.length + line.length > 18000 && pending) { contentChunks.push(pending); pending = ""; }
  pending += line;
}
if (pending) contentChunks.push(pending);
contentChunks.forEach((chunk, index) => fs.writeFileSync(path.join(contentChunkDir, `${String(index + 1).padStart(2, "0")}.txt`), chunk, "utf8"));

const formatChunks = [];
for (let i = 0; i < requests.length; i += 160) formatChunks.push(requests.slice(i, i + 160));
formatChunks.forEach((chunk, index) => fs.writeFileSync(path.join(formatChunkDir, `${String(index + 1).padStart(2, "0")}.json`), JSON.stringify(chunk), "utf8"));

const buildIndex = {
  contentFiles: contentChunks.map((_, index) => `${String(index + 1).padStart(2, "0")}.txt`),
  formatFiles: formatChunks.map((_, index) => `${String(index + 1).padStart(2, "0")}.json`),
  contentLength: content.length,
  endIndex: manifest.endIndex,
  formattingRequests: requests.length,
};
fs.writeFileSync(path.join(outDir, "build-index.json"), JSON.stringify(buildIndex), "utf8");
console.log(JSON.stringify(manifest.stats));
console.log(JSON.stringify(buildIndex));
