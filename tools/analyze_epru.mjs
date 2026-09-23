#!/usr/bin/env node
// M0 diagnostic: inspect an .epru document stream and report the facts needed
// to calibrate the format (record types, layer table, raw numeric magnitudes).
//
// Usage: node tools/analyze_epru.mjs <path-to.epru>

import fs from "node:fs";

const args = process.argv.slice(2);
const rawIdx = args.indexOf("--raw");
const rawType = rawIdx === -1 ? null : args[rawIdx + 1];
const rawLimit = Number(args[args.indexOf("--limit") + 1]) || 3;
const path = args.find((a) => !a.startsWith("--") && a !== rawType);

if (!path) {
  console.error("usage: node tools/analyze_epru.mjs <file.epru> [--raw TYPE] [--limit N]");
  process.exit(1);
}

const raw = fs.readFileSync(path, "utf8");
const lines = raw.split(/\r?\n/).filter((l) => l.trim() !== "");

/** Split "{header}||{body}|" into its two JSON objects. */
function splitRecord(line) {
  const sep = line.indexOf("}||");
  if (sep === -1) return null;
  const headerText = line.slice(0, sep + 1);
  let bodyText = line.slice(sep + 3);
  if (bodyText.endsWith("|")) bodyText = bodyText.slice(0, -1);
  try {
    return { header: JSON.parse(headerText), body: JSON.parse(bodyText) };
  } catch {
    return null;
  }
}

const docHeads = [];
const typeCounts = new Map();   // "docType/TYPE" -> count
const perDocSamples = new Map(); // "docType/TYPE" -> first body
let currentDoc = "(none)";
let parseFailures = 0;
const failureSamples = [];
const byType = new Map();       // TYPE -> [body, ...]

for (const [i, line] of lines.entries()) {
  const rec = splitRecord(line);
  if (!rec) {
    parseFailures++;
    if (failureSamples.length < 3) failureSamples.push([i + 1, line.slice(0, 300)]);
    continue;
  }
  const { header, body } = rec;

  if (header.type === "DOCHEAD") {
    currentDoc = body.docType ?? "?";
    docHeads.push(body);
    continue;
  }

  const key = `${currentDoc}/${header.type}`;
  typeCounts.set(key, (typeCounts.get(key) ?? 0) + 1);
  if (!perDocSamples.has(key)) perDocSamples.set(key, body);
  if (!byType.has(header.type)) byType.set(header.type, []);
  byType.get(header.type).push(body);
}

console.log(`file        : ${path}`);
console.log(`bytes       : ${raw.length}`);
console.log(`lines       : ${lines.length}`);
console.log(`parse fails : ${parseFailures}`);
console.log();

if (rawType) {
  const want = rawType.toUpperCase();
  let shown = 0;
  console.log(`=== raw lines for type ${want} ===`);
  for (const line of lines) {
    const rec = splitRecord(line);
    if (!rec || rec.header.type.toUpperCase() !== want) continue;
    if (shown++ >= rawLimit) break;
    console.log(`  [${shown}] len=${line.length}`);
    console.log("      " + line.slice(0, 900));
  }
  if (shown === 0) console.log("  (none)");
  process.exit(0);
}

if (failureSamples.length) {
  console.log("=== unparsed lines ===");
  for (const [n, text] of failureSamples) console.log(`  line ${n}: ${text}`);
  console.log();
}

console.log("=== DOCHEAD sections ===");
for (const h of docHeads) {
  console.log(`  ${h.docType.padEnd(12)} uuid=${h.uuid}  client=${h.client ?? "-"}`);
}
console.log();

console.log("=== record types (grouped by document) ===");
for (const [k, v] of [...typeCounts].sort()) {
  console.log(`  ${k.padEnd(28)} ${v}`);
}
console.log();

function fmt(v) {
  return typeof v === "number" ? v.toFixed(4) : JSON.stringify(v);
}

function stat(values) {
  const nums = values.filter((v) => typeof v === "number" && Number.isFinite(v));
  if (!nums.length) return "n/a";
  const uniq = [...new Set(nums)].sort((a, b) => a - b);
  const min = uniq[0];
  const max = uniq[uniq.length - 1];
  return `n=${nums.length} min=${min} max=${max} uniq=[${uniq.slice(0, 12).join(", ")}${uniq.length > 12 ? ", ..." : ""}]`;
}

// Numeric fields that reveal the underlying unit system.
const interesting = {
  LINE: ["width"],
  ARC: ["width"],
  POLY: ["width"],
  FILL: ["width"],
  POUR: ["width"],
  VIA: ["holeDiameter", "viaDiameter"],
  PAD: ["width", "height", "holeDiameter"],
  CANVAS: ["gridXSize", "gridYSize", "snapXSize", "snapYSize"],
  PREFERENCE: ["lastTrackWidth"],
};

console.log("=== numeric field magnitudes (unit calibration) ===");
for (const [type, fields] of Object.entries(interesting)) {
  const recs = byType.get(type);
  if (!recs?.length) continue;
  for (const f of fields) {
    const vals = recs.map((r) => r?.[f]).filter((v) => v !== undefined && v !== null);
    if (vals.length) console.log(`  ${type}.${f}`.padEnd(24) + stat(vals));
  }
}
console.log();

console.log("=== sample records per type ===");
for (const [k, body] of perDocSamples) {
  console.log(`  --- ${k} ---`);
  console.log("      " + JSON.stringify(body).slice(0, 400));
}
console.log();

console.log("=== coordinate magnitudes ===");
for (const f of ["startX", "startY", "endX", "endY", "x", "y", "centerX", "centerY"]) {
  const vals = [];
  for (const recs of byType.values()) for (const r of recs) if (typeof r?.[f] === "number") vals.push(r[f]);
  if (vals.length) console.log(`  ${f}`.padEnd(12) + stat(vals));
}
console.log();

// Board outline layer: report any record sitting on layerId 11.
const outlineRecs = [];
for (const [type, recs] of byType) {
  for (const r of recs) if (r?.layerId === 11) outlineRecs.push([type, r]);
}
console.log(`=== records on layerId 11 (OUTLINE): ${outlineRecs.length} ===`);
for (const [type, r] of outlineRecs.slice(0, 5)) {
  console.log(`  ${type.padEnd(8)} ${JSON.stringify(r).slice(0, 300)}`);
}
