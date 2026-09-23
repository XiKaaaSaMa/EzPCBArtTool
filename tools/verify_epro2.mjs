#!/usr/bin/env node
// Independent structural check of a generated .epro2, written in Node so it
// shares no code with the Python writer it is validating.
//
// Usage: node tools/verify_epro2.mjs <file.epro2>

import fs from "node:fs";
import zlib from "node:zlib";

const path = process.argv[2];
if (!path) {
  console.error("usage: node tools/verify_epro2.mjs <file.epro2>");
  process.exit(1);
}

// --- minimal ZIP reader (stored + deflate), independent of any library ------
function readZipEntries(buffer) {
  const eocdSignature = 0x06054b50;
  let eocd = -1;
  for (let i = buffer.length - 22; i >= 0 && i > buffer.length - 66000; i--) {
    if (buffer.readUInt32LE(i) === eocdSignature) {
      eocd = i;
      break;
    }
  }
  if (eocd === -1) throw new Error("not a ZIP: end-of-central-directory not found");

  const entryCount = buffer.readUInt16LE(eocd + 10);
  let offset = buffer.readUInt32LE(eocd + 16);
  const entries = [];

  for (let n = 0; n < entryCount; n++) {
    if (buffer.readUInt32LE(offset) !== 0x02014b50) throw new Error("bad central directory");
    const method = buffer.readUInt16LE(offset + 10);
    const compressedSize = buffer.readUInt32LE(offset + 20);
    const uncompressedSize = buffer.readUInt32LE(offset + 24);
    const nameLen = buffer.readUInt16LE(offset + 28);
    const extraLen = buffer.readUInt16LE(offset + 30);
    const commentLen = buffer.readUInt16LE(offset + 32);
    const localOffset = buffer.readUInt32LE(offset + 42);
    const name = buffer.toString("utf8", offset + 46, offset + 46 + nameLen);

    // local header -> data
    const lhNameLen = buffer.readUInt16LE(localOffset + 26);
    const lhExtraLen = buffer.readUInt16LE(localOffset + 28);
    const dataStart = localOffset + 30 + lhNameLen + lhExtraLen;
    const raw = buffer.subarray(dataStart, dataStart + compressedSize);
    const data = method === 0 ? raw : zlib.inflateRawSync(raw);

    entries.push({ name, method, size: uncompressedSize, data });
    offset += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

const buffer = fs.readFileSync(path);
const entries = readZipEntries(buffer);

console.log(`archive : ${path}`);
console.log(`entries : ${entries.length}`);
for (const entry of entries) {
  console.log(`  ${entry.name.padEnd(28)} method=${entry.method} bytes=${entry.size}`);
}

const required = ["project2.json"];
const epruEntry = entries.find((e) => e.name.toLowerCase().endsWith(".epru"));
if (!epruEntry) throw new Error("no .epru entry in archive");
required.push(epruEntry.name);

for (const name of required) {
  if (!entries.some((e) => e.name === name)) throw new Error(`missing required entry: ${name}`);
}
console.log("\nrequired entries present: ok");

const meta = JSON.parse(entries.find((e) => e.name === "project2.json").data.toString("utf8"));
console.log(`project title: ${meta.title}`);
for (const key of ["cbb_project", "editorVersion", "introduction", "description", "tags"]) {
  if (!(key in meta)) throw new Error(`project2.json missing field: ${key}`);
}
console.log("project2.json fields: ok");

// --- parse the log ----------------------------------------------------------
const text = epruEntry.data.toString("utf8");
const lines = text.split(/\r?\n/).filter((l) => l.trim());
let bad = 0;
let docs = [];
let inPcb = false;
let outlineBBox = null;
let malformedTerminator = 0;
const outlineRings = [];
const pads = [];
const fills = [];
const lines_ = [];
const layerPhys = [];

for (const line of lines) {
  if (!line.endsWith("|")) malformedTerminator++;
  const sep = line.indexOf("}||");
  if (sep === -1) {
    bad++;
    continue;
  }
  let header;
  let body = null;
  try {
    header = JSON.parse(line.slice(0, sep + 1));
    const bodyText = line.slice(sep + 3).replace(/\|$/, "");
    if (bodyText.length) body = JSON.parse(bodyText);
  } catch {
    bad++;
    continue;
  }

  if (header.type === "DOCHEAD") {
    docs.push(body?.docType);
    inPcb = body?.docType === "PCB";
    continue;
  }

  if (inPcb && header.type === "PAD" && body) {
    pads.push(body);
  }

  // FILL and LINE are what the copper layers are made of. They were outside
  // this validator's coverage, which meant a board full of copper could pass
  // verification untouched -- the same gap that once let a malformed PAD be
  // silently dropped by the editor.
  if (inPcb && header.type === "FILL" && body) {
    fills.push(body);
  }
  if (inPcb && header.type === "LINE" && body) {
    lines_.push(body);
  }

  // LAYER_PHYS carries isKeepIsland, which decides whether the editor keeps
  // copper that belongs to no net. Everything this tool writes is net-less, so
  // a false here silently deletes the whole copper layer on load -- the file
  // still verifies perfectly and the board opens as nothing but an outline.
  if (inPcb && header.type === "LAYER_PHYS" && body) {
    layerPhys.push({ id: header.id, body });
  }

  if (inPcb && body?.layerId === 11 && (body.path || body.startX !== undefined)) {
    // Decode the path into (x, y) vertices in the design frame (Y negated).
    const points = [];
    const path = body.path ?? [];
    const tokens = Array.isArray(path[0]) ? path[0] : path;

    if (tokens[0] === "CIRCLE") {
      // Native circle form: ['CIRCLE', cx, cy, r]. Sample it so the ring can
      // still be measured and nesting-tested like any other.
      const [, cx, cy, r] = tokens;
      const ring = [];
      for (let i = 0; i < 64; i++) {
        const a = (2 * Math.PI * i) / 64;
        // File Y is already up-positive, so no sign flip here.
        ring.push([(cx + r * Math.cos(a)) * 0.0254, (cy + r * Math.sin(a)) * 0.0254]);
      }
      outlineRings.push(ring);
      points.push(...ring);
    } else {
      let pending = [];
      let i = 0;
      const flush = () => {
        for (let k = 0; k + 1 < pending.length; k += 2) {
          // File Y is already up-positive; no sign flip.
          points.push([pending[k] * 0.0254, pending[k + 1] * 0.0254]);
        }
        pending = [];
      };
      while (i < tokens.length) {
        const token = tokens[i];
        if (typeof token === "string") {
          flush();
          i += 1;
          if (token === "ARC" || token === "CARC") i += 3;
          continue;
        }
        pending.push(token);
        i += 1;
      }
      flush();
      outlineRings.push(points);
    }

    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    if (xs.length && ys.length) {
      const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
      outlineBBox = outlineBBox
        ? [
            Math.min(outlineBBox[0], box[0]),
            Math.min(outlineBBox[1], box[1]),
            Math.max(outlineBBox[2], box[2]),
            Math.max(outlineBBox[3], box[3]),
          ]
        : box;
    }
  }
}

if (bad) throw new Error(`${bad} unparseable record lines`);
if (malformedTerminator) throw new Error(`${malformedTerminator} lines missing the '|' terminator`);

console.log(`\nlog records  : ${lines.length}`);
console.log(`documents    : ${docs.join(", ")}`);
console.log("all lines parse and terminate correctly: ok");

if (!outlineBBox) {
  throw new Error("no board outline primitive found on layer 11");
}

const validRings = outlineRings.filter((r) => r && r.length >= 3);
const [x0, y0, x1, y1] = outlineBBox;
console.log(
  `\nboard outline: ${(x1 - x0).toFixed(4)} x ${(y1 - y0).toFixed(4)} mm` +
    `  (x ${x0.toFixed(4)}..${x1.toFixed(4)}, y ${y0.toFixed(4)}..${y1.toFixed(4)})`,
);
console.log(`outline rings: ${validRings.length}`);

if (validRings.length > 1) {
  // With cutouts, the outer ring encloses the rest. Report each area so a
  // missing or wrongly-oriented hole is visible in the output.
  // Signed area: positive is counter-clockwise in the design frame. The outer
  // ring is clockwise (negative) and holes counter-clockwise (positive), so
  // the outer ring is the one with the largest *absolute* area.
  const areas = validRings.map((ring) => {
    let sum = 0;
    for (let i = 0; i < ring.length; i++) {
      const [ax, ay] = ring[i];
      const [bx, by] = ring[(i + 1) % ring.length];
      sum += ax * by - bx * ay;
    }
    return sum / 2;
  });
  const outerIndex = areas
    .map((a, i) => [Math.abs(a), i])
    .sort((p, q) => q[0] - p[0])[0][1];
  console.log(
    `  areas (mm^2): ${areas
      .map(
        (a, i) =>
          `${i === outerIndex ? "outer" : "hole"}=${Math.abs(a).toFixed(2)}${a < 0 ? "(CW)" : "(CCW)"}`,
      )
      .join(", ")}`,
  );
  const outerArea = areas[outerIndex];
  if (outerArea >= 0) {
    console.log("  note: outer ring is counter-clockwise; EasyEDA Pro expects clockwise");
  }
  for (let i = 0; i < areas.length; i++) {
    if (i === outerIndex) continue;
    if (areas[i] <= 0) {
      console.log(`  note: hole ring ${i} is clockwise; EasyEDA Pro expects counter-clockwise`);
    }
  }
  const outerBox = boundingBox(validRings[outerIndex]);
  for (let i = 0; i < validRings.length; i++) {
    if (i === outerIndex) continue;
    const innerBox = boundingBox(validRings[i]);
    const contained =
      innerBox[0] >= outerBox[0] &&
      innerBox[1] >= outerBox[1] &&
      innerBox[2] <= outerBox[2] &&
      innerBox[3] <= outerBox[3];
    if (!contained) throw new Error(`hole ring ${i} is not inside the outer ring`);
  }
  console.log("  all holes lie inside the outer ring: ok");
}

function boundingBox(ring) {
  const xs = ring.map((p) => p[0]);
  const ys = ring.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

// Every ring must be closed and have non-zero extent.
for (const [i, ring] of validRings.entries()) {
  const first = ring[0];
  const last = ring[ring.length - 1];
  const closed = Math.abs(first[0] - last[0]) < 1e-6 && Math.abs(first[1] - last[1]) < 1e-6;
  if (!closed) {
    console.log(`  note: ring ${i} is not explicitly closed (closing edge is implicit)`);
  }
}

// --- pad-encoded round holes -----------------------------------------------
// A round cutout is expressed as a drilled pad whose pad diameter equals its
// drill diameter, so no copper ring is left behind. EasyEDA Pro only reads the
// object form of a shape; the positional array the specification documents is
// accepted silently and then dropped, which leaves the board with no hole.
console.log(`\npads         : ${pads.length}`);

const toMm = (units) => units * 0.0254;

for (const [i, pad] of pads.entries()) {
  const where = `pad ${i} (layer ${pad.layerId}, num ${JSON.stringify(pad.num)})`;

  if (Array.isArray(pad.defaultPad)) {
    throw new Error(
      `${where}: defaultPad is written as an array; EasyEDA Pro only reads the ` +
        `object form {"padType":..., "width":..., "height":...}`,
    );
  }
  if (!pad.defaultPad || typeof pad.defaultPad !== "object") {
    throw new Error(`${where}: defaultPad is missing or malformed`);
  }
  if (typeof pad.defaultPad.padType !== "string") {
    throw new Error(`${where}: defaultPad.padType must be a string`);
  }

  const hasDrill = pad.hole !== null && pad.hole !== undefined;
  if (hasDrill) {
    if (Array.isArray(pad.hole)) {
      throw new Error(
        `${where}: hole is written as an array; EasyEDA Pro only reads the ` +
          `object form {"holeType":..., "width":..., "height":...}`,
      );
    }
    if (typeof pad.hole !== "object") {
      throw new Error(`${where}: hole is malformed`);
    }
    if (typeof pad.hole.holeType !== "string") {
      throw new Error(`${where}: hole.holeType must be a string`);
    }
  }

  for (const [field, shape] of [
    ["hole", pad.hole],
    ["defaultPad", pad.defaultPad],
  ]) {
    if (!shape) continue;
    for (const key of ["width", "height"]) {
      if (typeof shape[key] !== "number" || !(shape[key] > 0)) {
        throw new Error(`${where}: ${field}.${key} must be a positive number`);
      }
    }
  }

  if (typeof pad.centerX !== "number" || typeof pad.centerY !== "number") {
    throw new Error(`${where}: centre coordinates must be numbers`);
  }

  const cxMm = toMm(pad.centerX);
  const cyMm = toMm(pad.centerY);

  if (hasDrill) {
    const drillMm = `${toMm(pad.hole.width).toFixed(3)}x${toMm(pad.hole.height).toFixed(3)}`;
    const padMm = `${toMm(pad.defaultPad.width).toFixed(3)}x${toMm(pad.defaultPad.height).toFixed(3)}`;
    if (
      pad.hole.width !== pad.defaultPad.width ||
      pad.hole.height !== pad.defaultPad.height
    ) {
      throw new Error(
        `${where}: ${padMm} mm pad on a ${drillMm} mm drill leaves an annular ring; ` +
          `a plain cutout needs them equal`,
      );
    }

    const r = Math.max(toMm(pad.hole.width), toMm(pad.hole.height)) / 2;
    if (cxMm - r < x0 || cxMm + r > x1 || cyMm - r < y0 || cyMm + r > y1) {
      throw new Error(
        `${where}: hole at (${cxMm.toFixed(3)}, ${cyMm.toFixed(3)}) mm reaches outside ` +
          `the board outline`,
      );
    }

    console.log(
      `  ${where}: ${drillMm} mm ${pad.hole.holeType} drill, ${padMm} mm ` +
        `${pad.defaultPad.padType} pad, plated=${pad.plated === true}, ` +
        `centre=(${cxMm.toFixed(3)}, ${cyMm.toFixed(3)}) mm`,
    );
  }
}

if (pads.length) {
  console.log("  every pad uses the object form, has no annular ring, and sits on the board: ok");
}

// --- copper fills and traces ----------------------------------------------
// Decodes every FILL, measuring each ring and its winding. This is the part
// the real-project capture could not settle: all 75 fills in it were
// single-ring, and no POLY anywhere carried an interior ring either, so
// nothing captured says how a hole should be wound.
if (fills.length) {
  const MM = 0.0254;
  const byLayer = new Map();
  let withHoles = 0;
  const windings = { CW: 0, CCW: 0 };

  const decodeRing = (tokens) => {
    const pts = [];
    let pending = [];
    let i = 0;
    while (i < tokens.length) {
      const token = tokens[i];
      if (typeof token === "string") {
        for (let k = 0; k + 1 < pending.length; k += 2) {
          pts.push([pending[k] * MM, pending[k + 1] * MM]);
        }
        pending = [];
        i += 1;
        if (token === "ARC" || token === "CARC") i += 3;
        continue;
      }
      pending.push(token);
      i += 1;
    }
    for (let k = 0; k + 1 < pending.length; k += 2) {
      pts.push([pending[k] * MM, pending[k + 1] * MM]);
    }
    return pts;
  };

  const signedArea = (ring) => {
    let sum = 0;
    for (let i = 0; i < ring.length; i++) {
      const [ax, ay] = ring[i];
      const [bx, by] = ring[(i + 1) % ring.length];
      sum += ax * by - bx * ay;
    }
    return sum / 2;
  };

  for (const body of fills) {
    const layer = body.layerId;
    byLayer.set(layer, (byLayer.get(layer) ?? 0) + 1);

    const path = body.path;
    if (!Array.isArray(path) || !path.length) {
      throw new Error("FILL on layer " + layer + " has no path");
    }
    const ringTokens = Array.isArray(path[0]) ? path : [path];
    if (ringTokens.length > 1) withHoles++;

    for (let index = 0; index < ringTokens.length; index++) {
      const ring = decodeRing(ringTokens[index]);
      if (ring.length < 3) {
        throw new Error("FILL on layer " + layer + " has a degenerate ring");
      }
      const area = signedArea(ring);
      if (index === 0) windings[area < 0 ? "CW" : "CCW"] += 1;
    }
  }

  console.log("\nfills        : " + fills.length);
  console.log(
    "  by layer    : " +
      [...byLayer.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([id, n]) => "layer " + id + ": " + n)
        .join(", "),
  );
  console.log("  fills with interior rings: " + withHoles);
  console.log(
    "  outer ring winding: " +
      windings.CW +
      " clockwise, " +
      windings.CCW +
      " counter-clockwise",
  );
  console.log("  every fill has a well-formed path and at least one ring: ok");
}

if (lines_.length) {
  const MM = 0.0254;
  const byLayer = new Map();
  const widths = new Set();
  for (const body of lines_) {
    byLayer.set(body.layerId, (byLayer.get(body.layerId) ?? 0) + 1);
    for (const key of ["startX", "startY", "endX", "endY", "width"]) {
      if (typeof body[key] !== "number") {
        throw new Error("LINE on layer " + body.layerId + " is missing " + key);
      }
    }
    const length =
      Math.hypot(body.endX - body.startX, body.endY - body.startY) * MM;
    if (length <= 0) {
      throw new Error("LINE on layer " + body.layerId + " has zero length");
    }
    if (body.width * MM <= 0) {
      throw new Error("LINE on layer " + body.layerId + " has zero width");
    }
    widths.add((body.width * MM).toFixed(3));
  }
  console.log("\ntraces       : " + lines_.length);
  console.log(
    "  by layer    : " +
      [...byLayer.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([id, n]) => "layer " + id + ": " + n)
        .join(", "),
  );
  console.log("  widths (mm) : " + [...widths].sort().join(", "));
  console.log("  every trace has both endpoints and a positive width: ok");
}

// --- layer physics ---------------------------------------------------------
// isKeepIsland must be true wherever this tool writes copper. All artwork
// copper is net-less, and a false here makes the editor discard it while
// loading -- producing a board that opens cleanly with no copper on it at all.
//
// Only checked when the file actually carries copper. A board that is nothing
// but an outline has no islands to keep, so failing it would be a false alarm
// on every older sample.
if (layerPhys.length && (fills.length || lines_.length)) {
  const copperLayers = new Set([1, 2]);
  const offenders = [];
  for (const entry of layerPhys) {
    const isCopper =
      entry.body.material === "COPPER" ||
      copperLayers.has(Number(String(entry.id).split(",")[1]?.replace(/\D/g, "")));
    if (isCopper && entry.body.isKeepIsland !== true) {
      offenders.push(`${entry.id}: isKeepIsland=${entry.body.isKeepIsland}`);
    }
  }
  console.log("\nlayer physics: " + layerPhys.length);
  if (offenders.length) {
    throw new Error(
      "copper layer with isKeepIsland=false will be emptied by the editor: " +
        offenders.join("; "),
    );
  }
  console.log("  every copper layer keeps isolated copper: ok");
}

console.log("\nVERIFY OK");
