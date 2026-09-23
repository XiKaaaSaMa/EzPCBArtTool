#!/usr/bin/env node
// Show the header shape (and one sample body) for selected record types,
// grouped by the document section they appear in. Useful for reproducing the
// exact id conventions EasyEDA Pro uses.
//
// Usage: node tools/dump_headers.mjs <file.epru> [TYPE,TYPE,...]

import fs from "node:fs";

const file = process.argv[2];
const filter = process.argv[3] ? new Set(process.argv[3].split(",")) : null;
if (!file) {
  console.error("usage: node tools/dump_headers.mjs <file.epru> [TYPE,TYPE,...]");
  process.exit(1);
}

const lines = fs.readFileSync(file, "utf8").split(/\r?\n/);
let doc = "(none)";
const seen = new Set();

for (const line of lines) {
  if (!line.trim()) continue;
  const sep = line.indexOf("}||");
  if (sep === -1) continue;
  let header;
  let body;
  try {
    header = JSON.parse(line.slice(0, sep + 1));
    body = JSON.parse(line.slice(sep + 3).replace(/\|$/, ""));
  } catch {
    continue;
  }

  if (header.type === "DOCHEAD") {
    doc = body?.docType ?? "?";
    continue;
  }
  if (filter && !filter.has(header.type)) continue;

  const key = `${doc}/${header.type}`;
  if (seen.has(key)) continue;
  seen.add(key);

  console.log(`${key}`);
  console.log(`  HDR  ${JSON.stringify(header)}`);
  console.log(`  BODY ${JSON.stringify(body).slice(0, 420)}`);
}
