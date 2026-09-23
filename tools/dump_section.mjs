#!/usr/bin/env node
// Dump records belonging to one document section of an .epru log.
//
// Usage: node tools/dump_section.mjs <file.epru> <docType> [limit]

import fs from "node:fs";

const [file, wantDoc, limitArg] = process.argv.slice(2);
if (!file || !wantDoc) {
  console.error("usage: node tools/dump_section.mjs <file.epru> <docType> [limit]");
  process.exit(1);
}
const limit = Number(limitArg) || 20;

const lines = fs.readFileSync(file, "utf8").split(/\r?\n/);
let inDoc = false;
let shown = 0;

for (const [i, line] of lines.entries()) {
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
    inDoc = body.docType === wantDoc;
    if (inDoc) console.log(`DOCHEAD  ${JSON.stringify(body)}`);
    continue;
  }
  if (!inDoc) continue;
  if (shown++ >= limit) break;
  console.log(
    `${String(i + 1).padStart(4)} ${header.type.padEnd(14)} ticket=${String(header.ticket).padStart(5)}` +
      (header.id ? ` id=${header.id}` : ""),
  );
  console.log(`     ${JSON.stringify(body).slice(0, 700)}`);
}
