// Records what styles/*.css looked like when the stylesheet was last
// compiled, so run.sh / run.ps1 can tell whether static/tailwind-*.css is
// still in step with its source.
//
// Timestamps can't answer that: the Tailwind CLI leaves the output file
// untouched when a rebuild produces identical bytes, so the built file
// stays "older" than its source forever and a date comparison warns for
// the rest of time. A hash of the sources is exact.
//
// Run as the last step of `npm run css`; never run by hand.
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const SOURCES = ["styles/index.css", "styles/result.css"];

const lines = SOURCES.map((rel) => {
  const sha = crypto.createHash("sha256").update(fs.readFileSync(path.join(ROOT, rel))).digest("hex");
  return `${sha}  ${rel}`;
});
fs.writeFileSync(path.join(ROOT, "static/styles.sha256"), lines.join("\n") + "\n");
console.log("stamped static/styles.sha256");
