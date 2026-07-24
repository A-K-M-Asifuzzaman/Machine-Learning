#!/usr/bin/env node
/**
 * check_math.js — validate every LaTeX expression in the repository's markdown.
 *
 * GitHub renders math with KaTeX, which is stricter than LaTeX and rejects several
 * constructs that look fine in a .tex file. Rather than guessing which ones, this script
 * extracts every $...$ and $$...$$ expression and runs it through KaTeX itself, reporting
 * the file, line, and error for anything that fails.
 *
 * It also flags constructs that KaTeX *accepts* but GitHub mangles before KaTeX sees
 * them — the markdown parser runs first, and it eats some backslashes and treats `|` as a
 * table separator.
 *
 * Usage:
 *     node tools/check_math.js [root]
 *
 * Exit code 0 if everything renders, 1 otherwise — so it can gate a commit or CI job.
 */

const fs = require("fs");
const path = require("path");
const katex = require("katex");

// ---------------------------------------------------------------------------
// Constructs KaTeX accepts but GitHub's markdown pipeline breaks first.
// ---------------------------------------------------------------------------
const MARKDOWN_HAZARDS = [
  {
    pattern: /\\#/,
    message:
      "\\# — markdown consumes the backslash, leaving a bare # that KaTeX rejects " +
      "as a macro parameter character. Use \\lvert\\{...\\}\\rvert for cardinality.",
  },
  {
    pattern: /\\\|/,
    message:
      "\\| — inside a table this is parsed as an escaped column separator. Use \\Vert.",
  },
  {
    pattern: /\\operatorname/,
    message: "\\operatorname is not in GitHub's KaTeX macro allowlist. Use \\mathrm.",
  },
  {
    pattern: /\\(emph|textsc|mbox|includegraphics|newcommand|renewcommand|def)\b/,
    message: "LaTeX command with no KaTeX equivalent.",
  },
  {
    // `\\` followed by `[` starts an OPTIONAL spacing argument `\\[dimen]`. That is fine
    // when the bracket holds a real dimension (`\\[4pt]`), but a `cases`/`aligned` row
    // whose first cell is a bracketed expression — `\\ [-1, 1]` — is mis-parsed: the
    // renderer tries to read `-1, 1` as a length, fails, and the row structure collapses
    // into "Extra close brace / missing open brace". KaTeX is lenient about it; MathJax
    // and stricter renderers are not. Brace the cell: `\\ {[-1, 1]}`.
    pattern: /\\\\\s*\[(?![0-9.\s]*(pt|px|em|ex|mu|cm|mm|in|bp|pc|dd|cc|sp|ex|em)\])/,
    message:
      "\\\\ followed by [ that is not a spacing dimension — a cases/aligned row " +
      "starting with a bracket is mis-parsed as \\\\[dimen]. Brace it: \\\\ {[...]}.",
  },
];

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/** Replace fenced code blocks with blank lines so line numbers stay correct. */
function stripCodeBlocks(text) {
  const lines = text.split("\n");
  let inFence = false;
  return lines
    .map((line) => {
      if (/^\s*```/.test(line)) {
        inFence = !inFence;
        return "";
      }
      return inFence ? "" : line;
    })
    .join("\n");
}

/** Replace inline `code` spans, which may legitimately contain $ or \. */
function stripInlineCode(text) {
  return text.replace(/`[^`\n]*`/g, (m) => " ".repeat(m.length));
}

function lineOf(text, index) {
  return text.slice(0, index).split("\n").length;
}

function extractExpressions(raw) {
  const text = stripInlineCode(stripCodeBlocks(raw));
  const found = [];

  // Display math first, so its inner $ are not re-matched as inline.
  const displayRanges = [];
  const displayRe = /\$\$([\s\S]+?)\$\$/g;
  let m;
  while ((m = displayRe.exec(text)) !== null) {
    found.push({ tex: m[1], display: true, line: lineOf(text, m.index) });
    displayRanges.push([m.index, m.index + m[0].length]);
  }

  const insideDisplay = (i) => displayRanges.some(([a, b]) => i >= a && i < b);

  // Inline math: a $ not adjacent to another $, with no blank line inside.
  const inlineRe = /(?<!\$)\$(?!\$)([^\n$]+?)(?<!\$)\$(?!\$)/g;
  while ((m = inlineRe.exec(text)) !== null) {
    if (insideDisplay(m.index)) continue;
    found.push({ tex: m[1], display: false, line: lineOf(text, m.index) });
  }

  return found;
}

// ---------------------------------------------------------------------------
// Checking
// ---------------------------------------------------------------------------

function checkFile(file) {
  const raw = fs.readFileSync(file, "utf8");
  const problems = [];

  for (const expr of extractExpressions(raw)) {
    for (const hazard of MARKDOWN_HAZARDS) {
      if (hazard.pattern.test(expr.tex)) {
        problems.push({
          line: expr.line,
          kind: "markdown",
          message: hazard.message,
          tex: expr.tex.trim().slice(0, 90),
        });
      }
    }

    try {
      katex.renderToString(expr.tex, {
        displayMode: expr.display,
        throwOnError: true,
        strict: false,
      });
    } catch (err) {
      problems.push({
        line: expr.line,
        kind: "katex",
        message: String(err.message).replace(/\s+/g, " ").slice(0, 160),
        tex: expr.tex.trim().slice(0, 90),
      });
    }
  }

  return problems;
}

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === ".git" || entry.name === "node_modules") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (entry.name.endsWith(".md")) out.push(full);
  }
  return out;
}

// ---------------------------------------------------------------------------

function main() {
  const root = process.argv[2] || ".";
  const files = walk(root).sort();

  let totalExpressions = 0;
  let totalProblems = 0;

  for (const file of files) {
    const relative = path.relative(root, file);
    const raw = fs.readFileSync(file, "utf8");
    const count = extractExpressions(raw).length;
    totalExpressions += count;

    const problems = checkFile(file);
    if (problems.length === 0) {
      if (count > 0) console.log(`  ok    ${relative}  (${count} expressions)`);
      continue;
    }

    console.log(`  FAIL  ${relative}`);
    for (const p of problems) {
      console.log(`          line ${p.line} [${p.kind}] ${p.message}`);
      console.log(`            ${p.tex}`);
      totalProblems++;
    }
  }

  console.log(
    `\n${totalExpressions} expressions checked, ${totalProblems} problem(s).`
  );
  process.exit(totalProblems === 0 ? 0 : 1);
}

main();
