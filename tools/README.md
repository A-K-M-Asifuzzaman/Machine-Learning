# tools/

Repository maintenance scripts.

## `check_math.js` — validate every LaTeX expression

GitHub renders math with **KaTeX**, which is stricter than LaTeX and rejects constructs that
look perfectly fine in a `.tex` file. Worse, GitHub's *markdown* parser runs **before** KaTeX and
mangles a few things on the way through — so an expression can be valid KaTeX and still break.

This script extracts every `$...$` and `$$...$$` expression in the repository and runs it through
KaTeX itself, so the check is exhaustive rather than a guess at what might be wrong.

```bash
npm install katex
node tools/check_math.js .
```

Exit code is 0 if everything renders, 1 otherwise — so it can gate a commit or a CI job.

### The traps it catches

| Written | What GitHub does | Use instead |
|---|---|---|
| `\#` | markdown eats the backslash, leaving a bare `#`, which KaTeX rejects as a **macro parameter character** | `\lvert\{\dots\}\rvert` |
| `\|` inside a table | parsed as an escaped **column separator**, so the math is cut in half | `\Vert` |
| `\operatorname{...}` | not in GitHub's KaTeX **macro allowlist** | `\mathrm{...}` |
| `\emph`, `\mbox`, `\newcommand`, … | no KaTeX equivalent | `\textit`, `\text`, or drop it |

The first two are the nasty ones, because the expression is *correct LaTeX* — it is the markdown
layer, not the math layer, that breaks it. Neither is visible until you look at the rendered page
on GitHub.

### Notes on the extractor

- Fenced code blocks and inline `` `code` `` spans are stripped first, so a `$` in a shell snippet
  or a `\` in a docstring is not mistaken for math.
- Display math is matched before inline math, so the `$` inside a `$$...$$` block is not
  re-matched.
- Line numbers are preserved through stripping, so failures point at the right place.
