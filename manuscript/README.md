# Manuscript — submission deliverables

Snapshot of the submission of the CMLTRPCNN manuscript ABO₃
microwave-dielectric study. Copied 2026-05-26 from the build directory; **byte-identical**
to the rendered sources at the time of copy.

| File | What it is |
|------|------------|
| `manuscript.docx` | Main manuscript (Abstract → Introduction → Results → Discussion → Methods → References), figures embedded, captions at 9 pt |
| `supplementary.docx` | Supplementary Information (supplementary figures + notes + tables) |
| `captions.md` | Source for all main + supplementary figure captions |
| `references.json` | Citation source of truth (author / journal / volume / pages / year), used to number citations |

Manuscript in preparation / submitted; journal information to be added on acceptance.

## Reproducing the `.docx`
These are a **snapshot**, not a live link. They are rebuilt in `piml_ceramic/manuscript_build/`:

```bash
cd piml_ceramic/manuscript_build
node build_manuscript.js      # → manuscript.docx   (reads manuscript_numbered.md + captions.md + figures)
node build_supplementary.js   # → supplementary.docx (reads captions.md + figures)
```

If you re-edit and rebuild there, re-copy the four files into this folder to keep the all-in-one bundle current.
