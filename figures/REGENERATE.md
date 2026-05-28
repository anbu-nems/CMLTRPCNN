# Regenerating the figures (PCNN v7.7 release)

Final published figures ship in `../figures_output/main/` (6) and `../figures_output/supp/` (23).
**Every figure now has exactly one script, named after its output PNG**, in this `figures/` folder.
Scripts auto-locate the release root (`RELEASE_ROOT` walks up to the folder containing
`model_weights/`) and read bundled data from `../results/`, `../data/`, `../extracted_data/`.
No values are hard-coded.

## Regenerate
```bash
cd figures
python figS13_counterfactual_heatmap.py     # one supplementary figure
python gen_fig6_limits.py                    # one main figure
# or everything:
for f in figS*.py;     do python "$f"; done   # -> ../figures_output/supp/
for f in gen_fig*_*.py; do python "$f"; done  # -> ../figures_output/main/
```

See **`FIGURE_MAP.md`** for the full figure → script → data table, including the main-figure
manuscript-number remap (file `fig4_discovery` = manuscript Fig 2, etc.). Superseded bundled
generators and alternate/style scripts are kept in **`archive/`** (not needed for reproduction).

## Two PNG names kept for build compatibility (name ≠ current content)
- `figS13_counterfactual_heatmap.png` → now B-site LST-routing chart
- `figS22_element_attribution.png` → now A-site CM/LST/tilt decomposition

## Expected sanity-check values
- Fig 6a LOFO: Pb −0.097 … La 0.849 · 6b in-AD Pb 74.4 … Sr 23.3 · 6c 70/72 · 6d GA Pb-free ≈147
- figS13 96.0 % (min Y 80.3 %) · figS14 A-site range 1.622 (z≈17σ) · figS22 f_LST Pb 2.28 … Ba 0.66
- figS3 99 of 100 Optuna trials, best trial 98 (R²=0.9284)
