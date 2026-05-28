#!/usr/bin/env python3
# Auto-split from regen_supp_fixes.py :: fig_s13()  — self-contained single-figure script
"""
Regenerate four supplementary figures to match their (grounded) captions / text:

  figS11  PCNN SHAP feature attribution      (60_shap_values.npy + feature_partition_v7.json)
  figS13  B-site LST-routing fractions       (43_counterfactual_permutation.json)
  figS14  A-site / regime permutation null   (reproduced from decomposition_per_sample.csv)
  figS22  Per-A-site CM/LST/tilt/res decomp  (decomposition_per_sample.csv, A-site parsed)

Every number is read from a results file or reproduced deterministically from the
per-sample decomposition. No values are hard-coded.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json, os, re
PIML = '.'
OUT = './figures_output/all_figures/supp'
EXTRACT = './extracted_data'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': True, 'legend.framealpha': 0.92, 'legend.edgecolor': '#CCCCCC', 'figure.dpi': 150, 'savefig.dpi': 300, 'axes.linewidth': 0.9, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.grid': False})

def style(ax, lw=0.9):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
    ax.tick_params(direction='in', top=False, right=False, bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)
BRANCH_COLORS = {'LST': '#2A9D8F', 'Tilt': '#E9C46A', 'Residual': '#E76F51', 'Aux': '#264653'}
ASITE_COLORS = {'Pb': '#7A6F5B', 'Ba': '#5A8C3E', 'Sr': '#D4A82E', 'Ca': '#D67238', 'La': '#4A6FA5'}

def savefig(fig, stem):
    for ext in ('pdf', 'png'):
        p = f'{OUT}/{stem}.{ext}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print('saved', p)
    plt.close(fig)

def fig_s13():
    bs = json.load(open(f'{PIML}/results/43_counterfactual_permutation.json'))
    bs = bs['counterfactual']['b_site_summaries']
    rows = [(k, v['n_pairs'], v['lst_fraction']) for k, v in bs.items() if v['n_pairs'] >= 10]
    rows.sort(key=lambda r: r[2])
    labels = [r[0] for r in rows]
    ns = [r[1] for r in rows]
    fr = [r[2] * 100 for r in rows]
    allf = np.array([v['lst_fraction'] for v in bs.values()])
    mean_pct = allf.mean() * 100
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    style(ax)
    y = np.arange(len(labels))
    cols = ['#C0392B' if l == 'Y' else '#2A9D8F' for l in labels]
    ax.barh(y, fr, height=0.7, color=cols, edgecolor='white')
    ax.axvline(mean_pct, color='#333', lw=1.1, ls='--', zorder=4, label=f'mean = {mean_pct:.1f}%')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{l}  ($n$={n})' for l, n in zip(labels, ns)], fontsize=6.5)
    for yi, v in zip(y, fr):
        ax.text(v - 1.0, yi, f'{v:.1f}', va='center', ha='right', fontsize=6.0, color='white', fontweight='bold')
    ax.set_xlim(75, 101)
    ax.set_xlabel('Fraction of $\\Delta\\varepsilon_r$ routed through the LST branch (%)')
    ax.set_title('Counterfactual permutation: B-site-resolved LST routing', loc='left')
    ax.annotate(f'minimum: Y = {fr[0]:.1f}%', xy=(fr[0], 0), xytext=(fr[0] + 6, 1.4), fontsize=6.5, color='#C0392B', arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.9))
    ax.legend(loc='lower right', fontsize=7)
    fig.tight_layout()
    savefig(fig, 'figS13_counterfactual_heatmap')
    print(f'  S13 mean={mean_pct:.1f}%  min=Y {fr[0]:.1f}%  n_bsites={len(allf)}')

def _load_decomp():
    df = pd.read_csv(f'{EXTRACT}/decomposition_per_sample.csv')
    df = df[df['er_cm'] > 0].copy()

    def asite(f):
        m = re.match('([A-Z][a-z]?)', str(f))
        e = m.group(1) if m else 'Other'
        return e if e in ('Pb', 'Ca', 'La', 'Sr', 'Ba') else 'Other'
    df['A'] = df['formula'].apply(asite)
    df['flst'] = df['delta_lst'] / df['er_cm']
    return df
if __name__ == '__main__':
    fig_s13()
