#!/usr/bin/env python3
# Auto-split from regen_supp_fixes.py :: fig_s22()  — self-contained single-figure script
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
OUT = './figures_output/supp'
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

def fig_s22():
    df = _load_decomp()
    order = ['Pb', 'Ca', 'La', 'Sr', 'Ba']
    g = df[df['A'].isin(order)].groupby('A').agg(n=('idx', 'size'), er_cm=('er_cm', 'mean'), dlst=('delta_lst', 'mean'), dtilt=('delta_tilt', 'mean'), dres=('delta_res', 'mean'), flst=('flst', 'mean')).reindex(order)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'width_ratios': [1.5, 1.0]})
    style(a1)
    style(a2)
    x = np.arange(len(order))
    a1.bar(x, g['er_cm'], color='#264653', edgecolor='white', label='$\\varepsilon_r^{CM}$')
    a1.bar(x, g['dlst'], bottom=g['er_cm'], color='#2A9D8F', edgecolor='white', label='$\\delta_{LST}$')
    a1.bar(x, g['dtilt'], bottom=g['er_cm'] + g['dlst'], color='#E9C46A', edgecolor='white', label='$\\delta_{tilt}$')
    a1.bar(x, g['dres'], bottom=g['er_cm'] + g['dlst'] + g['dtilt'], color='#E76F51', edgecolor='white', label='$\\delta_{res}$')
    tot = g['er_cm'] + g['dlst'] + g['dtilt'] + g['dres']
    for xi, t in zip(x, tot):
        a1.text(xi, t + 1.5, f'{t:.0f}', ha='center', fontsize=6.5, fontweight='bold')
    a1.set_xticks(x)
    a1.set_xticklabels([f"{a}\n($n$={int(g.loc[a, 'n'])})" for a in order])
    a1.set_ylabel('Mean contribution to $\\varepsilon_r$')
    a1.set_title('a   Per-A-site decomposition', loc='left')
    a1.legend(loc='upper right', fontsize=6.3, ncol=2, handlelength=1.0)
    a1.set_ylim(0, tot.max() * 1.18)   # bars anchored at zero (touch x-axis)
    a2.bar(x, g['flst'], color=[ASITE_COLORS[a] for a in order], edgecolor='white')
    for xi, v in zip(x, g['flst']):
        a2.text(xi, v + 0.03, f'{v:.2f}', ha='center', fontsize=6.8, fontweight='bold')
    a2.set_xticks(x)
    a2.set_xticklabels(order)
    a2.set_ylabel('$f_{LST} = \\langle\\delta_{LST}/\\varepsilon_r^{CM}\\rangle$')
    a2.set_title('b   Soft-mode enhancement', loc='left')
    a2.set_ylim(0, g['flst'].max() * 1.2)   # bars anchored at zero (touch x-axis)
    fig.suptitle('A-site-resolved physics decomposition', fontsize=9.5, fontweight='bold', y=1.03)
    fig.tight_layout()
    savefig(fig, 'figS22_element_attribution')
    print('  S22 f_LST:', dict(g['flst'].round(3)))
if __name__ == '__main__':
    fig_s22()
