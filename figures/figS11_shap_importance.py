#!/usr/bin/env python3
# Auto-split from regen_supp_fixes.py :: fig_s11()  — self-contained single-figure script
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

def fig_s11():
    part = json.load(open(f'{PIML}/data/processed/feature_partition_v7.json'))
    feat_names = part['LST'] + part['Tilt'] + part['Residual'] + ['er_CM', 'has_cm', 'cm_approx', 'GII_norm', 'phase_tr']

    def branch(f):
        if f in part['LST']:
            return 'LST'
        if f in part['Tilt']:
            return 'Tilt'
        if f in part['Residual']:
            return 'Residual'
        return 'Aux'
    shap = np.load(f'{PIML}/results/60_shap_values.npy', allow_pickle=False)
    assert shap.shape[1] == len(feat_names), (shap.shape, len(feat_names))
    fdf = pd.DataFrame({'feature': feat_names, 'branch': [branch(f) for f in feat_names], 'mean_abs': np.abs(shap).mean(0), 'mean_signed': shap.mean(0)}).sort_values('mean_abs', ascending=False).reset_index(drop=True)
    bdf = fdf.groupby('branch').agg(mean_abs=('mean_abs', 'sum'), n=('feature', 'count')).reindex(['LST', 'Residual', 'Tilt', 'Aux'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 5.2), gridspec_kw={'width_ratios': [2.0, 1.0]})
    style(a1)
    style(a2)
    top = fdf.head(20)
    y = np.arange(len(top))[::-1]
    a1.barh(y, top['mean_abs'], height=0.72, color=[BRANCH_COLORS[b] for b in top['branch']], edgecolor='white')
    a1.set_yticks(y)
    a1.set_yticklabels(top['feature'], fontsize=6.3)
    for yi, v, s in zip(y, top['mean_abs'], top['mean_signed']):
        a1.text(v + 0.08, yi, f"{v:.2f} {('↑' if s > 0 else '↓')}", va='center', fontsize=6.2, color='#222')
    a1.set_xlim(0, top['mean_abs'].max() * 1.22)   # bars anchored at zero (touch y-axis)
    a1.set_xlabel('Mean |SHAP|  (impact on $\\varepsilon_r$)')
    a1.set_title('a   Top-20 features by SHAP importance', loc='left')
    a1.legend(handles=[mpatches.Patch(color=BRANCH_COLORS[b], label=b) for b in ['LST', 'Tilt', 'Residual', 'Aux']], loc='lower right', title='Physics branch', title_fontsize=7, fontsize=6.5, handlelength=1.0)
    bp = np.arange(len(bdf))
    a2.barh(bp, bdf['mean_abs'], height=0.6, color=[BRANCH_COLORS[b] for b in bdf.index], edgecolor='white')
    a2.set_yticks(bp)
    a2.set_yticklabels(bdf.index, fontsize=8, fontweight='bold')
    a2.invert_yaxis()
    for yi, v, n in zip(bp, bdf['mean_abs'], bdf['n']):
        a2.text(v + bdf['mean_abs'].max() * 0.02, yi, f'{v:.1f}', va='center', fontsize=6.5, color='#222')
    a2.set_xlim(0, bdf['mean_abs'].max() * 1.25)   # bars anchored at zero (touch y-axis)
    a2.set_xlabel('Aggregated mean |SHAP|')
    a2.set_title('b   By physics branch', loc='left')
    fig.suptitle('PCNN feature attribution (SHAP)', fontsize=10, fontweight='bold', y=1.02)
    # (the "n test compositions · GradientExplainer · arrows show sign of mean effect" note lives in the caption)
    fig.tight_layout()
    savefig(fig, 'figS11_shap_importance')
    print('  S11 branch totals:', dict(bdf['mean_abs'].round(1)))

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
    fig_s11()
