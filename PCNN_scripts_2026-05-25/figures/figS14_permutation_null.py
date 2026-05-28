#!/usr/bin/env python3
# Auto-split from regen_supp_fixes.py :: fig_s14()  — self-contained single-figure script
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
PIML = '/Users/anbu/Desktop/PIML/piml_ceramic'
OUT = '/Users/anbu/Desktop/NC figures/all_figures/supp'
EXTRACT = '/Users/anbu/Desktop/NC figures/all_figures/extracted_data'
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

def fig_s14(n_perm=2000, seed=0):
    df = _load_decomp()
    rng = np.random.default_rng(seed)

    def null_range(sub, label_col, groups):
        lab = sub[label_col].values
        flst = sub['flst'].values
        obs_means = {g: flst[lab == g].mean() for g in groups}
        obs_range = max(obs_means.values()) - min(obs_means.values())
        ranges = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(lab)
            m = [flst[perm == g].mean() for g in groups]
            ranges[i] = max(m) - min(m)
        p = np.sum(ranges >= obs_range) / n_perm
        z = (obs_range - ranges.mean()) / ranges.std()
        return (obs_range, ranges, p, z, obs_means)
    a_groups = ['Pb', 'Ca', 'La', 'Sr', 'Ba']
    oa, ra, pa, za, ma = null_range(df[df['A'].isin(a_groups)], 'A', a_groups)
    r_groups = sorted(df['regime'].dropna().unique().tolist())
    orr, rr, pr, zr, mr = null_range(df[df['regime'].isin(r_groups)], 'regime', r_groups)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4))
    style(a1)
    style(a2)
    for ax, (obs, null, p, z, ttl, xl) in zip((a1, a2), [(oa, ra, pa, za, 'a   A-site label-permutation null', 'Range of family-mean $f_{LST}$'), (orr, rr, pr, zr, 'b   Regime label-permutation null', 'Range of regime-mean $f_{LST}$')]):
        ax.hist(null, bins=40, color='#9FB8C8', edgecolor='white', linewidth=0.3, zorder=2)
        ax.axvline(obs, color='#C0392B', lw=1.6, zorder=4)
        ax.set_xlim(0, obs * 1.08)
        ptxt = 'p < 0.0005' if p == 0 else f'p = {p:.3f}'
        ax.text(0.5, 0.92, f'observed = {obs:.3f}\n{ptxt}\nz ≈ {z:.0f}$\\sigma$', transform=ax.transAxes, ha='center', va='top', fontsize=6.8, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCC', alpha=0.92))
        ax.annotate('observed', xy=(obs, ax.get_ylim()[1] * 0.5), xytext=(obs * 0.86, ax.get_ylim()[1] * 0.62), fontsize=6.3, color='#C0392B', ha='right', arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.8))
        ax.set_title(ttl, loc='left')
        ax.set_xlabel(xl)
        ax.set_ylabel('Count')
    fig.suptitle('Label-permutation null distributions (2,000 randomisations)', fontsize=9.5, fontweight='bold', y=1.03)
    fig.tight_layout()
    savefig(fig, 'figS14_permutation_null')
    print(f'  S14 A-site: obs={oa:.3f} p={pa:.4f} z={za:.1f}  means={ {k: round(v, 3) for k, v in ma.items()}}')
    print(f'  S14 regime: obs={orr:.3f} p={pr:.4f} z={zr:.1f}')
if __name__ == '__main__':
    fig_s14()
