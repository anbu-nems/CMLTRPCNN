#!/usr/bin/env python3
# Auto-split from gen_remaining_supplementary.py :: fig_sigma_conf_vs_error()  — self-contained single-figure script
"""
Generate 6 remaining supplementary figures using extracted per-sample data
+ Optuna database.

Outputs (all in all_figures/):
  figS3_optuna_convergence.pdf/png
  figS11_shap_importance.pdf/png
  figS12_lofo_parity.pdf/png
  figS13_counterfactual_heatmap.pdf/png
  figS14_permutation_null.pdf/png
  figS15_sigma_conf_vs_error.pdf/png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sqlite3
import json
import os
ROOT_DIR = './figures_output/all_figures'
OUT_DIR = f'{ROOT_DIR}/supp'
EXTRACT_DIR = f'{ROOT_DIR}/extracted_data'
PIML = '.'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': True, 'legend.framealpha': 0.92, 'legend.edgecolor': '#CCCCCC', 'figure.constrained_layout.use': True, 'figure.dpi': 150, 'savefig.dpi': 300, 'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.grid': False})

def style4(ax, lw=1.0):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
    ax.tick_params(direction='in', top=False, right=False, bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)
REGIME_COLORS = {'Ia': '#264653', 'Ib': '#2A9D8F', 'II': '#E9C46A', 'III': '#E76F51'}
REGIME_MARK = {'Ia': 'o', 'Ib': 's', 'II': '^', 'III': 'D'}
ASITE_COLORS = {'Pb': '#E76F51', 'Ca': '#F4A261', 'La': '#2A9D8F', 'Sr': '#E9C46A', 'Ba': '#264653'}
OURS = '#E76F51'
GRAY = '#B0BEC5'
TEAL = '#2A9D8F'
GOLD = '#E9C46A'

def save(fig, name):
    for ext in ('pdf', 'png'):
        p = f'{OUT_DIR}/{name}.{ext}'
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'Saved: {p}')
    plt.close(fig)

def fig_sigma_conf_vs_error():
    d = pd.read_csv(f'{EXTRACT_DIR}/decomposition_per_sample.csv')
    d['abs_error'] = (d['er_measured'] - d['er_predicted']).abs()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))
    style4(ax1)
    style4(ax2)
    for reg in ['Ia', 'Ib', 'II', 'III']:
        sub = d[d['regime'] == reg]
        ax1.scatter(sub['sigma_conf'], sub['abs_error'], s=10, alpha=0.55, color=REGIME_COLORS[reg], marker=REGIME_MARK[reg], edgecolors='white', linewidths=0.25, label=f'Regime {reg}', zorder=3)
    ax1.axhline(14.38, color='#555', lw=0.8, ls='--', label='Conformal $\\hat{q}_{90}=14.38$')
    ax1.axvline(0.35, color='#888', lw=0.6, ls=':', label='Layer-1 gate = 0.35')
    ax1.set_xlabel('$\\sigma_{\\rm conf}$ (confidence gate)')
    ax1.set_ylabel('|Predicted − Measured| $\\varepsilon_r$')
    ax1.set_title('a  Applicability domain validation', loc='left')
    ax1.legend(loc='upper right', fontsize=6.0, handlelength=0.8)
    bins = np.linspace(0, 1.0, 11)
    centers = 0.5 * (bins[1:] + bins[:-1])
    means, stds = ([], [])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d['sigma_conf'] >= lo) & (d['sigma_conf'] < hi)
        if m.sum() < 5:
            means.append(np.nan)
            stds.append(np.nan)
            continue
        means.append(d.loc[m, 'abs_error'].mean())
        stds.append(d.loc[m, 'abs_error'].std())
    means = np.array(means)
    stds = np.array(stds)
    ax2.errorbar(centers, means, yerr=stds, marker='o', markersize=5, color=OURS, ecolor='#555', elinewidth=0.8, capsize=2, zorder=3, label='Bin mean ± std')
    ax2.axhline(14.38, color='#555', lw=0.8, ls='--')
    ax2.set_xlabel('$\\sigma_{\\rm conf}$ bin centre')
    ax2.set_ylabel('Mean $|$error$|$ in bin')
    ax2.set_title('b  Error grows with σ_conf — physics gate works', loc='left')
    ax2.set_xlim(0, 1)
    ax2.legend(loc='upper left', fontsize=6.5)
    save(fig, 'figS15_sigma_conf_vs_error')
if __name__ == '__main__':
    fig_sigma_conf_vs_error()
