#!/usr/bin/env python3
# Auto-split from gen_remaining_supplementary.py :: fig_lofo_parity()  — self-contained single-figure script
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
ROOT_DIR = './figures_output'
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

def fig_lofo_parity():
    lofo = pd.read_csv(f'{EXTRACT_DIR}/lofo_predictions_v77.csv')
    lofo['target'] = lofo['label'].str.replace('LOFO-', '')
    asite_targets = {'Pb', 'Ca', 'Ba', 'Sr', 'La'}
    regime_targets = {'Ia', 'Ib', 'II', 'III'}
    asite_rows = lofo[lofo['target'].isin(asite_targets)].copy()
    asite_rows['a_site'] = asite_rows['target']
    regime_rows = lofo[lofo['target'].isin(regime_targets)].copy()
    regime_rows['regime'] = regime_rows['target']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    style4(ax1)
    style4(ax2)
    lim = 150
    for ax in (ax1, ax2):
        ax.plot([0, lim], [0, lim], color='#333', lw=0.8, ls='--', zorder=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal', adjustable='box')
    if len(asite_rows) > 0:
        order = ['Pb', 'Ca', 'Ba', 'Sr', 'La']
        for a in order:
            sub = asite_rows[asite_rows['a_site'] == a]
            if len(sub) == 0:
                continue
            r2 = sub['lofo_r2'].iloc[0]
            ax1.scatter(sub['er_measured'], sub['er_predicted'], s=20, color=ASITE_COLORS.get(a, GRAY), alpha=0.75, edgecolors='white', linewidths=0.3, label=f'{a} ($R^2$={r2:+.3f})', zorder=3)
    ax1.set_xlabel('Measured $\\varepsilon_r$')
    ax1.set_ylabel('Predicted $\\varepsilon_r$')
    ax1.set_title('a  LOFO by A-site', loc='left')
    ax1.legend(loc='upper left', fontsize=6.5, handlelength=0.8, title='Held-out A-site', title_fontsize=6.5)
    if len(regime_rows) > 0:
        order = ['Ia', 'Ib', 'II', 'III']
        for r in order:
            sub = regime_rows[regime_rows['regime'] == r]
            if len(sub) == 0:
                continue
            r2 = sub['lofo_r2'].iloc[0]
            ax2.scatter(sub['er_measured'], sub['er_predicted'], s=20, color=REGIME_COLORS[r], alpha=0.75, edgecolors='white', linewidths=0.3, marker=REGIME_MARK[r], label=f'Regime {r} ($R^2$={r2:+.2f})', zorder=3)
    ax2.set_xlabel('Measured $\\varepsilon_r$')
    ax2.set_ylabel('Predicted $\\varepsilon_r$')
    ax2.set_title('b  LOFO by Reaney regime', loc='left')
    ax2.legend(loc='upper left', fontsize=6.5, handlelength=0.8, title='Held-out regime', title_fontsize=6.5)
    fig.suptitle('Leave-One-Family-Out generalisation', fontsize=9, fontweight='bold', y=1.03)
    save(fig, 'figS12_lofo_parity')
if __name__ == '__main__':
    fig_lofo_parity()
