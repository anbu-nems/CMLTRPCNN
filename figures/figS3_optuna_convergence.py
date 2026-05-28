#!/usr/bin/env python3
# Auto-split from gen_remaining_supplementary.py :: fig_optuna_convergence()  — self-contained single-figure script
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
ROOT_DIR = '/Users/anbu/Desktop/NC figures/all_figures'
OUT_DIR = f'{ROOT_DIR}/supp'
EXTRACT_DIR = f'{ROOT_DIR}/extracted_data'
PIML = '/Users/anbu/Desktop/PIML/piml_ceramic'
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

def fig_optuna_convergence():
    con = sqlite3.connect(f'{PIML}/results/hp_tuning_optuna.db')
    df = pd.read_sql_query("SELECT t.trial_id, t.state, v.value FROM trials t JOIN trial_values v ON t.trial_id = v.trial_id WHERE t.state='COMPLETE' AND v.objective=0 ORDER BY t.trial_id", con)
    con.close()
    df['best_so_far'] = df['value'].cummax()
    best_idx = df['value'].idxmax()
    best_trial = int(df.loc[best_idx, 'trial_id'])
    best_val = float(df.loc[best_idx, 'value'])
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    style4(ax)
    ax.scatter(df['trial_id'], df['value'], s=12, color=GRAY, alpha=0.6, edgecolors='white', linewidths=0.3, label='Trial value', zorder=3)
    ax.plot(df['trial_id'], df['best_so_far'], color=OURS, lw=1.5, label='Best so far', zorder=4)
    ax.axhline(best_val, color=OURS, ls=':', lw=0.8, zorder=2)
    ax.scatter([best_trial], [best_val], s=80, color=OURS, edgecolor='white', linewidth=1.0, zorder=5, marker='*', label=f'Best trial #{best_trial}\n($R^2$={best_val:.4f})')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Formula-split CV $R^2$')
    ax.set_title(f'Optuna HP tuning convergence  ($n$={len(df)} completed trials)', loc='left')
    ax.set_ylim(min(0.4, df['value'].min() - 0.02), 1.0)
    ax.legend(loc='lower right')
    save(fig, 'figS3_optuna_convergence')
if __name__ == '__main__':
    fig_optuna_convergence()
