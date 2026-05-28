#!/usr/bin/env python3
# Auto-split from gen_all_figures_additional.py :: fig_ad_dual()  — self-contained single-figure script
"""
Additional PCNN figures generated from results JSONs.

Outputs to ./figures_output/:
  fig_decomposition.pdf       — εr decomposition (CM 49.3 / LST 51.0 / Tilt -1.5 / Res 0.6)
  fig_asite_hierarchy.pdf     — A-site f_LST hierarchy + literature validation (Spearman=1.000)
  fig_law_ablation.pdf        — 2^3 factorial ablation of physics laws I/II/III
  fig_branch_ablation.pdf     — R² drop when each branch is removed
  fig_flst_by_regime.pdf      — f_LST by Reaney regime (Ia/Ib/II/III)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
RESULTS_DIR = './results'
OUT_DIR = './figures_output/supp'
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': True, 'legend.framealpha': 0.92, 'legend.edgecolor': '#CCCCCC', 'figure.constrained_layout.use': True, 'figure.dpi': 150, 'savefig.dpi': 300, 'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.grid': False})

def style4(ax, lw=1.0):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
    ax.tick_params(direction='in', top=False, right=False, bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)
C = {'cm': '#264653', 'lst': '#2A9D8F', 'tilt': '#E9C46A', 'res': '#E76F51', 'ours': '#E76F51', 'gray': '#78909C'}
ASITE_COLORS = {'Pb': '#E76F51', 'Ca': '#F4A261', 'La': '#2A9D8F', 'Sr': '#E9C46A', 'Ba': '#264653'}
REGIME_COLORS = {'Ia': '#264653', 'Ib': '#2A9D8F', 'II': '#E9C46A', 'III': '#E76F51'}

def save(fig, name):
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)

def fig_ad_dual():
    with open(f'{RESULTS_DIR}/49_ad_v77.json') as f:
        d = json.load(f)
    L1 = d['layer1_physics_gate']
    L2 = d['layer2_kde']
    COMB = d['combined_ad']
    VC = d['validation_calibration']
    asite = d['asite_ad']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    ax1, ax2, ax3 = axes
    for ax in axes:
        style4(ax)
    labels = ['Layer 1\nphysics gate\n(σ_conf)', 'Layer 2\nKDE density\n(PCA+KDE)', 'Combined\n(both layers)']
    pct_total = [L1['pct_physics_reliable'], L2['pct_train_in_domain'], COMB['pct_total_in_ad']]
    cols = [C['lst'], C['tilt'], C['ours']]
    x = np.arange(len(labels))
    bars = ax1.bar(x, pct_total, width=0.55, color=cols, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylabel('In-domain coverage (%)')
    ax1.set_title('a  Dual-layer AD coverage', loc='left')
    ax1.set_ylim(0, 110)
    ax1.axhline(100, color='#888', lw=0.6, ls=':')
    for b, v in zip(bars, pct_total):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#222')
    layer_lbls = ['Layer 1', 'Layer 2', 'Combined']
    r2_in = [VC['r2_layer1_in'], VC['r2_layer2_in'], VC['r2_combined_in']]
    r2_out = [VC['r2_layer1_out'], VC['r2_layer2_out'], VC['r2_combined_out']]
    x2 = np.arange(len(layer_lbls))
    w = 0.35
    b1 = ax2.bar(x2 - w / 2, r2_in, w, color=C['lst'], edgecolor='white', label='In-domain')
    b2 = ax2.bar(x2 + w / 2, r2_out, w, color='#B0BEC5', edgecolor='white', label='Out-of-domain')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(layer_lbls, fontsize=7)
    ax2.set_ylabel('Holdout $R^2$')
    ax2.set_title('b  Reliability — in vs out', loc='left')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower left', fontsize=6.5)
    for bars, vals in [(b1, r2_in), (b2, r2_out)]:
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, color='#222')
    a_order = ['Pb', 'La', 'Ba', 'Ca', 'Sr']
    pct_L1 = [asite[a]['pct_layer1_physics'] for a in a_order]
    pct_L2 = [asite[a]['pct_layer2_kde'] for a in a_order]
    pct_comb = [asite[a]['pct_combined_in_ad'] for a in a_order]
    ns = [asite[a]['n'] for a in a_order]
    y = np.arange(len(a_order))
    h = 0.27
    ax3.barh(y - h, pct_L1, h, color=C['lst'], edgecolor='white', label='Layer 1 (σ_conf)')
    ax3.barh(y, pct_L2, h, color=C['tilt'], edgecolor='white', label='Layer 2 (KDE)')
    ax3.barh(y + h, pct_comb, h, color=C['ours'], edgecolor='white', label='Combined')
    ax3.set_yticks(y)
    ax3.set_yticklabels([f'{a}\n($n$={n})' for a, n in zip(a_order, ns)], fontsize=7)
    ax3.set_xlabel('In-domain (%)')
    ax3.set_title('c  By A-site cation', loc='left')
    ax3.invert_yaxis()
    ax3.set_xlim(0, 110)   # bars anchored at zero (touch y-axis)
    ax3.legend(loc='lower right', fontsize=6.0, handlelength=0.8)
    fig.suptitle('Dual-layer applicability domain  (Layer 1: physics gate · Layer 2: KDE)', fontsize=9, fontweight='bold', y=1.04)
    save(fig, 'figS9_applicability_domain')
if __name__ == '__main__':
    fig_ad_dual()
