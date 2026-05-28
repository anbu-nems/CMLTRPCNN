#!/usr/bin/env python3
"""
Fig 2 — PCNN Architecture Diagram
Nature Communications: Physics-Constrained Neural Network for ABO3 dielectric prediction

Layout (left → right):
  [Input Features] → [Trunk Network] → [3 Physics Branches] → [Decomposed Output]
  CM baseline runs independently and feeds directly to output sum.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = "./figures_output/main"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 150, "savefig.dpi": 600,
})

# Ocean Dusk palette
C = {
    "input":   "#5E81AC",   # slate blue   — input
    "trunk":   "#264653",   # deep teal    — trunk
    "cm":      "#6C757D",   # gray         — CM physics anchor
    "lst":     "#2A9D8F",   # teal         — LST branch
    "tilt":    "#E9C46A",   # gold         — Tilt branch
    "res":     "#F4A261",   # orange       — Residual branch
    "output":  "#E76F51",   # coral        — output
    "arrow":   "#444444",
    "bg":      "#FFFFFF",
    "band":    "#F7F9FB",
}

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.set_xlim(0, 7.2)
ax.set_ylim(0, 3.8)
ax.axis("off")
ax.set_facecolor(C["bg"])
fig.patch.set_facecolor(C["bg"])

# ── Helper functions ──────────────────────────────────────────────────────────
def box(cx, cy, w, h, label, color, sub=None, fontsize=7.5, textcolor="white",
        radius=0.08, lw=1.0, alpha=1.0):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=color, edgecolor="white", linewidth=lw,
                       alpha=alpha, zorder=3)
    ax.add_patch(b)
    yo = cy + 0.055 if sub else cy
    ax.text(cx, yo, label, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight="bold", zorder=4)
    if sub:
        ax.text(cx, cy - 0.065, sub, ha="center", va="center",
                fontsize=fontsize - 1.0, color=textcolor, alpha=0.88, zorder=4)

def arrow(x1, y1, x2, y2, color="#444", lw=1.2, rad=0.0, dashed=False):
    ls = (0, (4, 3)) if dashed else "solid"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                linestyle=ls, mutation_scale=8),
                zorder=5)

def label(x, y, txt, color="#555", fs=6.5, bold=False, ha="center"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 — Input features  (x ≈ 0.62)
# ══════════════════════════════════════════════════════════════════════════════
x_in = 0.68
box(x_in, 2.85, 1.10, 0.38, "LST features", C["lst"], sub="30 features", fontsize=7)
box(x_in, 2.27, 1.10, 0.38, "Tilt features", C["tilt"], sub="30 features", fontsize=7, textcolor="#333")
box(x_in, 1.69, 1.10, 0.38, "Residual features", C["res"], sub="43 features", fontsize=7)

# CM anchor box (separate physics path, lower)
box(x_in, 0.90, 1.10, 0.38, "CM baseline", C["cm"], sub=r"$\varepsilon_r^{CM}$ (Shannon)", fontsize=7)

# Feature group label
ax.text(x_in, 3.40, "Input  (103 features)", ha="center", va="center",
        fontsize=8, color=C["input"], fontweight="bold")

# Bracket around feature inputs
for yb in [2.85, 2.27, 1.69]:
    arrow(x_in + 0.55, yb, 1.50, yb, color=C["arrow"], lw=1.0)

# CM goes directly to output (bypass trunk)
arrow(x_in + 0.55, 0.90, 5.90, 0.90, color=C["cm"], lw=1.2, dashed=True)
label(3.25, 0.73, "Physics anchor (parameter-free)", color=C["cm"], fs=6.5)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 2 — Trunk  (x ≈ 2.05)
# ══════════════════════════════════════════════════════════════════════════════
x_tr = 2.05
trunk_box = FancyBboxPatch(
    (x_tr - 0.45, 2.27 - 0.71), 0.90, 1.42,
    boxstyle="round,pad=0,rounding_size=0.10",
    facecolor=C["trunk"], edgecolor="white", linewidth=1.0, zorder=3,
)
ax.add_patch(trunk_box)
ax.text(x_tr, 2.63, "Shared\ntrunk", ha="center", va="center",
        fontsize=8.0, color="white", fontweight="bold", linespacing=1.15, zorder=4)
ax.text(x_tr, 2.20, "384 → 384 → 192", ha="center", va="center",
        fontsize=6.0, color="white", alpha=0.90, zorder=4)
ax.text(x_tr, 1.93, "ResBlock + SiLU", ha="center", va="center",
        fontsize=5.8, color="white", alpha=0.90, zorder=4)
ax.text(x_tr, 3.40, "Shared", ha="center", va="center",
        fontsize=8, color=C["trunk"], fontweight="bold")

# Trunk → 3 branch inputs
for yb, xb_out in [(2.85, 2.80), (2.27, 2.80), (1.69, 2.80)]:
    arrow(x_tr + 0.45, yb, xb_out - 0.01, yb, color=C["arrow"], lw=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 — Three physics branches  (x ≈ 3.50)
# ══════════════════════════════════════════════════════════════════════════════
x_br = 3.50

# LST branch
box(x_br, 2.85, 1.30, 0.40, "LST branch", C["lst"],
    sub="96→48→Softplus  (δ≥0)", fontsize=7)

# Tilt branch
box(x_br, 2.27, 1.30, 0.40, "Tilt branch", C["tilt"],
    sub="64→32→−Softplus  (δ≤0)", fontsize=7, textcolor="#333")

# Residual branch
box(x_br, 1.69, 1.30, 0.40, "Residual branch", C["res"],
    sub="256→256→128→σ·tanh", fontsize=7)

ax.text(x_br, 3.40, "Physics Branches", ha="center", va="center",
        fontsize=8, color="#555", fontweight="bold")

# constraint annotations (right of branch boxes)
for y_c, txt, col in [(2.85, "≥ 0", C["lst"]),
                       (2.27, "≤ 0", "#C8A000"),
                       (1.69, "confidence\ngated", C["res"])]:
    ax.text(x_br + 0.72, y_c, txt, ha="left", va="center",
            fontsize=6.0, color=col, fontweight="bold", zorder=6)

# Branch → summation node
x_sum = 5.22
for yb in [2.85, 2.27, 1.69]:
    arrow(x_br + 0.65, yb, x_sum - 0.22, yb, color=C["arrow"], lw=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 4 — Summation node  (x ≈ 5.22)
# ══════════════════════════════════════════════════════════════════════════════
circle = plt.Circle((x_sum, 2.27), 0.20, color=C["trunk"],
                    ec="white", lw=1.0, zorder=3)
ax.add_patch(circle)
ax.text(x_sum, 2.27, "Σ", ha="center", va="center",
        fontsize=13, color="white", fontweight="bold", zorder=4)

# CM dashed arrow arrives at sum node
arrow(5.90, 0.90, x_sum, 2.07, color=C["cm"], lw=1.2, dashed=True, rad=-0.25)

# Sum → output
arrow(x_sum + 0.20, 2.27, 5.90, 2.27, color=C["arrow"], lw=1.3)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 5 — Decomposed output  (x ≈ 6.45)
# ══════════════════════════════════════════════════════════════════════════════
x_out = 6.45
box(x_out, 2.27, 1.38, 1.52, "Prediction", C["output"], fontsize=8, radius=0.10)

# Decomposition formula lines inside output box
lines = [
    (r"$\varepsilon_r = \varepsilon_r^{CM}$", "#ddd", 7.5),
    (r"$+ \,\delta_{LST}\;\;(\geq 0)$",       C["lst"],  7.0),
    (r"$+ \,\delta_{tilt}\,(\leq 0)$",        "#F8D56A", 7.0),
    (r"$+ \,\delta_{res}$",                   C["res"],  7.0),
]
y_start = 2.70
for txt, col, fs in lines:
    ax.text(x_out, y_start, txt, ha="center", va="center",
            fontsize=fs, color=col, fontweight="bold", zorder=5)
    y_start -= 0.31

# conformal CI annotation
ax.text(x_out, 1.57,
        "90% CI: pred ± 14.38\n(conformal, exact coverage)",
        ha="center", va="center", fontsize=5.8, color="white", alpha=0.85,
        zorder=5)

ax.text(x_out, 3.40, "Output", ha="center", va="center",
        fontsize=8, color=C["output"], fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# Performance stats box (bottom right)
# ══════════════════════════════════════════════════════════════════════════════
stats_box = FancyBboxPatch((5.58, 0.12), 1.55, 0.62,
                            boxstyle="round,pad=0,rounding_size=0.06",
                            facecolor="#F0F4F8", edgecolor="#B0C4DE",
                            linewidth=0.8, zorder=3)
ax.add_patch(stats_box)
ax.text(6.35, 0.62, "Model performance", ha="center", va="center",
        fontsize=6.5, color="#264653", fontweight="bold", zorder=4)
ax.text(6.35, 0.43, r"$R^2$ = 0.941  (held-out,  $n$=116)", ha="center", va="center",
        fontsize=6.2, color="#333", zorder=4)
ax.text(6.35, 0.27, "Strat-GSS CV:  0.893 ± 0.044", ha="center", va="center",
        fontsize=6.2, color="#333", zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# Legend: branch colour map
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(color=C["lst"],    label=r"$\delta_{LST}$: soft-mode (≥0)"),
    mpatches.Patch(color=C["tilt"],   label=r"$\delta_{tilt}$: octahedral tilt (≤0)"),
    mpatches.Patch(color=C["res"],    label=r"$\delta_{res}$: confidence-gated residual"),
    mpatches.Patch(color=C["cm"],     label=r"$\varepsilon_r^{CM}$: Clausius-Mossotti"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=6.0,
          handlelength=1.0, handleheight=0.9,
          framealpha=0.93, edgecolor="#CCCCCC",
          bbox_to_anchor=(0.0, 0.0))

# Title
ax.text(3.60, 3.72,
        "Physics-Constrained Neural Network (PCNN) — CMLTRPCNNv7.7",
        ha="center", va="center", fontsize=9, color="#1a1a2e", fontweight="bold")

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ["pdf", "png", "svg"]:
    path = os.path.join(OUT_DIR, f"fig2_architecture.{ext}")
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor=C["bg"])
    print(f"Saved: {path}")
plt.close(fig)
print("Fig 2 — Architecture done.")
