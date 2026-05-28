import os

# --- self-contained release root (auto-injected) ---
RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
# ----------------------------------------------------

"""
Figure S6 — Feature partition schematic
Publication-quality figure for Nature Communications supplementary.
Run from: /Users/anbu/Desktop/PIML/piml_ceramic
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch



# ── NC journal style (font only — diagram uses ax.axis("off")) ────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 150, "savefig.dpi": 300,
})

C_ANCHOR = "#264653"
C_LST    = "#2A9D8F"
C_TILT   = "#E9C46A"
C_RES    = "#E76F51"
FIG_HALF = (3.5, 3.5)

ANCHOR_LABEL = "$\\varepsilon_r^{\\mathrm{CM}}$  (0 params) — Clausius-Mossotti anchor"
LST_LABEL    = "LST Branch  |  30 features"
LST_ITEMS    = (
    "Goldschmidt $t$, soft-mode proxy,\n"
    "regime $I_a / I_b / II / III$, ionic radii\n"
    "ratios, B-site $d$-electron count,\n"
    "polarizability index…"
)
TILT_LABEL  = "Tilt Branch  |  30 features"
TILT_ITEMS  = (
    "Octahedral tilt-angle proxy, bond-\n"
    "angle variance, A-site size mismatch,\n"
    "B-site charge ordering…"
)
RES_LABEL  = "Residual Branch  |  43 features"
RES_ITEMS  = (
    "Sintering temp / time, crystal system,\n"
    "space group, processing history,\n"
    "VCA flag, GII…"
)

fig, ax = plt.subplots(figsize=FIG_HALF, layout="constrained")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def draw_band(ax, y_bot, height, color, label, items_text,
              label_color="white", items_color="white", alpha=0.92):
    rect = FancyBboxPatch((0.01, y_bot), 0.98, height,
                          boxstyle="round,pad=0.01",
                          linewidth=0, facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(rect)
    ax.text(0.05, y_bot + height - 0.035, label,
            va="top", ha="left", fontsize=8.5, fontweight="bold",
            color=label_color, zorder=3)
    if items_text:
        ax.text(0.06, y_bot + height - 0.075, items_text,
                va="top", ha="left", fontsize=7,
                color=items_color, zorder=3, linespacing=1.4)

draw_band(ax, 0.87, 0.10, C_ANCHOR, ANCHOR_LABEL, "",
          label_color="white", alpha=1.0)
draw_band(ax, 0.62, 0.23, C_LST,    LST_LABEL,    LST_ITEMS,
          label_color="white", items_color="#e8f5f4")
draw_band(ax, 0.35, 0.25, C_TILT,   TILT_LABEL,   TILT_ITEMS,
          label_color="#264653", items_color="#5c4a00")
draw_band(ax, 0.01, 0.32, C_RES,    RES_LABEL,    RES_ITEMS,
          label_color="white", items_color="#fce8e0")

# Brace annotation
ax.text(0.99, 0.43,
        "103\nphysics\nfeatures",
        va="center", ha="right", fontsize=8, fontweight="bold",
        color="#333333", zorder=4,
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="#cccccc", lw=0.8, alpha=0.9))
for yy in [0.62, 0.35, 0.01]:
    ax.plot([0.955, 0.965], [yy, yy], color="#aaaaaa", lw=0.8, zorder=4)
ax.plot([0.965, 0.965], [0.01, 0.62], color="#aaaaaa", lw=0.8, zorder=4)
ax.plot([0.965, 0.975], [0.315, 0.315], color="#aaaaaa", lw=0.8, zorder=4)

ax.set_title("Feature partition: 103 physics-informed features (30 LST + 30 Tilt + 43 Residual)",
             fontsize=9, fontweight="bold", loc="center", pad=6)

OUT = os.path.join(RELEASE_ROOT, "supplementary")
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/figS6_feature_partition.{ext}", dpi=300)
print("Saved figS6_feature_partition.pdf and .png")
plt.close(fig)
