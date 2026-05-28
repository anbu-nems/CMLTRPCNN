#!/usr/bin/env python3
"""
Crystal-structure schematic — the three physics mechanisms encoded in PCNN.

Three panels, each a 2D projection of the ABO₃ perovskite unit cell:
  (a) Ideal cubic perovskite        ← εᵣ_CM baseline (CM polarizability)
  (b) Soft-mode B-site displacement ← δ_LST (Lyddane-Sachs-Teller)
  (c) Octahedral tilt about [001]   ← δ_tilt (Goldschmidt/Reaney)

Output: all_figures/supp/figS20_crystal_structure.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, RegularPolygon
from matplotlib.lines import Line2D
import numpy as np
import os

OUT_DIR = "./figures_output/supp"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 300,
})

# Element colours (Shannon / VESTA-style)
C_A = "#5A8C3E"       # green — A-site (Ba-like)
C_B = "#4A6FA5"       # blue — B-site (Ti-like)
C_O = "#D67238"       # orange — oxygen
C_OCT = "#A8C2DC"     # pale blue — BO₆ polyhedron fill
C_OCT_TILT = "#E9C46A"   # tilted octahedron tinge
C_BOND = "#555555"

# ─────────────────────────────────────────────────────────────────────────────
def draw_cell_outline(ax, x0, y0, L, color="#222", lw=1.0, ls="-"):
    """Square unit cell outline (perovskite pseudocubic projection)."""
    ax.plot([x0, x0+L, x0+L, x0, x0],
            [y0, y0, y0+L, y0+L, y0], color=color, lw=lw, ls=ls, zorder=1)

def draw_atom(ax, x, y, color, r=0.08, label=None, label_offset=(0.0, 0.0),
              fs=8, lw=0.8, zorder=4):
    circ = Circle((x, y), r, facecolor=color, edgecolor="white",
                  linewidth=lw, zorder=zorder)
    ax.add_patch(circ)
    if label:
        ax.text(x + label_offset[0], y + label_offset[1], label,
                ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", zorder=zorder+1)

def draw_octahedron(ax, cx, cy, half_d=0.40, fill=C_OCT, edge=C_BOND,
                    lw=1.1, alpha=0.55, rotate_deg=0.0, zorder=2):
    """2D projection of BO₆ octahedron — diamond (rotated square)."""
    th = np.deg2rad(rotate_deg + 45)
    pts = []
    for k in range(4):
        a = th + k * np.pi/2
        pts.append((cx + half_d * np.cos(a), cy + half_d * np.sin(a)))
    poly = Polygon(pts, closed=True, facecolor=fill, edgecolor=edge,
                   linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(poly)
    return pts   # corners are the 4 in-plane O atoms

def draw_arrow(ax, x1, y1, x2, y2, color="#222", lw=1.6, alpha=1.0,
               mutation_scale=14, zorder=6):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=mutation_scale,
                        color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(a)

def panel_setup(ax, title, subtitle):
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.20)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, 1.16, title, ha="center", va="top",
            fontsize=10, color="#222", fontweight="bold")
    ax.text(0.5, 1.05, subtitle, ha="center", va="top",
            fontsize=7.5, color="#444", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4))

# ── PANEL A: Ideal cubic perovskite ─────────────────────────────────────────
ax = axes[0]
L = 0.8
x0, y0 = 0.1, 0.1
draw_cell_outline(ax, x0, y0, L, color="#888", lw=0.7, ls=":")
cx, cy = x0 + L/2, y0 + L/2

# Equatorial O atoms (at mid-faces of cube projection)
draw_octahedron(ax, cx, cy, half_d=L*0.5, rotate_deg=0)
# Apical O atoms (projected onto the centre — shown at top/bottom of cell midline)
draw_atom(ax, cx, y0 + L,   C_O, r=0.045, zorder=5)   # top apical
draw_atom(ax, cx, y0,        C_O, r=0.045, zorder=5)   # bottom apical
# Equatorial O atoms at octahedron corners
for k in range(4):
    a = np.deg2rad(45 + k*90)
    ax_x = cx + L*0.5*np.cos(a); ax_y = cy + L*0.5*np.sin(a)
    draw_atom(ax, ax_x, ax_y, C_O, r=0.045, zorder=5)

# B at centre
draw_atom(ax, cx, cy, C_B, r=0.06, zorder=6)
# A at corners
for (cx_a, cy_a) in [(x0, y0), (x0+L, y0), (x0, y0+L), (x0+L, y0+L)]:
    draw_atom(ax, cx_a, cy_a, C_A, r=0.07, zorder=5)

# B–O bonds (just two visible apical bonds for clarity)
ax.plot([cx, cx], [cy, y0+L], color=C_BOND, lw=0.8, zorder=3)
ax.plot([cx, cx], [cy, y0],    color=C_BOND, lw=0.8, zorder=3)

panel_setup(ax, "(a) Ideal cubic ABO$_3$",
            r"$\varepsilon_r^{CM}$  Clausius-Mossotti baseline")
# Mechanism label
ax.text(0.5, -0.02, "Polarizability of static lattice",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=7, color=C_B, fontweight="bold")

# ── PANEL B: Soft-mode B-site displacement ──────────────────────────────────
ax = axes[1]
draw_cell_outline(ax, x0, y0, L, color="#888", lw=0.7, ls=":")
draw_octahedron(ax, cx, cy, half_d=L*0.5, rotate_deg=0)
# Apical O atoms
draw_atom(ax, cx, y0 + L,   C_O, r=0.045, zorder=5)
draw_atom(ax, cx, y0,        C_O, r=0.045, zorder=5)
for k in range(4):
    a = np.deg2rad(45 + k*90)
    ax_x = cx + L*0.5*np.cos(a); ax_y = cy + L*0.5*np.sin(a)
    draw_atom(ax, ax_x, ax_y, C_O, r=0.045, zorder=5)
# A corners
for (cx_a, cy_a) in [(x0, y0), (x0+L, y0), (x0, y0+L), (x0+L, y0+L)]:
    draw_atom(ax, cx_a, cy_a, C_A, r=0.07, zorder=5)

# B-site DISPLACED downward (toward bottom apical O)
B_disp = 0.10
draw_atom(ax, cx, cy - B_disp, C_B, r=0.06, zorder=6)
# Ghost position (original)
ax.add_patch(Circle((cx, cy), 0.06, facecolor="none",
                    edgecolor=C_B, lw=0.8, ls=(0,(3,2)), alpha=0.55, zorder=4))
# Off-centering arrow
draw_arrow(ax, cx, cy + 0.02, cx, cy - B_disp + 0.04,
           color="#C0392B", lw=1.6, mutation_scale=12)
# Soft-mode label
ax.text(cx + 0.10, cy - B_disp/2, r"$\omega_{TO}\!\to\!0$",
        fontsize=8, color="#C0392B", fontweight="bold", va="center")

# B–O bonds: short upward (compressed), long downward (stretched)
ax.plot([cx, cx], [cy - B_disp, y0+L], color=C_BOND, lw=0.8, zorder=3)
ax.plot([cx, cx], [cy - B_disp, y0],    color=C_BOND, lw=1.6, zorder=3)

panel_setup(ax, "(b) Soft-mode displacement",
            r"$\delta_{LST}$  Lyddane-Sachs-Teller enhancement")
ax.text(0.5, -0.02, r"$\varepsilon_r \propto 1/\omega_{TO}^2$  ·  B-site off-centering",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=7, color="#C0392B", fontweight="bold")

# ── PANEL C: Octahedral tilt (rotation about [001]) ─────────────────────────
ax = axes[2]
draw_cell_outline(ax, x0, y0, L, color="#888", lw=0.7, ls=":")
TILT_DEG = 15
# Tilted octahedron (rotated)
pts_tilt = draw_octahedron(ax, cx, cy, half_d=L*0.5,
                            rotate_deg=TILT_DEG,
                            fill=C_OCT_TILT, alpha=0.50)
# Ghost untilted octahedron for reference
draw_octahedron(ax, cx, cy, half_d=L*0.5, rotate_deg=0,
                fill="none", edge="#888", alpha=0.65, lw=0.6)

# Equatorial O at tilted corners
for (px, py) in pts_tilt:
    draw_atom(ax, px, py, C_O, r=0.045, zorder=5)
# Apical O (still on z-axis in projection)
draw_atom(ax, cx, y0 + L,   C_O, r=0.045, zorder=5)
draw_atom(ax, cx, y0,        C_O, r=0.045, zorder=5)
# A corners
for (cx_a, cy_a) in [(x0, y0), (x0+L, y0), (x0, y0+L), (x0+L, y0+L)]:
    draw_atom(ax, cx_a, cy_a, C_A, r=0.07, zorder=5)
# B centre (unchanged)
draw_atom(ax, cx, cy, C_B, r=0.06, zorder=6)
# Curved rotation arrow
theta0 = np.deg2rad(35)
theta1 = np.deg2rad(15)
r_arrow = 0.30
arc_pts = np.array([[cx + r_arrow*np.cos(t), cy + r_arrow*np.sin(t)]
                    for t in np.linspace(theta0, theta1, 30)])
ax.plot(arc_pts[:,0], arc_pts[:,1], color="#8A6E16", lw=1.3, zorder=6)
# Arrowhead
draw_arrow(ax, arc_pts[-2,0], arc_pts[-2,1], arc_pts[-1,0], arc_pts[-1,1],
           color="#8A6E16", lw=1.3, mutation_scale=10)
ax.text(cx + 0.40, cy + 0.18, "tilt", fontsize=8.5,
        color="#8A6E16", fontweight="bold")

panel_setup(ax, "(c) Octahedral tilt",
            r"$\delta_{tilt}$  Goldschmidt/Reaney suppression")
ax.text(0.5, -0.02, r"$t = (r_A+r_O)/[\sqrt{2}\,(r_B+r_O)]<1$  ·  reduces $\varepsilon_r$",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=7, color="#8A6E16", fontweight="bold")

# Shared legend at top
legend_items = [
    Line2D([0],[0], marker="o", linestyle="",
           markerfacecolor=C_A, markeredgecolor="white", markeredgewidth=0.6,
           markersize=10, label="A-site (e.g. Ba, Sr, Ca, La, Pb)"),
    Line2D([0],[0], marker="o", linestyle="",
           markerfacecolor=C_B, markeredgecolor="white", markeredgewidth=0.6,
           markersize=8,  label="B-site (e.g. Ti, Zr, Nb, Mg)"),
    Line2D([0],[0], marker="o", linestyle="",
           markerfacecolor=C_O, markeredgecolor="white", markeredgewidth=0.6,
           markersize=6,  label="Oxygen"),
]
fig.legend(handles=legend_items, loc="upper center",
           bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=7,
           frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

fig.suptitle("ABO$_3$ perovskite — three physics mechanisms encoded in PCNN",
             fontsize=10, fontweight="bold", y=1.02)

# Save
for ext in ("pdf", "png"):
    path = f"{OUT_DIR}/figS20_crystal_structure.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
plt.close(fig)
