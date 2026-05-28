#!/usr/bin/env python3
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json, os, re

RELEASE_ROOT = os.path.abspath(os.path.dirname(__file__))
while RELEASE_ROOT != os.path.dirname(RELEASE_ROOT) and not os.path.isdir(os.path.join(RELEASE_ROOT, 'model_weights')):
    RELEASE_ROOT = os.path.dirname(RELEASE_ROOT)
PIML    = RELEASE_ROOT
OUT     = os.path.join(RELEASE_ROOT, "figures_output", "supp")
EXTRACT = os.path.join(RELEASE_ROOT, "extracted_data")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": True,
    "legend.framealpha": 0.92, "legend.edgecolor": "#CCCCCC",
    "figure.dpi": 150, "savefig.dpi": 300,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": False,
})

def style(ax, lw=0.9):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
    ax.tick_params(direction="in", top=False, right=False,
                   bottom=True, left=True, width=0.8, length=3)
    ax.set_axisbelow(False)

BRANCH_COLORS = {"LST": "#2A9D8F", "Tilt": "#E9C46A", "Residual": "#E76F51", "Aux": "#264653"}
ASITE_COLORS  = {"Pb": "#7A6F5B", "Ba": "#5A8C3E", "Sr": "#D4A82E", "Ca": "#D67238", "La": "#4A6FA5"}

def savefig(fig, stem):
    for ext in ("pdf", "png"):
        p = f"{OUT}/{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("saved", p)
    plt.close(fig)


# ───────────────────────── figS11 — SHAP attribution ─────────────────────────
def fig_s11():
    part = json.load(open(f"{PIML}/data/processed/feature_partition_v7.json"))
    feat_names = (part["LST"] + part["Tilt"] + part["Residual"] +
                  ["er_CM", "has_cm", "cm_approx", "GII_norm", "phase_tr"])
    def branch(f):
        if f in part["LST"]:      return "LST"
        if f in part["Tilt"]:     return "Tilt"
        if f in part["Residual"]: return "Residual"
        return "Aux"
    shap = np.load(f"{PIML}/results/60_shap_values.npy", allow_pickle=False)
    assert shap.shape[1] == len(feat_names), (shap.shape, len(feat_names))

    fdf = pd.DataFrame({
        "feature": feat_names,
        "branch":  [branch(f) for f in feat_names],
        "mean_abs": np.abs(shap).mean(0),
        "mean_signed": shap.mean(0),
    }).sort_values("mean_abs", ascending=False).reset_index(drop=True)
    bdf = (fdf.groupby("branch")
              .agg(mean_abs=("mean_abs", "sum"), n=("feature", "count"))
              .reindex(["LST", "Residual", "Tilt", "Aux"]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 5.2),
                                 gridspec_kw={"width_ratios": [2.0, 1.0]})
    style(a1); style(a2)

    top = fdf.head(20)
    y = np.arange(len(top))[::-1]
    a1.barh(y, top["mean_abs"], height=0.72,
            color=[BRANCH_COLORS[b] for b in top["branch"]],
            edgecolor="white", linewidth=0.4, zorder=3)
    a1.set_yticks(y); a1.set_yticklabels(top["feature"], fontsize=6.3)
    for yi, v, s in zip(y, top["mean_abs"], top["mean_signed"]):
        a1.text(v + 0.08, yi, f"{v:.2f} {'↑' if s > 0 else '↓'}",
                va="center", fontsize=6.2, color="#222")
    a1.set_xlim(0, top["mean_abs"].max() * 1.22)
    a1.set_xlabel(r"Mean |SHAP|  (impact on $\varepsilon_r$)")
    a1.set_title("a   Top-20 features by SHAP importance", loc="left")
    a1.legend(handles=[mpatches.Patch(color=BRANCH_COLORS[b], label=b)
                       for b in ["LST", "Tilt", "Residual", "Aux"]],
              loc="lower right", title="Physics branch", title_fontsize=7,
              fontsize=6.5, handlelength=1.0)

    bp = np.arange(len(bdf))
    a2.barh(bp, bdf["mean_abs"], height=0.6,
            color=[BRANCH_COLORS[b] for b in bdf.index],
            edgecolor="white", linewidth=0.5, zorder=3)
    a2.set_yticks(bp); a2.set_yticklabels(bdf.index, fontsize=8, fontweight="bold")
    a2.invert_yaxis()
    for yi, v, n in zip(bp, bdf["mean_abs"], bdf["n"]):
        a2.text(v + bdf["mean_abs"].max() * 0.02, yi, f"{v:.1f}",
                va="center", fontsize=6.5, color="#222")
    a2.set_xlim(0, bdf["mean_abs"].max() * 1.25)
    a2.set_xlabel(r"Aggregated mean |SHAP|")
    a2.set_title("b   By physics branch", loc="left")

    fig.suptitle("PCNN feature attribution (SHAP)", fontsize=10, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, f"n = {shap.shape[0]} test compositions · GradientExplainer · "
             "arrows show sign of mean effect", ha="center", fontsize=6.0,
             style="italic", color="#555")
    fig.tight_layout()
    savefig(fig, "figS11_shap_importance")
    print("  S11 branch totals:", dict(bdf["mean_abs"].round(1)))


# ───────────────────────── figS13 — B-site LST routing ───────────────────────
def fig_s13():
    bs = json.load(open(f"{PIML}/results/43_counterfactual_permutation.json"))
    bs = bs["counterfactual"]["b_site_summaries"]
    rows = [(k, v["n_pairs"], v["lst_fraction"]) for k, v in bs.items() if v["n_pairs"] >= 10]
    rows.sort(key=lambda r: r[2])                       # ascending for barh
    labels = [r[0] for r in rows]; ns = [r[1] for r in rows]; fr = [r[2] * 100 for r in rows]
    allf = np.array([v["lst_fraction"] for v in bs.values()])
    mean_pct = allf.mean() * 100

    fig, ax = plt.subplots(figsize=(6.4, 5.2)); style(ax)
    y = np.arange(len(labels))
    cols = ["#C0392B" if l == "Y" else "#2A9D8F" for l in labels]
    ax.barh(y, fr, height=0.7, color=cols, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(mean_pct, color="#333", lw=1.1, ls="--", zorder=4,
               label=f"mean = {mean_pct:.1f}%")
    ax.set_yticks(y); ax.set_yticklabels([f"{l}  ($n$={n})" for l, n in zip(labels, ns)],
                                         fontsize=6.5)
    for yi, v in zip(y, fr):
        ax.text(v - 1.0, yi, f"{v:.1f}", va="center", ha="right",
                fontsize=6.0, color="white", fontweight="bold")
    ax.set_xlim(75, 101)
    ax.set_xlabel(r"Fraction of $\Delta\varepsilon_r$ routed through the LST branch (%)")
    ax.set_title("Counterfactual permutation: B-site-resolved LST routing", loc="left")
    ax.annotate(f"minimum: Y = {fr[0]:.1f}%", xy=(fr[0], 0), xytext=(fr[0] + 6, 1.4),
                fontsize=6.5, color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.9))
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    savefig(fig, "figS13_counterfactual_heatmap")
    print(f"  S13 mean={mean_pct:.1f}%  min=Y {fr[0]:.1f}%  n_bsites={len(allf)}")


# ───────────────────── shared: per-composition f_LST & A-site ─────────────────
def _load_decomp():
    df = pd.read_csv(f"{EXTRACT}/decomposition_per_sample.csv")
    df = df[df["er_cm"] > 0].copy()
    def asite(f):
        m = re.match(r"([A-Z][a-z]?)", str(f))
        e = m.group(1) if m else "Other"
        return e if e in ("Pb", "Ca", "La", "Sr", "Ba") else "Other"
    df["A"] = df["formula"].apply(asite)
    df["flst"] = df["delta_lst"] / df["er_cm"]
    return df


# ───────────────────────── figS14 — permutation null ─────────────────────────
def fig_s14(n_perm=2000, seed=0):
    df = _load_decomp()
    rng = np.random.default_rng(seed)

    def null_range(sub, label_col, groups):
        lab  = sub[label_col].values
        flst = sub["flst"].values
        obs_means = {g: flst[lab == g].mean() for g in groups}
        obs_range = max(obs_means.values()) - min(obs_means.values())
        ranges = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(lab)
            m = [flst[perm == g].mean() for g in groups]
            ranges[i] = max(m) - min(m)
        p = np.sum(ranges >= obs_range) / n_perm
        z = (obs_range - ranges.mean()) / ranges.std()
        return obs_range, ranges, p, z, obs_means

    a_groups = ["Pb", "Ca", "La", "Sr", "Ba"]
    oa, ra, pa, za, ma = null_range(df[df["A"].isin(a_groups)], "A", a_groups)

    r_groups = sorted(df["regime"].dropna().unique().tolist())
    orr, rr, pr, zr, mr = null_range(df[df["regime"].isin(r_groups)], "regime", r_groups)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4)); style(a1); style(a2)
    for ax, (obs, null, p, z, ttl, xl) in zip(
        (a1, a2),
        [(oa, ra, pa, za, "a   A-site label-permutation null", "Range of family-mean $f_{LST}$"),
         (orr, rr, pr, zr, "b   Regime label-permutation null", "Range of regime-mean $f_{LST}$")]):
        ax.hist(null, bins=40, color="#9FB8C8", edgecolor="white", linewidth=0.3, zorder=2)
        ax.axvline(obs, color="#C0392B", lw=1.6, zorder=4)
        ax.set_xlim(0, obs * 1.08)
        ptxt = "p < 0.0005" if p == 0 else f"p = {p:.3f}"
        ax.text(0.50, 0.92, f"observed = {obs:.3f}\n{ptxt}\nz ≈ {z:.0f}$\\sigma$",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCC", alpha=0.92))
        ax.annotate("observed", xy=(obs, ax.get_ylim()[1] * 0.5),
                    xytext=(obs * 0.86, ax.get_ylim()[1] * 0.62), fontsize=6.3,
                    color="#C0392B", ha="right",
                    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8))
        ax.set_title(ttl, loc="left"); ax.set_xlabel(xl); ax.set_ylabel("Count")
    fig.suptitle("Label-permutation null distributions (2,000 randomisations)",
                 fontsize=9.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "figS14_permutation_null")
    print(f"  S14 A-site: obs={oa:.3f} p={pa:.4f} z={za:.1f}  means={ {k:round(v,3) for k,v in ma.items()} }")
    print(f"  S14 regime: obs={orr:.3f} p={pr:.4f} z={zr:.1f}")


# ───────────────────────── figS22 — per-A-site decomposition ──────────────────
def fig_s22():
    df = _load_decomp()
    order = ["Pb", "Ca", "La", "Sr", "Ba"]
    g = (df[df["A"].isin(order)].groupby("A")
         .agg(n=("idx", "size"), er_cm=("er_cm", "mean"),
              dlst=("delta_lst", "mean"), dtilt=("delta_tilt", "mean"),
              dres=("delta_res", "mean"), flst=("flst", "mean"))
         .reindex(order))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                 gridspec_kw={"width_ratios": [1.5, 1.0]})
    style(a1); style(a2)
    x = np.arange(len(order))

    # Panel a: stacked decomposition CM + dLST (+ dtilt, dres)
    a1.bar(x, g["er_cm"], color="#264653", edgecolor="white", label=r"$\varepsilon_r^{CM}$")
    a1.bar(x, g["dlst"], bottom=g["er_cm"], color="#2A9D8F", edgecolor="white",
           label=r"$\delta_{LST}$")
    a1.bar(x, g["dtilt"], bottom=g["er_cm"] + g["dlst"], color="#E9C46A",
           edgecolor="white", label=r"$\delta_{tilt}$")
    a1.bar(x, g["dres"], bottom=g["er_cm"] + g["dlst"] + g["dtilt"], color="#E76F51",
           edgecolor="white", label=r"$\delta_{res}$")
    tot = g["er_cm"] + g["dlst"] + g["dtilt"] + g["dres"]
    for xi, t in zip(x, tot):
        a1.text(xi, t + 1.5, f"{t:.0f}", ha="center", fontsize=6.5, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels([f"{a}\n($n$={int(g.loc[a,'n'])})" for a in order])
    a1.set_ylabel(r"Mean contribution to $\varepsilon_r$")
    a1.set_title("a   Per-A-site decomposition", loc="left")
    a1.legend(loc="upper right", fontsize=6.3, ncol=2, handlelength=1.0)
    a1.set_ylim(0, tot.max() * 1.18)

    # Panel b: f_LST hierarchy
    a2.bar(x, g["flst"], color=[ASITE_COLORS[a] for a in order],
           edgecolor="white", zorder=3)
    for xi, v in zip(x, g["flst"]):
        a2.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.8, fontweight="bold")
    a2.set_xticks(x); a2.set_xticklabels(order)
    a2.set_ylabel(r"$f_{LST} = \langle\delta_{LST}/\varepsilon_r^{CM}\rangle$")
    a2.set_title("b   Soft-mode enhancement", loc="left")
    a2.set_ylim(0, g["flst"].max() * 1.20)

    fig.suptitle("A-site-resolved physics decomposition", fontsize=9.5,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "figS22_element_attribution")
    print("  S22 f_LST:", dict(g["flst"].round(3)))


if __name__ == "__main__":
    fig_s11()
    fig_s13()
    fig_s14()
    fig_s22()
    print("done.")
