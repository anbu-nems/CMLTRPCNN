"""
Script 49 — Dual Applicability Domain for CMLTRPCNNv77
=======================================================
Layer 1 (physics-mechanism gate): sigma_conf from model
  - Derived from cm_approx_flag, GII, phase_transition
  - High sigma_conf → ML compensating for unreliable physics (VCA, strained, phase boundary)
  - Low sigma_conf  → physics-driven, reliable constraints active

Layer 2 (compositional OOD): PCA + KDE density
  - Catches rare families statistically far from training cloud
  - Flags what the physics gate misses (e.g. Ba_Eu, 5 samples — normal physics, extreme OOD)

Combined in-domain:
  in_ad = kde_in_domain AND physics_reliable
  where physics_reliable = sigma_conf < threshold (physics driving, not ML compensating)

Key validation:
  - R² in-AD vs out-AD on calibration holdout
  - Pb: high KDE (86 samples) but physics-compensated → correctly flagged by Layer 1
  - La/II shortlist: high KDE + physics-reliable → safe design target
"""

import sys, os, json, warnings
import numpy as np, torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model_CMLTRPCNNv77 import CMLTRPCNNv71

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
RES_DIR  = os.path.join(ROOT, 'results')
FIG_DIR  = os.path.join(ROOT, 'figures')
MDL_DIR  = os.path.join(ROOT, 'models')

# ── Thresholds ────────────────────────────────────────────────────────────────
SIGMA_CONF_THRESHOLD = 0.35   # above this → ML compensating → physics unreliable
KDE_PERCENTILE       = 5      # below 5th percentile of training density → OOD
PCA_COMPONENTS       = 10

# ── Data loading ──────────────────────────────────────────────────────────────
print("=" * 68)
print("Script 49 — Dual AD for CMLTRPCNNv77")
print("=" * 68)

df = pd.read_parquet(os.path.join(PROC_DIR, 'feature_matrix_v7.parquet'))
with open(os.path.join(PROC_DIR, 'feature_partition_v7.json')) as f:
    partition = json.load(f)
with open(os.path.join(PROC_DIR, 'calibration_split_idx.json')) as f:
    calib_info = json.load(f)

def _get(cols):
    present = [c for c in cols if c in df.columns]
    return df[present].fillna(0.0).values.astype(np.float32), present

Xl, lcols = _get(partition['LST'])
Xt, _     = _get(partition['Tilt'])
Xr, _     = _get(partition['Residual'])
er_cm     = df['er_CM'].fillna(0.0).values.astype(np.float32)
has_cm    = df['has_sigma_CM'].fillna(0.0).values.astype(np.float32)
cm_approx = df['cm_approx_flag'].fillna(0.0).values.astype(np.float32)
gii       = df['GII'].fillna(0.0).values.astype(np.float32)
phase_tr  = df['phase_transition'].fillna(0.0).values.astype(np.float32) \
            if 'phase_transition' in df.columns else np.zeros(len(df), np.float32)
y         = df['epsilon_r'].values.astype(np.float32)
groups    = df['chemistry_family'].values
formula   = df['formula'].values if 'formula' in df.columns else np.array(['']*len(df))
# Derive A-site from chemistry_family (first token)
a_site = df['chemistry_family'].str.split('_').str[0].values
# Derive regime from one-hot columns
reg_cols = ['regime_Ia', 'regime_Ib', 'regime_II', 'regime_III']
reg_mat  = df[reg_cols].values
regime   = np.array(['unknown'] * len(df))
for i, col in enumerate(reg_cols):
    regime[reg_mat[:, i] > 0.5] = col.replace('regime_', '')

train_idx = np.array(calib_info['train_idx'])
calib_idx = np.array(calib_info['calib_idx'])

regime_names = ['regime_Ia', 'regime_Ib', 'regime_II', 'regime_III']
regime_idx   = [lcols.index(r) for r in regime_names if r in lcols]

# Scale
def scale(X, idx):
    s = StandardScaler(); s.fit(X[idx])
    return s.transform(X).astype(np.float32)

Xl_s = scale(Xl, train_idx)
Xt_s = scale(Xt, train_idx)
Xr_s = scale(Xr, train_idx)

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(os.path.join(MDL_DIR, 'cmltrv77_final.pt'),
                  map_location='cpu', weights_only=False)
hp = ckpt['hp']

def make_model():
    return CMLTRPCNNv71(
        n_lst=Xl.shape[1], n_tilt=Xt.shape[1], n_res=Xr.shape[1],
        trunk_hidden=hp['trunk_hidden'], n_trunk_blocks=hp['n_trunk_blocks'],
        lst_hidden=hp['lst_hidden'], tilt_hidden=hp['tilt_hidden'],
        res_hidden=hp['res_hidden'], residual_scale=hp['residual_scale'],
        dropout=0.0)

# ── Full inference on all samples ─────────────────────────────────────────────
print("\n[1/4] Extracting sigma_conf and predictions for all samples ...")
preds_all  = []
sigma_all  = []
flst_all   = []

for sd in ckpt['models']:
    m = make_model(); m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        out = m(torch.tensor(Xl_s), torch.tensor(Xt_s), torch.tensor(Xr_s),
                torch.tensor(er_cm), torch.tensor(has_cm), torch.tensor(cm_approx),
                torch.tensor(gii), torch.tensor(phase_tr), regime_idx)
    preds_all.append(out['pred'].numpy())
    sigma_all.append(out['sigma_conf'].numpy())
    if 'delta_lst' in out:
        delta_lst = out['delta_lst'].numpy()
        er_cm_np  = er_cm.copy()
        f_lst     = np.where(er_cm_np > 0, delta_lst / np.maximum(er_cm_np, 1e-6), 0.0)
        flst_all.append(f_lst)

pred      = np.mean(preds_all, axis=0)
sigma_conf = np.mean(sigma_all, axis=0)
f_lst_arr  = np.mean(flst_all, axis=0) if flst_all else np.zeros(len(y))

print(f"  sigma_conf: mean={sigma_conf.mean():.3f}  min={sigma_conf.min():.3f}  max={sigma_conf.max():.3f}")
print(f"  VCA samples (cm_approx=1): n={cm_approx.sum():.0f}")
print(f"  Mean sigma_conf VCA:     {sigma_conf[cm_approx > 0.5].mean():.3f}")
print(f"  Mean sigma_conf exact CM:{sigma_conf[cm_approx < 0.5].mean():.3f}")

# ── Layer 1: Physics-mechanism gate ──────────────────────────────────────────
print("\n[2/4] Layer 1 — Physics-mechanism gate ...")
# Low sigma_conf = physics-driven (reliable), High = ML compensating (uncertain)
physics_reliable = sigma_conf < SIGMA_CONF_THRESHOLD
print(f"  Physics-reliable (sigma_conf < {SIGMA_CONF_THRESHOLD}): {physics_reliable.sum()} / {len(y)} ({100*physics_reliable.mean():.1f}%)")
print(f"  Physics-compensated (ML taking over):                  {(~physics_reliable).sum()} ({100*(~physics_reliable).mean():.1f}%)")

# Per A-site sigma_conf
print("\n  Mean sigma_conf by A-site:")
for a in ['Pb', 'Ba', 'Ca', 'Sr', 'La']:
    mask = np.array([str(x) == a for x in a_site])
    if mask.sum():
        print(f"    {a:4s}: sigma_conf={sigma_conf[mask].mean():.3f}  n={mask.sum()}")

# ── Layer 2: KDE compositional OOD ───────────────────────────────────────────
print("\n[3/4] Layer 2 — PCA + KDE compositional density ...")
X_all = np.hstack([Xl_s, Xt_s, Xr_s])

pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_all)
var_retained = pca.explained_variance_ratio_.sum()
print(f"  PCA: {PCA_COMPONENTS} components, {100*var_retained:.1f}% variance retained")

# Fit KDE on training set only
kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
kde.fit(X_pca[train_idx])

log_density = kde.score_samples(X_pca)
train_log_density = log_density[train_idx]
threshold_density = np.percentile(train_log_density, KDE_PERCENTILE)
kde_in_domain = log_density > threshold_density

print(f"  KDE threshold (p{KDE_PERCENTILE}): {threshold_density:.3f}")
print(f"  Training in-domain:    {kde_in_domain[train_idx].mean()*100:.1f}%")
print(f"  Calibration in-domain: {kde_in_domain[calib_idx].mean()*100:.1f}%")

# ── Combined AD ───────────────────────────────────────────────────────────────
in_ad = kde_in_domain & physics_reliable

# ── Validation on calibration set ────────────────────────────────────────────
print("\n[4/4] AD validation on calibration holdout ...")

yt_calib = y[calib_idx]
yp_calib = pred[calib_idx]

calib_in_ad  = in_ad[calib_idx]
calib_out_ad = ~calib_in_ad

r2_all   = r2_score(yt_calib, yp_calib)
mae_all  = mean_absolute_error(yt_calib, yp_calib)

if calib_in_ad.sum() > 5:
    r2_in  = r2_score(yt_calib[calib_in_ad],  yp_calib[calib_in_ad])
    mae_in = mean_absolute_error(yt_calib[calib_in_ad], yp_calib[calib_in_ad])
else:
    r2_in = mae_in = float('nan')

if calib_out_ad.sum() > 5:
    r2_out  = r2_score(yt_calib[calib_out_ad],  yp_calib[calib_out_ad])
    mae_out = mean_absolute_error(yt_calib[calib_out_ad], yp_calib[calib_out_ad])
else:
    r2_out = mae_out = float('nan')

# Layer 1 only validation
layer1_in  = physics_reliable[calib_idx]
r2_phys_in  = r2_score(yt_calib[layer1_in],  yp_calib[layer1_in])  if layer1_in.sum()>5 else float('nan')
r2_phys_out = r2_score(yt_calib[~layer1_in], yp_calib[~layer1_in]) if (~layer1_in).sum()>5 else float('nan')

# Layer 2 only validation
layer2_in  = kde_in_domain[calib_idx]
r2_kde_in  = r2_score(yt_calib[layer2_in],  yp_calib[layer2_in])  if layer2_in.sum()>5 else float('nan')
r2_kde_out = r2_score(yt_calib[~layer2_in], yp_calib[~layer2_in]) if (~layer2_in).sum()>5 else float('nan')

print(f"\n  Calibration set (N={len(calib_idx)}):")
print(f"    Overall R²:           {r2_all:.4f}  MAE={mae_all:.2f}")
print(f"    Layer1 physics in-AD: R²={r2_phys_in:.4f}  (N={layer1_in.sum()})")
print(f"    Layer1 physics out-AD:R²={r2_phys_out:.4f}  (N={(~layer1_in).sum()})")
print(f"    Layer2 KDE in-AD:     R²={r2_kde_in:.4f}  (N={layer2_in.sum()})")
print(f"    Layer2 KDE out-AD:    R²={r2_kde_out:.4f}  (N={(~layer2_in).sum()})")
print(f"    Combined in-AD:       R²={r2_in:.4f}  MAE={mae_in:.2f}  (N={calib_in_ad.sum()})")
print(f"    Combined out-AD:      R²={r2_out:.4f}  MAE={mae_out:.2f}  (N={calib_out_ad.sum()})")

# ── A-site AD breakdown ───────────────────────────────────────────────────────
print("\n  AD coverage by A-site:")
asite_ad = {}
for a in ['Pb', 'Ba', 'Ca', 'Sr', 'La']:
    mask = np.array([str(x) == a for x in a_site])
    if mask.sum() == 0: continue
    pct_in  = 100 * in_ad[mask].mean()
    pct_l1  = 100 * physics_reliable[mask].mean()
    pct_l2  = 100 * kde_in_domain[mask].mean()
    sc_mean = sigma_conf[mask].mean()
    asite_ad[a] = {
        'n': int(mask.sum()),
        'pct_combined_in_ad': round(pct_in, 1),
        'pct_layer1_physics': round(pct_l1, 1),
        'pct_layer2_kde':     round(pct_l2, 1),
        'mean_sigma_conf':    round(float(sc_mean), 3)
    }
    print(f"    {a:4s} (n={mask.sum():3d}): combined={pct_in:5.1f}%  L1-physics={pct_l1:5.1f}%  L2-KDE={pct_l2:5.1f}%  σ_conf={sc_mean:.3f}")

# ── Regime AD breakdown ───────────────────────────────────────────────────────
print("\n  AD coverage by Reaney regime:")
regime_ad = {}
for rk in ['Ia', 'Ib', 'II', 'III']:
    mask = np.array([str(r).replace('Regime ', '').replace('regime_', '') == rk for r in regime])
    if mask.sum() == 0: continue
    pct_in = 100 * in_ad[mask].mean()
    pct_l1 = 100 * physics_reliable[mask].mean()
    pct_l2 = 100 * kde_in_domain[mask].mean()
    regime_ad[rk] = {
        'n': int(mask.sum()),
        'pct_combined_in_ad': round(pct_in, 1),
        'pct_layer1_physics': round(pct_l1, 1),
        'pct_layer2_kde':     round(pct_l2, 1),
    }
    print(f"    Regime {rk:3s} (n={mask.sum():3d}): combined={pct_in:5.1f}%  L1-physics={pct_l1:5.1f}%  L2-KDE={pct_l2:5.1f}%")

# ── La/II shortlist AD check ──────────────────────────────────────────────────
print("\n  La/Regime II shortlist AD check:")
try:
    with open(os.path.join(RES_DIR, '46_application_relevance.json')) as f:
        app = json.load(f)
    shortlist = app.get('claim_2_design_target', {}).get('synthesis_shortlist', [])
    shortlist_formulas = [s['formula'] for s in shortlist]
    shortlist_mask = np.array([f in shortlist_formulas for f in formula])
    if shortlist_mask.sum() > 0:
        pct_sl_in   = 100 * in_ad[shortlist_mask].mean()
        pct_sl_phys = 100 * physics_reliable[shortlist_mask].mean()
        pct_sl_kde  = 100 * kde_in_domain[shortlist_mask].mean()
        print(f"    {shortlist_mask.sum()} shortlist found: combined={pct_sl_in:.1f}%  physics={pct_sl_phys:.1f}%  KDE={pct_sl_kde:.1f}%")
    else:
        pct_sl_in = pct_sl_phys = pct_sl_kde = float('nan')
        print("    Shortlist formulas not found in current dataset index")
except Exception as e:
    pct_sl_in = pct_sl_phys = pct_sl_kde = float('nan')
    print(f"    Could not load shortlist: {e}")

# ── Figures ───────────────────────────────────────────────────────────────────
print("\nGenerating figures ...")

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ASITE_COLOR = {'Pb':'#E76F51','Ba':'#2A9D8F','Ca':'#E9C46A','Sr':'#264653','La':'#A8DADC','Other':'#aaaaaa'}

# Panel A — PCA space coloured by KDE density
ax_a = fig.add_subplot(gs[0, 0])
sc = ax_a.scatter(X_pca[:, 0], X_pca[:, 1],
                  c=log_density, cmap='RdYlGn', s=6, alpha=0.5, linewidths=0)
plt.colorbar(sc, ax=ax_a, label='log density', shrink=0.8)
ax_a.axhline(0, color='gray', lw=0.4, ls='--', alpha=0.4)
ax_a.axvline(0, color='gray', lw=0.4, ls='--', alpha=0.4)
ax_a.set_xlabel('PC1'); ax_a.set_ylabel('PC2')
ax_a.set_title('Layer 2: KDE Density (PCA space)')
# Mark OOD
ood_mask = ~kde_in_domain
ax_a.scatter(X_pca[ood_mask, 0], X_pca[ood_mask, 1],
             s=20, facecolors='none', edgecolors='red', lw=0.8, alpha=0.7, label='OOD')
ax_a.legend(loc='upper right', fontsize=7)

# Panel B — sigma_conf distribution by A-site
ax_b = fig.add_subplot(gs[0, 1])
for a in ['Pb', 'Ba', 'Ca', 'Sr', 'La']:
    mask = np.array([str(x) == a for x in a_site])
    if mask.sum():
        ax_b.hist(sigma_conf[mask], bins=30, alpha=0.5,
                  color=ASITE_COLOR.get(a, '#888'), label=f'{a} (n={mask.sum()})', density=True)
ax_b.axvline(SIGMA_CONF_THRESHOLD, color='red', lw=1.5, ls='--', label=f'threshold={SIGMA_CONF_THRESHOLD}')
ax_b.set_xlabel('sigma_conf (physics gate)')
ax_b.set_ylabel('Density')
ax_b.set_title('Layer 1: Physics Gate by A-site')
ax_b.legend(fontsize=6.5)

# Panel C — Dual AD status per A-site (stacked bar)
ax_c = fig.add_subplot(gs[0, 2])
asites = [a for a in ['Pb', 'Ba', 'Ca', 'Sr', 'La'] if asite_ad.get(a)]
x = np.arange(len(asites))
pct_combined = [asite_ad[a]['pct_combined_in_ad'] for a in asites]
pct_l1_only  = [max(0, asite_ad[a]['pct_layer1_physics'] - asite_ad[a]['pct_combined_in_ad']) for a in asites]
pct_l2_only  = [max(0, asite_ad[a]['pct_layer2_kde'] - asite_ad[a]['pct_combined_in_ad']) for a in asites]
pct_out      = [100 - asite_ad[a]['pct_combined_in_ad'] for a in asites]
ax_c.bar(x, pct_combined, 0.5, label='Both layers in-AD', color='#2A9D8F', alpha=0.9)
ax_c.bar(x, pct_out, 0.5, bottom=pct_combined, label='Out of AD', color='#E76F51', alpha=0.7)
ax_c.set_xticks(x); ax_c.set_xticklabels(asites)
ax_c.set_ylabel('% samples'); ax_c.set_ylim(0, 110)
ax_c.set_title('Dual AD Coverage by A-site')
ax_c.legend(fontsize=7)
for i, (pc, po) in enumerate(zip(pct_combined, pct_out)):
    ax_c.text(i, pc/2, f'{pc:.0f}%', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')

# Panel D — Parity: in-AD vs out-AD (calibration)
ax_d = fig.add_subplot(gs[1, 0])
lim = (0, 148)
ax_d.plot(lim, lim, 'k--', lw=0.8, alpha=0.5)
if calib_in_ad.sum():
    ax_d.scatter(yt_calib[calib_in_ad], yp_calib[calib_in_ad],
                 s=10, alpha=0.6, color='#2A9D8F', label=f'In-AD  R²={r2_in:.3f}', linewidths=0)
if calib_out_ad.sum():
    ax_d.scatter(yt_calib[calib_out_ad], yp_calib[calib_out_ad],
                 s=14, alpha=0.7, color='#E76F51', marker='^', label=f'Out-AD R²={r2_out:.3f}', linewidths=0)
ax_d.set_xlim(lim); ax_d.set_ylim(lim)
ax_d.set_xlabel('Measured er'); ax_d.set_ylabel('Predicted er')
ax_d.set_title('AD Validation (Calibration Set)')
ax_d.legend(fontsize=7.5)

# Panel E — R² comparison bars
ax_e = fig.add_subplot(gs[1, 1])
labels = ['Overall', 'L1 Physics\nIn-AD', 'L1 Physics\nOut-AD', 'L2 KDE\nIn-AD', 'L2 KDE\nOut-AD', 'Combined\nIn-AD', 'Combined\nOut-AD']
values = [r2_all, r2_phys_in, r2_phys_out, r2_kde_in, r2_kde_out, r2_in, r2_out]
colors = ['#888888','#2A9D8F','#E76F51','#2A9D8F','#E76F51','#264653','#E9C46A']
valid  = [(l, v, c) for l, v, c in zip(labels, values, colors) if not np.isnan(v)]
lv, vv, cv = zip(*valid)
bars = ax_e.bar(range(len(vv)), vv, color=cv, alpha=0.85, width=0.55)
ax_e.set_xticks(range(len(lv))); ax_e.set_xticklabels(lv, fontsize=6.5)
ax_e.set_ylim(min(0, min(vv)) - 0.1, 1.0)
ax_e.set_ylabel('R2')
ax_e.set_title('R2 In-AD vs Out-AD')
ax_e.axhline(0, color='black', lw=0.7)
for bar, v in zip(bars, vv):
    ax_e.text(bar.get_x()+bar.get_width()/2,
              bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.05,
              f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)

# Panel F — sigma_conf vs f_LST scatter (mechanism reliability map)
ax_f = fig.add_subplot(gs[1, 2])
for a in ['Pb', 'Ba', 'Ca', 'Sr', 'La']:
    mask = np.array([str(x) == a for x in a_site])
    if mask.sum():
        ax_f.scatter(f_lst_arr[mask], sigma_conf[mask], s=8, alpha=0.45,
                     color=ASITE_COLOR.get(a, '#888'), label=a, linewidths=0)
ax_f.axhline(SIGMA_CONF_THRESHOLD, color='red', lw=1.2, ls='--', alpha=0.8, label=f'gate threshold')
ax_f.set_xlabel('f_LST (soft-mode factor)')
ax_f.set_ylabel('sigma_conf (physics gate)')
ax_f.set_title('Mechanism Reliability Map')
ax_f.legend(fontsize=6.5, markerscale=1.5)
ax_f.text(0.02, 0.97, 'Physics-driven\n(reliable)', transform=ax_f.transAxes,
          fontsize=7, va='top', color='#2A9D8F')
ax_f.text(0.02, 0.55, 'ML-compensated\n(uncertain)', transform=ax_f.transAxes,
          fontsize=7, va='top', color='#E76F51')

fig.suptitle('CMLTRPCNNv77 — Dual Applicability Domain', fontsize=11, fontweight='bold', y=1.01)
fig_path = os.path.join(FIG_DIR, '49_ad_v77.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  Figure → {fig_path}")

# ── Save results ──────────────────────────────────────────────────────────────
out = {
    "model": "CMLTRPCNNv77",
    "ad_concept": "Dual: Layer1=physics-mechanism gate (sigma_conf), Layer2=PCA+KDE density",
    "novelty": "Primary reliability from physics (CM quality, GII, phase proximity); KDE hardens against compositional OOD",
    "layer1_physics_gate": {
        "signal": "sigma_conf = sigmoid(beta_0 + beta_1*cm_approx + beta_2*gii + beta_3*phase_tr)",
        "beta": [-2.0, 3.0, 2.0, 2.0],
        "threshold": SIGMA_CONF_THRESHOLD,
        "n_physics_reliable": int(physics_reliable.sum()),
        "pct_physics_reliable": round(100*physics_reliable.mean(), 1),
        "mean_sigma_vca": round(float(sigma_conf[cm_approx > 0.5].mean()), 3) if (cm_approx > 0.5).sum() > 0 else None,
        "mean_sigma_exact_cm": round(float(sigma_conf[cm_approx < 0.5].mean()), 3),
    },
    "layer2_kde": {
        "pca_components": PCA_COMPONENTS,
        "pca_variance": round(float(var_retained), 4),
        "kde_bandwidth": 1.0,
        "threshold_percentile": KDE_PERCENTILE,
        "threshold_log_density": round(float(threshold_density), 4),
        "pct_train_in_domain": round(float(kde_in_domain[train_idx].mean()*100), 1),
        "pct_calib_in_domain": round(float(kde_in_domain[calib_idx].mean()*100), 1),
    },
    "combined_ad": {
        "pct_total_in_ad": round(float(in_ad.mean()*100), 1),
        "pct_train_in_ad": round(float(in_ad[train_idx].mean()*100), 1),
        "pct_calib_in_ad": round(float(in_ad[calib_idx].mean()*100), 1),
    },
    "validation_calibration": {
        "n_calib": len(calib_idx),
        "r2_overall": round(float(r2_all), 4),
        "r2_layer1_in": round(float(r2_phys_in), 4),
        "r2_layer1_out": round(float(r2_phys_out), 4),
        "r2_layer2_in": round(float(r2_kde_in), 4),
        "r2_layer2_out": round(float(r2_kde_out), 4),
        "r2_combined_in": round(float(r2_in), 4),
        "r2_combined_out": round(float(r2_out), 4),
        "mae_combined_in": round(float(mae_in), 3),
        "mae_combined_out": round(float(mae_out), 3),
    },
    "asite_ad": asite_ad,
    "regime_ad": regime_ad,
    "shortlist_ad": {
        "pct_combined_in_ad": round(float(pct_sl_in), 1) if not np.isnan(pct_sl_in) else None,
        "pct_physics_reliable": round(float(pct_sl_phys), 1) if not np.isnan(pct_sl_phys) else None,
        "pct_kde_in_domain": round(float(pct_sl_kde), 1) if not np.isnan(pct_sl_kde) else None,
    }
}

res_path = os.path.join(RES_DIR, '49_ad_v77.json')
with open(res_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"  Results → {res_path}")

print("\n" + "=" * 68)
print("SUMMARY — Dual AD for CMLTRPCNNv77")
print("=" * 68)
print(f"  Layer 1 (physics gate): {physics_reliable.sum()} in-domain ({100*physics_reliable.mean():.1f}%)")
print(f"  Layer 2 (KDE):          {kde_in_domain.sum()} in-domain ({100*kde_in_domain.mean():.1f}%)")
print(f"  Combined in-AD:         {in_ad.sum()} ({100*in_ad.mean():.1f}%)")
print(f"  R² combined in-AD:      {r2_in:.4f}")
print(f"  R² combined out-AD:     {r2_out:.4f}")
print(f"  R² gap (in - out):      {r2_in - r2_out:.4f}")
print(f"  Key: Pb physics-compensated = model self-flags Pb unreliability")
print("=" * 68)
