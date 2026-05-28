"""
PIRNN, MD-PINN, and CM-LTR-PINN-v2 for ABO3 dielectric prediction.

PIRNN: Physics-Input Residual Neural Network
  εr = εr_CM + Δεr_LST(≥0) + Δεr_tilt(≤0) + Δεr_dis(±)
  Note: physics enters only as input features — no architectural constraints.
  Contrast with CMLTRPCNN (PCNN) where constraints are structural.

MD-PINN: Mechanistically Decomposed Physics-Informed Neural Network
  εr = CM_layer(α,M,ρ) × LST_layer(d0_per_field) × Reaney_layer(tilt) + ε_residual

CM-LTR-PINN-v2: Clausius-Mossotti → LST → Tilt → Residual (Physics++ version)
  Sequential physics chain with hard-signed branches and LST capacity bound.
  Physics+ descriptors feed directly into mechanism-specific branches.
  Confirmed best hyperparams: hidden=128, layers=4, lr=0.000738, lam_cap=0.324
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

# ── PIRNN ─────────────────────────────────────────────────────────────────────
class PIRNN(nn.Module):
    def __init__(self, n_in, hidden=64, sigma_noise=15.0):
        super().__init__()
        self.sigma = sigma_noise
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.lst_head  = nn.Sequential(nn.Linear(hidden,32), nn.SiLU(), nn.Linear(32,1), nn.Softplus())
        self.tilt_head = nn.Sequential(nn.Linear(hidden,32), nn.SiLU(), nn.Linear(32,1), nn.Softplus())
        self.dis_head  = nn.Sequential(nn.Linear(hidden,32), nn.SiLU(), nn.Linear(32,1), nn.Tanh())

    def forward(self, x, er_cm, d0pf, tilt):
        h = self.enc(x)
        dl = self.lst_head(h).squeeze(-1)  * (1 + 8*d0pf)   # LST ≥ 0
        dt = -self.tilt_head(h).squeeze(-1)* (1 + 4*tilt)   # Tilt ≤ 0
        dd = self.dis_head(h).squeeze(-1)  * self.sigma      # bounded ±
        return {"pred": er_cm+dl+dt+dd, "dl": dl, "dt": dt, "dd": dd}

# ── MD-PINN ───────────────────────────────────────────────────────────────────
class MDPINN(nn.Module):
    """
    Layer 0: CM equation (exact, no params)
    Layer 1: LST phonon enhancement (1 learnable scale per regime)
    Layer 2: Reaney tilt suppression (1 learnable scale)
    Layer 3: Residual MLP (bounded ±sigma)
    """
    def __init__(self, n_in, hidden=64, sigma_noise=10.0):
        super().__init__()
        self.sigma = sigma_noise
        # LST scale — one per Reaney regime (Ia, Ib, II, III)
        self.lst_scale = nn.Parameter(torch.tensor([1.0, 5.0, 2.0, 1.5]))
        # Tilt suppression factor (positive → multiplied to reduce)
        self.tilt_scale = nn.Parameter(torch.tensor([0.5]))
        # Small residual MLP
        self.res_mlp = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Tanh(),
        )

    def forward(self, x, er_cm, d0pf, tilt, regime):
        # Layer 0: CM (already computed analytically as er_cm)
        er0 = er_cm

        # Layer 1: LST  εs ≈ ε∞ × (1 + scale × d0_per_field)
        lst_s = F.softplus(self.lst_scale[regime])           # positive constraint
        er1 = er0 * (1 + lst_s * d0pf)

        # Layer 2: Reaney tilt suppression  εr_tilt = εr_LST × (1 - scale×tilt²)
        tilt_s = torch.sigmoid(self.tilt_scale) * 2.0       # ∈ (0,2)
        er2 = er1 * (1 - tilt_s * tilt.pow(2)).clamp(min=0.1)

        # Layer 3: Residual
        res = self.res_mlp(x).squeeze(-1) * self.sigma
        return {"pred": er2+res, "er_cm": er0, "er_lst": er1, "er_reaney": er2, "res": res}

# ── Losses ────────────────────────────────────────────────────────────────────
def pirnn_loss(out, y, lam=1.0):
    rmse = torch.sqrt(F.mse_loss(out["pred"], y))
    pen  = lam * (F.relu(-out["dl"]).pow(2).mean() +
                  F.relu( out["dt"]).pow(2).mean() +
                  0.01 * out["dd"].pow(2).mean())
    return rmse + pen, rmse

def mdpinn_loss(out, y, lam=0.5):
    rmse = torch.sqrt(F.mse_loss(out["pred"], y))
    pen_lst   = lam * F.relu(out["er_cm"] - out["er_lst"]).pow(2).mean()
    pen_tilt  = lam * F.relu(out["er_reaney"] - out["er_lst"]).pow(2).mean()
    return rmse + pen_lst + pen_tilt, rmse


# ── CM-LTR-PINN-v2 (Physics++) ────────────────────────────────────────────────
class CMLTRPINNv2(nn.Module):
    """
    Sequential physics chain: CM → LST → Tilt → Residual.

    Physics++ features:
      - Hard-signed branches: LST always ≥ 0, tilt always ≤ 0 (Softplus + negation)
      - LST capacity bound: prediction cannot exceed er_CM × lst_capacity
      - Physics+ descriptors fed to mechanism-specific branches:
          LST branch  ← soft_mode_activity, b_o_reduced_mass, lst_enhancement_proxy
          Tilt branch ← tilt_severity, charge_imbalance_proxy, continuous_tilt_strain
      - Regime-Ib specialized high-capacity LST head for d0_B_polarizable_A samples
      - Sample weighting: er>80 weighted 3×, er>150 weighted 5× in loss

    Confirmed best hyperparams (from collaborator 100-trial Optuna):
      hidden_dim=128, num_layers=4, lr=0.000738, batch_size=256,
      soft_capacity_strength=0.5, lambda_capacity=0.324,
      residual_scale=80.0, lst_scale=2.081
    """

    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4,
                 residual_scale=80.0, lst_scale=2.081,
                 soft_capacity_strength=0.5):
        super().__init__()
        self.residual_scale = residual_scale
        self.soft_capacity_strength = soft_capacity_strength

        # Shared encoder
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)

        # LST branch — standard (all regimes)
        self.lst_head = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()      # hard ≥ 0
        )

        # LST branch — Regime Ib specialized (high-capacity for soft-mode materials)
        self.lst_head_ib = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()       # hard ≥ 0
        )

        # Tilt branch — hard ≤ 0 via negated Softplus
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()        # output negated in forward
        )

        # Residual MLP — bounded ±residual_scale
        self.res_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Tanh()            # ∈ (-1, 1) × scale
        )

        # Learnable LST capacity (soft upper bound on LST enhancement)
        self.lst_capacity = nn.Parameter(torch.tensor(lst_scale))

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        """
        x         : full feature tensor [B, n_in]
        er_cm     : CM baseline [B]
        lst_feats : [soft_mode_activity, b_o_reduced_mass, lst_enhancement_proxy] [B, 3]
        tilt_feats: [tilt_severity, charge_imbalance_proxy, continuous_tilt_strain] [B, 3]
        is_ib     : Regime Ib flag [B] — float 0/1
        """
        h = self.encoder(x)

        # LST branch — blend standard and Ib-specialized heads
        h_lst = torch.cat([h, lst_feats], dim=1)
        delta_lst_std = self.lst_head(h_lst).squeeze(-1)
        delta_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst = (1 - is_ib) * delta_lst_std + is_ib * delta_lst_ib  # always ≥ 0

        # LST capacity bound: Δεr_LST ≤ (capacity - 1) × εr_CM
        capacity = F.softplus(self.lst_capacity) * self.soft_capacity_strength
        delta_lst = torch.clamp(delta_lst, max=capacity * er_cm.abs())

        # Tilt branch — architecturally forced ≤ 0
        h_tilt = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = -self.tilt_head(h_tilt).squeeze(-1)  # negated → always ≤ 0

        # Residual — bounded ±residual_scale
        delta_res = self.res_head(h).squeeze(-1) * self.residual_scale

        er_pred = er_cm + delta_lst + delta_tilt + delta_res

        return {
            "pred":       er_pred,
            "er_cm":      er_cm,
            "delta_lst":  delta_lst,
            "delta_tilt": delta_tilt,
            "delta_res":  delta_res,
        }


def cmltr_loss(out, y, sample_weights=None, lambda_capacity=0.324):
    """
    Physics++ loss for CM-LTR-PINN-v2.

    Data loss   : weighted RMSE (er>80 weighted 3×, er>150 weighted 5×)
    Physics loss: LST capacity violation penalty
    Sign losses : enforced architecturally — no penalty needed
    """
    pred = out["pred"]

    # Sample weighting for high-εr regime
    if sample_weights is None:
        w = torch.ones_like(y)
        w[y > 80]  = 3.0
        w[y > 150] = 5.0
    else:
        w = sample_weights

    # Weighted RMSE
    loss_data = torch.sqrt((w * (pred - y).pow(2)).mean())

    # LST capacity violation: penalise if delta_lst exceeds soft capacity
    capacity = F.softplus(out.get("lst_capacity",
                          torch.tensor(2.081))) * 0.5
    lst_violation = F.relu(out["delta_lst"] - capacity * out["er_cm"].abs())
    loss_cap = lambda_capacity * lst_violation.pow(2).mean()

    total = loss_data + loss_cap
    return total, loss_data


def get_cmltr_input_indices(feature_cols):
    """
    Return index arrays for physics-specific inputs.
    Call after building the feature matrix to get correct column positions.
    """
    lst_feats  = ["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]
    tilt_feats = ["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]
    ib_feat    = "d0_B_polarizable_A"

    lst_idx  = [feature_cols.index(f) for f in lst_feats  if f in feature_cols]
    tilt_idx = [feature_cols.index(f) for f in tilt_feats if f in feature_cols]
    ib_idx   = feature_cols.index(ib_feat) if ib_feat in feature_cols else None

    return lst_idx, tilt_idx, ib_idx


# ── CM-LTR-PINN-v3 ────────────────────────────────────────────────────────────
class CMLTRPINNv3(nn.Module):
    """
    Sequential CM → LST → Tilt → Residual with 6 graph-trace physics constraints.

    Over CMLTRPINNv2:
      - ptype flags + ordering_proxy fed directly into encoder via X (add in training script)
      - Residual reliance exposed as uncertainty signal (sequential analog of gate entropy)
      - DC ≤ 143 filter applied upstream: removes ferroelectrics, no er_CM=0 edge cases
      - Loss adds: residual_cap penalty + GII-conditional CM floor penalty

    Dataset: DC ≤ 143 filtered, n=1360, CM-stable n=994
    """

    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4,
                 residual_scale=80.0, lst_scale=2.081,
                 soft_capacity_strength=0.5):
        super().__init__()
        self.residual_scale         = residual_scale
        self.soft_capacity_strength = soft_capacity_strength

        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)

        self.lst_head = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.lst_head_ib = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.res_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.SiLU(),
            nn.Linear(32, 1), nn.Tanh()
        )
        self.lst_capacity = nn.Parameter(torch.tensor(lst_scale))

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib):
        h = self.encoder(x)

        h_lst         = torch.cat([h, lst_feats], dim=1)
        delta_lst_std = self.lst_head(h_lst).squeeze(-1)
        delta_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst     = (1 - is_ib) * delta_lst_std + is_ib * delta_lst_ib

        capacity  = F.softplus(self.lst_capacity) * self.soft_capacity_strength
        delta_lst = delta_lst.clamp(max=capacity * er_cm.abs())

        h_tilt    = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = -self.tilt_head(h_tilt).squeeze(-1)

        delta_res = self.res_head(h).squeeze(-1) * self.residual_scale

        er_pred = er_cm + delta_lst + delta_tilt + delta_res

        # Residual reliance: |delta_res| fraction of total physics correction
        total_delta  = (delta_lst + delta_tilt + delta_res).abs().clamp(min=1e-6)
        resid_reliance = delta_res.abs() / total_delta

        return {
            "pred":          er_pred,
            "er_cm":         er_cm,
            "delta_lst":     delta_lst,
            "delta_tilt":    delta_tilt,
            "delta_res":     delta_res,
            "resid_reliance": resid_reliance,   # uncertainty proxy (constraint 5)
        }


def cmltrv3_loss(out, y, b_size_mismatch, tilt_severity,
                 sample_weights=None,
                 lambda_capacity=0.324,
                 lambda_rcap=0.10,
                 lambda_floor=0.10,
                 reliance_threshold=0.50,
                 er_threshold=80.0,
                 gii_threshold=0.05):
    """
    Loss for CMLTRPINNv3.

    L_data   : weighted RMSE  (er>80 weighted 3×)
    L_cap    : LST capacity violation
    L_rcap   : residual cap — penalise high reliance when er_pred > er_threshold
    L_floor  : GII-conditional CM floor — stable structures respect CM lower bound
    """
    pred = out["pred"]

    w = sample_weights if sample_weights is not None else torch.ones_like(y)
    if sample_weights is None:
        w = w.clone()
        w[y > 80] = 3.0

    loss_data = torch.sqrt((w * (pred - y).pow(2)).mean())

    # LST capacity (reuse same formula as cmltr_loss)
    capacity      = F.softplus(torch.tensor(2.081)) * 0.5
    lst_violation = F.relu(out["delta_lst"] - capacity * out["er_cm"].abs())
    loss_cap      = lambda_capacity * lst_violation.pow(2).mean()

    # Residual cap for high-εr (constraint 2)
    high_er_mask = (pred > er_threshold).float()
    rcap_viol    = F.relu(out["resid_reliance"] - reliance_threshold) * high_er_mask
    loss_rcap    = lambda_rcap * rcap_viol.pow(2).mean()

    # GII-conditional CM floor (constraint 6)
    gii_proxy   = b_size_mismatch * tilt_severity
    stable_mask = (gii_proxy < gii_threshold).float()
    floor_viol  = F.relu(out["er_cm"] - pred) * stable_mask
    loss_floor  = lambda_floor * floor_viol.pow(2).mean()

    total = loss_data + loss_cap + loss_rcap + loss_floor
    return total, loss_data


def get_cmltrv3_input_indices(feature_cols):
    """Indices for CMLTRPINNv3 physics inputs + GII proxy columns."""
    lst_feats  = ["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]
    tilt_feats = ["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]
    ib_feat    = "d0_B_polarizable_A"
    bsm_feat   = "b_size_mismatch"
    tsev_feat  = "tilt_severity"

    lst_idx  = [feature_cols.index(f) for f in lst_feats  if f in feature_cols]
    tilt_idx = [feature_cols.index(f) for f in tilt_feats if f in feature_cols]
    ib_idx   = feature_cols.index(ib_feat)  if ib_feat  in feature_cols else None
    bsm_idx  = feature_cols.index(bsm_feat) if bsm_feat in feature_cols else None
    tsev_idx = feature_cols.index(tsev_feat) if tsev_feat in feature_cols else None

    return lst_idx, tilt_idx, ib_idx, bsm_idx, tsev_idx


# ── PIRNNv2 ───────────────────────────────────────────────────────────────────
class PIRNNv2(nn.Module):
    """
    PIRNNv2 (Physics-Input Residual Neural Network v2):
      - Tolerance-weighted LST branch  (Goldschmidt: t≈1 → max LST potential)
      - Spectral-normed disorder head  (Lipschitz bound → OOD stability)
      - Same sign conventions as PIRNN (LST≥0, tilt≤0, disorder±)
    """
    def __init__(self, n_in, hidden=64, sigma_noise=15.0):
        super().__init__()
        self.sigma = sigma_noise
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.lst_head  = nn.Sequential(nn.Linear(hidden,32), nn.SiLU(), nn.Linear(32,1), nn.Softplus())
        self.tilt_head = nn.Sequential(nn.Linear(hidden,32), nn.SiLU(), nn.Linear(32,1), nn.Softplus())
        self.dis_head  = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden, 32)), nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(32, 1)),      nn.Tanh()
        )

    def forward(self, x, er_cm, d0pf, tilt, tol_factor):
        h = self.enc(x)
        tol_w = torch.exp(-torch.abs(tol_factor - 1.0) * 10.0)   # ∈ (0,1], peaks at t=1
        dl = self.lst_head(h).squeeze(-1)  * (1 + 8*d0pf) * tol_w   # LST ≥ 0
        dt = -self.tilt_head(h).squeeze(-1)* (1 + 4*tilt)             # Tilt ≤ 0
        dd = self.dis_head(h).squeeze(-1)  * self.sigma               # bounded ±
        return {"pred": er_cm+dl+dt+dd, "dl": dl, "dt": dt, "dd": dd}


def pirnn_v2_loss(out, y, lam=1.0):
    """Weighted RMSE + sign-penalty loss for PIRNNv2 (adds high-εr sample weights)."""
    w = torch.ones_like(y)
    w[y > 80]  = 3.0
    w[y > 150] = 5.0
    rmse = torch.sqrt((w * (out["pred"] - y).pow(2)).mean())
    pen  = lam * (F.relu(-out["dl"]).pow(2).mean() +
                  F.relu( out["dt"]).pow(2).mean() +
                  0.01 * out["dd"].pow(2).mean())
    return rmse + pen, rmse


# ── CM-MAG-PINN ───────────────────────────────────────────────────────────────
class CMAGPINNv1(nn.Module):
    """
    CM-MAG-PINN: Mechanism-Adaptive Gating PINN

    Three parallel physics branches selected by a learned softmax gate:
      εr = εr_CM + g_LST·ΔεLST(≥0) + g_tilt·Δεtilt(≤0) + g_dis·Δεdis(±)
      [g_LST, g_tilt, g_dis] = Softmax(gate_head(h, regime_feats) / T)

    Physics laws embedded:
      LST capacity bound  : ΔεLST ≤ capacity × εr_CM   (Lyddane-Sachs-Teller)
      Tilt sign           : −Softplus → always ≤0        (Reaney 1994)
      Tolerance weighting : ΔεLST ∝ exp(−|t−1|×10)      (Goldschmidt geometry)
      Spectral norm       : Lipschitz bound on disorder   (OOD stability)
      Gate temperature    : controls mechanism sharpness  (physics specialisation)
    """
    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3, n_regime_feats=4,
                 hidden=128, n_layers=4, residual_scale=60.0, lst_scale=2.5,
                 soft_capacity_strength=0.5, gate_temperature=0.5):
        super().__init__()
        self.residual_scale         = residual_scale
        self.soft_capacity_strength = soft_capacity_strength
        self.gate_temp              = gate_temperature

        # Shared encoder
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)

        # LST branch — hard ≥ 0
        self.lst_head = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.lst_capacity = nn.Parameter(torch.tensor(lst_scale))

        # Tilt branch — output negated in forward → hard ≤ 0
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )

        # Disorder branch — spectral norm for Lipschitz stability, bounded ±
        self.dis_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden, 32)), nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(32, 1)),      nn.Tanh()
        )

        # Gate head — conditioned on encoder output + regime one-hots
        self.gate_head = nn.Sequential(
            nn.Linear(hidden + n_regime_feats, 32), nn.SiLU(),
            nn.Linear(32, 3)   # logits for [g_LST, g_tilt, g_dis]
        )

    def forward(self, x, er_cm, lst_feats, tilt_feats, regime_feats, tol_factor,
                physics_only=False):
        """
        x            : full feature tensor [B, n_in]
        er_cm        : CM baseline [B]
        lst_feats    : [soft_mode_activity, b_o_reduced_mass, lst_enhancement_proxy] [B,3]
        tilt_feats   : [tilt_severity, charge_imbalance_proxy, continuous_tilt_strain] [B,3]
        regime_feats : [regime_Ia, regime_Ib, regime_II, regime_III] [B,4]
        tol_factor   : tolerance factor [B]
        physics_only : if True, zero the disorder branch (warmup stage 1)
        """
        h = self.encoder(x)

        # Tolerance weighting: max at t=1 (cubic), decays for distorted structures
        tol_w = torch.exp(-torch.abs(tol_factor - 1.0) * 10.0)

        # LST branch
        h_lst    = torch.cat([h, lst_feats], dim=1)
        delta_lst = self.lst_head(h_lst).squeeze(-1) * tol_w
        capacity  = F.softplus(self.lst_capacity) * self.soft_capacity_strength
        delta_lst = delta_lst.clamp(max=capacity * er_cm.abs())

        # Tilt branch (negated)
        h_tilt    = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = -self.tilt_head(h_tilt).squeeze(-1)

        # Disorder branch (zeroed during warmup)
        if physics_only:
            delta_dis = torch.zeros_like(er_cm)
        else:
            delta_dis = self.dis_head(h).squeeze(-1) * self.residual_scale

        # Mechanism gate
        h_gate = torch.cat([h, regime_feats], dim=1)
        gate   = F.softmax(self.gate_head(h_gate) / self.gate_temp, dim=-1)

        er_pred = er_cm + gate[:,0]*delta_lst + gate[:,1]*delta_tilt + gate[:,2]*delta_dis

        return {
            "pred":       er_pred,
            "er_cm":      er_cm,
            "delta_lst":  delta_lst,
            "delta_tilt": delta_tilt,
            "delta_dis":  delta_dis,
            "gate":       gate,
        }


def cmag_loss(out, y, sample_weights=None, lambda_capacity=0.2, lambda_entropy=0.05,
              lst_capacity_val=2.5, soft_capacity_strength=0.5):
    """
    CM-MAG-PINN loss:
      weighted RMSE  + LST capacity violation + gate entropy penalty
    Gate entropy penalty encourages mechanism specialisation (physics-grounded gating).
    """
    pred = out["pred"]

    w = sample_weights if sample_weights is not None else torch.ones_like(y)
    if sample_weights is None:
        w[y > 80]  = 3.0
        w[y > 150] = 5.0

    loss_data = torch.sqrt((w * (pred - y).pow(2)).mean())

    # LST capacity violation
    capacity     = F.softplus(torch.tensor(lst_capacity_val)) * soft_capacity_strength
    lst_violation = F.relu(out["delta_lst"] - capacity * out["er_cm"].abs())
    loss_cap      = lambda_capacity * lst_violation.pow(2).mean()

    # Gate entropy penalty (lower entropy = more specialised = better physics)
    gate        = out["gate"].clamp(min=1e-8)
    gate_entropy = -(gate * gate.log()).sum(dim=-1).mean()
    loss_entropy = lambda_entropy * gate_entropy

    total = loss_data + loss_cap + loss_entropy
    return total, loss_data


def mixup_batch(tensors, alpha=0.4):
    """
    Mixup augmentation across all input tensors simultaneously.
    Interpolates composition, CM baseline, physics features, and target εr.
    Forces smooth learned behaviour across composition space.
    Returns mixed tensors with same structure.
    """
    lam = float(np.random.beta(alpha, alpha))
    n   = tensors[0].shape[0]
    idx = torch.randperm(n, device=tensors[0].device)
    return tuple(lam * t + (1 - lam) * t[idx] for t in tensors)


def get_cmag_input_indices(feature_cols):
    """
    Return column index arrays for all CM-MAG-PINN physics inputs.
    Also used by PIRNNv2 for tol_idx and d0pf_idx.
    """
    lst_feats    = ["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]
    tilt_feats   = ["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]
    regime_feats = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    tol_feat     = "tolerance_factor"
    d0pf_feat    = "d0_per_field"
    tilt_sev     = "tilt_severity"

    lst_idx    = [feature_cols.index(f) for f in lst_feats    if f in feature_cols]
    tilt_idx   = [feature_cols.index(f) for f in tilt_feats   if f in feature_cols]
    regime_idx = [feature_cols.index(f) for f in regime_feats if f in feature_cols]
    tol_idx    = feature_cols.index(tol_feat)  if tol_feat  in feature_cols else None
    d0pf_idx   = feature_cols.index(d0pf_feat) if d0pf_feat in feature_cols else None
    tsev_idx   = feature_cols.index(tilt_sev)  if tilt_sev  in feature_cols else None

    return lst_idx, tilt_idx, regime_idx, tol_idx, d0pf_idx, tsev_idx


# ── CM-MAG-PINN v2 ────────────────────────────────────────────────────────────
class CMAGPINNv2(nn.Module):
    """
    CM-MAG-PINN v2: Mechanism-Adaptive Gating PINN with 6 graph-trace physics constraints.

    New vs CMAGPINNv1:
      1. ptype flags in gate conditioning (ptype_double, ptype_complex_b, ptype_re_b_site)
         — site-misassignment → gate shifts away from CM branch for complex perovskites
      2. Regime-conditional gate prior (KL in loss) — Ib pushes g_LST, II/III push g_tilt
      3. Residual cap for high-εr (loss penalty) — limits residual branch when er_pred > 60
      4. S_config × T_sint interaction feature expected as part of x
         — ordering_proxy = b_config_entropy × T_sint/T_ref (computed in training script)
      5. Gate entropy exposed as uncertainty signal (gate_entropy in output dict)
      6. GII-conditional CM floor (loss penalty) — enforces εr ≥ εr_CM for stable structures
    """
    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 n_regime_feats=4, n_ptype_feats=3,
                 hidden=128, n_layers=4,
                 residual_scale=60.0, lst_scale=2.5,
                 soft_capacity_strength=0.5, gate_temperature=0.5):
        super().__init__()
        self.residual_scale         = residual_scale
        self.soft_capacity_strength = soft_capacity_strength
        self.gate_temp              = gate_temperature

        # Shared encoder
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.1)]
        self.encoder = nn.Sequential(*layers)

        # LST branch — hard ≥ 0
        self.lst_head = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.lst_capacity = nn.Parameter(torch.tensor(lst_scale))

        # Tilt branch — output negated → hard ≤ 0
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )

        # Disorder/residual branch — spectral norm for Lipschitz OOD stability
        self.dis_head = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden, 32)), nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(32, 1)),      nn.Tanh()
        )

        # Gate head — conditioned on encoder + regime one-hots + ptype flags (constraint 1)
        gate_in = hidden + n_regime_feats + n_ptype_feats
        self.gate_head = nn.Sequential(
            nn.Linear(gate_in, 32), nn.SiLU(),
            nn.Linear(32, 3)   # logits for [g_LST, g_tilt, g_dis]
        )

    def forward(self, x, er_cm, lst_feats, tilt_feats, regime_feats, tol_factor,
                ptype_feats, physics_only=False):
        """
        x            : full feature tensor [B, n_in]  (includes ordering_proxy column)
        er_cm        : CM baseline [B]  (0.0 for ptype_double rows, not NaN)
        lst_feats    : [soft_mode_activity, b_o_reduced_mass, lst_enhancement_proxy] [B,3]
        tilt_feats   : [tilt_severity, charge_imbalance_proxy, continuous_tilt_strain] [B,3]
        regime_feats : [regime_Ia, regime_Ib, regime_II, regime_III] [B,4]
        tol_factor   : tolerance factor [B]
        ptype_feats  : [ptype_double, ptype_complex_b, ptype_re_b_site] [B,3]
        physics_only : zero disorder branch during warmup stage
        """
        h = self.encoder(x)

        # Tolerance weighting: peak at t=1 (cubic), decays for distorted structures
        tol_w = torch.exp(-torch.abs(tol_factor - 1.0) * 10.0)

        # LST branch
        h_lst    = torch.cat([h, lst_feats], dim=1)
        delta_lst = self.lst_head(h_lst).squeeze(-1) * tol_w
        capacity  = F.softplus(self.lst_capacity) * self.soft_capacity_strength
        # For ptype_double rows (er_cm≈0), fall back to a fixed capacity of 20
        effective_cm = er_cm.abs().clamp(min=20.0)
        delta_lst    = delta_lst.clamp(max=capacity * effective_cm)

        # Tilt branch
        h_tilt    = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = -self.tilt_head(h_tilt).squeeze(-1)

        # Disorder/residual branch
        if physics_only:
            delta_dis = torch.zeros_like(er_cm)
        else:
            delta_dis = self.dis_head(h).squeeze(-1) * self.residual_scale

        # Gate: conditioned on regime + ptype (constraint 1 & 3)
        h_gate = torch.cat([h, regime_feats, ptype_feats], dim=1)
        gate   = F.softmax(self.gate_head(h_gate) / self.gate_temp, dim=-1)

        er_pred = er_cm + gate[:,0]*delta_lst + gate[:,1]*delta_tilt + gate[:,2]*delta_dis

        # Gate entropy as uncertainty signal (constraint 5)
        gate_clamped  = gate.clamp(min=1e-8)
        gate_entropy  = -(gate_clamped * gate_clamped.log()).sum(dim=-1)  # [B]

        # Residual reliance (constraint 2) — fraction of prediction carried by disorder
        total_delta = (gate[:,0]*delta_lst + gate[:,1]*delta_tilt + gate[:,2]*delta_dis).abs().clamp(min=1e-6)
        resid_reliance = (gate[:,2] * delta_dis.abs()) / total_delta

        return {
            "pred":          er_pred,
            "er_cm":         er_cm,
            "delta_lst":     delta_lst,
            "delta_tilt":    delta_tilt,
            "delta_dis":     delta_dis,
            "gate":          gate,
            "gate_entropy":  gate_entropy,    # uncertainty signal (higher = less certain)
            "resid_reliance": resid_reliance, # fraction of Δεr from disorder branch
        }


# ── Physics-derived regime gate priors ────────────────────────────────────────
# From graph traces: LST↔CM equivalence + Reaney regime physics
# [g_LST, g_tilt, g_dis]
_GATE_PRIOR = {
    "Ia":  torch.tensor([0.10, 0.65, 0.25]),  # antiphase+in-phase tilts dominate
    "Ib":  torch.tensor([0.70, 0.10, 0.20]),  # soft-mode LST dominates (d0+Ba/Sr)
    "II":  torch.tensor([0.20, 0.55, 0.25]),  # antiphase tilts only
    "III": torch.tensor([0.15, 0.60, 0.25]),  # antiphase+in-phase, lower t
}


def regime_gate_prior_kl(gate, regime_feats, lambda_prior=0.05):
    """
    KL divergence from gate distribution toward physics-motivated regime priors.
    regime_feats: [regime_Ia, regime_Ib, regime_II, regime_III] [B,4]
    Constraint 1: regime determines which mechanism should dominate.
    """
    device = gate.device
    prior_matrix = torch.stack([
        _GATE_PRIOR["Ia"], _GATE_PRIOR["Ib"],
        _GATE_PRIOR["II"], _GATE_PRIOR["III"]
    ], dim=0).to(device)  # [4, 3]

    # Blend priors by regime membership (handles multi-regime edge cases gracefully)
    regime_w = regime_feats / regime_feats.sum(dim=1, keepdim=True).clamp(min=1e-6)
    target   = regime_w @ prior_matrix  # [B, 3]
    target   = target.clamp(min=1e-8)
    target   = target / target.sum(dim=1, keepdim=True)

    # KL(prior || gate) — penalise gate deviating from physics expectation
    kl = (target * (target.log() - gate.clamp(min=1e-8).log())).sum(dim=-1).mean()
    return lambda_prior * kl


def residual_cap_loss(out, lambda_rcap=0.10, er_threshold=60.0, reliance_threshold=0.30):
    """
    Penalise high residual reliance when predicted εr > threshold.
    Addresses collaborator weakness: 29% residual-dominant predictions for high-εr.
    Constraint 2: residual branch should not dominate in physically-understood regimes.
    """
    er_pred  = out["pred"]
    reliance = out["resid_reliance"]

    high_er_mask = (er_pred > er_threshold).float()
    violation    = F.relu(reliance - reliance_threshold) * high_er_mask
    return lambda_rcap * violation.pow(2).mean()


def gii_floor_loss(out, b_size_mismatch, tilt_severity, lambda_floor=0.10, gii_threshold=0.05):
    """
    Enforce εr ≥ εr_CM for structurally stable perovskites (low GII proxy).
    GII proxy = b_size_mismatch × tilt_severity (BVS/GII community split insight).
    Constraint 6: when octahedra are stable, CM provides a reliable lower bound.
    """
    gii_proxy    = b_size_mismatch * tilt_severity
    stable_mask  = (gii_proxy < gii_threshold).float()
    floor_viol   = F.relu(out["er_cm"] - out["pred"]) * stable_mask
    return lambda_floor * floor_viol.pow(2).mean()


def magv2_loss(out, y, b_size_mismatch, tilt_severity, regime_feats,
               sample_weights=None,
               lambda_capacity=0.20, lambda_entropy=0.05, lambda_prior=0.05,
               lambda_rcap=0.10, lambda_floor=0.10,
               lst_capacity_val=2.5, soft_capacity_strength=0.5):
    """
    Full loss for CMAGPINNv2. Combines:
      L_data     : weighted RMSE (high-εr oversampled)
      L_cap      : LST capacity violation (Lyddane-Sachs-Teller bound)
      L_entropy  : gate entropy (encourages mechanism specialisation)
      L_prior    : regime gate prior KL (physics-motivated gate targets)
      L_rcap     : residual cap for high-εr (disorder branch discipline)
      L_floor    : GII-conditional CM floor (stable-structure lower bound)
    """
    pred = out["pred"]

    w = sample_weights if sample_weights is not None else torch.ones_like(y)
    if sample_weights is None:
        w = w.clone()
        w[y > 80]  = 3.0
        w[y > 150] = 5.0

    loss_data = torch.sqrt((w * (pred - y).pow(2)).mean())

    capacity      = F.softplus(torch.tensor(lst_capacity_val)) * soft_capacity_strength
    effective_cm  = out["er_cm"].abs().clamp(min=20.0)
    lst_violation = F.relu(out["delta_lst"] - capacity * effective_cm)
    loss_cap      = lambda_capacity * lst_violation.pow(2).mean()

    gate         = out["gate"].clamp(min=1e-8)
    gate_entropy = -(gate * gate.log()).sum(dim=-1).mean()
    loss_entropy = lambda_entropy * gate_entropy

    loss_prior = regime_gate_prior_kl(out["gate"], regime_feats, lambda_prior)
    loss_rcap  = residual_cap_loss(out, lambda_rcap)
    loss_floor = gii_floor_loss(out, b_size_mismatch, tilt_severity, lambda_floor)

    total = loss_data + loss_cap + loss_entropy + loss_prior + loss_rcap + loss_floor
    return total, loss_data


def get_magv2_input_indices(feature_cols):
    """
    Return column index arrays for all CMAGPINNv2 physics inputs.
    ptype_feats: ptype_double, ptype_complex_b, ptype_re_b_site
    Also returns ordering_proxy index (b_config_entropy × T_sint_norm — added in training script).
    """
    lst_feats    = ["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]
    tilt_feats   = ["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]
    regime_feats = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]
    ptype_feats  = ["ptype_double", "ptype_complex_b", "ptype_re_b_site"]
    tol_feat     = "tolerance_factor"
    bsm_feat     = "b_size_mismatch"
    tsev_feat    = "tilt_severity"

    lst_idx    = [feature_cols.index(f) for f in lst_feats    if f in feature_cols]
    tilt_idx   = [feature_cols.index(f) for f in tilt_feats   if f in feature_cols]
    regime_idx = [feature_cols.index(f) for f in regime_feats if f in feature_cols]
    ptype_idx  = [feature_cols.index(f) for f in ptype_feats  if f in feature_cols]
    tol_idx    = feature_cols.index(tol_feat) if tol_feat in feature_cols else None
    bsm_idx    = feature_cols.index(bsm_feat) if bsm_feat in feature_cols else None
    tsev_idx   = feature_cols.index(tsev_feat) if tsev_feat in feature_cols else None

    return lst_idx, tilt_idx, regime_idx, ptype_idx, tol_idx, bsm_idx, tsev_idx


# ── CM-MAG-PINN v3 (Regime-Conditional + Mechanism-Adaptive Gate) ─────────────
class CMAGPINNv3(nn.Module):
    """
    Combines option 1 (regime-conditional heads) + option 3 (mechanism-adaptive gate).

    Each physics branch (LST, tilt, disorder) has 4 regime-specific output heads
    (Ia / Ib / II / III) sharing a common early encoder, blended by physics-computed
    soft regime weights.  A separate learned softmax gate then weights the three
    mechanisms against each other.

    Physics constraints:
      LST heads   : Softplus  → always ≥ 0,  capacity-bounded by LST theory
      Tilt heads  : Softplus, negated → always ≤ 0  (Reaney 1994)
      Disorder    : spectral-norm Tanh → Lipschitz-stable, bounded ±
      Gate        : regime KL prior  (Ib → g_LST, II/III → g_tilt)
      CM floor    : εr ≥ εr_CM for structurally stable samples
    """

    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 n_regime_feats=4, n_ptype_feats=3,
                 hidden=256, n_layers=5,
                 residual_scale=60.0, lst_scale=2.5,
                 soft_capacity_strength=0.5, gate_temperature=0.5,
                 dropout=0.20):
        super().__init__()
        self.residual_scale         = residual_scale
        self.soft_capacity_strength = soft_capacity_strength
        self.gate_temp              = gate_temperature

        # ── Shared backbone ───────────────────────────────────────────────────
        enc_layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden),
                      nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            enc_layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                           nn.SiLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*enc_layers)

        # ── LST branch: shared early layer + 4 regime-specific heads ─────────
        branch_dim = hidden // 2
        self.lst_shared = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, branch_dim), nn.SiLU()
        )
        self.lst_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(branch_dim, 32), nn.SiLU(),
                          nn.Linear(32, 1), nn.Softplus())
            for _ in range(4)   # Ia, Ib, II, III
        ])
        self.lst_capacity = nn.Parameter(torch.tensor(lst_scale))

        # ── Tilt branch: shared early layer + 4 regime-specific heads ────────
        self.tilt_shared = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, branch_dim), nn.SiLU()
        )
        self.tilt_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(branch_dim, 32), nn.SiLU(),
                          nn.Linear(32, 1), nn.Softplus())
            for _ in range(4)
        ])

        # ── Disorder branch: shared early layer + 4 regime-specific heads ────
        self.dis_shared = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden, branch_dim)), nn.SiLU()
        )
        self.dis_heads = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(branch_dim, 32)), nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(32, 1)), nn.Tanh()
            )
            for _ in range(4)
        ])

        # ── Mechanism gate: conditioned on backbone + regime + ptype ─────────
        gate_in = hidden + n_regime_feats + n_ptype_feats
        self.gate_head = nn.Sequential(
            nn.Linear(gate_in, 32), nn.SiLU(),
            nn.Linear(32, 3)   # logits → [g_LST, g_tilt, g_dis]
        )

    def forward(self, x, er_cm, lst_feats, tilt_feats, regime_feats,
                tol_factor, ptype_feats):
        """
        x           : [B, n_in]
        er_cm       : [B]
        lst_feats   : [B, 3]  (soft_mode_activity, b_o_reduced_mass, lst_enhancement_proxy)
        tilt_feats  : [B, 3]  (tilt_severity, charge_imbalance_proxy, continuous_tilt_strain)
        regime_feats: [B, 4]  (regime_Ia, regime_Ib, regime_II, regime_III) — one-hot
        tol_factor  : [B]
        ptype_feats : [B, 3]  (ptype_double, ptype_complex_b, ptype_re_b_site)
        """
        h = self.encoder(x)

        # Soft regime weights (one-hot in practice, sum-to-1 normalised)
        regime_w = regime_feats / regime_feats.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B,4]

        # Tolerance weighting for LST (peaks at t=1, decays for distorted structures)
        tol_w = torch.exp(-torch.abs(tol_factor - 1.0) * 10.0)  # [B]

        # ── LST branch ───────────────────────────────────────────────────────
        h_lst = self.lst_shared(torch.cat([h, lst_feats], dim=1))  # [B, branch_dim]
        lst_per_r = torch.stack(
            [head(h_lst).squeeze(-1) for head in self.lst_heads], dim=1  # [B, 4]
        )
        delta_lst_raw = (regime_w * lst_per_r).sum(dim=1)  # [B], ≥0 (Softplus)
        delta_lst     = delta_lst_raw * tol_w
        capacity      = F.softplus(self.lst_capacity) * self.soft_capacity_strength
        effective_cm  = er_cm.abs().clamp(min=20.0)
        delta_lst     = delta_lst.clamp(max=capacity * effective_cm)

        # ── Tilt branch ──────────────────────────────────────────────────────
        h_tilt = self.tilt_shared(torch.cat([h, tilt_feats], dim=1))
        tilt_per_r  = torch.stack(
            [head(h_tilt).squeeze(-1) for head in self.tilt_heads], dim=1  # [B,4], ≥0
        )
        delta_tilt_raw = (regime_w * tilt_per_r).sum(dim=1)
        delta_tilt     = -delta_tilt_raw   # always ≤ 0

        # ── Disorder branch ──────────────────────────────────────────────────
        h_dis = self.dis_shared(h)
        dis_per_r = torch.stack(
            [head(h_dis).squeeze(-1) for head in self.dis_heads], dim=1  # [B,4], ∈(-1,1)
        )
        delta_dis_raw = (regime_w * dis_per_r).sum(dim=1)
        delta_dis     = delta_dis_raw * self.residual_scale

        # ── Mechanism gate ───────────────────────────────────────────────────
        h_gate = torch.cat([h, regime_feats, ptype_feats], dim=1)
        gate   = F.softmax(self.gate_head(h_gate) / self.gate_temp, dim=-1)  # [B, 3]

        er_pred = er_cm + gate[:, 0]*delta_lst + gate[:, 1]*delta_tilt + gate[:, 2]*delta_dis

        # Gate entropy (higher = more uncertain about mechanism)
        gate_clamped = gate.clamp(min=1e-8)
        gate_entropy = -(gate_clamped * gate_clamped.log()).sum(dim=-1)  # [B]

        # Residual reliance (fraction of |Δεr| from disorder branch)
        total_delta  = (gate[:, 0]*delta_lst + gate[:, 1]*delta_tilt +
                        gate[:, 2]*delta_dis).abs().clamp(min=1e-6)
        resid_reliance = (gate[:, 2] * delta_dis.abs()) / total_delta

        return {
            "pred":           er_pred,
            "er_cm":          er_cm,
            "delta_lst":      delta_lst,
            "delta_tilt":     delta_tilt,
            "delta_dis":      delta_dis,
            "gate":           gate,
            "gate_entropy":   gate_entropy,
            "resid_reliance": resid_reliance,
        }


def magv3_loss(out, y, b_size_mismatch, tilt_severity, regime_feats,
               sample_weights=None,
               lambda_capacity=0.20, lambda_entropy=0.05, lambda_prior=0.05,
               lambda_rcap=0.10, lambda_floor=0.10,
               lst_capacity_val=2.5, soft_capacity_strength=0.5):
    """
    Full loss for CMAGPINNv3.  Same 6 terms as magv2_loss — architecture change only.
    """
    pred = out["pred"]
    w    = sample_weights if sample_weights is not None else torch.ones_like(y)

    loss_data = torch.sqrt((w * (pred - y).pow(2)).mean())

    capacity      = F.softplus(torch.tensor(lst_capacity_val)) * soft_capacity_strength
    effective_cm  = out["er_cm"].abs().clamp(min=20.0)
    lst_violation = F.relu(out["delta_lst"] - capacity * effective_cm)
    loss_cap      = lambda_capacity * lst_violation.pow(2).mean()

    gate         = out["gate"].clamp(min=1e-8)
    gate_entropy = -(gate * gate.log()).sum(dim=-1).mean()
    loss_entropy = lambda_entropy * gate_entropy

    loss_prior = regime_gate_prior_kl(out["gate"], regime_feats, lambda_prior)
    loss_rcap  = residual_cap_loss(out, lambda_rcap)
    loss_floor = gii_floor_loss(out, b_size_mismatch, tilt_severity, lambda_floor)

    total = loss_data + loss_cap + loss_entropy + loss_prior + loss_rcap + loss_floor
    return total, loss_data


def get_magv3_input_indices(feature_cols):
    """Column indices for CMAGPINNv3 physics inputs (same layout as v2)."""
    return get_magv2_input_indices(feature_cols)


# ── CMLTRPINNv4 — Multiplicative Decomposition ────────────────────────────────
class CMLTRPINNv4(nn.Module):
    """
    Multiplicative physics decomposition:
        εr = εr_CM × f_LST × f_tilt × f_dis

    Physics rationale:
      - Lyddane-Sachs-Teller: εr(0) = εr(∞) × ∏(ω_LO/ω_TO)²  — naturally multiplicative
      - Tilt suppression scales proportionally with CM baseline, not absolutely
      - Covalency renormalises the polarisability multiplicatively

    Sign constraints (in log-space):
      log_f_LST  ≥ 0  (LST always enhances) → softplus output
      log_f_tilt ≤ 0  (tilt always suppresses) → −softplus output
      log_f_dis unconstrained (disorder / processing ± effect)

    Fallback for er_CM = 0:
      A separate additive head predicts εr directly, selected by has_cm flag.
    """

    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=256, n_layers=4, dropout=0.1,
                 lst_log_cap=2.0, tilt_log_cap=2.0):
        super().__init__()
        self.lst_log_cap  = lst_log_cap   # max log(f_LST)  → f_LST ≤ exp(cap) ≈ 7.4
        self.tilt_log_cap = tilt_log_cap  # max |log(f_tilt)|

        # Shared encoder
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)

        # LST branch: log(f_LST) ≥ 0; standard + Ib-boosted heads
        self.lst_head = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()       # output ≥ 0
        )
        self.lst_head_ib = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )

        # Tilt branch: log(f_tilt) ≤ 0 — softplus then negate
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()       # positive → negate below
        )

        # Disorder/residual: unconstrained log-factor
        self.dis_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Fallback additive head for samples without valid er_CM
        self.fallback_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()       # εr > 0
        )

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib, has_cm=None):
        h = self.encoder(x)

        # LST factor: log(f_LST) ≥ 0
        h_lst        = torch.cat([h, lst_feats], dim=1)
        log_f_lst_std = self.lst_head(h_lst).squeeze(-1)
        log_f_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        log_f_lst     = (1 - is_ib) * log_f_lst_std + is_ib * log_f_lst_ib
        log_f_lst     = log_f_lst.clamp(max=self.lst_log_cap)

        # Tilt factor: log(f_tilt) ≤ 0
        h_tilt     = torch.cat([h, tilt_feats], dim=1)
        log_f_tilt = -self.tilt_head(h_tilt).squeeze(-1)
        log_f_tilt  = log_f_tilt.clamp(min=-self.tilt_log_cap)

        # Disorder factor: unconstrained
        log_f_dis = self.dis_head(h).squeeze(-1)

        # Multiplicative prediction: εr = er_CM × exp(Σ log_f)
        log_er_mult = torch.log(er_cm.abs().clamp(min=1.0)) + log_f_lst + log_f_tilt + log_f_dis
        er_mult     = torch.exp(log_er_mult)

        # Fallback for er_CM = 0 samples (no valid CM baseline)
        er_fallback = self.fallback_head(h).squeeze(-1) + 8.0   # offset: εr ≥ 8

        # Blend: use multiplicative path where CM is valid, fallback otherwise
        if has_cm is None:
            has_cm = (er_cm.abs() > 1.0).float()
        er_pred = has_cm * er_mult + (1 - has_cm) * er_fallback

        total_log = (log_f_lst + log_f_tilt + log_f_dis).abs().clamp(min=1e-6)
        dis_reliance = log_f_dis.abs() / total_log

        return {
            "pred":         er_pred,
            "er_cm":        er_cm,
            "log_f_lst":    log_f_lst,
            "log_f_tilt":   log_f_tilt,
            "log_f_dis":    log_f_dis,
            "f_lst":        torch.exp(log_f_lst),
            "f_tilt":       torch.exp(log_f_tilt),
            "f_dis":        torch.exp(log_f_dis),
            "dis_reliance": dis_reliance,
        }


def cmltrv4_loss(out, y, b_size_mismatch, tilt_severity,
                 sample_weights=None,
                 lambda_floor=0.10,
                 lambda_rcap=0.10,
                 lambda_reg=0.01,
                 er_threshold=80.0,
                 reliance_threshold=0.50,
                 gii_threshold=0.05):
    """
    Loss for CMLTRPINNv4 (multiplicative).

    L_data     : weighted RMSE  (εr>80 weighted 3×)
    L_log      : log-space MSE  (stabilises high-εr tail)
    L_floor    : GII-conditional CM floor — f_total should not push below CM
    L_rcap     : residual cap — disorder factor shouldn't dominate high-εr
    L_reg      : entropy-like regularisation — keep factors near 1 (log_f ≈ 0)
    """
    pred      = out["pred"]
    log_f_dis = out["log_f_dis"]
    log_f_lst = out["log_f_lst"]
    dis_rel   = out["dis_reliance"]

    w = sample_weights if sample_weights is not None else torch.ones_like(y)
    if sample_weights is None:
        w = w.clone()
        w[y > 80] = 3.0

    # Primary: weighted RMSE
    loss_data = torch.sqrt((w * (pred - y).pow(2)).mean())

    # Log-space MSE — reduces dominance of high-εr outliers
    loss_log = ((torch.log(pred.clamp(min=1.0)) - torch.log(y.clamp(min=1.0))).pow(2)).mean()

    # GII-conditional CM floor: stable structures → f_total ≥ 1 (pred ≥ er_CM)
    gii_proxy   = b_size_mismatch * tilt_severity
    stable_mask = (gii_proxy < gii_threshold).float()
    floor_viol  = F.relu(out["er_cm"] - pred) * stable_mask
    loss_floor  = lambda_floor * floor_viol.pow(2).mean()

    # Residual cap: disorder factor shouldn't dominate for high-εr predictions
    high_er_mask = (pred > er_threshold).float()
    rcap_viol    = F.relu(dis_rel - reliance_threshold) * high_er_mask
    loss_rcap    = lambda_rcap * rcap_viol.pow(2).mean()

    # Regularisation: penalise extreme log-factors (keep corrections modest)
    loss_reg = lambda_reg * (log_f_dis.pow(2).mean() + log_f_lst.pow(2).mean())

    total = loss_data + 0.1 * loss_log + loss_floor + loss_rcap + loss_reg
    return total, loss_data


def get_cmltrv4_input_indices(feature_cols):
    """Same physics inputs as v3."""
    return get_cmltrv3_input_indices(feature_cols)


# ── CMLTRPINNv5: Confidence-Gated Physics PINN ───────────────────────────────
class CMLTRPINNv5(nn.Module):
    """
    Confidence-Gated Physics PINN.

    Architecture:
        εr = εr_CM + δ_LST(≥0) + δ_tilt(≤0) + gated_scale × δ_res_raw
        gated_scale = residual_base × (1 - confidence)
        confidence  = σ(conf_head([er_CM_norm, σ_CM_norm, has_cm, cm_flag, regime×4]))

    When confidence → 1 (exact CM, low σ, in-domain regime):
        gated_scale → 0 → physics branches must carry the prediction
    When confidence → 0 (VCA, high σ, OOD):
        gated_scale → residual_base → ML compensates → R² maintained

    Physics dominance is structural, not enforced by a penalty.
    """

    def __init__(self, n_in, n_lst_feats=3, n_tilt_feats=3,
                 hidden=128, n_layers=4,
                 residual_base=25.0,
                 lst_scale=2.081,
                 soft_capacity_strength=0.5,
                 n_conf_feats=8,
                 dropout=0.1):
        super().__init__()
        self.residual_base          = residual_base
        self.soft_capacity_strength = soft_capacity_strength

        # Shared encoder
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                  nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                       nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)

        # Confidence head — small MLP on physics reliability signals
        self.conf_head = nn.Sequential(
            nn.Linear(n_conf_feats, 32), nn.SiLU(),
            nn.Linear(32, 16),           nn.SiLU(),
            nn.Linear(16, 1),            nn.Sigmoid()
        )
        # Start confidence ≈ 0 → gated_scale ≈ residual_base (full capacity).
        # Physics branches mature first; confidence then grows to suppress residual.
        nn.init.constant_(self.conf_head[-2].bias, -4.0)

        # LST branch: δ_LST ≥ 0 (Lyddane-Sachs-Teller enhancement)
        self.lst_head = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.lst_head_ib = nn.Sequential(
            nn.Linear(hidden + n_lst_feats, 128), nn.SiLU(),
            nn.Linear(128, 64),                   nn.SiLU(),
            nn.Linear(64, 1),                     nn.Softplus()
        )
        self.lst_capacity = nn.Parameter(torch.tensor(lst_scale))

        # Tilt branch: δ_tilt ≤ 0 (octahedral tilt suppression)
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden + n_tilt_feats, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Softplus()
        )

        # Residual branch (gated — scale set by confidence)
        self.res_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.SiLU(),
            nn.Linear(32, 1),      nn.Tanh()
        )

        # Fallback for has_cm=0 samples
        self.fallback_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.SiLU(),
            nn.Linear(64, 32),     nn.SiLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, er_cm, lst_feats, tilt_feats, is_ib, conf_feats, has_cm):
        """
        conf_feats : (B, 8) — [er_CM_norm, σ_CM_norm, has_cm, cm_approx_flag, Ia, Ib, II, III]
        has_cm     : (B,)   — 1 if εr_CM is available, else 0
        """
        h = self.encoder(x)

        # Confidence ∈ (0,1) — clamped for numerical stability
        confidence  = self.conf_head(conf_feats).squeeze(-1).clamp(0.0, 1.0)
        gated_scale = (self.residual_base * (1.0 - confidence)).clamp(0.0, 150.0)

        # LST branch — no hard clamp (gradient must flow to lst_capacity)
        h_lst         = torch.cat([h, lst_feats], dim=1)
        delta_lst_std = self.lst_head(h_lst).squeeze(-1)
        delta_lst_ib  = self.lst_head_ib(h_lst).squeeze(-1)
        delta_lst     = (1 - is_ib) * delta_lst_std + is_ib * delta_lst_ib
        # Soft capacity is enforced via loss_cap penalty; hard clamp blocks gradient to lst_capacity

        # Tilt branch
        h_tilt     = torch.cat([h, tilt_feats], dim=1)
        delta_tilt = -self.tilt_head(h_tilt).squeeze(-1)

        # Gated residual — clamp to prevent overflow
        delta_res = (self.res_head(h).squeeze(-1) * gated_scale).clamp(-150.0, 150.0)

        # Physics path vs fallback
        er_physics  = (er_cm + delta_lst + delta_tilt + delta_res).clamp(1.0, 500.0)
        er_fallback = (self.fallback_head(h).squeeze(-1) + 8.0).clamp(1.0, 500.0)
        er_pred     = has_cm * er_physics + (1.0 - has_cm) * er_fallback

        total_corr   = (delta_lst.abs() + delta_tilt.abs() + delta_res.abs()).clamp(min=1e-6)
        physics_frac = (delta_lst.abs() + delta_tilt.abs()) / total_corr

        return {
            "pred":         er_pred,
            "er_cm":        er_cm,
            "delta_lst":    delta_lst,
            "delta_tilt":   delta_tilt,
            "delta_res":    delta_res,
            "confidence":   confidence,
            "gated_scale":  gated_scale,
            "physics_frac": physics_frac,
        }


def cmltrv5_loss(out, y, b_size_mismatch, tilt_severity, sigma_cm, has_cm,
                 sample_weights=None,
                 lambda_cap=0.1,
                 lambda_floor=0.1,
                 lambda_conf_cal=0.5,
                 gii_threshold=0.05,
                 sigma_cm_reliable_thresh=0.3):
    """
    Loss for CMLTRPINNv5.

    L_data     : weighted RMSE (εr>80 weighted 3×)
    L_cap      : LST capacity violation (Lyddane-Sachs-Teller bound)
    L_floor    : GII-conditional CM floor (stable structures respect CM)
    L_conf_cal : calibration — push confidence high for reliable samples
                 (exact CM + low σ_CM) so residual gate naturally closes
    """
    pred = out["pred"]

    w = sample_weights if sample_weights is not None else torch.ones_like(y)
    if sample_weights is None:
        w = w.clone()
        w[y > 80] = 3.0
    loss_data = torch.sqrt((w * (pred - y).pow(2)).mean())

    # LST capacity
    capacity  = F.softplus(torch.tensor(2.081)) * 0.5
    lst_viol  = F.relu(out["delta_lst"] - capacity * out["er_cm"].abs())
    loss_cap  = lambda_cap * lst_viol.pow(2).mean()

    # GII-conditional CM floor
    gii_proxy   = b_size_mismatch * tilt_severity
    stable_mask = (gii_proxy < gii_threshold).float()
    floor_viol  = F.relu(out["er_cm"] - pred) * stable_mask
    loss_floor  = lambda_floor * floor_viol.pow(2).mean()

    # Confidence calibration: push conf ≥ 0.7 for reliable samples
    sigma_norm    = sigma_cm / (sigma_cm.max().clamp(min=1e-6))
    reliable_mask = has_cm * (sigma_norm < sigma_cm_reliable_thresh).float()
    conf_viol     = F.relu(0.7 - out["confidence"]) * reliable_mask
    loss_conf     = lambda_conf_cal * conf_viol.pow(2).mean()

    total = loss_data + loss_cap + loss_floor + loss_conf
    return total, loss_data


def get_cmltrv5_input_indices(feature_cols):
    """Indices for CMLTRPINNv5 physics inputs and confidence features."""
    lst_feats   = ["soft_mode_activity", "b_o_reduced_mass", "lst_enhancement_proxy"]
    tilt_feats  = ["tilt_severity", "charge_imbalance_proxy", "continuous_tilt_strain"]
    regime_feats = ["regime_Ia", "regime_Ib", "regime_II", "regime_III"]

    lst_idx    = [feature_cols.index(f) for f in lst_feats  if f in feature_cols]
    tilt_idx   = [feature_cols.index(f) for f in tilt_feats if f in feature_cols]
    regime_idx = [feature_cols.index(f) for f in regime_feats if f in feature_cols]

    ib_idx   = feature_cols.index("d0_B_polarizable_A")  if "d0_B_polarizable_A"  in feature_cols else None
    bsm_idx  = feature_cols.index("b_size_mismatch")     if "b_size_mismatch"     in feature_cols else None
    tsev_idx = feature_cols.index("tilt_severity")       if "tilt_severity"       in feature_cols else None
    flag_idx = feature_cols.index("cm_approx_flag")      if "cm_approx_flag"      in feature_cols else None

    return lst_idx, tilt_idx, ib_idx, bsm_idx, tsev_idx, regime_idx, flag_idx


# ── CMLTRPCNNv6 ───────────────────────────────────────────────────────────────
class CMLTRPCNNv6(nn.Module):
    """
    CM-LTR Physics-Constrained Neural Network v6.

    Architecture (LOCKED after 4 LLM councils):
        εr = εr_CM + δ_LST(≥0) + δ_tilt(≤0) + δ_res(confidence-gated)

    Design principles:
      - ZERO shared encoder — each branch sees ONLY its own feature partition.
        This is the fix for V1 identifiability (corr(δ_LST,δ_res) < 0.3).
      - Sign constraints via activation: softplus for LST, −softplus for tilt.
        Not enforced by loss penalties (which were shown to be ineffective).
      - Regime multiplier on δ_LST: FIXED lookup table (no training parameters).
        Ia=1.0, Ib=1.5, II=0.9, III=0.7 — applied after branch output.
      - Confidence gate: FIXED β priors (not learned).
        σ_conf = sigmoid(β₀ + β₁·cm_approx_flag + β₂·gii_norm + β₃·phase_tr)
        β = [−2, +3, +2, +2]. Calibrated on 10% held-out split.
      - Residual scale: single learned scalar (allows magnitude to adapt).

    Physics embedded:
      - εr_CM: zero-parameter analytic formula (Clausius-Mossotti + Shannon)
      - δ_LST ≥ 0: Lyddane-Sachs-Teller soft-mode enhancement
      - δ_tilt ≤ 0: Reaney octahedral tilt suppression
      - δ_res: residual suppressed structurally where physics is reliable
    """

    # Fixed regime multipliers (Reaney 1994)
    REGIME_SCALES = torch.tensor([1.0, 1.5, 0.9, 0.7])   # [Ia, Ib, II, III]

    # Fixed confidence gate β priors
    GATE_BETA = torch.tensor([-2.0, 3.0, 2.0, 2.0])       # [β₀, β_vca, β_gii, β_phase]

    def __init__(self,
                 n_lst:  int,   # number of LST branch features (30)
                 n_tilt: int,   # number of Tilt branch features (30)
                 n_res:  int,   # number of Residual branch features (40)
                 lst_hidden:  int = 64,
                 tilt_hidden: int = 64,
                 res_hidden:  int = 128,
                 residual_scale: float = 30.0,
                 dropout: float = 0.05):
        super().__init__()

        # ── LST branch: 2-layer MLP → regime_scale × softplus → δ_LST ≥ 0 ────
        self.lst_mlp = nn.Sequential(
            nn.Linear(n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden // 2), nn.SiLU(),
            nn.Linear(lst_hidden // 2, 1),
        )
        # Separate regime-enhanced head for Regime Ib (stronger soft-mode)
        self.lst_mlp_ib = nn.Sequential(
            nn.Linear(n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden), nn.SiLU(),
            nn.Linear(lst_hidden, 1),
        )

        # ── Tilt branch: 2-layer MLP → −softplus → δ_tilt ≤ 0 ───────────────
        self.tilt_mlp = nn.Sequential(
            nn.Linear(n_tilt, tilt_hidden), nn.LayerNorm(tilt_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tilt_hidden, tilt_hidden // 2), nn.SiLU(),
            nn.Linear(tilt_hidden // 2, 1),
        )

        # ── Residual branch: 3-layer MLP, gated ──────────────────────────────
        # +1 input: normalised partial physics prediction (er_cm+δ_LST+δ_tilt)/143
        # fed stop-gradient so physics branches stay independent
        self.res_mlp = nn.Sequential(
            nn.Linear(n_res + 1, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden // 2), nn.SiLU(),
            nn.Linear(res_hidden // 2, 1),
        )

        # Learned residual scale (initialised to residual_scale)
        self.log_res_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale))))

        # Fallback head for no-CM samples (100 layered / complex structures)
        self.fallback_mlp = nn.Sequential(
            nn.Linear(n_res, res_hidden), nn.SiLU(),
            nn.Linear(res_hidden, 64),   nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _regime_scale(self, lst_feats: torch.Tensor, regime_indices: list) -> torch.Tensor:
        """Compute per-sample regime multiplier from one-hot regime features."""
        # regime_indices: [ia_idx, ib_idx, ii_idx, iii_idx] within lst_feats
        # Threshold back to binary: after StandardScaler, 0→≈-0.45, 1→≈+2.2
        if len(regime_indices) == 4:
            ia   = (lst_feats[:, regime_indices[0]] > 0.0).float()
            ib   = (lst_feats[:, regime_indices[1]] > 0.0).float()
            ii   = (lst_feats[:, regime_indices[2]] > 0.0).float()
            iii  = (lst_feats[:, regime_indices[3]] > 0.0).float()
            no_regime = (1.0 - ia - ib - ii - iii).clamp(0.0, 1.0)  # VCA / unclassified → Ia
            scales = self.REGIME_SCALES.to(lst_feats.device)
            return (ia + no_regime) * scales[0] + ib * scales[1] + ii * scales[2] + iii * scales[3]
        return torch.ones(lst_feats.shape[0], device=lst_feats.device)

    def _confidence_gate(self, cm_approx: torch.Tensor,
                         gii_norm: torch.Tensor,
                         phase_tr: torch.Tensor) -> torch.Tensor:
        """Fixed β confidence gate — no learned parameters."""
        beta = self.GATE_BETA.to(cm_approx.device)
        logit = beta[0] + beta[1] * cm_approx + beta[2] * gii_norm + beta[3] * phase_tr
        return torch.sigmoid(logit)

    def forward(self,
                lst_feats:   torch.Tensor,   # (B, n_lst)
                tilt_feats:  torch.Tensor,   # (B, n_tilt)
                res_feats:   torch.Tensor,   # (B, n_res)
                er_cm:       torch.Tensor,   # (B,)
                has_cm:      torch.Tensor,   # (B,) — 1 if CM available
                cm_approx:   torch.Tensor,   # (B,) — cm_approx_flag
                gii_norm:    torch.Tensor,   # (B,) — GII normalised to [0,1]
                phase_tr:    torch.Tensor,   # (B,) — phase_transition flag
                regime_idx:  list):          # indices of [Ia,Ib,II,III] in lst_feats
        """
        Returns dict with pred, er_cm, delta_lst, delta_tilt, delta_res,
        sigma_conf, physics_frac.
        """
        res_scale = torch.exp(self.log_res_scale)

        # ── LST branch ────────────────────────────────────────────────────────
        is_ib = (lst_feats[:, regime_idx[1]] > 0.0).float() if len(regime_idx) >= 2 else torch.zeros(lst_feats.shape[0], device=lst_feats.device)
        h_lst_std = self.lst_mlp(lst_feats).squeeze(-1)
        h_lst_ib  = self.lst_mlp_ib(lst_feats).squeeze(-1)
        h_lst     = (1 - is_ib) * h_lst_std + is_ib * h_lst_ib

        # Architectural sign constraint: softplus guarantees δ_LST ≥ 0
        delta_lst_raw = F.softplus(h_lst)
        # Fixed regime multiplier (no training)
        r_scale   = self._regime_scale(lst_feats, regime_idx)
        delta_lst = r_scale * delta_lst_raw

        # ── Tilt branch ───────────────────────────────────────────────────────
        h_tilt    = self.tilt_mlp(tilt_feats).squeeze(-1)
        # Architectural sign constraint: −softplus guarantees δ_tilt ≤ 0
        delta_tilt = -F.softplus(h_tilt)

        # ── Confidence gate (fixed β, no training) ────────────────────────────
        sigma_conf = self._confidence_gate(cm_approx, gii_norm, phase_tr)

        # ── Residual branch (gated) ───────────────────────────────────────────
        # Augment residual features with normalised partial physics prediction
        # (stop-gradient: physics branches stay independent of residual opt)
        er_phys_partial = (er_cm + delta_lst.detach() + delta_tilt.detach()) / 143.0
        res_feats_aug = torch.cat([res_feats, er_phys_partial.unsqueeze(-1)], dim=-1)
        h_res      = self.res_mlp(res_feats_aug).squeeze(-1)
        delta_res_raw = torch.tanh(h_res) * res_scale
        delta_res  = sigma_conf * delta_res_raw

        # ── Output assembly ───────────────────────────────────────────────────
        er_physics  = (er_cm + delta_lst + delta_tilt + delta_res).clamp(1.0, 600.0)

        # Fallback for no-CM samples (cm_approx=2 or has_cm=0)
        er_fallback = (self.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
        er_pred     = has_cm * er_physics + (1.0 - has_cm) * er_fallback

        total_delta  = (delta_lst.abs() + delta_tilt.abs() + delta_res.abs()).clamp(min=1e-6)
        physics_frac = (delta_lst.abs() + delta_tilt.abs()) / total_delta

        return {
            "pred":         er_pred,
            "er_cm":        er_cm,
            "delta_lst":    delta_lst,
            "delta_tilt":   delta_tilt,
            "delta_res":    delta_res,
            "sigma_conf":   sigma_conf,
            "physics_frac": physics_frac,
            "res_scale":    res_scale,
        }


def cmltrv6_loss(out, y, gii, cm_approx_flag,
                 sample_weights=None,
                 lambda_lst=0.01,
                 lambda_gii=0.05,
                 lambda_vca=0.02,
                 er_cm=None):
    """
    CMLTRPCNNv6 loss.

    L_total = L_MSE
            + λ_lst  × LST_capacity_bound   (δ_LST ≤ 5 × εr_CM)
            + λ_gii  × GII_floor            (εr_pred ≥ εr_CM when GII low)
            + λ_vca  × VCA_regularisation   (suppress δ_res for VCA samples to prevent
                                               over-correcting the approximate CM formula)
    """
    pred = out["pred"]

    w = sample_weights if sample_weights is not None else torch.ones_like(y)
    loss_data = F.mse_loss(pred * w, y * w)

    # LST capacity: δ_LST ≤ 5 × εr_CM (very generous — soft ceiling only)
    if er_cm is not None:
        cap_viol = F.relu(out["delta_lst"] - 5.0 * er_cm.abs())
        loss_lst = lambda_lst * cap_viol.pow(2).mean()
    else:
        loss_lst = torch.tensor(0.0, device=pred.device)

    # GII floor: for structurally stable samples (GII < 0.3), εr_pred ≥ εr_CM
    if er_cm is not None:
        stable_mask = (gii < 0.3).float()
        floor_viol  = F.relu(er_cm - pred) * stable_mask
        loss_gii    = lambda_gii * floor_viol.pow(2).mean()
    else:
        loss_gii = torch.tensor(0.0, device=pred.device)

    # VCA regularisation: penalise large δ_res for VCA (cm_approx=1) samples.
    # Gate already opens for VCA (high σ_conf), so this soft penalty prevents
    # the residual from fully absorbing CM approximation errors.
    vca_mask = (cm_approx_flag > 0.5).float()
    loss_vca = lambda_vca * (out["delta_res"].pow(2) * vca_mask).mean()

    total = loss_data + loss_lst + loss_gii + loss_vca
    return total, loss_data


# ── CMLTRPCNNv7 ───────────────────────────────────────────────────────────────
class CMLTRPCNNv7(nn.Module):
    """
    CM-LTR Physics-Constrained Neural Network v7.

    Improvement over v6: SHARED TRUNK over all features.
      v6: strict partition — each branch sees ONLY its own features (no cross-talk)
      v7: shared trunk encoder over ALL features → physics-guided heads receive
          trunk output concatenated with branch-specific features.

    Cross-branch interactions captured: composition context (res features) can
    inform LST/tilt behaviour; tilt geometry can modulate LST enhancement.
    All physics constraints preserved:
      δ_LST ≥ 0  (softplus + regime scale)
      δ_tilt ≤ 0 (−softplus)
      δ_res gated (fixed β confidence gate)
      εr_CM anchor (zero-parameter CM formula)
    """

    REGIME_SCALES = torch.tensor([1.0, 1.5, 0.9, 0.7])  # [Ia, Ib, II, III]
    GATE_BETA     = torch.tensor([-2.0, 3.0, 2.0, 2.0])  # [β₀, β_vca, β_gii, β_phase]

    def __init__(self,
                 n_lst:  int,
                 n_tilt: int,
                 n_res:  int,
                 trunk_hidden:  int   = 128,
                 lst_hidden:    int   = 64,
                 tilt_hidden:   int   = 64,
                 res_hidden:    int   = 128,
                 residual_scale: float = 80.0,
                 dropout: float = 0.20):
        super().__init__()
        n_feat = n_lst + n_tilt + n_res
        t_out  = trunk_hidden // 2

        # ── Shared trunk: all features → compressed representation ─────────────
        self.trunk = nn.Sequential(
            nn.Linear(n_feat, trunk_hidden), nn.LayerNorm(trunk_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden), nn.LayerNorm(trunk_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, t_out), nn.SiLU(),
        )

        # ── LST head: trunk_out + lst_feats → δ_LST (two heads for regime Ib) ─
        self.lst_head = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden // 2), nn.SiLU(),
            nn.Linear(lst_hidden // 2, 1),
        )
        self.lst_head_ib = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden), nn.SiLU(),
            nn.Linear(lst_hidden, 1),
        )

        # ── Tilt head: trunk_out + tilt_feats → δ_tilt ────────────────────────
        self.tilt_head = nn.Sequential(
            nn.Linear(t_out + n_tilt, tilt_hidden), nn.LayerNorm(tilt_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tilt_hidden, tilt_hidden // 2), nn.SiLU(),
            nn.Linear(tilt_hidden // 2, 1),
        )

        # ── Residual head: trunk_out + res_feats + physics_context → δ_res ─────
        self.res_head = nn.Sequential(
            nn.Linear(t_out + n_res + 1, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden // 2), nn.SiLU(),
            nn.Linear(res_hidden // 2, 1),
        )

        self.log_res_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale))))

        # Fallback for no-CM samples
        self.fallback_mlp = nn.Sequential(
            nn.Linear(n_res, res_hidden), nn.SiLU(),
            nn.Linear(res_hidden, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _regime_scale(self, lst_feats: torch.Tensor, regime_indices: list) -> torch.Tensor:
        if len(regime_indices) == 4:
            ia  = (lst_feats[:, regime_indices[0]] > 0.0).float()
            ib  = (lst_feats[:, regime_indices[1]] > 0.0).float()
            ii  = (lst_feats[:, regime_indices[2]] > 0.0).float()
            iii = (lst_feats[:, regime_indices[3]] > 0.0).float()
            no_regime = (1.0 - ia - ib - ii - iii).clamp(0.0, 1.0)
            sc = self.REGIME_SCALES.to(lst_feats.device)
            return (ia + no_regime)*sc[0] + ib*sc[1] + ii*sc[2] + iii*sc[3]
        return torch.ones(lst_feats.shape[0], device=lst_feats.device)

    def _confidence_gate(self, cm_approx, gii_norm, phase_tr):
        beta  = self.GATE_BETA.to(cm_approx.device)
        logit = beta[0] + beta[1]*cm_approx + beta[2]*gii_norm + beta[3]*phase_tr
        return torch.sigmoid(logit)

    def forward(self, lst_feats, tilt_feats, res_feats,
                er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx):
        res_scale = torch.exp(self.log_res_scale)

        # Shared trunk over concatenated all features
        all_feats = torch.cat([lst_feats, tilt_feats, res_feats], dim=-1)
        t = self.trunk(all_feats)  # (B, trunk//2)

        # ── LST head ──────────────────────────────────────────────────────────
        lst_in  = torch.cat([t, lst_feats], dim=-1)
        is_ib   = (lst_feats[:, regime_idx[1]] > 0.0).float() if len(regime_idx) >= 2 else torch.zeros(lst_feats.shape[0], device=lst_feats.device)
        h_lst   = (1 - is_ib) * self.lst_head(lst_in).squeeze(-1) + is_ib * self.lst_head_ib(lst_in).squeeze(-1)
        r_scale   = self._regime_scale(lst_feats, regime_idx)
        delta_lst = r_scale * F.softplus(h_lst)

        # ── Tilt head ─────────────────────────────────────────────────────────
        tilt_in   = torch.cat([t, tilt_feats], dim=-1)
        delta_tilt = -F.softplus(self.tilt_head(tilt_in).squeeze(-1))

        # ── Confidence gate ───────────────────────────────────────────────────
        sigma_conf = self._confidence_gate(cm_approx, gii_norm, phase_tr)

        # ── Residual head (stop-gradient physics context) ─────────────────────
        er_phys_ctx = (er_cm + delta_lst.detach() + delta_tilt.detach()) / 143.0
        res_in       = torch.cat([t, res_feats, er_phys_ctx.unsqueeze(-1)], dim=-1)
        h_res        = self.res_head(res_in).squeeze(-1)
        delta_res    = sigma_conf * torch.tanh(h_res) * res_scale

        # ── Assembly ──────────────────────────────────────────────────────────
        er_physics  = (er_cm + delta_lst + delta_tilt + delta_res).clamp(1.0, 600.0)
        er_fallback = (self.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
        er_pred     = has_cm * er_physics + (1.0 - has_cm) * er_fallback

        total_delta  = (delta_lst.abs() + delta_tilt.abs() + delta_res.abs()).clamp(min=1e-6)
        physics_frac = (delta_lst.abs() + delta_tilt.abs()) / total_delta

        return {
            "pred":         er_pred,
            "er_cm":        er_cm,
            "delta_lst":    delta_lst,
            "delta_tilt":   delta_tilt,
            "delta_res":    delta_res,
            "sigma_conf":   sigma_conf,
            "physics_frac": physics_frac,
            "res_scale":    res_scale,
        }


# ── CMLTRPCNN ──────────────────────────────────────────────────────────────
class _ResBlock(nn.Module):
    """Pre-norm residual block: LN → Linear(dim→dim) → SiLU → Dropout → Linear(dim→dim) + skip.
    No hidden expansion — keeps param count proportional to dim, not dim²."""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CMLTRPCNN(nn.Module):
    """
    CM-LTR Physics-Constrained Neural Network v7.1.

    Improvements over v7:
      1. ResNet trunk — 3 pre-norm residual blocks; depth without vanishing gradients
      2. Wider trunk (256) and wider heads (96/96/192)
    Training add-ons (script 29): SWA + Mixup alpha=0.2
    Physics constraints identical to v7.
    """

    REGIME_SCALES = torch.tensor([1.0, 1.5, 0.9, 0.7])
    GATE_BETA     = torch.tensor([-2.0, 3.0, 2.0, 2.0])

    def __init__(self, n_lst, n_tilt, n_res,
                 trunk_hidden=256, n_trunk_blocks=3,
                 lst_hidden=96, tilt_hidden=96, res_hidden=192,
                 residual_scale=80.0, dropout=0.15):
        super().__init__()
        n_feat = n_lst + n_tilt + n_res
        t_out  = trunk_hidden // 2

        self.trunk_in = nn.Sequential(
            nn.Linear(n_feat, trunk_hidden), nn.LayerNorm(trunk_hidden),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.trunk_blocks = nn.Sequential(
            *[_ResBlock(trunk_hidden, dropout) for _ in range(n_trunk_blocks)]
        )
        self.trunk_proj = nn.Sequential(nn.Linear(trunk_hidden, t_out), nn.SiLU())

        self.lst_head = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden // 2), nn.SiLU(),
            nn.Linear(lst_hidden // 2, 1),
        )
        self.lst_head_ib = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden), nn.SiLU(),
            nn.Linear(lst_hidden, 1),
        )
        self.tilt_head = nn.Sequential(
            nn.Linear(t_out + n_tilt, tilt_hidden), nn.LayerNorm(tilt_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tilt_hidden, tilt_hidden // 2), nn.SiLU(),
            nn.Linear(tilt_hidden // 2, 1),
        )
        self.res_head = nn.Sequential(
            nn.Linear(t_out + n_res + 1, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden // 2), nn.SiLU(),
            nn.Linear(res_hidden // 2, 1),
        )
        self.log_res_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale))))
        self.fallback_mlp = nn.Sequential(
            nn.Linear(n_res, res_hidden), nn.SiLU(),
            nn.Linear(res_hidden, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _regime_scale(self, lst_feats, regime_indices):
        if len(regime_indices) == 4:
            ia  = (lst_feats[:, regime_indices[0]] > 0.0).float()
            ib  = (lst_feats[:, regime_indices[1]] > 0.0).float()
            ii  = (lst_feats[:, regime_indices[2]] > 0.0).float()
            iii = (lst_feats[:, regime_indices[3]] > 0.0).float()
            no  = (1.0 - ia - ib - ii - iii).clamp(0.0, 1.0)
            sc  = self.REGIME_SCALES.to(lst_feats.device)
            return (ia + no)*sc[0] + ib*sc[1] + ii*sc[2] + iii*sc[3]
        return torch.ones(lst_feats.shape[0], device=lst_feats.device)

    def _confidence_gate(self, cm_approx, gii_norm, phase_tr):
        beta = self.GATE_BETA.to(cm_approx.device)
        return torch.sigmoid(beta[0] + beta[1]*cm_approx + beta[2]*gii_norm + beta[3]*phase_tr)

    def _trunk(self, lst_feats, tilt_feats, res_feats):
        x = torch.cat([lst_feats, tilt_feats, res_feats], dim=-1)
        return self.trunk_proj(self.trunk_blocks(self.trunk_in(x)))

    def forward(self, lst_feats, tilt_feats, res_feats,
                er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx):
        res_scale = torch.exp(self.log_res_scale)
        t = self._trunk(lst_feats, tilt_feats, res_feats)

        lst_in = torch.cat([t, lst_feats], dim=-1)
        is_ib  = (lst_feats[:, regime_idx[1]] > 0.0).float() if len(regime_idx) >= 2 \
                 else torch.zeros(lst_feats.shape[0], device=lst_feats.device)
        h_lst     = (1-is_ib)*self.lst_head(lst_in).squeeze(-1) + \
                    is_ib*self.lst_head_ib(lst_in).squeeze(-1)
        delta_lst  = self._regime_scale(lst_feats, regime_idx) * F.softplus(h_lst)
        delta_tilt = -F.softplus(self.tilt_head(torch.cat([t, tilt_feats], dim=-1)).squeeze(-1))
        sigma_conf = self._confidence_gate(cm_approx, gii_norm, phase_tr)

        ctx       = (er_cm + delta_lst.detach() + delta_tilt.detach()) / 143.0
        res_in    = torch.cat([t, res_feats, ctx.unsqueeze(-1)], dim=-1)
        delta_res = sigma_conf * torch.tanh(self.res_head(res_in).squeeze(-1)) * res_scale

        er_physics  = (er_cm + delta_lst + delta_tilt + delta_res).clamp(1.0, 600.0)
        er_fallback = (self.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
        er_pred     = has_cm * er_physics + (1.0 - has_cm) * er_fallback

        total_delta  = (delta_lst.abs() + delta_tilt.abs() + delta_res.abs()).clamp(min=1e-6)
        physics_frac = (delta_lst.abs() + delta_tilt.abs()) / total_delta

        return {
            "pred":         er_pred,
            "er_cm":        er_cm,
            "delta_lst":    delta_lst,
            "delta_tilt":   delta_tilt,
            "delta_res":    delta_res,
            "sigma_conf":   sigma_conf,
            "physics_frac": physics_frac,
            "res_scale":    res_scale,
        }


class CMLTRPCNNv78(nn.Module):
    """
    CM-LTR Physics-Constrained Neural Network v7.8 — Unified multiplicative LST.

    Key change from v71/v77:
      v71/v77:  delta_lst = regime_scale * softplus(h_lst)        [absolute add-on]
      v78:      f_lst     = 1 + regime_scale * softplus(h_lst)    [multiplicative factor]
                delta_lst = er_cm * (f_lst - 1)                   [scales with CM magnitude]

    f_LST is now a direct model output, matching the LST relation physically.
    All other architecture, constraints, and hyperparameters identical to v71.
    """

    REGIME_SCALES = torch.tensor([1.0, 1.5, 0.9, 0.7])
    GATE_BETA     = torch.tensor([-2.0, 3.0, 2.0, 2.0])

    def __init__(self, n_lst, n_tilt, n_res,
                 trunk_hidden=256, n_trunk_blocks=3,
                 lst_hidden=96, tilt_hidden=96, res_hidden=192,
                 residual_scale=80.0, dropout=0.15):
        super().__init__()
        n_feat = n_lst + n_tilt + n_res
        t_out  = trunk_hidden // 2

        self.trunk_in = nn.Sequential(
            nn.Linear(n_feat, trunk_hidden), nn.LayerNorm(trunk_hidden),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.trunk_blocks = nn.Sequential(
            *[_ResBlock(trunk_hidden, dropout) for _ in range(n_trunk_blocks)]
        )
        self.trunk_proj = nn.Sequential(nn.Linear(trunk_hidden, t_out), nn.SiLU())

        self.lst_head = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden // 2), nn.SiLU(),
            nn.Linear(lst_hidden // 2, 1),
        )
        self.lst_head_ib = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden), nn.SiLU(),
            nn.Linear(lst_hidden, 1),
        )
        self.tilt_head = nn.Sequential(
            nn.Linear(t_out + n_tilt, tilt_hidden), nn.LayerNorm(tilt_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tilt_hidden, tilt_hidden // 2), nn.SiLU(),
            nn.Linear(tilt_hidden // 2, 1),
        )
        self.res_head = nn.Sequential(
            nn.Linear(t_out + n_res + 1, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden // 2), nn.SiLU(),
            nn.Linear(res_hidden // 2, 1),
        )
        self.log_res_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale))))
        self.fallback_mlp = nn.Sequential(
            nn.Linear(n_res, res_hidden), nn.SiLU(),
            nn.Linear(res_hidden, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _regime_scale(self, lst_feats, regime_indices):
        if len(regime_indices) == 4:
            ia  = (lst_feats[:, regime_indices[0]] > 0.0).float()
            ib  = (lst_feats[:, regime_indices[1]] > 0.0).float()
            ii  = (lst_feats[:, regime_indices[2]] > 0.0).float()
            iii = (lst_feats[:, regime_indices[3]] > 0.0).float()
            no  = (1.0 - ia - ib - ii - iii).clamp(0.0, 1.0)
            sc  = self.REGIME_SCALES.to(lst_feats.device)
            return (ia + no)*sc[0] + ib*sc[1] + ii*sc[2] + iii*sc[3]
        return torch.ones(lst_feats.shape[0], device=lst_feats.device)

    def _confidence_gate(self, cm_approx, gii_norm, phase_tr):
        beta = self.GATE_BETA.to(cm_approx.device)
        return torch.sigmoid(beta[0] + beta[1]*cm_approx + beta[2]*gii_norm + beta[3]*phase_tr)

    def _trunk(self, lst_feats, tilt_feats, res_feats):
        x = torch.cat([lst_feats, tilt_feats, res_feats], dim=-1)
        return self.trunk_proj(self.trunk_blocks(self.trunk_in(x)))

    def forward(self, lst_feats, tilt_feats, res_feats,
                er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx):
        res_scale = torch.exp(self.log_res_scale)
        t = self._trunk(lst_feats, tilt_feats, res_feats)

        lst_in = torch.cat([t, lst_feats], dim=-1)
        is_ib  = (lst_feats[:, regime_idx[1]] > 0.0).float() if len(regime_idx) >= 2 \
                 else torch.zeros(lst_feats.shape[0], device=lst_feats.device)
        h_lst  = (1-is_ib)*self.lst_head(lst_in).squeeze(-1) + \
                 is_ib*self.lst_head_ib(lst_in).squeeze(-1)

        # Multiplicative LST: f_LST is the direct output; delta_lst scales with CM magnitude
        f_lst      = 1.0 + self._regime_scale(lst_feats, regime_idx) * F.softplus(h_lst)
        delta_lst  = er_cm * (f_lst - 1.0)

        delta_tilt = -F.softplus(self.tilt_head(torch.cat([t, tilt_feats], dim=-1)).squeeze(-1))
        sigma_conf = self._confidence_gate(cm_approx, gii_norm, phase_tr)

        ctx       = (er_cm + delta_lst.detach() + delta_tilt.detach()) / 143.0
        res_in    = torch.cat([t, res_feats, ctx.unsqueeze(-1)], dim=-1)
        delta_res = sigma_conf * torch.tanh(self.res_head(res_in).squeeze(-1)) * res_scale

        er_physics  = (er_cm + delta_lst + delta_tilt + delta_res).clamp(1.0, 600.0)
        er_fallback = (self.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
        er_pred     = has_cm * er_physics + (1.0 - has_cm) * er_fallback

        total_delta  = (delta_lst.abs() + delta_tilt.abs() + delta_res.abs()).clamp(min=1e-6)
        physics_frac = (delta_lst.abs() + delta_tilt.abs()) / total_delta

        return {
            "pred":         er_pred,
            "er_cm":        er_cm,
            "delta_lst":    delta_lst,
            "delta_tilt":   delta_tilt,
            "delta_res":    delta_res,
            "f_lst":        f_lst,
            "sigma_conf":   sigma_conf,
            "physics_frac": physics_frac,
            "res_scale":    res_scale,
        }


class CMLTRPCNNv79(nn.Module):
    """
    CM-LTR Physics-Constrained Neural Network v7.9 — Regime Ia scale fix.

    Change from v71/v77:
      REGIME_SCALES[0] (Ia) : 1.0 → 0.2
      Rationale: Regime Ia = centrosymmetric cubic paraelectrics.
      No soft-mode activity → LST enhancement should be near-zero.
      Previous Ia scale=1.0 (default) caused over-prediction of delta_LST
      for centrosymmetric compositions, explaining SMAPE=17.2%, R²=-0.35.
    All other architecture and hyperparameters identical to v71/v77.
    """

    REGIME_SCALES = torch.tensor([0.2, 1.5, 0.9, 0.7])   # Ia reduced from 1.0
    GATE_BETA     = torch.tensor([-2.0, 3.0, 2.0, 2.0])

    def __init__(self, n_lst, n_tilt, n_res,
                 trunk_hidden=256, n_trunk_blocks=3,
                 lst_hidden=96, tilt_hidden=96, res_hidden=192,
                 residual_scale=80.0, dropout=0.15):
        super().__init__()
        n_feat = n_lst + n_tilt + n_res
        t_out  = trunk_hidden // 2

        self.trunk_in = nn.Sequential(
            nn.Linear(n_feat, trunk_hidden), nn.LayerNorm(trunk_hidden),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.trunk_blocks = nn.Sequential(
            *[_ResBlock(trunk_hidden, dropout) for _ in range(n_trunk_blocks)]
        )
        self.trunk_proj = nn.Sequential(nn.Linear(trunk_hidden, t_out), nn.SiLU())

        self.lst_head = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden // 2), nn.SiLU(),
            nn.Linear(lst_hidden // 2, 1),
        )
        self.lst_head_ib = nn.Sequential(
            nn.Linear(t_out + n_lst, lst_hidden), nn.LayerNorm(lst_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lst_hidden, lst_hidden), nn.SiLU(),
            nn.Linear(lst_hidden, 1),
        )
        self.tilt_head = nn.Sequential(
            nn.Linear(t_out + n_tilt, tilt_hidden), nn.LayerNorm(tilt_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tilt_hidden, tilt_hidden // 2), nn.SiLU(),
            nn.Linear(tilt_hidden // 2, 1),
        )
        self.res_head = nn.Sequential(
            nn.Linear(t_out + n_res + 1, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden), nn.LayerNorm(res_hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(res_hidden, res_hidden // 2), nn.SiLU(),
            nn.Linear(res_hidden // 2, 1),
        )
        self.log_res_scale = nn.Parameter(torch.log(torch.tensor(float(residual_scale))))
        self.fallback_mlp = nn.Sequential(
            nn.Linear(n_res, res_hidden), nn.SiLU(),
            nn.Linear(res_hidden, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _regime_scale(self, lst_feats, regime_indices):
        if len(regime_indices) == 4:
            ia  = (lst_feats[:, regime_indices[0]] > 0.0).float()
            ib  = (lst_feats[:, regime_indices[1]] > 0.0).float()
            ii  = (lst_feats[:, regime_indices[2]] > 0.0).float()
            iii = (lst_feats[:, regime_indices[3]] > 0.0).float()
            no  = (1.0 - ia - ib - ii - iii).clamp(0.0, 1.0)
            sc  = self.REGIME_SCALES.to(lst_feats.device)
            return (ia + no)*sc[0] + ib*sc[1] + ii*sc[2] + iii*sc[3]
        return torch.ones(lst_feats.shape[0], device=lst_feats.device)

    def _confidence_gate(self, cm_approx, gii_norm, phase_tr):
        beta = self.GATE_BETA.to(cm_approx.device)
        return torch.sigmoid(beta[0] + beta[1]*cm_approx + beta[2]*gii_norm + beta[3]*phase_tr)

    def _trunk(self, lst_feats, tilt_feats, res_feats):
        x = torch.cat([lst_feats, tilt_feats, res_feats], dim=-1)
        return self.trunk_proj(self.trunk_blocks(self.trunk_in(x)))

    def forward(self, lst_feats, tilt_feats, res_feats,
                er_cm, has_cm, cm_approx, gii_norm, phase_tr, regime_idx):
        res_scale = torch.exp(self.log_res_scale)
        t = self._trunk(lst_feats, tilt_feats, res_feats)

        lst_in = torch.cat([t, lst_feats], dim=-1)
        is_ib  = (lst_feats[:, regime_idx[1]] > 0.0).float() if len(regime_idx) >= 2 \
                 else torch.zeros(lst_feats.shape[0], device=lst_feats.device)
        h_lst     = (1-is_ib)*self.lst_head(lst_in).squeeze(-1) + \
                    is_ib*self.lst_head_ib(lst_in).squeeze(-1)
        delta_lst  = self._regime_scale(lst_feats, regime_idx) * F.softplus(h_lst)
        delta_tilt = -F.softplus(self.tilt_head(torch.cat([t, tilt_feats], dim=-1)).squeeze(-1))
        sigma_conf = self._confidence_gate(cm_approx, gii_norm, phase_tr)

        ctx       = (er_cm + delta_lst.detach() + delta_tilt.detach()) / 143.0
        res_in    = torch.cat([t, res_feats, ctx.unsqueeze(-1)], dim=-1)
        delta_res = sigma_conf * torch.tanh(self.res_head(res_in).squeeze(-1)) * res_scale

        er_physics  = (er_cm + delta_lst + delta_tilt + delta_res).clamp(1.0, 600.0)
        er_fallback = (self.fallback_mlp(res_feats).squeeze(-1) + 8.0).clamp(1.0, 600.0)
        er_pred     = has_cm * er_physics + (1.0 - has_cm) * er_fallback

        total_delta  = (delta_lst.abs() + delta_tilt.abs() + delta_res.abs()).clamp(min=1e-6)
        physics_frac = (delta_lst.abs() + delta_tilt.abs()) / total_delta

        return {
            "pred":         er_pred,
            "er_cm":        er_cm,
            "delta_lst":    delta_lst,
            "delta_tilt":   delta_tilt,
            "delta_res":    delta_res,
            "sigma_conf":   sigma_conf,
            "physics_frac": physics_frac,
            "res_scale":    res_scale,
        }

# ── Backward-compatibility alias ────────────────────────────────────────────
# Architecture froze at v7.1; the published model is CMLTRPCNN v7.7 (checkpoint
# cmltrv77_final.pt). Old code/pickles that reference CMLTRPCNNv71 keep working.
CMLTRPCNNv71 = CMLTRPCNN
