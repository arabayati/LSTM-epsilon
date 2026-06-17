#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Oracle-Q0 inference diagnostic for LSTM-epsilon.

Purpose
-------
This script tests whether the learned Q0 / q_base head is limiting performance.

It loads a trained LSTM-epsilon model, extracts learned epsilon, AET, alpha,
lp, gamma, and q_base, then recomputes the recession trajectory manually.

The important correction:
    q_base is component-wise: [Time, Batch, N_MUL].
    The model does not use one scalar Q0 internally.
    Therefore the best oracle-Q0 mode preserves the predicted component-wise
    q_base pattern and rescales it so the ensemble mean equals observed Q.

Modes
-----
Q0_ORACLE_MODE = "predicted"
    Uses predicted q_base exactly. This is a sanity-check mode.

Q0_ORACLE_MODE = "observed_scalar"
    Gives every component the same observed Q. This is NOT recommended as
    the main diagnostic because it destroys the component-wise ensemble pattern.

Q0_ORACLE_MODE = "observed_rescaled"
    Recommended. Preserves component-wise predicted q_base ratios, but rescales
    each q_base vector so its mean equals observed Q.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from custom_loader import LoaderConfig, load_cluster_data
from model import EpsilonStateResetModel

warnings.filterwarnings("ignore")

# ============================================================
# 1. USER OPTIONS
# ============================================================

EPOCH_TO_LOAD = 120
SAVE_SIMULATIONS = True

# Recommended diagnostic mode:
#   "predicted"          = sanity check using model-predicted q_base
#   "observed_scalar"    = old/simple oracle; all components get same observed Q
#   "observed_rescaled"  = recommended oracle; mean Q0 is observed, component pattern preserved
Q0_ORACLE_MODE = "observed_rescaled"

# For your old results_v3_with_10_ODE_ model, use "first_eval".
# For the newer last-hidden-state model, use "last_eval".
STATIC_STATE_MODE = "first_eval"

SOURCE_RESULTS_ROOT = "/project/6107743/ARA_A/LSTM_Epsilon_code/results_v3_with_10_ODE_"
ORACLE_RESULTS_ROOT = "/project/6107743/ARA_A/LSTM_Epsilon_code/results_v3_with_10_ODE_oracleQ0"


# ============================================================
# 2. HYPERPARAMETERS & CONFIGURATION
# ============================================================

slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
CLUSTER_ID = int(slurm_id) if slurm_id is not None else 1

print(f"[Oracle-Q0 Inference Config] CLUSTER_ID = {CLUSTER_ID}")
print(f"[Oracle-Q0 Inference Config] EPOCH_TO_LOAD = {EPOCH_TO_LOAD}")
print(f"[Oracle-Q0 Inference Config] SAVE_SIMULATIONS = {SAVE_SIMULATIONS}")
print(f"[Oracle-Q0 Inference Config] STATIC_STATE_MODE = {STATIC_STATE_MODE}")
print(f"[Oracle-Q0 Inference Config] Q0_ORACLE_MODE = {Q0_ORACLE_MODE}")

FORCING_ROOT = "/project/6107743/ARA_A/data/clustered_forcing_data"
METADATA_CSV = "/project/6107743/ARA_A/attributes/merged_catchments_metadata.csv"
SNOW_MASK_CSV = "/project/6107743/ARA_A/data/monthly_climatology_q70.csv"
AET_BOUNDS_CSV = "/project/6107743/ARA_A/data/lp_gamma_fit_summary_with_recession.csv"

WEIGHTS_DIR = os.path.join(SOURCE_RESULTS_ROOT, f"cluster_{CLUSTER_ID}", "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, f"Epsilon_Model_Epoch_{EPOCH_TO_LOAD}.pth")

RESULT_ROOT = os.path.join(ORACLE_RESULTS_ROOT, f"cluster_{CLUSTER_ID}", "accuracy_metrics")
SIM_DIR = os.path.join(RESULT_ROOT, "simulations")

os.makedirs(RESULT_ROOT, exist_ok=True)
if SAVE_SIMULATIONS:
    os.makedirs(SIM_DIR, exist_ok=True)

DATE_COL = "date"
DYNAMIC_COLUMNS = ["precipitation_mmd", "temperature_C", "pet_mmd", "SM_%"]
TARGET_COL = "streamflow_mmd"

STATIC_COLUMNS = [
    "elevation_mean_m", "mean_slope_degree", "Median_DepthToBedrock_cm",
    "Prec_mm", "Temp_C", "PET_mm", "AET_mm", "P_AET_mm", "Aridity", "SF",
    "max_soil_moisture", "Porosity", "Seasonality_of_Moisture_Index",
    "low_high_ratio", "wet_days_ratio_1mm", "wet_days_ratio_5mm",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"
]

BUFFTIME = 365
TRAIN_FRAC = 0.6
N_MUL = 10
HIDDEN_SIZE = 256
EPS = 1e-6

MIN_RECESSION_DAYS = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. METRICS
# ============================================================

def safe_nan_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    x1, y1 = x[mask], y[mask]

    if np.std(x1) < 1e-12 or np.std(y1) < 1e-12:
        return np.nan

    return float(np.corrcoef(x1, y1)[0, 1])


def calc_nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan

    o, s = obs[mask], sim[mask]
    denom = np.sum((o - np.mean(o)) ** 2)

    return float(1.0 - np.sum((s - o) ** 2) / denom) if denom > 0 else np.nan


def calc_kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2:
        return np.nan

    o, s = obs[mask], sim[mask]

    mo, ms = np.mean(o), np.mean(s)
    so, ss = np.std(o), np.std(s)

    if abs(mo) < 1e-12 or abs(so) < 1e-12:
        return np.nan

    r = safe_nan_corrcoef(o, s)
    if not np.isfinite(r):
        return np.nan

    alpha = ss / so
    beta = ms / mo

    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def calc_r2(obs, sim):
    r = safe_nan_corrcoef(obs, sim)
    return float(r ** 2) if np.isfinite(r) else np.nan


def calc_mae(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(np.abs(sim[mask] - obs[mask]))) if mask.sum() > 0 else np.nan


def calc_rmse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((sim[mask] - obs[mask]) ** 2))) if mask.sum() > 0 else np.nan


def calc_bias(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(sim[mask] - obs[mask])) if mask.sum() > 0 else np.nan


def compute_metric_row(obs: np.ndarray, sim: np.ndarray, prefix: str) -> dict:
    valid = np.isfinite(obs) & np.isfinite(sim)

    return {
        f"{prefix}_NSE": calc_nse(obs, sim),
        f"{prefix}_KGE": calc_kge(obs, sim),
        f"{prefix}_R2": calc_r2(obs, sim),
        f"{prefix}_MAE": calc_mae(obs, sim),
        f"{prefix}_RMSE": calc_rmse(obs, sim),
        f"{prefix}_Bias": calc_bias(obs, sim),
        f"{prefix}_N": int(valid.sum()),
    }


# ============================================================
# 4. MODEL PARAMETER EXTRACTION
# ============================================================

def extract_model_parameters_without_qhat(
    model,
    z_seq,
    pet_seq,
    sm_seq,
    bounds,
    bufftime,
    static_state_mode="first_eval",
):
    """
    Extract eps, q_base, AET, alpha, lp, gamma directly from the model modules.

    This intentionally does not use model_out["q_hat"], because q_hat is generated
    internally using predicted q_base. We need to recompute q_hat with oracle-Q0.
    """

    lstm_out_full, _ = model.lstm(z_seq)
    lstm_out_full = model.dropout(lstm_out_full)

    if lstm_out_full.shape[0] <= bufftime:
        raise RuntimeError(
            f"Sequence too short for bufftime: "
            f"seq_len={lstm_out_full.shape[0]}, bufftime={bufftime}"
        )

    lstm_eval = lstm_out_full[bufftime:, :, :]
    pet_eval = pet_seq[bufftime:, :, :]
    sm_eval = sm_seq[bufftime:, :, :]

    time_len, batch_size, _ = lstm_eval.shape

    if batch_size != 1:
        raise RuntimeError(
            f"This script expects one catchment at a time. Got batch_size={batch_size}"
        )

    # Dynamic epsilon
    raw_eps = model.eps_head(lstm_eval)
    eps_t = F.softplus(raw_eps)  # [T, 1, N_MUL]

    # Component-wise predicted q_base
    raw_qbase = model.peak_head(lstm_eval)
    q_base_t = F.softplus(raw_qbase)  # [T, 1, N_MUL]

    # Static alpha/lp/gamma hidden state
    if static_state_mode == "first_eval":
        h_static = lstm_eval[0, :, :]
    elif static_state_mode == "last_eval":
        h_static = lstm_eval[-1, :, :]
    elif static_state_mode == "warmup_end":
        h_static = lstm_out_full[bufftime - 1, :, :]
    else:
        raise ValueError(
            "STATIC_STATE_MODE must be one of: "
            "'first_eval', 'last_eval', 'warmup_end'"
        )

    static_raw = model.static_head(h_static).view(batch_size, model.n_mul, 3)

    a_min, a_max = bounds[:, 0:1], bounds[:, 1:2]
    l_min, l_max = bounds[:, 2:3], bounds[:, 3:4]
    g_min, g_max = bounds[:, 4:5], bounds[:, 5:6]

    alpha = a_min + (a_max - a_min) * static_raw[:, :, 0]  # [1, N_MUL]
    lp = l_min + (l_max - l_min) * static_raw[:, :, 1]     # [1, N_MUL]
    gamma = g_min + (g_max - g_min) * static_raw[:, :, 2]  # [1, N_MUL]

    # AET calculation, same as model.py
    sm_term = torch.clamp(sm_eval / (lp.unsqueeze(0) + EPS), min=EPS)
    aet_t = pet_eval * torch.pow(sm_term, gamma.unsqueeze(0))
    aet_t = torch.clamp(aet_t, max=pet_eval)

    return {
        "eps": eps_t,
        "q_base": q_base_t,
        "aet": aet_t,
        "alpha": alpha,
        "lp": lp,
        "gamma": gamma,
        "time_len": time_len,
    }


# ============================================================
# 5. ORACLE-Q0 INTEGRATION
# ============================================================

def make_q0_vector(q_base_vec: np.ndarray, q_obs_scalar: float, mode: str) -> np.ndarray:
    """
    Create component-wise Q0 vector.

    q_base_vec:
        [N_MUL], predicted component-wise q_base from the model.

    q_obs_scalar:
        observed Q used as oracle mean state.

    mode:
        "predicted":
            returns q_base_vec unchanged.

        "observed_scalar":
            returns [Qobs, Qobs, ..., Qobs].

        "observed_rescaled":
            preserves q_base_vec component-wise pattern, but rescales it so
            mean(output) = Qobs.
    """

    q_base_vec = np.asarray(q_base_vec, dtype=np.float64)
    q_base_vec = np.maximum(q_base_vec, EPS)

    q_obs_scalar = float(q_obs_scalar)
    if not np.isfinite(q_obs_scalar):
        q_obs_scalar = EPS
    q_obs_scalar = max(q_obs_scalar, EPS)

    if mode == "predicted":
        return q_base_vec.copy()

    if mode == "observed_scalar":
        return np.full_like(q_base_vec, q_obs_scalar, dtype=np.float64)

    if mode == "observed_rescaled":
        base_mean = np.mean(q_base_vec)

        if not np.isfinite(base_mean) or base_mean <= EPS:
            return np.full_like(q_base_vec, q_obs_scalar, dtype=np.float64)

        q0_vec = q_base_vec * (q_obs_scalar / base_mean)
        return np.maximum(q0_vec, EPS)

    raise ValueError(
        "Q0_ORACLE_MODE must be one of: "
        "'predicted', 'observed_scalar', 'observed_rescaled'"
    )


def integrate_oracle_q0_numpy(
    eps_ts: np.ndarray,
    aet_ts: np.ndarray,
    alpha_opt: np.ndarray,
    q_base_ts: np.ndarray,
    obs_q_eval: np.ndarray,
    rec_mask_eval: np.ndarray,
    start_mask_eval: np.ndarray,
    q0_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recompute Q trajectory using model eps/AET/alpha but with controlled Q0.

    q0_mode="observed_rescaled" is the recommended oracle.
    """

    time_len, n_mul = eps_ts.shape

    if aet_ts.shape != (time_len, n_mul):
        raise RuntimeError(f"aet_ts shape mismatch: {aet_ts.shape}, expected {(time_len, n_mul)}")

    if q_base_ts.shape != (time_len, n_mul):
        raise RuntimeError(f"q_base_ts shape mismatch: {q_base_ts.shape}, expected {(time_len, n_mul)}")

    if alpha_opt.shape[0] != n_mul:
        raise RuntimeError(f"alpha shape mismatch: {alpha_opt.shape}, expected ({n_mul},)")

    if obs_q_eval.shape[0] != time_len:
        raise RuntimeError(f"obs_q_eval length mismatch: {obs_q_eval.shape[0]}, expected {time_len}")

    if rec_mask_eval.shape[0] != time_len:
        raise RuntimeError(f"rec_mask_eval length mismatch: {rec_mask_eval.shape[0]}, expected {time_len}")

    if start_mask_eval.shape[0] != time_len:
        raise RuntimeError(f"start_mask_eval length mismatch: {start_mask_eval.shape[0]}, expected {time_len}")

    obs_q_safe = np.asarray(obs_q_eval, dtype=np.float64).copy()
    obs_q_safe[~np.isfinite(obs_q_safe)] = EPS
    obs_q_safe = np.maximum(obs_q_safe, EPS)

    alpha_opt = np.asarray(alpha_opt, dtype=np.float64)

    q_components = np.full((time_len, n_mul), np.nan, dtype=np.float64)

    # Original model initializes q_prev = q_base_t[0].
    # Here we use the selected Q0 mode.
    q_prev = make_q0_vector(
        q_base_vec=q_base_ts[0, :],
        q_obs_scalar=obs_q_safe[0],
        mode=q0_mode,
    )

    for t in range(time_len):
        # Original model:
        # reset_val = q_base_t[t-1] if t > 0 else q_base_t[0]
        reset_idx = t - 1 if t > 0 else 0

        reset_val = make_q0_vector(
            q_base_vec=q_base_ts[reset_idx, :],
            q_obs_scalar=obs_q_safe[reset_idx],
            mode=q0_mode,
        )

        if start_mask_eval[t]:
            q_curr = reset_val
        else:
            q_curr = q_prev

        b_t = eps_ts[t, :].astype(np.float64)
        aet_t = aet_ts[t, :].astype(np.float64)

        b_t = np.maximum(b_t, EPS)
        aet_t = np.maximum(aet_t, 0.0)

        a_t = b_t * (alpha_opt * aet_t)

        # Formula A: exact integral with AET
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denom = (b_t * q_curr + a_t) * np.exp(a_t) - (b_t * q_curr)

        denom = np.where(np.isfinite(denom), denom, np.inf)
        denom = np.maximum(denom, EPS)

        q_next_aet = (a_t * q_curr) / denom

        # Formula B: zero-AET limit
        q_next_zero_aet = q_curr / (1.0 + b_t * q_curr)

        q_next = np.where(a_t < 1e-6, q_next_zero_aet, q_next_aet)
        q_next = np.where(np.isfinite(q_next), q_next, EPS)
        q_next = np.maximum(q_next, EPS)

        # Original model:
        # q_prev = q_next if rec_mask[t] else q_base_t[t]
        if rec_mask_eval[t]:
            q_prev = q_next
        else:
            q_prev = make_q0_vector(
                q_base_vec=q_base_ts[t, :],
                q_obs_scalar=obs_q_safe[t],
                mode=q0_mode,
            )

        q_components[t, :] = q_prev

    q_hat = np.mean(q_components, axis=1, keepdims=True)

    return q_hat.astype(np.float32), q_components.astype(np.float32)


# ============================================================
# 6. MAIN INFERENCE LOOP
# ============================================================

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Saving outputs to: {RESULT_ROOT}")

    if SAVE_SIMULATIONS:
        print(f"Saving simulation files to: {SIM_DIR}")
    else:
        print("SAVE_SIMULATIONS=False, simulation CSVs will not be written.")

    loader_cfg = LoaderConfig(
        cluster_id=CLUSTER_ID,
        forcing_root=FORCING_ROOT,
        metadata_csv=METADATA_CSV,
        static_columns=STATIC_COLUMNS,
        dynamic_columns=DYNAMIC_COLUMNS,
        target_col=TARGET_COL,
        date_col=DATE_COL,
        warmup_years=1,
        train_frac=TRAIN_FRAC,
        snow_mask_csv=SNOW_MASK_CSV,
        snow_free_threshold=25.0,
        aet_bounds_csv=AET_BOUNDS_CSV,
        global_bounds={
            "alpha": (0.0, 1.0),
            "lp": (0.1, 1.0),
            "gamma": (0.1, 5.0),
        },
    )

    print("\nLoading Cluster Data...")
    dataset = load_cluster_data(loader_cfg)
    gridcodes = dataset["gridcodes"]

    n_inv = len(DYNAMIC_COLUMNS) + len(STATIC_COLUMNS)

    model = EpsilonStateResetModel(
        input_dim=n_inv,
        hidden_size=HIDDEN_SIZE,
        n_mul=N_MUL,
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    summary_results = []

    print(f"\nStarting Oracle-Q0 full-sequence inference for {len(gridcodes)} catchments...")

    with torch.no_grad():
        for gc in gridcodes:
            b = dataset["basins"][gc]
            nt = b["nt"]

            if nt <= BUFFTIME:
                print(f"[SKIPPED] gc={gc}: nt={nt} <= BUFFTIME={BUFFTIME}")
                continue

            time_len = nt - BUFFTIME

            rec_mask_eval_np = b["rec_mask"][BUFFTIME:] > 0.5
            start_mask_eval_np = b["start_mask"][BUFFTIME:] > 0.5

            n_valid_recessions = int(np.sum(rec_mask_eval_np))

            if n_valid_recessions < MIN_RECESSION_DAYS:
                print(
                    f"[SKIPPED] gc={gc}: Only {n_valid_recessions} valid recession days "
                    f"(Requires {MIN_RECESSION_DAYS})."
                )
                continue

            # --- Observed AET from raw CSV ---
            obs_aet_full = np.full(nt, np.nan, dtype=np.float32)

            if "forcing_csv" in b and os.path.exists(b["forcing_csv"]):
                df_raw = pd.read_csv(b["forcing_csv"])

                if "aet_mm" in df_raw.columns:
                    obs_aet_full = df_raw["aet_mm"].to_numpy(dtype=np.float32)
                elif "AET_mm" in df_raw.columns:
                    obs_aet_full = df_raw["AET_mm"].to_numpy(dtype=np.float32)

            # --- Prepare tensors ---
            z_norm = b["z_norm"]
            c_rep = np.repeat(b["c_norm"][None, :], nt, axis=0)
            z_seq = np.concatenate([z_norm, c_rep], axis=1)

            z_batch = torch.from_numpy(z_seq).float().unsqueeze(1).to(device)
            pet_batch = torch.from_numpy(b["x_model"][:, 2:3]).float().unsqueeze(1).to(device)
            sm_batch = torch.from_numpy(b["x_model"][:, 3:4]).float().unsqueeze(1).to(device)
            bounds_batch = torch.from_numpy(b["bounds"]).float().unsqueeze(0).to(device)

            # --- Extract model parameters, including component-wise q_base ---
            param_out = extract_model_parameters_without_qhat(
                model=model,
                z_seq=z_batch,
                pet_seq=pet_batch,
                sm_seq=sm_batch,
                bounds=bounds_batch,
                bufftime=BUFFTIME,
                static_state_mode=STATIC_STATE_MODE,
            )

            if param_out["time_len"] != time_len:
                raise RuntimeError(
                    f"time_len mismatch for gc={gc}: "
                    f"param_out={param_out['time_len']}, expected={time_len}"
                )

            eps_ts = param_out["eps"].cpu().numpy().reshape(time_len, N_MUL)
            q_base_ts = param_out["q_base"].cpu().numpy().reshape(time_len, N_MUL)
            aet_ts = param_out["aet"].cpu().numpy().reshape(time_len, N_MUL)

            alpha_opt = param_out["alpha"].cpu().numpy().reshape(N_MUL)
            lp_opt = param_out["lp"].cpu().numpy().reshape(N_MUL)
            gamma_opt = param_out["gamma"].cpu().numpy().reshape(N_MUL)

            obs_q_full = b["y_raw"].flatten().astype(np.float32)
            obs_q_eval = obs_q_full[BUFFTIME:]

            # --- Oracle-Q0 integration ---
            q_hat, q_comps = integrate_oracle_q0_numpy(
                eps_ts=eps_ts,
                aet_ts=aet_ts,
                alpha_opt=alpha_opt,
                q_base_ts=q_base_ts,
                obs_q_eval=obs_q_eval,
                rec_mask_eval=rec_mask_eval_np,
                start_mask_eval=start_mask_eval_np,
                q0_mode=Q0_ORACLE_MODE,
            )

            abs_idx = np.arange(BUFFTIME, nt)[rec_mask_eval_np]

            sim_q_final = np.full(nt, np.nan, dtype=np.float32)
            sim_q_comps = np.full((nt, N_MUL), np.nan, dtype=np.float32)

            eps_mean_final = np.full(nt, np.nan, dtype=np.float32)
            eps_eff_final = np.full(nt, np.nan, dtype=np.float32)
            eps_comps_final = np.full((nt, N_MUL), np.nan, dtype=np.float32)

            qbase_mean_final = np.full(nt, np.nan, dtype=np.float32)
            qbase_comps_final = np.full((nt, N_MUL), np.nan, dtype=np.float32)

            aet_mean_final = np.full(nt, np.nan, dtype=np.float32)
            aet_comps_final = np.full((nt, N_MUL), np.nan, dtype=np.float32)

            alpha_eff_final = np.full(nt, np.nan, dtype=np.float32)

            q_val = q_comps[rec_mask_eval_np, :]
            eps_val = eps_ts[rec_mask_eval_np, :]
            qbase_val = q_base_ts[rec_mask_eval_np, :]
            aet_val = aet_ts[rec_mask_eval_np, :]
            q_hat_val = q_hat[rec_mask_eval_np, 0].flatten()

            alpha_val = alpha_opt[np.newaxis, :]

            eps_mean = np.mean(eps_val, axis=1).flatten()
            qbase_mean = np.mean(qbase_val, axis=1).flatten()
            aet_eff = np.mean(aet_val, axis=1).flatten()

            e_eps_q2 = np.mean(eps_val * (q_val ** 2), axis=1).flatten()
            eps_eff = e_eps_q2 / (q_hat_val ** 2 + EPS)

            e_eps_alpha_aet_q = np.mean(eps_val * alpha_val * aet_val * q_val, axis=1).flatten()
            denom_alpha = (eps_eff * aet_eff * q_hat_val) + EPS
            alpha_eff = e_eps_alpha_aet_q / denom_alpha

            sim_q_final[abs_idx] = q_hat_val
            sim_q_comps[abs_idx, :] = q_val

            eps_mean_final[abs_idx] = eps_mean
            eps_eff_final[abs_idx] = eps_eff
            eps_comps_final[abs_idx, :] = eps_val

            qbase_mean_final[abs_idx] = qbase_mean
            qbase_comps_final[abs_idx, :] = qbase_val

            aet_mean_final[abs_idx] = aet_eff
            aet_comps_final[abs_idx, :] = aet_val

            alpha_eff_final[abs_idx] = alpha_eff

            df_sim = pd.DataFrame({
                "date": pd.to_datetime(b["dates"]).strftime("%Y-%m-%d"),
                "Split": b["split_labels"],
                "observed_Q_mmd": obs_q_full,
                "simulated_Q_mmd": sim_q_final,
                "observed_AET_mm": obs_aet_full,
                "simulated_AET_mm": aet_mean_final,
                "epsilon_mean": eps_mean_final,
                "epsilon_effective": eps_eff_final,
                "qbase_mean_predicted": qbase_mean_final,
                "alpha_effective": alpha_eff_final,
                "q0_source": Q0_ORACLE_MODE,
                "epoch_loaded": EPOCH_TO_LOAD,
                "static_state_mode": STATIC_STATE_MODE,
            })

            for m in range(N_MUL):
                df_sim[f"sim_Q_comp_{m+1:02d}"] = sim_q_comps[:, m]
                df_sim[f"eps_comp_{m+1:02d}"] = eps_comps_final[:, m]
                df_sim[f"qbase_comp_{m+1:02d}"] = qbase_comps_final[:, m]
                df_sim[f"aet_comp_{m+1:02d}"] = aet_comps_final[:, m]

            if SAVE_SIMULATIONS:
                sim_file = os.path.join(SIM_DIR, f"simulation_{gc}.csv")
                df_sim.to_csv(sim_file, index=False)

            train_mask = df_sim["Split"] == "train"
            val_mask = df_sim["Split"] == "val"

            obs_q_tr = df_sim.loc[train_mask, "observed_Q_mmd"].values
            sim_q_tr = df_sim.loc[train_mask, "simulated_Q_mmd"].values

            obs_q_val = df_sim.loc[val_mask, "observed_Q_mmd"].values
            sim_q_val = df_sim.loc[val_mask, "simulated_Q_mmd"].values

            valid_mask = train_mask | val_mask
            obs_aet_all = df_sim.loc[valid_mask, "observed_AET_mm"].values
            sim_aet_all = df_sim.loc[valid_mask, "simulated_AET_mm"].values

            row = {
                "gridcode": gc,
                "cluster_id": CLUSTER_ID,
                "epoch_loaded": EPOCH_TO_LOAD,
                "q0_source": Q0_ORACLE_MODE,
                "static_state_mode": STATIC_STATE_MODE,
                "n_total_days": nt,
                "n_warmup_days": int((df_sim["Split"] == "warmup").sum()),
                "n_train_days": int(train_mask.sum()),
                "n_val_days": int(val_mask.sum()),
                "n_valid_recessions": int(n_valid_recessions),
            }

            row.update(compute_metric_row(obs_q_tr, sim_q_tr, "Train_Q"))
            row.update(compute_metric_row(obs_q_val, sim_q_val, "Val_Q"))
            row.update(compute_metric_row(obs_aet_all, sim_aet_all, "Total_AET"))

            for m in range(N_MUL):
                row[f"alpha_comp_{m+1:02d}"] = float(alpha_opt[m])
                row[f"lp_comp_{m+1:02d}"] = float(lp_opt[m])
                row[f"gamma_comp_{m+1:02d}"] = float(gamma_opt[m])

            row["alpha_mean_static"] = float(np.mean(alpha_opt))
            row["lp_mean"] = float(np.mean(lp_opt))
            row["gamma_mean"] = float(np.mean(gamma_opt))

            summary_results.append(row)

    if len(summary_results) > 0:
        summary_df = pd.DataFrame(summary_results)

        summary_file = os.path.join(
            RESULT_ROOT,
            f"accuracy_cluster_{CLUSTER_ID}_plus_AET_metrics.csv"
        )

        summary_df.to_csv(summary_file, index=False)

        med_cols = [
            "Train_Q_NSE", "Train_Q_KGE",
            "Val_Q_NSE", "Val_Q_KGE",
            "Total_AET_NSE"
        ]

        print("\nOracle-Q0 Inference Complete.")
        print(f"Data exported to: {RESULT_ROOT}")

        if SAVE_SIMULATIONS:
            print(f"Simulation files: {SIM_DIR}")
        else:
            print("Simulation files were not saved.")

        print(f"Metrics file: {summary_file}")

        print("\nMedian metrics for this cluster:")
        print(summary_df[med_cols].median(numeric_only=True))

    else:
        print("\n[WARNING] No catchments passed the MIN_RECESSION_DAYS filter. No metrics file generated.")


if __name__ == "__main__":
    main()