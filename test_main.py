#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import warnings
import numpy as np
import pandas as pd
import torch

from custom_loader import LoaderConfig, load_cluster_data
from model import EpsilonStateResetModel

warnings.filterwarnings("ignore")

# ============================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ============================================================

slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
CLUSTER_ID = int(slurm_id) if slurm_id is not None else 1
print(f"[Inference Config] CLUSTER_ID = {CLUSTER_ID}")

# --- File Paths ---
FORCING_ROOT = "/project/6107743/ARA_A/data/clustered_forcing_data"
METADATA_CSV = "/project/6107743/ARA_A/attributes/merged_catchments_metadata.csv"
SNOW_MASK_CSV = "/project/6107743/ARA_A/data/monthly_climatology_q70.csv"
AET_BOUNDS_CSV = "/project/6107743/ARA_A/data/lp_gamma_fit_summary_with_recession.csv"

# --- Model Weights Location ---
#WEIGHTS_DIR = f"/project/6107743/ARA_A/LSTM_Epsilon_code/results/cluster_{CLUSTER_ID}/weights/"

WEIGHTS_DIR = f"/project/6107743/ARA_A/LSTM_Epsilon_code/results_v3_with_10_ODE/cluster_{CLUSTER_ID}/weights/"

EPOCH_TO_LOAD = 150 
MODEL_PATH = os.path.join(WEIGHTS_DIR, f"Epsilon_Model_Epoch_{EPOCH_TO_LOAD}.pth")

# --- Export Locations ---
RESULT_ROOT = f"/project/6107743/ARA_A/LSTM_Epsilon_code/results_v3_with_10_ODE/cluster_{CLUSTER_ID}/accuracy_metrics/"
SIM_DIR = os.path.join(RESULT_ROOT, "simulations")
os.makedirs(SIM_DIR, exist_ok=True)

# --- Feature Columns ---
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

# --- Physics & Split Params ---
BUFFTIME = 365
TRAIN_FRAC = 0.6
N_MUL = 10#5
HIDDEN_SIZE = 256
EPS = 1e-6

# --- Catchment Quality Control ---
MIN_RECESSION_DAYS = 25 # Catchments with fewer valid recession days will be skipped.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. METRICS CALCULATOR
# ============================================================

def safe_nan_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2: return np.nan
    x1, y1 = x[mask], y[mask]
    if np.std(x1) < 1e-12 or np.std(y1) < 1e-12: return np.nan
    return float(np.corrcoef(x1, y1)[0, 1])

def calc_nse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2: return np.nan
    o, s = obs[mask], sim[mask]
    denom = np.sum((o - np.mean(o))**2)
    return float(1.0 - np.sum((s - o)**2) / denom) if denom > 0 else np.nan

def calc_kge(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 2: return np.nan
    o, s = obs[mask], sim[mask]
    mo, ms, so, ss = np.mean(o), np.mean(s), np.std(o), np.std(s)
    if abs(mo) < 1e-12 or abs(so) < 1e-12: return np.nan
    r = safe_nan_corrcoef(o, s)
    if not np.isfinite(r): return np.nan
    alpha, beta = ss / so, ms / mo
    return float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))

def calc_r2(obs, sim):
    r = safe_nan_corrcoef(obs, sim)
    return float(r**2) if np.isfinite(r) else np.nan

def calc_mae(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(np.abs(sim[mask] - obs[mask]))) if mask.sum() > 0 else np.nan

def calc_rmse(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.sqrt(np.mean((sim[mask] - obs[mask])**2))) if mask.sum() > 0 else np.nan

def calc_bias(obs, sim):
    mask = np.isfinite(obs) & np.isfinite(sim)
    return float(np.mean(sim[mask] - obs[mask])) if mask.sum() > 0 else np.nan

def compute_metric_row(obs: np.ndarray, sim: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_NSE": calc_nse(obs, sim),
        f"{prefix}_KGE": calc_kge(obs, sim),
        f"{prefix}_R2": calc_r2(obs, sim),
        f"{prefix}_MAE": calc_mae(obs, sim),
        f"{prefix}_RMSE": calc_rmse(obs, sim),
        f"{prefix}_Bias": calc_bias(obs, sim),
        f"{prefix}_N": int(np.isfinite(obs).sum() & np.isfinite(sim).sum()),
    }

# ============================================================
# 3. MAIN INFERENCE LOOP
# ============================================================

def main():
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
        global_bounds={'alpha':(0.0, 1.0), 'lp':(0.1, 1.0), 'gamma':(0.1, 5.0)}
    )
    
    print("Loading Cluster Data...")
    dataset = load_cluster_data(loader_cfg)
    gridcodes = dataset["gridcodes"]
    
    n_inv = len(DYNAMIC_COLUMNS) + len(STATIC_COLUMNS)
    model = EpsilonStateResetModel(input_dim=n_inv, hidden_size=HIDDEN_SIZE, n_mul=N_MUL).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    summary_results = []
    print(f"Starting full sequence inference for {len(gridcodes)} catchments...")
    
    with torch.no_grad():
        for gc in gridcodes:
            b = dataset["basins"][gc]
            nt = b["nt"]
            
            # --- Identify Valid Recessions FIRST ---
            mask_np = b["rec_mask"][BUFFTIME:] > 0.5
            n_valid_recessions = np.sum(mask_np)
            
            if n_valid_recessions < MIN_RECESSION_DAYS:
                print(f"[SKIPPED] gc={gc} : Only {n_valid_recessions} valid recession days (Requires {MIN_RECESSION_DAYS}).")
                continue
            
            # --- Extract Observed AET from the raw CSV securely ---
            obs_aet_full = np.full(nt, np.nan)
            if "forcing_csv" in b and os.path.exists(b["forcing_csv"]):
                df_raw = pd.read_csv(b["forcing_csv"])
                if "aet_mm" in df_raw.columns:
                    obs_aet_full = df_raw["aet_mm"].values
                elif "AET_mm" in df_raw.columns:
                    obs_aet_full = df_raw["AET_mm"].values
            
            # --- Prepare Full Sequence Tensors ---
            z_norm = b["z_norm"]
            c_rep = np.repeat(b["c_norm"][None, :], nt, axis=0)
            z_seq = np.concatenate([z_norm, c_rep], axis=1)
            z_batch = torch.from_numpy(z_seq).float().unsqueeze(1).to(device) 
            
            pet_batch = torch.from_numpy(b["x_model"][:, 2:3]).float().unsqueeze(1).to(device)
            sm_batch = torch.from_numpy(b["x_model"][:, 3:4]).float().unsqueeze(1).to(device)
            
            rec_mask_full = torch.from_numpy(b["rec_mask"]).float().unsqueeze(1).unsqueeze(-1).to(device)
            start_mask_full = torch.from_numpy(b["start_mask"]).float().unsqueeze(1).unsqueeze(-1).to(device)
            bounds_batch = torch.from_numpy(b["bounds"]).float().unsqueeze(0).to(device)

            # --- Forward Pass ---
            model_out = model(z_batch, pet_batch, sm_batch, rec_mask_full, start_mask_full, bounds_batch, bufftime=BUFFTIME)
            
            # --- DEFENSIVE SHAPE FIX --- 
            # Crushing any ghost dimensions generated by model.py
            time_len = nt - BUFFTIME
            
            q_hat = model_out['q_hat'].cpu().numpy().reshape(time_len, 1)
            q_comps = model_out['q_components'].cpu().numpy().reshape(time_len, N_MUL)
            eps_ts = model_out['eps'].cpu().numpy().reshape(time_len, N_MUL)
            aet_ts = model_out['aet'].cpu().numpy().reshape(time_len, N_MUL)
            
            alpha_opt = model_out['alpha'].cpu().numpy().reshape(N_MUL)
            lp_opt = model_out['lp'].cpu().numpy().reshape(N_MUL)
            gamma_opt = model_out['gamma'].cpu().numpy().reshape(N_MUL)
            
            # --- Apply NaN Masking Framework ---
            abs_idx = np.arange(BUFFTIME, nt)[mask_np]
            
            # Initialize blank arrays with NaNs
            sim_q_final = np.full(nt, np.nan)
            sim_q_comps = np.full((nt, N_MUL), np.nan)
            
            eps_mean_final = np.full(nt, np.nan)
            eps_eff_final = np.full(nt, np.nan)
            eps_comps_final = np.full((nt, N_MUL), np.nan)
            
            aet_mean_final = np.full(nt, np.nan)
            aet_comps_final = np.full((nt, N_MUL), np.nan)
            
            alpha_eff_final = np.full(nt, np.nan)
            
            # --- EXACT MACROSCOPIC PHYSICS CALCULATIONS ---
            # Extract only the valid recession days. The .flatten() prevents numpy broadcasting crashes.
            q_val = q_comps[mask_np, :]                 # [Valid_Days, 16]
            eps_val = eps_ts[mask_np, :]                # [Valid_Days, 16]
            aet_val = aet_ts[mask_np, :]                # [Valid_Days, 16]
            q_hat_val = q_hat[mask_np, 0].flatten()     # [Valid_Days]
            alpha_val = alpha_opt[np.newaxis, :]        # [1, 16] Broadcast to time
            
            # 1. Simple Arithmetic Means
            eps_mean = np.mean(eps_val, axis=1).flatten()
            aet_eff = np.mean(aet_val, axis=1).flatten()
            
            # 2. True Effective Epsilon (Q-Squared Weighted)
            e_eps_q2 = np.mean(eps_val * (q_val**2), axis=1).flatten()
            eps_eff = e_eps_q2 / (q_hat_val**2 + EPS)
            
            # 3. True Effective Alpha (Coupled Sink Weighting)
            e_eps_alpha_aet_q = np.mean(eps_val * alpha_val * aet_val * q_val, axis=1).flatten()
            denom_alpha = (eps_eff * aet_eff * q_hat_val) + EPS
            alpha_eff = e_eps_alpha_aet_q / denom_alpha
            
            # --- Inject values into the absolute timeline arrays ---
            sim_q_final[abs_idx] = q_hat_val
            sim_q_comps[abs_idx, :] = q_val
            
            eps_mean_final[abs_idx] = eps_mean
            eps_eff_final[abs_idx] = eps_eff
            eps_comps_final[abs_idx, :] = eps_val
            
            aet_mean_final[abs_idx] = aet_eff
            aet_comps_final[abs_idx, :] = aet_val
            
            alpha_eff_final[abs_idx] = alpha_eff
            
            # --- Create DataFrame ---
            df_sim = pd.DataFrame({
                "date": pd.to_datetime(b["dates"]).strftime("%Y-%m-%d"),
                "Split": b["split_labels"],
                "observed_Q_mmd": b["y_raw"].flatten(),
                "simulated_Q_mmd": sim_q_final,
                "observed_AET_mm": obs_aet_full,
                "simulated_AET_mm": aet_mean_final,
                "epsilon_mean": eps_mean_final,
                "epsilon_effective": eps_eff_final,
                "alpha_effective": alpha_eff_final
            })
            
            # Append individual N_MUL components dynamically
            for m in range(N_MUL):
                df_sim[f"sim_Q_comp_{m+1:02d}"] = sim_q_comps[:, m]
                df_sim[f"eps_comp_{m+1:02d}"] = eps_comps_final[:, m]
                df_sim[f"aet_comp_{m+1:02d}"] = aet_comps_final[:, m]

            # Save Simulation CSV
            df_sim.to_csv(os.path.join(SIM_DIR, f"simulation_{gc}.csv"), index=False)
            
            # --- Accuracy & Metrics Calculations ---
            # Streamflow (Q) Metrics - Strict Train/Val Split
            train_mask = df_sim["Split"] == "train"
            val_mask = df_sim["Split"] == "val"
            
            obs_q_tr = df_sim.loc[train_mask, "observed_Q_mmd"].values
            sim_q_tr = df_sim.loc[train_mask, "simulated_Q_mmd"].values
            
            obs_q_val = df_sim.loc[val_mask, "observed_Q_mmd"].values
            sim_q_val = df_sim.loc[val_mask, "simulated_Q_mmd"].values
            
            # Evapotranspiration (AET) Metrics - One metric for all valid data
            valid_mask = train_mask | val_mask  # Both train and val periods
            obs_aet_all = df_sim.loc[valid_mask, "observed_AET_mm"].values
            sim_aet_all = df_sim.loc[valid_mask, "simulated_AET_mm"].values
            
            row = {
                "gridcode": gc,
                "cluster_id": CLUSTER_ID,
                "n_total_days": nt,
                "n_warmup_days": int((df_sim["Split"] == "warmup").sum()),
                "n_train_days": int(train_mask.sum()),
                "n_val_days": int(val_mask.sum()),
                "n_valid_recessions": int(n_valid_recessions)
            }
            
            # Add Q and AET metric dictionaries to the row
            row.update(compute_metric_row(obs_q_tr, sim_q_tr, "Train_Q"))
            row.update(compute_metric_row(obs_q_val, sim_q_val, "Val_Q"))
            row.update(compute_metric_row(obs_aet_all, sim_aet_all, "Total_AET"))
            
            # Append Static Parameters (16 Components + Arithmetic Averages)
            for m in range(N_MUL):
                row[f"alpha_comp_{m+1:02d}"] = float(alpha_opt[m])
                row[f"lp_comp_{m+1:02d}"] = float(lp_opt[m])
                row[f"gamma_comp_{m+1:02d}"] = float(gamma_opt[m])
                
            row["alpha_mean_static"] = float(np.mean(alpha_opt))
            row["lp_mean"] = float(np.mean(lp_opt))
            row["gamma_mean"] = float(np.mean(gamma_opt))
            
            summary_results.append(row)
            
    # --- Final Save ---
    if len(summary_results) > 0:
        summary_df = pd.DataFrame(summary_results)
        summary_file = os.path.join(RESULT_ROOT, f"accuracy_cluster_{CLUSTER_ID}_plus_AET_metrics.csv")
        summary_df.to_csv(summary_file, index=False)
        print("\nInference Complete.")
        print(f"Data exported to: {RESULT_ROOT}")
        print(f"Simulation files: {SIM_DIR}")
        print(f"Metrics file: {summary_file}")
    else:
        print("\n[WARNING] No catchments passed the MIN_RECESSION_DAYS filter. No metrics file generated.")

if __name__ == "__main__":
    main()