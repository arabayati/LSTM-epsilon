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
# CONFIG
# ============================================================

slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
CLUSTER_ID = int(slurm_id) if slurm_id is not None else 1
print(f"[Quick Epoch Sweep] CLUSTER_ID = {CLUSTER_ID}")

FORCING_ROOT = "/project/6107743/ARA_A/data/clustered_forcing_data"
METADATA_CSV = "/project/6107743/ARA_A/attributes/merged_catchments_metadata.csv"
SNOW_MASK_CSV = "/project/6107743/ARA_A/data/monthly_climatology_q70.csv"
AET_BOUNDS_CSV = "/project/6107743/ARA_A/data/lp_gamma_fit_summary_with_recession.csv"

# IMPORTANT: set this to your NEW experiment folder
EXP_ROOT = "/project/6107743/ARA_A/LSTM_Epsilon_code/results_v3_with_10_ODE/"

WEIGHTS_DIR = f"{EXP_ROOT}/cluster_{CLUSTER_ID}/weights/"
OUT_DIR = f"{EXP_ROOT}/cluster_{CLUSTER_ID}/epoch_sweep_no_sim/"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS_TO_CHECK = [50, 70, 90, 100, 110, 120, 130, 140, 150]

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
MIN_RECESSION_DAYS = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# METRICS
# ============================================================

def safe_nan_corrcoef(x, y):
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


def metric_pair(obs, sim, prefix):
    valid = np.isfinite(obs) & np.isfinite(sim)
    return {
        f"{prefix}_NSE": calc_nse(obs, sim),
        f"{prefix}_KGE": calc_kge(obs, sim),
        f"{prefix}_N": int(valid.sum()),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Using device:", device)

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

    print("Loading cluster data once...")
    dataset = load_cluster_data(loader_cfg)
    gridcodes = dataset["gridcodes"]
    print(f"Loaded {len(gridcodes)} basins.")

    n_inv = len(DYNAMIC_COLUMNS) + len(STATIC_COLUMNS)
    model = EpsilonStateResetModel(
        input_dim=n_inv,
        hidden_size=HIDDEN_SIZE,
        n_mul=N_MUL
    ).to(device)

    # ------------------------------------------------------------
    # Cache basin tensors once, so we do not rebuild them for every epoch.
    # ------------------------------------------------------------
    basin_cache = []

    print("Preparing basin tensor cache...")
    for gc in gridcodes:
        b = dataset["basins"][gc]
        nt = b["nt"]

        mask_np = b["rec_mask"][BUFFTIME:] > 0.5
        n_valid_recessions = int(mask_np.sum())

        if n_valid_recessions < MIN_RECESSION_DAYS:
            continue

        z_norm = b["z_norm"]
        c_rep = np.repeat(b["c_norm"][None, :], nt, axis=0)
        z_seq = np.concatenate([z_norm, c_rep], axis=1)

        item = {
            "gc": gc,
            "nt": nt,
            "mask_np": mask_np,
            "abs_idx": np.arange(BUFFTIME, nt)[mask_np],
            "split_labels": b["split_labels"],
            "obs_q": b["y_raw"].flatten(),
            "n_valid_recessions": n_valid_recessions,
            "z_batch": torch.from_numpy(z_seq).float().unsqueeze(1).to(device),
            "pet_batch": torch.from_numpy(b["x_model"][:, 2:3]).float().unsqueeze(1).to(device),
            "sm_batch": torch.from_numpy(b["x_model"][:, 3:4]).float().unsqueeze(1).to(device),
            "rec_mask_full": torch.from_numpy(b["rec_mask"]).float().unsqueeze(1).unsqueeze(-1).to(device),
            "start_mask_full": torch.from_numpy(b["start_mask"]).float().unsqueeze(1).unsqueeze(-1).to(device),
            "bounds_batch": torch.from_numpy(b["bounds"]).float().unsqueeze(0).to(device),
        }
        basin_cache.append(item)

    print(f"Cached {len(basin_cache)} basins after MIN_RECESSION_DAYS filter.")

    all_rows = []
    summary_rows = []

    for epoch in EPOCHS_TO_CHECK:
        model_path = os.path.join(WEIGHTS_DIR, f"Epsilon_Model_Epoch_{epoch}.pth")

        if not os.path.exists(model_path):
            print(f"[SKIP] Missing checkpoint: {model_path}")
            continue

        print(f"\nEvaluating epoch {epoch}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        epoch_rows = []

        with torch.no_grad():
            for item in basin_cache:
                gc = item["gc"]
                nt = item["nt"]
                time_len = nt - BUFFTIME

                model_out = model(
                    item["z_batch"],
                    item["pet_batch"],
                    item["sm_batch"],
                    item["rec_mask_full"],
                    item["start_mask_full"],
                    item["bounds_batch"],
                    bufftime=BUFFTIME,
                )

                q_hat = model_out["q_hat"].detach().cpu().numpy().reshape(time_len, 1)

                sim_q_final = np.full(nt, np.nan, dtype=np.float32)
                sim_q_final[item["abs_idx"]] = q_hat[item["mask_np"], 0]

                split_labels = item["split_labels"]
                train_mask = split_labels == "train"
                val_mask = split_labels == "val"

                obs_q_tr = item["obs_q"][train_mask]
                sim_q_tr = sim_q_final[train_mask]

                obs_q_val = item["obs_q"][val_mask]
                sim_q_val = sim_q_final[val_mask]

                row = {
                    "epoch": epoch,
                    "cluster_id": CLUSTER_ID,
                    "gridcode": gc,
                    "n_total_days": nt,
                    "n_valid_recessions": item["n_valid_recessions"],
                }
                row.update(metric_pair(obs_q_tr, sim_q_tr, "Train_Q"))
                row.update(metric_pair(obs_q_val, sim_q_val, "Val_Q"))

                epoch_rows.append(row)
                all_rows.append(row)

                del model_out

        epoch_df = pd.DataFrame(epoch_rows)

        summary = {
            "epoch": epoch,
            "cluster_id": CLUSTER_ID,
            "n_basins": int(len(epoch_df)),
            "Train_Q_NSE_median": float(epoch_df["Train_Q_NSE"].median()),
            "Train_Q_KGE_median": float(epoch_df["Train_Q_KGE"].median()),
            "Val_Q_NSE_median": float(epoch_df["Val_Q_NSE"].median()),
            "Val_Q_KGE_median": float(epoch_df["Val_Q_KGE"].median()),
        }
        summary_rows.append(summary)

        print(
            f"Epoch {epoch:03d} | "
            f"Train NSE median = {summary['Train_Q_NSE_median']:.4f} | "
            f"Val NSE median = {summary['Val_Q_NSE_median']:.4f} | "
            f"Val KGE median = {summary['Val_Q_KGE_median']:.4f}"
        )

    all_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    all_file = os.path.join(OUT_DIR, f"epoch_sweep_metrics_cluster_{CLUSTER_ID}.csv")
    summary_file = os.path.join(OUT_DIR, f"epoch_sweep_summary_cluster_{CLUSTER_ID}.csv")

    all_df.to_csv(all_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    print("\nDone.")
    print(f"Per-basin metrics saved to: {all_file}")
    print(f"Cluster summary saved to: {summary_file}")


if __name__ == "__main__":
    main()