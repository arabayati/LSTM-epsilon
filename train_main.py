#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import warnings
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

from custom_loader import LoaderConfig, load_cluster_data, build_dynamic_batch, get_valid_train_gridcodes
from model import EpsilonStateResetModel
from crit import PhysicsInformedLoss

warnings.filterwarnings("ignore")

# ============================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ============================================================

# --- Cluster & Environment ---
slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
CLUSTER_ID = int(slurm_id) if slurm_id is not None else 1
print(f"[Run Config] CLUSTER_ID = {CLUSTER_ID}")

# --- File Paths ---
#FORCING_ROOT = f"/project/6107743/ARA_A/data/clustered_forcing_data/cluster_{CLUSTER_ID}/"
FORCING_ROOT = "/project/6107743/ARA_A/data/clustered_forcing_data"
METADATA_CSV = "/project/6107743/ARA_A/attributes/merged_catchments_metadata.csv"
SNOW_MASK_CSV = "/project/6107743/ARA_A/data/monthly_climatology_q70.csv"

# --- Outputs ---
OUT_DIR = f"/project/6107743/ARA_A/LSTM_Epsilon_code/results_v3_with_10_ODE/cluster_{CLUSTER_ID}/weights/"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Feature Columns ---
DATE_COL = "date"
# CRITICAL FIX: SM_% must be included to satisfy the AET calculation in model.py
DYNAMIC_COLUMNS = ["precipitation_mmd", "temperature_C", "pet_mmd", "SM_%"]
TARGET_COL = "streamflow_mmd"
STATIC_COLUMNS = [
    "elevation_mean_m", "mean_slope_degree", "Median_DepthToBedrock_cm",
    "Prec_mm", "Temp_C", "PET_mm", "AET_mm", "P_AET_mm", "Aridity", "SF",
    "max_soil_moisture", "Porosity", "Seasonality_of_Moisture_Index",
    "low_high_ratio", "wet_days_ratio_1mm", "wet_days_ratio_5mm",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"
]

# --- Physics Bounds Configuration ---
USE_BOUNDS_CSV = True 
AET_BOUNDS_CSV = "/project/6107743/ARA_A/data/lp_gamma_fit_summary_with_recession.csv"
GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX = 0.0, 1.0
GLOBAL_LP_MIN, GLOBAL_LP_MAX = 0.1, 1.0
GLOBAL_GAMMA_MIN, GLOBAL_GAMMA_MAX = 0.1, 5.0

# --- Loss Function Weights ---
LAMBDA_PATH =25.0#10.0
LAMBDA_RHS = 10.0
LAMBDA_SMOOTH = 0.1
LAMBDA_Q0 = 5.0

# --- Training & Architecture Params ---
N_MUL = 10#5
HIDDEN_SIZE = 256
BATCH_SIZE = 64#100#512
RHO = 365              # Prediction window
BUFFTIME = 365         # Warmup window
EPOCHS = 150#100           # From your original specification
LEARNING_RATE = 1e-4

# --- Recession Detection Params ---
SNOW_FREE_THRESHOLD = 25.0

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. MAIN TRAINING ROUTINE
# ============================================================
def compute_epoch_iterations(train_gridcodes, basin_dict, batch_size, rho, bufftime):
    """
    Calculates the exact number of iterations needed to ensure 99% of the training 
    data is seen per epoch, matching the original LSTM-HBV mathematical formulation.
    """
    total_train_points = 0
    for gc in train_gridcodes:
        b = basin_dict[gc]
        split_idx = b["split_idx"]
        start_min = max(split_idx["warmup_end"], bufftime)
        train_end = split_idx["train_end"]
        total_train_points += max(0, train_end - start_min)

    if total_train_points <= 0:
        raise RuntimeError("No valid training points available.")

    effective_batch = min(batch_size, len(train_gridcodes))
    p = (effective_batch * rho) / float(total_train_points)
    p = min(max(p, 1e-6), 0.99) # Prevent log(0) domain errors

    n_iter = int(np.ceil(np.log(0.01) / np.log(1.0 - p)))
    return max(n_iter, 1)
    
    
def main():
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # --- Setup DataLoader Config ---
    loader_cfg = LoaderConfig(
        cluster_id=CLUSTER_ID,
        forcing_root=FORCING_ROOT,
        metadata_csv=METADATA_CSV,
        static_columns=STATIC_COLUMNS,
        dynamic_columns=DYNAMIC_COLUMNS,
        target_col=TARGET_COL,
        date_col=DATE_COL,
        warmup_years=1, 
        train_frac=0.6,
        snow_mask_csv=SNOW_MASK_CSV,
        snow_free_threshold=SNOW_FREE_THRESHOLD,
        aet_bounds_csv=AET_BOUNDS_CSV if USE_BOUNDS_CSV else None,
        global_bounds={
            'alpha': (GLOBAL_ALPHA_MIN, GLOBAL_ALPHA_MAX),
            'lp': (GLOBAL_LP_MIN, GLOBAL_LP_MAX),
            'gamma': (GLOBAL_GAMMA_MIN, GLOBAL_GAMMA_MAX)
        }
    )

    # --- Load Data ---
    start_time = time.time()
    #print("\nLoading datasets and generating Physics Masks (CPU)...")
    #data = load_cluster_data(loader_cfg)
    #TrainLS = get_valid_train_gridcodes(data, rho=RHO, bufftime=BUFFTIME)
    print("\nLoading datasets and generating Physics Masks (CPU)...")
    data = load_cluster_data(loader_cfg)
    
    # ------------------------------------------------------------
    # AET-bound diagnostic: confirms the widened lp/gamma bounds
    # actually entered the loaded basin dictionary.
    # bounds columns:
    # [alpha_min, alpha_max, lp_min, lp_max, gamma_min, gamma_max]
    # ------------------------------------------------------------
    bounds_arr = np.stack(
        [b["bounds"] for b in data["basins"].values()],
        axis=0
    )
    
    print("\n[Bounds diagnostic]")
    print(f"Number of loaded basins: {bounds_arr.shape[0]}")
    print(f"alpha_min range:  {bounds_arr[:, 0].min():.4f} to {bounds_arr[:, 0].max():.4f}")
    print(f"alpha_max range:  {bounds_arr[:, 1].min():.4f} to {bounds_arr[:, 1].max():.4f}")
    
    print(f"lp_min range:     {bounds_arr[:, 2].min():.4f} to {bounds_arr[:, 2].max():.4f}")
    print(f"lp_max range:     {bounds_arr[:, 3].min():.4f} to {bounds_arr[:, 3].max():.4f}")
    
    print(f"gamma_min range:  {bounds_arr[:, 4].min():.4f} to {bounds_arr[:, 4].max():.4f}")
    print(f"gamma_max range:  {bounds_arr[:, 5].min():.4f} to {bounds_arr[:, 5].max():.4f}")
    
    bad_lp = np.sum(bounds_arr[:, 2] >= bounds_arr[:, 3])
    bad_gamma = np.sum(bounds_arr[:, 4] >= bounds_arr[:, 5])
    
    print(f"Bad lp bounds count:    {bad_lp}")
    print(f"Bad gamma bounds count: {bad_gamma}")
    print("------------------------------------------------------------\n")
    
    TrainLS = get_valid_train_gridcodes(data, rho=RHO, bufftime=BUFFTIME)
    
    if len(TrainLS) == 0:
        raise RuntimeError("No valid basins have enough training data.")
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds. {len(TrainLS)} valid training basins.")

    # --- Initialize Architecture ---
    N_inv = len(DYNAMIC_COLUMNS) + len(STATIC_COLUMNS)
    model = EpsilonStateResetModel(input_dim=N_inv, hidden_size=HIDDEN_SIZE, n_mul=N_MUL).to(device)
    criterion = PhysicsInformedLoss(LAMBDA_PATH, LAMBDA_RHS, LAMBDA_SMOOTH, LAMBDA_Q0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Determine iterations per epoch (Guarantee at least 50 iterations, or scale by dataset size)
    #ITERS_PER_EPOCH = max(50, len(TrainLS) // BATCH_SIZE * 3)
    #print(f"\nStarting Training: {EPOCHS} Epochs, {ITERS_PER_EPOCH} Iters/Epoch.")
    # Determine iterations per epoch using the strict 99% coverage rule
    ITERS_PER_EPOCH = compute_epoch_iterations(
        train_gridcodes=TrainLS,
        basin_dict=data["basins"],
        batch_size=BATCH_SIZE,
        rho=RHO,
        bufftime=BUFFTIME
    )
    print(f"\nStarting Training: {EPOCHS} Epochs, {ITERS_PER_EPOCH} Iters/Epoch.")
    

    loss_history = []

    # --- The Global Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # Track loss components for reporting
        ep_loss = {'total': 0.0, 'path': 0.0, 'rhs': 0.0, 'smooth': 0.0, 'q0': 0.0}

        for iIter in range(ITERS_PER_EPOCH):
            # 1. Fetch Global Randomized Mini-Batch
            batch_data = build_dynamic_batch(
                train_gridcodes=TrainLS,
                basin_dict=data["basins"],
                batch_size=BATCH_SIZE,
                rho=RHO,
                bufftime=BUFFTIME,
                device=device
            )

            # 2. Unpack the Tensors
            x_batch, z_batch, y_batch, rec_mask, start_mask, tau_t, bounds, gc_batch = batch_data

            # 3. Extract Physical Forcings safely using aligned indices
            # index 2: pet_mmd | index 3: SM_%
            pet_seq = x_batch[:, :, 2:3]
            sm_seq  = x_batch[:, :, 3:4]

            # 4. Forward Pass
            optimizer.zero_grad()
            model_out = model(z_batch, pet_seq, sm_seq, rec_mask, start_mask, bounds, bufftime=BUFFTIME) 
            
            if epoch == 1 and iIter == 0:
                print("SHAPE CHECK")
                print("y_batch:", y_batch.shape)
                print("rec_mask:", rec_mask.shape)
                print("q_hat:", model_out["q_hat"].shape)
                print("q_components:", model_out["q_components"].shape)
                print("eps:", model_out["eps"].shape)
                print("aet:", model_out["aet"].shape)            
            
            # 5. Masked Physics Loss
            loss_dict = criterion(model_out, y_batch, rec_mask, start_mask)
            loss = loss_dict['total']
            
            # 6. Backprop & Step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate logs
            ep_loss['total'] += loss.item()
            ep_loss['path'] += loss_dict['l_path'].item()
            ep_loss['rhs'] += loss_dict['l_rhs'].item()
            ep_loss['smooth'] += loss_dict['l_smooth'].item()
            ep_loss['q0'] += loss_dict['l_q0'].item()

        # --- Epoch-average losses ---
        avg_total = ep_loss['total'] / ITERS_PER_EPOCH
        avg_path = ep_loss['path'] / ITERS_PER_EPOCH
        avg_rhs = ep_loss['rhs'] / ITERS_PER_EPOCH
        avg_smooth = ep_loss['smooth'] / ITERS_PER_EPOCH
        avg_q0 = ep_loss['q0'] / ITERS_PER_EPOCH

        # --- Save losses in memory for CSV ---
        loss_history.append({
            "epoch": epoch,
            "loss_total": avg_total,
            "loss_path": avg_path,
            "loss_rhs": avg_rhs,
            "loss_smooth": avg_smooth,
            "loss_q0": avg_q0
        })

        # --- Logging for every epoch ---
        print(f"Epoch {epoch:04d} | L_Total: {avg_total:.5f} | "
              f"L_Path: {avg_path:.5f} | "
              f"L_RHS: {avg_rhs:.5f} | "
              f"L_Smooth: {avg_smooth:.5f} | "
              f"L_Q0: {avg_q0:.5f}")

        # --- Checkpoint Saving ---
        if epoch % 10 == 0 or epoch == EPOCHS:
            save_path = os.path.join(OUT_DIR, f"Epsilon_Model_Epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

    # --- Save loss log as one CSV ---
    loss_log_path = os.path.join(OUT_DIR, f"loss_log_cluster_{CLUSTER_ID}.csv")
    pd.DataFrame(loss_history).to_csv(loss_log_path, index=False)

    print("\nTraining Complete.")
    print(f"Final Weights saved to: {OUT_DIR}")
    print(f"Loss log saved to: {loss_log_path}")

if __name__ == "__main__":
    main()