#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import math
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any
from scipy.signal import find_peaks

EPS = 1e-6

@dataclass
class LoaderConfig:
    cluster_id: int
    forcing_root: str
    metadata_csv: str
    static_columns: Sequence[str]
    date_col: str = "date"
    dynamic_columns: Sequence[str] = field(default_factory=lambda: ["precipitation_mmd", "temperature_C", "pet_mmd"])
    target_col: str = "streamflow_mmd"
    warmup_years: int = 1
    train_frac: float = 0.7
    
    # Custom physics parameters
    snow_mask_csv: Optional[str] = None
    snow_free_threshold: float = 0.25
    aet_bounds_csv: Optional[str] = None
    global_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    verbose: bool = True

# --- Recession Helpers from your old code ---
def get_valid_train_gridcodes(data: Dict[str, Any], rho: int, bufftime: int) -> List[int]:
    """
    Filters out catchments that do not have enough continuous training data 
    to support the required warmup (bufftime) and prediction sequence length (rho).
    """
    valid_gcs = []
    basins = data.get("basins", {})
    
    for gc, b in basins.items():
        split = b.get("split_idx", {})
        
        # Determine the earliest valid start day and the end of the train split
        warmup_end = split.get("warmup_end", bufftime)
        train_end = split.get("train_end", b.get("nt", 0))
        
        train_start = max(warmup_end, bufftime)
        
        # Keep the catchment only if it has enough days for at least one full batch extraction
        if train_end - train_start >= rho:
            valid_gcs.append(gc)
            
    return valid_gcs
    

def detect_recession_simple(Q):
    Q_proc = np.asarray(Q, dtype=float).copy()
    Q_proc[np.isnan(Q_proc)] = np.inf
    mask = np.full_like(Q_proc, False, dtype=bool)
    peaks, _ = find_peaks(Q_proc, prominence=0.0)
    for peak in peaks:
        if peak >= len(Q_proc) - 1: continue
        i = peak
        while i < len(Q_proc) - 1 and Q_proc[i + 1] < Q_proc[i]:
            mask[i + 1] = True
            i += 1
    return mask
    
def detect_recession_paper(Q, min_len=4, drop_first=1, decreasing_rate=True):
    Q_proc = np.asarray(Q, dtype=float).copy()
    Q_proc[np.isnan(Q_proc)] = np.inf
    N = len(Q_proc)
    mask = np.full(N, False, dtype=bool)
    i = 0
    while i < N - 1:
        if Q_proc[i + 1] < Q_proc[i]:
            seg = [i, i + 1]
            R_prev = Q_proc[i] - Q_proc[i + 1] 
            j = i + 1
            while j < N - 1 and Q_proc[j + 1] < Q_proc[j]:
                R_cur = Q_proc[j] - Q_proc[j + 1]
                if (not decreasing_rate) or (R_cur < R_prev):
                    seg.append(j + 1)
                    R_prev = R_cur
                    j += 1
                else: break
            if len(seg) >= min_len:
                for idx in seg[drop_first:]:
                    if 0 <= idx < N: mask[idx] = True
            i = j
        else: i += 1
    return mask

def apply_qp_threshold(raw_mask, Q, P, threshold=1.0, min_prop=0.6, max_gap=2):
    N = len(Q)
    final_mask = np.zeros(N, dtype=bool)
    ratio_now = np.full(N, np.nan, dtype=float)
    posP = P > 0
    ratio_now[posP] = Q[posP] / P[posP]
    pass_day = (P <= 0) | ((P > 0) & (ratio_now > threshold))

    idx = np.where(raw_mask)[0]
    if idx.size == 0: return final_mask
    
    splits = np.where(np.diff(idx) != 1)[0]
    segments = np.split(idx, splits + 1)
    
    for seg in segments:
        if seg.size == 0: continue
        seg_pass = pass_day[seg].copy()
        
        # fill internal gaps up to max_gap
        z = seg_pass
        i = 0
        while i < len(z):
            if not z[i]:
                j = i
                while j < len(z) and not z[j]: j += 1
                if (i - 1 >= 0) and z[i - 1] and (j < len(z)) and z[j] and (j - i <= max_gap):
                    z[i:j] = True
                i = j
            else:
                i += 1
        
        if z.mean() >= min_prop:
            pass_idx = np.where(z)[0]
            if pass_idx.size > 0:
                final_mask[seg[pass_idx[0]: pass_idx[-1]+1]] = True

    return final_mask

def apply_snow_mask(final_mask, dates, gridcode, snow_df, threshold):
    """Zeroes out the recession mask for snowy months."""
    if snow_df is None or gridcode not in snow_df['gridcode'].values:
        return final_mask
        
    gc_snow = snow_df[snow_df['gridcode'] == gridcode]
    snowy_months = gc_snow[gc_snow['mean_q'] > threshold]['month'].values
    
    months = dates.dt.month.values
    is_snowy = np.isin(months, snowy_months)
    
    # Overwrite the mask
    final_mask[is_snowy] = False
    return final_mask

def generate_state_reset_tensors(mask):
    """
    Given a boolean mask [0, 0, 1, 1, 1, 0, 1, 1]
    Returns:
      start_mask: [0, 0, 1, 0, 0, 0, 1, 0]  (1 on the first day of a recession)
      tau_t:      [0, 0, 1, 2, 3, 0, 1, 2]  (Time since start of current recession)
    """
    mask_int = mask.astype(int)
    start_mask = np.zeros_like(mask_int)
    tau_t = np.zeros_like(mask_int, dtype=float)
    
    # Edge detection: It is a start if today is 1 and yesterday was 0
    start_mask[1:] = (mask_int[1:] == 1) & (mask_int[:-1] == 0)
    start_mask[0] = mask_int[0] # Edge case for first day
    
    current_tau = 0
    for i in range(len(mask_int)):
        if mask_int[i] == 0:
            current_tau = 0
        else:
            current_tau += 1
            tau_t[i] = current_tau
            
    return start_mask, tau_t

# --- (Standard Loader functions: _normalize_2d, _feature_stats_init, etc. go here exactly as before) ---
# [Omitted for brevity, assume standard implementation from your LSTM-HBV loader]

def load_cluster_data(cfg: LoaderConfig) -> Dict[str, Any]:
    print(f"[Loader] Processing Cluster {cfg.cluster_id}")
    
    # 1. Load Snow & Bounds CSVs
    snow_df = pd.read_csv(cfg.snow_mask_csv) if cfg.snow_mask_csv else None
    bounds_df = pd.read_csv(cfg.aet_bounds_csv) if cfg.aet_bounds_csv else None
    
    # [Standard reading of metadata and global feature stats goes here]
    # ...
    
    basins = {}
    for gc in gridcodes_ok:
        # [Standard reading of raw arrays goes here]
        # df = load_single_basin_forcing(...)
        
        Q = df[cfg.target_col].values
        P = df["precipitation_mmd"].values
        dates = pd.to_datetime(df[cfg.date_col])
        
        # 2. Physics Masking Pipeline
        #raw_rec = detect_recession_simple(Q)
        #rec_mask = apply_qp_threshold(raw_rec, Q, P)
        #rec_mask = apply_snow_mask(rec_mask, dates, gc, snow_df, cfg.snow_free_threshold)
        
        #start_mask, tau_t = generate_state_reset_tensors(rec_mask)
        # 2. Physics Masking Pipeline
        #Strictly enforce min_len=4 and drop_first=1
        raw_rec = detect_recession_paper(Q, min_len=4, drop_first=1, decreasing_rate=True)
        rec_mask = apply_qp_threshold(raw_rec, Q, P)
        rec_mask = apply_snow_mask(rec_mask, dates, gc, snow_df, cfg.snow_free_threshold)
        
        start_mask, tau_t = generate_state_reset_tensors(rec_mask)
        
        # 3. Bounds Lookup
        gc_bounds = [
            cfg.global_bounds['alpha'][0], cfg.global_bounds['alpha'][1],
            cfg.global_bounds['lp'][0], cfg.global_bounds['lp'][1],
            cfg.global_bounds['gamma'][0], cfg.global_bounds['gamma'][1]
        ]
        if bounds_df is not None and gc in bounds_df['gridcode'].values:
            row = bounds_df[bounds_df['gridcode'] == gc].iloc[0]
            gc_bounds[2] = float(row['Lp_lower_CI'])
            gc_bounds[3] = float(row['Lp_higer_CI'])
            gc_bounds[4] = float(row['gamma_low'])
            gc_bounds[5] = float(row['gamma_high'])
            
        basins[gc] = {
            "x_model": x_model,           
            "z_norm": z_norm_model,       
            "y_raw": y_raw,               
            "c_norm": c_norm_model,
            "rec_mask": rec_mask.astype(np.float32),
            "start_mask": start_mask.astype(np.float32),
            "tau_t": tau_t.astype(np.float32),
            "bounds": np.array(gc_bounds, dtype=np.float32)
        }

    return {"gridcodes": gridcodes_ok, "basins": basins}

# --- Batch Builder ---
def build_dynamic_batch(train_gridcodes, basin_dict, batch_size, rho, bufftime, device):
    """
    Fetches chunks of data, returning standard inputs AND the new physics masks.
    """
    effective_batch = min(batch_size, len(train_gridcodes))
    idx = np.random.randint(0, len(train_gridcodes), size=effective_batch)
    gc_batch = [train_gridcodes[i] for i in idx]

    x_list, z_list, y_list = [], [], []
    mask_list, start_list, tau_list, bounds_list = [], [], [], []

    for gc in gc_batch:
        b = basin_dict[gc]
        split_idx = b["split_idx"]
        train_start = max(split_idx["warmup_end"], bufftime)
        iT = np.random.randint(train_start, split_idx["train_end"] - rho + 1)

        x_list.append(b["x_model"][iT - bufftime:iT + rho, :])
        
        # Z includes static attrs appended to dynamic attrs
        z_chunk = b["z_norm"][iT - bufftime:iT + rho, :]
        c_rep = np.repeat(b["c_norm"][None, :], bufftime + rho, axis=0)
        z_list.append(np.concatenate([z_chunk, c_rep], axis=1))
        
        # Targets & Masks (only needed for the RHO prediction window)
        y_list.append(b["y_raw"][iT:iT + rho, :])
        mask_list.append(b["rec_mask"][iT:iT + rho])
        start_list.append(b["start_mask"][iT:iT + rho])
        tau_list.append(b["tau_t"][iT:iT + rho])
        bounds_list.append(b["bounds"]) # Appended per-basin

    # Convert to Tensors (Time, Batch, Features)
    def to_tensor(lst, is_time_first=True):
        arr = np.stack(lst, axis=0)
        if is_time_first and arr.ndim == 3:
            arr = np.swapaxes(arr, 0, 1)
        elif is_time_first and arr.ndim == 2: # for 1D masks
            arr = np.swapaxes(arr, 0, 1)
        return torch.from_numpy(arr).float().to(device)

    return (
        to_tensor(x_list),          # x_batch
        to_tensor(z_list),          # z_batch
        to_tensor(y_list),          # y_batch
        to_tensor(mask_list).unsqueeze(-1),  # rec_mask
        to_tensor(start_list).unsqueeze(-1), # start_mask
        to_tensor(tau_list).unsqueeze(-1),   # tau_t
        to_tensor(bounds_list, False),       # bounds (Batch, 6)
        gc_batch
    )