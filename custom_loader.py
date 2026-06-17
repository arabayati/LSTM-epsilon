#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom loader for clustered catchment data (Epsilon PINN Version).

Core design
-----------
- Preserve original modeling ideology:
    * HBV forcing x is kept in physical units
    * inversion input z is normalized
    * static attributes are normalized
    * target y stays in physical units
- Do NOT force a basin-wide date intersection
- Keep each basin on its own full record
- Compute warmup/train/val split per basin
- Compute global normalization from pooled TRAIN portions across all basins
- Keep target NaNs as NaN
- Make input-side arrays safe for the model
- **NEW: Inject physics masks, bounds, and tau_t for PINN engine**
"""

from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

EPS = 1e-6

@dataclass
class LoaderConfig:
    # Required run-specific settings
    cluster_id: int
    forcing_root: str
    metadata_csv: str
    static_columns: Sequence[str]

    # Optional run-specific settings
    gridcode_file: Optional[str] = None
    metadata_key: str = "gridcode"

    date_col: str = "Date"
    dynamic_columns: Sequence[str] = field(default_factory=lambda: [
        "precipitation_mmd",
        "temperature_C",
        "pet_mmd",
        "SM_%"
    ])
    z_columns: Optional[Sequence[str]] = None
    target_col: str = "streamflow_mmd"

    warmup_years: int = 1
    train_frac: float = 0.7
    drop_negative_target: bool = False
    verbose: bool = True
    
    # Custom physics parameters
    snow_mask_csv: Optional[str] = None
    snow_free_threshold: float = 25.0
    aet_bounds_csv: Optional[str] = None
    global_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.z_columns is None:
            self.z_columns = list(self.dynamic_columns)

        self.cluster_id = int(self.cluster_id)
        self.warmup_years = int(self.warmup_years)
        self.train_frac = float(self.train_frac)

        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac must be between 0 and 1.")
        if self.warmup_years < 0:
            raise ValueError("warmup_years must be >= 0.")
        if not self.forcing_root:
            raise ValueError("forcing_root must be provided.")
        if not self.metadata_csv:
            raise ValueError("metadata_csv must be provided.")
        if not self.static_columns:
            raise ValueError("static_columns must be provided.")


# ==============================================================================
# PHYSICS HELPERS
# ==============================================================================

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

# ==============================================================================
# STANDARD LOADER HELPERS
# ==============================================================================

def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_gridcodes(gridcode_file: str) -> List[int]:
    if gridcode_file is None:
        raise ValueError("gridcode_file must be provided.")
    gridcodes: List[int] = []
    with open(gridcode_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for token in re.split(r"[,\s]+", s):
                if token:
                    gridcodes.append(int(token))
    return sorted(set(gridcodes))

def list_cluster_gridcodes(cluster_dir: str) -> List[int]:
    out: List[int] = []
    if not os.path.isdir(cluster_dir):
        return out
    for name in os.listdir(cluster_dir):
        m = re.match(r"^\s*(\d+)\s*\.csv\s*$", name, flags=re.IGNORECASE)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))

def find_forcing_file(forcing_root: str, cluster_id: int, gridcode: int) -> str:
    cluster_dir = os.path.join(forcing_root, f"cluster_{cluster_id}")
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")

    pattern = re.compile(rf"^\s*{int(gridcode)}\s*\.csv\s*$", flags=re.IGNORECASE)
    matches = [os.path.join(cluster_dir, name) for name in os.listdir(cluster_dir) if pattern.match(name)]

    if len(matches) == 1: return matches[0]
    if len(matches) > 1:
        raise FileExistsError(f"Multiple forcing files matched gridcode={gridcode}: {matches}")

    glob_matches = glob.glob(os.path.join(cluster_dir, f"*{int(gridcode)}*.csv"))
    filtered = []
    for path in glob_matches:
        base = os.path.basename(path)
        m = re.match(r"^\s*(\d+)\s*\.csv\s*$", base, flags=re.IGNORECASE)
        if m and int(m.group(1)) == int(gridcode):
            filtered.append(path)

    if len(filtered) == 1: return filtered[0]
    if len(filtered) > 1:
        raise FileExistsError(f"Multiple forcing files matched gridcode={gridcode}: {filtered}")
    raise FileNotFoundError(f"No forcing file found for gridcode={gridcode} in cluster_{cluster_id}")

def load_metadata_table(metadata_csv: str, metadata_key: str, static_columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    if metadata_key not in df.columns:
        raise KeyError(f"Metadata key '{metadata_key}' not found in {metadata_csv}")

    missing = [c for c in static_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing static columns in metadata file: {missing}")

    keep_cols = [metadata_key] + list(static_columns)
    df = df[keep_cols].copy()
    df[metadata_key] = pd.to_numeric(df[metadata_key], errors="coerce")
    df = df.dropna(subset=[metadata_key]).copy()
    df[metadata_key] = df[metadata_key].astype(int)
    df = df.drop_duplicates(subset=[metadata_key])
    return df

def load_single_basin_forcing(forcing_csv: str, date_col: str, dynamic_columns: Sequence[str], target_col: str, drop_negative_target: bool = False) -> pd.DataFrame:
    df = pd.read_csv(forcing_csv)
    required = [date_col] + list(dynamic_columns) + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing: raise KeyError(f"Missing columns in {forcing_csv}: {missing}")

    df = df[required].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col])

    for c in dynamic_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    if drop_negative_target:
        df.loc[df[target_col] < 0, target_col] = np.nan

    return df

def compute_split_indices(nt: int, warmup_years: int, train_frac: float) -> Dict[str, int]:
    if nt <= 0: raise ValueError("nt must be positive.")
    warmup_days = int(round(365 * warmup_years))
    warmup_end = min(max(warmup_days, 0), nt)
    remaining = nt - warmup_end
    if remaining < 2:
        raise ValueError(f"Not enough data after warmup: nt={nt}, warmup_days={warmup_days}")

    n_train = int(math.floor(train_frac * remaining))
    n_train = max(1, min(n_train, remaining - 1))
    train_end = warmup_end + n_train
    return {"warmup_end": warmup_end, "train_end": train_end, "nt": nt}

def make_split_labels(nt: int, split_idx: Dict[str, int]) -> np.ndarray:
    labels = np.empty(nt, dtype=object)
    labels[:] = "val"
    labels[:split_idx["warmup_end"]] = "warmup"
    labels[split_idx["warmup_end"]:split_idx["train_end"]] = "train"
    return labels

def _replace_nan_with_zero(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[np.isnan(out)] = 0.0
    return out.astype(np.float32)

def _feature_stats_init(nf: int) -> Dict[str, np.ndarray]:
    return {
        "count": np.zeros(nf, dtype=np.float64),
        "sum": np.zeros(nf, dtype=np.float64),
        "sumsq": np.zeros(nf, dtype=np.float64),
    }

def _feature_stats_update(stats: Dict[str, np.ndarray], arr2d: np.ndarray) -> None:
    if arr2d.ndim != 2: raise ValueError("arr2d must be 2D")
    for j in range(arr2d.shape[1]):
        col = arr2d[:, j]
        mask = np.isfinite(col)
        if np.any(mask):
            vals = col[mask].astype(np.float64)
            stats["count"][j] += vals.size
            stats["sum"][j] += vals.sum()
            stats["sumsq"][j] += np.square(vals).sum()

def _feature_stats_finalize(stats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    count = stats["count"]
    mean = np.zeros_like(count, dtype=np.float64)
    std = np.ones_like(count, dtype=np.float64)

    valid = count > 0
    mean[valid] = stats["sum"][valid] / count[valid]
    var = np.zeros_like(count, dtype=np.float64)
    var[valid] = stats["sumsq"][valid] / count[valid] - np.square(mean[valid])
    var = np.maximum(var, 0.0)
    std[valid] = np.sqrt(var[valid])
    std[~np.isfinite(std) | (std < EPS)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean.astype(np.float32), std.astype(np.float32)

def _normalize_2d(arr2d: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr2d - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)

def _normalize_1d_vector(vec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((vec - mean) / std).astype(np.float32)

def denormalize_2d(arr2d_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr2d_norm * std.reshape(1, -1) + mean.reshape(1, -1)

def basin_has_valid_train_window(basin: Dict[str, Any], rho: int, bufftime: int) -> bool:
    split_idx = basin["split_idx"]
    train_start = max(split_idx["warmup_end"], bufftime)
    train_end = split_idx["train_end"]
    return (train_end - train_start) >= rho
    
def get_valid_train_gridcodes(dataset: Dict[str, Any], rho: int, bufftime: int) -> List[int]:
    out = []
    for gc in dataset["gridcodes"]:
        if basin_has_valid_train_window(dataset["basins"][gc], rho=rho, bufftime=bufftime):
            out.append(gc)
    return out

# ==============================================================================
# MAIN CLUSTER LOAD (FIRST & SECOND PASS)
# ==============================================================================

def load_cluster_data(config: LoaderConfig) -> Dict[str, Any]:
    cfg = config
    _log(f"[custom_loader] Loading cluster {cfg.cluster_id}", cfg.verbose)

    cluster_dir = os.path.join(cfg.forcing_root, f"cluster_{cfg.cluster_id}")
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")

    if cfg.gridcode_file is not None:
        gridcodes = read_gridcodes(cfg.gridcode_file)
        _log(f"[custom_loader] Read {len(gridcodes)} gridcodes from file.", cfg.verbose)
    else:
        gridcodes = list_cluster_gridcodes(cluster_dir)
        _log(f"[custom_loader] Discovered {len(gridcodes)} gridcodes.", cfg.verbose)

    if len(gridcodes) == 0: raise ValueError("No gridcodes found.")

    meta = load_metadata_table(
        metadata_csv=cfg.metadata_csv,
        metadata_key=cfg.metadata_key,
        static_columns=cfg.static_columns,
    ).set_index(cfg.metadata_key)

    # Pre-load the physics CSVs
    snow_df = pd.read_csv(cfg.snow_mask_csv) if cfg.snow_mask_csv else None
    bounds_df = pd.read_csv(cfg.aet_bounds_csv) if cfg.aet_bounds_csv else None

    basins_raw: Dict[int, Dict[str, Any]] = {}
    forcing_files: Dict[int, str] = {}
    skipped: List[Tuple[int, str]] = []

    x_stats = _feature_stats_init(len(cfg.dynamic_columns))
    z_stats = _feature_stats_init(len(cfg.z_columns))
    y_stats = _feature_stats_init(1)
    c_stats = _feature_stats_init(len(cfg.static_columns))

    metadata_used_rows = []

    for gc in gridcodes:
        try:
            forcing_csv = find_forcing_file(cfg.forcing_root, cfg.cluster_id, gc)
            if gc not in meta.index:
                raise KeyError(f"gridcode={gc} not found in metadata file")

            df = load_single_basin_forcing(
                forcing_csv=forcing_csv,
                date_col=cfg.date_col,
                dynamic_columns=cfg.dynamic_columns,
                target_col=cfg.target_col,
                drop_negative_target=cfg.drop_negative_target,
            )

            dates = pd.to_datetime(df[cfg.date_col]).reset_index(drop=True)
            nt = len(df)
            split_idx = compute_split_indices(nt=nt, warmup_years=cfg.warmup_years, train_frac=cfg.train_frac)
            split_labels = make_split_labels(nt, split_idx)

            x_raw = df[list(cfg.dynamic_columns)].to_numpy(dtype=np.float32)
            z_raw = df[list(cfg.z_columns)].to_numpy(dtype=np.float32)
            y_raw = df[[cfg.target_col]].to_numpy(dtype=np.float32)

            c_raw_series = pd.to_numeric(meta.loc[gc, list(cfg.static_columns)], errors="coerce")
            c_raw = c_raw_series.to_numpy(dtype=np.float32)

            tr0 = split_idx["warmup_end"]
            tr1 = split_idx["train_end"]

            _feature_stats_update(x_stats, x_raw[tr0:tr1, :])
            _feature_stats_update(z_stats, z_raw[tr0:tr1, :])
            _feature_stats_update(y_stats, y_raw[tr0:tr1, :])
            _feature_stats_update(c_stats, c_raw.reshape(1, -1))

            # --- Physics Processing ---
            Q = df[cfg.target_col].values
            P = df["precipitation_mmd"].values
            
            raw_rec = detect_recession_paper(Q, min_len=4, drop_first=1, decreasing_rate=True)
            
            # Keep only the classical recession detector:
            # - at least 4-day decreasing Q segment
            # - decreasing recession rate if decreasing_rate=True
            # No Q/P precipitation filter.
            rec_mask = raw_rec.copy()
            
            # Keep the snow filter.
            rec_mask = apply_snow_mask(rec_mask, dates, gc, snow_df, cfg.snow_free_threshold)
             
            start_mask, tau_t = generate_state_reset_tensors(rec_mask)

            gc_bounds = [
                cfg.global_bounds.get('alpha', (0.0, 0.2))[0], cfg.global_bounds.get('alpha', (0.0, 0.2))[1],
                cfg.global_bounds.get('lp', (0.1, 1.0))[0], cfg.global_bounds.get('lp', (0.1, 1.0))[1],
                cfg.global_bounds.get('gamma', (0.1, 5.0))[0], cfg.global_bounds.get('gamma', (0.1, 5.0))[1]
            ]
            #if bounds_df is not None and gc in bounds_df['gridcode'].values:
            #    row = bounds_df[bounds_df['gridcode'] == gc].iloc[0]
            #    gc_bounds[2] = float(row['Lp_lower_CI'])
            #    gc_bounds[3] = float(row['Lp_higer_CI'])
            #    gc_bounds[4] = float(row['gamma_low'])
            #    gc_bounds[5] = float(row['gamma_high'])
            if bounds_df is not None and gc in bounds_df['gridcode'].values:
                row = bounds_df[bounds_df['gridcode'] == gc].iloc[0]
            
                # Widen the AET prior bounds slightly while keeping them inside global physical limits.
                LP_PAD = 0.01
                GAMMA_PAD = 0.01
            
                lp_global_min, lp_global_max = cfg.global_bounds.get('lp', (0.1, 1.0))
                gamma_global_min, gamma_global_max = cfg.global_bounds.get('gamma', (0.1, 5.0))
            
                lp_low = float(row['Lp_lower_CI'])
                lp_high = float(row['Lp_higer_CI'])
                gamma_low = float(row['gamma_low'])
                gamma_high = float(row['gamma_high'])
            
                gc_bounds[2] = max(lp_global_min, lp_low - LP_PAD)
                gc_bounds[3] = min(lp_global_max, lp_high + LP_PAD)
            
                gc_bounds[4] = max(gamma_global_min, gamma_low - GAMMA_PAD)
                gc_bounds[5] = min(gamma_global_max, gamma_high + GAMMA_PAD)            
            

            basins_raw[gc] = {
                "gridcode": gc,
                "forcing_csv": forcing_csv,
                "dates": dates.to_numpy(),
                "split_idx": split_idx,
                "split_labels": split_labels,
                "x_raw": x_raw,
                "z_raw": z_raw,
                "y_raw": y_raw,
                "c_raw": c_raw,
                "nt": nt,
                "rec_mask": rec_mask.astype(np.float32),
                "start_mask": start_mask.astype(np.float32),
                "tau_t": tau_t.astype(np.float32),
                "bounds": np.array(gc_bounds, dtype=np.float32)
            }
            forcing_files[gc] = forcing_csv
            metadata_used_rows.append([gc] + list(c_raw))

        except Exception as e:
            skipped.append((gc, str(e)))

    if skipped:
        _log(f"[custom_loader] Skipped {len(skipped)} basins.", cfg.verbose)

    gridcodes_ok = sorted(basins_raw.keys())
    if len(gridcodes_ok) == 0: raise RuntimeError("No valid basins remained after loading.")

    x_mean, x_std = _feature_stats_finalize(x_stats)
    z_mean, z_std = _feature_stats_finalize(z_stats)
    y_mean, y_std = _feature_stats_finalize(y_stats)
    c_mean, c_std = _feature_stats_finalize(c_stats)

    statDict = {
        "x": {"columns": list(cfg.dynamic_columns), "mean": x_mean.tolist(), "std": x_std.tolist()},
        "z": {"columns": list(cfg.z_columns), "mean": z_mean.tolist(), "std": z_std.tolist()},
        "y": {"columns": [cfg.target_col], "mean": y_mean.tolist(), "std": y_std.tolist()},
        "c": {"columns": list(cfg.static_columns), "mean": c_mean.tolist(), "std": c_std.tolist()},
        "split": {"warmup_years": int(cfg.warmup_years), "train_frac": float(cfg.train_frac)},
        "cluster_id": int(cfg.cluster_id),
        "ngrid": int(len(gridcodes_ok)),
    }

    # Second pass: normalized / model-safe arrays
    basins: Dict[int, Dict[str, Any]] = {}
    for gc in gridcodes_ok:
        b = basins_raw[gc]

        z_norm = _normalize_2d(b["z_raw"], z_mean, z_std)
        y_norm = _normalize_2d(b["y_raw"], y_mean, y_std)
        c_norm = _normalize_1d_vector(b["c_raw"], c_mean, c_std)

        x_model = _replace_nan_with_zero(b["x_raw"])
        z_norm_model = _replace_nan_with_zero(z_norm)
        c_norm_model = _replace_nan_with_zero(c_norm)

        basins[gc] = {
            "gridcode": gc,
            "forcing_csv": b["forcing_csv"],
            "dates": b["dates"],
            "split_idx": b["split_idx"],
            "split_labels": b["split_labels"],
            "x_raw": b["x_raw"],
            "x_model": x_model,
            "z_raw": b["z_raw"],
            "z_norm": z_norm_model,
            "y_raw": b["y_raw"],
            "y_norm": y_norm.astype(np.float32),
            "c_raw": b["c_raw"],
            "c_norm": c_norm_model,
            "nt": b["nt"],
            "rec_mask": b["rec_mask"],
            "start_mask": b["start_mask"],
            "tau_t": b["tau_t"],
            "bounds": b["bounds"]
        }

    metadata_used = pd.DataFrame(metadata_used_rows, columns=[cfg.metadata_key] + list(cfg.static_columns))

    return {
        "gridcodes": gridcodes_ok,
        "basins": basins,
        "statDict": statDict,
        "metadata_used": metadata_used,
        "forcing_files": forcing_files,
        "config_dict": asdict(cfg),
    }

def denormalize_x(arr2d_norm: np.ndarray, statDict: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(statDict["x"]["mean"], dtype=np.float32)
    std = np.asarray(statDict["x"]["std"], dtype=np.float32)
    return denormalize_2d(arr2d_norm, mean, std)

def denormalize_z(arr2d_norm: np.ndarray, statDict: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(statDict["z"]["mean"], dtype=np.float32)
    std = np.asarray(statDict["z"]["std"], dtype=np.float32)
    return denormalize_2d(arr2d_norm, mean, std)

def denormalize_y(arr2d_norm: np.ndarray, statDict: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(statDict["y"]["mean"], dtype=np.float32)
    std = np.asarray(statDict["y"]["std"], dtype=np.float32)
    return denormalize_2d(arr2d_norm, mean, std)

def denormalize_c(arr1d_norm: np.ndarray, statDict: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(statDict["c"]["mean"], dtype=np.float32)
    std = np.asarray(statDict["c"]["std"], dtype=np.float32)
    return arr1d_norm * std + mean

# ==============================================================================
# BATCH BUILDER (CALLED BY TRAIN_MAIN.PY)
# ==============================================================================

def build_dynamic_batch(train_gridcodes, basin_dict, batch_size, rho, bufftime, device):
    import torch
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
        
        z_chunk = b["z_norm"][iT - bufftime:iT + rho, :]
        c_rep = np.repeat(b["c_norm"][None, :], bufftime + rho, axis=0)
        z_list.append(np.concatenate([z_chunk, c_rep], axis=1))
        
        y_list.append(b["y_raw"][iT:iT + rho, :])
        mask_list.append(b["rec_mask"][iT:iT + rho])
        start_list.append(b["start_mask"][iT:iT + rho])
        tau_list.append(b["tau_t"][iT:iT + rho])
        bounds_list.append(b["bounds"])

    def to_tensor(lst, is_time_first=True):
        arr = np.stack(lst, axis=0)
        if is_time_first and arr.ndim == 3:
            arr = np.swapaxes(arr, 0, 1)
        elif is_time_first and arr.ndim == 2:
            arr = np.swapaxes(arr, 0, 1)
        return torch.from_numpy(arr).float().to(device)

    return (
        to_tensor(x_list),
        to_tensor(z_list),
        to_tensor(y_list),
        to_tensor(mask_list).unsqueeze(-1),
        to_tensor(start_list).unsqueeze(-1),
        to_tensor(tau_list).unsqueeze(-1),
        to_tensor(bounds_list, False),
        gc_batch
    )