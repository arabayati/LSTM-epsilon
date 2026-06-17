#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd

from custom_loader import LoaderConfig, load_cluster_data, get_valid_train_gridcodes

# ============================================================
# SETTINGS
# ============================================================

CLUSTERS = list(range(1, 2))

FORCING_ROOT = "/project/6107743/ARA_A/data/clustered_forcing_data"
METADATA_CSV = "/project/6107743/ARA_A/attributes/merged_catchments_metadata.csv"
SNOW_MASK_CSV = "/project/6107743/ARA_A/data/monthly_climatology_q70.csv"
AET_BOUNDS_CSV = "/project/6107743/ARA_A/data/lp_gamma_fit_summary_with_recession.csv"

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

TRAIN_FRAC = 0.6
BUFFTIME = 365
RHO = 730
BATCH_SIZE = 100
SNOW_FREE_THRESHOLD = 25.0

# Keep this moderate because we are doing 9 clusters × 2 loaders.
N_SIM_BATCHES = 100

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOADER
# ============================================================

def make_loader_cfg(cluster_id: int, use_snow_filter: bool) -> LoaderConfig:
    return LoaderConfig(
        cluster_id=cluster_id,
        forcing_root=FORCING_ROOT,
        metadata_csv=METADATA_CSV,
        static_columns=STATIC_COLUMNS,
        dynamic_columns=DYNAMIC_COLUMNS,
        target_col=TARGET_COL,
        date_col=DATE_COL,
        warmup_years=1,
        train_frac=TRAIN_FRAC,
        snow_mask_csv=SNOW_MASK_CSV if use_snow_filter else None,
        snow_free_threshold=SNOW_FREE_THRESHOLD,
        aet_bounds_csv=AET_BOUNDS_CSV,
        global_bounds={
            "alpha": (0.0, 1.0),
            "lp": (0.1, 1.0),
            "gamma": (0.1, 5.0),
        },
        verbose=False,
    )


# ============================================================
# WINDOW PRECOMPUTATION
# ============================================================

def rhs_pair_mask(mask_1d):
    m = np.asarray(mask_1d).astype(bool)
    if len(m) < 2:
        return np.zeros(0, dtype=bool)
    return m[:-1] & m[1:]


def precompute_windows_for_basin(b):
    split_idx = b["split_idx"]
    train_start = max(split_idx["warmup_end"], BUFFTIME)
    train_end = split_idx["train_end"]

    if train_end - train_start < RHO:
        return None

    valid_starts = np.arange(train_start, train_end - RHO + 1)

    rec = np.asarray(b["rec_mask"]).astype(int)
    starts = np.asarray(b["start_mask"]).astype(int)
    rhs = rhs_pair_mask(rec).astype(int)

    rec_cum = np.concatenate([[0], np.cumsum(rec)])
    start_cum = np.concatenate([[0], np.cumsum(starts)])
    rhs_cum = np.concatenate([[0], np.cumsum(rhs)])

    rec_counts = rec_cum[valid_starts + RHO] - rec_cum[valid_starts]
    start_counts = start_cum[valid_starts + RHO] - start_cum[valid_starts]
    rhs_counts = rhs_cum[valid_starts + RHO - 1] - rhs_cum[valid_starts]

    return {
        "rec_counts": rec_counts.astype(float),
        "rhs_counts": rhs_counts.astype(float),
        "start_counts": start_counts.astype(float),
        "total_rec_weight": float(rec_counts.sum()),
    }


def precompute_all_windows(data, train_gridcodes):
    bank = {}
    for gc in train_gridcodes:
        w = precompute_windows_for_basin(data["basins"][gc])
        if w is not None:
            bank[gc] = w
    return bank


# ============================================================
# SAMPLING
# ============================================================

def sample_old_uniform(bank, available_gcs):
    gc = int(np.random.choice(available_gcs))
    w = bank[gc]
    idx = np.random.randint(0, len(w["rec_counts"]))
    return w["rec_counts"][idx], w["rhs_counts"][idx], w["start_counts"][idx]


def sample_pair_weighted(bank, weighted_gcs, gc_probs):
    gc = int(np.random.choice(weighted_gcs, p=gc_probs))
    w = bank[gc]

    weights = w["rec_counts"]
    idx = np.random.choice(np.arange(len(weights)), p=weights / weights.sum())

    return w["rec_counts"][idx], w["rhs_counts"][idx], w["start_counts"][idx]


def simulate_old_uniform(bank):
    available_gcs = sorted(bank.keys())
    effective_batch = min(BATCH_SIZE, len(available_gcs))

    batch_rec = []
    batch_rhs = []
    zero_windows = []

    for _ in range(N_SIM_BATCHES):
        rec_sum = 0.0
        rhs_sum = 0.0
        zero_count = 0

        for _ in range(effective_batch):
            rec, rhs, _ = sample_old_uniform(bank, available_gcs)
            rec_sum += rec
            rhs_sum += rhs
            zero_count += int(rec == 0)

        batch_rec.append(rec_sum)
        batch_rhs.append(rhs_sum)
        zero_windows.append(zero_count)

    return {
        "rec_mean": float(np.mean(batch_rec)),
        "rhs_mean": float(np.mean(batch_rhs)),
        "zero_mean": float(np.mean(zero_windows)),
    }


def simulate_pair_weighted(bank):
    available_gcs = sorted(bank.keys())
    effective_batch = min(BATCH_SIZE, len(available_gcs))

    weights = np.array([bank[gc]["total_rec_weight"] for gc in available_gcs], dtype=float)
    keep = weights > 0

    if keep.sum() == 0:
        return {
            "rec_mean": np.nan,
            "rhs_mean": np.nan,
            "zero_mean": np.nan,
        }

    weighted_gcs = np.array(available_gcs)[keep]
    weights = weights[keep]
    gc_probs = weights / weights.sum()

    batch_rec = []
    batch_rhs = []
    zero_windows = []

    for _ in range(N_SIM_BATCHES):
        rec_sum = 0.0
        rhs_sum = 0.0
        zero_count = 0

        for _ in range(effective_batch):
            rec, rhs, _ = sample_pair_weighted(bank, weighted_gcs, gc_probs)
            rec_sum += rec
            rhs_sum += rhs
            zero_count += int(rec == 0)

        batch_rec.append(rec_sum)
        batch_rhs.append(rhs_sum)
        zero_windows.append(zero_count)

    return {
        "rec_mean": float(np.mean(batch_rec)),
        "rhs_mean": float(np.mean(batch_rhs)),
        "zero_mean": float(np.mean(zero_windows)),
    }


# ============================================================
# SUMMARY
# ============================================================

def bank_availability(bank):
    n_catchments = len(bank)
    total_windows = sum(len(w["rec_counts"]) for w in bank.values())
    active_windows = sum(int((w["rec_counts"] > 0).sum()) for w in bank.values())

    max_rec_by_catchment = np.array(
        [w["rec_counts"].max() for w in bank.values()],
        dtype=float
    )

    active_catchments = int((max_rec_by_catchment > 0).sum())

    return {
        "n_catchments": n_catchments,
        "active_catchments": active_catchments,
        "active_catchment_frac": active_catchments / max(n_catchments, 1),
        "active_window_frac": active_windows / max(total_windows, 1),
    }


def diagnose_one_case(cluster_id, case_name, data, train_gridcodes):
    bank = precompute_all_windows(data, train_gridcodes)

    avail = bank_availability(bank)
    old = simulate_old_uniform(bank)
    pair = simulate_pair_weighted(bank)

    return {
        "cluster": cluster_id,
        "case": case_name,
        "eligible_basins": len(train_gridcodes),

        "active_catchment_frac": avail["active_catchment_frac"],
        "active_window_frac": avail["active_window_frac"],

        "old_rec_mean": old["rec_mean"],
        "old_rhs_mean": old["rhs_mean"],
        "old_zero_windows_mean": old["zero_mean"],

        "pair_rec_mean": pair["rec_mean"],
        "pair_rhs_mean": pair["rhs_mean"],
        "pair_zero_windows_mean": pair["zero_mean"],

        "rec_gain": pair["rec_mean"] / old["rec_mean"] if old["rec_mean"] > 0 else np.nan,
        "rhs_gain": pair["rhs_mean"] / old["rhs_mean"] if old["rhs_mean"] > 0 else np.nan,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    rows = []

    for cluster_id in CLUSTERS:
        print(f"\nProcessing cluster {cluster_id}...")

        data_snow = load_cluster_data(make_loader_cfg(cluster_id, use_snow_filter=True))
        data_no = load_cluster_data(make_loader_cfg(cluster_id, use_snow_filter=False))

        train_snow = set(get_valid_train_gridcodes(data_snow, rho=RHO, bufftime=BUFFTIME))
        train_no = set(get_valid_train_gridcodes(data_no, rho=RHO, bufftime=BUFFTIME))
        train_gridcodes = sorted(train_snow.intersection(train_no))

        if len(train_gridcodes) == 0:
            rows.append({
                "cluster": cluster_id,
                "case": "with_snow_filter",
                "eligible_basins": 0,
            })
            rows.append({
                "cluster": cluster_id,
                "case": "without_snow_filter",
                "eligible_basins": 0,
            })
            continue

        rows.append(diagnose_one_case(cluster_id, "with_snow_filter", data_snow, train_gridcodes))
        rows.append(diagnose_one_case(cluster_id, "without_snow_filter", data_no, train_gridcodes))

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("ALL-CLUSTER DIAGNOSTIC SUMMARY")
    print("=" * 120)

    cols = [
        "cluster", "case", "eligible_basins",
        "active_catchment_frac", "active_window_frac",
        "old_rec_mean", "old_rhs_mean", "old_zero_windows_mean",
        "pair_rec_mean", "pair_rhs_mean", "pair_zero_windows_mean",
        "rec_gain", "rhs_gain",
    ]

    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
