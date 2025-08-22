"""
How to run
==========
python crime_forecast.py \
  --csv_path ./crimes.csv \
  --date_col Date \
  --lat_col Latitude \
  --lon_col Longitude \
  --type_col "Primary Type" \
  --start_year 2014 \
  --end_year 2021 \
  --grid_precision 3 \
  --category "ALL"

If you want category‑specific forecasting (e.g., THEFT only), set --category "THEFT".
"""

from __future__ import annotations
import argparse
import os
import math
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def _week_start(dt: pd.Timestamp) -> pd.Timestamp:
    # Normalize to Monday of the same week (ISO)
    return (dt - pd.to_timedelta(dt.weekday(), unit="D")).normalize()


def make_grid(lat: pd.Series, lon: pd.Series, precision: int = 3) -> pd.Series:
    """Simple spatial cells by rounding coordinates. precision=3 ~ ~100m-150m bins in Chicago.
    Adjust as needed. Returns a string key like "41.89_-87.65"."""
    lat_round = lat.round(precision).astype(str)
    lon_round = lon.round(precision).astype(str)
    return lat_round + "_" + lon_round


# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------

def build_spatiotemporal_frame(df: pd.DataFrame,
                               date_col: str,
                               lat_col: str,
                               lon_col: str,
                               type_col: Optional[str] = None,
                               category: str = "ALL",
                               grid_precision: int = 3) -> pd.DataFrame:
    # Parse datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, lat_col, lon_col])

    # Optionally filter to one crime category
    if type_col and category and category != "ALL":
        df = df[df[type_col].astype(str).str.upper() == category.upper()]

    # Spatial cell key
    df["cell"] = make_grid(df[lat_col].astype(float), df[lon_col].astype(float), precision=grid_precision)

    # Week key (Monday start)
    df["week"] = df[date_col].apply(_week_start)

    # Aggregate to WEEK x CELL count
    weekly = (
        df.groupby(["cell", "week"], as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )

    # Add calendar features
    weekly["weekofyear"] = weekly["week"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week"].dt.month.astype(int)
    weekly["year"] = weekly["week"].dt.year.astype(int)

    # Create per‑cell time index for lags
    weekly = weekly.sort_values(["cell", "week"]).reset_index(drop=True)

    # Lags and rolling means
    def add_lags(group: pd.DataFrame, lags=(1, 2, 4, 8), roll_windows=(2, 4, 8)) -> pd.DataFrame:
        g = group.copy()
        for L in lags:
            g[f"lag_{L}"] = g["count"].shift(L)
        for W in roll_windows:
            g[f"rollmean_{W}"] = g["count"].rolling(W).mean()
            g[f"rollstd_{W}"] = g["count"].rolling(W).std()
        g["trend_4"] = g["count"].diff(4)
        return g

    weekly = weekly.groupby("cell", group_keys=False).apply(add_lags)

    # Drop rows with NaNs from lags at the start of each cell series
    weekly = weekly.dropna().reset_index(drop=True)

    return weekly


# --------------------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------------------

def time_split(df: pd.DataFrame, test_weeks: int = 12, val_weeks: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # last N weeks for test, previous N for validation, rest for train
    last_week = df["week"].max()
    val_start = last_week - pd.to_timedelta(test_weeks + val_weeks - 1, unit="W")
    test_start = last_week - pd.to_timedelta(test_weeks - 1, unit="W")

    train = df[df["week"] < val_start]
    val   = df[(df["week"] >= val_start) & (df["week"] < test_start)]
    test  = df[df["week"] >= test_start]
    return train, val, test


def fit_regressor(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> GradientBoostingRegressor:
    Xtr, ytr = train[features], train["count_next"]
    Xva, yva = val[features], val["count_next"]
    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xtr, ytr)
    preds = model.predict(Xva)
    print("[Regressor] Val MAE:", mean_absolute_error(yva, preds))
    print("[Regressor] Val R2 :", r2_score(yva, preds))
    return model


def fit_classifier(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> RandomForestClassifier:
    Xtr, ytr = train[features], train["is_high_risk_next"]
    Xva, yva = val[features], val["is_high_risk_next"]
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xva)[:, 1]
    try:
        auc = roc_auc_score(yva, proba)
    except ValueError:
        auc = float("nan")
    ap = average_precision_score(yva, proba)
    print(f"[Classifier] Val ROC‑AUC: {auc:.3f}  |  PR‑AUC: {ap:.3f}")
    return clf


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--lat_col", default="Latitude")
    ap.add_argument("--lon_col", default="Longitude")
    ap.add_argument("--type_col", default=None)
    ap.add_argument("--category", default="ALL")
    ap.add_argument("--start_year", type=int, default=None)
    ap.add_argument("--end_year", type=int, default=None)
    ap.add_argument("--grid_precision", type=int, default=3)
    ap.add_argument("--risk_percentile", type=float, default=0.8, help="threshold for high‑risk flag on next‑week counts")
    ap.add_argument("--out_dir", default="./outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    print("Loading:", args.csv_path)
    raw = pd.read_csv(args.csv_path)

    # Optional year filter (on date_col)
    if args.start_year or args.end_year:
        raw[args.date_col] = pd.to_datetime(raw[args.date_col], errors="coerce")
        if args.start_year:
            raw = raw[raw[args.date_col].dt.year >= args.start_year]
        if args.end_year:
            raw = raw[raw[args.date_col].dt.year <= args.end_year]

    # Build WEEK x CELL frame with lags
    weekly = build_spatiotemporal_frame(
        raw,
        date_col=args.date_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        type_col=args.type_col,
        category=args.category,
        grid_precision=args.grid_precision,
    )

    # Target: next‑week count per cell
    weekly = weekly.sort_values(["cell", "week"])  # safety
    weekly["count_next"] = weekly.groupby("cell")["count"].shift(-1)
    weekly = weekly.dropna(subset=["count_next"]).reset_index(drop=True)

    # High‑risk label based on next‑week distribution across all cells
    thr = weekly["count_next"].quantile(args.risk_percentile)
    weekly["is_high_risk_next"] = (weekly["count_next"] >= thr).astype(int)

    # Train/val/test time split
    train, val, test = time_split(weekly)

    # Feature list (exclude leakage)
    base_feats = [
        "weekofyear", "month", "year",
        "lag_1", "lag_2", "lag_4", "lag_8",
        "rollmean_2", "rollmean_4", "rollmean_8",
        "rollstd_2", "rollstd_4", "rollstd_8",
        "trend_4",
    ]

    # Regressor: predict numeric next‑week counts
    reg = fit_regressor(train, val, base_feats)

    # Classifier: predict high‑risk probability
    clf = fit_classifier(train, val, base_feats)

    # Evaluate on test
    Xte = test[base_feats]

    # Regressor metrics
    reg_preds = reg.predict(Xte)
    mae = mean_absolute_error(test["count_next"], reg_preds)
    r2  = r2_score(test["count_next"], reg_preds)
    print("[Regressor] Test MAE:", mae)
    print("[Regressor] Test R2 :", r2)

    # Classifier metrics
    proba = clf.predict_proba(Xte)[:, 1]
    try:
        auc = roc_auc_score(test["is_high_risk_next"], proba)
    except ValueError:
        auc = float("nan")
    ap = average_precision_score(test["is_high_risk_next"], proba)
    print(f"[Classifier] Test ROC‑AUC: {auc:.3f}  |  PR‑AUC: {ap:.3f}")

    # Save outputs
    out_preds = test[["cell", "week", "count", "count_next", "is_high_risk_next"]].copy()
    out_preds["pred_count_next"] = reg_preds
    out_preds["pred_highrisk_proba"] = proba
    out_path = os.path.join(args.out_dir, "predictions.csv")
    out_preds.to_csv(out_path, index=False)

    # Feature importances (GBR + RF)
    fi_reg = pd.DataFrame({"feature": base_feats, "importance": reg.feature_importances_})
    fi_clf = pd.DataFrame({"feature": base_feats, "importance": clf.feature_importances_})
    fi_reg.to_csv(os.path.join(args.out_dir, "fi_regressor.csv"), index=False)
    fi_clf.to_csv(os.path.join(args.out_dir, "fi_classifier.csv"), index=False)

    print("Saved:", out_path)
    print("Saved:", os.path.join(args.out_dir, "fi_regressor.csv"))
    print("Saved:", os.path.join(args.out_dir, "fi_classifier.csv"))


if __name__ == "__main__":
    main()
