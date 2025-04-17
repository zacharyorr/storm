# outage_direct_monthly.py  (steps 1-8 working, ASCII logging)
# ---------------------------------------------------------
# Build state-level monthly outage percentage dataset from
# county-level climate features + Excel outage workbook.
# ---------------------------------------------------------

from __future__ import annotations

import os
import sys
import logging
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import pandas as pd

# =========================================================
# 1. CONFIG & PATHS
# =========================================================

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

SHAPEFILE_PATH = PROJECT_ROOT / "data/geospatial/tl_2024_us_county.shp"

CLIMATE_FILE = PROJECT_ROOT / "output/processed_climate/county_monthly_climate_variables_target_states.parquet"
OUTAGE_EXCEL = PROJECT_ROOT / "data/outages/Outages.xlsx"

TARGET_STATES: List[str] = [
    "12", "20", "29", "40", "48", "05", "22", "28",
    "01", "13", "45", "37", "47", "51",
]
STATE_NAME_TO_FIPS: Dict[str, str] = {
    'Florida': '12', 'Kansas': '20', 'Missouri': '29', 'Oklahoma': '40',
    'Texas': '48', 'Arkansas': '05', 'Louisiana': '22', 'Mississippi': '28',
    'Alabama': '01', 'Georgia': '13', 'South Carolina': '45',
    'North Carolina': '37', 'Tennessee': '47', 'Virginia': '51',
}

TRAIN_END_Y = 2021
TEST_START_Y = 2022
FORECAST_Y0, FORECAST_Y1 = 2025, 2035
LAG_MONTHS = [1, 3, 6]

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "outage_direct_monthly.log"

logger = logging.getLogger("outage_direct_monthly")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)

# =========================================================
# 2. LOAD DATA
# =========================================================

def load_climate(path: Path, target_states: List[str]) -> pd.DataFrame:
    if not path.exists():
        logger.error("Climate file not found: %s", path)
        sys.exit(1)
    logger.info("Reading climate file %s", path.relative_to(PROJECT_ROOT))
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    if "county_geoid" not in df.columns:
        logger.error("county_geoid column missing in climate file")
        sys.exit(1)

    df["county_geoid"] = df["county_geoid"].astype(str).str.zfill(5)
    df["state"] = df["county_geoid"].str[:2]
    before = len(df)
    df = df[df["state"].isin(target_states)].copy()
    logger.info("Filtered climate rows: %d -> %d for target states", before, len(df))

    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    elif {"year", "month"}.issubset(df.columns):
        df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    else:
        logger.error("time or year/month columns missing in climate data")
        sys.exit(1)

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    yr_min, yr_max = int(df.year.min()), int(df.year.max())
    logger.info("Climate years: %s-%s | columns: %d", yr_min, yr_max, df.shape[1])
    return df


def load_outages(path: Path, target_states: List[str]) -> pd.DataFrame:
    if not path.exists():
        logger.error("Outage workbook not found: %s", path)
        sys.exit(1)
    logger.info("Reading outage workbook %s (sheet 0)", path.relative_to(PROJECT_ROOT))
    df_raw = pd.read_excel(path, sheet_name=0)
    logger.debug("Loaded %d raw outage rows", len(df_raw))

    if "STATE" not in df_raw.columns:
        logger.error("STATE column missing in outage workbook")
        sys.exit(1)
    df_raw["state"] = df_raw["STATE"].map(STATE_NAME_TO_FIPS)
    unmapped = df_raw["state"].isna().sum()
    if unmapped:
        logger.warning("%d outage rows with unmapped STATE dropped", unmapped)
    df_raw = df_raw.dropna(subset=["state"])

    for col in ("Year", "Month"):
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    df_raw.dropna(subset=["Year", "Month"], inplace=True)

    required = {"CustomersTrackedTotal", "MaxCustomersOutTotal"}
    if not required.issubset(df_raw.columns):
        logger.error("Required outage metric columns missing in workbook")
        sys.exit(1)

    tracked = df_raw["CustomersTrackedTotal"].astype(float)
    peak = df_raw["MaxCustomersOutTotal"].astype(float)
    df_raw["pct_out"] = np.where((tracked > 0) & peak.notna(), (peak / tracked) * 100, np.nan)

    before = len(df_raw)
    df_raw = df_raw[df_raw["state"].isin(target_states)].copy()
    logger.info("Filtered outage rows: %d -> %d for target states", before, len(df_raw))

    df_raw["date"] = pd.to_datetime(dict(year=df_raw.Year, month=df_raw.Month, day=1))
    df_state_month = (
        df_raw.groupby(["state", "Year", "Month"], observed=True)["pct_out"].mean().reset_index()
    )
    df_state_month.rename(columns={"Year": "year", "Month": "month"}, inplace=True)
    df_state_month["date"] = pd.to_datetime(dict(year=df_state_month.year, month=df_state_month.month, day=1))

    logger.info("Outage state-month rows: %d (%s-%s)", len(df_state_month), int(df_state_month.year.min()), int(df_state_month.year.max()))
    return df_state_month

# =========================================================
# 3. MERGE & AGGREGATE
# =========================================================

def aggregate_climate_state(df: pd.DataFrame) -> pd.DataFrame:
    non_feat = {"county_geoid", "state", "year", "month", "date", "time"}
    base_cols = [c for c in df.columns if c not in non_feat]

    logger.info("Aggregating %d climate columns to state-month", len(base_cols))
    df_state = (
        df.groupby(["state", "year", "month", "date"], observed=True)[base_cols]
        .mean()
        .reset_index()
    )

    for col in base_cols:
        if pd.api.types.is_numeric_dtype(df_state[col]):
            clim_col = f"{col}_clim"
            df_state[clim_col] = (
                df_state.groupby(["state", "month"], observed=True)[col]
                .transform("mean")
            )
            df_state[f"{col}_anom"] = df_state[col] - df_state[clim_col]
    return df_state


def merge_climate_outages(clim_state: pd.DataFrame, outages: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging climate and outage data ...")
    merged = clim_state.merge(
        outages,
        on=["state", "year", "month", "date"],
        how="left",
        validate="1:1"
    )
    before = len(merged)
    # For historical years (< first forecast year), require pct_out present;
    # keep future climate rows even if pct_out is missing
    hist_mask = merged["year"] < FORECAST_Y0
    merged = merged.loc[~hist_mask | merged["pct_out"].notna()].copy()
    logger.info(
        "Merged rows: %d -> %d after filtering historical pct_out",
        before,
        len(merged)
    )
    return merged

# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    anomaly_cols = [c for c in df.columns if c.endswith("_anom")]
    index_cols = [c for c in ("TNA_ASO", "Nino34_ASO", "AMO_Annual") if c in df.columns]
    fixed_cols = ["Altitude"]; fixed_cols = [c for c in fixed_cols if c in df.columns]

    df = df.copy().sort_values(["state", "date"])

    # Create lagged anomaly features only
    lagged_cols: List[str] = []
    for col in anomaly_cols:
        for lag in LAG_MONTHS:
            new_col = f"{col}_lag{lag}"
            df[new_col] = df.groupby("state")[col].shift(lag)
            lagged_cols.append(new_col)

    # Generate cyclic time features
    df["month_sin"] = np.sin(2 * np.pi * df["month"]/12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]/12)

    # Assemble feature list without pct_out lags
    feat_cols = anomaly_cols + lagged_cols + index_cols + fixed_cols + ["month_sin", "month_cos"]

    # Forward/backward fill and drop any remaining NaNs
    for col in feat_cols:
        if col in df.columns:
            df[col] = df.groupby("state")[col].ffill().bfill()
    df.dropna(subset=feat_cols, inplace=True)

    logger.info("Engineered features, final shape %s | feature cols: %d", df.shape, len(feat_cols))
    return df, feat_cols

# =========================================================
# 5. DATA SPLIT
# =========================================================

def split_sets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df.year <= TRAIN_END_Y].copy()
    test_df = df[(df.year >= TEST_START_Y) & (df.year <= 2024)].copy()
    future_df = df[(df.year >= FORECAST_Y0) & (df.year <= FORECAST_Y1)].copy()
    logger.info("Split sets -> train: %d, test: %d, future: %d", len(train_df), len(test_df), len(future_df))
    return train_df, test_df, future_df

# =========================================================
# 6. MODEL TRAINING
# =========================================================

MODEL_DIR = PROJECT_ROOT / "output/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_TEST_PATH = MODEL_DIR / "xgb_pct_out_test.joblib"
MODEL_FULL_PATH = MODEL_DIR / "xgb_pct_out_full.joblib"
SCALER_PATH = MODEL_DIR / "scaler_pct_out.joblib"


def train_models(train_df: pd.DataFrame, test_df: pd.DataFrame, all_df: pd.DataFrame, feat_cols: List[str]):
    scaler = StandardScaler()
    X_train = train_df[feat_cols].values
    y_train = train_df["pct_out"].values
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(test_df[feat_cols].values) if not test_df.empty else None
    y_test = test_df["pct_out"].values if not test_df.empty else None

    eval_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    eval_model.fit(X_train_scaled, y_train)
    joblib.dump(eval_model, MODEL_TEST_PATH)
    joblib.dump(scaler, SCALER_PATH)

    if X_test_scaled is not None:
        preds = eval_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        logger.info("Eval-model MAE: %.4f | RMSE: %.4f on 2022-24 set", mae, rmse)
    else:
        logger.warning("No test rows -> skipped evaluation metrics")

    hist_df = pd.concat([train_df, test_df], ignore_index=True)
    hist_df = hist_df[hist_df["pct_out"].notna()]
    X_full = hist_df[feat_cols].values
    y_full = hist_df["pct_out"].values

    # scale

    X_full_scaled = scaler.transform(X_full)

    full_model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    full_model.fit(X_full_scaled, y_full)
    joblib.dump(full_model, MODEL_FULL_PATH)
    logger.info("Saved eval model -> %s | full model -> %s", MODEL_TEST_PATH.name, MODEL_FULL_PATH.name)

    imp = pd.Series(full_model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    logger.info("Top 10 feature importances (full model):\n%s", imp.head(10).to_string())

    return eval_model, full_model, scaler

# =========================================================
# 7. FORECAST
# =========================================================
FORECAST_CSV = PROJECT_ROOT / "output/analysis/future_pct_out_by_state.csv"

def forecast_future(full_model, scaler, future_df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    logger.info("Forecasting future pct_out for %d rows...", len(future_df))
    if future_df.empty:
        logger.warning("No future rows to forecast, skipping.")
        return pd.DataFrame()
    Xf = future_df[feat_cols].values
    Xf_scaled = scaler.transform(Xf)
    preds = full_model.predict(Xf_scaled)
    df_f = future_df[["state", "date"]].copy()
    df_f["pred_pct_out"] = preds
    df_agg = (
        df_f.groupby(["state", pd.Grouper(key="date", freq="MS")], observed=True)["pred_pct_out"]
        .mean()
        .reset_index()
    )
    FORECAST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(FORECAST_CSV, index=False)
    logger.info("Saved future forecast to %s", FORECAST_CSV)
    return df_agg

# =========================================================
# 8. METRICS / DIAGNOSTICS
# =========================================================
METRICS_DIR = PROJECT_ROOT / "output/analysis"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
STATE_MAE_CSV = METRICS_DIR / "state_mae.csv"
OVERALL_METRICS_TXT = METRICS_DIR / "overall_metrics.txt"

def diagnostics(eval_model, scaler, train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: List[str]):
    logger.info("Running diagnostics...")
    # run per‐set MAE
    for df_set, name in [(train_df, "train"), (test_df, "test")]:
        if df_set.empty:
            logger.warning("%s set empty, skipping metrics", name)
            continue
        X = df_set[feat_cols].values
        y = df_set["pct_out"].values
        Xs = scaler.transform(X)
        preds = eval_model.predict(Xs)
        mae = mean_absolute_error(y, preds)
        logger.info("%s MAE: %.4f", name, mae)
        print(f"{name.capitalize()} MAE: {mae:.4f}")
        if name == "test":
            df_tmp = df_set.copy()
            df_tmp["pred"] = preds
            state_mae = (
                df_tmp.groupby("state", observed=True)
                .apply(lambda g: mean_absolute_error(g["pct_out"], g["pred"]))
                .reset_index(name="mae")
            )
            state_mae.to_csv(STATE_MAE_CSV, index=False)
            logger.info("Saved state MAE to %s", STATE_MAE_CSV)
    # print overall metrics instead of saving to file
    overall = []
    train_mae = mean_absolute_error(
        train_df["pct_out"], eval_model.predict(scaler.transform(train_df[feat_cols].values))
    )
    overall.append(f"Train MAE: {train_mae:.4f}")
    print(f"Overall Train MAE: {train_mae:.4f}")
    if not test_df.empty:
        test_mae = mean_absolute_error(
            test_df["pct_out"], eval_model.predict(scaler.transform(test_df[feat_cols].values))
        )
        overall.append(f"Test MAE: {test_mae:.4f}")
        print(f"Overall Test MAE: {test_mae:.4f}")
    # no file write

# =========================================================
# 9. MAP PLOT
# =========================================================
def plot_state_pct_out_change(df_agg: pd.DataFrame) -> None:
    """
    Choropleth of projected average %‑out change (2035 vs 2025).

    ‑3 pp … +3 pp diverging scale
      < 0  → shades of red
       0   → grey
      > 0  → shades of green
    """

    # ── 1. summarise start/end averages ──────────────────────────
    df_agg["year"] = df_agg["date"].dt.year
    start = (
        df_agg[df_agg["year"] == FORECAST_Y0]
        .groupby("state", observed=True)["pred_pct_out"]
        .mean()
        .reset_index(name="start")
    )
    end = (
        df_agg[df_agg["year"] == FORECAST_Y1]
        .groupby("state", observed=True)["pred_pct_out"]
        .mean()
        .reset_index(name="end")
    )
    df_chg = start.merge(end, on="state")
    df_chg["change"] = df_chg["end"] - df_chg["start"]

    # ── 2. geometry (dissolve counties → states) ─────────────────
    gdf = gpd.read_file(SHAPEFILE_PATH)
    gdf["state_fips"] = gdf["STATEFP"].str.zfill(2)
    gdf_states = (
        gdf[gdf["state_fips"].isin(TARGET_STATES)]
        .dissolve(by="state_fips")
        .reset_index()
        .rename(columns={"state_fips": "state"})
    )

    gdf_states = gdf_states.merge(df_chg, on="state", how="left")

    # ── 3. colour map with fixed bounds (‑3 … +3 pp) ─────────────
    vmin, vmax = -3.0, 3.0
    #    red → grey → green
    cmap = colors.LinearSegmentedColormap.from_list(
        "red_grey_green", ["#b40426", "#cccccc", "#006837"], N=256
    )
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # ── 4. plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf_states.plot(
        column="change",
        cmap=cmap,
        norm=norm,
        linewidth=0.2,
        edgecolor="0.7",
        ax=ax,
        legend=True,
        legend_kwds={
            "label": f"% Out Change {FORECAST_Y0}-{FORECAST_Y1}",
            "orientation": "horizontal",
            "shrink": 0.7,
            "extend": "both",
        },
        missing_kwds={"color": "lightgrey", "label": "Missing"},
    )

    ax.set_title(f"Projected % Out Change {FORECAST_Y0}-{FORECAST_Y1}", fontsize=15)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================



# =========================================================
# MAIN
# =========================================================
def plot_customers_out_bars(outage_xlsx: Path,
                            future_pct_df: pd.DataFrame,
                            start_year_hist: int = 2017,
                            end_year_future: int = FORECAST_Y1) -> None:
    """
    Build two bar charts:
      1) yearly sum of *customers out* across all target‑states
      2) monthly sum of *customers out* across all target‑states

    ‑ Historical part uses actual MaxCustomersOutTotal from the workbook
    ‑ Future part multiplies last‑known CustomersTrackedTotal per‑state by
      the model‑projected percentage outage.
    """
    # ---------- read raw outage workbook (again – only the needed cols) ----
    cols = ["STATE", "Year", "Month", "MaxCustomersOutTotal",
            "CustomersTrackedTotal"]
    df_raw = pd.read_excel(outage_xlsx, sheet_name=0, usecols=cols)
    df_raw["state"] = df_raw["STATE"].map(STATE_NAME_TO_FIPS)
    df_raw = df_raw.dropna(subset=["state"])                 # keep mapped rows
    df_raw["date"] = pd.to_datetime(dict(year=df_raw.Year,
                                         month=df_raw.Month,
                                         day=1))
    # ---------- HISTORICAL customers‑out -------------------------------
    df_hist = (df_raw
               .loc[df_raw["Year"] >= start_year_hist,
                    ["state", "date", "MaxCustomersOutTotal"]]
               .rename(columns={"MaxCustomersOutTotal": "cust_out"}))

    # ---------- FUTURE customers‑out (approx) --------------------------
    if future_pct_df.empty:
        logger.warning("future_pct_df is empty – future bars will be omitted")
        df_future = pd.DataFrame(columns=["state", "date", "cust_out"])
    else:
        # latest tracked total per‑state (last 3 yrs of history)
        latest_tracked = (df_raw[df_raw["Year"] >= TRAIN_END_Y]
                          .groupby("state")["CustomersTrackedTotal"]
                          .mean())
        df_future = future_pct_df.copy()
        df_future["tracked"] = df_future["state"].map(latest_tracked)
        df_future["cust_out"] = df_future["pred_pct_out"] / 100.0 \
                                * df_future["tracked"]
        df_future = df_future[["state", "date", "cust_out"]]

    # ---------- CONCAT  ------------------------------------------------
    df_all = pd.concat([df_hist, df_future], ignore_index=True)

    # ---------- AGG – YEARLY ------------------------------------------
    df_year = (df_all
               .assign(year=lambda d: d.date.dt.year)
               .groupby("year", as_index=False)["cust_out"].sum())

    plt.figure(figsize=(12, 5))
    sns.barplot(data=df_year, x="year", y="cust_out",
                palette="Blues_d", edgecolor="k")
    plt.xticks(rotation=45, ha="right")
    plt.title("Total Customers Out ‑ All Target States (Annual)")
    plt.ylabel("Customers Out")
    plt.tight_layout()
    plt.show()

    # ---------- AGG – MONTHLY -----------------------------------------
    df_mon = (df_all
              .groupby("date", as_index=False)["cust_out"].sum())

    plt.figure(figsize=(14, 5))
    sns.barplot(data=df_mon, x="date", y="cust_out",
                palette="Blues_r", edgecolor="k")
    plt.xticks(rotation=45, ha="right")
    plt.title("Total Customers Out ‑ All Target States (Monthly)")
    plt.ylabel("Customers Out")
    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------------------



# ---------------------------------------------------------------------
# UPDATED  main()  – just the final lines shown here


def main():
    clim_raw = load_climate(CLIMATE_FILE, TARGET_STATES)
    outages = load_outages(OUTAGE_EXCEL, TARGET_STATES)
    clim_state = aggregate_climate_state(clim_raw)
    merged = merge_climate_outages(clim_state, outages)
    fe_df, feature_cols = engineer_features(merged)

    train_df, test_df, future_df = split_sets(fe_df)
    logger.info(
        "Data split complete | train years ≤%d: %d rows | test %d-24: %d rows | future %d-%d: %d rows",
        TRAIN_END_Y, len(train_df), TEST_START_Y, len(test_df), FORECAST_Y0, FORECAST_Y1, len(future_df),
    )

    eval_m, full_m, scaler = train_models(train_df, test_df, fe_df, feature_cols)
    df_future_agg = forecast_future(full_m, scaler, future_df, feature_cols)
    diagnostics(eval_m, scaler, train_df, test_df, feature_cols)
    plot_state_pct_out_change(df_future_agg)
    plot_customers_out_bars(OUTAGE_EXCEL, df_future_agg)


    return train_df, test_df, future_df, feature_cols, df_future_agg

if __name__ == "__main__":
    main()
