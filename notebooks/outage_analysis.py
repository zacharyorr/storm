# --- Python Script: 08_outage_analysis.py ---
# Purpose: Links predicted storm likelihood to state-level power outages and projects future outages.

import pandas as pd
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb # For loading model object structure
from sklearn.calibration import CalibratedClassifierCV # For loading model object structure
import statsmodels.formula.api as smf # For linking model example
import statsmodels.api as sm # For loading saved OLS model
import gc
import requests # For loading indices if needed
from io import StringIO # For loading indices if needed
import warnings
import geopandas as gpd # For mapping

# --- Determine Project Root based on script location ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- Configuration ---\
# --- Input Paths (relative to PROJECT_ROOT) ---
CLIMATE_FILE_HIST_FUTURE = os.path.join(PROJECT_ROOT, "output", "processed_climate", "county_monthly_climate_variables_target_states.parquet") # Should contain historical AND future
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "models", "scs_wind_target_AnomInd_lgbm_calibrated.joblib") # Calibrated binary classifier path
SCALER_PATH = os.path.join(PROJECT_ROOT, "output", "models", "scs_wind_target_states_scaler.joblib") # Scaler path used for the model
OUTAGE_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "outages", "Outages.xlsx") # Corrected filename casing & absolute path
COUNTY_SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, "data", "geospatial", "tl_2024_us_county.shp") # For state map generation
OUTAGE_SHEET_NAME = 0 # Read first sheet

# --- Climate Index URLs ---
AMO_URL = "https://psl.noaa.gov/data/correlation/amon.us.data"
TNA_URL = "https://psl.noaa.gov/data/correlation/tna.data"
NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.data"

# --- Parameters ---
TARGET_EVENT_TYPE = 'scs_wind'
target_state_fips = [
    "12", "20", "29", "40", "48", "05", "22", "28",
    "01", "13", "45", "37", "47", "51"
]
TARGET_OUTAGE_METRICS = ['CustomerHoursOutTotal', 'MaxCustomersOutTotal', 'CustomersTrackedTotal']
TARGET_LINKING_METRIC = 'peak_pct_out' # Outage metric to analyze/model/project
GEOID_COLUMN_IN_SHAPEFILE = "GEOID"  # Column name for County FIPS in shapefile
STATE_FIPS_COLUMN_IN_SHAPEFILE = "STATEFP" # Column name for State FIPS in shapefile
STATE_NAME_TO_FIPS = { # Add ALL states present in outage file that are also in target_state_fips
    'Florida': '12', 'Kansas': '20', 'Missouri': '29', 'Oklahoma': '40',
    'Texas': '48', 'Arkansas': '05', 'Louisiana': '22', 'Mississippi': '28',
    'Alabama': '01', 'Georgia': '13', 'South Carolina': '45',
    'North Carolina': '37', 'Tennessee': '47', 'Virginia': '51'
    # Add others if necessary
}
MAX_EVALUATION_YEAR = 2024 # Last year of historical climate/event data to use for analysis/linking model
HISTORICAL_START_YEAR_MODEL = 2015 # First year of climate features for model input
LAG_MONTHS = [1, 2, 3, 6] # Define lag months for feature engineering - ADDED LAG 6
FUTURE_PROJECTION_START_YEAR = 2025
FUTURE_PROJECTION_END_YEAR = 2035

# --- Output Paths (relative to PROJECT_ROOT) ---
ANALYSIS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "analysis")
LINKING_MODEL_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, f"link_model_{TARGET_EVENT_TYPE}_{TARGET_LINKING_METRIC}.pickle")
MERGED_DATA_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, f"merged_hist_{TARGET_EVENT_TYPE}_outage_pred.csv") # Save historical merged data
CORR_PLOT_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, f"corr_hist_{TARGET_EVENT_TYPE}_{TARGET_LINKING_METRIC}.png")
HIST_ACTUAL_VS_PRED_PLOT_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, f"link_model_eval_{TARGET_EVENT_TYPE}_{TARGET_LINKING_METRIC}.png")
FUTURE_OUTAGE_PROJ_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, f"future_proj_{TARGET_LINKING_METRIC}_by_state_{FUTURE_PROJECTION_START_YEAR}-{FUTURE_PROJECTION_END_YEAR}.csv")
STATE_OUTAGE_CHANGE_DRIVERS_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, f"state_level_change_drivers_{TARGET_LINKING_METRIC}_{FUTURE_PROJECTION_START_YEAR}-{FUTURE_PROJECTION_END_YEAR}.csv") # For change analysis
STATE_OUTAGE_CHANGE_MAP_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, f"map_state_outage_change_{TARGET_LINKING_METRIC}_{FUTURE_PROJECTION_START_YEAR}-{FUTURE_PROJECTION_END_YEAR}.png")

# --- Setup Logger (relative to PROJECT_ROOT) ---
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "08_outage_analysis_script_log.log") # Separate log file
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) # Ensure log directory exists
logger = logging.getLogger("OutageAnalysisScriptLogger")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, mode='w'); fh.setLevel(logging.DEBUG); fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"); fh.setFormatter(fh_formatter); logger.addHandler(fh)
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"); sh.setFormatter(sh_formatter); logger.addHandler(sh)

# --- Helper Function to Load Indices (Copied from Notebook Cell 3.5) ---
def load_psl_index(url, missing_val=-99.99, date_col_name='time'):
    """ Loads and parses space-delimited text index files from NOAA PSL. """
    logger.info(f"Attempting load from: {url}")
    df_final = pd.DataFrame(columns=[date_col_name, 'Index_Value'])
    df_final[date_col_name] = pd.to_datetime(df_final[date_col_name])
    try:
        logger.debug(f"Fetching raw content (SSL Verification Disabled)...")
        # WARNING: verify=False disables SSL certificate verification.
        # This might be necessary for some NOAA sites but introduces a security risk.
        # Consider downloading the data manually or configuring certificate verification properly if possible.
        response = requests.get(url, verify=False) # Skip SSL check
        response.raise_for_status()
        data_io = StringIO(response.text)
        col_names = ['Year_Str'] + list(range(1, 13))
        df_index = pd.read_csv( data_io, sep=r'\s+', skiprows=1, header=None, names=col_names, na_values=[str(missing_val), missing_val, '-999', '-999.0', '-99.99'] )
        logger.debug(f"Read {len(df_index)} lines.")
        df_index['Year'] = pd.to_numeric(df_index['Year_Str'], errors='coerce')
        rows_before_drop = len(df_index); df_index.dropna(subset=['Year'], inplace=True); rows_after_drop = len(df_index)
        if rows_before_drop != rows_after_drop: logger.warning(f"Dropped {rows_before_drop - rows_after_drop} non-numeric Year rows.")
        if df_index.empty: raise ValueError("No valid Year rows.")
        df_index['Year'] = df_index['Year'].astype(int)
        id_vars = ['Year']; value_vars = list(range(1, 13))
        df_long = df_index.melt(id_vars=id_vars, value_vars=value_vars, var_name='Month', value_name='Index_Value')
        df_long['Month'] = pd.to_numeric(df_long['Month'], errors='coerce'); df_long.dropna(subset=['Month'], inplace=True); df_long['Month'] = df_long['Month'].astype(int)
        df_long['Index_Value'] = pd.to_numeric(df_long['Index_Value'], errors='coerce'); df_long.dropna(subset=['Index_Value'], inplace=True)
        if df_long.empty: raise ValueError("No valid numeric Index_Value.")
        date_str = df_long['Year'].astype(str) + '-' + df_long['Month'].astype(str).str.zfill(2) + '-01'
        df_long['time'] = pd.to_datetime(date_str, format='%Y-%m-%d')
        df_final = df_long[['time', 'Index_Value']].sort_values('time').reset_index(drop=True)
        logger.info(f"Processed index: {os.path.basename(url)}. Shape: {df_final.shape}")
    except requests.exceptions.RequestException as req_e: logger.exception(f"HTTP Request failed for {url}: {req_e}")
    except Exception as e: logger.exception(f"Failed load/process {url}: {e}")
    return df_final

# --- Function to Load & Clean Outage Data ---
def load_clean_outages(file_path, sheet_name, state_map, target_fips, metric_cols):
    logger.info(f"Loading outage data from {file_path}, sheet '{sheet_name}'...")
    if not os.path.exists(file_path): raise FileNotFoundError(f"Outage file not found: {file_path}")
    df_outages_raw = pd.read_excel(file_path, sheet_name=sheet_name, dtype={'CountyFIPS': str})
    logger.info(f"Loaded {len(df_outages_raw)} records.")

    req_cols = ['STATE', 'Year', 'Month'] + metric_cols
    if not all(col in df_outages_raw.columns for col in req_cols): raise ValueError("Outage data missing required columns.")

    df_outages_raw['state_fips'] = df_outages_raw['STATE'].map(state_map)
    mapped = df_outages_raw['state_fips'].notna().sum(); unmapped = df_outages_raw[df_outages_raw['state_fips'].isna()]['STATE'].unique()
    logger.info(f"Mapped {mapped} records to FIPS.");
    if len(unmapped) > 0: logger.warning(f"Unmapped states: {unmapped}.")

    df_outages_state = df_outages_raw[df_outages_raw['state_fips'].isin(target_fips)].copy()
    logger.info(f"Filtered to {len(df_outages_state)} records for target states.")
    if df_outages_state.empty: return pd.DataFrame() # Return empty if no target state data

    df_outages_state['Year'] = pd.to_numeric(df_outages_state['Year'], errors='coerce'); df_outages_state['Month'] = pd.to_numeric(df_outages_state['Month'], errors='coerce')
    df_outages_state.dropna(subset=['Year', 'Month'], inplace=True)
    df_outages_state['time'] = pd.to_datetime(df_outages_state[['Year', 'Month']].astype(int).assign(DAY=1))

    for col in metric_cols: df_outages_state[col] = pd.to_numeric(df_outages_state[col], errors='coerce')
    logger.info("Dropping rows with NaN in key numeric metrics/time...")
    key_num_cols = ['time'] + metric_cols
    df_outages_state.dropna(subset=[col for col in key_num_cols if col in df_outages_state.columns], inplace=True)

    # Calculate peak pct out
    tracked = df_outages_state['CustomersTrackedTotal']; max_out = 'MaxCustomersOutTotal'
    if max_out in df_outages_state.columns:
        out = df_outages_state[max_out]; df_outages_state['peak_pct_out'] = np.where(tracked.notna() & (tracked > 0) & out.notna(), (out / tracked) * 100, np.nan)
    else: df_outages_state['peak_pct_out'] = np.nan
    # Fix FutureWarning: Use assignment instead of inplace=True on a slice
    df_outages_state['peak_pct_out'] = df_outages_state['peak_pct_out'].fillna(0)

    final_cols = ['state_fips', 'time', 'Year', 'Month'] + metric_cols + ['peak_pct_out']
    df_outages_state = df_outages_state[[col for col in final_cols if col in df_outages_state.columns]].copy()
    df_outages_state.sort_values(by=['state_fips', 'time'], inplace=True)
    logger.info("Outage data cleaned.")
    return df_outages_state

# --- Function to Prepare Climate Features ---
def prepare_climate_features(climate_df, indices_df, fixed_vars, tv_vars, lag_months):
    logger.info("Preparing climate features (anomalies, lags, indices)...")
    df = climate_df.copy()
    df['year'] = df['time'].dt.year; df['month'] = df['time'].dt.month
    anomaly_cols = []
    all_climate_cols = tv_vars + fixed_vars
    for var in all_climate_cols:
        # Check if column exists before processing
        if var not in df.columns:
            logger.warning(f"Variable '{var}' listed in configuration not found in climate data. Skipping.")
            continue
        clim_col = f'{var}_clim'; anom_col = f'{var}_anom'
        # Calculate climatology only if there are enough non-NaN values
        if df[var].notna().any():
            df[clim_col] = df.groupby(['county_geoid', 'month'])[var].transform('mean')
            df[anom_col] = df[var] - df[clim_col]; anomaly_cols.append(anom_col)
        else:
            logger.warning(f"Variable '{var}' is all NaN. Cannot calculate climatology/anomaly.")
            df[clim_col] = np.nan
            df[anom_col] = np.nan


    indices_df['year'] = indices_df['year'].astype(df['year'].dtype) # Match type
    df = pd.merge(df, indices_df, on='year', how='left')
    index_cols = [col for col in indices_df.columns if col != 'year']

    impute_cols = anomaly_cols + index_cols + fixed_vars
    impute_cols = [col for col in impute_cols if col in df.columns] # Ensure columns exist
    logger.info(f"Imputing NaNs in features...")
    df.sort_values(by=['county_geoid', 'time'], inplace=True)
    anom_impute = [c for c in anomaly_cols if c in df.columns]; index_impute = [c for c in index_cols if c in df.columns]; fixed_impute = [c for c in fixed_vars if c in df.columns]
    if anom_impute: df[anom_impute] = df.groupby('county_geoid')[anom_impute].ffill().bfill()
    if index_impute: df[index_impute] = df[index_impute].ffill().bfill()
    if fixed_impute:
        for fv in fixed_impute:
             if df[fv].isnull().any(): med = df[fv].median(); df[fv].fillna(med, inplace=True)
    df.dropna(subset=impute_cols, inplace=True) # Drop any rows still having NaNs

    logger.info(f"Creating lagged features: {lag_months}...")
    lag_cols = []
    # Use only anomaly columns that were successfully created
    tv_anom_cols = [f'{var}_anom' for var in tv_vars if f'{var}_anom' in df.columns]
    if tv_anom_cols:
        for lag in lag_months:
            for anom_var in tv_anom_cols: lag_col_name = f'{anom_var}_lag{lag}'; df = df.assign(**{lag_col_name: df.groupby('county_geoid')[anom_var].shift(lag)}); lag_cols.append(lag_col_name)

    df['month_sin'] = np.sin(2 * np.pi * df['month']/12); df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    lag_cols_exist = [col for col in lag_cols if col in df.columns]
    if lag_cols_exist: df.dropna(subset=lag_cols_exist, inplace=True)
    logger.info("Feature engineering complete.")
    return df

# --- Function to Plot State-Level Outage Change --- MODIFIED ---
def plot_state_outage_change(gdf_states_plot, output_path, start_yr, end_yr, metric_col_name, change_col_name):
    """Plots the change in a metric on a state map. Expects a GeoDataFrame with geometry and the change data.

    Args:
        gdf_states_plot (GeoDataFrame): GeoDataFrame with state geometries and the change column.
        output_path (str): Path to save the map image.
        start_yr (int): Start year for the change calculation.
        end_yr (int): End year for the change calculation.
        metric_col_name (str): Base name of the metric being plotted (e.g., 'Projected_peak_pct_out').
        change_col_name (str): Name of the column in gdf_states_plot containing the change values.
    """
    logger.info(f"--- Generating State-Level Change Map ({start_yr} vs {end_yr}) ---")
    try:
        if gdf_states_plot is None or gdf_states_plot.empty:
            raise ValueError("Input GeoDataFrame for plotting is empty or None.")
        if change_col_name not in gdf_states_plot.columns:
            raise ValueError(f"Change column '{change_col_name}' not found in GeoDataFrame.")

        logger.info("Creating choropleth map...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Use RdYlGn: Red = Decrease (-), Yellow = ~0, Green = Increase (+)
        gdf_states_plot.plot(column=change_col_name,
                             ax=ax,
                             legend=True,
                             legend_kwds={'label': f"Change in Avg Annual {metric_col_name} ({end_yr} vs {start_yr})\n(Green=Increase, Red=Decrease)",
                                          'orientation': "horizontal"},
                             cmap='RdYlGn', # Flipped colormap
                             missing_kwds={"color": "lightgrey", "label": "Missing Data"})
        ax.set_axis_off()
        ax.set_title(f"Projected Change in Avg Annual {metric_col_name} ({end_yr} vs {start_yr}) - Target States", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Saved state outage change map to {output_path}")
        plt.close(fig) # Close plot to free memory

    except Exception as e:
        logger.exception(f"Failed to generate state map: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Outage Analysis Pipeline...")
    ols_model = None # Initialize linking model variable

    # --- Step 1: Load and Clean Outage Data ---
    try:
        df_outages_monthly_agg = load_clean_outages(OUTAGE_FILE_PATH, OUTAGE_SHEET_NAME, STATE_NAME_TO_FIPS, target_state_fips, TARGET_OUTAGE_METRICS)
        if df_outages_monthly_agg.empty:
            logger.error("No valid outage data for target states. Exiting.")
            exit() # Cannot proceed without outage data
    except Exception as e:
        logger.exception("Failed during outage data loading/cleaning.")
        raise e

    # --- Step 2: Load FULL Climate Data (Historical + Future) ---
    logger.info(f"Loading FULL climate features (Hist+Future) from: {CLIMATE_FILE_HIST_FUTURE}")
    try:
        if not os.path.exists(CLIMATE_FILE_HIST_FUTURE):
            raise FileNotFoundError(f"Climate data file not found: {CLIMATE_FILE_HIST_FUTURE}")
        if CLIMATE_FILE_HIST_FUTURE.endswith(".parquet"): df_climate_raw = pd.read_parquet(CLIMATE_FILE_HIST_FUTURE)
        else: df_climate_raw = pd.read_csv(CLIMATE_FILE_HIST_FUTURE, parse_dates=['time'])

        df_climate_raw['county_geoid'] = df_climate_raw['county_geoid'].astype(str).str.zfill(5)
        df_climate_raw['time'] = pd.to_datetime(df_climate_raw['time']).dt.normalize()
        df_climate_raw['year'] = df_climate_raw['time'].dt.year
        # Filter only for target states here
        df_climate_raw['state_fips'] = df_climate_raw['county_geoid'].str[:2]
        df_climate_raw = df_climate_raw[df_climate_raw['state_fips'].isin(target_state_fips)].copy()
        logger.info(f"Loaded raw climate data (Hist+Future) shape: {df_climate_raw.shape}. Year range: {df_climate_raw['year'].min()}-{df_climate_raw['year'].max()}")
    except Exception as e: logger.exception(f"Failed loading climate data: {e}"); raise e

    # --- Step 3: Load Climate Indices ---
    logger.info("Loading climate indices...")
    df_amo = load_psl_index(AMO_URL); df_tna = load_psl_index(TNA_URL); df_nino = load_psl_index(NINO34_URL)
    if df_amo.empty or df_tna.empty or df_nino.empty: raise ValueError("Failed to load one or more indices.")
    # Calculate yearly/seasonal averages
    df_amo_yr = df_amo.set_index('time').resample('YS-JAN')['Index_Value'].mean().reset_index().rename(columns={'Index_Value':'AMO_Annual'}); df_amo_yr['year'] = df_amo_yr['time'].dt.year
    def calc_aso(df, name): df_c=df.copy(); df_c['year']=df_c['time'].dt.year; df_c['month']=df_c['time'].dt.month; df_aso=df_c[df_c['month'].isin([8,9,10])]; return df_aso.groupby('year')['Index_Value'].mean().reset_index().rename(columns={'Index_Value':name})
    df_tna_aso = calc_aso(df_tna, 'TNA_ASO'); df_nino_aso = calc_aso(df_nino, 'Nino34_ASO')
    df_indices = pd.merge(df_tna_aso, df_nino_aso, on='year', how='outer').merge(df_amo_yr[['year','AMO_Annual']], on='year', how='outer')
    logger.info(f"Processed climate indices. Shape: {df_indices.shape}")

    # --- Step 4: Engineer Features on ALL Climate Data ---
    tv_vars = ['Max Temp', 'Spec Humid', 'Sens Heat Flux', 'Precip', 'Sea Level Press', 'Soil Moisture', 'Wind Speed']
    fixed_vars = ['Altitude']
    try:
        df_all_featured = prepare_climate_features(df_climate_raw, df_indices, fixed_vars, tv_vars, LAG_MONTHS)
        logger.info(f"Engineered features for all data. Shape: {df_all_featured.shape}")
        del df_climate_raw; gc.collect() # Free memory
    except Exception as e:
        logger.exception("Failed during feature engineering.")
        raise e

    # --- Step 5: Load Model & Scaler ---
    logger.info("Loading scaler and storm model...")
    try:
         scaler = joblib.load(SCALER_PATH)
         model = joblib.load(MODEL_PATH)
         feature_cols = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None
         if feature_cols is None: raise ValueError("Scaler did not have feature names stored.")
         logger.info(f"Loaded scaler and model. Model expects {len(feature_cols)} features.")
         # Verify all expected features are in the engineered data
         missing_eng_features = [f for f in feature_cols if f not in df_all_featured.columns]
         if missing_eng_features:
              raise ValueError(f"Engineered features are missing columns required by the scaler: {missing_eng_features}")
    except Exception as e: logger.exception(f"Failed loading model/scaler: {e}"); raise e

    # --- Step 6: Split into Historical and Future Data ---
    logger.info(f"Splitting data into Historical (<={MAX_EVALUATION_YEAR}) and Future ({FUTURE_PROJECTION_START_YEAR}-{FUTURE_PROJECTION_END_YEAR})...")
    df_hist_featured = df_all_featured[df_all_featured['year'] <= MAX_EVALUATION_YEAR].copy()
    df_future_featured = df_all_featured[
        (df_all_featured['year'] >= FUTURE_PROJECTION_START_YEAR) &
        (df_all_featured['year'] <= FUTURE_PROJECTION_END_YEAR)
    ].copy()
    logger.info(f"Historical features shape: {df_hist_featured.shape}")
    logger.info(f"Future features shape: {df_future_featured.shape}")
    del df_all_featured; gc.collect()

    # --- Step 7: Generate Historical Storm Predictions ---
    logger.info("Generating historical predictions...")
    df_hist_pred = pd.DataFrame()
    try:
         X_hist = df_hist_featured[feature_cols].copy() # Select features
         X_hist_scaled = scaler.transform(X_hist)
         with warnings.catch_warnings(): warnings.filterwarnings("ignore"); hist_pred_proba = model.predict_proba(X_hist_scaled)[:, 1]
         df_hist_pred = df_hist_featured[['county_geoid', 'time', 'state_fips']].copy()
         df_hist_pred['predicted_prob'] = hist_pred_proba
         logger.info("Historical predictions generated.")
         del X_hist, X_hist_scaled, hist_pred_proba; gc.collect()
    except Exception as e: logger.exception(f"Failed prediction generation: {e}"); raise e

    # --- Step 8: Aggregate Historical Predictions to State-Month ---
    logger.info("Aggregating historical predictions to state-month...")
    df_hist_state_monthly_expected = pd.DataFrame()
    if not df_hist_pred.empty:
        try:
             df_hist_state_monthly_expected = df_hist_pred.groupby(['state_fips', pd.Grouper(key='time', freq='MS')])['predicted_prob'].sum().reset_index()
             df_hist_state_monthly_expected.rename(columns={'predicted_prob': 'pred_expected_hits_hist'}, inplace=True)
             logger.info(f"Historical state-month prediction aggregation complete. Shape: {df_hist_state_monthly_expected.shape}")
        except Exception as e: logger.exception(f"Failed historical prediction aggregation: {e}"); raise e
    else:
        logger.warning("Skipping historical aggregation as historical predictions are empty.")


    # --- Step 9: Merge Historical Outage & Prediction Data ---
    logger.info("Merging historical outage metrics and storm predictions...")
    df_merged_analysis = pd.DataFrame()
    if not df_outages_monthly_agg.empty and not df_hist_state_monthly_expected.empty:
        try:
             df_outages_monthly_agg['time'] = pd.to_datetime(df_outages_monthly_agg['time'])
             df_hist_state_monthly_expected['time'] = pd.to_datetime(df_hist_state_monthly_expected['time'])
             df_merged_analysis = pd.merge(df_outages_monthly_agg, df_hist_state_monthly_expected, on=['state_fips', 'time'], how='inner')
             logger.info(f"Historical merge complete. Shape for analysis: {df_merged_analysis.shape}")
             if df_merged_analysis.empty: logger.warning("Historical merged dataframe is empty!")
             else:
                 # Save merged historical data
                 try:
                     os.makedirs(os.path.dirname(MERGED_DATA_PATH), exist_ok=True)
                     df_merged_analysis.to_csv(MERGED_DATA_PATH, index=False)
                     logger.info(f"Saved merged historical data to {MERGED_DATA_PATH}")
                 except Exception as save_e:
                     logger.warning(f"Could not save merged historical data: {save_e}")
        except Exception as e: logger.exception(f"Failed merging historical data: {e}"); raise e
    else:
        logger.warning("Skipping historical merge as one or both inputs are empty.")

    # --- Step 10: Analyze Historical Relationship & Fit Linking Model ---
    if not df_merged_analysis.empty:
        logger.info("--- Analyzing Historical Relationship ---")
        os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True) # Ensure analysis output directory exists
        predictor_col = 'pred_expected_hits_hist' # Column with predicted storm hits

        if TARGET_LINKING_METRIC not in df_merged_analysis.columns:
            logger.error(f"Target outage metric '{TARGET_LINKING_METRIC}' not found in merged data. Cannot perform analysis.")
        else:
             df_analysis = df_merged_analysis[[TARGET_LINKING_METRIC, predictor_col]].dropna()
             if len(df_analysis) > 1:
                  correlation = df_analysis[TARGET_LINKING_METRIC].corr(df_analysis[predictor_col])
                  logger.info(f"Correlation ({TARGET_LINKING_METRIC} vs Pred. Hits): {correlation:.4f}")
                  print(f"\nCorrelation ({TARGET_LINKING_METRIC} vs Pred. Hits): {correlation:.4f}")

                  # Scatter Plot
                  plt.figure(figsize=(8, 6)); sns.scatterplot(data=df_analysis, x=predictor_col, y=TARGET_LINKING_METRIC, alpha=0.5);
                  plt.title(f'Historical {TARGET_LINKING_METRIC} vs. Predicted Expected SCS Hits'); plt.xlabel("Predicted Expected SCS Hits (State-Month Sum)"); plt.ylabel(TARGET_LINKING_METRIC); plt.grid(True); plt.tight_layout();
                  plt.savefig(CORR_PLOT_PATH); logger.info(f"Saved correlation plot to {CORR_PLOT_PATH}"); plt.close() # Close plot

                  # Simple Linking Model (OLS Example)
                  try:
                       # Rename columns for statsmodels formula
                       df_analysis_renamed = df_analysis.rename(columns={TARGET_LINKING_METRIC: 'Outage_Metric', predictor_col: 'Pred_Expected_Hits'})
                       formula = f"Outage_Metric ~ Pred_Expected_Hits"
                       logger.info(f"Fitting OLS model: {formula}")
                       ols_model = smf.ols(formula, data=df_analysis_renamed).fit()
                       logger.info("OLS linking model fitted.")
                       print("\n--- Simple Linking Model Summary (OLS) ---"); print(ols_model.summary())
                       # Save linking model
                       try:
                           os.makedirs(os.path.dirname(LINKING_MODEL_PATH), exist_ok=True) # Ensure analysis output directory exists before saving
                           ols_model.save(LINKING_MODEL_PATH, remove_data=True); logger.info(f"Linking model saved to {LINKING_MODEL_PATH}")
                       except Exception as save_e: logger.warning(f"Could not save linking model: {save_e}")

                       # Add Actual vs Predicted plot for historical data
                       df_analysis_renamed['Linked_Pred_Outage'] = ols_model.predict(df_analysis_renamed[['Pred_Expected_Hits']])
                       plt.figure(figsize=(8, 6))
                       plt.scatter(df_analysis_renamed['Outage_Metric'], df_analysis_renamed['Linked_Pred_Outage'], alpha=0.3, s=10, label='State-Month')
                       plot_min = min(0, df_analysis_renamed['Outage_Metric'].min(), df_analysis_renamed['Linked_Pred_Outage'].min()) * 0.95
                       plot_max = max(df_analysis_renamed['Outage_Metric'].max(), df_analysis_renamed['Linked_Pred_Outage'].max()) * 1.05
                       plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='Ideal Fit (y=x)')
                       plt.xlabel(f"Actual Historical {TARGET_LINKING_METRIC}")
                       plt.ylabel(f"Predicted {TARGET_LINKING_METRIC} (Linked Model)")
                       plt.title('Linking Model Performance (Historical Data)')
                       plt.xlim(plot_min, plot_max); plt.ylim(plot_min, plot_max)
                       plt.grid(True); plt.legend(); plt.tight_layout();
                       plt.savefig(HIST_ACTUAL_VS_PRED_PLOT_PATH); logger.info(f"Saved linking model eval plot to {HIST_ACTUAL_VS_PRED_PLOT_PATH}"); plt.close()

                  except Exception as model_e: logger.exception(f"Failed linking model fitting/evaluation: {model_e}")
             else: logger.warning("Not enough historical data for correlation/regression after dropping NaNs.")
    else: logger.warning("Skipping historical analysis as merged dataframe is empty.")


    # --- Step 11: Generate Future Storm Predictions ---
    df_future_pred = pd.DataFrame()
    if not df_future_featured.empty:
        logger.info("Generating future storm predictions...")
        try:
            X_future = df_future_featured[feature_cols].copy()
            X_future_scaled = scaler.transform(X_future)
            with warnings.catch_warnings(): warnings.filterwarnings("ignore"); future_pred_proba = model.predict_proba(X_future_scaled)[:, 1]
            df_future_pred = df_future_featured[['county_geoid', 'time', 'state_fips']].copy()
            df_future_pred['predicted_prob'] = future_pred_proba
            logger.info(f"Future predictions generated. Shape: {df_future_pred.shape}")
            del X_future, X_future_scaled, future_pred_proba; gc.collect()
        except Exception as e:
            logger.exception(f"Failed future prediction generation: {e}")
            df_future_pred = pd.DataFrame() # Ensure empty on error
    else:
        logger.warning("No future climate data found. Skipping future predictions.")


    # --- Step 12: Aggregate Future Predictions to State-Month ---
    df_future_state_monthly_expected = pd.DataFrame()
    if not df_future_pred.empty:
        logger.info("Aggregating future predictions to state-month...")
        try:
            df_future_state_monthly_expected = df_future_pred.groupby(['state_fips', pd.Grouper(key='time', freq='MS')])['predicted_prob'].sum().reset_index()
            # Rename predictor column to match linking model input
            df_future_state_monthly_expected.rename(columns={'predicted_prob': 'Pred_Expected_Hits'}, inplace=True)
            logger.info(f"Future state-month prediction aggregation complete. Shape: {df_future_state_monthly_expected.shape}")
        except Exception as e:
            logger.exception(f"Failed future prediction aggregation: {e}")
            df_future_state_monthly_expected = pd.DataFrame() # Ensure empty on error
    else:
        logger.warning("Skipping future aggregation as future predictions are empty.")


    # --- Step 13: Project Future Outages using Linking Model ---
    df_future_outage_proj = pd.DataFrame()
    if not df_future_state_monthly_expected.empty:
        # Load the saved linking model if it wasn't fitted in this run (or failed)
        if ols_model is None:
            logger.info(f"OLS model not fitted in this run. Attempting to load from: {LINKING_MODEL_PATH}")
            try:
                if os.path.exists(LINKING_MODEL_PATH):
                    ols_model = sm.load(LINKING_MODEL_PATH)
                    logger.info("Successfully loaded saved linking model.")
                else:
                    logger.error("Saved linking model not found. Cannot project future outages.")
            except Exception as load_e:
                logger.exception(f"Failed to load linking model: {load_e}")
                ols_model = None # Ensure it's None if loading fails

        if ols_model is not None:
            logger.info(f"Applying linking model to project future '{TARGET_LINKING_METRIC}'...")
            try:
                # Ensure the predictor column exists
                if 'Pred_Expected_Hits' not in df_future_state_monthly_expected.columns:
                    raise ValueError("'Pred_Expected_Hits' column missing from future aggregated predictions.")

                # Predict using the loaded/fitted OLS model
                future_outage_predictions = ols_model.predict(df_future_state_monthly_expected[['Pred_Expected_Hits']])

                # Combine predictions with identifiers
                df_future_outage_proj = df_future_state_monthly_expected[['state_fips', 'time']].copy()
                df_future_outage_proj[f'Projected_{TARGET_LINKING_METRIC}'] = future_outage_predictions

                # Clip predictions if necessary (e.g., outage counts/pct cannot be negative)
                if TARGET_LINKING_METRIC in ['peak_pct_out', 'CustomerHoursOutTotal', 'MaxCustomersOutTotal']:
                    logger.info(f"Clipping negative projections for {TARGET_LINKING_METRIC} to 0.")
                    proj_col = f'Projected_{TARGET_LINKING_METRIC}'
                    df_future_outage_proj[proj_col] = np.maximum(0, df_future_outage_proj[proj_col])

                logger.info(f"Future outage projections generated. Shape: {df_future_outage_proj.shape}")

                # Save future projections
                try:
                     os.makedirs(os.path.dirname(FUTURE_OUTAGE_PROJ_PATH), exist_ok=True)
                     df_future_outage_proj.to_csv(FUTURE_OUTAGE_PROJ_PATH, index=False)
                     logger.info(f"Saved future outage projections to {FUTURE_OUTAGE_PROJ_PATH}")
                except Exception as save_e:
                     logger.warning(f"Could not save future outage projections: {save_e}")

            except Exception as proj_e:
                logger.exception(f"Failed applying linking model for future projections: {proj_e}")
                df_future_outage_proj = pd.DataFrame() # Ensure empty on error
        else:
            logger.warning("Linking model is not available. Skipping future outage projection.")
    else:
         logger.warning("Skipping future outage projection as future storm predictions are empty.")


    # --- Step 13.5: Analyze Drivers of Change ---
    state_changes_df = pd.DataFrame()
    if not df_future_outage_proj.empty and not df_future_state_monthly_expected.empty:
        logger.info(f"--- Analyzing Drivers of Change ({FUTURE_PROJECTION_START_YEAR} vs {FUTURE_PROJECTION_END_YEAR}) ---")
        try:
            # Calculate average annual values for start and end years
            proj_metric_col = f'Projected_{TARGET_LINKING_METRIC}'
            storm_hits_col = 'Pred_Expected_Hits'
            df_future_outage_proj['year'] = df_future_outage_proj['time'].dt.year
            df_future_state_monthly_expected['year'] = df_future_state_monthly_expected['time'].dt.year

            # Avg Outage Metric
            outage_start = df_future_outage_proj[df_future_outage_proj['year'] == FUTURE_PROJECTION_START_YEAR].groupby('state_fips')[proj_metric_col].mean()
            outage_end = df_future_outage_proj[df_future_outage_proj['year'] == FUTURE_PROJECTION_END_YEAR].groupby('state_fips')[proj_metric_col].mean()

            # Avg Storm Hits
            hits_start = df_future_state_monthly_expected[df_future_state_monthly_expected['year'] == FUTURE_PROJECTION_START_YEAR].groupby('state_fips')[storm_hits_col].mean()
            hits_end = df_future_state_monthly_expected[df_future_state_monthly_expected['year'] == FUTURE_PROJECTION_END_YEAR].groupby('state_fips')[storm_hits_col].mean()

            # Combine into a single dataframe
            state_changes_df = pd.DataFrame({
                f'{proj_metric_col}_{FUTURE_PROJECTION_START_YEAR}': outage_start,
                f'{proj_metric_col}_{FUTURE_PROJECTION_END_YEAR}': outage_end,
                f'{storm_hits_col}_{FUTURE_PROJECTION_START_YEAR}': hits_start,
                f'{storm_hits_col}_{FUTURE_PROJECTION_END_YEAR}': hits_end
            }).reset_index()

            # Calculate absolute changes
            state_changes_df['outage_change'] = state_changes_df[f'{proj_metric_col}_{FUTURE_PROJECTION_END_YEAR}'] - state_changes_df[f'{proj_metric_col}_{FUTURE_PROJECTION_START_YEAR}']
            state_changes_df['storm_hits_change'] = state_changes_df[f'{storm_hits_col}_{FUTURE_PROJECTION_END_YEAR}'] - state_changes_df[f'{storm_hits_col}_{FUTURE_PROJECTION_START_YEAR}']

            logger.info(f"Calculated state-level changes in projected outages and storm hits.")

            # Analyze relationship between changes
            if len(state_changes_df) > 1 and state_changes_df['storm_hits_change'].notna().all() and state_changes_df['outage_change'].notna().all():
                change_corr = state_changes_df['outage_change'].corr(state_changes_df['storm_hits_change'])
                logger.info(f"Correlation between change in Storm Hits and change in Projected Outages: {change_corr:.4f}")

                # Log OLS coefficient again for context
                if ols_model is not None and 'Pred_Expected_Hits' in ols_model.params:
                    ols_coeff = ols_model.params['Pred_Expected_Hits']
                    logger.info(f"OLS Linking Model Coefficient for Pred_Expected_Hits: {ols_coeff:.4f}")
                    logger.info(f"Interpretation: A 1-unit increase in avg monthly storm hits is associated with a {ols_coeff:.4f} unit change in {TARGET_LINKING_METRIC}.")
                else:
                    logger.warning("Could not retrieve OLS coefficient for context.")
            else:
                logger.warning("Not enough data or NaNs present to calculate correlation between changes.")

            # Save the change driver data
            try:
                os.makedirs(os.path.dirname(STATE_OUTAGE_CHANGE_DRIVERS_PATH), exist_ok=True)
                state_changes_df.to_csv(STATE_OUTAGE_CHANGE_DRIVERS_PATH, index=False)
                logger.info(f"Saved state-level change driver data to {STATE_OUTAGE_CHANGE_DRIVERS_PATH}")
            except Exception as save_e:
                logger.warning(f"Could not save state-level change driver data: {save_e}")

        except Exception as change_e:
            logger.exception(f"Failed during analysis of change drivers: {change_e}")
            state_changes_df = pd.DataFrame() # Ensure empty on error
    else:
        logger.warning("Skipping analysis of change drivers as future projections or storm hits are missing.")


    # --- Step 14: Create State-Level Change Map --- MODIFIED ---
    gdf_states_plot = None
    if not state_changes_df.empty:
        logger.info("Preparing data for state map...")
        try:
            # Load county boundaries and dissolve to states
            if not os.path.exists(COUNTY_SHAPEFILE_PATH):
                raise FileNotFoundError(f"County shapefile not found: {COUNTY_SHAPEFILE_PATH}")
            gdf_counties = gpd.read_file(COUNTY_SHAPEFILE_PATH)
            if STATE_FIPS_COLUMN_IN_SHAPEFILE not in gdf_counties.columns:
                 raise ValueError(f"State FIPS column '{STATE_FIPS_COLUMN_IN_SHAPEFILE}' not found in shapefile.")
            gdf_counties[STATE_FIPS_COLUMN_IN_SHAPEFILE] = gdf_counties[STATE_FIPS_COLUMN_IN_SHAPEFILE].astype(str).str.strip().str.zfill(2)
            gdf_target_counties = gdf_counties[gdf_counties[STATE_FIPS_COLUMN_IN_SHAPEFILE].isin(target_state_fips)].copy()
            if gdf_target_counties.empty:
                raise ValueError(f"No geometries found for target states {target_state_fips} in shapefile.")
            gdf_states = gdf_target_counties.dissolve(by=STATE_FIPS_COLUMN_IN_SHAPEFILE).reset_index()
            gdf_states.rename(columns={STATE_FIPS_COLUMN_IN_SHAPEFILE: 'state_fips'}, inplace=True)
            gdf_states = gdf_states[['state_fips', 'geometry']].copy()

            # Merge change data
            gdf_states_plot = gdf_states.merge(state_changes_df, on='state_fips', how='left')
            missing_map_data = gdf_states_plot['outage_change'].isnull().sum()
            if missing_map_data > 0:
                logger.warning(f"Could not merge change data for {missing_map_data} states for the map.")

        except Exception as map_prep_e:
            logger.exception(f"Failed preparing map data: {map_prep_e}")
            gdf_states_plot = None # Ensure None if prep fails

    if gdf_states_plot is not None and not gdf_states_plot.empty:
        plot_state_outage_change(
            gdf_states_plot=gdf_states_plot,
            output_path=STATE_OUTAGE_CHANGE_MAP_PATH,
            start_yr=FUTURE_PROJECTION_START_YEAR,
            end_yr=FUTURE_PROJECTION_END_YEAR,
            metric_col_name=f'Projected_{TARGET_LINKING_METRIC}', # Base name of the metric
            change_col_name='outage_change' # Name of the column with the change value
        )
    else:
        logger.warning("Skipping state map generation as data preparation failed or resulted in empty GeoDataFrame.")


    logger.info("--- Outage Analysis Script Finished ---")