import gc
# process_climate_data.py
from dask.distributed import Client, LocalCluster
import multiprocessing
from sklearn.neighbors import BallTree
import warnings # To suppress specific warnings if needed
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import regionmask
import logging
from tqdm import tqdm # Import tqdm
import time # For simple timing
from functools import reduce # For merging multiple dataframes

# --- Configuration ---
CLIMATE_DATA_DIR = "data/climate/"
COUNTY_SHAPEFILE_PATH = "data/geospatial/tl_2024_us_county.shp"
GEOID_COLUMN_IN_SHAPEFILE = "GEOID"
STATE_FIPS_COLUMN_IN_SHAPEFILE = "STATEFP" # Column name for State FIPS (e.g., '12' for FL) <-- CHECK THIS
# Define the list of target state FIPS codes
TARGET_STATE_FIPS_LIST = [
    "12", # FL
    "20", # KS
    "29", # MO
    "40", # OK
    "48", # TX
    "05", # AR
    "22", # LA
    "28", # MS
    "01", # AL
    "13", # GA
    "45", # SC
    "37", # NC
    "47", # TN
    "51"  # VA
]
OUTPUT_DIR = "output/processed_climate"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- !! NEW: Define the latest year to include in processing !! ---
MAX_PROCESSING_YEAR = 2035

# --- !! TESTING FLAG: Set to True to run on a sample OF THE TARGET STATE, False for full state run !! ---
PROCESS_SAMPLE_COUNTIES = False # Set back to True for testing, False for full FL run
SAMPLE_FRACTION = 0.05 # Sample 5% OF FLORIDA if PROCESS_SAMPLE_COUNTIES is True
RANDOM_STATE = 42

TARGET_VARIABLES = {
    'tasmax': 'Max Temp',        # CESM2, Amon
    'huss': 'Spec Humid',        # CESM2, Amon
    'hfss': 'Sens Heat Flux',    # CESM2, Amon
    'pr': 'Precip',            # CESM2, Amon
    # 'uas': 'U Wind',           # GFDL - REMOVED
    # 'vas': 'V Wind',           # GFDL - REMOVED
    'psl': 'Sea Level Press',   # CESM2, Amon
    'mrsos': 'Soil Moisture',    # CESM2, Lmon (Monthly Land)
    'orog': 'Altitude',         # CESM2, fx (Fixed/Time-Invariant) <-- ADDED
    'sfcWind': 'Wind Speed'      # CESM2, Amon <-- ADDED
}

MONTHLY_AGG_METHOD = { 'Precip': 'sum' }

# --- Setup Logger ---
LOG_FILE = "logs/climate_processing_states_log.log" # Updated log file name
logger = logging.getLogger("ClimateProcessLoggerStates") # Updated logger name
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # File handler
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"); fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    # Console handler
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO)
    sh_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"); sh.setFormatter(sh_formatter)
    logger.addHandler(sh)

logger.info(f"--- Climate Data Processing Script Started (Target States: {TARGET_STATE_FIPS_LIST}, Max Year: {MAX_PROCESSING_YEAR}) ---")
logger.info(f"PROCESS_SAMPLE_COUNTIES: {PROCESS_SAMPLE_COUNTIES}, SAMPLE_FRACTION: {SAMPLE_FRACTION if PROCESS_SAMPLE_COUNTIES else 'N/A'}")
# ... (logging other configurations) ...
logger.info(f"  MAX_PROCESSING_YEAR: {MAX_PROCESSING_YEAR}")

logger.info("--- Climate Data Processing Script Started ---")
logger.info(f"Script Arguments/Configuration:")
logger.info(f"  CLIMATE_DATA_DIR: {CLIMATE_DATA_DIR}")
logger.info(f"  COUNTY_SHAPEFILE_PATH: {COUNTY_SHAPEFILE_PATH}")
logger.info(f"  GEOID_COLUMN_IN_SHAPEFILE: {GEOID_COLUMN_IN_SHAPEFILE}")
logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"  TARGET_VARIABLES: {TARGET_VARIABLES}")

# --- Step 1: Find NetCDF Files ---
def find_climate_files(base_dir, target_vars):
    """ Recursively finds .nc files, potentially identifying variables """
    nc_files = {}
    logger.info(f"Searching for .nc files under: {base_dir}")
    all_files = glob.glob(os.path.join(base_dir, "**", "*.nc"), recursive=True)
    logger.info(f"Found {len(all_files)} total .nc files.")
    logger.info(f"First few files found: {all_files[:5]}")

    for var_short_name in target_vars.keys():
        found_files_for_var = []
        for f in all_files:
            # Improved check: Look for _{var}_ or /{var}_ patterns
            # This helps avoid matching parts of model names, etc.
            filename = os.path.basename(f)
            # Check path sep too, case insensitive check
            f_lower = f.lower().replace("\\", "/")
            var_lower = var_short_name.lower()
            if f"_{var_lower}_" in filename.lower() or f"/{var_lower}_" in f_lower or \
               filename.lower().startswith(var_lower + "_"):
                 found_files_for_var.append(f)
                 logger.info(f"Found potential file for {var_short_name}: {f}")

        if not found_files_for_var:
            logger.warning(f"Could not find any files matching variable pattern: {var_short_name}")
        else:
            nc_files[var_short_name] = found_files_for_var
            logger.info(f"Found {len(found_files_for_var)} file(s) for variable: {var_short_name}")

    return nc_files

# --- Step 2 & 3: Load and Preprocess NetCDF ---
# --- Step 2 & 3: Load and Preprocess NetCDF (Handle time-invariant) ---
# --- Step 2 & 3: Load and Preprocess NetCDF (Unit Conversions Removed Here) ---
def load_and_preprocess_netcdf(filepaths, variable_short_name):
    """ Loads NetCDF file(s), decodes time, selects variable, basic checks, filters time """
    logger.info(f"Processing variable '{variable_short_name}' from {len(filepaths)} file(s)...")
    start_time = time.time()
    ds = None # Initialize dataset variable
    try:
        # --- Use chunks={} ---
        if len(filepaths) > 1:
             logger.debug(f"Using open_mfdataset for {len(filepaths)} files (disabling auto chunks).")
             try: ds = xr.open_mfdataset(filepaths, combine='by_coords', decode_times=True, chunks={}, parallel=True)
             except Exception as decode_err:
                  logger.warning(f"decode_times=True failed for {variable_short_name}, trying decode_times=False. Error: {decode_err}")
                  ds = xr.open_mfdataset(filepaths, combine='by_coords', decode_times=False, chunks={}, parallel=True)
        elif len(filepaths) == 1:
             logger.debug(f"Using open_dataset for single file: {filepaths[0]} (disabling auto chunks).")
             try: ds = xr.open_dataset(filepaths[0], decode_times=True, chunks={})
             except Exception as decode_err:
                 logger.warning(f"decode_times=True failed for {variable_short_name}, trying decode_times=False. Error: {decode_err}")
                 ds = xr.open_dataset(filepaths[0], decode_times=False, chunks={})
        else: logger.warning(f"No file paths provided for {variable_short_name}."); return None
        if ds is None: return None # Ensure ds was loaded

        logger.debug(f"Opened dataset for {variable_short_name}. Variables: {list(ds.data_vars)}. Coords: {list(ds.coords)}")

        # Find variable (same logic as before)
        data_var_name = None; var_lower = variable_short_name.lower()
        potential_matches_exact = [v for v in ds.data_vars if var_lower == v.lower()]
        if len(potential_matches_exact) == 1: data_var_name = potential_matches_exact[0]; #... (warning if case mismatch) ...
        elif len(potential_matches_exact) > 1: logger.error(f"Multi exact match {variable_short_name}"); ds.close(); return None
        else: potential_matches_substr = [v for v in ds.data_vars if var_lower in v.lower()]; #... (substring check) ...
        if data_var_name is None: logger.error(f"Var '{variable_short_name}' not found."); ds.close(); return None
        da = ds[data_var_name]

        logger.debug(f"Selected DataArray '{data_var_name}'. Initial shape: {da.shape}, Dims: {da.dims}")
        if da.dtype == 'O': logger.warning(f"{data_var_name} object dtype BEFORE.")

        # Coord Renaming & Cleaning (same as before)
        coord_renames = {}; #... (build coord_renames dict) ...
        if ('latitude' in da.coords and 'lat' not in da.coords): coord_renames['latitude'] = 'lat'
        if ('longitude' in da.coords and 'lon' not in da.coords): coord_renames['longitude'] = 'lon'
        if ('nav_lat' in da.coords and 'lat' not in da.coords): coord_renames['nav_lat'] = 'lat'
        if ('nav_lon' in da.coords and 'lon' not in da.coords): coord_renames['nav_lon'] = 'lon'
        if coord_renames: logger.info(f"Renaming coords: {coord_renames}"); da = da.rename(coord_renames)
        if 'lat' not in da.coords or 'lon' not in da.coords: logger.error("Missing lat/lon"); ds.close(); return None
        da = da.sortby('lat');
        if da['lon'].max() > 180: logger.info("Converting lon to -180-180"); da = da.assign_coords(lon=(((da.coords['lon'] + 180) % 360) - 180)); da = da.sortby('lon')

        # Time Coordinate Handling & Filtering (same as before)
        is_time_varying = 'time' in da.coords
        if is_time_varying:
            if not np.issubdtype(da['time'].dtype, np.datetime64):
                logger.warning("Non-standard time coord. Converting cftime.");
                try:
                     with warnings.catch_warnings(): warnings.simplefilter("ignore"); datetimeindex = da.indexes['time'].to_datetimeindex()
                     da['time'] = datetimeindex; logger.info("Converted cftime ok.")
                except Exception as cftime_err: logger.error(f"Failed cftime: {cftime_err}."); ds.close(); return None
            logger.debug(f"Time range BEFORE filter: {da['time'].min().values} to {da['time'].max().values}")

            if MAX_PROCESSING_YEAR is not None:
                logger.info(f"Filtering up to {MAX_PROCESSING_YEAR}...")
                if np.issubdtype(da['time'].dtype, np.datetime64):
                    original_time_size = da.time.size
                    da = da.where(da.time.dt.year <= MAX_PROCESSING_YEAR, drop=True)
                    logger.info(f"Filtered time: {original_time_size} -> {da.time.size} steps.")
                    if da.time.size == 0: logger.warning("No data after time filter."); ds.close(); return None
                else: logger.warning("Cannot filter time: not datetime.")
        else: logger.info(f"'{variable_short_name}' is time-invariant.")

        # Standard Name (same as before)
        standard_name = TARGET_VARIABLES.get(variable_short_name, variable_short_name); da.attrs['standard_name'] = standard_name; da = da.rename(standard_name)

        # --- Unit Conversion Section - Conversions REMOVED/Commented Out ---
        if 'units' in da.attrs:
            original_units = da.attrs['units']
            logger.debug(f"Variable: {standard_name}, Original Units: {original_units} (Conversion will happen later)")
            # COMMENTED OUT:
            # converted = False
            # if original_units.lower() in ['k', 'kelvin'] and standard_name == 'Max Temp':
            #     # da = da - 273.15; da.attrs['units'] = 'C'; converted = True
            #     pass
            # elif original_units.lower() in ['kg m-2 s-1'] and standard_name == 'Precip':
            #      # da = da * 86400; da.attrs['units'] = 'mm/day'; converted = True
            #      pass
            # elif original_units.lower() in ['pa'] and standard_name == 'Sea Level Press':
            #      # da = da / 100.0; da.attrs['units'] = 'hPa'; converted = True
            #      pass
            # if converted: logger.info(f"Converted units for {standard_name} to {da.attrs['units']}")
        else:
             logger.warning(f"Units attribute missing for {standard_name}. Assuming standard units required later.")
        # --- End Unit Conversion Section Modification ---

        # Dimension handling (same as before)
        dims_to_keep = ['time', 'lat', 'lon'] if is_time_varying else ['lat', 'lon']; # ... (squeeze/select logic) ...
        extra_dims = [d for d in da.dims if d not in dims_to_keep];
        if extra_dims: # ... (selection/squeeze logic same as before) ...
             selected = False;
             for dim in extra_dims:
                 if dim in ['height', 'lev', 'plev', 'depth'] and da[dim].size > 1: da = da.isel({dim: 0}, drop=True); logger.warning(f"Selected first index for dim '{dim}'."); selected = True; break
                 elif da[dim].size == 1: continue
                 else: logger.warning(f"Dim '{dim}' size {da[dim].size} > 1, no selection.");
             dims_to_squeeze = [d for d in da.dims if d not in dims_to_keep and da[d].size == 1];
             if dims_to_squeeze: logger.warning(f"Squeezing dims: {dims_to_squeeze}"); da = da.squeeze(dims_to_squeeze, drop=True);
             final_extra_dims = [d for d in da.dims if d not in dims_to_keep];
             if final_extra_dims: logger.error(f"Extra dims remain: {final_extra_dims}"); ds.close(); return None
        if set(da.dims) != set(dims_to_keep): logger.error(f"Unexpected final dims: {da.dims}"); ds.close(); return None

        # Dtype/FillValue checks (same as before)
        if da.dtype == 'O': logger.error(f"{standard_name} object dtype."); ds.close(); return None
        fill_value = da.encoding.get('_FillValue', da.attrs.get('_FillValue', None)); # ... (handle fill value) ...
        if fill_value is not None: logger.debug(f"Replacing fill {fill_value}"); da = da.where(da != fill_value)

        logger.info(f"Successfully preprocessed '{standard_name}'. Shape: {da.shape}, Dtype: {da.dtype}. Time: {time.time() - start_time:.2f}s")
        ds.close(); return da

    except Exception as e:
        logger.exception(f"Failed load/preprocess {variable_short_name}: {e}")
        if ds is not None: ds.close();
        return None

# --- Step 5 & 6: Aggregate to Counties (with Nearest Neighbor Fallback) ---
# --- Step 5 & 6: Aggregate to Counties (Using Pre-computed Mask, NO Fallback, More Logging) ---
# --- Step 5 & 6: Aggregate to Counties (Removed internal dropna, Corrected mask check) ---
# --- Step 5 & 6: Aggregate to Counties (with Nearest Neighbor Fallback) ---
# --- Step 5 & 6: Aggregate to Counties (Logging after compute) ---
# --- Step 5 & 6: Aggregate to Counties (FINAL - No Value Dropna, with Fallback Logic Included but likely unused) ---
# --- Step 5 & 6: Aggregate to Counties (Using Pre-computed Mask, NO Fallback) ---
# --- Step 5 & 6: Aggregate to Counties (Nearest Neighbor Method) ---
# --- Step 5 & 6: Aggregate to Counties (Handle Time-Invariant Vars) ---
def aggregate_to_counties(da_climate, gdf_counties, standard_name, grid_tree, grid_shape):
    """
    Aggregates gridded climate data (time-varying or fixed) to county polygons
    using the NEAREST NEIGHBOR method.
    """
    if da_climate is None: logger.error(f"{standard_name}: Input data None."); return None
    if gdf_counties is None or gdf_counties.empty: logger.error(f"{standard_name}: County GDF empty."); return None
    if grid_tree is None: logger.error(f"{standard_name}: Grid BallTree missing."); return None
    if grid_shape is None: logger.error(f"{standard_name}: Grid shape missing."); return None

    logger.info(f"Aggregating '{standard_name}' to {len(gdf_counties)} counties using Nearest Neighbor...")
    agg_start_time = time.time()
    is_time_varying = 'time' in da_climate.coords

    results_list = []
    df_agg = pd.DataFrame() # Initialize result DataFrame

    try:
        # --- Prepare County Centroids (same as before) ---
        logger.debug(f"  {standard_name} - Calculating county centroids...")
        prep_start = time.time()
        if gdf_counties.crs is None: gdf_counties_geo = gdf_counties.set_crs("EPSG:4326")
        elif gdf_counties.crs != "EPSG:4326": gdf_counties_geo = gdf_counties.to_crs("EPSG:4326")
        else: gdf_counties_geo = gdf_counties
        with warnings.catch_warnings(): # Suppress centroid warning
            warnings.simplefilter("ignore", category=UserWarning)
            centroids_geo = gdf_counties_geo.geometry.centroid
        county_coords = centroids_geo.get_coordinates()
        county_centroids_latlon = county_coords.loc[gdf_counties_geo.index, ['y', 'x']].values
        county_centroids_rad = np.radians(county_centroids_latlon)
        county_geoids_ordered = gdf_counties_geo['county_geoid'].tolist()
        logger.debug(f"  {standard_name} - Centroid prep took: {time.time() - prep_start:.2f}s")

        # --- Find Nearest Grid Cell Indices (same as before) ---
        logger.debug(f"  {standard_name} - Querying grid BallTree...")
        query_start = time.time()
        distances, flat_indices = grid_tree.query(county_centroids_rad, k=1, return_distance=True)
        flat_indices = flat_indices[:,0]
        nearest_grid_indices = np.unravel_index(flat_indices, grid_shape)
        logger.debug(f"  {standard_name} - BallTree query took: {time.time() - query_start:.2f}s")
        lat_indexer = xr.DataArray(nearest_grid_indices[0], dims='county')
        lon_indexer = xr.DataArray(nearest_grid_indices[1], dims='county')

        # --- Extract Data based on Time Dimension ---
        extract_start = time.time()
        if is_time_varying:
            # --- Process TIME-VARYING data (existing logic) ---
            logger.info(f"  {standard_name} - Extracting & resampling time series...")
            county_timeseries = da_climate.isel(lat=lat_indexer, lon=lon_indexer)
            county_timeseries = county_timeseries.rename({'county': 'county_geoid'})
            county_timeseries['county_geoid'] = county_geoids_ordered
            logger.debug(f"  {standard_name} - Advanced indexing complete. Shape: {county_timeseries.shape}")

            # Resample monthly
            agg_method = MONTHLY_AGG_METHOD.get(standard_name, 'mean')
            logger.info(f"  {standard_name} - Using monthly aggregation: '{agg_method}'")
            resample_start = time.time();
            monthly_agg = county_timeseries.resample(time='MS')
            temporal_agg_func = monthly_agg.sum if agg_method == 'sum' else monthly_agg.mean
            monthly_county_data = temporal_agg_func(skipna=True)
            logger.debug(f"  {standard_name} - Resample setup took: {time.time() - resample_start:.2f}s")

            # Compute
            compute_start_time = time.time(); logger.info(f"  {standard_name} - Computing final aggregated data...")
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); monthly_county_data = monthly_county_data.compute()
            logger.info(f"  {standard_name} - Computation complete. Took: {time.time() - compute_start_time:.2f}s")
            logger.debug(f"  {standard_name} - Shape after compute: {monthly_county_data.shape}")
            # No need to map numbers back, county_geoid is already the dimension

            # Convert to DataFrame
            df_agg = monthly_county_data.to_dataframe(name=standard_name).reset_index()
            # Ensure time column is date
            if 'time' in df_agg.columns: df_agg['time'] = pd.to_datetime(df_agg['time']).dt.date

        else:
            # --- Process TIME-INVARIANT data (e.g., orog) ---
            logger.info(f"  {standard_name} - Extracting fixed values...")
            # Use isel to get the value at the nearest grid point for each county
            fixed_values = da_climate.isel(lat=lat_indexer, lon=lon_indexer)
            # Assign county_geoid coordinate
            fixed_values = fixed_values.rename({'county': 'county_geoid'})
            fixed_values['county_geoid'] = county_geoids_ordered
            fixed_values = fixed_values.compute() # Compute if it's a dask array
            logger.debug(f"  {standard_name} - Extraction complete. Shape: {fixed_values.shape}")

            # Convert directly to DataFrame (no time dimension)
            df_agg = fixed_values.to_dataframe(name=standard_name).reset_index()
            # Add a placeholder 'time' column if needed for merging consistency? No, merge separately later.

        logger.debug(f"  {standard_name} - Extraction/Resample took: {time.time() - extract_start:.2f}s")

    except Exception as e:
         logger.exception(f"Failed during nearest neighbor aggregation/resampling for {standard_name}: {e}")
         return None

    # --- Convert to DataFrame (if not already done for fixed vars) & Final Checks ---
    df_conv_start_time = time.time(); logger.debug("Converting final result to DataFrame (if needed)...")
    try:
        # If df_agg wasn't created above (e.g., error), initialize empty
        if 'df_agg' not in locals() or df_agg is None: df_agg = pd.DataFrame()

        # Check if DataFrame is valid and has expected columns
        if not isinstance(df_agg, pd.DataFrame) or df_agg.empty:
             logger.warning(f"DataFrame generation failed or is empty for {standard_name}.")
             return None

        if 'county_geoid' not in df_agg.columns: raise ValueError("'county_geoid' column missing.")
        if standard_name not in df_agg.columns: raise ValueError(f"'{standard_name}' column missing.")

        # Ensure time column exists and is correct format IF it's time-varying data
        if is_time_varying:
             if 'time' not in df_agg.columns: raise ValueError("'time' column missing for time-varying data.")
             df_agg['time'] = pd.to_datetime(df_agg['time']).dt.date
             # Drop rows only if essential keys are missing
             key_cols = ['county_geoid', 'time']
             final_cols = ['county_geoid', 'time', standard_name]
        else: # Time-invariant data
             key_cols = ['county_geoid']
             final_cols = ['county_geoid', standard_name]

        logger.debug(f"  {standard_name} - Shape before final checks: {df_agg.shape}.")
        rows_before_key_dropna = len(df_agg)
        df_agg.dropna(subset=key_cols, inplace=True)
        rows_after_key_dropna = len(df_agg)
        if rows_before_key_dropna != rows_after_key_dropna: logger.warning(f"Dropped rows with missing keys.")

        final_nan_count = df_agg[standard_name].isnull().sum()
        final_row_count = len(df_agg)
        logger.info(f"Aggregation for '{standard_name}' complete. Output shape ({final_row_count}, {len(df_agg.columns)}) ({final_nan_count} NaNs). Total func time: {time.time() - agg_start_time:.2f}s")
        if df_agg.empty: logger.warning(f"Aggregation for {standard_name} resulted EMPTY DataFrame.")
        return df_agg[final_cols] # Return only essential columns

    except Exception as final_e:
         logger.exception(f"Error during final processing/DataFrame conversion for {standard_name}: {final_e}")
         return None

# --- Main Script Logic ---
if __name__ == "__main__":
    logger.info("--- Starting Main Processing ---")
    main_start_time = time.time()
    client = None; cluster = None # Initialize

    # --- Dask Cluster Setup ---
    try: # ... (Dask setup) ...
        n_cores = multiprocessing.cpu_count(); workers = max(1, n_cores - 2); threads = 1
        logger.info(f"Attempting Dask Cluster: {workers}w, {threads}t."); cluster = LocalCluster(n_workers=workers, threads_per_worker=threads, memory_limit='auto')
        client = Client(cluster); logger.info(f"Dask dashboard: {client.dashboard_link}")
    except Exception as dask_e: logger.warning(f"Dask LocalCluster failed: {dask_e}.")

    # --- Main Processing ---
    try:
        # 1. Find files
        nc_files_dict = find_climate_files(CLIMATE_DATA_DIR, TARGET_VARIABLES)
        nc_files_dict = {k:v for k,v in nc_files_dict.items() if k in TARGET_VARIABLES}
        logger.info(f"Processing variables: {list(nc_files_dict.keys())}")

        # 2. Load County Polygons & Filter/Sample
        try: # ... (County loading, filtering, sampling, number_to_geoid_final mapping) ...
            gdf_counties_raw = gpd.read_file(COUNTY_SHAPEFILE_PATH)
            # ... (Checks for GEOID, STATEFP) ...
            if GEOID_COLUMN_IN_SHAPEFILE not in gdf_counties_raw.columns: raise ValueError("GEOID col missing.")
            if STATE_FIPS_COLUMN_IN_SHAPEFILE not in gdf_counties_raw.columns: raise ValueError("State FIPS col missing.")
            gdf_counties_raw = gdf_counties_raw[gdf_counties_raw.geometry.notna() & gdf_counties_raw.geometry.is_valid & ~gdf_counties_raw.geometry.is_empty]
            logger.info(f"Filtering counties for State FIPS in: {TARGET_STATE_FIPS_LIST}")
            gdf_counties_raw[STATE_FIPS_COLUMN_IN_SHAPEFILE] = gdf_counties_raw[STATE_FIPS_COLUMN_IN_SHAPEFILE].astype(str).str.strip()
            # Filter using the list of target states
            gdf_target_state_counties = gdf_counties_raw[gdf_counties_raw[STATE_FIPS_COLUMN_IN_SHAPEFILE].isin(TARGET_STATE_FIPS_LIST)].copy()
            if gdf_target_state_counties.empty: raise ValueError(f"No counties found for Target State FIPS list: {TARGET_STATE_FIPS_LIST}.")
            # ... (Process gdf_counties_processed: rename, zfill, dropna, drop_duplicates, set CRS) ...
            gdf_counties_processed = gdf_target_state_counties[[GEOID_COLUMN_IN_SHAPEFILE, 'geometry']].copy()
            gdf_counties_processed.rename(columns={GEOID_COLUMN_IN_SHAPEFILE: 'county_geoid'}, inplace=True)
            gdf_counties_processed['county_geoid'] = gdf_counties_processed['county_geoid'].astype(str).str.strip().str.zfill(5)
            gdf_counties_processed.drop_duplicates(subset=['county_geoid'], keep='first', inplace=True)
            gdf_counties_processed = gdf_counties_processed.dropna(subset=['county_geoid', 'geometry'])
            if gdf_counties_processed.crs is None: gdf_counties_processed = gdf_counties_processed.set_crs("EPSG:4326")
            elif gdf_counties_processed.crs != "EPSG:4326": gdf_counties_processed = gdf_counties_processed.to_crs("EPSG:4326")
            logger.info(f"Loaded {len(gdf_counties_processed)} valid counties for the target states.")
            gdf_counties_processed = gdf_counties_processed.reset_index(drop=True)
            gdf_counties_processed['county_num_id'] = gdf_counties_processed.index
            number_to_geoid_full = pd.Series(gdf_counties_processed['county_geoid'].values, index=gdf_counties_processed['county_num_id'].values)
            # Sampling Logic
            if PROCESS_SAMPLE_COUNTIES:
                 n_total = len(gdf_counties_processed); n_sample = max(1, int(n_total * SAMPLE_FRACTION))
                 logger.warning(f"SAMPLE RUN: Selecting {n_sample} counties."); gdf_to_process = gdf_counties_processed.sample(n=n_sample, random_state=RANDOM_STATE).copy()
                 gdf_to_process = gdf_to_process.reset_index(drop=True); gdf_to_process['county_num_id'] = gdf_to_process.index
                 number_to_geoid_final = pd.Series(gdf_to_process['county_geoid'].values, index=gdf_to_process['county_num_id'].values)
            else: gdf_to_process = gdf_counties_processed; number_to_geoid_final = number_to_geoid_full
            logger.info(f"Processing climate data for {len(gdf_to_process)} counties.")

        except Exception as e: logger.exception(f"CRITICAL: Failed county loading: {e}"); raise e


        # 3. Process each variable and aggregate
        all_county_monthly_dfs = []
        fixed_dfs_dict = {}
        variable_items = list(nc_files_dict.items())
        grid_tree = None
        grid_shape = None

        for var_short_name, filepaths in tqdm(variable_items, desc="Processing Variables"):
            # ... (Get variable_standard_name) ...
            variable_standard_name = TARGET_VARIABLES.get(var_short_name, var_short_name)
            logger.info(f"--- Processing Variable: {variable_standard_name} ---")
            da_climate_var = load_and_preprocess_netcdf(filepaths, var_short_name)

            if da_climate_var is None: logger.error(f"Skipping aggregation: load failed."); continue

            # Build Grid Tree ONCE
            if grid_tree is None and 'lat' in da_climate_var.coords and 'lon' in da_climate_var.coords:
                 try: # ... (Build grid tree logic) ...
                    logger.info("Building BallTree..."); grid_lat = da_climate_var['lat'].values; grid_lon = da_climate_var['lon'].values
                    grid_lat_flat, grid_lon_flat = np.meshgrid(grid_lat, grid_lon, indexing='ij'); grid_points_rad = np.radians(np.vstack([grid_lat_flat.ravel(), grid_lon_flat.ravel()]).T)
                    grid_tree = BallTree(grid_points_rad, metric='haversine'); grid_shape = (da_climate_var['lat'].size, da_climate_var['lon'].size)
                    logger.info(f"BallTree built. Grid shape: {grid_shape}")
                 except Exception as tree_build_e: logger.exception(f"Failed BallTree build: {tree_build_e}"); grid_tree = None


            # Call aggregation (Nearest Neighbor version)
            df_county_result = aggregate_to_counties(
                da_climate_var,
                gdf_to_process,
                variable_standard_name,
                grid_tree,
                grid_shape
            )

            # Separate time-varying and fixed results
            if df_county_result is not None and not df_county_result.empty:
                 logger.info(f"Generated {len(df_county_result)} rows for {variable_standard_name}.")
                 if 'time' in df_county_result.columns:
                      all_county_monthly_dfs.append(df_county_result)
                 else: # Assumed fixed if no time column
                      fixed_dfs_dict[variable_standard_name] = df_county_result[['county_geoid', variable_standard_name]].drop_duplicates(subset=['county_geoid'])
                      logger.info(f"Stored fixed variable: {variable_standard_name}")
            else: logger.warning(f"No data generated for: {variable_standard_name}")

            da_climate_var = None; gc.collect(); logger.debug(f"Memory collected.")


        # 4. Combine results from all variables
        if not all_county_monthly_dfs and not fixed_dfs_dict:
            logger.error("No county-level climate data was generated.")
        else:
            df_final_county_monthly = None
            # --- Merge TIME-VARYING variables ---
            if not all_county_monthly_dfs: logger.warning("No time-varying data.")
            elif len(all_county_monthly_dfs) == 1: df_final_county_monthly = all_county_monthly_dfs[0]
            else:
                logger.info(f"Combining {len(all_county_monthly_dfs)} time-varying variables...")
                df_final_county_monthly = reduce(lambda left, right: pd.merge(left, right, on=['county_geoid', 'time'], how='outer'), all_county_monthly_dfs)

            # --- Merge FIXED variables ---
            if fixed_dfs_dict:
                logger.info(f"Merging {len(fixed_dfs_dict)} fixed variables...")
                for var_name, df_fixed in fixed_dfs_dict.items():
                    if df_final_county_monthly is None or df_final_county_monthly.empty:
                         # If ONLY fixed vars were processed, need a base DF - less ideal
                         logger.warning(f"Only fixed var {var_name} processed. Final DF will lack time.")
                         df_final_county_monthly = df_fixed # Assign first fixed df if base is empty
                    elif 'county_geoid' in df_final_county_monthly.columns:
                        logger.debug(f"Merging fixed variable: {var_name}")
                        df_final_county_monthly = pd.merge(df_final_county_monthly, df_fixed, on='county_geoid', how='left')
                    else:
                        logger.error(f"Cannot merge fixed var {var_name}, base df missing county_geoid.")

            # --- Final Check and Unit Conversion ---
            if df_final_county_monthly is not None and 'county_geoid' in df_final_county_monthly.columns and not df_final_county_monthly.empty:
                logger.info(f"Final combined DataFrame shape BEFORE unit conversion: {df_final_county_monthly.shape}")

                # --- !! ADDED: Perform Unit Conversions on Final DataFrame !! ---
                logger.info("Performing final unit conversions on merged DataFrame...")
                converted_cols = []
                if 'Max Temp' in df_final_county_monthly.columns:
                    logger.debug("Converting Max Temp K -> C")
                    df_final_county_monthly['Max Temp'] = df_final_county_monthly['Max Temp'] - 273.15
                    converted_cols.append('Max Temp')
                if 'Precip' in df_final_county_monthly.columns:
                    logger.debug("Converting Precip kg m-2 s-1 -> mm/day")
                    df_final_county_monthly['Precip'] = df_final_county_monthly['Precip'] * 86400
                    converted_cols.append('Precip')
                if 'Sea Level Press' in df_final_county_monthly.columns:
                    logger.debug("Converting Sea Level Press Pa -> hPa")
                    df_final_county_monthly['Sea Level Press'] = df_final_county_monthly['Sea Level Press'] / 100.0
                    converted_cols.append('Sea Level Press')
                # Add more if needed
                logger.info(f"Final unit conversions applied for: {converted_cols}")
                # --- End Added Conversions ---

                logger.info(f"Final columns: {df_final_county_monthly.columns.tolist()}")
                print("\n--- Final Combined County-Monthly Data (Sample with CONVERTED Units) ---")
                print(df_final_county_monthly.head().to_string()) # Use print()

                # 5. Save output
                # Use a more general suffix for the output file
                output_filename_suffix = "_target_states_SAMPLE" if PROCESS_SAMPLE_COUNTIES else "_target_states"
                output_filename_parquet = os.path.join(OUTPUT_DIR, f"county_monthly_climate_variables{output_filename_suffix}.parquet")
                output_filename_csv = os.path.join(OUTPUT_DIR, f"county_monthly_climate_variables{output_filename_suffix}.csv")
                try: df_final_county_monthly.to_parquet(output_filename_parquet, index=False); logger.info(f"Saved parquet: {output_filename_parquet}")
                except ImportError: logger.error("pyarrow missing."); logger.info(f"Saving CSV: {output_filename_csv}"); #... (CSV save logic) ...
                except Exception as save_e: logger.exception(f"Failed parquet save: {save_e}"); logger.info(f"Saving CSV: {output_filename_csv}"); #... (CSV save logic) ...
            else: logger.error("Final dataframe empty or failed generation.")

    # --- Use finally for Dask cleanup ---
    finally:
        if client is not None:
            logger.info("Closing Dask client and cluster...")
            # ... (Dask cleanup logic) ...
            try: client.close()
            except Exception: pass
            try:
                 if cluster is not None: cluster.close()
                 logger.info("Dask resources closed.")
            except Exception: pass

    logger.info(f"--- Climate Data Processing Script Finished --- Total Time: {(time.time() - main_start_time)/60:.2f} minutes ---")