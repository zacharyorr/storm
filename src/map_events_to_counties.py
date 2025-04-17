# map_events_to_counties.py

import os
import pandas as pd
import numpy as np
import logging
import math
from tqdm import tqdm
from geopy.distance import geodesic # Still used for final check if needed, but BallTree handles initial query
from sklearn.neighbors import BallTree # <-- The key optimization library

# CONSTANTS
HURRICANE_RADIUS_MILES = 50.0
FEET_PER_MILE = 5280.0
SQFT_PER_ACRE = 43560.0
EARTH_RADIUS_MILES = 3958.8 # Approx. radius for Haversine in BallTree

def setup_logger():
    """
    Configure a logger that prints to console and can be adjusted as needed.
    """
    logger = logging.getLogger("MapEventsLogger")
    logger.setLevel(logging.INFO) # or DEBUG, WARNING, etc.

    # Avoid adding multiple handlers if already set
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

logger = setup_logger()

def load_counties_txt(counties_file: str) -> pd.DataFrame:
    """
    Loads the '2024_counties.txt' file, using the correct delimiter and stripping headers.
    """
    logger.info(f"Loading county data from: {counties_file}")
    try:
        # Use the delimiter you confirmed works ('\t' in your case)
        df = pd.read_csv(counties_file, delimiter='\t', dtype=str)
    except Exception as e:
        logger.error(f"Failed to read county file with tab delimiter: {e}")
        logger.info("Attempting with whitespace delimiter as fallback...")
        try:
            df = pd.read_csv(counties_file, delim_whitespace=True, dtype=str)
        except Exception as e2:
            logger.error(f"Fallback with whitespace delimiter also failed: {e2}")
            raise ValueError("Could not parse county file. Check delimiter and format.") from e2

    # Strip whitespace around column names
    df.columns = df.columns.str.strip()
    logger.debug(f"County file columns after stripping: {df.columns.tolist()}")

    # Convert lat/lon to float, ensuring the columns exist
    required_cols = ['GEOID', 'INTPTLAT', 'INTPTLONG', 'NAME', 'USPS']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"File missing required column '{col}'. Check file headers.")

    df['INTPTLAT'] = df['INTPTLAT'].astype(float)
    df['INTPTLONG'] = df['INTPTLONG'].astype(float)

    # We'll rename columns for clarity
    df.rename(columns={
        'GEOID': 'county_geoid',
        'INTPTLAT': 'lat',
        'INTPTLONG': 'lon'
    }, inplace=True)

    keep_cols = ['county_geoid', 'lat', 'lon', 'NAME', 'USPS']
    df = df[keep_cols].copy()
    # Ensure county_geoid is string and potentially padded if needed
    df['county_geoid'] = df['county_geoid'].astype(str).str.strip()
    # df['county_geoid'] = df['county_geoid'].str.zfill(5) # Uncomment if padding is necessary

    # Add coordinates in radians for BallTree
    df['lat_rad'] = np.radians(df['lat'])
    df['lon_rad'] = np.radians(df['lon'])

    logger.info(f"Loaded {len(df)} counties from {counties_file}")
    return df

def build_county_kdtree(df_counties: pd.DataFrame):
    """
    Builds a BallTree from county centroid coordinates (in radians) for fast spatial querying.
    """
    logger.info("Building BallTree for county centroids...")
    # Use radian coordinates for Haversine distance metric in BallTree
    county_coords_rad = df_counties[['lat_rad', 'lon_rad']].values
    tree = BallTree(county_coords_rad, metric='haversine')
    logger.info("BallTree built successfully.")
    return tree

def map_hurricanes_to_counties(df_hurdat: pd.DataFrame, df_counties: pd.DataFrame, county_tree: BallTree) -> pd.DataFrame:
    """
    Optimized function using BallTree to map hurricanes to counties within 50 miles.
    """
    logger.info("Mapping Hurricanes to Counties using BallTree (50mi radius)...")
    df_hurdat['datetime'] = pd.to_datetime(df_hurdat['datetime'], errors='coerce').dt.tz_localize(None) # Ensure no timezone for comparison if needed
    df_hurdat = df_hurdat.sort_values(['storm_id', 'datetime'])
    df_hurdat.dropna(subset=['latitude', 'longitude', 'datetime'], inplace=True) # Ensure valid inputs

    results = []
    assigned = set()  # track (storm_id, county_geoid)

    unique_storms = df_hurdat['storm_id'].unique()
    logger.info(f"Found {len(unique_storms)} unique storms in HURDAT.")

    # Convert radius from miles to radians for BallTree query
    radius_radians = HURRICANE_RADIUS_MILES / EARTH_RADIUS_MILES

    # Outer loop: storms
    for storm in tqdm(unique_storms, desc="Hurricanes", unit="storm"):
        sub = df_hurdat[df_hurdat['storm_id'] == storm]
        # Inner loop: each track point
        for _, row in sub.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            current_time = row['datetime']

            # Convert point to radians for query
            try:
                point_rad = np.radians([lat, lon])
            except TypeError:
                logger.warning(f"Skipping invalid lat/lon for storm {storm}: {lat}, {lon}")
                continue

            # Query the tree for county indices within the radius
            indices = county_tree.query_radius(point_rad.reshape(1, -1), r=radius_radians)[0]

            # Process only the nearby counties found by the tree
            for county_idx in indices:
                geoid = df_counties.iloc[county_idx]['county_geoid']

                if (storm, geoid) in assigned:
                    continue # Already assigned for this storm

                # Assuming BallTree result is good enough:
                assigned.add((storm, geoid))
                results.append({
                    'event_id': storm,
                    'county_geoid': geoid,
                    'first_impact_dt': current_time,
                    'event_type': 'hurricane'
                })

    logger.info(f"Hurricane->County mapping complete. Generated {len(results)} records.")
    return pd.DataFrame(results)


def map_wildfires_to_counties(df_wildfires: pd.DataFrame, df_counties: pd.DataFrame, county_tree: BallTree) -> pd.DataFrame:
    """
    Optimized function using BallTree to map wildfires to counties based on acreage radius.
    """
    logger.info("Mapping Wildfires to Counties using BallTree (acreage circle assumption)...")
    df_wildfires['incident_date_created'] = pd.to_datetime(df_wildfires['incident_date_created'], errors='coerce').dt.tz_localize(None)
    df_wildfires.dropna(subset=['incident_latitude', 'incident_longitude', 'incident_date_created'], inplace=True)

    results = []

    # We'll iterate over each wildfire with tqdm
    for _, fire_row in tqdm(df_wildfires.iterrows(), total=len(df_wildfires), desc="Wildfires", unit="fire"):
        event_id = fire_row['incident_id']
        lat = fire_row['incident_latitude']
        lon = fire_row['incident_longitude']
        acres = fire_row.get('incident_acres_burned', 0) or 0

        # Calculate radius in miles
        area_sqft = acres * SQFT_PER_ACRE
        if area_sqft > 0:
            radius_feet = math.sqrt(area_sqft / math.pi)
            radius_miles = radius_feet / FEET_PER_MILE
        else:
            radius_miles = 0

        start_time = fire_row['incident_date_created']

        # If radius is positive, query the tree
        if radius_miles > 0:
            # Convert radius to radians and point to radians
            radius_radians = radius_miles / EARTH_RADIUS_MILES
            try:
                point_rad = np.radians([lat, lon])
            except TypeError:
                 logger.warning(f"Skipping invalid lat/lon for wildfire {event_id}: {lat}, {lon}")
                 continue

            # Query the tree
            indices = county_tree.query_radius(point_rad.reshape(1, -1), r=radius_radians)[0]

            # Add results for nearby counties
            for county_idx in indices:
                results.append({
                    'event_id': event_id,
                    'county_geoid': df_counties.iloc[county_idx]['county_geoid'],
                    'first_impact_dt': start_time,
                    'event_type': 'wildfire'
                })

    logger.info(f"Wildfire->County mapping complete. Generated {len(results)} records.")
    # Ensure uniqueness at the wildfire-county level
    if results:
        return pd.DataFrame(results).drop_duplicates(subset=['event_id', 'county_geoid'])
    else:
        return pd.DataFrame(columns=['event_id', 'county_geoid', 'first_impact_dt', 'event_type'])


def map_scs_wind_to_counties(df_stormevents: pd.DataFrame, df_counties: pd.DataFrame) -> pd.DataFrame:
    """
    For SCS wind events, use the official StormEvents approach, BUT FILTER FOR COUNTY ZONES ('C').
     - Filter for wind-related event types
     - *** Filter for CZ_TYPE == 'C' ***
     - Combine STATE_FIPS + CZ_FIPS => a 'fips5' code
     - Merge with counties by matching first 5 digits of county_geoid
    """
    logger.info("Mapping SCS Wind events to Counties using StormEvents FIPS data (CZ_TYPE='C' only)...")
    wind_types = ['Thunderstorm Wind', 'High Wind', 'Strong Wind', 'Marine High Wind'] # Add others if needed

    # --- Check if CZ_TYPE column exists ---
    if 'CZ_TYPE' not in df_stormevents.columns:
        logger.error("Required column 'CZ_TYPE' not found in StormEvents data. Cannot filter by county zone.")
        logger.warning("Proceeding without CZ_TYPE filter - results may include non-county zones and cause mismatches.")
        df_wind_filtered = df_stormevents[df_stormevents['EVENT_TYPE'].isin(wind_types)].copy()
    else:
        # --- Apply Filters ---
        df_wind_filtered = df_stormevents[
            (df_stormevents['EVENT_TYPE'].isin(wind_types)) &
            (df_stormevents['CZ_TYPE'] == 'C') # Filter for County types only
        ].copy()
        logger.info(f"Filtered {len(df_wind_filtered)} SCS wind rows with CZ_TYPE='C' from StormEvents.")

    if df_wind_filtered.empty:
        logger.warning("No SCS wind events found with CZ_TYPE='C'. Returning empty DataFrame.")
        return pd.DataFrame(columns=['event_id', 'county_geoid', 'first_impact_dt', 'event_type'])

    # Handle potential NaN or non-numeric FIPS before formatting
    required_fips_cols = ['STATE_FIPS', 'CZ_FIPS', 'EVENT_ID', 'BEGIN_DATE_TIME']
    df_wind_filtered = df_wind_filtered.dropna(subset=required_fips_cols)
    df_wind_filtered['STATE_FIPS'] = pd.to_numeric(df_wind_filtered['STATE_FIPS'], errors='coerce')
    df_wind_filtered['CZ_FIPS'] = pd.to_numeric(df_wind_filtered['CZ_FIPS'], errors='coerce')
    df_wind_filtered = df_wind_filtered.dropna(subset=['STATE_FIPS', 'CZ_FIPS']) # Drop if conversion failed

    if df_wind_filtered.empty:
        logger.warning("No valid STATE_FIPS/CZ_FIPS found after cleaning for SCS wind events. Returning empty DataFrame.")
        return pd.DataFrame(columns=['event_id', 'county_geoid', 'first_impact_dt', 'event_type'])

    # Combine state_fips + cz_fips into a 5-digit code
    # Use try-except within apply for robustness if some rows still cause issues
    def combine_fips(row):
        try:
            return f"{int(row['STATE_FIPS']):02d}{int(row['CZ_FIPS']):03d}"
        except (ValueError, TypeError):
            return None # Handle potential conversion errors

    df_wind_filtered['fips5'] = df_wind_filtered.apply(combine_fips, axis=1)
    df_wind_filtered.dropna(subset=['fips5'], inplace=True) # Drop rows where FIPS construction failed

    df_wind_filtered['BEGIN_DATE_TIME'] = pd.to_datetime(df_wind_filtered['BEGIN_DATE_TIME'], errors='coerce').dt.tz_localize(None)
    df_wind_filtered.dropna(subset=['BEGIN_DATE_TIME'], inplace=True) # Drop rows with invalid dates


    # Keep relevant columns
    df_wind_filtered = df_wind_filtered[['EVENT_ID', 'fips5', 'BEGIN_DATE_TIME']].drop_duplicates()
    df_wind_filtered.rename(columns={
        'EVENT_ID': 'event_id',
        'BEGIN_DATE_TIME': 'first_impact_dt'
    }, inplace=True)
    df_wind_filtered['event_type'] = 'scs_wind'

    # Prepare counties for merge (ensure cnty_5 exists and is clean)
    if 'county_geoid' not in df_counties.columns:
         logger.error("df_counties missing 'county_geoid'. Cannot create 'cnty_5' for merge.")
         return pd.DataFrame(columns=['event_id', 'county_geoid', 'first_impact_dt', 'event_type'])
    df_counties['cnty_5'] = df_counties['county_geoid'].str.strip().str[:5]


    # --- Perform the Merge ---
    logger.debug(f"Attempting merge. SCS wind 'fips5' sample: {df_wind_filtered['fips5'].head().tolist()}. County 'cnty_5' sample: {df_counties['cnty_5'].head().tolist()}")
    merged = pd.merge(
        df_wind_filtered,
        df_counties[['cnty_5', 'county_geoid']], # Select only needed columns from counties
        how='inner', # Use inner merge to keep only matched records
        left_on='fips5',
        right_on='cnty_5'
    )
    n_merged = len(merged)
    n_unmerged = len(df_wind_filtered) - n_merged
    logger.info(f"SCS wind FIPS merge complete. Matched {n_merged} records. Could not match {n_unmerged} SCS wind events to a county GEOID via FIPS.")
    if n_unmerged > 0:
         # Log samples of unmerged FIPS5 codes for further investigation
         unmerged_fips = df_wind_filtered[~df_wind_filtered['fips5'].isin(merged['fips5'])]['fips5'].unique()
         sample_unmerged = min(20, len(unmerged_fips))
         logger.warning(f"Sample of {sample_unmerged} unmerged SCS FIPS codes (format: SSCCC):\n{unmerged_fips[:sample_unmerged]}")


    # Final selection and deduplication
    merged = merged[['event_id','county_geoid','first_impact_dt','event_type']].drop_duplicates()
    logger.info(f"Final unique SCS wind->County records generated: {len(merged)}")
    return merged


def main():
    logger.info("=== Starting map_events_to_counties script (Optimized with BallTree & SCS Fix) ===")

    # 1. Load counties
    counties_file = "data/counties/2024_counties.txt"
    try:
        df_counties = load_counties_txt(counties_file)
    except ValueError as e:
        logger.error(f"Failed to load or process counties file: {e}. Exiting.")
        return # Stop script if counties cannot be loaded

    # 2. Build the spatial index for counties
    county_tree = build_county_kdtree(df_counties)

    # 3. Load cleaned data
    logger.info("Loading cleaned HURDAT, Wildfire, and StormEvents data...")
    try:
        df_hurdat = pd.read_csv("output/cleaned/noaa_hurdat_clean.csv")
        # --- Add cleaning for HURDAT -99 wind speeds here ---
        if 'max_wind_knots' in df_hurdat.columns:
            logger.info("Applying fix for HURDAT -99 max_wind_knots -> NaN")
            df_hurdat['max_wind_knots'] = pd.to_numeric(df_hurdat['max_wind_knots'], errors='coerce')
            df_hurdat.loc[df_hurdat['max_wind_knots'] <= 0, 'max_wind_knots'] = np.nan # Set -99 and any other non-positives to NaN
        # --- End HURDAT Fix ---
    except FileNotFoundError: logger.error("HURDAT file not found."); df_hurdat = pd.DataFrame()
    except Exception as e: logger.exception(f"Error loading HURDAT: {e}"); df_hurdat = pd.DataFrame()

    try:
        df_wildfires = pd.read_csv("output/cleaned/california_wildfires_clean.csv")
        df_wildfires['incident_latitude'] = pd.to_numeric(df_wildfires['incident_latitude'], errors='coerce')
        df_wildfires['incident_longitude'] = pd.to_numeric(df_wildfires['incident_longitude'], errors='coerce')
        # Drop invalid coords AFTER reading acres, in case acres are valid but coords aren't
        df_wildfires.dropna(subset=['incident_acres_burned'], inplace=True) # Acres are needed for mapping
    except FileNotFoundError: logger.error("Wildfires file not found."); df_wildfires = pd.DataFrame()
    except Exception as e: logger.exception(f"Error loading Wildfires: {e}"); df_wildfires = pd.DataFrame()

    try:
        # Ensure CZ_TYPE is loaded for the SCS fix
        # If the file is huge, consider loading only necessary columns:
        storm_cols_needed = ['EVENT_ID', 'EVENT_TYPE', 'STATE_FIPS', 'CZ_FIPS', 'CZ_TYPE', 'BEGIN_DATE_TIME']
        df_stormevents = pd.read_csv("output/cleaned/stormevents_2014_2024_cleaned.csv", low_memory=False, usecols=lambda c: c in storm_cols_needed)
        # Alternatively, load all if memory allows:
        # df_stormevents = pd.read_csv("output/cleaned/stormevents_2014_2024_cleaned.csv", low_memory=False)
        if 'CZ_TYPE' not in df_stormevents.columns:
             logger.error("CRITICAL: 'CZ_TYPE' column not found in loaded StormEvents data. SCS mapping will likely fail or be incorrect.")
    except FileNotFoundError: logger.error("StormEvents file not found."); df_stormevents = pd.DataFrame()
    except ValueError as e: logger.error(f"Column error loading StormEvents (check usecols?): {e}"); df_stormevents = pd.DataFrame()
    except Exception as e: logger.exception(f"Error loading StormEvents: {e}"); df_stormevents = pd.DataFrame()


    logger.info(f"Data loaded: HURDAT rows={len(df_hurdat)}, Wildfires rows={len(df_wildfires)}, StormEvents rows={len(df_stormevents)}")

    # --- Run Mapping Functions (only if data loaded) ---
    df_hurr_counties = pd.DataFrame()
    if not df_hurdat.empty:
        df_hurr_counties = map_hurricanes_to_counties(df_hurdat, df_counties, county_tree)

    df_fire_counties = pd.DataFrame()
    if not df_wildfires.empty:
         # Drop wildfires with invalid coords before mapping
         df_wildfires_valid_coords = df_wildfires.dropna(subset=['incident_latitude', 'incident_longitude']).copy()
         if len(df_wildfires_valid_coords) < len(df_wildfires):
              logger.warning(f"Dropped {len(df_wildfires) - len(df_wildfires_valid_coords)} wildfires with missing lat/lon before mapping.")
         if not df_wildfires_valid_coords.empty:
              df_fire_counties = map_wildfires_to_counties(df_wildfires_valid_coords, df_counties, county_tree)

    df_scs_counties = pd.DataFrame()
    if not df_stormevents.empty:
        df_scs_counties = map_scs_wind_to_counties(df_stormevents, df_counties)

    # 7. Combine
    logger.info("Combining all county-event mappings...")
    all_dfs = [df for df in [df_hurr_counties, df_fire_counties, df_scs_counties] if not df.empty]
    if not all_dfs:
        logger.error("No event-county mappings were generated. Cannot create final output.")
        return

    df_county_events = pd.concat(all_dfs, ignore_index=True)

    # Ensure final uniqueness
    final_rows_before_drop = len(df_county_events)
    df_county_events.drop_duplicates(subset=['event_id','county_geoid','event_type'], inplace=True, keep='first')
    final_rows_after_drop = len(df_county_events)
    logger.info(f"Combined records: {final_rows_before_drop}. After final drop_duplicates: {final_rows_after_drop}")


    # 8. Save
    os.makedirs("output/final", exist_ok=True)
    outpath = "output/final/county_events.csv"
    df_county_events.to_csv(outpath, index=False)
    logger.info(f"Saved final county_events table to: {outpath}")
    logger.info("=== map_events_to_counties script completed successfully. ===")

if __name__ == "__main__":
    main()