# src/data_cleaning.py
import os
import logging
import pandas as pd
import numpy as np

from src.logging_config import setup_logger

logger = setup_logger(name="pipeline_logger")

def clean_calfire_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the California Wildfire DataFrame by removing duplicates,
    handling missing or erroneous values, etc.
    """
    logger.info("Cleaning CAL FIRE DataFrame...")
    
    initial_rows = df.shape[0]

    # Drop exact duplicates
    df.drop_duplicates(inplace=True)
    after_dup = df.shape[0]
    logger.debug(f"Removed {initial_rows - after_dup} duplicate rows. Remaining: {after_dup}")

    # For incident_acres_burned, remove or fix negative or impossible values
    if "incident_acres_burned" in df.columns:
        invalid_acres = df[df["incident_acres_burned"] < 0].shape[0]
        if invalid_acres > 0:
            logger.warning(f"Found {invalid_acres} rows with negative acres. Setting them to NaN.")
            df.loc[df["incident_acres_burned"] < 0, "incident_acres_burned"] = np.nan

    # Handle missing required columns
    if "incident_date_created" in df.columns:
        missing_dates = df["incident_date_created"].isna().sum()
        if missing_dates > 0:
            logger.warning(f"{missing_dates} rows have missing creation date. Dropping them.")
            df = df.dropna(subset=["incident_date_created"])

    logger.info(f"Finished cleaning CAL FIRE DataFrame. Final row count: {df.shape[0]}")
    return df


def clean_hurdat_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the NOAA HURDAT DataFrame by removing duplicates, handling
    special missing codes (-999), and ensuring valid lat/lon/wind speeds.
    """
    logger.info("Cleaning HURDAT DataFrame...")
    
    initial_rows = df.shape[0]

def clean_hurdat_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning HURDAT DataFrame...")

    # Example: remove duplicates
    df.drop_duplicates(inplace=True)

    # 1. Convert “-99” to NaN for max wind, and “-999” if you see that too
    if "max_wind_knots" in df.columns:
        mask_neg99 = df["max_wind_knots"] == -99
        mask_neg999 = df["max_wind_knots"] == -999
        total_neg99 = mask_neg99.sum()
        total_neg999 = mask_neg999.sum()
        if total_neg99 > 0 or total_neg999 > 0:
            logger.warning(f"Found {total_neg99} occurrences of -99 and {total_neg999} of -999 in max_wind_knots. Converting to NaN.")
            df.loc[mask_neg99, "max_wind_knots"] = np.nan
            df.loc[mask_neg999, "max_wind_knots"] = np.nan

    # 2. Convert -999 to NaN for min_pressure_mb (or other columns)
    if "min_pressure_mb" in df.columns:
        mask_pressure = df["min_pressure_mb"] == -999
        if mask_pressure.any():
            logger.warning(f"Found {mask_pressure.sum()} rows with min_pressure_mb = -999. Converting to NaN.")
            df.loc[mask_pressure, "min_pressure_mb"] = np.nan

    # Similarly for max_wind_knots
    if "max_wind_knots" in df.columns:
        mask_wind = df["max_wind_knots"] == -999
        if mask_wind.sum() > 0:
            logger.warning(f"Found {mask_wind.sum()} rows with max_wind_knots = -999. Setting them to NaN.")
            df.loc[mask_wind, "max_wind_knots"] = np.nan

    # Optionally remove rows with missing lat/lon
    if "latitude" in df.columns and "longitude" in df.columns:
        missing_coords = df[df["latitude"].isna() | df["longitude"].isna()].shape[0]
        if missing_coords > 0:
            logger.info(f"Dropping {missing_coords} rows with missing lat/lon.")
            df = df.dropna(subset=["latitude", "longitude"])

    logger.info(f"Finished cleaning HURDAT DataFrame. Final row count: {df.shape[0]}")
    return df

def clean_storm_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the combined StormEvents DataFrame by removing duplicates,
    handling missing columns, etc.
    
    :param df: The raw, combined DataFrame of StormEvents
    :return: A cleaned DataFrame
    """
    logger.info("Cleaning StormEvents DataFrame...")

    if df.empty:
        logger.warning("StormEvents DataFrame is empty. Nothing to clean.")
        return df

    # 1. Drop exact duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    after_dups = df.shape[0]
    logger.debug(f"Removed {initial_rows - after_dups} duplicate rows. Remaining: {after_dups}")

    # 2. Handle missing or invalid values in critical columns
    # Example: If there's a 'BEGIN_DATE_TIME' or 'END_DATE_TIME' col, parse them as datetimes
    date_cols = ["BEGIN_DATE_TIME", "END_DATE_TIME"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            missing_dates = df[col].isna().sum()
            if missing_dates > 0:
                logger.warning(f"{missing_dates} rows have invalid/missing {col}")

    # 3. Example: If there's a numeric column like 'BEGIN_LAT', 'BEGIN_LON', convert to numeric
    numeric_cols = ["BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Drop rows missing essential location data (if that matters)
    if "BEGIN_LAT" in df.columns and "BEGIN_LON" in df.columns:
        missing_coords = df[df["BEGIN_LAT"].isna() | df["BEGIN_LON"].isna()]
        logger.warning(f"{missing_coords.shape[0]} rows have missing BEGIN_LAT or BEGIN_LON. Keeping them for review.")
        
        # Optional: Save them separately for inspection
        os.makedirs("./output/debug", exist_ok=True)
        debug_path = "./output/debug/stormevents_missing_latlon.csv"
        missing_coords.to_csv(debug_path, index=False)
        logger.info(f"Saved rows with missing lat/lon to: {debug_path}")


    logger.info(f"Final StormEvents row count after cleaning: {df.shape[0]}")
    return df
