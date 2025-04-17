# src/data_ingestion.py

import os
import re
import logging
import pandas as pd
from datetime import datetime
from typing import List, Union
import glob
# We import our logger from the logging_config
from src.logging_config import setup_logger

logger = setup_logger(name="pipeline_logger")

def load_calfire_data(file_path: str) -> pd.DataFrame:
    """
    Loads the California Wildfire data from a CSV file.
    
    :param file_path: Path to the CSV containing CAL FIRE incidents
    :return: DataFrame with parsed wildfire data
    """
    logger.info(f"Loading CAL FIRE data from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Could not find file: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Successfully read {df.shape[0]} rows and {df.shape[1]} columns from {file_path}")
        
        # Example columns to parse: 
        # "incident_date_created", "incident_date_last_update", "incident_acres_burned"
        # Convert date columns
        date_cols = ["incident_date_created", "incident_date_last_update", "incident_date_extinguished"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.debug(f"Parsed '{col}' as datetime")

        # Numeric columns
        numeric_cols = ["incident_acres_burned", "incident_latitude", "incident_longitude"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.debug(f"Converted '{col}' to numeric")

        logger.info("Finished loading California Wildfire data.")
        return df

    except Exception as e:
        logger.exception("Failed to load CAL FIRE data.")
        raise e


def load_hurdat_data(file_path: str) -> pd.DataFrame:
    """
    Loads and parses NOAA HURDAT data from a text file.
    Each storm has 1 header line + N data lines. 
    For example:
      AL092021, IDA, 40,
      20210826, 1200, , TD, ...
      (40 data lines)

    :param file_path: Path to HURDAT .txt file
    :return: DataFrame with parsed HURDAT data
    """
    logger.info(f"Loading NOAA HURDAT data from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    all_records = []  # will hold dictionaries for each data line
    
    try:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        
        idx = 0
        while idx < len(lines):
            header_line = lines[idx]
            idx += 1
            
            # Parse header line, e.g.: "AL092021, IDA, 40,"
            header_parts = [p.strip() for p in header_line.split(",")]
            if len(header_parts) < 3:
                logger.warning(f"Skipping malformed header line: {header_line}")
                continue
            
            storm_id = header_parts[0]  # e.g. "AL092021"
            storm_name = header_parts[1]  # e.g. "IDA"
            num_data_lines = int(header_parts[2])  # e.g. 40

            # logger.info(f"Found storm {storm_id}, name {storm_name}, with {num_data_lines} data lines.")

            # Now parse the next N lines for data
            for _ in range(num_data_lines):
                if idx >= len(lines):
                    logger.error("Reached end of file before parsing all lines for storm.")
                    break
                
                data_line = lines[idx]
                idx += 1

                data_parts = [p.strip() for p in data_line.split(",")]
                # data line example:
                # "20210826, 1200, , TD, 16.5N, 78.9W, 30, 1006, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60"
                
                if len(data_parts) < 8:
                    logger.warning(f"Skipping malformed data line: {data_line}")
                    continue

                # Extract data from data_parts
                # data_parts[0] -> '20210826' (YYYYMMDD)
                # data_parts[1] -> '1200' (HHMM)
                # data_parts[2] -> record identifier (could be '', 'L', etc.)
                # data_parts[3] -> system status (TD, TS, HU, EX, etc.)
                # data_parts[4] -> latitude (e.g. '16.5N')
                # data_parts[5] -> longitude (e.g. '78.9W')
                # data_parts[6] -> max sustained wind (knots)
                # data_parts[7] -> min pressure (mb)
                # ...
                
                date_str = data_parts[0].strip()
                time_str = data_parts[1].strip()
                
                # Convert date/time
                # e.g. '20210826' -> datetime(2021,08,26)
                # '1200' -> 12:00 UTC
                # If time_str is not strictly 4 digits, handle carefully
                # For safety, we can do a check:
                
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                if len(time_str) == 4:
                    hour = int(time_str[0:2])
                    minute = int(time_str[2:4])
                else:
                    hour = 0
                    minute = 0
                
                # Build a datetime object
                dt = datetime(year, month, day, hour, minute)
                
                record_id = data_parts[2] if data_parts[2] != "" else None
                status = data_parts[3]
                
                # Parse lat and lon
                lat_hem = data_parts[4][-1].upper()  # last char: 'N' or 'S'
                lat_val = float(data_parts[4][:-1])  # everything except last char
                if lat_hem == 'S':
                    lat_val = -lat_val
                    
                lon_hem = data_parts[5][-1].upper()  # 'W' or 'E'
                lon_val = float(data_parts[5][:-1])
                if lon_hem == 'W':
                    lon_val = -lon_val
                
                # Attempt to parse max wind and pressure
                try:
                    max_wind_knots = float(data_parts[6])
                except ValueError:
                    max_wind_knots = None

                try:
                    min_pressure_mb = float(data_parts[7])
                except ValueError:
                    min_pressure_mb = None

                # We can keep going for 34-kt wind radii, 50-kt wind radii, etc.
                # data_parts[8] -> 34 kt wind NE quadrant
                # ...
                # data_parts[19 or 20?] -> radius of max wind
                # Because these can be optional or -999, let's parse them carefully
                # For brevity, we parse just a few more:
                if len(data_parts) > 8:
                    try:
                        wind_radii_34kt_ne = float(data_parts[8]) if data_parts[8] != "-999" else None
                    except ValueError:
                        wind_radii_34kt_ne = None
                else:
                    wind_radii_34kt_ne = None
                
                # We'll store everything in a dictionary
                record_dict = {
                    "storm_id": storm_id,
                    "storm_name": storm_name,
                    "datetime": dt,
                    "record_identifier": record_id,
                    "status": status,
                    "latitude": lat_val,
                    "longitude": lon_val,
                    "max_wind_knots": max_wind_knots,
                    "min_pressure_mb": min_pressure_mb,
                    "wind_radii_34kt_ne": wind_radii_34kt_ne,
                    # add more as needed...
                }
                all_records.append(record_dict)
        
        # Convert all_records to DataFrame
        df = pd.DataFrame(all_records)
        logger.info(f"Parsed {df.shape[0]} HURDAT data rows from file: {file_path}")
        return df

    except Exception as e:
        logger.exception("Failed to parse HURDAT data.")
        raise e

def load_storm_events(data_dir: str) -> pd.DataFrame:
    """
    Loads and combines multiple NOAA StormEvents CSV.gz files from the given directory.
    The files typically follow a naming pattern, e.g.:
      StormEvents_details-ftp_v1.0_d2021_c20250401.csv.gz
    spanning multiple years (e.g., 2014-2024).
    
    :param data_dir: Path to the directory containing StormEvents .csv.gz files
    :return: A single DataFrame containing all rows from all files
    """
    logger.info(f"Loading StormEvents data from directory: {data_dir}")

    # List all .csv.gz files
    pattern = os.path.join(data_dir, "*.csv.gz")
    files = glob.glob(pattern)
    if not files:
        logger.warning(f"No StormEvents CSV files found in {data_dir}")
        return pd.DataFrame()  # return empty DataFrame

    all_dfs = []
    for file in files:
        try:
            # Example:
            # StormEvents_details-ftp_v1.0_d2021_c20250401.csv.gz
            file_name = os.path.basename(file)
            logger.info(f"Reading file: {file_name}")
            
            # Read gzipped CSV
            df = pd.read_csv(file, compression='gzip', low_memory=False)
            logger.debug(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {file_name}")
            
            # OPTIONAL: If you want to label each row by year or file, you can do:
            # year = extract_year_from_filename(file_name)  # e.g. parse out '2021'
            # df['year_file'] = year

            all_dfs.append(df)
        except Exception as e:
            logger.exception(f"Error reading {file_name}: {e}")

    if not all_dfs:
        logger.warning("No valid dataframes were loaded.")
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    logger.info(f"Combined StormEvents data shape: {combined_df.shape}")

    return combined_df
