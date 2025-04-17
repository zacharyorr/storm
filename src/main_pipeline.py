# src/main_pipeline.py

import os
from src.logging_config import setup_logger
from src.data_ingestion import (
    load_calfire_data,
    load_hurdat_data,
    load_storm_events  # new function
)
from src.data_cleaning import (
    clean_calfire_data,
    clean_hurdat_data,
    clean_storm_events  # new function
)

logger = setup_logger(name="pipeline_logger")

def main():
    logger.info("=== Starting main pipeline for Wildfire + HURDAT + StormEvents data ===")

    # ---------------------------
    # Example existing logic
    # ---------------------------
    # 1. Ingest & clean CAL FIRE
    wildfires_path = "./data/wildfires/california_wildfires.csv"
    df_wildfires = load_calfire_data(wildfires_path)
    df_wildfires_clean = clean_calfire_data(df_wildfires)

    # 2. Ingest & clean NOAA HURDAT
    hurdat_path = "./data/hurdat/NOAA_Hurdat.txt"
    df_hurdat = load_hurdat_data(hurdat_path)
    df_hurdat_clean = clean_hurdat_data(df_hurdat)

    # ---------------------------
    # New: StormEvents ingestion
    # ---------------------------
    storm_dir = "./data/Stormevents"
    df_storm_raw = load_storm_events(storm_dir)
    df_storm_clean = clean_storm_events(df_storm_raw)

    # 3. Save all cleaned data
    os.makedirs("./output/cleaned", exist_ok=True)
    df_wildfires_clean.to_csv("./output/cleaned/california_wildfires_clean.csv", index=False)
    df_hurdat_clean.to_csv("./output/cleaned/noaa_hurdat_clean.csv", index=False)

    # NEW: Save combined StormEvents file
    df_storm_clean.to_csv("./output/cleaned/stormevents_2014_2024_cleaned.csv", index=False)

    logger.info("=== Pipeline completed successfully. ===")


if __name__ == "__main__":
    main()
