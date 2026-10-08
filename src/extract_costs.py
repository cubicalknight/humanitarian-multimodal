# %%
import polars as pl
import numpy as np
from pathlib import Path
from data_processing import DataProcessing
import matplotlib.pyplot as plt

def _explode_airline_names(frame: pl.DataFrame, airline_col: str = "Airline") -> pl.DataFrame:
    if airline_col not in frame.columns:
        raise ValueError(f"Expected '{airline_col}' to be present in the data frame.")

    return (
        frame
        .select(pl.all())
        .with_columns(
            pl.col(airline_col)
            .cast(pl.Utf8)
            .str.split(",")
            .alias("airline_name")
        )
        .explode("airline_name")
        .with_columns(pl.col("airline_name").str.strip_chars())
        .filter(pl.col("airline_name").is_not_null() & (pl.col("airline_name") != ""))
    )

def _build_airline_price_profile(train_frame: pl.DataFrame) -> tuple[dict[str, float], float, int]:
    weight_column_candidates = ["shipment_weight_kg", "AW (kg)"]
    weight_column = next((c for c in weight_column_candidates if c in train_frame.columns), None)

    required_columns = {"Airline", "Commercial Cost for Long-Haul", "distance"}
    missing_columns = sorted(required_columns.difference(train_frame.columns))
    if missing_columns:
        raise ValueError(f"Cannot build airline price profile; missing columns: {missing_columns}")
    if weight_column is None:
        raise ValueError(
            "Cannot build airline price profile; missing shipment weight column. "
            f"Expected one of {weight_column_candidates}."
        )

    priced_rows = (
        _explode_airline_names(
            train_frame.select(["Airline", "Commercial Cost for Long-Haul", "distance", weight_column]),
            airline_col="Airline",
        )
        .with_columns([
            pl.col("Commercial Cost for Long-Haul")
            .cast(pl.Utf8)
            .str.replace_all(",", "")
            .cast(pl.Float64, strict=False)
            .alias("long_haul_cost"),
            pl.col("distance").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False).alias("distance_km"),
            pl.col(weight_column).cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False).alias("shipment_weight_kg"),
        ])
        .with_columns((pl.col("distance_km") * 0.621371).alias("distance_miles"))
        .filter(
            pl.col("long_haul_cost").is_not_null()
            & pl.col("distance_miles").is_not_null()
            & pl.col("shipment_weight_kg").is_not_null()
            & (pl.col("long_haul_cost") > 0)
            & (pl.col("distance_miles") > 0)
            & (pl.col("shipment_weight_kg") > 0)
        )
        .with_columns((pl.col("long_haul_cost") / (pl.col("distance_miles") * pl.col("shipment_weight_kg"))).alias("price_per_mile_kg"))
    )

    if priced_rows.is_empty():
        raise ValueError("No valid airline pricing rows remained after cleaning training data.")

    airline_mean_price_per_mile = (
        priced_rows
        .group_by("airline_name")
        .agg(pl.col("price_per_mile_kg").mean().alias("mean_price_per_mile"))
        .sort("airline_name")
    )

    airline_price_map = {
        str(row[0]): float(row[1])
        for row in airline_mean_price_per_mile.select(["airline_name", "mean_price_per_mile"]).iter_rows()
    }
    if not airline_price_map:
        raise ValueError("No airline price-per-mile statistics could be computed.")

    global_mean_price_per_mile = float(priced_rows.select(pl.col("price_per_mile_kg").mean()).item())
    if not np.isfinite(global_mean_price_per_mile) or global_mean_price_per_mile <= 0:
        raise ValueError(f"Invalid global mean price-per-mile computed: {global_mean_price_per_mile}")

    return airline_price_map, global_mean_price_per_mile, int(priced_rows.height)

def main():
    # Initialize data processing to load the data
    dp = DataProcessing()
    
    print(f"Loading data from: {dp.excel_path}")
    df = dp.load_shipping_data(dp.excel_path)
    
    # The cost extraction in main.py expects 'distance' to be present.
    # In DataProcessing, _geolocate_nodes and _calculate_distance provide 'DISTANCE' (uppercase).
    # We need to ensure column names match what _build_airline_price_profile expects.
    
    df = dp._geolocate_nodes(df)
    df = dp._calculate_distance(df)
    
    # Rename DISTANCE to distance for compatibility with the extraction function
    if "DISTANCE" in df.columns:
        df = df.rename({"DISTANCE": "distance"})
        
    # Fix: DataProcessing renames "AW (kg)" to "AW (lbs)". 
    # We map "AW (lbs)" back to "shipment_weight_kg" so the extraction function can find it.
    if "AW (lbs)" in df.columns:
        df = df.rename({"AW (lbs)": "shipment_weight_kg"})

    print("Extracting airline price profiles...")
    try:
        airline_map, global_mean, row_count = _build_airline_price_profile(df)
        
        print("\n=== Extraction Results ===")
        print(f"Total valid pricing rows used: {row_count}")
        print(f"Global mean price per mile-kg: {global_mean:.6f}")
        print(f"Global std price per mile-kg: {np.std(list(airline_map.values())):.6f}")

        fig = plt.figure(figsize=(12, 6))
        # x axis should be values, y axis counts in buckets
        plt.hist(list(airline_map.values()), bins=5, color='skyblue', edgecolor='black')
        plt.show()

        print(f"Number of airlines found: {len(airline_map)}")
        print("\nAirline Price Map:")
        for airline, price in sorted(airline_map.items()):
            print(f"  {airline}: {price:.6f}")
            
    except Exception as e:
        print(f"Error during cost extraction: {e}")

if __name__ == "__main__":
    main()
