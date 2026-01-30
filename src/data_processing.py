# %%
import numpy as np
import torch
import polars as pl

from pathlib import Path

import airportsdata as ad

class DataProcessing:
    def __init__(self):
        self.cwd = Path.cwd().parent
        self.excel_path = Path(self.cwd, "data/Airlink - UMichiagn - Data Collection - 9.8.2025.xlsx")
        self.mapping = {}
        self.airports = ad.load('IATA')


    def geolocate_nodes(self, df: pl.DataFrame) -> pl.DataFrame:
        def get_lat_lon(code):
            airport = self.airports.get(code)
            if airport is None:
                # TODO add in geopandas based or OSM api based geolocation for cities that are not airports
                return {"lat": None, "lon": None}
            return {"lat": airport["lat"], "lon": airport["lon"]}

        geo_dtype = pl.Struct([
            pl.Field("lat", pl.Float64),
            pl.Field("lon", pl.Float64),
        ])

        ret = df.with_columns([
            pl.col("Origin")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lat")
            .alias("origin_Lat"),

            pl.col("Origin")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lon")
            .alias("origin_Lon"),

            pl.col("Destination")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lat")
            .alias("destination_Lat"),

            pl.col("Destination")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lon")
            .alias("destination_Lon"),
        ])
        
        return ret
    

    def calculate_distance(self, df: pl.DataFrame) -> pl.DataFrame:
        R = 6371.0  # Earth's radius in km

        dlat = (pl.col("destination_Lat") - pl.col("origin_Lat")).radians()
        dlon = (pl.col("destination_Lon") - pl.col("origin_Lon")).radians()
        lat1 = pl.col("origin_Lat").radians()
        lat2 = pl.col("destination_Lat").radians()

        # Haversine formula
        a = ((dlat / 2).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2).sin().pow(2))

        c = 2 * a.sqrt().arcsin()

        # NOTE this gives distance in km
        ret =  df.with_columns((c * R).alias("distance"))

        return ret
        

    def load_shipping_data(self, filepath: Path) -> pl.DataFrame:
        # Read Raw (as strings) to handle mixed headers
        df_raw = pl.read_excel(filepath, has_header=False, infer_schema_length=0)

        # Extract Year & Clean
        df_clean = (
            df_raw
            .with_columns(
                pl.col("column_1")
                .str.extract(r"^(\d{4})", 1)
                .forward_fill()
                .alias("Year")
            )
            .filter(
                ~pl.col("column_1").str.contains(r"^\d{4}.*Completed Shipments"),
                pl.col("column_1") != "NGO ID",
                pl.col("column_1") != "Mirror",
                pl.col("column_2").is_not_null(),
                pl.col("column_2").str.strip_chars() != ""
            )
        )

        # Rename Columns (Mapping from row 1 of raw file)
        header_row = df_raw.row(1)
        rename_map = {f"column_{i+1}": name for i, name in enumerate(header_row) if name}
        
        df_final = (
            df_clean
            .rename(rename_map)
            .with_columns([
                pl.col("Year").cast(pl.Int32)
                # pl.lit(1.0).alias("s_label") # These are all Positives
            ])
        )
        return df_final


    def preprocess_to_tensor(self, df: pl.DataFrame) -> torch.Tensor:
        # NOTE these feature columns may need to be adjusted based on data available
        feature_cols = [
            "Origin",
            'Destination',
            'origin_Lat',
            'origin_Lon',
            'destination_Lat',
            'destination_Lon',
            "AW (kg)",
            "Pallets",
            'distance'
        ]

        # check if the values in the column can be directly converted to numbers if not build a map and encode
        for col in feature_cols:
            try: 
                # NOTE : fill Nones are creating issues changing to zero for now, need to fully decide how these are handled going forward
                df = df.with_columns(pl.col(col).replace("", None).cast(pl.Float32).fill_null(0.0))
            except:          
                uniq_vals = df[col].unique().to_list()
                self.mapping[col] = {val: idx for idx, val in enumerate(uniq_vals)}
                df = df.with_columns(
                    pl.Series(col, [self.mapping[col][val] for val in df[col]], dtype=pl.Int32)
                )

        data_np = df.select(feature_cols).to_numpy().astype(np.float32)

        # Normalize data for ease of training
        data_tensor = torch.tensor(data_np)
        mean = data_tensor.mean(dim=0)
        std = data_tensor.std(dim=0) + 1e-6
        data_tensor = (data_tensor - mean) / (std + 1e-6)

        return data_tensor
    
    def transform_new_data(self, df: pl.DataFrame) -> torch.Tensor:
        # TODO : verify correctness and ability to handle unseen categories if not conditions managed above in preprocessing
        feature_cols = [
            "Origin",
            'Dest',
            "AW (kg)",
            "Pallets",
            'origin_Lat',
            'origin_Lon',
            'destination_Lat',
            'destination_Lon'
        ]

        for col, map in self.mapping.items():
            if col in self.mapping:
                df = df.with_columns(
                    pl.Series(col, [self.mapping[col].get(val, -1) for val in df[col]], dtype=pl.Int32)
                )
            else:
                df = df.with_columns(pl.col(col).cast(pl.Float32))

        data_np = df.select(feature_cols).to_numpy().astype(np.float32)
        return torch.tensor(data_np)
    
# %%
if __name__ == "__main__":
    dp = DataProcessing()
    df = dp.load_shipping_data(dp.excel_path)
    df_geo = dp.geolocate_nodes(df)
    df_dist = dp.calculate_distance(df_geo)
    # TODO integrate processing
    data_tensor = dp.preprocess_to_tensor(df_dist)
    print(data_tensor.shape)