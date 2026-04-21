# %%
import numpy as np
import torch
import polars as pl

from pathlib import Path
from dataclasses import dataclass, field

import airportsdata as ad

# %%
@dataclass
class FeatureConfig:
    target_column: str = "Aircraft Type"

    categorical_features: list[str] = field(default_factory=lambda: [
        "Origin", "Destination"])
    
    numerical_features: list[str] = field(default_factory=lambda: [
        "AW (kg)",
        "Pallets",
        'Origin_Lat',
        'Origin_Lon',
        'Destination_Lat',
        'Destination_Lon',
        'distance'
    ])

    
    def add_feature(self, feature_name: str, is_categorical: bool):
        if is_categorical:
            self.categorical_features.append(feature_name)
        else:
            self.numerical_features.append(feature_name)

    @property
    def all_features(self) -> list[str]:
        features = self.categorical_features + self.numerical_features            
        return features
    
    @property
    def features_with_target(self) -> list[str]:
        return self.all_features + [self.target_column]
    

class DataProcessing:
    def __init__(self):
        self.data_dir = Path(__file__).resolve().parent.parent
        self.excel_path = Path(self.data_dir, "data/Airlink - UMichiagn - Data Collection - 9.8.2025.xlsx")
        self.mapping = {}
        self.airports = ad.load('IATA')

        self.features = FeatureConfig()


    def _geolocate_nodes(self, df: pl.DataFrame) -> pl.DataFrame:
        def get_lat_lon(code):
            airport = self.airports.get(code)
            if airport is None:
                # TODO add in geopandas based or OSM api based geolocation for cities that are not airports
                return {"lat": None, "lon": None}
            return {"lat": airport["lat"], "lon": airport["lon"]}

        # NOTE defined a new dtype for the geolocation struct to ensure correct typing and avoid issues with missing values which can cause type inference problems in Polars when using map_elements
        geo_dtype = pl.Struct([
            pl.Field("lat", pl.Float64),
            pl.Field("lon", pl.Float64),
        ])

        ret = df.with_columns([
            pl.col("Origin")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lat")
            .alias("Origin_Lat"),

            pl.col("Origin")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lon")
            .alias("Origin_Lon"),

            pl.col("Destination")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lat")
            .alias("Destination_Lat"),

            pl.col("Destination")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lon")
            .alias("Destination_Lon"),
        ])
        
        return ret
    

    def _calculate_distance(self, df: pl.DataFrame) -> pl.DataFrame:
        # Standard Haversine formula implementation to calculate distance between two lat/lon points
        R = 6371.0  # Earth's radius in km

        dlat = (pl.col("Destination_Lat") - pl.col("Origin_Lat")).radians()
        dlon = (pl.col("Destination_Lon") - pl.col("Origin_Lon")).radians()
        lat1 = pl.col("Origin_Lat").radians()
        lat2 = pl.col("Destination_Lat").radians()

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
        
        exclude_types = ["Trucking", "Multimodal", "Ocean Freight"]

        df_final = (
            df_clean
            .rename(rename_map)
            .with_columns(pl.col("Year").cast(pl.Int32))
            .filter(
                pl.col("Aircraft Type").is_not_null()
                & (pl.col("Aircraft Type").str.strip_chars() != "")
                & ~pl.col("Aircraft Type").is_in(exclude_types)
            )
        )

        # TODO in df final, check origin destination, if either is not in airports data, raise error with list of unknown codes
        origin_codes = set(df_final["Origin"].unique().to_list())
        destination_codes = set(df_final["Destination"].unique().to_list())
        unknown_origin_codes = sorted(origin_codes.difference(self.airports.keys()), key=lambda value: str(value))
        unknown_destination_codes = sorted(destination_codes.difference(self.airports.keys()), key=lambda value: str(value))
        # if unknown_origin_codes or unknown_destination_codes:
        #     raise ValueError(
        #         f"Unknown airport codes found. Unknown Origins: {unknown_origin_codes}, Unknown Destinations: {unknown_destination_codes}. "
        #         "Please ensure all Origin and Destination codes are valid IATA codes present in the airports dataset."
        #     )

        # NOTE drop these rows for now
        df_final = df_final.filter(
            ~pl.col("Origin").is_in(unknown_origin_codes) &
            ~pl.col("Destination").is_in(unknown_destination_codes)
        )

        return df_final


    def _encode_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # Encode target variable
        target_col = self.features.target_column
        canonical_target_map = {
            "Narrowbody": 0,
            "Widebody": 1,
            "Freighter": 2,
        }

        observed_types = set(df[target_col].unique().to_list())
        unknown_types = sorted(observed_types.difference(canonical_target_map.keys()), key=lambda value: str(value))
        if unknown_types:
            raise ValueError(
                f"Unexpected aircraft types in '{target_col}': {unknown_types}. "
                "Expected only ['Narrowbody', 'Widebody', 'Freighter']."
            )

        self.mapping[target_col] = {
            "Narrowbody": 0,
            "Widebody": 1,
            "Freighter": 2,
        }

        df = df.with_columns(
            pl.Series(
                target_col,
                [canonical_target_map[val] for val in df[target_col]],
                dtype=pl.Int32,
            )
        )

        # NOTE these feature columns may need to be adjusted based on data available
        feature_cols = self.features.all_features

        # check if the values in the column can be directly converted to numbers if not build a map and encode
        for col in feature_cols:
            try: 
                # NOTE : fill Nones are creating issues changing to zero for now, need to fully decide how these are handled going forward
                df = df.with_columns(pl.col(col).replace("", None).cast(pl.Float32).fill_null(0.0))
            except:          
                # Sort unique values to keep categorical mapping stable across runs.
                uniq_vals = sorted(df[col].unique().to_list(), key=lambda value: str(value))
                self.mapping[col] = {val: idx for idx, val in enumerate(uniq_vals)}
                df = df.with_columns(
                    pl.Series(col, [self.mapping[col][val] for val in df[col]], dtype=pl.Int32)
                )

        return df


    def _normalize_data(self, data_tensor: torch.Tensor) -> torch.Tensor:
        # data_tensor = torch.tensor(data_np)
        mean = data_tensor.mean(dim=0)
        std = data_tensor.std(dim=0) + 1e-6
        data_tensor = (data_tensor - mean) / (std + 1e-6)

        return data_tensor
    
    def to_tensor(self, df: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        data_ = self._encode_features(df)
        
        input_tensor = torch.tensor(data_.select(self.features.all_features).to_numpy().astype(np.float32))
        # Normalize data for ease of training
        # TODO do normalization elsewhere to prevent data leakage and ensure correct handling of new data
        data_tensor = self._normalize_data(input_tensor)

        target_tensor = torch.tensor(data_.select(self.features.target_column).to_numpy().astype(np.int64))

        return data_tensor, target_tensor
    

    def transform_new_data(self, df: pl.DataFrame) -> torch.Tensor:
        raise NotImplementedError("This method needs to be implemented to handle new data preprocessing.")
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
    
    def process_data(self, filepath: Path = Path()) -> tuple[torch.Tensor, torch.Tensor]:
        if filepath == Path():
            filepath = self.excel_path

        df_shipping = self.load_shipping_data(filepath)
        df_geo = self._geolocate_nodes(df_shipping)
        df_dist = self._calculate_distance(df_geo)
        self.data_tensor, self.target_tensor = self.to_tensor(df_dist)

        return self.data_tensor, self.target_tensor
    
# %%
class T100DataProcessing(DataProcessing):
    def __init__(self):
        super().__init__()
        self.excel_path = Path(self.data_dir, "data/T_T100I_SEGMENT_ALL_CARRIER.csv")

    def filter_data(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError("This method needs to be implemented to handle specific filtering for T100 data.")

# %%
if __name__ == "__main__":
    dp = DataProcessing()
    # df = dp.load_shipping_data(dp.excel_path)
    # df_geo = dp.geolocate_nodes(df)
    # df_dist = dp.calculate_distance(df_geo)
    # TODO integrate processing
    # data_tensor = dp.preprocess_to_tensor(df_dist)
    data_tensor, target_tensor = dp.process_data()
    print(data_tensor.shape)
# %%