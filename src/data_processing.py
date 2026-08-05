# %%
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import airportsdata as ad
import numpy as np
import polars as pl
import torch
from scipy.stats import truncnorm


# %%
@dataclass
class FeatureConfig:
    target_column: str = "Aircraft Type"

    # standardized naming conventions
    categorical_features: list[str] = field(default_factory=lambda: [
        "ORIGIN", "DEST", "UNIQUE_CARRIER"])

    # categorical_features: list[str] = field(default_factory=lambda: [
    #     "Origin", "Destination"])
    
    numerical_features: list[str] = field(default_factory=lambda: [
        "Origin_Lat",
        "Origin_Lon",
        "Destination_Lat",
        "Destination_Lon",
        "DISTANCE"
    ])
    # numerical_features: list[str] = field(default_factory=lambda: [
    #     "AW (kg)",
    #     "Pallets",
    #     'Origin_Lat',
    #     'Origin_Lon',
    #     'Destination_Lat',
    #     'Destination_Lon',
    #     'distance'
    # ])

    
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
    def __init__(self, feature_config: FeatureConfig | None = None, separate_target: bool = False):
        self.data_dir = Path(__file__).resolve().parent.parent
        self.excel_path = Path(self.data_dir, "data/Airlink - UMichiagn - Data Collection - 9.8.2025.xlsx")
        self.mapping = {}
        self.airports = ad.load('IATA')

        self.separate_target = separate_target
        self.features = feature_config or FeatureConfig()
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def _canonicalize_route_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        rename_map = {
            "Origin": "ORIGIN",
            "Destination": "DEST",
            "Airline": "UNIQUE_CARRIER",
            "distance": "DISTANCE",
        }
        active_renames = {
            source: target
            for source, target in rename_map.items()
            if source in df.columns and target not in df.columns
        }
        if active_renames:
            df = df.rename(active_renames)
        return df

    def export_alignment_state(self) -> dict[str, object]:
        return {
            "mapping": deepcopy(self.mapping),
            "feature_mean": None if self.feature_mean is None else self.feature_mean.clone(),
            "feature_std": None if self.feature_std is None else self.feature_std.clone(),
            "features": deepcopy(self.features),
        }

    def import_alignment_state(self, state: dict[str, object]) -> None:
        self.mapping = deepcopy(state.get("mapping", {}))

        feature_mean = state.get("feature_mean")
        feature_std = state.get("feature_std")
        features = state.get("features")

        self.feature_mean = feature_mean.clone() if isinstance(feature_mean, torch.Tensor) else feature_mean
        self.feature_std = feature_std.clone() if isinstance(feature_std, torch.Tensor) else feature_std
        if isinstance(features, FeatureConfig):
            self.features = deepcopy(features)

    def align_from(self, reference: "DataProcessing") -> None:
        self.import_alignment_state(reference.export_alignment_state())


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
            pl.col("ORIGIN")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lat")
            .alias("Origin_Lat"),

            pl.col("ORIGIN")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lon")
            .alias("Origin_Lon"),

            pl.col("DEST")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lat")
            .alias("Destination_Lat"),

            pl.col("DEST")
            .map_elements(get_lat_lon, return_dtype=geo_dtype)
            .struct.field("lon")
            .alias("Destination_Lon"),
        ])

        # filter out rows where geolocation failed for either origin or destination
        ret = ret.filter(
            pl.col("Origin_Lat").is_not_null() &
            pl.col("Origin_Lon").is_not_null() &
            pl.col("Destination_Lat").is_not_null() &
            pl.col("Destination_Lon").is_not_null()
        )
        
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
        ret =  df.with_columns((c * R).alias("DISTANCE"))

        return ret
        

    def load_shipping_data(self, filepath: Path | None = None) -> pl.DataFrame:
        # Read Raw (as strings) to handle mixed headers
        if filepath is None:
            filepath = self.excel_path

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

        df_final = self._canonicalize_route_columns(df_final)

        # TODO in df final, check origin destination, if either is not in airports data, raise error with list of unknown codes
        origin_codes = set(df_final["ORIGIN"].unique().to_list())
        destination_codes = set(df_final["DEST"].unique().to_list())
        unknown_origin_codes = sorted(origin_codes.difference(self.airports.keys()), key=lambda value: str(value))
        unknown_destination_codes = sorted(destination_codes.difference(self.airports.keys()), key=lambda value: str(value))

        # Warn (rather than raise) when unknown airport codes are present so callers
        # can inspect problematic codes without stopping execution. Rows with these
        # codes will be dropped afterwards.
        if unknown_origin_codes or unknown_destination_codes:
            warnings.warn(
                f"Unknown airport codes found. Unknown Origins: {unknown_origin_codes}, "
                f"Unknown Destinations: {unknown_destination_codes}. Rows with these codes will be dropped.",
                UserWarning,
            )

        # NOTE drop these rows for now
        df_final = df_final.filter(
            ~pl.col("ORIGIN").is_in(unknown_origin_codes) &
            ~pl.col("DEST").is_in(unknown_destination_codes)
        )
        # .rename({
        #     "Origin": "ORIGIN",
        #     "Destination": "DEST",
        #     "Airline": "UNIQUE_CARRIER",
        # })

        return df_final


    def _encode_features(self, df: pl.DataFrame,
                         is_training: bool = True) -> pl.DataFrame:
        target_col = self.features.target_column
        df = self._canonicalize_route_columns(df)
        
        # 1. Encode Target (if it exists in this dataset)
        if target_col in df.columns and self.separate_target:
            if is_training and target_col not in self.mapping:
                uniq_targets = sorted(df[target_col].drop_nulls().unique().to_list())
                self.mapping[target_col] = {val: idx for idx, val in enumerate(uniq_targets)}
            
            target_map = self.mapping[target_col]
            df = df.with_columns(
                pl.Series(target_col, [target_map.get(val, -1) for val in df[target_col]], dtype=pl.Int32)
            )

        # 2. Encode Features
        for col in self.features.all_features:
            if col not in df.columns:
                # If a feature is missing from the dataset, fill with 0
                if col in self.features.categorical_features:
                    df = df.with_columns(pl.lit(0).alias(col))
                else:
                    df = df.with_columns(pl.lit(0.0).alias(col))
                continue

            col_dtype = df[col].dtype
            
            # Is it defined as categorical or is it a String type?
            if col in self.features.categorical_features or col_dtype in (pl.String, pl.Utf8, pl.Categorical):
                # Clean strings and fill nulls
                df = df.with_columns(pl.col(col).cast(pl.String).fill_null("UNKNOWN"))

                if is_training:
                    if col not in self.mapping:
                        self.mapping[col] ={}

                    curr_max_idx = max(self.mapping[col].values(), default=0)

                    uniq_vals = sorted(df[col].unique().to_list(), key=lambda v: str(v))
                    for val in uniq_vals:
                        if val not in self.mapping[col]:
                            curr_max_idx += 1
                            self.mapping[col][val] = curr_max_idx
                # # Build mapping if not already seen and in training mode
                # if is_training and col not in self.mapping:
                #     uniq_vals = sorted(df[col].unique().to_list(), key=lambda v: str(v))
                #     self.mapping[col] = {val: idx for idx, val in enumerate(uniq_vals)}
                
                # Apply integer mapping
                cat_map = self.mapping[col]
                df = df.with_columns(
                    pl.Series(col, [cat_map.get(val, 0) for val in df[col]], dtype=pl.Int32)
                )

            # Otherwise, treat as numerical
            else:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float32, strict=False).fill_null(0.0)
                )

        return df

    
    def _encode_features_old(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError("This method is deprecated. Use 'transform_new_data' for new data preprocessing.")
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
            except (pl.exceptions.PolarsError, TypeError, ValueError):
                # Sort unique values to keep categorical mapping stable across runs.
                uniq_vals = sorted(df[col].unique().to_list(), key=lambda value: str(value))
                self.mapping[col] = {val: idx for idx, val in enumerate(uniq_vals)}
                df = df.with_columns(
                    pl.Series(col, [self.mapping[col][val] for val in df[col]], dtype=pl.Int32)
                )

        return df



    def _normalize_data(self, data_tensor: torch.Tensor, is_training: bool ) -> torch.Tensor:
        if is_training:
            self.feature_mean = data_tensor.mean(dim=0)
            self.feature_std = data_tensor.std(dim=0) + 1e-6
        elif self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Feature normalization statistics are not initialized.")

        # data_tensor = torch.tensor(data_np)
        # mean = data_tensor.mean(dim=0)
        # std = data_tensor.std(dim=0) + 1e-6
        data_tensor = (data_tensor - self.feature_mean) / (self.feature_std)

        return data_tensor

    
    def to_tensor(self, df: pl.DataFrame, is_training: bool = True, simple: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        warnings.warn('Ensure revised implementation to handle split num and cat tensors')
        
        data_ = self._encode_features(df, is_training=is_training)

        cat_cols = [c for c in self.features.categorical_features if c in data_.columns]
        if cat_cols:
            cat_tensor = torch.tensor(data_.select(cat_cols).to_numpy().astype(np.int64), dtype=torch.long)
        else:
            cat_tensor = torch.empty((data_.shape[0], 0), dtype=torch.long)

        num_cols = [c for c in self.features.numerical_features if c in data_.columns]
        num_tensor = torch.empty((data_.shape[0], 0), dtype=torch.float32)
        if num_cols:    
            input_tensor = torch.tensor(data_.select(num_cols).to_numpy().astype(np.float32), dtype=torch.float32)
        # # Normalize data for ease of training
        # # TODO do normalization elsewhere to prevent data leakage and ensure correct handling of new data
            # num_tensor = self._normalize_data(input_tensor, is_training=is_training)
            # NOTE normalization will be handleded in model training pipeline
            num_tensor = input_tensor

        target_tensor = None
        if self.features.target_column in data_.columns and self.separate_target:
            target_array = data_.select(self.features.target_column).to_numpy().astype(np.int64)
            target_tensor = torch.tensor(target_array, dtype=torch.int64)
        # torch.tensor(data_.select(self.features.target_column).to_numpy().astype(np.int64))
        # breakpoint()
        if simple:
            # combine cat and num
            data_tensor = torch.cat([cat_tensor, num_tensor], dim=1)
            return data_tensor, target_tensor

        return cat_tensor, num_tensor, target_tensor
    

    def transform_new_data(self, df: pl.DataFrame) -> torch.Tensor:
        df = self._canonicalize_route_columns(df)
        df_geo = self._geolocate_nodes(df)
        df_dist = self._calculate_distance(df_geo)

        data_tensor, _ = self.to_tensor(df_dist, is_training=False, simple=True)
        return data_tensor


    def process_data(self, filepath: Path = Path()) -> tuple[torch.Tensor, torch.Tensor]:
        if filepath == Path():
            filepath = self.excel_path

        df_shipping = self.load_shipping_data(filepath)
        df_geo = self._geolocate_nodes(df_shipping)
        df_dist = self._calculate_distance(df_geo)
        self.cat_tensor, self.num_tensor, self.target_tensor = self.to_tensor(df_dist)

        return self.cat_tensor, self.num_tensor, self.target_tensor
    

# %%
class T100DataProcessing(DataProcessing):
    def __init__(self):
        # t100_feature_config = FeatureConfig(
        #     target_column="SLACK",
        #     categorical_features=[
        #         "UNIQUE_CARRIER",
        #         "ORIGIN",
        #         "DEST"
        #     ],
        #     numerical_features=[
        #         "DISTANCE"]
        # )

        super().__init__(separate_target=False)
        self.excel_path = Path(self.data_dir, "data/T_T100I_SEGMENT_ALL_CARRIER.csv")
        self.t100_data = pl.read_csv(self.excel_path)

        self.rng = np.random.default_rng(seed=42)
        
        # Passenger-level moments (kg)
        self.mu_pax = 166.7 + 16.8 + 35.1
        self.sigma_pax = np.sqrt(36.8**2 + 11.0**2 + 12.3**2)

    def _truncated_slack_samples(self, mu: np.ndarray, sigma: np.ndarray, n_draws: int = 1) -> np.ndarray:
        """Generates samples from a truncated normal distribution bounded at 0."""
        out = np.zeros((mu.size, n_draws))
        positive_sigma = sigma > 0

        # Standardized truncation bounds (a = lower bound, b = upper bound)
        a = (0 - mu[positive_sigma]) / sigma[positive_sigma]
        b = np.full_like(a, np.inf)

        out[positive_sigma] = truncnorm.rvs(
            a[:, None],
            b[:, None],
            loc=mu[positive_sigma][:, None],
            scale=sigma[positive_sigma][:, None],
            size=(positive_sigma.sum(), n_draws),
            random_state=self.rng,
        )

        # Deterministic rows (where sigma is 0, e.g., cargo-only flights)
        out[~positive_sigma] = np.maximum(mu[~positive_sigma], 0.0)[:, None]
        return out

    def filter_data(self, df: pl.DataFrame = None) -> pl.DataFrame:
        if df is None:
            df = self.t100_data

        df = self._canonicalize_route_columns(df)

        df_clean = (df.filter(
            (pl.col("DEPARTURES_PERFORMED") > 0) &
            (pl.col("CLASS").is_in(["A", "C", "E", "F"]))
            ).sort(by="FREIGHT", descending=True)
        )

        # df_clean = df_clean.rename({
        #     "ORIGIN": "Origin",
        #     "DEST": "Destination",
        # })

        df_normalized = df_clean.with_columns([
            (pl.col("PAYLOAD") / pl.col("DEPARTURES_PERFORMED")).alias("PAYLOAD_PER_FLIGHT"),
            (pl.col("FREIGHT") / pl.col("DEPARTURES_PERFORMED")).alias("FREIGHT_PER_FLIGHT"),
            (pl.col("MAIL") / pl.col("DEPARTURES_PERFORMED")).alias("MAIL_PER_FLIGHT"),
            (pl.col("PASSENGERS") / pl.col("DEPARTURES_PERFORMED")).alias("PASSENGERS_PER_FLIGHT")
        ])

        # NOTE currently sweeps and eliminates routes with 0 departures whatsoever, 
        # while assumed to be fine for now, may need to shift to a mask instead to 
        # maintain dataset integrity

        u_max = df_normalized["PAYLOAD_PER_FLIGHT"].cast(pl.Float32).to_numpy()
        u_cargo = df_normalized["FREIGHT_PER_FLIGHT"].cast(pl.Float32).to_numpy()
        u_mail = df_normalized["MAIL_PER_FLIGHT"].cast(pl.Float32).to_numpy()
        u_pax = df_normalized["PASSENGERS_PER_FLIGHT"].cast(pl.Float32).to_numpy()

        # Calcuate the distributions
        mu_z = u_max - (u_cargo + u_mail + (self.mu_pax * u_pax))
        sigma_z = np.sqrt(np.maximum(u_pax, 0.0)) * self.sigma_pax

        # Generate truncated slack samples
        slack_samples = self._truncated_slack_samples(mu_z, sigma_z, n_draws=100)
        expected_slack = np.mean(slack_samples, axis=1)

        df_final = df_normalized.with_columns([
            (pl.Series("SLACK_ARRAY", slack_samples)),
            (pl.Series("SLACK", expected_slack, dtype=pl.Float32))
        ])

        return df_final


# %%
if __name__ == "__main__":
    # dp = DataProcessing()
    # df = dp.load_shipping_data(dp.excel_path)
    # df_geo = dp._geolocate_nodes(df)
    # df_dist = dp._calculate_distance(df_geo)
    # # TODO integrate processing
    # # data_tensor = dp.preprocess_to_tensor(df_dist)
    # # data_tensor, target_tensor = dp.process_data()
    # # print(data_tensor.shape)
    # # sys.exit(0)
    t100 = T100DataProcessing()
    df = t100.filter_data()
    t100geo = t100._geolocate_nodes(df)
    t100dist = t100._calculate_distance(t100geo)
    tensor = t100.to_tensor(t100dist, is_training=True, simple=True)

    # make sure encoding aligns
    ship = DataProcessing()
    ship.align_from(t100)
    df_ship = ship.load_shipping_data()
    ship_geo = ship._geolocate_nodes(df_ship)
    ship_dist = ship._calculate_distance(ship_geo)
    ship_tensor = ship.to_tensor(ship_dist, is_training=False, simple=True)

# %%
