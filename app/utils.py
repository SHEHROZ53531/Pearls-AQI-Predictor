import os
import io
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "aqi_forecast"
FEATURES_COLLECTION = "features"
MODELS_META_COLLECTION = "model_metadata"

CITIES = ["islamabad", "rawalpindi", "lahore", "faisalabad"]
CITY_DISPLAY_NAMES = {
    "islamabad": "Islamabad",
    "rawalpindi": "Rawalpindi",
    "lahore": "Lahore",
    "faisalabad": "Faisalabad",
}

HORIZON_BUCKETS = {
    "short": (1, 24),
    "medium": (25, 48),
    "long": (49, 72),
}

BASE_MODEL_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi_change_rate", "day_of_month",
    "aqi_lag_6h", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_168h",
    "aqi_rolling_mean_24h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
]
CITY_DUMMY_COLUMNS = [f"city_{c}" for c in CITIES]
MODEL_FEATURE_COLUMNS = BASE_MODEL_COLUMNS + CITY_DUMMY_COLUMNS

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def classify_aqi(value: float) -> tuple:
    for low, high, name, color in AQI_CATEGORIES:
        if low <= value <= high:
            return name, color
    return "Hazardous", "#7e0023"


def connect_to_mongo():
    uri = os.environ["MONGODB_URI"]
    client = MongoClient(uri)
    return client[DB_NAME]


def load_full_feature_table(db) -> pd.DataFrame:
    """Read the ENTIRE features collection once. Cache and reuse across city switches."""
    collection = db[FEATURES_COLLECTION]
    records = list(collection.find({}))
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def engineer_features_for_city(full_df: pd.DataFrame, city: str, hours_needed: int = 2000) -> pd.DataFrame:
    """Slice one city out of the already-loaded table and build its lag/cyclical/one-hot features."""
    df = full_df[full_df["city"] == city].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.tail(hours_needed + 24).copy()

    df = df.set_index("timestamp")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "timestamp"
    df["city"] = city

    raw_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "pressure", "wind_speed",
        "hour", "day_of_week", "day_of_month", "month", "aqi_change_rate",
    ]
    df[raw_cols] = df[raw_cols].ffill().bfill()

    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_168h"] = df["aqi"].shift(168)
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    for c in CITIES:
        df[f"city_{c}"] = 1.0 if c == city else 0.0

    engineered_cols = ["aqi_lag_6h", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_168h", "aqi_rolling_mean_24h"]
    df[engineered_cols] = df[engineered_cols].ffill().bfill()

    return df


def load_bucket_model(db, bucket_name: str) -> dict:
    """Load one bucket's model + scaler from GridFS, using the metadata pointer document."""
    metadata_collection = db[MODELS_META_COLLECTION]
    meta = metadata_collection.find_one({"bucket_name": bucket_name})
    if meta is None:
        raise ValueError(f"No trained model found for bucket '{bucket_name}' yet.")

    fs = gridfs.GridFS(db)
    model_bytes = fs.get(meta["model_file_id"]).read()
    scaler_bytes = fs.get(meta["scaler_file_id"]).read()

    scaler = joblib.load(io.BytesIO(scaler_bytes))

    if meta["model_kind"] == "sklearn":
        model = joblib.load(io.BytesIO(model_bytes))
    else:
        
        from tensorflow import keras
        tmp_path = f"/tmp/{bucket_name}_nn_model.keras"
        with open(tmp_path, "wb") as f:
            f.write(model_bytes)
        model = keras.models.load_model(tmp_path)
        os.remove(tmp_path)

    return {
        "model": model,
        "scaler": scaler,
        "type": meta["model_kind"],
        "metrics": meta["metrics"],
        "trained_at": meta["trained_at"],
    }


def load_all_bucket_models(db) -> dict:
    bundles = {}
    for bucket_name in HORIZON_BUCKETS:
        bundles[bucket_name] = load_bucket_model(db, bucket_name)
    return bundles


def bucket_for_horizon(horizon: int) -> str:
    for name, (low, high) in HORIZON_BUCKETS.items():
        if low <= horizon <= high:
            return name
    return "long"


def predict_next_72_hours(history_df: pd.DataFrame, model_bundles: dict) -> pd.DataFrame:
    filled_df = history_df[MODEL_FEATURE_COLUMNS].ffill().bfill()
    latest_timestamp = history_df.index.max()
    latest_row = filled_df.loc[latest_timestamp]

    if latest_row.isna().any():
        missing_cols = latest_row[latest_row.isna()].index.tolist()
        raise ValueError(
            f"Cannot build a forecast: still missing values for {missing_cols} "
            f"even after filling. Need more history collected first."
        )

    predictions = []
    for horizon in range(1, 73):
        bucket_name = bucket_for_horizon(horizon)
        bundle = model_bundles[bucket_name]

        feature_values = latest_row[MODEL_FEATURE_COLUMNS].tolist() + [horizon]
        X = pd.DataFrame([feature_values], columns=MODEL_FEATURE_COLUMNS + ["horizon"])
        X_scaled = bundle["scaler"].transform(X)

        if bundle["type"] == "sklearn":
            pred = bundle["model"].predict(X_scaled)[0]
        else:
            pred = bundle["model"].predict(X_scaled, verbose=0).flatten()[0]

        pred = max(0.0, float(pred))
        forecast_time = latest_timestamp + pd.Timedelta(hours=horizon)
        predictions.append({"timestamp": forecast_time, "horizon": horizon, "predicted_aqi": pred})

    return pd.DataFrame(predictions)