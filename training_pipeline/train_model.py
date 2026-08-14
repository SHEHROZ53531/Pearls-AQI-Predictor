"""
Training pipeline: pulls historical AQI features for multiple cities
from MongoDB Atlas, adds lagged/rolling AQI features and cyclical time
encodings (computed separately per city), one-hot encodes the city
itself, then trains SEPARATE models for three forecast-horizon
buckets (short/medium/long) shared across all cities.

For each bucket, four model types compete (Ridge, Random Forest,
Gradient Boosting, a small regularized neural network) and the best
one is saved to MongoDB via GridFS, replacing Hopsworks' Model Registry.

Runs once a day via GitHub Actions.
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = "aqi_forecast"
FEATURES_COLLECTION = "features"
MODELS_META_COLLECTION = "model_metadata"

CITIES = ["islamabad", "rawalpindi", "lahore", "faisalabad"]

HORIZON_BUCKETS = {
    "short": (1, 24),
    "medium": (25, 48),
    "long": (49, 72),
}

RAW_STORE_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "hour", "day_of_week", "day_of_month", "month", "aqi_change_rate",
]

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


def get_db():
    client = MongoClient(MONGODB_URI)
    return client[DB_NAME]


def load_feature_data(db) -> pd.DataFrame:
    collection = db[FEATURES_COLLECTION]
    records = list(collection.find({}))
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    return df


def process_one_city(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.set_index("timestamp")
    full_range = pd.date_range(city_df.index.min(), city_df.index.max(), freq="h")
    city_df = city_df.reindex(full_range)
    city_df.index.name = "timestamp"

    city_df["city"] = city_df["city"].ffill()
    city_df[RAW_STORE_COLUMNS] = city_df[RAW_STORE_COLUMNS].ffill(limit=3)

    city_df["aqi_lag_6h"] = city_df["aqi"].shift(6)
    city_df["aqi_lag_24h"] = city_df["aqi"].shift(24)
    city_df["aqi_lag_48h"] = city_df["aqi"].shift(48)
    city_df["aqi_lag_168h"] = city_df["aqi"].shift(168)
    city_df["aqi_rolling_mean_24h"] = city_df["aqi"].rolling(window=24, min_periods=6).mean()

    city_df["hour_sin"] = np.sin(2 * np.pi * city_df["hour"] / 24)
    city_df["hour_cos"] = np.cos(2 * np.pi * city_df["hour"] / 24)
    city_df["month_sin"] = np.sin(2 * np.pi * city_df["month"] / 12)
    city_df["month_cos"] = np.cos(2 * np.pi * city_df["month"] / 12)
    city_df["dow_sin"] = np.sin(2 * np.pi * city_df["day_of_week"] / 7)
    city_df["dow_cos"] = np.cos(2 * np.pi * city_df["day_of_week"] / 7)

    return city_df


def add_city_dummies(df: pd.DataFrame) -> pd.DataFrame:
    for city in CITIES:
        df[f"city_{city}"] = (df["city"] == city).astype("float64")
    return df


def build_dataset_for_bucket(all_cities_df: pd.DataFrame, min_h: int, max_h: int) -> pd.DataFrame:
    rows = []
    for city in CITIES:
        city_df = all_cities_df[all_cities_df["city"] == city]
        aqi_series = city_df["aqi"]

        for horizon in range(min_h, max_h + 1):
            future_aqi = aqi_series.shift(-horizon)
            chunk = city_df[MODEL_FEATURE_COLUMNS].copy()
            chunk["horizon"] = horizon
            chunk["target_aqi"] = future_aqi.values
            rows.append(chunk)

    combined = pd.concat(rows, ignore_index=False)
    combined = combined.dropna()
    return combined


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    df = df.sort_index()
    cutoff_index = int(len(df) * (1 - test_fraction))
    cutoff_time = df.index.sort_values()[cutoff_index]
    train = df[df.index <= cutoff_time]
    test = df[df.index > cutoff_time]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_pick_best(X_train, y_train, X_test, y_test, bucket_name: str):
    results = {}
    is_long_bucket = bucket_name == "long"

    print(f"  [{bucket_name}] training ridge...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    results["ridge"] = {"model": ridge, "metrics": evaluate(y_test, preds)}

    print(f"  [{bucket_name}] training random_forest...")
    rf = RandomForestRegressor(
        n_estimators=400 if is_long_bucket else 200,
        max_depth=20 if is_long_bucket else 14,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    results["random_forest"] = {"model": rf, "metrics": evaluate(y_test, preds)}

    print(f"  [{bucket_name}] training gradient_boosting...")
    gbdt = GradientBoostingRegressor(
        n_estimators=350 if is_long_bucket else 150,
        max_depth=6 if is_long_bucket else 4,
        learning_rate=0.05 if is_long_bucket else 0.1,
        random_state=42,
    )
    gbdt.fit(X_train, y_train)
    preds = gbdt.predict(X_test)
    results["gradient_boosting"] = {"model": gbdt, "metrics": evaluate(y_test, preds)}

    # neural network is optional -- only imported/trained if TensorFlow is
    # available, keeps this script lighter and avoids native library
    # conflicts seen elsewhere in this project
    try:
        from tensorflow import keras
        print(f"  [{bucket_name}] training neural_network...")
        nn = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ])
        nn.compile(optimizer="adam", loss="mse")
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        nn.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=512, callbacks=[early_stop], verbose=0)
        preds = nn.predict(X_test, verbose=0).flatten()
        results["neural_network"] = {"model": nn, "metrics": evaluate(y_test, preds)}
    except Exception as e:
        print(f"  [{bucket_name}] skipping neural_network: {e}")

    for name, r in results.items():
        m = r["metrics"]
        print(f"  [{bucket_name}] {name}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["metrics"]["rmse"])
    print(f"  [{bucket_name}] BEST: {best_name} (RMSE={results[best_name]['metrics']['rmse']:.2f})")
    return best_name, results[best_name]


def save_model_to_mongo(db, bucket_name: str, model, model_kind: str, scaler, metrics: dict):
    """
    Save a trained model + its scaler to GridFS (for the binary files),
    and its metadata (metrics, which model type won, timestamp) to a
    regular collection -- this replaces Hopsworks' Model Registry.
    """
    fs = gridfs.GridFS(db)

    # serialize model and scaler to bytes in memory, then store in GridFS
    model_buffer = io.BytesIO()
    if model_kind == "sklearn":
        joblib.dump(model, model_buffer)
    else:
        # keras models need a real file path to save, so use a temp file
        tmp_path = f"/tmp/{bucket_name}_nn_model.keras"
        model.save(tmp_path)
        with open(tmp_path, "rb") as f:
            model_buffer.write(f.read())
        os.remove(tmp_path)
    model_buffer.seek(0)

    scaler_buffer = io.BytesIO()
    joblib.dump(scaler, scaler_buffer)
    scaler_buffer.seek(0)

    # remove any old files for this bucket so GridFS doesn't accumulate versions forever
    for old_file in fs.find({"bucket_name": bucket_name}):
        fs.delete(old_file._id)

    model_file_id = fs.put(model_buffer.read(), filename=f"{bucket_name}_model", bucket_name=bucket_name)
    scaler_file_id = fs.put(scaler_buffer.read(), filename=f"{bucket_name}_scaler", bucket_name=bucket_name)

    metadata_collection = db[MODELS_META_COLLECTION]
    metadata_collection.replace_one(
        {"bucket_name": bucket_name},
        {
            "bucket_name": bucket_name,
            "model_kind": model_kind,
            "model_file_id": model_file_id,
            "scaler_file_id": scaler_file_id,
            "metrics": metrics,
            "trained_at": datetime.now(timezone.utc),
        },
        upsert=True,
    )


def main():
    db = get_db()

    print("Loading feature data for all cities...")
    raw_df = load_feature_data(db)
    print(f"Loaded {len(raw_df)} raw rows across {raw_df['city'].nunique()} cities.")

    print("Processing each city's timeline (hourly grid + lag/cyclical features)...")
    processed = []
    for city in CITIES:
        city_slice = raw_df[raw_df["city"] == city]
        if len(city_slice) == 0:
            print(f"  Skipping {city}: no data found.")
            continue
        processed.append(process_one_city(city_slice))
    all_cities_df = pd.concat(processed)
    all_cities_df = add_city_dummies(all_cities_df)

    feature_cols = MODEL_FEATURE_COLUMNS + ["horizon"]

    for bucket_name, (min_h, max_h) in HORIZON_BUCKETS.items():
        print(f"\n=== Bucket: {bucket_name} (horizons {min_h}-{max_h}) ===")

        dataset = build_dataset_for_bucket(all_cities_df, min_h, max_h)
        print(f"Built {len(dataset)} training examples across all cities.")

        train_df, test_df = time_based_split(dataset)
        X_train_raw, y_train = train_df[feature_cols], train_df["target_aqi"]
        X_test_raw, y_test = test_df[feature_cols], test_df["target_aqi"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        best_name, best = train_and_pick_best(X_train, y_train, X_test, y_test, bucket_name)
        model_kind = "sklearn" if best_name != "neural_network" else "tensorflow"

        save_model_to_mongo(db, bucket_name, best["model"], model_kind, scaler, best["metrics"])
        print(f"Saved '{bucket_name}' model ({best_name}) to MongoDB.")

    print("\nAll 3 bucket models trained and saved (multi-city).")


if __name__ == "__main__":
    main()