"""
Training pipeline: pulls historical AQI features from Hopsworks, adds
lagged/rolling AQI features and cyclical time encodings, then trains
SEPARATE models for three forecast-horizon buckets (short/medium/long)
since near-term and far-term AQI prediction are very different problems.

For each bucket, four model types compete (Ridge, Random Forest,
Gradient Boosting, a small regularized neural network) and the best
one is registered in the Hopsworks Model Registry.

Runs once a day via GitHub Actions.
"""

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT = os.environ["HOPSWORKS_PROJECT"]

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

# instead of one model for all 72 hours, we split into 3 difficulty tiers
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

MODEL_FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi_change_rate", "day_of_month",
    "aqi_lag_6h", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_168h",
    "aqi_rolling_mean_24h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
]


def load_feature_data(fs) -> pd.DataFrame:
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def make_hourly_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "timestamp"
    df[RAW_STORE_COLUMNS] = df[RAW_STORE_COLUMNS].ffill(limit=3)
    return df


def add_lag_and_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_168h"] = df["aqi"].shift(168)  # same hour, one week ago
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=6).mean()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def build_dataset_for_bucket(df: pd.DataFrame, min_h: int, max_h: int) -> pd.DataFrame:
    """Same multi-horizon trick as before, but only for this bucket's horizon range."""
    rows = []
    aqi_series = df["aqi"]

    for horizon in range(min_h, max_h + 1):
        future_aqi = aqi_series.shift(-horizon)
        chunk = df[MODEL_FEATURE_COLUMNS].copy()
        chunk["horizon"] = horizon
        chunk["target_aqi"] = future_aqi.values
        rows.append(chunk)

    combined = pd.concat(rows, ignore_index=False)
    combined = combined.dropna()
    return combined


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    df = df.sort_index()
    cutoff_index = int(len(df) * (1 - test_fraction))
    cutoff_time = df.index[cutoff_index]
    train = df[df.index <= cutoff_time]
    test = df[df.index > cutoff_time]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_neural_network(input_dim: int) -> keras.Model:
    """Smaller + dropout this time, to stop it from memorizing the training set."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_and_pick_best(X_train, y_train, X_test, y_test, bucket_name: str):
    """Train all 4 candidates for one horizon bucket, return the winner."""
    results = {}

    print(f"  [{bucket_name}] training ridge...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    results["ridge"] = {"model": ridge, "metrics": evaluate(y_test, preds), "type": "sklearn"}

    print(f"  [{bucket_name}] training random_forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    results["random_forest"] = {"model": rf, "metrics": evaluate(y_test, preds), "type": "sklearn"}

    print(f"  [{bucket_name}] training gradient_boosting...")
    gbdt = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
    )
    gbdt.fit(X_train, y_train)
    preds = gbdt.predict(X_test)
    results["gradient_boosting"] = {"model": gbdt, "metrics": evaluate(y_test, preds), "type": "sklearn"}

    print(f"  [{bucket_name}] training neural_network...")
    nn = build_neural_network(input_dim=X_train.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    nn.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=512,
        callbacks=[early_stop],
        verbose=0,
    )
    preds = nn.predict(X_test, verbose=0).flatten()
    results["neural_network"] = {"model": nn, "metrics": evaluate(y_test, preds), "type": "tensorflow"}

    for name, r in results.items():
        m = r["metrics"]
        print(f"  [{bucket_name}] {name}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")

    best_name = min(results, key=lambda n: results[n]["metrics"]["rmse"])
    print(f"  [{bucket_name}] BEST: {best_name} (RMSE={results[best_name]['metrics']['rmse']:.2f})")
    return best_name, results[best_name]


def main():
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    print("Loading feature data...")
    raw_df = load_feature_data(fs)
    print(f"Loaded {len(raw_df)} raw hourly rows.")

    hourly_df = make_hourly_and_clean(raw_df)
    hourly_df = add_lag_and_cyclical_features(hourly_df)

    mr = project.get_model_registry()
    feature_cols = MODEL_FEATURE_COLUMNS + ["horizon"]

    for bucket_name, (min_h, max_h) in HORIZON_BUCKETS.items():
        print(f"\n=== Bucket: {bucket_name} (horizons {min_h}-{max_h}) ===")

        dataset = build_dataset_for_bucket(hourly_df, min_h, max_h)
        print(f"Built {len(dataset)} training examples.")

        train_df, test_df = time_based_split(dataset)
        X_train_raw, y_train = train_df[feature_cols], train_df["target_aqi"]
        X_test_raw, y_test = test_df[feature_cols], test_df["target_aqi"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        best_name, best = train_and_pick_best(X_train, y_train, X_test, y_test, bucket_name)

        model_output_dir = f"model_output_{bucket_name}"
        os.makedirs(model_output_dir, exist_ok=True)
        joblib.dump(scaler, f"{model_output_dir}/scaler.pkl")

        registry_name = f"aqi_forecast_model_{bucket_name}"
        description = (
            f"AQI forecast model for Islamabad, horizons {min_h}-{max_h}h "
            f"({best_name}), with lag+cyclical features"
        )

        if best["type"] == "sklearn":
            joblib.dump(best["model"], f"{model_output_dir}/model.pkl")
            registered_model = mr.sklearn.create_model(
                name=registry_name,
                metrics=best["metrics"],
                description=description,
                input_example=X_train_raw[:1],
            )
        else:
            best["model"].save(f"{model_output_dir}/nn_model.keras")
            registered_model = mr.tensorflow.create_model(
                name=registry_name,
                metrics=best["metrics"],
                description=description,
                input_example=X_train_raw[:1],
            )

        registered_model.save(model_output_dir)
        print(f"Saved '{registry_name}' ({best_name}) to Model Registry.")

    print("\nAll 3 bucket models trained and registered.")


if __name__ == "__main__":
    main()