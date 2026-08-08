"""
Historical backfill: pulls past AQI + weather data for multiple
Pakistani cities from Open-Meteo's free archive API and stores it in
the same Hopsworks feature group used by the live hourly pipeline.

Run this once (or a few times) to build up enough history to train on.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import hopsworks

CITIES = {
    "islamabad": {"lat": 33.6844, "lon": 73.0479},
    "rawalpindi": {"lat": 33.5651, "lon": 73.0169},
    "lahore": {"lat": 31.5497, "lon": 74.3436},
    "faisalabad": {"lat": 31.4504, "lon": 73.1350},
}

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT = os.environ["HOPSWORKS_PROJECT"]

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

NUMERIC_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def fetch_air_quality_history(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])


def fetch_weather_history(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])


def build_feature_rows_for_city(city_name: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    aq = fetch_air_quality_history(lat, lon, start_date, end_date)
    weather = fetch_weather_history(lat, lon, start_date, end_date)
    merged = pd.merge(aq, weather, on="time", how="inner")

    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(merged["time"], utc=True)
    df["city"] = city_name
    df["aqi"] = merged["us_aqi"]
    df["pm25"] = merged["pm2_5"]
    df["pm10"] = merged["pm10"]
    df["o3"] = merged["ozone"]
    df["no2"] = merged["nitrogen_dioxide"]
    df["so2"] = merged["sulphur_dioxide"]
    df["co"] = merged["carbon_monoxide"]
    df["temperature"] = merged["temperature_2m"]
    df["humidity"] = merged["relative_humidity_2m"]
    df["pressure"] = merged["surface_pressure"]
    df["wind_speed"] = merged["wind_speed_10m"]

    df = df.dropna(subset=["aqi"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour.astype("int64")
    df["day_of_week"] = df["timestamp"].dt.weekday.astype("int64")
    df["day_of_month"] = df["timestamp"].dt.day.astype("int64")
    df["month"] = df["timestamp"].dt.month.astype("int64")
    return df


def add_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Computed within one city's own timeline -- caller must pass one city at a time."""
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0.0)
    return df


def main():
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365)

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI readings for multiple Pakistani cities with engineered features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )

    for city_name, coords in CITIES.items():
        print(f"\nBackfilling {city_name} from {start_date} to {end_date}...")
        df = build_feature_rows_for_city(city_name, coords["lat"], coords["lon"], str(start_date), str(end_date))
        df = add_time_features(df)
        df = add_change_rate(df)

        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        print(f"Built {len(df)} hourly rows for {city_name}. Uploading...")
        feature_group.insert(df)
        print(f"Done with {city_name}: inserted {len(df)} rows.")

    print("\nAll cities backfilled.")


if __name__ == "__main__":
    main()