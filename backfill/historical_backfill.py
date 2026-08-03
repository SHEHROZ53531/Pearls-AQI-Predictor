"""
Historical backfill: pulls past AQI + weather data for Islamabad from
Open-Meteo's free archive API and stores it in the same Hopsworks
feature group used by the live hourly pipeline.

Run this once (or a few times) to build up enough history to train on.
AQICN doesn't offer free historical data, so we use Open-Meteo here instead --
same feature columns, different data source, just for building training data.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import hopsworks

CITY = "islamabad"
LATITUDE = 33.6844
LONGITUDE = 73.0479

HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT = os.environ["HOPSWORKS_PROJECT"]

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

NUMERIC_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def fetch_air_quality_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Get hourly pollutant + AQI history from Open-Meteo's air quality archive."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,us_aqi",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.DataFrame(hourly)


def fetch_weather_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Get hourly temperature/humidity/pressure/wind from Open-Meteo's weather archive."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.DataFrame(hourly)


def build_feature_rows(start_date: str, end_date: str) -> pd.DataFrame:
    """Combine air quality + weather history into one row per hour."""
    aq = fetch_air_quality_history(start_date, end_date)
    weather = fetch_weather_history(start_date, end_date)

    # both come back with a "time" column of matching hourly timestamps
    merged = pd.merge(aq, weather, on="time", how="inner")

    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(merged["time"], utc=True)
    df["city"] = CITY
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

    # drop rows where AQI itself is missing -- can't train on those anyway
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
    """
    Same idea as the live pipeline: how much AQI moved since the previous hour.
    Here we have the full history at once, so we just use pandas' diff()
    instead of querying Hopsworks row by row.
    """
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0.0)
    return df


def main():
    # last 365 days -- lead's guidance: aim for 6 months minimum, 1-2 years ideal
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365)

    print(f"Backfilling {CITY} data from {start_date} to {end_date}...")

    df = build_feature_rows(str(start_date), str(end_date))
    df = add_time_features(df)
    df = add_change_rate(df)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    print(f"Built {len(df)} hourly rows. Uploading to Hopsworks...")

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI readings for Islamabad with engineered features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    feature_group.insert(df)
    print(f"Done. Inserted {len(df)} historical rows.")


if __name__ == "__main__":
    main()
