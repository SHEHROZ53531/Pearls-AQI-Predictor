"""
Feature pipeline: fetches current AQI + weather data for multiple
Pakistani cities from Open-Meteo, engineers features, and writes them
to the Hopsworks Feature Store.

We use Open-Meteo (not AQICN) for the live pipeline because AQICN's
real-time station network has no coverage in some of our cities
(Rawalpindi, Faisalabad) -- Open-Meteo's model-based data covers all
of them consistently, and it's the same source we already use for
historical backfill.

Runs every hour via GitHub Actions.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
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


def current_hour_index(time_list: list, now: datetime) -> int:
    """Find which row in Open-Meteo's hourly list matches the current hour."""
    target = now.strftime("%Y-%m-%dT%H:00")
    if target in time_list:
        return time_list.index(target)
    return len(time_list) - 1  # fallback: closest available if exact hour missing


def fetch_current_conditions(lat: float, lon: float) -> dict:
    """Get this city's current-hour AQI, pollutants, and weather from Open-Meteo."""
    now = datetime.now(timezone.utc)

    aq_response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,us_aqi",
            "timezone": "UTC",
        },
        timeout=30,
    )
    aq_response.raise_for_status()
    aq_hourly = aq_response.json()["hourly"]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
            "timezone": "UTC",
        },
        timeout=30,
    )
    weather_response.raise_for_status()
    weather_hourly = weather_response.json()["hourly"]

    aq_idx = current_hour_index(aq_hourly["time"], now)
    weather_idx = current_hour_index(weather_hourly["time"], now)

    return {
        "timestamp": now,
        "aqi": aq_hourly["us_aqi"][aq_idx],
        "pm25": aq_hourly["pm2_5"][aq_idx],
        "pm10": aq_hourly["pm10"][aq_idx],
        "o3": aq_hourly["ozone"][aq_idx],
        "no2": aq_hourly["nitrogen_dioxide"][aq_idx],
        "so2": aq_hourly["sulphur_dioxide"][aq_idx],
        "co": aq_hourly["carbon_monoxide"][aq_idx],
        "temperature": weather_hourly["temperature_2m"][weather_idx],
        "humidity": weather_hourly["relative_humidity_2m"][weather_idx],
        "pressure": weather_hourly["surface_pressure"][weather_idx],
        "wind_speed": weather_hourly["wind_speed_10m"][weather_idx],
    }


def add_time_features(row: dict) -> dict:
    ts = row["timestamp"]
    row["hour"] = ts.hour
    row["day_of_week"] = ts.weekday()
    row["day_of_month"] = ts.day
    row["month"] = ts.month
    return row


def add_change_rate(row: dict, city_name: str, feature_store) -> dict:
    """Compare this reading to the most recent stored reading for the SAME city."""
    try:
        fg = feature_store.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        recent_df = fg.read()
        recent_df = recent_df[recent_df["city"] == city_name]
        if len(recent_df) > 0:
            recent_df = recent_df.sort_values("timestamp")
            last_aqi = recent_df.iloc[-1]["aqi"]
            row["aqi_change_rate"] = row["aqi"] - last_aqi
        else:
            row["aqi_change_rate"] = 0.0
    except Exception:
        row["aqi_change_rate"] = 0.0
    return row


def build_row_for_city(city_name: str, lat: float, lon: float, feature_store) -> dict:
    row = fetch_current_conditions(lat, lon)
    row["city"] = city_name
    row = add_time_features(row)
    row = add_change_rate(row, city_name, feature_store)
    return row


def main():
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    rows = []
    for city_name, coords in CITIES.items():
        try:
            row = build_row_for_city(city_name, coords["lat"], coords["lon"], fs)
            rows.append(row)
            print(f"Fetched {city_name}: AQI={row['aqi']}")
        except Exception as e:
            print(f"Skipped {city_name} due to error: {e}")

    if not rows:
        print("No cities fetched successfully this run.")
        return

    df = pd.DataFrame(rows)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI readings for multiple Pakistani cities with engineered features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    feature_group.insert(df)
    print(f"Inserted {len(df)} rows for {list(df['city'])}")


if __name__ == "__main__":
    main()