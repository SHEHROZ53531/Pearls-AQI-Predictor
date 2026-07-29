"""
Feature pipeline: fetches live AQI data for Islamabad from AQICN,
engineers features, and writes them to the Hopsworks Feature Store.

Runs every hour via GitHub Actions.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
import hopsworks

CITY = "islamabad"
AQICN_TOKEN = os.environ["AQICN_TOKEN"]
HOPSWORKS_API_KEY = os.environ["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT = os.environ["HOPSWORKS_PROJECT"]

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

NUMERIC_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed",
    "aqi_change_rate",
]


def fetch_current_aqi(city: str) -> dict:
    """Hit the AQICN API and pull out the fields we care about."""
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if payload["status"] != "ok":
        raise RuntimeError(f"AQICN returned an error: {payload}")

    data = payload["data"]
    iaqi = data.get("iaqi", {})  # individual pollutant readings

    record = {
        "timestamp": datetime.now(timezone.utc),
        "city": city,
        "aqi": data.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "temperature": iaqi.get("t", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
        "pressure": iaqi.get("p", {}).get("v"),
        "wind_speed": iaqi.get("w", {}).get("v"),
    }
    return record


def add_time_features(row: dict) -> dict:
    """Break the timestamp into features the model can actually learn from."""
    ts = row["timestamp"]
    row["hour"] = ts.hour
    row["day_of_week"] = ts.weekday()   # 0 = Monday
    row["day_of_month"] = ts.day
    row["month"] = ts.month
    return row


def add_change_rate(row: dict, feature_store) -> dict:
    """
    Compare this AQI reading to the most recent one already stored,
    so the model can see whether AQI is rising or falling, not just its level.
    """
    try:
        fg = feature_store.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        recent_df = fg.read()
        if len(recent_df) > 0:
            recent_df = recent_df.sort_values("timestamp")
            last_aqi = recent_df.iloc[-1]["aqi"]
            row["aqi_change_rate"] = row["aqi"] - last_aqi
        else:
            row["aqi_change_rate"] = 0.0
    except Exception:
        # first ever run, feature group might not have data yet
        row["aqi_change_rate"] = 0.0
    return row


def main():
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()

    row = fetch_current_aqi(CITY)
    row = add_time_features(row)
    row = add_change_rate(row, fs)

    df = pd.DataFrame([row])

    # force every pollutant/weather column to a real float so a station's
    # missing reading (None) becomes NaN instead of an ambiguous null type
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI readings for Islamabad with engineered features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
    )
    feature_group.insert(df)
    print(f"Inserted row for {row['timestamp']} — AQI={row['aqi']}")


if __name__ == "__main__":
    main()