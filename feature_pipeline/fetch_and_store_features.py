"""
Feature pipeline: fetches current AQI + weather data for multiple
Pakistani cities from Open-Meteo, engineers features, and writes them
to MongoDB Atlas.

Runs every hour via GitHub Actions.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

CITIES = {
    "islamabad": {"lat": 33.6844, "lon": 73.0479},
    "rawalpindi": {"lat": 33.5651, "lon": 73.0169},
    "lahore": {"lat": 31.5497, "lon": 74.3436},
    "faisalabad": {"lat": 31.4504, "lon": 73.1350},
}

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = "aqi_forecast"
FEATURES_COLLECTION = "features"


def get_features_collection():
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    return db[FEATURES_COLLECTION]


def current_hour_index(time_list: list, now: datetime) -> int:
    target = now.strftime("%Y-%m-%dT%H:00")
    if target in time_list:
        return time_list.index(target)
    return len(time_list) - 1


def fetch_current_conditions(lat: float, lon: float) -> dict:
    now = datetime.now(timezone.utc)

    aq_response = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat, "longitude": lon,
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
            "latitude": lat, "longitude": lon,
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


def add_change_rate(row: dict, city_name: str, collection) -> dict:
    """Compare this reading to the most recent stored reading for the SAME city."""
    try:
        last_doc = collection.find_one(
            {"city": city_name}, sort=[("timestamp", -1)]
        )
        if last_doc is not None:
            row["aqi_change_rate"] = row["aqi"] - last_doc["aqi"]
        else:
            row["aqi_change_rate"] = 0.0
    except Exception:
        row["aqi_change_rate"] = 0.0
    return row


def build_row_for_city(city_name: str, lat: float, lon: float, collection) -> dict:
    row = fetch_current_conditions(lat, lon)
    row["city"] = city_name
    row = add_time_features(row)
    row = add_change_rate(row, city_name, collection)
    return row


def main():
    collection = get_features_collection()

    # keep (city, timestamp) unique, same role the primary key played in Hopsworks
    collection.create_index([("city", 1), ("timestamp", 1)], unique=True)

    inserted = 0
    for city_name, coords in CITIES.items():
        try:
            row = build_row_for_city(city_name, coords["lat"], coords["lon"], collection)
            collection.insert_one(row)
            inserted += 1
            print(f"Fetched {city_name}: AQI={row['aqi']}")
        except Exception as e:
            print(f"Skipped {city_name} due to error: {e}")

    print(f"Inserted {inserted} rows this run.")


if __name__ == "__main__":
    main()