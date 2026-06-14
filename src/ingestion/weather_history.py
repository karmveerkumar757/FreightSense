# -*- coding: utf-8 -*-
"""
Fetches historical and forecast weather data from Open-Meteo API
(FREE, no API key required)
Used for the LSTM Delay Predictor
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches hourly historical weather data for training.
    Dates must be in 'YYYY-MM-DD' format.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation,visibility,wind_speed_10m,temperature_2m,weathercode",
        "timezone": "Asia/Kolkata"
    }
    
    print(f"📡 Fetching historical weather from Open-Meteo for {lat},{lon} ({start_date} to {end_date})...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly = data.get("hourly", {})
        df = pd.DataFrame(hourly)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            # rename for consistency
            df.rename(columns={
                "precipitation": "precipitation_mm",
                "wind_speed_10m": "wind_speed_kmh",
                "temperature_2m": "temperature_c"
            }, inplace=True)
            return df
    else:
        print(f"⚠️ Open-Meteo API Error: {response.status_code}")
        print(response.text)
        
    return pd.DataFrame()

def fetch_forecast_weather(lat: float, lon: float, hours_ahead: int = 24) -> pd.DataFrame:
    """
    Fetches hourly forecast for inference.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation,visibility,wind_speed_10m,temperature_2m,weathercode",
        "forecast_days": (hours_ahead // 24) + 1,
        "timezone": "Asia/Kolkata"
    }
    
    print(f"📡 Fetching forecast weather from Open-Meteo for {lat},{lon}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly = data.get("hourly", {})
        df = pd.DataFrame(hourly)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.rename(columns={
                "precipitation": "precipitation_mm",
                "wind_speed_10m": "wind_speed_kmh",
                "temperature_2m": "temperature_c"
            }, inplace=True)
            
            # Filter to just the requested hours ahead from now
            now = datetime.now()
            end_time = now + timedelta(hours=hours_ahead)
            df = df[(df.index >= now) & (df.index <= end_time)]
            
            # If less than 24 hours available, pad it (safety check)
            if len(df) < hours_ahead and not df.empty:
                last_row = df.iloc[-1:]
                pad_length = hours_ahead - len(df)
                padding = pd.concat([last_row] * pad_length)
                # update index
                new_idx = [df.index[-1] + timedelta(hours=i+1) for i in range(pad_length)]
                padding.index = new_idx
                df = pd.concat([df, padding])
                
            return df.head(hours_ahead)
    else:
        print(f"⚠️ Open-Meteo API Error: {response.status_code}")
        
    return pd.DataFrame()
