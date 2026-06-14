# -*- coding: utf-8 -*-
"""
Generates synthetic delay training dataset by combining:
- Historical weather from Open-Meteo (real data, free)
- Indian festival calendar (hardcoded dict - no API needed)
- Day/time patterns (Monday morning = high congestion, etc.)

Labels: binary - 0 = on time, 1 = delayed 2+ hours
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.weather_history import fetch_historical_weather

INDIAN_FESTIVALS_2023 = [
    "2023-01-14", # Makar Sankranti
    "2023-01-26", # Republic Day
    "2023-03-08", # Holi
    "2023-08-15", # Independence Day
    "2023-08-30", # Raksha Bandhan
    "2023-09-19", # Ganesh Chaturthi
    "2023-10-24", # Dussehra
    "2023-11-12", # Diwali
    "2023-12-25"  # Christmas
]

def generate_dataset(days_history: int = 365) -> pd.DataFrame:
    print(f"🔄 Generating synthetic delay dataset for the last {days_history} days...")
    
    end_date = datetime(2023, 12, 31)
    start_date = end_date - timedelta(days=days_history)
    
    # We use Delhi coordinates as a representative Indian city
    lat, lon = 28.6139, 77.2090
    
    # Fetch real weather
    df_weather = fetch_historical_weather(lat, lon, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if df_weather.empty:
        print("⚠️ Failed to fetch weather data. Generating fully synthetic weather...")
        # Fallback fully synthetic if API fails
        date_rng = pd.date_range(start=start_date, end=end_date, freq='H')
        df_weather = pd.DataFrame(index=date_rng)
        df_weather['precipitation_mm'] = np.random.exponential(scale=0.5, size=(len(date_rng)))
        df_weather['visibility'] = np.random.normal(loc=10000, scale=2000, size=(len(date_rng)))
        df_weather['wind_speed_kmh'] = np.random.normal(loc=10, scale=5, size=(len(date_rng)))
        df_weather['temperature_c'] = np.random.normal(loc=25, scale=10, size=(len(date_rng)))
        
    df = df_weather.copy()
    
    # Feature Engineering
    df['is_fog'] = (df['visibility'] < 500).astype(int) if 'visibility' in df.columns else 0
    df['is_heavy_rain'] = (df['precipitation_mm'] > 5.0).astype(int)
    
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 8-10 AM, 5-8 PM
    df['is_peak_hour'] = df['hour_of_day'].isin([8, 9, 17, 18, 19]).astype(int)
    
    # Dates
    date_strs = df.index.strftime('%Y-%m-%d')
    df['is_festival_day'] = date_strs.isin(INDIAN_FESTIVALS_2023).astype(int)
    
    # Generate random routes context for each row
    df['route_base_risk'] = np.random.uniform(0.1, 0.5, size=len(df))
    df['is_highway'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
    df['has_city_entry'] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
    df['cargo_risk_weight'] = np.random.choice([0.3, 0.6, 1.0], size=len(df), p=[0.7, 0.2, 0.1])
    
    # Label generation heuristic (Base Probability)
    prob = 0.1 # base delay prob
    prob += df['is_heavy_rain'] * 0.4
    prob += df['is_fog'] * 0.3
    prob += df['is_peak_hour'] * 0.15 * df['has_city_entry']
    prob += df['is_festival_day'] * 0.2
    prob += df['route_base_risk'] * 0.2
    
    # Add gaussian noise
    prob += np.random.normal(0, 0.1, size=len(df))
    
    # Clip and threshold
    prob = np.clip(prob, 0, 1)
    df['delay_label'] = (prob > 0.5).astype(int)
    
    # Handle NaN
    df.fillna(0, inplace=True)
    
    output_path = os.path.join("data", "delay_training_data.csv")
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path)
    
    print(f"✅ Generated {len(df)} rows. Saved to {output_path}")
    print(f"Class distribution: {df['delay_label'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    generate_dataset()
