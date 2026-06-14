# -*- coding: utf-8 -*-
"""
LSTM Delay Risk Predictor for FreightSense
Input: sequence of 24 hourly weather + temporal features
Output: probability of 2+ hour delay during the trip window
Architecture: 2-layer LSTM -> Dropout -> Linear -> Sigmoid
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import List

class DelayPredictorLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=2, dropout=0.3):
        super(DelayPredictorLSTM, self).__init__()
        # input_size = number of features per timestep
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # Get last timestep output
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return self.sigmoid(out)


class DelayPredictorService:
    def __init__(self):
        self.model_path = os.path.join("models", "delay_predictor", "lstm_model.pt")
        self.scaler_path = os.path.join("models", "delay_predictor", "scaler.joblib")
        self.model = None
        self.scaler = None
        self.input_size = 13 # Match dataset feature count
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = DelayPredictorLSTM(input_size=self.input_size)
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.scaler = joblib.load(self.scaler_path)
        else:
            print(f"⚠️ DelayPredictor model not found at {self.model_path}. Run training.")

    def _extract_features(self, df_forecast: pd.DataFrame) -> np.ndarray:
        """Extract exact same features used in training"""
        # Ensure correct columns exist
        cols = ['precipitation_mm', 'visibility', 'wind_speed_kmh', 'temperature_c',
                'is_fog', 'is_heavy_rain', 'hour_of_day', 'day_of_week', 'month',
                'is_weekend', 'is_peak_hour', 'is_festival_day', 'is_highway']
                
        df = df_forecast.copy()
        
        # We might be missing some columns during inference, fill with defaults
        if 'visibility' not in df.columns:
            df['visibility'] = 10000
        
        df['is_fog'] = (df['visibility'] < 500).astype(int)
        df['is_heavy_rain'] = (df.get('precipitation_mm', 0) > 5.0).astype(int)
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = df['hour_of_day'].isin([8, 9, 17, 18, 19]).astype(int)
        df['is_festival_day'] = 0 # simplifying for inference
        df['is_highway'] = 1 # Assuming highway for cross-city
        
        # Keep only required columns
        for c in cols:
            if c not in df.columns:
                df[c] = 0
                
        X = df[cols].values
        if self.scaler:
            X = self.scaler.transform(X)
        return X

    def predict(self, lat: float, lon: float, departure_datetime: datetime = None) -> dict:
        """
        Main inference method.
        Fetches 24h forecast, runs LSTM, returns probability.
        """
        if self.model is None or self.scaler is None:
            return {
                "delay_probability": 0.0,
                "delay_risk_level": "UNKNOWN",
                "recommendation": "Predictor model offline."
            }

        # Lazy import to avoid circular dependency
        from src.ingestion.weather_history import fetch_forecast_weather
        
        df_forecast = fetch_forecast_weather(lat, lon, hours_ahead=24)
        if df_forecast.empty or len(df_forecast) < 24:
            return {
                "delay_probability": 0.5,
                "delay_risk_level": "MEDIUM",
                "recommendation": "Unable to fetch complete weather forecast. Use caution."
            }
            
        features = self._extract_features(df_forecast)
        
        # Reshape for LSTM: (batch=1, seq_len=24, input_size)
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prob = self.model(x_tensor).item()
            
        risk_level = "LOW"
        if prob > 0.6:
            risk_level = "HIGH"
        elif prob > 0.3:
            risk_level = "MEDIUM"
            
        recommendation = "Normal conditions expected."
        if prob > 0.6:
            recommendation = "Consider departing before peak hours or exploring alternate routes."
            
        # Try to find peak risk hour from heavy rain or peak hour flags
        peak_hour = None
        if features[:, 5].sum() > 0: # is_heavy_rain
            idx = np.argmax(features[:, 5])
            peak_hour = df_forecast.index[idx].strftime("%H:%00")
            
        return {
            "delay_probability": float(round(prob, 2)),
            "delay_risk_level": risk_level,
            "peak_risk_hour": peak_hour,
            "recommendation": recommendation
        }
