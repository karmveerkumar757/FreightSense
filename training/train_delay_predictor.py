# -*- coding: utf-8 -*-
"""
Training script for the LSTM Delay Risk Predictor
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nlp.delay_predictor import DelayPredictorLSTM

def create_sequences(data: np.ndarray, labels: np.ndarray, seq_length: int = 24):
    """Create rolling window sequences"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        # We predict if delay happens in the window, using the label at the end of the window
        y = labels[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    print("🚀 Starting LSTM Delay Predictor Training Pipeline...")
    
    data_path = os.path.join("data", "delay_training_data.csv")
    if not os.path.exists(data_path):
        print("Data not found. Running generator...")
        from data.generate_delay_dataset import generate_dataset
        df = generate_dataset()
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
    print(f"Loaded {len(df)} hourly records.")
    
    # Select features exactly matching the predictor service
    feature_cols = ['precipitation_mm', 'visibility', 'wind_speed_kmh', 'temperature_c',
                    'is_fog', 'is_heavy_rain', 'hour_of_day', 'day_of_week', 'month',
                    'is_weekend', 'is_peak_hour', 'is_festival_day', 'is_highway']
                    
    # Handle missing columns safely
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X_raw = df[feature_cols].values
    y_raw = df['delay_label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    seq_length = 24
    X_seq, y_seq = create_sequences(X_scaled, y_raw, seq_length)
    
    print(f"Created {len(X_seq)} sequences of length {seq_length}.")
    
    # Splits (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Tensors
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_size = len(feature_cols)
    model = DelayPredictorLSTM(input_size=input_size).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 20
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0
    
    model_dir = os.path.join("models", "delay_predictor")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lstm_model.pt")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        y_val_preds, y_val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                y_val_preds.extend(outputs.cpu().numpy())
                y_val_true.extend(y_batch.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        try:
            auc = roc_auc_score(y_val_true, y_val_preds)
        except ValueError:
            auc = 0.5
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val AUC: {auc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
    # Final eval on test set
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds_prob = model(X_t).cpu().numpy()
        preds_bin = (preds_prob > 0.5).astype(int)
        
    print("\n✅ Final Test Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, preds_bin):.4f}")
    print(f"Precision: {precision_score(y_test, preds_bin, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds_bin, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds_bin, zero_division=0):.4f}")
    try:
        print(f"AUC-ROC:   {roc_auc_score(y_test, preds_prob):.4f}")
    except ValueError:
        pass
        
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()

# COLAB SETUP INSTRUCTIONS:
# !pip install torch pandas scikit-learn numpy
# !python training/train_delay_predictor.py
