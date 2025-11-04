import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path="data/weather_data.csv", date_col="date"):
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df

def create_features(df, target_col="temperature", lags=(1,2,3,7,14)):
    df = df.copy()
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

    for lag in lags:
        df[f"temp_lag_{lag}"] = df[target_col].shift(lag)
        if "humidity" in df.columns:
            df[f"hum_lag_{lag}"] = df["humidity"].shift(lag)
        if "rainfall" in df.columns:
            df[f"rain_lag_{lag}"] = df["rainfall"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ['date', target_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, df

def train_test_split_timeaware(X, y, test_size=0.2):
    n = len(X)
    split_at = int(n * (1 - test_size))
    X_train = X.iloc[:split_at].reset_index(drop=True)
    X_test = X.iloc[split_at:].reset_index(drop=True)
    y_train = y.iloc[:split_at].reset_index(drop=True)
    y_test = y.iloc[split_at:].reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def scale_and_persist(X_train, X_test, scaler_path="models/scaler.joblib"):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, scaler
