import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

from preprocess import load_data, create_features

MODEL_PATH = "models/weather_model.joblib"
SCALER_PATH = "models/scaler.joblib"
DATA_PATH = "data/weather_data.csv"

def predict_next_day(model_path=MODEL_PATH, scaler_path=SCALER_PATH, data_path=DATA_PATH, target_col="temperature"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = load_data(data_path)
    X, y, df_features = create_features(df, target_col=target_col)
    last_row = X.iloc[[-1]].copy()
    last_date = df_features['date'].iloc[-1]
    X_scaled = scaler.transform(last_row)
    pred = model.predict(X_scaled)[0]
    print("Last available date in dataset:", last_date.date())
    print("Predicted", target_col, "for next day (approx):", round(pred, 2))
    return pred

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train model first by running: python src/train_model.py")
    else:
        predict_next_day()
