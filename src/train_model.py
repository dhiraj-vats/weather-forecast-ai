import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import load_data, create_features, train_test_split_timeaware, scale_and_persist

def train(save_model_path="models/weather_model.joblib",
          data_path="data/weather_data.csv",
          target_col="temperature"):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    df = load_data(data_path)
    X, y, df_with_features = create_features(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split_timeaware(X, y, test_size=0.2)
    X_train_s, X_test_s, scaler = scale_and_persist(X_train, X_test, scaler_path="models/scaler.joblib")
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    joblib.dump(model, save_model_path)
    joblib.dump({"mae": mae, "mse": mse, "r2": r2}, "models/metrics.joblib")
    print("Model saved to:", save_model_path)
    print(f"MAE: {mae:.3f} | MSE: {mse:.3f} | R2: {r2:.3f}")
    try:
        os.makedirs("models/plots", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(y_test.values[:200], label="True")
        plt.plot(y_pred[:200], label="Pred")
        plt.title("True vs Predicted (first 200 points of test set)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("models/plots/true_vs_pred.png")
        plt.close()
        print("Saved prediction plot to models/plots/true_vs_pred.png")
    except Exception as e:
        print("Plot failed:", e)

if __name__ == "__main__":
    train()
