import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate(start_date="2019-01-01", days=1200, seed=42):
    np.random.seed(seed)
    start = datetime.fromisoformat(start_date)
    dates = [start + timedelta(days=i) for i in range(days)]

    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)
    temp += np.random.normal(0, 2.2, size=days)

    humidity = 70 - 10 * np.sin(2 * np.pi * day_of_year / 365)
    humidity += np.random.normal(0, 5, size=days)

    prob_rain = 0.15 + 0.15 * (np.sin(2 * np.pi * (day_of_year-150) / 365) + 1)/2
    rain = (np.random.rand(days) < prob_rain).astype(float) * np.random.gamma(2, 5, size=days)

    df = pd.DataFrame({
        "date": dates,
        "temperature": temp.round(2),
        "humidity": humidity.round(1),
        "rainfall": rain.round(2)
    })
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/weather_data.csv", index=False)
    print("Saved data/weather_data.csv with", len(df), "rows")
