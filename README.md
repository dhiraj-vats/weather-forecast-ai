# 🌦️ Weather-ML (SkyCastML)

**Weather-ML** is a machine learning project built using **Python** and **scikit-learn** for predicting future weather conditions like temperature, humidity, and rainfall based on past climate data.  

This project demonstrates an end-to-end ML workflow — from data generation and preprocessing to model training, evaluation, and prediction.

---

## 🚀 Features

- End-to-end weather forecasting pipeline  
- Data cleaning and preprocessing using Pandas and NumPy  
- Feature engineering (lag-based features + cyclical encoding for seasonality)  
- Model training using Random Forest Regressor (scikit-learn)  
- Evaluation using MAE, MSE, and R² score  
- Visualization of predicted vs actual temperature  
- Lightweight and easily extendable  

---

## 🧠 Tech Stack

- **Language:** Python 3.8+  
- **Libraries:**  
  - pandas  
  - numpy  
  - scikit-learn  
  - matplotlib  
  - joblib  

---

## 🗂️ Project Structure

```
SkyCastML/
│
├── data/
│   ├── weather_data.csv              # Dataset (auto-generated)
│   └── generate_sample_data.py       # Script to create sample data
│
├── models/
│   ├── weather_model.joblib          # Trained ML model
│   ├── metrics.joblib                # Model evaluation metrics
│   └── plots/true_vs_pred.png        # Visualization of results
│
├── src/
│   ├── preprocess.py                 # Data preprocessing utilities
│   ├── train_model.py                # Model training and evaluation
│   └── predict.py                    # Prediction for next day
│
├── requirements.txt                  # Dependencies
├── .gitignore                        # Ignore unnecessary files
└── README.txt                        # Documentation
```

---

## ⚙️ Installation

### Step 1: Clone the repository
```
git clone git@github.com:dhiraj-vats/weather-forecast-ai.git
cd weather-forecast-ai
```

### Step 2: Create virtual environment (optional)
```
python -m venv venv
venv\Scripts\activate      (Windows)
# OR
source venv/bin/activate   (Linux/Mac)
```

### Step 3: Install dependencies
```
pip install -r requirements.txt
```

---

## 🌤️ Usage

### 1️⃣ Generate synthetic weather data
```
python data/generate_sample_data.py
```

### 2️⃣ Train the ML model
```
python src/train_model.py
```

### 3️⃣ Predict next day temperature
```
python src/predict.py
```

---

## 📊 Example Output

```
Model saved to: models/weather_model.joblib
MAE: 1.82 | MSE: 3.91 | R2: 0.94
Saved prediction plot to models/plots/true_vs_pred.png

Last available date in dataset: 2022-04-15
Predicted temperature for next day (approx): 30.6°C
```

---

## 📈 Visualization Example

(After training, you’ll find this plot at `models/plots/true_vs_pred.png`)

```
[True vs Predicted Temperature Graph]
```

---

## 🧪 Future Enhancements

- Integrate with OpenWeatherMap API for real-time data  
- Build a Streamlit web app dashboard for forecasts  
- Extend model for rainfall and humidity prediction  
- Add LSTM or Prophet for time-series forecasting  

---

## 🧑‍💻 Author

**Dhiraj Kumar**  
Team Lead – Full Stack Development  
GitHub: git@github.com:dhiraj-vats/weather-forecast-ai.git

---

## 🪪 License

This project is licensed under the **MIT License** —  
You can freely use, modify, and distribute it.

---

## 💡 Quick Summary

| Step | Command | Description |
|------|----------|-------------|
| 1 | `pip install -r requirements.txt` | Install required packages |
| 2 | `python data/generate_sample_data.py` | Generate dataset |
| 3 | `python src/train_model.py` | Train ML model |
| 4 | `python src/predict.py` | Predict next day weather |
| 5 | Check `models/plots/true_vs_pred.png` | View prediction results |

---

✅ **Now you’re ready to upload this project to GitHub!**  
Repo name suggestion: **weather-forecast-ai** or **skycast-ml**
