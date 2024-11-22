import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from WeatherPredict import WeatherPredict
import os

# Load data
csv_path = "processed_data.csv"
data = pd.read_csv(csv_path)
print(data.head())
print(data.columns)

# Feature engineering
data["Year"] = data["Serial"].astype(str).str[:4].astype(int)
data["Month"] = data["Serial"].astype(str).str[4:6].astype(int)
data["Day"] = data["Serial"].astype(str).str[6:8].astype(int)
data["hhmm"] = data["Serial"].astype(str).str[8:12].astype(int)
data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)
data["Weekday"] = pd.to_datetime(data["Serial"].astype(str).str[:8]).dt.weekday

# Load weather data
weather_model = "weather_model.joblib"
weather_data = WeatherPredict(weather_model, data)

# Add the weather data to the processed data
data["Pressure(hpa)"] = weather_data[:, 0]
data["WindSpeed(m/s)"] = weather_data[:, 1]
data["Temperature(°C)"] = weather_data[:, 2]
data["Sunlight(Lux)"] = weather_data[:, 3]
data["Humidity(%)"] = weather_data[:, 4]

# Define features and target
X = data[
    [
        "Year",
        "Month",
        "Day",
        "hhmm",
        "DeviceID",
        "Weekday",
        "Pressure(hpa)",
        "WindSpeed(m/s)",
        "Temperature(°C)",
        "Sunlight(Lux)",
        "Humidity(%)",
    ]
]
y = data["Power(mW)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "colsample_bytree": [0.3, 0.7, 0.9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [4, 5, 6, 7],
    "alpha": [10, 20, 30],
    "n_estimators": [100, 200, 300],
    "device": ["cuda"],
}

xgboost_model = xgb.XGBRegressor(objective="reg:squarederror")

grid_search = GridSearchCV(
    estimator=xgboost_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    verbose=1,
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

xgb.plot_importance(best_model)

# Save the model
joblib.dump(best_model, "main_model.joblib")
