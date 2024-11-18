import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
# from WeatherPredict import WeatherPredict

# Load data
csv_path = "processed_data.csv"
data = pd.read_csv(csv_path)
print(data.head())
print(data.columns)

# Feature engineering
data["mmdd"] = data["Serial"].astype(str).str[4:8].astype(int)
data["hhmm"] = data["Serial"].astype(str).str[8:12].astype(int)
data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)

# Load weather data
# weather_model = "weather_model.joblib"
# weather_data = WeatherPredict(weather_model, data)

# Add the weather data to the processed data
# data["Sunlight(Lux)"] = weather_data[:, 0]
# data["Temperature(°C)"] = weather_data[:, 1]

# Define features and target
X = data[
    [
        "DeviceID",
        "Sunlight(Lux)",
        "Temperature(°C)",
        "hhmm",
        "mmdd",
    ]
]
y = data["Power(mW)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    "colsample_bytree": [0.3, 0.7, 0.9],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [4, 5, 6, 7],
    "alpha": [10, 20, 30],
    "n_estimators": [100, 200, 300],
}

xgboost_model = xgb.XGBRegressor(objective="reg:squarederror")

random_search = RandomizedSearchCV(
    estimator=xgboost_model,
    param_distributions=param_dist,
    n_iter=100,
    cv=3,
    scoring="neg_mean_absolute_error",
    verbose=1,
    random_state=42,
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

xgb.plot_importance(best_model)

# Save the model
joblib.dump(best_model, "main_model.joblib")
