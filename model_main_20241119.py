import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from openWeather import openWeather
import optuna

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
# weather_model = "weather_model.joblib"
# weather_data = WeatherPredict(weather_model, data)
humidity_model = "humidity_model.joblib"
pressure_model = "pressure_model.joblib"
sunlight_model = "sunlight_model.joblib"
temperature_model = "temperature_model.joblib"
wind_speed_model = "wind_speed_model.joblib"

# Add the weather data to the processed data
# data["Pressure(hpa)"] = weather_data[:, 0]
# data["WindSpeed(m/s)"] = weather_data[:, 1]
# data["Temperature(°C)"] = weather_data[:, 2]
# data["Sunlight(Lux)"] = weather_data[:, 3]
# data["Humidity(%)"] = weather_data[:, 4]
data, weather_columns = openWeather(data)
X = data[["Year", "Month", "Day", "hhmm", "DeviceID", *weather_columns]]

data["Pressure(hpa)"] = joblib.load(pressure_model).predict(X)
data["WindSpeed(m/s)"] = joblib.load(wind_speed_model).predict(X)
data["Temperature(°C)"] = joblib.load(temperature_model).predict(X)
data["Sunlight(Lux)"] = joblib.load(sunlight_model).predict(X)
data["Humidity(%)"] = joblib.load(humidity_model).predict(X)

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

# Use a smaller subset of the training data for hyperparameter tuning
X_train_sub = X_train.sample(frac=0.1, random_state=42)
y_train_sub = y_train.loc[X_train_sub.index]


def objective(trial):
    param = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "alpha": trial.suggest_float("alpha", 0, 10),
        "lambda": trial.suggest_float("lambda", 0, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "tree_method": "hist",
        "device": "cuda",
    }

    model = xgb.XGBRegressor(**param)
    model.fit(
        X_train_sub,
        y_train_sub,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Reduce the number of trials

best_params = study.best_params
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

xgb.plot_importance(best_model)

# Save the model
joblib.dump(best_model, "main_model.joblib")
