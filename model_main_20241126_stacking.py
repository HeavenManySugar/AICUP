import pandas as pd
from catboost import CatBoostRegressor, Pool
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
data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)
data["Weekday"] = pd.to_datetime(data["Serial"].astype(str).str[:8]).dt.weekday

# Load weather data
humidity_model = "humidity_model.joblib"
pressure_model = "pressure_model.joblib"
sunlight_model = "sunlight_model.joblib"
temperature_model = "temperature_model.joblib"
wind_speed_model = "wind_speed_model.joblib"

# Add the weather data to the processed data
data, weather_columns = openWeather(data)
data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)

data["Datetime"] = pd.to_datetime(
    data["Serial"].astype(str).str[:12], format="%Y%m%d%H%M"
)
data["day_of_year"] = [i.dayofyear for i in data["Datetime"]]
data["hour"] = [i.hour for i in data["Datetime"]]
data["minute"] = [i.minute for i in data["Datetime"]]

# Filter data to get the rows where the time is 08:50
data_850 = data[(data["hour"] == 8) & (data["minute"] == 50)]

# Select only the required columns
data_850 = data_850[
    [
        "DeviceID",
        "day_of_year",
        "Pressure(hpa)",
        "WindSpeed(m/s)",
        "Temperature(°C)",
        "Sunlight(Lux)",
        "Humidity(%)",
    ]
]

# Rename columns to indicate they are from 08:50
data_850.columns = [
    "DeviceID",
    "day_of_year",
    "Pressure_850",
    "WindSpeed_850",
    "Temperature_850",
    "Sunlight_850",
    "Humidity_850",
]

data = pd.merge(
    data,
    data_850,
    on=["DeviceID", "day_of_year"],
    how="left",
    suffixes=("", "_duplicate"),
)

X = data[
    [
        "hour",
        "minute",
        "DeviceID",
        *weather_columns,
        "Pressure_850",
        "WindSpeed_850",
        "Temperature_850",
        "Sunlight_850",
        "Humidity_850",
    ]
]
data["_Pressure(hpa)"] = joblib.load(pressure_model).predict(X)
data["_WindSpeed(m/s)"] = joblib.load(wind_speed_model).predict(X)
data["_Temperature(°C)"] = joblib.load(temperature_model).predict(X)
data["_Sunlight(Lux)"] = joblib.load(sunlight_model).predict(X)
data["_Humidity(%)"] = joblib.load(humidity_model).predict(X)

weather_model = "weather_model.joblib"
X = data[
    [
        "hour",
        "minute",
        "DeviceID",
        "_Pressure(hpa)",
        "_WindSpeed(m/s)",
        "_Temperature(°C)",
        "_Sunlight(Lux)",
        "_Humidity(%)",
        *weather_columns,
        "Pressure_850",
        "WindSpeed_850",
        "Temperature_850",
        "Sunlight_850",
        "Humidity_850",
    ]
]
weather_data = joblib.load(weather_model).predict(X)
data["Pressure(hpa)"] = weather_data[:, 0]
data["WindSpeed(m/s)"] = weather_data[:, 1]
data["Temperature(°C)"] = weather_data[:, 2]
data["Sunlight(Lux)"] = weather_data[:, 3]
data["Humidity(%)"] = weather_data[:, 4]

# Define features and target
X = data[
    [
        "hour",
        "minute",
        "DeviceID",
        "_Pressure(hpa)",
        "_WindSpeed(m/s)",
        "_Temperature(°C)",
        "_Sunlight(Lux)",
        "_Humidity(%)",
        "Pressure(hpa)",
        "WindSpeed(m/s)",
        "Temperature(°C)",
        "Sunlight(Lux)",
        "Humidity(%)",
        *weather_columns,
        "Pressure_850",
        "WindSpeed_850",
        "Temperature_850",
        "Sunlight_850",
        "Humidity_850",
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
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "od_type": "Iter",
        "od_wait": 100,
    }

    model = CatBoostRegressor(**param)
    model.fit(
        X_train_sub,
        y_train_sub,
        eval_set=(X_test, y_test),
        verbose=False,
    )
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Reduce the number of trials

best_params = study.best_params
best_model = CatBoostRegressor(**best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the model
joblib.dump(best_model, "main_model_stacking.joblib")
