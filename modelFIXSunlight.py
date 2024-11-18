import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from openWeather import openWeather
import optuna

data = pd.read_csv("processed_data.csv")
data, weather_columns = openWeather(data)
print(data.head())
print(data.columns)

data["Year"] = data["Serial"].astype(str).str[:4].astype(int)
data["Month"] = data["Serial"].astype(str).str[4:6].astype(int)
data["Day"] = data["Serial"].astype(str).str[6:8].astype(int)
data["hhmm"] = data["Serial"].astype(str).str[8:12].astype(int)
data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)

X = data[
    [
        "Year",
        "Month",
        "Day",
        "hhmm",
        "DeviceID",
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Power(mW)",
    ]
]
y = data[
    [
        "Sunlight(Lux)",
    ]
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use a smaller subset of the training data for hyperparameter tuning
X_train_sub = X_train.sample(frac=0.1, random_state=42)
y_train_sub = y_train.loc[X_train_sub.index]


def objective(trial):
    param = {
        "max_depth": trial.suggest_int("max_depth", 6, 15),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
        "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
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
joblib.dump(best_model, "FIX_sunlight_model.joblib")
