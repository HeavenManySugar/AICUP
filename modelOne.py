import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
from openWeather import openWeather

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
data, weather_columns = openWeather(data)

# Define features and target
X = data[
    [
        "Year",
        "Month",
        "Day",
        "hhmm",
        "DeviceID",
        "Weekday",
        *weather_columns,
    ]
]
y = data["Power(mW)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning using GridSearchCV
param_dist = {
    "colsample_bytree": [
        round(float(x), 2) for x in np.linspace(start=0.1, stop=1, num=10)
    ],
    "learning_rate": [
        round(float(x), 2) for x in np.linspace(start=0.01, stop=0.2, num=10)
    ],
    "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
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
joblib.dump(best_model, "one_model.joblib")
