import pandas as pd
import lightgbm as lgb
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

# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    "num_leaves": [int(x) for x in np.linspace(20, 150, num=10)],
    "learning_rate": [
        round(float(x), 2) for x in np.linspace(start=0.01, stop=0.2, num=10)
    ],
    "max_depth": [int(x) for x in np.linspace(5, 50, num=10)],
    "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
}

lightgbm_model = lgb.LGBMRegressor(objective="regression")

random_search = RandomizedSearchCV(
    estimator=lightgbm_model,
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

lgb.plot_importance(best_model, max_num_features=10)

# Save the model
joblib.dump(best_model, "one_model_LightGBM.joblib")
