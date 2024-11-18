import joblib
import pandas as pd
from openWeather import openWeather


def WeatherPredict(model_path, data):
    # Load the model from the file
    data, weather_columns = openWeather(data)
    weather_model = joblib.load(model_path)

    # Make predictions
    data["Year"] = data["Serial"].astype(str).str[:4].astype(int)
    data["Month"] = data["Serial"].astype(str).str[4:6].astype(int)
    data["Day"] = data["Serial"].astype(str).str[6:8].astype(int)
    data["hhmm"] = data["Serial"].astype(str).str[8:12].astype(int)
    data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)

    X = data[["Year", "Month", "Day", "hhmm", "DeviceID", *weather_columns]]

    y_pred = weather_model.predict(X)

    return y_pred


if __name__ == "__main__":
    csv_path = "processed_data.csv"
    # Load the test data
    data = pd.read_csv(csv_path)
    y_pred = WeatherPredict("weather_model.joblib", data)
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    print(y_pred)
    y_test = data[
        [
            "Sunlight(Lux)",
            "Temperature(°C)",
        ]
    ]
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
