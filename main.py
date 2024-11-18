import joblib
import pandas as pd
from WeatherPredict import WeatherPredict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def PowerPredict(main_model_path, weather_model_path, data):
    # Load the model from the file
    main_model = joblib.load(main_model_path)

    # rename columns
    data.columns = ["Serial", "Power(mW)"]

    # Make predictions
    data["Year"] = data["Serial"].astype(str).str[:4].astype(int)
    data["Month"] = data["Serial"].astype(str).str[4:6].astype(int)
    data["Day"] = data["Serial"].astype(str).str[6:8].astype(int)
    data["hhmm"] = data["Serial"].astype(str).str[8:12].astype(int)
    data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)
    data["Weekday"] = pd.to_datetime(data["Serial"].astype(str).str[:8]).dt.weekday

    weather_data = WeatherPredict(weather_model_path, data)
    data["Pressure(hpa)"] = weather_data[:, 0]
    data["WindSpeed(m/s)"] = weather_data[:, 1]
    data["Temperature(°C)"] = weather_data[:, 2]
    data["Sunlight(Lux)"] = weather_data[:, 3]
    data["Humidity(%)"] = weather_data[:, 4]

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

    y_pred = main_model.predict(X)
    return y_pred


if __name__ == "__main__":
    csv_path = "upload(no answer).csv"
    data = pd.read_csv(csv_path)

    y_pred = PowerPredict("main_model.joblib", "weather_model.joblib", data)
    y_pred = np.maximum(y_pred, 0)

    print(y_pred)
    y_test = data["Power(mW)"]
    # Handle NaN values
    y_test = y_test.fillna(0)
    y_pred = np.nan_to_num(y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Save predictions to CSV
    output = pd.DataFrame({"序號": data["Serial"], "答案": y_pred})
    output.to_csv("predictions.csv", index=False, encoding="utf-8")

    # score = sum(abs(y_test - y_pred))
    score = sum(abs(y_test - y_pred))
    print(f"Score: {score}")
