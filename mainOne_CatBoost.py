import joblib
import pandas as pd
from openWeather import openWeather
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def PowerPredict(main_model_path, data):
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

    data, weather_columns = openWeather(data)

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

    y_pred = main_model.predict(X)
    return y_pred


if __name__ == "__main__":
    csv_path = "upload.csv"
    data = pd.read_csv(csv_path)

    y_pred = PowerPredict("one_model_CatBoost.joblib", data)
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.round(y_pred, 2)

    processed_data = pd.read_csv("processed_data.csv")
    # Ensure the Serial columns are of the same type
    processed_data["Serial"] = processed_data["Serial"].astype(str)
    print(processed_data)
    data["Serial"] = data["Serial"].astype(str)
    print(data)
    y_test = processed_data.loc[
        processed_data["Serial"].isin(data["Serial"]), "Power(mW)"
    ].values

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Save predictions to CSV
    output = pd.DataFrame({"序號": data["Serial"], "答案": y_pred})
    output.to_csv("predictions.csv", index=False, encoding="utf-8", header=False)

    # score = sum(abs(y_test - y_pred))
    score = sum(abs(y_test - y_pred))
    print(f"Score: {score}")
