import joblib
import pandas as pd

# from WeatherPredict import WeatherPredict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from openWeather import openWeather


def PowerPredict(main_model_path, data):
    # Load the model from the file
    main_model = joblib.load(main_model_path)

    # rename columns
    data.columns = ["Serial", "Power(mW)"]

    # Make predictions
    data["Year"] = data["Serial"].astype(str).str[:4].astype(int)
    data["Month"] = data["Serial"].astype(str).str[4:6].astype(int)
    data["Day"] = data["Serial"].astype(str).str[6:8].astype(int)
    data["Datetime"] = pd.to_datetime(
        data["Serial"].astype(str).str[:12], format="%Y%m%d%H%M"
    )
    data["day_of_year"] = [i.dayofyear for i in data["Datetime"]]
    data["hour"] = [i.hour for i in data["Datetime"]]
    data["minute"] = [i.minute for i in data["Datetime"]]
    data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)
    data["Weekday"] = pd.to_datetime(data["Serial"].astype(str).str[:8]).dt.weekday

    # weather_data = WeatherPredict(weather_model_path, data)
    # data["Pressure(hpa)"] = weather_data[:, 0]
    # data["WindSpeed(m/s)"] = weather_data[:, 1]
    # data["Temperature(°C)"] = weather_data[:, 2]
    # data["Sunlight(Lux)"] = weather_data[:, 3]
    # data["Humidity(%)"] = weather_data[:, 4]
    humidity_model = "humidity_model.joblib"
    pressure_model = "pressure_model.joblib"
    sunlight_model = "sunlight_model.joblib"
    temperature_model = "temperature_model.joblib"
    wind_speed_model = "wind_speed_model.joblib"

    data, weather_columns = openWeather(data)
    data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)
    # 打開參考資料
    SourceData = pd.read_csv("processed_data.csv")
    SourceData, weather_columns = openWeather(SourceData)
    SourceData["Datetime"] = pd.to_datetime(
        SourceData["Serial"].astype(str).str[:12], format="%Y%m%d%H%M"
    )
    SourceData["DeviceID"] = SourceData["Serial"].astype(str).str[12:14].astype(int)
    SourceData["day_of_year"] = [i.dayofyear for i in SourceData["Datetime"]]
    SourceData["hour"] = [i.hour for i in SourceData["Datetime"]]
    SourceData["minute"] = [i.minute for i in SourceData["Datetime"]]

    # Filter data to get the rows where the time is 08:50
    data_850 = SourceData[(SourceData["hour"] == 8) & (SourceData["minute"] == 50)]

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
    print(data.head())

    X = data[
        [
            # "Year",
            # "Month",
            # "Day",
            "hour",
            "minute",
            "DeviceID",
            # "day_of_year",
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
            # "Year",
            # "Month",
            # "Day",
            "hour",
            "minute",
            "DeviceID",
            # "day_of_year",
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

    # 輸出預測出來的天氣
    data[
        [
            "Serial",
            "WindSpeed(m/s)",
            "Pressure(hpa)",
            "Temperature(°C)",
            "Humidity(%)",
            "Sunlight(Lux)",
        ]
    ].to_csv("Pweather_data.csv", index=False)
    # Calculate MAE for predicted weather data
    # mae_weather = {}
    # for column in [
    #     "Pressure(hpa)",
    #     "WindSpeed(m/s)",
    #     "Temperature(°C)",
    #     "Sunlight(Lux)",
    #     "Humidity(%)",
    # ]:
    #     original = SourceData.loc[SourceData["Serial"].isin(data["Serial"]), column]
    #     predicted = data[column]
    #     mae_weather[column] = mean_absolute_error(original, predicted)
    #     r2_score = np.corrcoef(original, predicted)[0, 1] ** 2
    #     print(f"Mean Absolute Error for {column}: {mae_weather[column]}")
    #     print(f"誤差 for {column}: {mae_weather[column]/original.max()}")
    #     print(f"R2 Score for {column}: {r2_score}")

    X = data[
        [
            # "Year",
            # "Month",
            # "Day",
            "hour",
            "minute",
            "DeviceID",
            # "day_of_year",
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

    y_pred = main_model.predict(X)
    return y_pred


if __name__ == "__main__":
    csv_path = "upload(no answer).csv"
    data = pd.read_csv(csv_path)

    y_pred = PowerPredict("main_model_stacking.joblib", data)
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.round(y_pred, 2)
    try:
        print(y_pred)
        processed_data = pd.read_csv("processed_data.csv")
        y_test = processed_data.loc[processed_data["Serial"].isin(data["Serial"])][
            "Power(mW)"
        ]
        # Handle NaN values
        y_test = y_test.fillna(0)
        y_pred = np.nan_to_num(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        score = sum(abs(y_test - y_pred))
        print(f"Score: {score}")
    except:
        pass

    # Save predictions to CSV
    output = pd.DataFrame({"序號": data["Serial"], "答案": y_pred})
    output.to_csv("predictions.csv", index=False, encoding="utf-8", header=False)
