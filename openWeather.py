import os
from io import StringIO
import pandas as pd
import numpy as np


def _openWeatherCSV(csv_dir):
    # open weather data
    csv_files = []
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if file.endswith("lotsDataReports.csv"):
                csv_files.append(os.path.join(root, file))
    comment_chars = ['"', "*", "\ufeff"]
    weather_dataframes = []
    for csv_file in csv_files:
        with open(csv_file, "r", encoding="utf-8-sig") as file:
            lines = [
                line
                for line in file
                if not any(line.lstrip().startswith(c) for c in comment_chars)
            ]
        weather_data = pd.read_csv(StringIO("\n".join(lines)))
        weather_data.columns = weather_data.columns.str.strip()
        weather_dataframes.append(weather_data)
    weather_data = pd.concat(weather_dataframes, ignore_index=True)
    weather_data["yyyymmddhh"] = pd.to_numeric(
        weather_data["yyyymmddhh"], errors="coerce"
    ).astype("Int64")
    return weather_data


def openWeather(data):
    data["yyyymmddhh"] = (
        data["Serial"].astype(str).str[:4]
        + data["Serial"].astype(str).str[4:6]
        + data["Serial"].astype(str).str[6:8]
        + data["Serial"].astype(str).str[8:10]
    )
    data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)
    data["original_index"] = data.index

    weather_data_C0Z100 = _openWeatherCSV("C0Z100")
    weather_data_C0Z100 = weather_data_C0Z100.set_index("yyyymmddhh")
    # ensure both yyyymmddhh columns are of the same type
    data["yyyymmddhh"] = pd.to_numeric(data["yyyymmddhh"], errors="coerce").astype(
        "Int64"
    )
    weather_data_C0Z100 = weather_data_C0Z100.add_suffix("_C0Z100")
    weather_data_C0Z100.drop(columns=["# stno_C0Z100"], inplace=True)

    # find the closest yyyymmddhh to merge
    result_data = pd.merge_asof(
        data.sort_values("yyyymmddhh"),
        weather_data_C0Z100.sort_index(),
        on="yyyymmddhh",
        direction="nearest",
    )
    weather_columns = weather_data_C0Z100.columns

    existing_columns_C0Z100 = weather_data_C0Z100.columns
    print(existing_columns_C0Z100)
    # Convert columns to numeric types
    for col in existing_columns_C0Z100:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find 466990
    weather_data_466990 = _openWeatherCSV("466990")
    weather_data_466990 = weather_data_466990.set_index("yyyymmddhh")
    weather_data_466990 = weather_data_466990.add_suffix("_466990")
    weather_data_466990.drop(columns=["# stno_466990"], inplace=True)
    # find the closest yyyymmddhh to merge
    result_data = pd.merge_asof(
        result_data.sort_values("yyyymmddhh"),
        weather_data_466990.sort_index(),
        on="yyyymmddhh",
        direction="nearest",
    )
    existing_columns_466990 = weather_data_466990.columns
    weather_columns = weather_columns.tolist() + existing_columns_466990.tolist()

    # Convert columns to numeric types
    for col in existing_columns_466990:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    weather_data_72T250 = _openWeatherCSV("72T250")
    weather_data_72T250 = weather_data_72T250.set_index("yyyymmddhh")
    weather_data_72T250 = weather_data_72T250.add_suffix("_72T250")
    weather_data_72T250.drop(columns=["# stno_72T250"], inplace=True)
    # find the closest yyyymmddhh to merge
    result_data = pd.merge_asof(
        result_data.sort_values("yyyymmddhh"),
        weather_data_72T250.sort_index(),
        on="yyyymmddhh",
        direction="nearest",
    )
    existing_columns_72T250 = weather_data_72T250.columns
    weather_columns = weather_columns + existing_columns_72T250.tolist()

    # Convert columns to numeric types
    for col in existing_columns_72T250:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # Restore the original order
    result_data = result_data.sort_values("original_index").drop(
        columns=["original_index"]
    )

    return result_data, weather_columns


if __name__ == "__main__":
    data = pd.read_csv("processed_data.csv")
    # data.columns = ["Serial", "Power(mW)"]
    result_data = openWeather(data)[0]
    print(result_data.head())
    print(result_data.columns)
    print(result_data[result_data["DeviceID"].between(15, 17)].head())
    # 缺失值數量
    print("缺失值數量")
    print(result_data.isnull().sum())
    # 輸出csv
    result_data.to_csv("open_weather.csv", index=False)
