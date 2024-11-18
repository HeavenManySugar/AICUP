import os
from io import StringIO
import pandas as pd


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

    weather_data = _openWeatherCSV("C0Z100")

    # ensure both yyyymmddhh columns are of the same type
    data["yyyymmddhh"] = pd.to_numeric(data["yyyymmddhh"], errors="coerce").astype(
        "Int64"
    )

    # find the closest yyyymmddhh to merge
    filtered_data = data[data["DeviceID"].between(1, 14)]
    merged_data = pd.merge_asof(
        filtered_data.sort_values("yyyymmddhh"),
        weather_data.sort_values("yyyymmddhh"),
        on="yyyymmddhh",
        direction="nearest",
    )
    weather_columns = [
        "SS01",
        "CD11",
        "TX01",
        "TS01",
        "RH01",
        "WD01",
        "PP01",
    ]
    existing_columns = [col for col in weather_columns if col in weather_data.columns]
    merged_data = merged_data[["yyyymmddhh"] + existing_columns]
    result_data = data.copy()
    for col in existing_columns:
        result_data.loc[result_data["DeviceID"].between(1, 14), col] = merged_data[
            col
        ].values

    # Convert columns to numeric types
    for col in existing_columns:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find 466990
    weather_data = _openWeatherCSV("466990")
    # find the closest yyyymmddhh to merge
    filtered_data = data[data["DeviceID"].between(15, 17)]
    merged_data = pd.merge_asof(
        filtered_data.sort_values("yyyymmddhh"),
        weather_data.sort_values("yyyymmddhh"),
        on="yyyymmddhh",
        direction="nearest",
    )
    existing_columns = [col for col in weather_columns if col in weather_data.columns]
    merged_data = merged_data[["yyyymmddhh"] + existing_columns]
    result_data = data.copy()
    for col in existing_columns:
        result_data.loc[result_data["DeviceID"].between(15, 17), col] = merged_data[
            col
        ].values

    # Convert columns to numeric types
    for col in existing_columns:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    return result_data, weather_columns


if __name__ == "__main__":
    data = pd.read_csv("processed_data.csv")
    result_data = openWeather(data)[0]
    print(result_data.head())
    print(result_data.columns)
    print(result_data[result_data["DeviceID"].between(15, 17)].head())
