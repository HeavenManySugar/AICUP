import pandas as pd


def openWeather(data):
    data["DateTime"] = pd.to_datetime(
        data["Serial"].astype(str).str[:12], format="%Y%m%d%H%M"
    )
    data["original_index"] = data.index

    weather_data_C0Z100 = pd.read_csv("C0Z100_2024.csv")
    # set index to the DateTime column
    weather_data_C0Z100.set_index("Unnamed: 0", inplace=True)
    weather_data_C0Z100.index.name = "DateTime"
    weather_data_C0Z100.index = pd.to_datetime(weather_data_C0Z100.index)
    weather_data_C0Z100 = weather_data_C0Z100.add_suffix("_C0Z100")

    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        data.sort_values("DateTime"),
        weather_data_C0Z100.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    weather_columns = weather_data_C0Z100.columns

    existing_columns_C0Z100 = weather_data_C0Z100.columns
    print(existing_columns_C0Z100)
    # Convert columns to numeric types
    for col in existing_columns_C0Z100:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find 466990
    weather_data_466990 = pd.read_csv("466990_2024.csv")
    weather_data_466990.set_index("Unnamed: 0", inplace=True)
    weather_data_466990.index.name = "DateTime"
    weather_data_466990.index = pd.to_datetime(weather_data_466990.index)
    weather_data_466990 = weather_data_466990.add_suffix("_466990")
    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        result_data.sort_values("DateTime"),
        weather_data_466990.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    existing_columns_466990 = weather_data_466990.columns
    weather_columns = weather_columns.tolist() + existing_columns_466990.tolist()
    # Convert columns to numeric types
    for col in existing_columns_466990:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find 72T250
    weather_data_72T250 = pd.read_csv("72T250_2024.csv")
    weather_data_72T250.set_index("Unnamed: 0", inplace=True)
    weather_data_72T250.index.name = "DateTime"
    weather_data_72T250.index = pd.to_datetime(weather_data_72T250.index)
    weather_data_72T250 = weather_data_72T250.add_suffix("_72T250")
    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        result_data.sort_values("DateTime"),
        weather_data_72T250.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
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
    # 缺失值數量
    print("缺失值數量")
    print(result_data.isnull().sum())
    # 輸出csv
    result_data.to_csv("open_weather.csv", index=False)
