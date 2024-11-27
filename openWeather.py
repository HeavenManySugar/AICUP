import pandas as pd
import matplotlib.pyplot as plt
import pvlib


def pvWeather(data):
    # 定義花蓮的地理位置
    locations = [
        [23.5358, 121.3240],
        [23.5359, 121.3241],
        [23.5359, 121.3242],
        [23.5358, 121.3240],
        [23.5358, 121.3241],
        [23.5358, 121.3240],
        [23.5358, 121.3240],
        [23.5359, 121.3242],
        [23.5358, 121.3240],
        [23.5358, 121.3240],
        [23.5359, 121.3241],
        [23.5359, 121.3241],
        [23.5352, 121.3222],
        [23.5352, 121.3222],
        [24.0033, 121.3702],
        [24.0032, 121.3702],
        [23.9751, 121.6132],
    ]
    tz = "Asia/Taipei"

    # 保存原始索引並複製資料
    data = data.copy()
    data["original_index"] = data.index

    # 轉換時間和 DeviceID
    data["DateTime"] = pd.to_datetime(
        data["Serial"].astype(str).str[:12], format="%Y%m%d%H%M"
    ).dt.tz_localize(tz)
    data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)

    # 先根據每個設備找出時間範圍
    start_time = data["DateTime"].min()
    end_time = data["DateTime"].max()

    # 建立晴空輻射量資料
    clearsky_list = []  # 使用列表來收集數據
    for i in range(1, len(locations) + 1):
        latitude, longitude = locations[i - 1]
        location = pvlib.location.Location(latitude, longitude, tz=tz)

        times = pd.date_range(start_time, end_time, freq="10min")

        clearsky = location.get_clearsky(times)
        clearsky.index.name = "DateTime"
        clearsky["DeviceID"] = i

        clearsky_list.append(clearsky.reset_index())

    # 合併所有 clearsky 數據
    clearsky_data = pd.concat(clearsky_list, ignore_index=True)

    # 確保數據類型一致
    data["DeviceID"] = data["DeviceID"].astype(int)
    clearsky_data["DeviceID"] = clearsky_data["DeviceID"].astype(int)

    # 分別處理每個設備的數據
    result_list = []

    for device_id in data["DeviceID"].unique():
        # 篩選當前設備的數據
        device_data = data[data["DeviceID"] == device_id].copy()
        device_clearsky = clearsky_data[clearsky_data["DeviceID"] == device_id].copy()

        # 確保時間排序
        device_data = device_data.sort_values("DateTime")
        device_clearsky = device_clearsky.sort_values("DateTime")

        # 進行合併
        merged = pd.merge_asof(
            device_data,
            device_clearsky,
            on="DateTime",
            direction="nearest",
            tolerance=pd.Timedelta("10min"),
        )

        result_list.append(merged)

    # 合併所有結果
    result_data = pd.concat(result_list)

    # 恢復原始順序
    result_data = result_data.sort_values("original_index").drop(
        columns=["original_index"]
    )

    print("合併後的資料筆數:", len(result_data))
    print("原始資料筆數:", len(data))
    print("樣本數據:")

    result_data = result_data.drop(
        columns=["DeviceID_x", "DeviceID_y"], errors="ignore"
    )
    print(result_data.head())

    return result_data


def openWeather(data):
    data = pvWeather(data)
    data["DateTime"] = pd.to_datetime(
        data["Serial"].astype(str).str[:12], format="%Y%m%d%H%M"
    )
    data["original_index"] = data.index

    weather_data_C0Z100 = pd.read_csv("C0Z100_2024.csv")
    weather_data_C0Z100.drop(
        columns=["StnPres", "Tx", "RH", "WS", "WD", "SunShine"], inplace=True
    )
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
    # Convert columns to numeric types
    for col in existing_columns_C0Z100:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find 466990
    weather_data_466990 = pd.read_csv("466990_2024.csv")
    weather_data_466990.drop(
        columns=[
            "StnPres",
            "Tx",
            "RH",
            "WS",
            "WD",
            "Visb",
            "Cloud Amount",
            "TxSoil0cm",
            "TxSoil5cm",
            "TxSoil10cm",
            "TxSoil20cm",
            "TxSoil30cm",
            "TxSoil50cm",
            "TxSoil100cm",
        ],
        inplace=True,
    )
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
    weather_data_72T250.drop(columns=["EvapA"], inplace=True)
    # 線性插值
    weather_data_72T250.interpolate(method="linear", inplace=True)
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

    # find C0T9D0
    weather_data_C0T9D0 = pd.read_csv("C0T9D0_2024.csv")
    weather_data_C0T9D0.drop(columns=["SunShine"], inplace=True)
    weather_data_C0T9D0.set_index("Unnamed: 0", inplace=True)
    weather_data_C0T9D0.index.name = "DateTime"
    weather_data_C0T9D0.index = pd.to_datetime(weather_data_C0T9D0.index)
    weather_data_C0T9D0 = weather_data_C0T9D0.add_suffix("_C0T9D0")
    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        result_data.sort_values("DateTime"),
        weather_data_C0T9D0.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    existing_columns_C0T9D0 = weather_data_C0T9D0.columns
    weather_columns = weather_columns + existing_columns_C0T9D0.tolist()
    # Convert columns to numeric types
    for col in existing_columns_C0T9D0:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find C0T900
    weather_data_C0T900 = pd.read_csv("C0T900_2024.csv")
    weather_data_C0T900.drop(
        columns=["StnPres", "Tx", "RH", "WS", "WD", "SunShine"], inplace=True
    )
    weather_data_C0T900.set_index("Unnamed: 0", inplace=True)
    weather_data_C0T900.index.name = "DateTime"
    weather_data_C0T900.index = pd.to_datetime(weather_data_C0T900.index)
    weather_data_C0T900 = weather_data_C0T900.add_suffix("_C0T900")
    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        result_data.sort_values("DateTime"),
        weather_data_C0T900.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    existing_columns_C0T900 = weather_data_C0T900.columns
    weather_columns = weather_columns + existing_columns_C0T900.tolist()
    # Convert columns to numeric types
    for col in existing_columns_C0T900:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find C0t960
    weather_data_C0T960 = pd.read_csv("C0T960_2024.csv")
    weather_data_C0T960.drop(
        columns=["StnPres", "Tx", "RH", "WS", "WD", "SunShine"], inplace=True
    )
    weather_data_C0T960.set_index("Unnamed: 0", inplace=True)
    weather_data_C0T960.index.name = "DateTime"
    weather_data_C0T960.index = pd.to_datetime(weather_data_C0T960.index)
    weather_data_C0T960 = weather_data_C0T960.add_suffix("_C0T960")
    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        result_data.sort_values("DateTime"),
        weather_data_C0T960.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    existing_columns_C0T960 = weather_data_C0T960.columns
    weather_columns = weather_columns + existing_columns_C0T960.tolist()
    # Convert columns to numeric types
    for col in existing_columns_C0T960:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find C0Z020
    weather_data_C0Z020 = pd.read_csv("C0Z020_2024.csv")
    weather_data_C0Z020.drop(
        columns=["StnPres", "Tx", "RH", "WS", "WD", "SunShine"], inplace=True
    )
    weather_data_C0Z020.set_index("Unnamed: 0", inplace=True)
    weather_data_C0Z020.index.name = "DateTime"
    weather_data_C0Z020.index = pd.to_datetime(weather_data_C0Z020.index)
    weather_data_C0Z020 = weather_data_C0Z020.add_suffix("_C0Z020")
    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        result_data.sort_values("DateTime"),
        weather_data_C0Z020.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    existing_columns_C0Z020 = weather_data_C0Z020.columns
    weather_columns = weather_columns + existing_columns_C0Z020.tolist()
    # Convert columns to numeric types
    for col in existing_columns_C0Z020:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # find C0Z290
    weather_data_C0Z290 = pd.read_csv("C0Z290_2024.csv")
    weather_data_C0Z290.drop(columns=["SunShine"], inplace=True)
    weather_data_C0Z290.set_index("Unnamed: 0", inplace=True)
    weather_data_C0Z290.index.name = "DateTime"
    weather_data_C0Z290.index = pd.to_datetime(weather_data_C0Z290.index)
    weather_data_C0Z290 = weather_data_C0Z290.add_suffix("_C0Z290")
    # find the closest DateTime to merge
    result_data = pd.merge_asof(
        result_data.sort_values("DateTime"),
        weather_data_C0Z290.sort_index(),
        on="DateTime",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )
    existing_columns_C0Z290 = weather_data_C0Z290.columns
    weather_columns = weather_columns + existing_columns_C0Z290.tolist()
    # Convert columns to numeric types
    for col in existing_columns_C0Z290:
        result_data[col] = pd.to_numeric(result_data[col], errors="coerce")

    # Restore the original order
    result_data = result_data.sort_values("original_index").drop(
        columns=["original_index"]
    )

    return result_data, weather_columns


if __name__ == "__main__":
    data = pd.read_csv("processed_data.csv")
    data = pvWeather(data)
    # data.columns = ["Serial", "Power(mW)"]
    result_data = openWeather(data)[0]
    print(result_data.head())
    # print(result_data.columns)
    # 缺失值數量
    # print("缺失值數量")
    # print(result_data.isnull().sum())
    # plt畫出缺失值比例
    plt.figure(figsize=(10, 5))
    result_data.isnull().mean().plot(kind="bar")
    plt.title("Missing values")
    plt.show()
    # 輸出csv
    result_data.to_csv("open_weather.csv", index=False)
