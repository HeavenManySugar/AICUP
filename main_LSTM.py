from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from openWeather import openWeather


# 載入模型
regressor = load_model("WheatherLSTM_2024-11-20T14_35_32Z.h5")

# 載入測試資料
DataName = "upload.csv"
SourceData = pd.read_csv(DataName, encoding="utf-8")
target = "序號"

# rename 序號 to Serial
SourceData.rename(columns={target: "Serial"}, inplace=True)

PredictOutput = []  # 存放預測值


SourceData["Year"] = SourceData["Serial"].astype(str).str[:4].astype(int)
SourceData["Month"] = SourceData["Serial"].astype(str).str[4:6].astype(int)
SourceData["Day"] = SourceData["Serial"].astype(str).str[6:8].astype(int)
SourceData["hhmm"] = SourceData["Serial"].astype(str).str[8:12].astype(int)
SourceData["DeviceID"] = SourceData["Serial"].astype(str).str[12:14].astype(int)

SourceData, weather_columns = openWeather(SourceData)
X = SourceData[["Year", "Month", "Day", "hhmm", "DeviceID", *weather_columns]]

humidity_model = "humidity_model.joblib"
pressure_model = "pressure_model.joblib"
sunlight_model = "sunlight_model.joblib"
temperature_model = "temperature_model.joblib"
wind_speed_model = "wind_speed_model.joblib"


SourceData["Pressure(hpa)"] = joblib.load(pressure_model).predict(X)
SourceData["WindSpeed(m/s)"] = joblib.load(wind_speed_model).predict(X)
SourceData["Temperature(°C)"] = joblib.load(temperature_model).predict(X)
SourceData["Sunlight(Lux)_FIX"] = joblib.load(sunlight_model).predict(X)
SourceData["Humidity(%)"] = joblib.load(humidity_model).predict(X)

selected_features = [
    "Year",
    "Month",
    "Day",
    "hhmm",
    "DeviceID",
    "Pressure(hpa)",
    "WindSpeed(m/s)",
    "Temperature(°C)",
    "Sunlight(Lux)_FIX",
    "Humidity(%)",
]

# 取得參考資料
X = SourceData[selected_features]

# 將參考資料轉換為3D張量
X = X.values.reshape(X.shape[0], 1, X.shape[1])

# 預測
y_pred = regressor.predict(X)
PredictOutput = y_pred.flatten()

# 將預測結果轉換為陣列
PredictOutput = np.array(PredictOutput)

# 定義 EXquestion 為序號的值
EXquestion = SourceData["Serial"].values.reshape(-1, 1)

# 寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame({"序號": EXquestion.flatten(), "答案": PredictOutput})

try:
    actual_values = []
    DataName = "processed_data.csv"
    SourceData = pd.read_csv(DataName, encoding="utf-8")
    for question in EXquestion:
        actual_value = SourceData.loc[
            SourceData["Serial"] == question[0], "Power(mW)"
        ].values
        if len(actual_value) > 0:
            actual_values.append(actual_value[0])

    mse = mean_squared_error(actual_values, PredictOutput)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_values, PredictOutput)
    r2 = r2_score(actual_values, PredictOutput)
    print(f"均方誤差(MSE): {mse:.4f}")
    print(f"均方根誤差(RMSE): {rmse:.4f}")
    print(f"平均絕對誤差(MAE): {mae:.4f}")
    print(f"R²分數: {r2:.4f}")
    # R²解讀：

    # 1.0：完美預測
    # 0：模型只是預測平均值
    # 負值：模型比單純預測平均值還差

    score = sum(abs(actual_values - PredictOutput))
    print(f"Score: {score}")

    PredictOutput = np.array(PredictOutput)
    actual_values = np.array(actual_values)
    errors = PredictOutput - actual_values

    plt.figure(figsize=(10, 6))
    plt.scatter(
        actual_values, PredictOutput, c=np.abs(errors), cmap="viridis", alpha=0.7
    )
    plt.colorbar(label="Absolute Error")
    plt.xlabel("Actual Power (mW)")
    plt.ylabel("Predicted Power (mW)")
    plt.title("Actual vs Predicted Power with Error Visualization")
    plt.plot(
        [actual_values.min(), actual_values.max()],
        [actual_values.min(), actual_values.max()],
        "r--",
        lw=2,
    )
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, errors, alpha=0.7)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Actual Power (mW)")
    plt.ylabel("Errors")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.show()

except:
    pass

# 將 DataFrame 寫入 CSV 檔案
df.to_csv("predictions.csv", index=False, encoding="utf-8", header=False)
print("Output CSV File Saved")
