import glob
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Get all CSV files in the directory
csv_files = glob.glob("36_TrainingData/*.csv")

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]

for i in range(len(df_list)):
    df_list[i]["DateTime"] = pd.to_datetime(df_list[i]["DateTime"])
df = pd.concat(df_list, ignore_index=True)

# Remove rows with any empty (NaN) values
df = df.dropna()

# Ensure LocationCode is an integer
df["LocationCode"] = df["LocationCode"].astype(int)

year = df["DateTime"].dt.year
month = df["DateTime"].dt.month
date = df["DateTime"].dt.day
time = df["DateTime"].dt.strftime("%H%M")

df["Serial"] = (
    year.astype(str).str.zfill(4)
    + month.astype(str).str.zfill(2)
    + date.astype(str).str.zfill(2)
    + time.astype(str).str.zfill(4)
    + df["LocationCode"].astype(str).str.zfill(2)
)

df = df.drop(columns=["LocationCode", "DateTime"])
df = df.reindex(columns=["Serial"] + [col for col in df.columns if col != "Serial"])

# 刪除重複資料，透過取平均
df = df.groupby("Serial").mean().round(2).reset_index()

df.to_csv("processed_data_1min.csv", index=False)

sunlight = df["Sunlight(Lux)"]
power = df["Power(mW)"]

# max_sunlight_value = sunlight.max()
max_sunlight_value = 117758.2
mask = sunlight < max_sunlight_value
sunlight_valid = sunlight[mask]
power_valid = power[mask]

model = joblib.load("FIX_sunlight_model.joblib")
data = df.copy()
data["Month"] = data["Serial"].astype(str).str[4:6].astype(int)
data["hhmm"] = data["Serial"].astype(str).str[8:12].astype(int)
data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)

X = data[
    [
        "Month",
        "hhmm",
        "DeviceID",
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Power(mW)",
    ]
]

# max_sunlight_value = 117758.2
# mask = data["Sunlight(Lux)"] < max_sunlight_value
# X_saturated = X.loc[~mask]

# estimated_lux = model.predict(X_saturated)
# estimated_lux_full = df["Sunlight(Lux)"].copy()
# estimated_lux_full[~mask] = estimated_lux
# df["Sunlight(Lux)_FIX"] = df["Sunlight(Lux)"].where(
#     df["Sunlight(Lux)"] < max_sunlight_value, estimated_lux_full
# )

# 預測全部的Sunlight(Lux)
estimated_lux = model.predict(X)
df["Sunlight(Lux)_FIX"] = estimated_lux


plt.figure(figsize=(10, 6))
plt.plot(df["Sunlight(Lux)"], label="Original Lux", alpha=0.7)
plt.plot(
    df["Sunlight(Lux)_FIX"], label="Compensated Lux", linestyle="--", color="orange"
)
plt.legend()
plt.title("Comparison before and after illumination correction")
plt.xlabel("Sample number")
plt.ylabel("Sunlight(Lux)")
# plt.show()

# 將Serial變成["LocationCode", "DateTime"]
df["Serial"] = df["Serial"].astype(str)
df["LocationCode"] = df["Serial"].str[-2:].astype(int)
df["DateTime"] = pd.to_datetime(df["Serial"].astype(str).str[:12], format="%Y%m%d%H%M")
df = df.drop(columns=["Serial"])
# 將df按照LocationCode分別處理
df_list = [df[df["LocationCode"] == i] for i in df["LocationCode"].unique()]
for i in range(len(df_list)):
    # 10min為一次平均
    df_list[i] = (
        df_list[i].groupby(pd.Grouper(key="DateTime", freq="10min")).mean().round(2)
    )
    df_list[i] = df_list[i].reset_index()
    df_list[i] = df_list[i].dropna()
    df_list[i]["LocationCode"] = df_list[i]["LocationCode"].astype(int)
    # 將DateTime轉換成Serial
    year = df_list[i]["DateTime"].dt.year
    month = df_list[i]["DateTime"].dt.month
    date = df_list[i]["DateTime"].dt.day
    time = df_list[i]["DateTime"].dt.strftime("%H%M")
    df_list[i]["Serial"] = (
        year.astype(str).str.zfill(4)
        + month.astype(str).str.zfill(2)
        + date.astype(str).str.zfill(2)
        + time.astype(str).str.zfill(4)
        + df_list[i]["LocationCode"].astype(str).str.zfill(2)
    )
    df_list[i] = df_list[i].drop(columns=["LocationCode", "DateTime"])
    df_list[i] = df_list[i].reindex(
        columns=["Serial"] + [col for col in df_list[i].columns if col != "Serial"]
    )
df = pd.concat(df_list, ignore_index=True)

df.to_csv("processed_data.csv", index=False)

plt.show()
