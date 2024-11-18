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
    df_list[i] = (
        df_list[i].resample("10min", on="DateTime").mean().round(2).reset_index()
    )
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

sunlight = df["Sunlight(Lux)"]
power = df["Power(mW)"]

# max_sunlight_value = sunlight.max()
max_sunlight_value = 117758.2
mask = sunlight < max_sunlight_value
sunlight_valid = sunlight[mask]
power_valid = power[mask]

model = joblib.load("FIX_sunlight_model.joblib")
data = df.copy()
data["Year"] = data["Serial"].astype(str).str[:4].astype(int)
data["Month"] = data["Serial"].astype(str).str[4:6].astype(int)
data["Day"] = data["Serial"].astype(str).str[6:8].astype(int)
data["hhmm"] = data["Serial"].astype(str).str[8:12].astype(int)
data["DeviceID"] = data["Serial"].astype(str).str[12:14].astype(int)

X = data[
    [
        "Year",
        "Month",
        "Day",
        "hhmm",
        "DeviceID",
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Power(mW)",
    ]
]

max_sunlight_value = 117758.2
mask = data["Sunlight(Lux)"] < max_sunlight_value
X_saturated = X.loc[~mask]

estimated_lux = model.predict(X_saturated)
estimated_lux_full = df["Sunlight(Lux)"].copy()
estimated_lux_full[~mask] = estimated_lux
df["Sunlight(Lux)_FIX"] = df["Sunlight(Lux)"].where(
    df["Sunlight(Lux)"] < max_sunlight_value, estimated_lux_full
)


plt.figure(figsize=(10, 6))
plt.plot(df["Sunlight(Lux)"], label="Original Lux", alpha=0.7)
plt.plot(
    df["Sunlight(Lux)_FIX"], label="Compensated Lux", linestyle="--", color="orange"
)
plt.legend()
plt.title("Comparison before and after illumination correction")
plt.xlabel("Sample number")
plt.ylabel("Sunlight(Lux)")
plt.show()


print(df.head())

df.to_csv("processed_data.csv", index=False)
