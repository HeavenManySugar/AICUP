import glob
import pandas as pd

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

print(df.head())

df.to_csv("processed_data.csv", index=False)
