import glob
import pandas as pd

# Get all CSV files in the directory
csv_files = glob.glob("ExampleTrainData(AVG)/*.csv") + glob.glob(
    "ExampleTrainData(IncompleteAVG)/*.csv"
)

# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in csv_files]

# Concatenate all CSV files
for i in range(len(df_list)):
    df_list[i].columns = [
        "Serial",
        "WindSpeed(m/s)",
        "Pressure(hpa)",
        "Temperature(°C)",
        "Humidity(%)",
        "Sunlight(Lux)",
        "Power(mW)",
    ]

data = pd.concat(df_list, ignore_index=True)

# output the processed data
data.to_csv("processed_data.csv", index=False)
