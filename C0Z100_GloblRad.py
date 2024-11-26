import pandas as pd
import os


def loadC0Z100_GloblRad():
    # Define the directory containing the CSV files
    directory = "C0Z100_GloblRad"
    # Initialize an empty list to store DataFrames
    df_list = []

    # Loop through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)

            # Extract the date from the filename
            date_str = (
                filename.split("-")[1]
                + filename.split("-")[2]
                + filename.split("-")[3].split(".")[0]
            )

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, skiprows=1)

            # Add the date to the DataFrame
            df["yyyymmddhh"] = date_str + df["ObsTime"].astype(str).str.zfill(2)
            df["yyyymmddhh"] = pd.to_numeric(df["yyyymmddhh"], errors="coerce").astype(
                "Int64"
            )

            df.drop(
                columns=[
                    "ObsTime",
                    "StnPres",
                    "Temperature",
                    "RH",
                    "WS",
                    "WD",
                    "WSGust",
                    "WDGust",
                    "Precp",
                    "SeaPres",
                    "Td dew point",
                    "PrecpHour",
                    "SunShine",
                    "Visb",
                    "UVI",
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
                errors="ignore",
            )

            # Append the DataFrame to the list
            df_list.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    df = loadC0Z100_GloblRad()
    print(df.head())
