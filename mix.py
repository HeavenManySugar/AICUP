import main
import main2

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


if __name__ == "__main__":
    csv_path = "upload(no answer).csv"
    data = pd.read_csv(csv_path)
    # rename columns
    data.columns = ["Serial", "Power(mW)"]

    y_pred1 = main.PowerPredict("main_model.joblib", pd.read_csv(csv_path))
    y_pred1 = np.maximum(y_pred1, 0)
    y_pred1 = np.round(y_pred1, 2)

    y_pred2 = main2.PowerPredict(
        "main_model2.joblib", "weather_model.joblib", pd.read_csv(csv_path)
    )
    y_pred2 = np.maximum(y_pred2, 0)
    y_pred2 = np.round(y_pred2, 2)

    y_pred = 0.5 * y_pred1 + 0.5 * y_pred2

    try:
        print(y_pred)
        processed_data = pd.read_csv("processed_data.csv")
        y_test = processed_data.loc[processed_data["Serial"].isin(data["Serial"])][
            "Power(mW)"
        ]
        # Handle NaN values
        y_test = y_test.fillna(0)
        y_pred = np.nan_to_num(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        score = sum(abs(y_test - y_pred))
        print(f"Score: {score}")
    except:
        pass

    # Save predictions to CSV
    output = pd.DataFrame({"序號": data["Serial"], "答案": y_pred})
    output.to_csv("predictions.csv", index=False, encoding="utf-8", header=False)
