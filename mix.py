import main
import main2

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


if __name__ == "__main__":
    csv_path = "upload.csv"
    data = pd.read_csv(csv_path)
    # rename columns
    data.columns = ["Serial", "Power(mW)"]

    y_pred1 = main.PowerPredict("main_model.joblib", pd.read_csv(csv_path))

    y_pred2 = main2.PowerPredict(
        "main_model2.joblib", "weather_model.joblib", pd.read_csv(csv_path)
    )

    y_pred = y_pred2
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.round(y_pred, 2)

    try:
        print(y_pred)
        processed_data = pd.read_csv("processed_data.csv")
        data = processed_data.loc[processed_data["Serial"].isin(data["Serial"])]
        y_test = data["Power(mW)"].values

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"均方誤差(MSE): {mse:.4f}")
        print(f"均方根誤差(RMSE): {rmse:.4f}")
        print(f"平均絕對誤差(MAE): {mae:.4f}")
        print(f"R²分數: {r2:.4f}")
        # R²解讀：

        # 1.0：完美預測
        # 0：模型只是預測平均值
        # 負值：模型比單純預測平均值還差

        score = sum(abs(y_test - y_pred))
        print(f"Score: {score}")

        errors = y_pred - y_test

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, c=np.abs(errors), cmap="viridis", alpha=0.7)
        plt.colorbar(label="Absolute Error")
        plt.xlabel("Actual Power (mW)")
        plt.ylabel("Predicted Power (mW)")
        plt.title("Actual vs Predicted Power with Error Visualization")
        plt.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
        )
        plt.tight_layout()

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, errors, alpha=0.7)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Actual Power (mW)")
        plt.ylabel("Errors")
        plt.title("Residual Plot")
        plt.tight_layout()
        plt.show()
    except:
        pass

    # Save predictions to CSV
    output = pd.DataFrame({"序號": data["Serial"], "答案": y_pred})
    output.to_csv("predictions.csv", index=False, encoding="utf-8", header=False)
