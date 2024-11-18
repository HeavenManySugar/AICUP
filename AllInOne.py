import mainOne
import mainOne_CatBoost
import mainOne_LightGBM
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == "__main__":
    csv_path = "upload(no answer).csv"
    data = pd.read_csv(csv_path)

    y_pred_xgboost = mainOne.PowerPredict("one_model.joblib", data)
    data = pd.read_csv(csv_path)
    y_pred_lightgbm = mainOne_LightGBM.PowerPredict("one_model_LightGBM.joblib", data)
    data = pd.read_csv(csv_path)
    y_pred_catboost = mainOne_CatBoost.PowerPredict("one_model_CatBoost.joblib", data)

    # Average the predictions from the three models
    y_pred = 0.39 * y_pred_xgboost + 0.02 * y_pred_lightgbm + 0.59 * y_pred_catboost
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.round(y_pred, 2)

    try:
        processed_data = pd.read_csv("processed_data.csv")
        # Ensure the Serial columns are of the same type
        processed_data["Serial"] = processed_data["Serial"].astype(str)
        print(processed_data)
        data["Serial"] = data["Serial"].astype(str)
        print(data)
        y_test = processed_data.loc[
            processed_data["Serial"].isin(data["Serial"]), "Power(mW)"
        ].values

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
