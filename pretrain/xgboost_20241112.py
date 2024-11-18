# %%
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import os

# 設定LSTM往前看的筆數和預測筆數
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48  # 預測筆數

# 載入訓練資料
SourceData = pd.read_csv("processed_data.csv")
# 選擇要留下來的資料欄位(發電量)
target = ["Power(mW)"]
AllOutPut = SourceData[target].values

X_train = []
y_train = []

# 設定每i-12筆資料(X_train)就對應到第i筆資料(y_train)
for i in range(LookBackNum, len(AllOutPut)):
    X_train.append(AllOutPut[i - LookBackNum : i, 0])
    y_train.append(AllOutPut[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# No reshaping needed for XGBoost

# Split the data into training and validation sets
X_train_data, X_val, y_train_data, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# %%
# ============================建置&訓練模型============================
# 建置XGBoost模型

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Initialize the model
xgb_model = XGBRegressor(objective="reg:squarederror")

# Perform grid search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
)
grid_search.fit(X_train_data, y_train_data)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model on the validation set
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print("Validation Mean Squared Error:", mse)

# 保存模型
from datetime import datetime

NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
best_model.save_model("WeatherXGBoost_" + NowDateTime + ".json")
print("Model Saved")

# %%
# ============================預測數據============================

# 載入模型
best_model = XGBRegressor()
best_model.load_model("WeatherXGBoost_2024-11-12T17_24_25Z.json")

# 載入測試資料
DataName = os.getcwd() + r"/ExampleTestData/upload.csv"
SourceData = pd.read_csv(DataName, encoding="utf-8")
target = ["序號"]
EXquestion = SourceData[target].values

inputs = []  # 存放參考資料
PredictOutput = []  # 存放預測值

count = 0
while count < len(EXquestion):
    print("count : ", count)
    LocationCode = int(EXquestion[count])
    strLocationCode = str(LocationCode)[-2:]
    if LocationCode < 10:
        strLocationCode = "0" + LocationCode

    DataName = (
        os.getcwd()
        + "/ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_"
        + strLocationCode
        + ".csv"
    )
    SourceData = pd.read_csv(DataName, encoding="utf-8")
    ReferTitle = SourceData[["Serial"]].values
    ReferData = SourceData[["Power(mW)"]].values

    inputs = []  # 重置存放參考資料

    # 找到相同的一天，把12個資料都加進inputs
    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]:
            inputs.append(ReferData[DaysCount][0])

    inputs = np.array(inputs)

    # Prepare test data for XGBoost
    X_test = []
    for i in range(len(inputs) - LookBackNum + 1):
        X_test.append(inputs[i : i + LookBackNum])

    X_test = np.array(X_test)

    # Predict using the XGBoost model
    predicted = best_model.predict(X_test)
    # Set negative predictions to zero
    predicted[predicted < 0] = 0
    PredictOutput.extend(predicted.tolist())

    # Adjust count to move to the next set
    count += 48

# 寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame(PredictOutput, columns=["答案"])

# 將 DataFrame 寫入 CSV 檔案
df.to_csv("output.csv", index=False)
print("Output CSV File Saved")

# %%
