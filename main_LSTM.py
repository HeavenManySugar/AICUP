from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd

# 設定LSTM往前看的筆數和預測筆數
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48  # 預測筆數

# 載入模型
regressor = load_model("WheatherLSTM_2024-11-19T17_05_16Z.keras")

# 載入測試資料
DataName = "upload(no answer).csv"
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

    DataName = "processed_data.csv"
    SourceData = pd.read_csv(DataName, encoding="utf-8")
    ReferTitle = SourceData[["Serial"]].values
    ReferData = SourceData[["Power(mW)"]].values

    inputs = []  # 重置存放參考資料

    # 找到相同的一天，把12個資料都加進inputs
    for DaysCount in range(len(ReferTitle)):
        if str(int(ReferTitle[DaysCount]))[:8] == str(int(EXquestion[count]))[:8]:
            inputs = np.append(inputs, ReferData[DaysCount])

    # 用迴圈不斷使新的預測值塞入參考資料，並預測下一筆資料
    for i in range(ForecastNum):
        # print(i)

        # 將新的預測值加入參考資料(用自己的預測值往前看)
        if i > 0:
            inputs = np.append(inputs, PredictOutput[i - 1])

        # 切出新的參考資料12筆(往前看12筆)
        X_test = []
        X_test.append(inputs[0 + i : LookBackNum + i])

        # Reshaping
        NewTest = np.array(X_test)
        NewTest = np.reshape(NewTest, (NewTest.shape[0], NewTest.shape[1], 1))
        predicted = regressor.predict(NewTest)
        PredictOutput.append(round(predicted[0, 0], 2))

    # 每次預測都要預測48個，因此加48個會切到下一天
    # 0~47,48~95,96~143...
    count += 48

# 寫預測結果寫成新的CSV檔案
# 將陣列轉換為 DataFrame
df = pd.DataFrame({"序號": EXquestion.flatten(), "答案": PredictOutput})

try:
    actual_values = []
    for question in EXquestion:
        actual_value = SourceData.loc[
            SourceData["Serial"] == question[0], "Power(mW)"
        ].values
        if len(actual_value) > 0:
            actual_values.append(actual_value[0])
    mae = mean_absolute_error(actual_values, PredictOutput)
    rmse = np.sqrt(mean_squared_error(actual_values, PredictOutput))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    score = sum(abs(np.array(actual_values) - np.array(PredictOutput)))
    print(f"Score: {score}")
except:
    pass

# 將 DataFrame 寫入 CSV 檔案
df.to_csv("predictions.csv", index=False, encoding="utf-8", header=False)
print("Output CSV File Saved")
