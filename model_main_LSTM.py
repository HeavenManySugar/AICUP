from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
import joblib
from openWeather import openWeather

# 設定LSTM往前看的筆數和預測筆數
LookBackNum = 12  # LSTM往前看的筆數
ForecastNum = 48  # 預測筆數

# 載入訓練資料
DataName = os.getcwd() + "/processed_data.csv"
SourceData = pd.read_csv(DataName, encoding="utf-8")

SourceData["Year"] = SourceData["Serial"].astype(str).str[:4].astype(int)
SourceData["Month"] = SourceData["Serial"].astype(str).str[4:6].astype(int)
SourceData["Day"] = SourceData["Serial"].astype(str).str[6:8].astype(int)
SourceData["hhmm"] = SourceData["Serial"].astype(str).str[8:12].astype(int)
SourceData["DeviceID"] = SourceData["Serial"].astype(str).str[12:14].astype(int)

SourceData, weather_columns = openWeather(SourceData)
X = SourceData[["Year", "Month", "Day", "hhmm", "DeviceID", *weather_columns]]

humidity_model = "humidity_model.joblib"
pressure_model = "pressure_model.joblib"
sunlight_model = "sunlight_model.joblib"
temperature_model = "temperature_model.joblib"
wind_speed_model = "wind_speed_model.joblib"

SourceData["Pressure(hpa)"] = joblib.load(pressure_model).predict(X)
SourceData["WindSpeed(m/s)"] = joblib.load(wind_speed_model).predict(X)
SourceData["Temperature(°C)"] = joblib.load(temperature_model).predict(X)
SourceData["Sunlight(Lux)_FIX"] = joblib.load(sunlight_model).predict(X)
SourceData["Humidity(%)"] = joblib.load(humidity_model).predict(X)

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

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Define the model
regressor = Sequential()

regressor.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=64, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=32))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units=1))

# Compile the model
optimizer = Adam(learning_rate=0.001)
regressor.compile(optimizer=optimizer, loss="mean_squared_error")

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss"
)

# 開始訓練
regressor.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
)

# 保存模型
from datetime import datetime

NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
regressor.save("WheatherLSTM_" + NowDateTime + ".keras")
print("Model Saved")
