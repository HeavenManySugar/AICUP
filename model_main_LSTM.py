from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
import joblib
from openWeather import openWeather
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime


def scale(train_features, test_features):
    scaler = MinMaxScaler()
    data = pd.concat([train_features, test_features])
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=test_features.keys())
    train_features = scaled_data.iloc[0 : train_features.shape[0]]
    test_features = scaled_data.iloc[train_features.shape[0] :]

    return train_features, test_features


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

selected_features = [
    "Year",
    "Month",
    "Day",
    "hhmm",
    "DeviceID",
    "Pressure(hpa)",
    "WindSpeed(m/s)",
    "Temperature(°C)",
    "Sunlight(Lux)_FIX",
    "Humidity(%)",
]

train_features = SourceData.drop("Power(mW)", axis=1)
y = SourceData["Power(mW)"]
X_test = SourceData
X_test = X_test[selected_features]
X_train = train_features[selected_features]

X_train, X_test = scale(X_train, X_test)

X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=333
)

model = Sequential()
model.add(
    LSTM(200, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2]))
)
model.add(Dense(1, activation="relu"))
model.compile(loss="mse", optimizer="adam")


# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss"
)

# 開始訓練
model.fit(
    X_train,
    y_train,
    epochs=1100,
    batch_size=100,
    validation_data=(X_val, y_val),
    verbose=2,
    shuffle=False,
    callbacks=[early_stopping, model_checkpoint],
)

# 保存模型
NowDateTime = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
model.save("WheatherLSTM_" + NowDateTime + ".h5")
print("Model Saved")
