import joblib

model = joblib.load("pressure_model.joblib")

joblib.dump(model, "pressure_model.joblib", compress="zlib")
