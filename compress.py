import joblib

model = joblib.load("sunlight_model.joblib")

joblib.dump(model, "sunlight_model.joblib", compress="lzma")
