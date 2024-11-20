import joblib

# compress joblib model
model = joblib.load("FIX_sunlight_model.joblib")
joblib.dump(model, "FIX_sunlight_model_compressed.joblib", compress="lzma")
