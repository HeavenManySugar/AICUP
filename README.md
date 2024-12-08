# AICUP

[Training Documentation](training.md)

# Solar Power Prediction Model
This project aims to predict solar power generation using machine learning techniques. It leverages historical data and meteorological information to forecast solar power output accurately.

## Table of Contents
+ Overview
+ Project Structure
+ Functionality
+ Installation
+ Usage
+ Dependencies
+ Notes

## Overview
The Solar Power Prediction Model processes input data, performs feature engineering, and utilizes pre-trained machine learning models to predict solar power output. The main script, main.py, orchestrates the data processing and prediction workflow.

## Project Structure
The workspace contains the following key files and directories:

+ main.py: The main script to run the prediction model.
+ processed_data.csv: The dataset containing processed historical data.
+ Pre-trained models (*.joblib files):
  + main_model_11281346.joblib
  + pressure_model.joblib
  + wind_speed_model.joblib
  + temperature_model.joblib
  + sunlight_model.joblib
  + humidity_model.joblib
+ openWeather.py: A module for handling weather-related data.
+ Other scripts and data files supporting the model.

## Functionality
The model operates through the following steps:

1. Data Preparation
+ Input Data: Reads input data from a CSV file (e.g., `upload(no answer).csv`).
+ Parsing Serial Numbers: Extracts date, time, and device ID from the Serial column.
+ Datetime Features: Converts serial numbers to datetime objects and computes features like day_of_year, hour, and minute.
2. Weather Data Integration
+ Historical Weather Data: Merges input data with historical weather data from `processed_data.csv` for specific timestamps (e.g., 08:50 AM).
+ Feature Engineering: Incorporates lag features using weather data from previous timestamps.
3. Weather Parameter Prediction
+ Pre-trained Models: Uses pre-trained models to predict weather parameters:
  + Pressure (`pressure_model.joblib`)
  + Wind Speed (`wind_speed_model.joblib`)
  + Temperature (`temperature_model.joblib`)
  + Sunlight (`sunlight_model.joblib`)
  + Humidity (`humidity_model.joblib`)
+ Prediction Application: Enhances input features with predicted weather data.
4. Solar Power Prediction
+ Main Model: Loads the main pre-trained model (`main_model_11281346.joblib`).
+ Feature Selection: Selects relevant features, including time variables, device ID, weather predictions, and historical weather data.
+ Generating Predictions: Predicts solar power output using the selected features.
+ Post-processing: Ensures predictions are non-negative and appropriately rounded.
5. Evaluation (Optional)
+ Error Metrics: If actual power data is available, computes metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
+ Scoring: Calculates a score based on the absolute differences between actual and predicted values.
6. Output Generation
+ Predictions CSV: Saves predictions in `predictions.csv` without headers.
+ Console Output: Prints predictions and evaluation metrics to the console.
## Installation
1. Clone the Repository
2. Create a Virtual Environment (Optional)
3. Install Dependencies
Install required Python packages:
`pip install pandas numpy scikit-learn joblib catboost`

## Usage
1. Prepare Input Data
+ Ensure your input CSV file (e.g., `upload(no answer).csv`) is correctly formatted with a Serial column.
+ Place the input CSV file in the project directory.
2. Run the Prediction Script
+ Execute the main script:
`python main.py`

3. View Results
+ Predictions are saved in `predictions.csv`.
+ If actual power data is available in `processed_data.csv`, evaluation metrics are displayed.

## Dependencies

The project requires the Python packages listed in `requirements.txt`  
Install them using:

```sh
pip install -r requirements.txt
```

## Notes
+ Data Files: Ensure all necessary data files, such as `processed_data.csv` and input CSV files, are present in the project directory.
+ Pre-trained Models: Place all pre-trained model files (`*.joblib`) in the same directory as `main.py`.
Output Format: The `predictions.csv` file contains two columns: 序號 (Serial number) and 答案 (Predicted power output).
+ Error Handling: The script includes error handling for missing or NaN values.
+ Customization: Adjust file paths and model names in `main.py` if your file structure differs.
## Conclusion
The Solar Power Prediction Model provides a robust framework for forecasting solar power generation using machine learning. By integrating weather data and historical patterns, it aims to deliver accurate predictions to support energy planning and management.