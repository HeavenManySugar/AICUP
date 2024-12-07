# Training the Solar Power Prediction Model

This document provides a step-by-step guide on how to train the Solar Power Prediction Model. The model utilizes machine learning techniques to forecast solar power generation based on historical data and meteorological information.

## Table of Contents

- Overview
- Prerequisites
- Project Structure
- Data Preparation
- Training the Model
  - 1. Install Dependencies
  - 2. Prepare the Data
  - 3. Run the Training Scripts
- Model Evaluation
- Saving and Loading Models
- Notes
- Conclusion

## Overview

The Solar Power Prediction Model is trained using machine learning algorithms like CatBoost and XGBoost. The training process involves:

- Data preprocessing and feature engineering
- Hyperparameter tuning using Optuna
- Training regression models for solar power and weather parameters
- Evaluating model performance

## Prerequisites

- **Python 3.6+**
- **Anaconda or Virtual Environment** (recommended)
- **Optional**: GPU support for faster training with CatBoost

## Project Structure

The workspace includes the following relevant files and directories:

- `model_main.py`: Main script for training the solar power prediction model.
- `processed_data.csv`: Processed historical data used for training.
- Weather model scripts:
  - `modelWindSpeed.py`
  - `modelTemperature.py`
  - `modelHumidity.py`
  - `modelSunlight.py`
  - `modelPressure.py`
- `openWeather.py`: Module for handling weather-related data.
- Pre-trained models (if any): `*.joblib` files
- Other supporting scripts and data files.

## Data Preparation

1. **Collect Data**: Ensure that `processed_data.csv` and other necessary data files are present in the project directory.

2. **Feature Engineering**:

   - **Extract Date and Time Features**: Parse the `Serial` column to extract `Year`, `Month`, `Day`, `Hour`, `Minute`, and `DeviceID`.
   - **Datetime Features**: Convert serial numbers to datetime objects to compute additional features like `Weekday` or `Day of Year`.
   - **Weather Data Integration**: Merge the input data with historical weather data for specific timestamps to create lag features.

## Training the Model

### 1. Install Dependencies

Install the required Python packages:

```sh
pip install -r requirements.txt
```

### 2. Prepare the Data

- **Verify Data Integrity**: Ensure that the data is clean and free of missing or NaN values.
- **Handle Missing Values**: Implement strategies to fill or drop missing data as appropriate.
- **Split Data**: Typically, the data is split into training and testing sets, e.g., 80% training and 20% testing.

### 3. Run the Training Scripts

#### Training the Main Solar Power Prediction Model

1. **Navigate to the Project Directory**:

   ```sh
   cd path/to/your/project
   ```

2. **Execute the Training Script**:

   ```sh
   python model_main.py
   ```

3. **Script Workflow**:

   - **Load Data**: The script reads `processed_data.csv` into a pandas DataFrame.
   - **Feature Selection**: Select relevant features for the model.
   - **Define Target Variable**: The target variable is typically `Power(mW)`.
   - **Hyperparameter Tuning with Optuna**:

     - An objective function is defined to minimize the error metric (e.g., Mean Absolute Error).
     - Optuna suggests hyperparameters like `iterations`, `depth`, `learning_rate`, etc.
     - The study optimizes these hyperparameters over several trials.

   - **Model Training**:

     - A `CatBoostRegressor` model is initialized with the best hyperparameters.
     - The model is trained on the training data.

   - **Model Evaluation**:

     - Predictions are made on the test set.
     - Performance metrics such as MAE and RMSE are calculated.

   - **Model Saving**:

     - The trained model is saved using `joblib`.

#### Training Weather Parameter Models

To improve prediction accuracy, individual models for weather parameters are trained.

1. **Run Weather Model Scripts**:

   - **Wind Speed Model**:

     ```sh
     python modelWindSpeed.py
     ```

   - **Temperature Model**:

     ```sh
     python modelTemperature.py
     ```

   - **Humidity Model**:

     ```sh
     python modelHumidity.py
     ```

   - **Sunlight Model**:

     ```sh
     python modelSunlight.py
     ```

   - **Pressure Model**:

     ```sh
     python modelPressure.py
     ```

2. **Script Workflow**:

   - Similar to the main model training script.
   - Each script trains a model to predict a specific weather parameter.
   - Hyperparameter tuning is performed using Optuna.
   - Models are saved upon completion.

## Model Evaluation

- **Performance Metrics**:

  - **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
  - **Root Mean Squared Error (RMSE)**: Penalizes larger errors more than MAE.

- **Cross-Validation**:

  - **K-Fold Cross-Validation**: Enhances the robustness of the model evaluation.
  - **K-Fold Parameters**: The number of splits (n_splits) can be adjusted based on dataset size.

- **Hyperparameter Tuning**:

  - **Optuna Trials**: The number of trials (n_trials) can be set to balance between computational resource constraints and the desire for optimal hyperparameters.

## Saving and Loading Models

- **Saving Models**:

  ```python
  import joblib
  joblib.dump(model, 'model_name.joblib')
  ```

- **Loading Models**:

  ```python
  model = joblib.load('model_name.joblib')
  ```

- **Model Versioning**: It's good practice to include timestamps or version numbers in your model filenames.

## Notes

- **GPU Support**:

  - For CatBoost, you can enable GPU training by setting `task_type='GPU'` in the model parameters.
  - Ensure that the appropriate GPU drivers and libraries are installed.

- **Data Files**:

  - Keep all necessary data files in the project directory.
  - Adjust file paths in the scripts if your directory structure is different.

- **Hyperparameter Tuning**:

  - Adjust the number of trials in Optuna studies based on available computational resources.
  - More trials can lead to better hyperparameter optimization at the cost of longer training times.

- **Dependencies**:

  - Ensure all dependencies are installed and up-to-date.
  - Use a virtual environment to manage packages without affecting the global Python installation.

## Conclusion

By following the steps outlined in this document, you can train the Solar Power Prediction Model and its associated weather parameter models. Proper training and evaluation will help in creating an accurate and reliable model for forecasting solar power generation, which is valuable for energy planning and management.

**Remember**: Model performance is highly dependent on the quality of data and the appropriateness of features selected. Continuous evaluation and iteration are key to improving model accuracy.