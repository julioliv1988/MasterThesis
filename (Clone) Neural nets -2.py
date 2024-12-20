# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import sklearn.metrics as metrics
from logic import *

# COMMAND ----------

dbutils.widgets.dropdown("Chosen Model", "Random Forest",["Linear Regression", "KNN", "Random Forest" ],label=None)
dbutils.widgets.dropdown("TS_truncated_experiment", "False",["True","False"],label=None)
dbutils.widgets.text("SerieNumber",defaultValue="0")
dbutils.widgets.text("Modified_serie",defaultValue="False")
dbutils.widgets.text("Models",defaultValue="False")

# COMMAND ----------

import ast

Chosen_Model = dbutils.widgets.get("Chosen Model")
TS_truncated_experiment = dbutils.widgets.get("TS_truncated_experiment") in ['True']
SerieNumber = dbutils.widgets.get('SerieNumber')
Modified_serie = ast.literal_eval(dbutils.widgets.get('Modified_serie'))

Models = ast.literal_eval(dbutils.widgets.get('Models'))

# COMMAND ----------

if not Modified_serie:
    decorator = ""
else:
    decorator = f"-{Modified_serie[0]}-{Modified_serie[1]}"

# COMMAND ----------

# MAGIC %md # Expanding window approach

# COMMAND ----------

# Setting variables/env

experiment_name = "/Users/jdeoliveira@microsoft.com/Master-Experiment"

# Set the experiment
mlflow.set_experiment(experiment_name)

experiment_path_ = "/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/"

_df_ = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_{SerieNumber}.csv")

# COMMAND ----------

print(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_{SerieNumber}.csv")

# COMMAND ----------

with mlflow.start_run(run_name=f"LSTM_predict_FULL") as run:

    _df_ = pd.read_csv("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_0.csv")
    # _df_ = _df_.iloc[:50]

    _df_ = _df_[["observation"]]

    data = _df_.values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.reshape(-1, 1)
    data = scaler.fit_transform(data)

    # Build the LSTM model
    def build_model():
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(None, 1)))
        model.add(Dense(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model = build_model()

    # Set initial sequence length
    initial_seq_length = 40

    # Function to train and predict using expanding window
    def train_and_predict_expanding_window(data, initial_seq_length, model, scaler):
        predictions = []
        
        for i in range(initial_seq_length, len(data)):
            # Get expanding window data
            window_data = data[:i]
            X = window_data[:-1].reshape(1, -1, 1)  # All data except the last point
            y = window_data[-1].reshape(1, 1)       # The last point
            
            # Train the model
            model.fit(X, y, epochs=10, verbose=0)  # Train for 1 epoch to update the model
            
            # Predict the next point
            prediction = model.predict(window_data.reshape(1, -1, 1))
            predictions.append(prediction[0, 0])
        
        # Invert scaling for predictions and actual data
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        actual_data = scaler.inverse_transform(data)
        
        return predictions, actual_data
    # Train and predict using expanding window
    predictions, actual_data = train_and_predict_expanding_window(data, initial_seq_length, model, scaler)
    actual_data_flat = actual_data.flatten()
    predictions_flat = predictions.flatten()
    pred_ew = pd.DataFrame({'observation': actual_data_flat.tolist()[:len(predictions)],'forcast': predictions_flat.tolist()})

    mae = metrics.mean_absolute_error(pred_ew.observation, pred_ew.forcast)
    mse = metrics.mean_squared_error(pred_ew.observation, pred_ew.forcast)
    rmse = np.sqrt(mse) # or mse**(0.5)
    r2 = metrics.r2_score(pred_ew.observation, pred_ew.forcast)
    smape = calculate_smape(pred_ew.observation, pred_ew.forcast)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("smape", smape)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(actual_data, label='Original Data')
    plt.plot(range(initial_seq_length, len(actual_data)), predictions, label='Predictions', linestyle='--')
    plt.title('LSTM Predictions Using Expanding Window')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import sklearn.metrics as metrics
from logic import *

# with mlflow.start_run(run_name=f"LSTM_predict_FULL") as run:

_df_ = pd.read_csv("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_0.csv")
# _df_ = _df_.iloc[:50]

_df_ = _df_[["observation"]]

data = _df_.values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
data = scaler.fit_transform(data)

# Build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(None, 1)))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model()

# Set initial sequence length
initial_seq_length = 40

# Function to train and predict using expanding window
def train_and_predict_expanding_window(data, initial_seq_length, model, scaler):
    predictions = []
    
    for i in range(initial_seq_length, len(data)):
        # Get expanding window data
        window_data = data[:i]
        X = window_data[:-1].reshape(1, -1, 1)  # All data except the last point
        y = window_data[-1].reshape(1, 1)       # The last point
        
        # Train the model
        model.fit(X, y, epochs=10, verbose=0)  # Train for 1 epoch to update the model
        
        # Predict the next point
        prediction = model.predict(window_data.reshape(1, -1, 1))
        predictions.append(prediction[0, 0])
    
    # Invert scaling for predictions and actual data
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actual_data = scaler.inverse_transform(data)
    
    return predictions, actual_data
# Train and predict using expanding window
predictions, actual_data = train_and_predict_expanding_window(data, initial_seq_length, model, scaler)
actual_data_flat = actual_data.flatten()
predictions_flat = predictions.flatten()
pred_ew = pd.DataFrame({'observation': actual_data_flat.tolist()[:len(predictions)],'forcast': predictions_flat.tolist()})

mae = metrics.mean_absolute_error(pred_ew.observation, pred_ew.forcast)
mse = metrics.mean_squared_error(pred_ew.observation, pred_ew.forcast)
rmse = np.sqrt(mse) # or mse**(0.5)
r2 = metrics.r2_score(pred_ew.observation, pred_ew.forcast)
smape = calculate_smape(pred_ew.observation, pred_ew.forcast)
# mlflow.log_metric("mae", mae)
# mlflow.log_metric("mse", mse)
# mlflow.log_metric("rmse", rmse)
# mlflow.log_metric("r2", r2)
# mlflow.log_metric("smape", smape)
print(f"mae: {mae}")
print(f"mse: {mse}")
print(f"rmse: {rmse}") 
print(f"r2: {r2}")
print(f"smape: {smape}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(actual_data, label='Original Data')
plt.plot(range(initial_seq_length, len(actual_data)), predictions, label='Predictions', linestyle='--')
plt.title('LSTM Predictions Using Expanding Window')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from logic import *


# COMMAND ----------

_df_ = pd.read_csv("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_0.csv")

_df_ = _df_[["observation"]]

# display(_df_)

data = _df_.values

# Plot the data
plt.plot(data)
plt.title('Synthetic Sine Wave Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
data = scaler.fit_transform(data)


# COMMAND ----------

# Build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(None, 1)))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_model()
model.summary()


# COMMAND ----------

# Set initial sequence length
initial_seq_length = 40

# Function to train and predict using expanding window
def train_and_predict_expanding_window(data, initial_seq_length, model, scaler):
    predictions = []
    
    for i in range(initial_seq_length, len(data)):
        # Get expanding window data
        window_data = data[:i]
        X = window_data[:-1].reshape(1, -1, 1)  # All data except the last point
        y = window_data[-1].reshape(1, 1)       # The last point
        
        # Train the model
        model.fit(X, y, epochs=10, verbose=0)  # Train for 1 epoch to update the model
        
        # Predict the next point
        prediction = model.predict(window_data.reshape(1, -1, 1))
        predictions.append(prediction[0, 0])
    
    # Invert scaling for predictions and actual data
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actual_data = scaler.inverse_transform(data)
    
    return predictions, actual_data

# COMMAND ----------

# Train and predict using expanding window
predictions, actual_data = train_and_predict_expanding_window(data, initial_seq_length, model, scaler)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(actual_data, label='Original Data')
plt.plot(range(initial_seq_length, len(actual_data)), predictions, label='Predictions', linestyle='--')
plt.title('LSTM Predictions Using Expanding Window')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# COMMAND ----------

score = calculate_smape(predictions,actual_data[:len(predictions)])

print(score)

# COMMAND ----------

score = calculate_smape(predictions,actual_data[:len(predictions)])

print(score)

# COMMAND ----------

predictions
