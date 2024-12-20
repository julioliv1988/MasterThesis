# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sklearn.metrics as metrics
from logic import *
import mlflow

# COMMAND ----------

dbutils.widgets.text("SerieNumber",defaultValue="0")
# dbutils.widgets.text("Modified_serie",defaultValue="False")

# COMMAND ----------

import ast

SerieNumber = dbutils.widgets.get('SerieNumber')
# Modified_serie = ast.literal_eval(dbutils.widgets.get('Modified_serie'))


# COMMAND ----------

# if not Modified_serie:
#     decorator = ""
# else:
#     decorator = f"-{Modified_serie[0]}-{Modified_serie[1]}"

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

try:
    _random_string_ = dbutils.jobs.taskValues.get(taskKey="TS_FeatureEng", key="job_reference")
    print(f"Retrieved from TS_FeatureEng task:{_random_string_}")
except:
    print("No reference found - creating new one")
    _random_string_ = generate_random_string(7)
    print(_random_string_)
    dbutils.fs.mkdirs(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}')
    # dbutils.jobs.taskValues.set(key="job_reference", value=_random_string_)

# COMMAND ----------

# MAGIC %md # Expanding window approach

# COMMAND ----------

_df_ = _df_[["observation"]]

# _df_ = _df_.head(70)

data = _df_.values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
data = scaler.fit_transform(data)



# COMMAND ----------

# Set initial sequence length
initial_seq_length = 40

# Build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(None, 1)))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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

def log_model_params(model):
    # Log parameters of each layer
    for i, layer in enumerate(model.layers):
        if isinstance(layer, LSTM):
            mlflow.log_param(f"LSTM_units_layer_{i+1}", layer.units)
        elif isinstance(layer, Dense):
            mlflow.log_param(f"Dense_units_layer_{i+1}", layer.units)

# COMMAND ----------


with mlflow.start_run(run_name=f"LSTM_predict_FULL") as run:

    model = build_model()
    # Train and predict using expanding window
    predictions, actual_data = train_and_predict_expanding_window(data, initial_seq_length, model, scaler)
    model.summary()
    print(data)
    print(scaler)
    print(initial_seq_length)

    actual_data_flat = actual_data.flatten()
    predictions_flat = predictions.flatten()


    pred_ew = pd.DataFrame({'observation': actual_data_flat.tolist()[:len(predictions)],'forcast': predictions_flat.tolist()})

    # random_string = generate_random_string(7)
    forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_FULL_LSTM.csv"
    pred_ew.to_csv(forcast_path, mode='x', index=False)
    mlflow.log_artifact(forcast_path, "forcast_FULL_LSTM-{_random_string_}.csv")

    display(pred_ew)

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

    plt = plot_forcast_vs_obs(pred_ew,f"LSTM")
    mlflow.log_figure(plt, f"forcast_FULL_LSTM-{_random_string_}.png")

    log_model_params(model)
    mlflow.sklearn.log_model(model, "LSTM")
    mlflow.set_tag("iternal_id", _random_string_)
    mlflow.set_tag("dataset", SerieNumber)


# COMMAND ----------

model = build_model()
# Train and predict using expanding window
predictions, actual_data = train_and_predict_expanding_window(data, initial_seq_length, model, scaler)
model.summary()
print(data)
print(scaler)
print(initial_seq_length)

actual_data_flat = actual_data.flatten()
predictions_flat = predictions.flatten()


pred_ew = pd.DataFrame({'observation': actual_data_flat.tolist()[:len(predictions)],'forcast': predictions_flat.tolist()})

# random_string = generate_random_string(7)
forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_FULL_LSTM.csv"
pred_ew.to_csv(forcast_path, mode='x', index=False)
# mlflow.log_artifact(forcast_path, "forcast_FULL_LSTM-{_random_string_}.csv")

display(pred_ew)

mae = metrics.mean_absolute_error(pred_ew.observation, pred_ew.forcast)
mse = metrics.mean_squared_error(pred_ew.observation, pred_ew.forcast)
rmse = np.sqrt(mse) # or mse**(0.5)
r2 = metrics.r2_score(pred_ew.observation, pred_ew.forcast)
smape = calculate_smape(pred_ew.observation, pred_ew.forcast)
print(f"mae: {mae}")
print(f"mse: {mse}")
print(f"rmse: {rmse}") 
print(f"r2: {r2}")
print(f"smape: {smape}")

# mlflow.log_metric("mae", mae)
# mlflow.log_metric("mse", mse)
# mlflow.log_metric("rmse", rmse)
# mlflow.log_metric("r2", r2)
# mlflow.log_metric("smape", smape)

plt = plot_forcast_vs_obs(pred_ew,f"LSTM")
plt.show()
# mlflow.log_figure(plt, f"forcast_FULL_LSTM-{_random_string_}.png")

# log_model_params(model)
# mlflow.sklearn.log_model(model, "LSTM")
# mlflow.set_tag("iternal_id", _random_string_)
# mlflow.set_tag("dataset", SerieNumber)

