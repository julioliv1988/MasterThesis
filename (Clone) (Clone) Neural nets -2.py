# Databricks notebook source
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.utils import plot_model
# import graphviz
# from tensorflow.keras.utils import model_to_dot

# COMMAND ----------

_df_ = pd.read_csv("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_0.csv")

_df_ = _df_[["observation"]]

# display(_df_)

# Example time series data
# data = np.sin(np.linspace(0, 50, 100))  # Generating synthetic sine wave data

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
initial_seq_length = 10

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

predictions

# COMMAND ----------

# MAGIC %md # Hyper tune

# COMMAND ----------

pip install keras-tuner


# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping

# COMMAND ----------

# Load data
df = pd.read_csv("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_0.csv")
df = df[["observation"]]
data = df.values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
data = scaler.fit_transform(data)


# COMMAND ----------

# Define the model with hyperparameters
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   input_shape=(None, 1)))
    if hp.Boolean('use_dropout'):
        model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=128, step=16)))
    model.add(Dense(1))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='mean_squared_error')
    return model

# COMMAND ----------

# Create the tuner
tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='time_series_tuning')

# Define a function for training and predicting using expanding window
def train_and_predict_expanding_window(data, initial_seq_length, model, scaler):
    predictions = []

    for i in range(initial_seq_length, len(data)):
        # Get expanding window data
        window_data = data[:i]
        X = window_data[:-1].reshape(1, -1, 1)  # All data except the last point
        y = window_data[-1].reshape(1, 1)       # The last point

        # Train the model
        model.fit(X, y, epochs=1, verbose=0)  # Train for 1 epoch to update the model

        # Predict the next point
        prediction = model.predict(window_data.reshape(1, -1, 1))
        predictions.append(prediction[0, 0])

    # Invert scaling for predictions and actual data
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actual_data = scaler.inverse_transform(data)

    return predictions, actual_data

# COMMAND ----------

# Sample a smaller portion of data for hyperparameter tuning
sample_size = 100  # Adjust as needed for performance
X_sample = data[:sample_size]
y_sample = data[1:sample_size + 1]

# Split sample data into training and validation sets
split_idx = int(len(X_sample) * 0.8)
X_train, X_val = X_sample[:split_idx], X_sample[split_idx:]
y_train, y_val = y_sample[:split_idx], y_sample[split_idx:]




# COMMAND ----------

# Define initial sequence length
initial_seq_length = 40

# Prepare the tuner search
tuner.search(X_train.reshape(-1, 1, 1), y_train, epochs=10, validation_data=(X_val.reshape(-1, 1, 1), y_val), callbacks=[EarlyStopping(patience=3)])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

# COMMAND ----------

# Rebuild the model with the best hyperparameters
best_model = build_model(best_hps)
# print(f"Best hyperparameters: {best_hps.values}")


# COMMAND ----------

# Train and predict using expanding window with the best hyperparameters
predictions, actual_data = train_and_predict_expanding_window(data, initial_seq_length, best_model, scaler)

# COMMAND ----------

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(actual_data, label='Actual Data')
plt.plot(range(initial_seq_length, len(actual_data)), predictions, label='Predictions')
plt.legend()
plt.s ou how()


# COMMAND ----------

from logic import *

score = calculate_smape(predictions,actual_data[:len(predictions)])

print(score)
