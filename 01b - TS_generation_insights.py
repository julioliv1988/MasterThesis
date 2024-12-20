# Databricks notebook source
# MAGIC %md
# MAGIC # Widgets

# COMMAND ----------

dbutils.widgets.dropdown("TS_FeatureEng_insights", "False",["True","False"],label=None)
dbutils.widgets.text("SerieNumber",defaultValue="0")

# COMMAND ----------

TS_FeatureEng_insights = dbutils.widgets.get("TS_FeatureEng_insights") in ['True']
SerieNumber = dbutils.widgets.get('SerieNumber')

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook validation

# COMMAND ----------

if not TS_FeatureEng_insights:
    # If condition is met, exit the notebook
    # sys.exit("Skipping the rest of the notebook because condition is met")
    dbutils.notebook.exit("Skipping")

# COMMAND ----------

# MAGIC %md
# MAGIC #Imports

# COMMAND ----------

import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import random
import math

# COMMAND ----------

# MAGIC %md
# MAGIC # Insights on the Generated Time Series

# COMMAND ----------

# MAGIC %md
# MAGIC ##Definition

# COMMAND ----------

def check_range_in_list(list_,target,tolerancy):
  found = False
# Iterate through the list to check for numbers in the range
  for number in list_:
      if target - tolerancy <= number <= target + tolerancy:
          found = True
          break  # If a number is found in the range, exit the loop

  return found

# COMMAND ----------

class series_generator:
      def __init__(self, _dic_proprieties_):
        self.proprieties = _dic_proprieties_

        self.num_samples = _dic_proprieties_["num_samples"]

        # white_noise
        self.std = _dic_proprieties_["white_noise"]["std"]
        self.mean = _dic_proprieties_["white_noise"]["mean"]

        # Seasonality
        self.amplitude = _dic_proprieties_["Seasonality"]["amplitude"]
        self.frequency = _dic_proprieties_["Seasonality"]["frequency"]
        self.frequency2 = _dic_proprieties_["Seasonality"]["frequency2"]
        self.frequency3 = _dic_proprieties_["Seasonality"]["frequency3"]
        self.duration = _dic_proprieties_["Seasonality"]["duration"]
        self.sampling_rate = _dic_proprieties_["Seasonality"]["sampling_rate"]

        # trend
        self.c0 = _dic_proprieties_["trend"]["c0"]
        self.c1 = _dic_proprieties_["trend"]["c1"]
        self.c2 = _dic_proprieties_["trend"]["c2"]
        self.c3 = _dic_proprieties_["trend"]["c3"]
        self.c4 = _dic_proprieties_["trend"]["c4"]
        self.c5 = _dic_proprieties_["trend"]["c5"]

        self.window_size2 = 20
        # after experimenting with several window_size2 20 outcomed good results

        self.series_collection = []
        self.decomp_serie_collection = []

        self.frequencies_found = False

        self.frequencies_calculated = []

      def summary_plot(self,i):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot the data on each subplot

        # Plot noise
        axes[0, 0].plot(self.series_collection[i].index.tolist(), self.series_collection[i]["noise"], color='b', alpha=0.7)
        axes[0, 0].set_title('White Noise Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True)

        # Plot Trend
        axes[0, 1].plot(self.series_collection[i].index.tolist(), self.series_collection[i]["trend"], color='b', alpha=0.7)
        axes[0, 1].set_title('Trend')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True)

        ################## 2 frequencies Wave ##################
        # Plot season
        axes[0, 2].plot(self.series_collection[i].index.tolist(), self.series_collection[i]["season"], color='b', alpha=0.7)
        axes[0, 2].set_title('2-frequencies Wave')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].grid(True)
        # axes[0, 2].set_xlim(0, 100)

        ################## Trend White noise ##################
        # Plot the white noise time series
        axes[1, 0].plot(self.series_collection[i].index.tolist(), self.series_collection[i]["trend_noise"], color='b', alpha=0.7)
        axes[1, 0].set_title('White Noise with trend')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True)

        ################## Cyclic Noise ##################
        # Plot the white noise time series
        axes[1, 1].plot(self.series_collection[i].index.tolist(), self.series_collection[i]["cyclic_noise"], color='b', alpha=0.7)
        axes[1, 1].set_title('White Seasonal Noise')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True)

        ################## Cyclic Noise + Trend ##################
        # Plot the white noise + Trend
        axes[1, 2].plot(self.series_collection[i].index.tolist(), self.series_collection[i]["observation"], color='b', alpha=0.7)
        axes[1, 2].set_title('White Seasonal Noise + Trend')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].grid(True)

      def decomp_serie(self,i):

        df_decomp = self.series_collection[i].copy()

        # Computing the rolling average set with window_size2
        df_decomp[f'rolling-{self.window_size2}'] = df_decomp['observation'].rolling(window=self.window_size2).mean()

        #Double-check it here: wouldn't be needed a shift here, not to include test data in the train data?

        df_decomp = df_decomp.iloc[self.window_size2-1:]  #truncating the DataFrame just to remove nulls

        df_decomp["Cyclic noise Calculated"] = df_decomp['observation'] - df_decomp[f'rolling-{self.window_size2}'] +10

        self.decomp_serie_collection.append(df_decomp)

      def decomp_fft(self,i):
        # Compute the FFT
        fft_result = np.fft.fft(self.decomp_serie_collection[i]["Cyclic noise Calculated"])

        # Frequency values corresponding to the FFT result

        freq = np.fft.fftfreq(len(self.decomp_serie_collection[i]["Cyclic noise Calculated"]), 1.0 / len(self.decomp_serie_collection[i]["Cyclic noise Calculated"]))*self.num_samples/(self.series_collection[i].shape[0]-self.window_size2)

        # Find the index of the maximum amplitude in the FFT result
        index_of_max_amplitude = np.argmax(np.abs(fft_result))

        # Extract the fundamental frequency
        fundamental_frequency = np.abs(freq[index_of_max_amplitude])

        peaks, _ = find_peaks(np.abs(fft_result), height=100, distance=10)
        peak_frequencies = freq[peaks]
        peak_frequencies = [x for x in peak_frequencies if x > 0]

        ff1 = check_range_in_list(peak_frequencies,self.frequency,1)
        ff2 = check_range_in_list(peak_frequencies,self.frequency2,1)

        self.frequencies_calculated = peak_frequencies

        if ff1 and ff2:
          print("frequencies found")
          self.frequencies_found = True
        else:
          # ANSI escape code for bold text
          bold_text = "\033[1m"
          reset_text = "\033[0m"

          # Text to be printed in bold
          text_to_print = "frequencies not found"

          # Print the text in bold
          print(f"{bold_text}{text_to_print}{reset_text}")

        print(f'Calculated peak frequencies: {peak_frequencies}')

      def plot_fft(self,i):

        freq = np.fft.fftfreq(len(self.decomp_serie_collection[i]["Cyclic noise Calculated"]), 1.0 / len(self.decomp_serie_collection[i]["Cyclic noise Calculated"]))*self.num_samples/(self.series_collection[i].shape[0]-self.window_size2)

        fft_result = np.fft.fft(self.decomp_serie_collection[i]["Cyclic noise Calculated"])
        # Find the index of the maximum amplitude in the FFT result
        index_of_max_amplitude = np.argmax(np.abs(fft_result))

        # Extract the fundamental frequency
        fundamental_frequency = np.abs(freq[index_of_max_amplitude])
        # Plot the FFT result
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(freq, np.abs(fft_result))
        plt.title('FFT Result')
        plt.xlim(min(self.frequency,self.frequency2)-1, max(self.frequency,self.frequency2)+1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        # Highlight the fundamental frequency
        plt.subplot(2, 1, 2)
        plt.plot(freq, np.abs(fft_result))
        plt.title('Fundamental Frequency')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 15)  # Limit the x-axis to the expected range of frequencies
        plt.annotate(f'Fundamental Frequency: {fundamental_frequency:.2f} Hz',
                    xy=(fundamental_frequency, np.abs(fft_result[index_of_max_amplitude])),
                    xytext=(fundamental_frequency, 20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', lw=1, color='red'),
                    fontsize=10,
                    color='red')

        plt.tight_layout()
        plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ##Retrieving the generated series

# COMMAND ----------

# Summary table containing all the propreties of all 100 series
summary_proprieties = pd.read_csv("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/summary_proprieties.csv", index_col=0)
display(summary_proprieties)

# COMMAND ----------

# Recreating the list of dicts
dic_proprieties_list = []

for i in range(100):
    specific_row = summary_proprieties.iloc[[i]]
    result_dict = {}
    for column in specific_row.columns:
        keys = column.split('.')
        current_dict = result_dict
        for key in keys[:-1]:
            current_dict = current_dict.setdefault(key, {})
        current_dict[keys[-1]] = specific_row[column].iloc[0]
    dic_proprieties_list.append(result_dict)


# COMMAND ----------

# MAGIC %md
# MAGIC - Periodo máximo de sazonalidade é de 183 dias (meio ano) (f = 5.46)
# MAGIC - Periodo mínimo de sazonalidade é de 7 dias (f = 142.86)

# COMMAND ----------

# MAGIC %md ##Outcomes
# MAGIC

# COMMAND ----------

#Tracking if fft are able to detect seasons accordingly
Series_list = []
counter_ff = 0
for i in range(100):
  Series_list.append(series_generator(dic_proprieties_list[i]))
  # Series_list[i].summary_plot(0)
  Series_list[i].series_collection.append(pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_{i}.csv", index_col=0))
  Series_list[i].decomp_serie(0)
  Series_list[i].decomp_fft(0)
  if Series_list[i].frequencies_found:
    counter_ff = counter_ff + 1
print(counter_ff)

# COMMAND ----------

Series_list[int(SerieNumber)].plot_fft(0)

# COMMAND ----------

Series_list[int(SerieNumber)].summary_plot(0)

# COMMAND ----------

# Detailed Dataframe generated
generated_ts = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_{SerieNumber}.csv", index_col=0)
display(generated_ts)

# COMMAND ----------

from logic import calculate_smape
import sklearn.metrics as metrics
import statistics

# COMMAND ----------

mean_list = [statistics.mean(generated_ts['noise']) for _ in range(1000)]

# COMMAND ----------

wn_mae = metrics.mean_absolute_error(generated_ts['noise'], mean_list)
wn_mse = metrics.mean_squared_error(generated_ts['noise'], mean_list)
wn_rmse = np.sqrt(wn_mse) # or mse**(0.5)
wn_r2 = metrics.r2_score(generated_ts['noise'], mean_list)
wn_smape = calculate_smape(generated_ts['noise'], mean_list)   

# COMMAND ----------

print(f"""wn_smape: {wn_smape}
wn_mae: {wn_mae}
wn_mse: {wn_mse}
wn_rmse: {wn_rmse}
wn_r2: {wn_r2}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Peak frequencies are not well calculated just when the real frequencies are close to each other. If I implement a constraint --- and this makes total sence
