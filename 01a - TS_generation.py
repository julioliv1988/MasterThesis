# Databricks notebook source
# MAGIC %md
# MAGIC # Widgets

# COMMAND ----------

dbutils.widgets.dropdown("TS_generation", "False",["True","False"],label=None)

# COMMAND ----------

TS_generation = dbutils.widgets.get("TS_generation") in ['True']

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook validation

# COMMAND ----------

if not TS_generation:
    # If condition is met, exit the notebook
    # sys.exit("Skipping the rest of the notebook because condition is met")
    dbutils.notebook.exit("Skipping")

# COMMAND ----------

# MAGIC %md
# MAGIC #Imports

# COMMAND ----------

import pandas as pd
from pandas import json_normalize
import numpy as np
import random
import warnings
warnings.simplefilter('ignore', np.RankWarning)


# COMMAND ----------

# MAGIC %md
# MAGIC # Artificial Generation of Time Series

# COMMAND ----------

# MAGIC %md
# MAGIC ##Definition

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

      def generate_serie(self):

        ### Pure Signals
        # white_noise
        white_noise = np.random.normal(self.mean, self.std, self.num_samples)
        white_noise = white_noise - min(white_noise) + 1
        time = np.arange(self.num_samples)
        # trend
        trend = []
        for i in range(self.num_samples):
          trend.append(self.c0 + self.c1*i + self.c2*i**2 + self.c3*i**3+ self.c4*i**4+ self.c5*i**5)
        # season
        t = np.linspace(0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        season = 1 + self.amplitude + \
                     self.amplitude * np.sin(2 * np.pi * self.frequency * t) \
                     + np.sin(2 * np.pi * self.frequency2 * t)

        ### Combined Signals
        # Trend Noise
        trend_noise = []
        for i in range(self.num_samples):
          trend_noise.append(trend[i]+white_noise[i])

        cyclic_noise = []
        for i in range(self.num_samples):
          cyclic_noise.append(white_noise[i]+season[i])

        cyclic_trend_noise  = []
        for i in range(len(white_noise)):
          cyclic_trend_noise.append(cyclic_noise[i]+trend[i])

        # Getting just half of the data to see how this behave

        _df_ = pd.DataFrame({'trend':trend, 'season': season, 'noise':white_noise, 'trend_noise': trend_noise, 'cyclic_noise': cyclic_noise, 'observation':cyclic_trend_noise})

        # cyclic_trend_noise

        # _df_ = _df_.iloc[:600]

        self.series_collection.append(_df_)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Generating 100 different time series

# COMMAND ----------

dic_proprieties_list = []

for i in range(100):
  ##################TREND_PARAMETERS
  # To fit a polynomial curve of 5th degree we need 4 points
  polynomial_points = []
  polynomial_points.append((200, random.uniform(0, 8)))
  polynomial_points.append((400, random.uniform(0, 8)))
  polynomial_points.append((600, random.uniform(0, 8)))
  polynomial_points.append((800, random.uniform(0, 8)))

  x = [inner_list[0] for inner_list in polynomial_points]
  y = [inner_list[1] for inner_list in polynomial_points]
  coefficients = np.polyfit(x, y, 5)

  # Create a polynomial function using the coefficients
  p5 = np.poly1d(coefficients)

  ##################FREQUENCY_SEASON_PARAMETERS
  min_diff_d = 7  # Minimum difference days
  min_diff_f = 10  # Minimum difference freq
  period_1 = random.randint(7, 183)
  period_2 = random.randint(7, 183)
  f1 = 1000/ period_1
  f2 = 1000 / period_2

  while (abs(period_1-period_2)<min_diff_d+1) or abs(f1-f2)<min_diff_f:
      period_2 = random.randint(7, 183)
      f2 = 1000 / period_2

  dic_proprieties = {
      "num_samples": 1000,
      "white_noise": {"std": 1, "mean":0},
      "Seasonality": {"amplitude": 3.0, "frequency" : f1, "frequency2": f2,"frequency3": 0, "duration": 1.0, "sampling_rate": 1000},
      "trend": { "c0": p5.coefficients[5] , "c1": p5.coefficients[4], "c2": p5.coefficients[3], "c3": p5.coefficients[2], "c4": p5.coefficients[1], "c5": p5.coefficients[0]}
      }
  dic_proprieties_list.append(dic_proprieties)
  # dic_proprieties

# COMMAND ----------

# MAGIC %md
# MAGIC - Periodo máximo de sazonalidade é de 183 dias (meio ano) (f = 5.46)
# MAGIC - Periodo mínimo de sazonalidade é de 7 dias (f = 142.86)

# COMMAND ----------

Series_list = []
for i in range(100):
  Series_list.append(series_generator(dic_proprieties_list[i]))
  Series_list[i].generate_serie()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Storing results

# COMMAND ----------

# Your list of JSON objects
json_list = [series.proprieties for series in Series_list]

# Convert the list of JSONs to a pandas DataFrame
summary_proprieties = json_normalize(json_list)

# COMMAND ----------

#Storing the CSVs
dbutils.fs.rm('/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/',True)
dbutils.fs.mkdirs('/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input')
for i in range(100):
  Series_list[i].series_collection[0].to_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_{i}.csv")
summary_proprieties.to_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/summary_proprieties.csv")

