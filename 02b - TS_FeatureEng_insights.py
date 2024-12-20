# Databricks notebook source
# MAGIC %md
# MAGIC #Imports

# COMMAND ----------

import pandas as pd
import ast

# COMMAND ----------

# MAGIC %md
# MAGIC # Widgets

# COMMAND ----------

dbutils.widgets.dropdown("TS_generation_insights", "False",["True","False"],label=None)
dbutils.widgets.text("SerieNumber",defaultValue="0")
dbutils.widgets.text("Modified_serie",defaultValue="False")

# COMMAND ----------

TS_generation_insights = dbutils.widgets.get("TS_generation_insights") in ['True']
SerieNumber = dbutils.widgets.get('SerieNumber')
Modified_serie = ast.literal_eval(dbutils.widgets.get('Modified_serie'))

# COMMAND ----------

if not Modified_serie:
    decorator = ""
else:
    decorator = f"-{Modified_serie[0]}-{Modified_serie[1]}"

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook validation

# COMMAND ----------

if not TS_generation_insights:
    # If condition is met, exit the notebook
    dbutils.notebook.exit("Skipping")

# COMMAND ----------

# MAGIC %md # Feature Engineered

# COMMAND ----------

# Feature engineer just the 1st Serie
feature_enged = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}{decorator}/train_set_featured/window_959.csv", index_col=0)
# traintest_df = traintest_df[["observation"]]
# # traintest_df
# serie_solv = timeseries_solver(traintest_df)
# serie_solv.feature_eng()
display(feature_enged)

# COMMAND ----------

freq_calculated = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}/Feature_eng/freq_calculated.csv", index_col=0)
display(freq_calculated)

# COMMAND ----------

# MAGIC %md
# MAGIC #Original Time Serie

# COMMAND ----------

# Original TS
original_ts = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Input/input_{SerieNumber}.csv", index_col=0)[['observation']]
display(original_ts)
