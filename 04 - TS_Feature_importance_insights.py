# Databricks notebook source
# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import re
import ast

# COMMAND ----------

try:
    _random_string_ = dbutils.jobs.taskValues.get(taskKey="TS_FeatureEng", key="job_reference")
    print(f"Retrieved from TS_FeatureEng task:{_random_string_}")
except:

    dbutils.notebook.exit("Skipping")


# COMMAND ----------

# MAGIC %md
# MAGIC # Widgets

# COMMAND ----------

dbutils.widgets.text("SerieNumber",defaultValue="0")
dbutils.widgets.text("Modified_serie",defaultValue="False")

# COMMAND ----------

SerieNumber = dbutils.widgets.get('SerieNumber')
Modified_serie = ast.literal_eval(dbutils.widgets.get('Modified_serie'))

# COMMAND ----------

# MAGIC %md
# MAGIC #Defitinions

# COMMAND ----------

if not Modified_serie:
    decorator = ""
else:
    decorator = f"-{Modified_serie[0]}-{Modified_serie[1]}"

# COMMAND ----------

print(decorator)

# COMMAND ----------

def extract_number(path):
    filename = path.split('/')[-1]  # Extract the filename from the path
    match = re.search(r'\d+', filename)  # Find the first sequence of digits in the filename
    return int(match.group()) if match else float('inf')  # Use 'inf' to handle cases with no digits

def transform_string(s):
    # Remove all occurrences of ":"
    transformed = s.replace(":", "")
    # Add "/" at the beginning
    result = "/" + transformed
    return result

# # Example usage:
# input_string = "12:34:56"
# output_string = transform_string(input_string)

# print(output_string)  # Output: "/123456"


# COMMAND ----------

list_files_importance_FULL_rf = sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_FULL_rf')], key=extract_number) 

list_files_importance_VEST_rf =  sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_VEST_rf')], key=extract_number)

list_files_importance_FULL_lr =  sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_FULL_lr')], key=extract_number)

list_files_importance_VEST_lr =  sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_VEST_lr')], key=extract_number)

list_files_importance_FULL_rf_pd = []
for files in list_files_importance_FULL_rf:
    list_files_importance_FULL_rf_pd.append(pd.read_csv(transform_string(files)))

list_files_importance_VEST_rf_pd = []
for files in list_files_importance_VEST_rf:
    list_files_importance_VEST_rf_pd.append(pd.read_csv(transform_string(files)))

list_files_importance_FULL_lr_pd = []
for files in list_files_importance_FULL_lr:
    list_files_importance_FULL_lr_pd.append(pd.read_csv(transform_string(files)))

list_files_importance_VEST_lr_pd = []
for files in list_files_importance_VEST_lr:
    list_files_importance_VEST_lr_pd.append(pd.read_csv(transform_string(files)))


# COMMAND ----------

try:
    freq_calculated = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}{decorator}/Feature_eng/freq_calculated.csv", index_col=0)
    freq_calculated[["f1","f2"]]
    # display(freq_calculated)

    # Retrieve the index of the first row that contains a non-NaN value
    first_non_nan_index = freq_calculated.first_valid_index()

    # Convert the index to the row number
    season_cut = freq_calculated.index.get_loc(first_non_nan_index)

    # Retrieve the index of the first row that contains a non-NaN value
    first_non_nan_index = freq_calculated["f2"].first_valid_index()

    # Convert the index to the row number
    season_cut_2 = freq_calculated["f2"].index.get_loc(first_non_nan_index)

    print(f"Season cut 1: {season_cut}")
    print(f"Season cut 2: {season_cut_2}")
except:
    season_cut = 25
    season_cut_2 = 250
    print(f"Season cut 1: {season_cut}")
    print(f"Season cut 2: {season_cut_2}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Linear Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full

# COMMAND ----------

# Considering entire serie
importance_FULL_lr_appended = pd.DataFrame(columns=list_files_importance_FULL_lr_pd[0].columns)
for _df_ in list_files_importance_FULL_lr_pd:
    importance_FULL_lr_appended = pd.concat([importance_FULL_lr_appended, _df_], ignore_index=True)

full_agerage_agg = importance_FULL_lr_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Considering entire serie")
display(full_agerage_agg.reset_index())

# COMMAND ----------

list_importance_FULL_lr__filtered_no_season_appended = pd.DataFrame(columns=list_files_importance_FULL_lr_pd[0].columns)
for i in range(season_cut):
    list_importance_FULL_lr__filtered_no_season_appended = pd.concat([list_importance_FULL_lr__filtered_no_season_appended, list_files_importance_FULL_lr_pd[i]], ignore_index=True)

FULL_lr__filtered_no_season_agg = list_importance_FULL_lr__filtered_no_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Linear Regression FULL - no season - Average")
display(FULL_lr__filtered_no_season_agg.reset_index())

list_importance_FULL_lr__filtered_1_season_appended = pd.DataFrame(columns=list_files_importance_FULL_lr_pd[0].columns)
for i in range(season_cut,season_cut_2):
    list_importance_FULL_lr__filtered_1_season_appended = pd.concat([list_importance_FULL_lr__filtered_1_season_appended, list_files_importance_FULL_lr_pd[i]], ignore_index=True)

FULL_lr__filtered_1_season_agg = list_importance_FULL_lr__filtered_1_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Linear Regression FULL - 1 season - Average")
display(FULL_lr__filtered_1_season_agg.reset_index())

list_importance_FULL_lr__filtered_2_season_appended = pd.DataFrame(columns=list_files_importance_FULL_lr_pd[0].columns)
for i in range(season_cut_2,len(list_files_importance_FULL_lr_pd)):
    list_importance_FULL_lr__filtered_2_season_appended = pd.concat([list_importance_FULL_lr__filtered_2_season_appended, list_files_importance_FULL_lr_pd[i]], ignore_index=True)

FULL_lr__filtered_2_season_agg = list_importance_FULL_lr__filtered_2_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Linear Regression FULL - 2 season - Average")
display(FULL_lr__filtered_2_season_agg.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## VEST

# COMMAND ----------

# Considering entire serie
importance_VEST_lr_appended = pd.DataFrame(columns=list_files_importance_VEST_lr_pd[0].columns)
for _df_ in list_files_importance_VEST_lr_pd:
    importance_VEST_lr_appended = pd.concat([importance_VEST_lr_appended, _df_], ignore_index=True)

VEST_agerage_agg = importance_VEST_lr_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Considering entire serie")
display(VEST_agerage_agg.reset_index())

# COMMAND ----------

list_importance_VEST_lr__filtered_no_season_appended = pd.DataFrame(columns=list_files_importance_VEST_lr_pd[0].columns)
for i in range(season_cut):
    list_importance_VEST_lr__filtered_no_season_appended = pd.concat([list_importance_VEST_lr__filtered_no_season_appended, list_files_importance_VEST_lr_pd[i]], ignore_index=True)

VEST_lr__filtered_no_season_agg = list_importance_VEST_lr__filtered_no_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Linear Regression VEST - no season - Average")
display(VEST_lr__filtered_no_season_agg.reset_index())

list_importance_VEST_lr__filtered_1_season_appended = pd.DataFrame(columns=list_files_importance_VEST_lr_pd[0].columns)
for i in range(season_cut,season_cut_2):
    list_importance_VEST_lr__filtered_1_season_appended = pd.concat([list_importance_VEST_lr__filtered_1_season_appended, list_files_importance_VEST_lr_pd[i]], ignore_index=True)

VEST_lr__filtered_1_season_agg = list_importance_VEST_lr__filtered_1_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Linear Regression VEST - 1 season - Average")
display(VEST_lr__filtered_1_season_agg.reset_index())

list_importance_VEST_lr__filtered_2_season_appended = pd.DataFrame(columns=list_files_importance_VEST_lr_pd[0].columns)
for i in range(season_cut_2,len(list_files_importance_VEST_lr_pd)):
    list_importance_VEST_lr__filtered_2_season_appended = pd.concat([list_importance_VEST_lr__filtered_2_season_appended, list_files_importance_VEST_lr_pd[i]], ignore_index=True)

VEST_lr__filtered_2_season_agg = list_importance_VEST_lr__filtered_2_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Linear Regression VEST - 2 season - Average")
display(VEST_lr__filtered_2_season_agg.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC #Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC ##Full

# COMMAND ----------

importance_FULL_rf_appended = pd.DataFrame(columns=list_files_importance_FULL_rf_pd[0].columns)
for _df_ in list_files_importance_FULL_rf_pd:
    importance_FULL_rf_appended = pd.concat([importance_FULL_rf_appended, _df_], ignore_index=True)

full_agerage_agg = importance_FULL_rf_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Considering entire serie")
display(full_agerage_agg.reset_index())

# COMMAND ----------

list_importance_FULL_rf__filtered_no_season_appended = pd.DataFrame(columns=list_files_importance_FULL_rf_pd[0].columns)
for i in range(season_cut):
    list_importance_FULL_rf__filtered_no_season_appended = pd.concat([list_importance_FULL_rf__filtered_no_season_appended, list_files_importance_FULL_rf_pd[i]], ignore_index=True)

FULL_rf__filtered_no_season_agg = list_importance_FULL_rf__filtered_no_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Random Forest FULL - no season - Average")
display(FULL_rf__filtered_no_season_agg.reset_index())

list_importance_FULL_rf__filtered_1_season_appended = pd.DataFrame(columns=list_files_importance_FULL_rf_pd[0].columns)
for i in range(season_cut,season_cut_2):
    list_importance_FULL_rf__filtered_1_season_appended = pd.concat([list_importance_FULL_rf__filtered_1_season_appended, list_files_importance_FULL_rf_pd[i]], ignore_index=True)

FULL_rf__filtered_1_season_agg = list_importance_FULL_rf__filtered_1_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Random Forest FULL - 1 season - Average")
display(FULL_rf__filtered_1_season_agg.reset_index())

list_importance_FULL_rf__filtered_2_season_appended = pd.DataFrame(columns=list_files_importance_FULL_rf_pd[0].columns)
for i in range(season_cut_2,len(list_files_importance_FULL_rf_pd)):
    list_importance_FULL_rf__filtered_2_season_appended = pd.concat([list_importance_FULL_rf__filtered_2_season_appended, list_files_importance_FULL_rf_pd[i]], ignore_index=True)

FULL_rf__filtered_2_season_agg = list_importance_FULL_rf__filtered_2_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Random Forest FULL - 2 season - Average")
display(FULL_rf__filtered_2_season_agg.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## VEST

# COMMAND ----------

# Considering entire serie
importance_VEST_rf_appended = pd.DataFrame(columns=list_files_importance_VEST_rf_pd[0].columns)
for _df_ in list_files_importance_VEST_rf_pd:
    importance_VEST_rf_appended = pd.concat([importance_VEST_rf_appended, _df_], ignore_index=True)

VEST_agerage_agg = importance_VEST_rf_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Considering entire serie")
display(VEST_agerage_agg.reset_index())

# COMMAND ----------

list_importance_VEST_rf__filtered_no_season_appended = pd.DataFrame(columns=list_files_importance_VEST_rf_pd[0].columns)
for i in range(season_cut):
    list_importance_VEST_rf__filtered_no_season_appended = pd.concat([list_importance_VEST_rf__filtered_no_season_appended, list_files_importance_VEST_rf_pd[i]], ignore_index=True)

VEST_rf__filtered_no_season_agg = list_importance_VEST_rf__filtered_no_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Random Forest VEST - no season - Average")
display(VEST_rf__filtered_no_season_agg.reset_index())

list_importance_VEST_rf__filtered_1_season_appended = pd.DataFrame(columns=list_files_importance_VEST_rf_pd[0].columns)
for i in range(season_cut,season_cut_2):
    list_importance_VEST_rf__filtered_1_season_appended = pd.concat([list_importance_VEST_rf__filtered_1_season_appended, list_files_importance_VEST_rf_pd[i]], ignore_index=True)

VEST_rf__filtered_1_season_agg = list_importance_VEST_rf__filtered_1_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Random Forest VEST - 1 season - Average")
display(VEST_rf__filtered_1_season_agg.reset_index())

list_importance_VEST_rf__filtered_2_season_appended = pd.DataFrame(columns=list_files_importance_VEST_rf_pd[0].columns)
for i in range(season_cut_2,len(list_files_importance_VEST_rf_pd)):
    list_importance_VEST_rf__filtered_2_season_appended = pd.concat([list_importance_VEST_rf__filtered_2_season_appended, list_files_importance_VEST_rf_pd[i]], ignore_index=True)

VEST_rf__filtered_2_season_agg = list_importance_VEST_rf__filtered_2_season_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Random Forest VEST - 2 season - Average")
display(VEST_rf__filtered_2_season_agg.reset_index())
