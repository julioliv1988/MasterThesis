# Databricks notebook source
import pandas as pd
import numpy as np
import re
import ast

# COMMAND ----------

# dbutils.widgets.text("SerieNumber",defaultValue="0")
# dbutils.widgets.text("Modified_serie",defaultValue="False")

# COMMAND ----------

# SerieNumber = dbutils.widgets.get('SerieNumber')
# Modified_serie = ast.literal_eval(dbutils.widgets.get('Modified_serie'))

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

# Function to find key by value
def get_key_by_value(nested_dict, target_value):
    for key, sub_dict in nested_dict.items():
        if sub_dict.get('internal_key') == target_value:
            return key
    return None  # If the value is not found

# COMMAND ----------

ifr_summary_dic = {}


ifr_summary_dic['SerieNumber_0'] = {}
ifr_summary_dic['SerieNumber_0']['internal_key'] = "YishP6I"

ifr_summary_dic['SerieNumber_1'] = {}
ifr_summary_dic['SerieNumber_1']['internal_key'] = "mMRVk4N"

ifr_summary_dic['SerieNumber_2'] = {}
ifr_summary_dic['SerieNumber_2']['internal_key'] = "Grbb0dz"

ifr_summary_dic['SerieNumber_3'] = {}
ifr_summary_dic['SerieNumber_3']['internal_key'] = "XznRtp7"

ifr_summary_dic['SerieNumber_4'] = {}
ifr_summary_dic['SerieNumber_4']['internal_key'] = "VhQxMtS"

ifr_summary_dic['SerieNumber_0-trend-season'] = {}
ifr_summary_dic['SerieNumber_0-trend-season']['internal_key'] = "xRmAioK"

ifr_summary_dic['SerieNumber_1-trend-season'] = {}
ifr_summary_dic['SerieNumber_1-trend-season']['internal_key'] = "6KC6QnM"

ifr_summary_dic['SerieNumber_2-trend-season'] = {}
ifr_summary_dic['SerieNumber_2-trend-season']['internal_key'] = "Ef93YW6"

ifr_summary_dic['SerieNumber_3-trend-season'] = {}
ifr_summary_dic['SerieNumber_3-trend-season']['internal_key'] = "h94TUJC"

ifr_summary_dic['SerieNumber_4-trend-season'] = {}
ifr_summary_dic['SerieNumber_4-trend-season']['internal_key'] = "MmpnJ41"

ifr_summary_dic['SerieNumber_0-season-noise'] = {}
ifr_summary_dic['SerieNumber_0-season-noise']['internal_key'] = "cGJPwDY"

ifr_summary_dic['SerieNumber_1-season-noise'] = {}
ifr_summary_dic['SerieNumber_1-season-noise']['internal_key'] = "4P9oiEy"

ifr_summary_dic['SerieNumber_2-season-noise'] = {}
ifr_summary_dic['SerieNumber_2-season-noise']['internal_key'] = "ckGrURE"

ifr_summary_dic['SerieNumber_3-season-noise'] = {}
ifr_summary_dic['SerieNumber_3-season-noise']['internal_key'] = "KLUPRil"

ifr_summary_dic['SerieNumber_4-season-noise'] = {}
ifr_summary_dic['SerieNumber_4-season-noise']['internal_key'] = "rxulmxt"

ifr_summary_dic['SerieNumber_0-trend-noise'] = {}
ifr_summary_dic['SerieNumber_0-trend-noise']['internal_key'] = "1aMpsha"

ifr_summary_dic['SerieNumber_1-trend-noise'] = {}
ifr_summary_dic['SerieNumber_1-trend-noise']['internal_key'] = "K6muDqb"

ifr_summary_dic['SerieNumber_2-trend-noise'] = {}
ifr_summary_dic['SerieNumber_2-trend-noise']['internal_key'] = "ig7jLMN"

ifr_summary_dic['SerieNumber_3-trend-noise'] = {}
ifr_summary_dic['SerieNumber_3-trend-noise']['internal_key'] = "bzJCZpi"

ifr_summary_dic['SerieNumber_4-trend-noise'] = {}
ifr_summary_dic['SerieNumber_4-trend-noise']['internal_key'] = "cXlFrik"



# COMMAND ----------

_random_string_ = "YishP6I"

# COMMAND ----------

get_key_by_value(ifr_summary_dic, _random_string_)

# COMMAND ----------

ifr_summary_dic["SerieNumber_0"]

# COMMAND ----------

ifr_summary_dic["SerieNumber_1"]["list_files_importance_FULL_rf"]

# COMMAND ----------

for SerieNumber in ifr_summary_dic.keys():
    # print(ifr_summary_dic[_random_string_]['internal_key'])
    print(SerieNumber)
    ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf'] = sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{ifr_summary_dic[SerieNumber]["internal_key"]}/importance_FULL_rf')], key=extract_number)

    # ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf_pd'] = []

    # for files in ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf']:
    #     ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf_pd'].append(pd.read_csv(transform_string(files)))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import avg
schema = StructType([
    StructField("Feature", StringType(), True),
    StructField("Importance", FloatType(), True)
])


# COMMAND ----------

# List of CSV file paths
csv_paths = ifr_summary_dic["SerieNumber_0"]['list_files_importance_FULL_rf']  # Add all your CSV paths here

# Initialize an empty DataFrame to union all CSVs
df_combined = spark.createDataFrame([], schema)

# Read each CSV, then union them into a single DataFrame
for csv_path in csv_paths:
    df = spark.read.csv(csv_path, header=True, schema=schema)
    df_combined = df_combined.union(df)

# Group by Feature and calculate the average Importance
df_avg_importance = df_combined.groupBy("Feature").agg(avg("Importance").alias("Average_Importance"))


# COMMAND ----------

csv_paths

# COMMAND ----------

df = spark.read.csv(csv_paths[0], header=True, schema=schema)

# COMMAND ----------



# COMMAND ----------

pd.read_csv('/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/YishP6I/importance_FULL_rf/_window-100.csv')

# COMMAND ----------

csv_paths[1]

# COMMAND ----------

      .option("header", "true")
      .schema(schemaWithCorruptedColumn)
      // mode PERMISSIVE: nulls for missing values
      .option("mode", "PERMISSIVE")
      .option("sep", ",")
      .option("quote", "\"")
      .option("escape", "\\")
      .option("ignoreLeadingWhiteSpace", "true")
      // corrupt column name
      .option("columnNameOfCorruptRecord", "_corrupt_record")
      .csv("src/main/resources/demo1/")

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").option("mode", "PERMISSIVE").option("quote",'"').option("columnNameOfCorruptRecord", "_corrupt_record").option("quote", "\"").option("ignoreLeadingWhiteSpace", "true").option("sep",",").schema(schema).load('dbfs:/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/_window-886.csv/part-00000-tid-2646239180962808764-46205746-2601-47f6-9948-23a5634ffa91-2408-1-c000.txt')
# df_importance = df.select("Importance")

df.show()

# master_data/_window-886.csv/part-00000-tid-2646239180962808764-46205746-2601-47f6-9948-23a5634ffa91-2408-1-c000.txt

# COMMAND ----------

spark.read.csv('dbfs:/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/_window-886.csv', header='true', sep=',', schema=schema).show()

# COMMAND ----------

pd.read_csv('/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/_window-886.csv', header='infer', sep=',', engine='python', quoting=3)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace


# Path to your CSV file
input_file_path = "dbfs:/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/_window-886.csv"
output_file_path = "dbfs:/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/_window-886.csv"

# Read the CSV file into a DataFrame
df = spark.read.option("header", "true").csv(input_file_path, schema=schema)

# Replace all hyphens with empty strings in all columns
for column in df.columns:
    df = df.withColumn(column, regexp_replace(col(column), "-", ""))

# Overwrite the CSV file with the modified DataFrame
df.write.mode("overwrite").option("header", "true").csv(output_file_path)




# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace
# Path to the folder containing files
input_path = "dbfs:/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/_window-886.csv"

# Read all files from the folder into a DataFrame
df = spark.read.text(input_path)

# Remove "-" from the content
df_cleaned = df.withColumn("value", regexp_replace(col("value"), "-", ""))

# Show the cleaned DataFrame (for verification)
df_cleaned.show(truncate=False)

# Optional: Write the cleaned DataFrame back to a folder
output_path = input_path
df_cleaned.write.mode("overwrite").text(output_path)

# COMMAND ----------

import pyspark.pandas as ps


# COMMAND ----------

import pandas_spark as ps

# COMMAND ----------

import pyspark.pandas as ps

# List of CSV file paths
csv_paths = ifr_summary_dic["SerieNumber_0"]['list_files_importance_FULL_rf']

# Read all CSVs in parallel into a single DataFrame
df_combined = ps.read_csv(csv_paths[0],header=str,)

# Show the combined DataFrame
df_combined.head()

# Extract just the Importance column


# Show the Importance column
# df_importance.head()


# COMMAND ----------

display(spark.read.csv(csv_paths[0], header=True, schema=schema))

# COMMAND ----------

df_combined = spark.read.csv(csv_paths[0], header=True, schema=schema).show()
# Group by Feature and calculate the average Importance
df_avg_importance = df_combined.groupBy("Feature").agg(avg("Importance").alias("Average_Importance"))

# Show the resulting DataFrame
df_avg_importance.show()

# COMMAND ----------

df.show()

# COMMAND ----------

for SerieNumber in ifr_summary_dic.keys():
    print(SerieNumber)
    ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf'] = sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{ifr_summary_dic[SerieNumber]["internal_key"]}/importance_FULL_rf')], key=extract_number)

    ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf_pd'] = []

    for files in ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf']:
        ifr_summary_dic[SerieNumber]['list_files_importance_FULL_rf_pd'].append(spark.read.csv(files, header=True, schema=schema))







# COMMAND ----------

ifr_summary_dic["SerieNumber_0"]['list_files_importance_FULL_rf']

ifr_summary_dic["SerieNumber_0"]['list_files_importance_FULL_lr']

ifr_summary_dic["SerieNumber_0"]['list_files_importance_FULL_rf_pd']

ifr_summary_dic["SerieNumber_0"]['list_files_importance_FULL_lr_pd']

try:
    freq_calculated = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/{"SerieNumber_0"}/Feature_eng/freq_calculated.csv", index_col=0)
    freq_calculated[["f1","f2"]]
    # display(freq_calculated)

    # Retrieve the index of the first row that contains a non-NaN value
    first_non_nan_index = freq_calculated.first_valid_index()

    # Convert the index to the row number
    ifr_summary_dic["SerieNumber_0"]["season_cut"]
    season_cut = freq_calculated.index.get_loc(first_non_nan_index)

    # Retrieve the index of the first row that contains a non-NaN value
    first_non_nan_index = freq_calculated["f2"].first_valid_index()

    # Convert the index to the row number
    ifr_summary_dic["SerieNumber_0"]["season_cut_2"]
    season_cut_2 = freq_calculated["f2"].index.get_loc(first_non_nan_index)

    print(f"Season cut 1: {season_cut}")
    print(f"Season cut 2: {season_cut_2}")
except:
    ifr_summary_dic["SerieNumber_0"]["season_cut"]
    season_cut = 25
    ifr_summary_dic["SerieNumber_0"]["season_cut_2"]
    season_cut_2 = 250
    print(f"Cut 1: {season_cut}")
    print(f"Cut 2: {season_cut_2}")

ifr_summary_dic["SerieNumber_0"]["full_agerage_agg"]

# Considering entire serie
importance_FULL_lr_appended = pd.DataFrame(columns=list_files_importance_FULL_lr_pd[0].columns)
for _df_ in list_files_importance_FULL_lr_pd:
    importance_FULL_lr_appended = pd.concat([importance_FULL_lr_appended, _df_], ignore_index=True)

full_agerage_agg = importance_FULL_lr_appended.groupby(['Feature']).mean().sort_values(by='Importance',ascending=False)
print("Considering entire serie")
display(full_agerage_agg.reset_index())


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

list_files_importance_FULL_rf = sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_FULL_rf')], key=extract_number) 

# list_files_importance_VEST_rf =  sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_VEST_rf')], key=extract_number)

list_files_importance_FULL_lr =  sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_FULL_lr')], key=extract_number)

# list_files_importance_VEST_lr =  sorted([sublist[0] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_VEST_lr')], key=extract_number)

list_files_importance_FULL_rf_pd = []
for files in list_files_importance_FULL_rf:
    list_files_importance_FULL_rf_pd.append(pd.read_csv(transform_string(files)))

# list_files_importance_VEST_rf_pd = []
# for files in list_files_importance_VEST_rf:
#     list_files_importance_VEST_rf_pd.append(pd.read_csv(transform_string(files)))

list_files_importance_FULL_lr_pd = []
for files in list_files_importance_FULL_lr:
    list_files_importance_FULL_lr_pd.append(pd.read_csv(transform_string(files)))

# list_files_importance_VEST_lr_pd = []
# for files in list_files_importance_VEST_lr:
#     list_files_importance_VEST_lr_pd.append(pd.read_csv(transform_string(files)))


# COMMAND ----------

try:
    freq_calculated = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/{SerieNumber}/Feature_eng/freq_calculated.csv", index_col=0)
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



# COMMAND ----------

# MAGIC %md # All Variables

# COMMAND ----------

# MAGIC %md ###Serie 0

# COMMAND ----------

# MAGIC %md
# MAGIC YishP6I

# COMMAND ----------

# MAGIC %md ###Serie 1

# COMMAND ----------

# MAGIC %md
# MAGIC mMRVk4N

# COMMAND ----------

# MAGIC %md ###Serie 2

# COMMAND ----------

# MAGIC %md
# MAGIC Grbb0dz

# COMMAND ----------

# MAGIC %md ###Serie 3

# COMMAND ----------

# MAGIC %md
# MAGIC XznRtp7

# COMMAND ----------

# MAGIC %md ###Serie 4

# COMMAND ----------

# MAGIC %md
# MAGIC VhQxMtS

# COMMAND ----------

# MAGIC %md ###Serie 0 - trend-season

# COMMAND ----------

# MAGIC %md
# MAGIC xRmAioK
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ###Serie 1 - trend-season

# COMMAND ----------

# MAGIC %md
# MAGIC 6KC6QnM
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

# MAGIC %md ###Serie 2 - trend-season

# COMMAND ----------

# MAGIC %md
# MAGIC Ef93YW6
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

# MAGIC %md ###Serie 3 - trend-season

# COMMAND ----------

# MAGIC %md
# MAGIC h94TUJC
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

# MAGIC %md ###Serie 4 - trend-season (outro workspace)

# COMMAND ----------

# MAGIC %md
# MAGIC MmpnJ41

# COMMAND ----------

#https://adb-6905234774964628.8.azuredatabricks.net/jobs/309844446891484/runs/561774085326083?o=6905234774964628

# COMMAND ----------

# MAGIC %md ###Serie 0 - Season-noise

# COMMAND ----------

# MAGIC %md
# MAGIC cGJPwDY
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

# MAGIC %md ###Serie 1 - Season-noise

# COMMAND ----------

# MAGIC %md
# MAGIC 4P9oiEy

# COMMAND ----------

# MAGIC %md ###Serie 2 - Season-noise (outro workspace)

# COMMAND ----------

# MAGIC %md
# MAGIC ckGrURE

# COMMAND ----------

# MAGIC %md ###Serie 3 - Season-noise (outro workspace)

# COMMAND ----------

# MAGIC %md
# MAGIC KLUPRil

# COMMAND ----------

# MAGIC %md
# MAGIC https://adb-6905234774964628.8.azuredatabricks.net/jobs/309844446891484/runs/653693321998623?o=6905234774964628

# COMMAND ----------

# MAGIC %md ###Serie 4 - Season-noise (outro workspace)

# COMMAND ----------

# MAGIC %md
# MAGIC rxulmxt

# COMMAND ----------

# MAGIC %md ###Serie 0 - trend-noise

# COMMAND ----------

# MAGIC %md
# MAGIC 1aMpsha

# COMMAND ----------

# MAGIC %md ###Serie 1 - trend-noise

# COMMAND ----------

# MAGIC %md
# MAGIC K6muDqb

# COMMAND ----------

# MAGIC %md ###Serie 2 - trend-noise

# COMMAND ----------

# MAGIC %md
# MAGIC ig7jLMN

# COMMAND ----------

# MAGIC %md ###Serie 3 - trend-noise (outro workspace)

# COMMAND ----------

# MAGIC %md
# MAGIC bzJCZpi

# COMMAND ----------

# MAGIC %md ###Serie 4 - trend-noise

# COMMAND ----------

# MAGIC %md
# MAGIC cXlFrik
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

# MAGIC %md # Test
