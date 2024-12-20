# Databricks notebook source
# MAGIC %md
# MAGIC # Widgets

# COMMAND ----------

dbutils.widgets.text("SerieNumber",defaultValue="0")
dbutils.widgets.text("Modified_serie",defaultValue="False")

# COMMAND ----------

import ast
SerieNumber = dbutils.widgets.get('SerieNumber')
Modified_serie = ast.literal_eval(dbutils.widgets.get('Modified_serie'))

# COMMAND ----------

if not Modified_serie:
    decorator = ""
else:
    decorator = f"-{Modified_serie[0]}-{Modified_serie[1]}"

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports
# MAGIC

# COMMAND ----------

import numpy as np
import pandas as pd
from logic import *
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Rank

# COMMAND ----------

class feature_rank:
      def __init__(self, SerieNumber, decorator):
        self.SerieNumber = SerieNumber

        self.decorator = decorator
        try:
            self.freq_calculated = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/Feature_eng/freq_calculated.csv", index_col=0)
            self.freq_calculated = self.freq_calculated[["f1","f2"]]
        except: pass

        self.path = f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/Feature_eng/"

        # display(freq_calculated)
        try:

            # Retrieve the index of the first row that contains a non-NaN value
            first_non_nan_index = self.freq_calculated.first_valid_index()

            # Convert the index to the row number
            self.season_cut = self.freq_calculated.index.get_loc(first_non_nan_index)

            # Retrieve the index of the first row that contains a non-NaN value
            first_non_nan_index = self.freq_calculated["f2"].first_valid_index()

            # Convert the index to the row number
            self.season_cut_2 = self.freq_calculated["f2"].index.get_loc(first_non_nan_index)

            self.order = ["No season", "1 season", "2 season"]

            self.order_points =[self.season_cut-1, self.season_cut_2-1, 959]

            # print(f"Season cut 1: {self.season_cut}") # No season
            # print(f"Season cut 2: {self.season_cut_2}") # 1 season
            # 2 seasons -> len of the Dataset
        except:
            self.season_cut = 160
            self.season_cut_2 = 480
            # print(f"Cut 1: {season_cut}")
            # print(f"Cut 2: {season_cut_2}")
            # cut 3 -> 800

            self.order = ["window-160", "window-480", "window-800"]
            self.order_points =[160, 480, 800]


        # Setting variables/env

        # Checking the ammount of windows
        # ammount_of_windows("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/test_set/")
        n_windows = ammount_of_windows(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/train_set/")

        train_featured_no_season_list = []
        test_featured_no_season_list = []

        for i in self.order_points:
            test_featured_no_season_list.append(pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/test_set_featured/window_{i}.csv", index_col=0))
            train_featured_no_season_list.append(pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/train_set_featured/window_{i}.csv", index_col=0))

        self.input_variables = train_test_convert_2_regressor(train_featured_no_season_list,test_featured_no_season_list)

      def summary_plot(self):
          for k in range(len(self.order)):
            print("################################################################################################################################################################")

            print(f"############################################################################ {self.order[k]} #################################################################################")

            print("################################################################################################################################################################")

            print(f"######################################################################### {self.order[k]} LR #########################################################################")
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.input_variables[1][k], self.input_variables[0][k], test_size=0.2, random_state=0)
            model = LinearRegression().fit(X_train, y_train)
            result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0, scoring='neg_mean_squared_error')

            # Step 5: Analyze and visualize the results
            # Create a DataFrame to hold the results
            features_ns = self.input_variables[1][k].columns.tolist()
            feature_names = [f'feature_{i}' for i in features_ns]
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean, 'Std': result.importances_std}).sort_values(by='Importance', ascending=False)

            display(importance_df)

            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'])
            plt.xlabel('Permutation Importance')
            plt.ylabel('Feature')
            plt.title(f'{self.order[k]} LR')
            plt.gca().invert_yaxis()
            plt.show()

            importance_df.to_csv(f'{self.path}FIR {self.order[k]} LR', index=False)

            print(f"######################################################################### {self.order[k]} KNN #########################################################################")
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.input_variables[1][k], self.input_variables[0][k], test_size=0.2, random_state=0)
            model = KNeighborsRegressor().fit(X_train, y_train)
            result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0, scoring='neg_mean_squared_error')

            # Step 5: Analyze and visualize the results
            # Create a DataFrame to hold the results
            features_ns = self.input_variables[1][k].columns.tolist()
            feature_names = [f'feature_{i}' for i in features_ns]
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean, 'Std': result.importances_std}).sort_values(by='Importance', ascending=False)

            display(importance_df)

            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'])
            plt.xlabel('Permutation Importance')
            plt.ylabel('Feature')
            plt.title(f'{self.order[k]} KNN')
            plt.gca().invert_yaxis()
            plt.show()

            importance_df.to_csv(f'{self.path}FIR {self.order[k]} KNN', index=False)

            print(f"######################################################################### {self.order[k]} RF #########################################################################")
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.input_variables[1][k], self.input_variables[0][k], test_size=0.2, random_state=0)
            model = RandomForestRegressor().fit(X_train, y_train)
            result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0, scoring='neg_mean_squared_error')

            # Step 5: Analyze and visualize the results
            # Create a DataFrame to hold the results
            features_ns = self.input_variables[1][k].columns.tolist()
            feature_names = [f'feature_{i}' for i in features_ns]
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean, 'Std': result.importances_std}).sort_values(by='Importance', ascending=False)

            display(importance_df)

            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'])
            plt.xlabel('Permutation Importance')
            plt.ylabel('Feature')
            plt.title(f'{self.order[k]} RF')
            plt.gca().invert_yaxis()
            plt.show()
            
            importance_df.to_csv(f'{self.path}FIR {self.order[k]} RF', index=False)
   
      def decomp_serie(self,i):
          pass

      def decomp_fft(self,i):
          pass

# COMMAND ----------

_feature_rank_ = feature_rank(SerieNumber,decorator)

# COMMAND ----------

_feature_rank_.summary_plot()

# COMMAND ----------

# pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}{decorator}/train_set_featured/window_{_feature_rank_.season_cut}.csv", index_col=0)

# COMMAND ----------

print(f"season_cut: {_feature_rank_.season_cut}")

print(f"season_cut_2: {_feature_rank_.season_cut_2}")

print(f"order: {_feature_rank_.order}")

print(f"order_points: {_feature_rank_.order_points}")

# COMMAND ----------

display(_feature_rank_.input_variables[1][0])

# COMMAND ----------

display(_feature_rank_.input_variables[0][1])

# COMMAND ----------

display(_feature_rank_.input_variables[0][2])

# COMMAND ----------

try:
    display(_feature_rank_.freq_calculated)

except: pass
