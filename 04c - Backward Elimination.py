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

# importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter("ignore")

## sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder #nao precisa
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
import sklearn.metrics as metrics
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, SparkTrials
from hyperopt.pyll.base import scope

import mlflow

from mlflow.models.signature import infer_signature

from logic import *



# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection

# COMMAND ----------

def backward_elimination(x_train, y_train, _model_):
    remaining_features = x_train.columns.tolist()    
    best_features = remaining_features[:]
    min_smape = float('inf')
    
    while len(remaining_features) > 0:
        smape_dict = {}
        for feature in remaining_features:
            temp_features = remaining_features[:]
            temp_features.remove(feature)
                       
            model = _model_.fit(x_train, y_train)
            predictions = model.predict(x_train)
            
            smape =calculate_smape(y_train, predictions)

            smape_dict[feature] = smape
        
        worst_feature = max(smape_dict, key=smape_dict.get)
        worst_smape = smape_dict[worst_feature]
        
        if worst_smape < min_smape:
            min_smape = worst_smape
            remaining_features.remove(worst_feature)
            best_features = remaining_features[:]
        else:
            break
    
    return best_features


# COMMAND ----------

def forward_elimination(x_train, y_train, _model_):
    initial_features = x_train.columns.tolist()
    selected_features = []
    min_smape = float('inf')
    
    while len(selected_features) < len(initial_features):
        smape_dict = {}
        
        for feature in initial_features:
            if feature in selected_features:
                continue
            
            temp_features = selected_features + [feature]
            model = _model_.fit(x_train[temp_features], y_train)
            predictions = model.predict(x_train[temp_features])
            
            smape = calculate_smape(y_train, predictions)
            smape_dict[feature] = smape
        
        best_feature = min(smape_dict, key=smape_dict.get)
        best_smape = smape_dict[best_feature]
        
        if best_smape < min_smape:
            min_smape = best_smape
            selected_features.append(best_feature)
        else:
            break
    
    return selected_features

# Example usage
# Assuming 'df' is your DataFrame and 'target' is the name of the target variable column
# X = df.drop(columns='target')
# y = df['target']

# Example model
from sklearn.linear_model import LinearRegression
# model = LinearRegression()

# selected_features = forward_elimination(X, y, model)
# print("Selected features:", selected_features)


# COMMAND ----------

def find_missing_elements(list1, list2):
    return list(set(list1) - set(list2))

# Example usage
# list1 = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
# list2 = ["banana", "cherry", "apple", "fig", "grape"]
# missing_elements = find_missing_elements(list1, list2)
# print(missing_elements)  # Output will be ['date', 'elderberry']


# COMMAND ----------

class feature_elimination:
      def __init__(self, SerieNumber, decorator):
        self.SerieNumber = SerieNumber

        self.decorator = decorator
        try:
            self.freq_calculated = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/Feature_eng/freq_calculated.csv", index_col=0)
            self.freq_calculated = self.freq_calculated[["f1","f2"]]
        except: pass


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

        self.list_bk_lr = []
        self.list_bk_lr_missing = []
        self.list_bk_knn = []
        self.list_bk_knn_missing  = []
        self.list_bk_rf= []
        self.list_bk_rf_missing = []

        self.list_fw_lr = []
        self.list_fw_lr_missing = []
        self.list_fw_knn = []
        self.list_fw_knn_missing  = []
        self.list_fw_rf= []
        self.list_fw_rf_missing = []

        self.summary_back_frame = 0
        self.summary_foward_frame = 0

      def back_summary_elimination(self):
          
          for k in range(len(self.order)):
            print("################################################################################################################################################################")

            print(f"############################################################################ {self.order[k]} #################################################################################")

            print("################################################################################################################################################################")

            print(f"######################################################################### {self.order[k]} LR #########################################################################")

            features_ns = self.input_variables[1][k].columns.tolist()
            print(f"total features: {features_ns}")
            print(f"ammout of total features: {len(features_ns)}")
                      
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.input_variables[1][k], self.input_variables[0][k], test_size=0.2, random_state=0)

            lr_features = backward_elimination(X_train, y_train,LinearRegression())
            print(f"LR: ammount of selected features: {len(lr_features)}")
            print(f"Removed from LR: {find_missing_elements(features_ns,lr_features)}")

            self.list_bk_lr.append(lr_features)
            self.list_bk_lr_missing.append(find_missing_elements(features_ns,lr_features))


            print(f"######################################################################### {self.order[k]} KNN #########################################################################")

            # Split the data into training and test sets

            knn_features = backward_elimination(X_train, y_train,KNeighborsRegressor())
            print(f"KNN: ammount of selected features: {len(knn_features)}")
            print(f"Removed from KNN: {find_missing_elements(features_ns,knn_features)}")

            self.list_bk_knn.append(knn_features)
            self.list_bk_knn_missing.append(find_missing_elements(features_ns,knn_features))

            print(f"######################################################################### {self.order[k]} RF #########################################################################")

            rf_features = backward_elimination(X_train, y_train,RandomForestRegressor())
            print(f"RF: ammount of selected features: {len(rf_features)}")
            print(f"Removed from RF: {find_missing_elements(features_ns,rf_features)}")

            self.list_bk_rf.append(rf_features)
            self.list_bk_rf_missing.append(find_missing_elements(features_ns,rf_features))
        
          data = {
            'type': [_feature_elimination_.order[0], _feature_elimination_.order[1] , _feature_elimination_.order[2]],
            'Points': [_feature_elimination_.order_points[0], _feature_elimination_.order_points[1], _feature_elimination_.order_points[2]],
            'LR features': [_feature_elimination_.list_bk_lr[0], _feature_elimination_.list_bk_lr[1], _feature_elimination_.list_bk_lr[2]],
            'LR features removed': [_feature_elimination_.list_bk_lr_missing[0], _feature_elimination_.list_bk_lr_missing[1], _feature_elimination_.list_bk_lr_missing[2]],
            'KNN features': [_feature_elimination_.list_bk_knn[0], _feature_elimination_.list_bk_knn[1], _feature_elimination_.list_bk_knn[2]],
            'KNN features removed': [_feature_elimination_.list_bk_knn_missing[0], _feature_elimination_.list_bk_knn_missing[1], _feature_elimination_.list_bk_knn_missing[2]],
            'RF features': [_feature_elimination_.list_bk_rf[0], _feature_elimination_.list_bk_rf[1], _feature_elimination_.list_bk_rf[2]],
            'RF features removed': [_feature_elimination_.list_bk_rf_missing[0], _feature_elimination_.list_bk_rf_missing[1], _feature_elimination_.list_bk_rf_missing[2]]
            }

          self.summary_back_frame = pd.DataFrame(data)


      def foward_summary_elimination(self):
          
          for k in range(len(self.order)):
            print("################################################################################################################################################################")

            print(f"############################################################################ {self.order[k]} #################################################################################")

            print("################################################################################################################################################################")

            print(f"######################################################################### {self.order[k]} LR #########################################################################")

            features_ns = self.input_variables[1][k].columns.tolist()
            print(f"total features: {features_ns}")
            print(f"ammout of total features: {len(features_ns)}")
                      
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.input_variables[1][k], self.input_variables[0][k], test_size=0.2, random_state=0)

            lr_features = forward_elimination(X_train, y_train,LinearRegression())
            print(f"LR: ammount of selected features: {len(lr_features)}")
            print(f"Removed from LR: {find_missing_elements(features_ns,lr_features)}")

            self.list_fw_lr.append(lr_features)
            self.list_fw_lr_missing.append(find_missing_elements(features_ns,lr_features))


            print(f"######################################################################### {self.order[k]} KNN #########################################################################")

            # Split the data into training and test sets

            knn_features = forward_elimination(X_train, y_train,KNeighborsRegressor())
            print(f"KNN: ammount of selected features: {len(knn_features)}")
            print(f"Removed from KNN: {find_missing_elements(features_ns,knn_features)}")

            self.list_fw_knn.append(knn_features)
            self.list_fw_knn_missing.append(find_missing_elements(features_ns,knn_features))

            print(f"######################################################################### {self.order[k]} RF #########################################################################")

            rf_features = forward_elimination(X_train, y_train,RandomForestRegressor())
            print(f"RF: ammount of selected features: {len(rf_features)}")
            print(f"Removed from RF: {find_missing_elements(features_ns,rf_features)}")

            self.list_fw_rf.append(rf_features)
            self.list_fw_rf_missing.append(find_missing_elements(features_ns,rf_features))

          data = {
            'type': [_feature_elimination_.order[0], _feature_elimination_.order[1] , _feature_elimination_.order[2]],
            'Points': [_feature_elimination_.order_points[0], _feature_elimination_.order_points[1], _feature_elimination_.order_points[2]],
            'LR features': [_feature_elimination_.list_fw_lr[0], _feature_elimination_.list_fw_lr[1], _feature_elimination_.list_fw_lr[2]],
            'LR features removed': [_feature_elimination_.list_fw_lr_missing[0], _feature_elimination_.list_fw_lr_missing[1], _feature_elimination_.list_fw_lr_missing[2]],
            'KNN features': [_feature_elimination_.list_fw_knn[0], _feature_elimination_.list_fw_knn[1], _feature_elimination_.list_fw_knn[2]],
            'KNN features removed': [_feature_elimination_.list_fw_knn_missing[0], _feature_elimination_.list_fw_knn_missing[1], _feature_elimination_.list_fw_knn_missing[2]],
            'RF features': [_feature_elimination_.list_fw_rf[0], _feature_elimination_.list_fw_rf[1], _feature_elimination_.list_fw_rf[2]],
            'RF features removed': [_feature_elimination_.list_fw_rf_missing[0], _feature_elimination_.list_fw_rf_missing[1], _feature_elimination_.list_fw_rf_missing[2]]
            }

          self.summary_foward_frame = pd.DataFrame(data)

      def store_backward_frame(self):
          self.summary_back_frame.to_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/Feature_eng/summary_back_frame.csv")
      

      def store_foward_frame(self):
          self.summary_foward_frame.to_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/Feature_eng/summary_foward_frame.csv")


# COMMAND ----------

_feature_elimination_ = feature_elimination(SerieNumber,decorator)

# COMMAND ----------

_feature_elimination_.back_summary_elimination()

# COMMAND ----------

_feature_elimination_.foward_summary_elimination()

# COMMAND ----------

display(_feature_elimination_.summary_foward_frame)

# COMMAND ----------

display(_feature_elimination_.summary_back_frame)

# COMMAND ----------

_feature_elimination_.store_foward_frame()


# COMMAND ----------

_feature_elimination_.store_backward_frame()
