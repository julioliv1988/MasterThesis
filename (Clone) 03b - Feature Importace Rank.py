# Databricks notebook source
# MAGIC %md
# MAGIC # Widgets

# COMMAND ----------

dbutils.widgets.text("SerieNumber",defaultValue="0")
dbutils.widgets.text("Modified_serie",defaultValue="False")

# COMMAND ----------

import ast
Chosen_Model = dbutils.widgets.get("Chosen Model")
TS_truncated_experiment = dbutils.widgets.get("TS_truncated_experiment") in ['True']
SerieNumber = dbutils.widgets.get('SerieNumber')
Modified_serie = ast.literal_eval(dbutils.widgets.get('Modified_serie'))

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

from hyperopt import fmin, tpe, hp, SparkTrials
from hyperopt.pyll.base import scope

import mlflow

from mlflow.models.signature import infer_signature

from logic import *



# COMMAND ----------

_random_string_ = "test002"

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection

# COMMAND ----------

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from sklearn.inspection import permutation_importance

# COMMAND ----------

# Step 1: Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# COMMAND ----------

# Step 4: Compute the permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0, scoring='neg_mean_squared_error')

# COMMAND ----------


# Step 5: Analyze and visualize the results
# Create a DataFrame to hold the results
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean, 'Std': result.importances_std})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'])
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Feature Importance using Permutation on Test Set')
plt.gca().invert_yaxis()
plt.show()

# Optionally, print the results
print(importance_df)


# COMMAND ----------

# Step 1: Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Step 2: Prepare your data
# Create a synthetic dataset (replace this with your actual dataset)
np.random.seed(0)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = X @ np.array([1.5, -2.0, 0.5, 1.0, -1.0]) + np.random.randn(100) * 0.5  # Linear combination + noise

# Check the dimensions of X and y
print("Shape of X:", X.shape)  # Should be (100, 5)
print("Shape of y:", y.shape)  # Should be (100,)

# COMMAND ----------

y

# COMMAND ----------




# Split the data into training and test sets (ensuring multiple samples in each)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 3: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Compute the permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0, scoring='neg_mean_squared_error')

# Step 5: Analyze and visualize the results
# Create a DataFrame to hold the results
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': result.importances_mean, 'Std': result.importances_std})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'])
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Feature Importance using Permutation on Test Set')
plt.gca().invert_yaxis()
plt.show()

# Optionally, print the results
print(importance_df)


# COMMAND ----------

# MAGIC %md
# MAGIC #Serie Number 0

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

class feature_rank:
      def __init__(self, SerieNumber, decorator):
        self.SerieNumber = SerieNumber

        self.decorator = decorator

        self.freq_calculated = pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{self.SerieNumber}{self.decorator}/Feature_eng/freq_calculated.csv", index_col=0)
        self.freq_calculated = self.freq_calculated[["f1","f2"]]



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
   
      def decomp_serie(self,i):
          pass

      def decomp_fft(self,i):
          pass

# COMMAND ----------

feature_rank_serie_0 = feature_rank(0,"")

# COMMAND ----------

_feature_rank_.summary_plot()

# COMMAND ----------

pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{0}/train_set_featured/window_{236}.csv", index_col=0)

# COMMAND ----------

print(f"season_cut: {_feature_rank_.season_cut}")

print(f"season_cut_2: {_feature_rank_.season_cut_2}")

print(f"order: {_feature_rank_.order}")

print(f"order_points: {_feature_rank_.order_points}")

# COMMAND ----------

display(_feature_rank_.input_variables[1][0])
# display(_feature_rank_.input_variables[0][1])
# display(_feature_rank_.input_variables[0][2])

# COMMAND ----------

display(_feature_rank_.freq_calculated)
