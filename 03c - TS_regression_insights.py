# Databricks notebook source
dbutils.widgets.dropdown("Chosen Model", "Random Forest",["Linear Regression", "KNN", "Random Forest" ],label=None)
dbutils.widgets.text("SerieNumber",defaultValue="0")

# COMMAND ----------

SerieNumber = dbutils.widgets.get('SerieNumber')
Chosen_Model = dbutils.widgets.get("Chosen Model")

# COMMAND ----------

# MAGIC %md
# MAGIC #TS_regression

# COMMAND ----------

# importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import warnings
warnings.simplefilter('ignore', np.RankWarning)

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

from logic import *



# COMMAND ----------

import warnings

# Define a filter function to suppress specific warnings
def suppress_mlflow_signature_warning(message, category, filename, lineno, file=None, line=None):
    if "Model logged without a signature" in str(message):
        return True
    return False

# Register the filter function with the warnings module
warnings.filterwarnings("always", category=UserWarning)
warnings.showwarning = suppress_mlflow_signature_warning

# COMMAND ----------

# Setting variables/env

# Checking the ammount of windows
# ammount_of_windows("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/test_set/")
n_windows = ammount_of_windows("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}/train_set/")

experiment_name = "/Users/jdeoliveira@microsoft.com/Master-Experiment"

# Set the experiment
mlflow.set_experiment(experiment_name)

train_featured_no_season_list = []
test_featured_no_season_list = []

for i in range(n_windows):
    test_featured_no_season_list.append(pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}/test_set_featured/window_{i}.csv", index_col=0))
    train_featured_no_season_list.append(pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}/train_set_featured/window_{i}.csv", index_col=0))


input_variables = train_test_convert_2_regressor(train_featured_no_season_list,test_featured_no_season_list)

experiment_path_ = "/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/"

# COMMAND ----------

print(Chosen_Model)

# COMMAND ----------

class timeseries_solver:
      def __init__(self):

        self.y_train_list_expanding = 0
        self.x_train_list_expanding = 0
        self.y_test_list_expanding = 0
        self.x_test_list_expanding = 0

        self.list_of_models = []


      def fit_expanding(self, name_model):


        self.list_of_models.append([name_model])

        #Fitting Expanding
        pred_ew = pd.DataFrame(self.y_test_list_expanding[0])
        pred_ew["forcast"] = np. zeros(1)
        pred_ew = pred_ew[0:0]

        ew_list = [] #regressor
        rfe_list = []
        forcast_list = []
        study_list = []
        trial_list = []
        params_list = []
        
        # Need to hyper-tune a fracion of the windows - every 80 window p.e.

        if name_model == "Linear Regression":
          with mlflow.start_run(run_name=f"LR_predict_FULL-Hyperopt-multiple") as run:
            for i in range(int(len(self.y_train_list_expanding))):
              (x_train,y_train,x_test,y_test) = (self.x_train_list_expanding[i],self.y_train_list_expanding[i],self.x_test_list_expanding[i],self.y_test_list_expanding[i])

              if i % 80 == 0:              
                ########################################################### Hyperparameter tune start ###########################################################
                # Define search space
                search_space = {"fit_intercept": hp.choice('fit_intercept', [True, False]),
                                "positive": hp.choice('positive', [True, False])}

                model = LinearRegression()

                def objective_lr(params):
                    model = LinearRegression(fit_intercept=bool(params["fit_intercept"]), 
                                                  positive=bool(params["positive"]))
                    model.fit(x_train, y_train)
                    pred = model.predict(x_train)
                    score = calculate_smape(pred, y_train)
                    # Hyperopt minimizes score, here we minimize mse. 
                    return score

                # Set parallelism (should be order of magnitude smaller than max_evals)
                spark_trials = SparkTrials(parallelism=4)

                with mlflow.start_run(run_name=f"LR_predict_FULL_window-{i}",nested=True):
                    argmin = fmin(fn=objective_lr,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=4,
                                  trials=spark_trials)
                    
                ########################################################### Hyperparameter tune end ###########################################################
                argmin = {key: bool(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"LR_predict_FULL_window-{i}-chosen",nested=True):
                params_list.append(argmin)
                ew_list.append(model.set_params(**argmin))
                ew_list[i].fit(self.x_train_list_expanding[i],self.y_train_list_expanding[i])
                forcast_list.append(ew_list[i].predict(self.x_test_list_expanding[i]))
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pred_ew.append(temp)
                mlflow.sklearn.log_model(ew_list[i], "lr_model-mindow-{i}")

            random_string = generate_random_string(7)
            forcast_path = f"{experiment_path_}forcast_FULL_lr-{random_string}.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_FULL_lr-{random_string}.csv")

            no_season_mae = metrics.mean_absolute_error(pred_ew.observation, pred_ew.forcast)
            no_season_mse = metrics.mean_squared_error(pred_ew.observation, pred_ew.forcast)
            no_season_rmse = np.sqrt(no_season_mse) # or mse**(0.5)
            no_season_r2 = metrics.r2_score(pred_ew.observation, pred_ew.forcast)
            no_season_smape = calculate_smape(pred_ew.observation, pred_ew.forcast)            
            mlflow.log_metric("mae", no_season_mae)
            mlflow.log_metric("mse", no_season_mse)
            mlflow.log_metric("rmse", no_season_rmse)
            mlflow.log_metric("r2", no_season_r2)
            mlflow.log_metric("smape", no_season_smape)

            plt = plot_forcast_vs_obs(pred_ew,f"Linear Regression")
            mlflow.log_figure(plt, f"forcast_FULL_lr-{random_string}.png")

        elif name_model == "KNN":
          with mlflow.start_run(run_name=f"KNN_predict_FULL") as run:
            for i in range(int(len(self.y_train_list_expanding))):
              (x_train,y_train,x_test,y_test) = (self.x_train_list_expanding[i],self.y_train_list_expanding[i],self.x_test_list_expanding[i],self.y_test_list_expanding[i])

              if i % 80 == 0:
                ########################################################### Hyperparameter tune start ###########################################################
                # Define search space
                search_space = {"n_neighbors": hp.uniformint('n_neighbors', 1, 5, q=1),
                                "leaf_size": hp.uniformint('leaf_size',  5, 21, q=3),
                                "p": hp.uniformint('p',  5, 21, q=3)}

                model = KNeighborsRegressor()

                def objective_knn(params):
                    model = KNeighborsRegressor(n_neighbors=int(params["n_neighbors"]), 
                                                  leaf_size=params["leaf_size"],
                                                  p=params["p"])                          
                    model.fit(x_train, y_train)
                    pred = model.predict(x_train)
                    score = calculate_smape(pred, y_train)
                    # Hyperopt minimizes score, here we minimize smape. 
                    return score
                  
                # Set parallelism (should be order of magnitude smaller than max_evals)
                spark_trials = SparkTrials(parallelism=25)

                with mlflow.start_run(run_name=f"KNN_predict_FULL_window-{i}",nested=True):
                    argmin = fmin(fn=objective_knn,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=25,
                                  trials=spark_trials)
                ########################################################### Hyperparameter tune end ###########################################################    
                argmin = {key: int(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"KNN_predict_FULL_window-{i}-chosen",nested=True):
                params_list.append(argmin)
                ew_list.append(model.set_params(**argmin))
                ew_list[i].fit(self.x_train_list_expanding[i],self.y_train_list_expanding[i])
                forcast_list.append(ew_list[i].predict(self.x_test_list_expanding[i]))
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pred_ew.append(temp)
                mlflow.sklearn.log_model(ew_list[i], "knn_model-mindow-{i}")

            random_string = generate_random_string(7)
            forcast_path = f"{experiment_path_}forcast_FULL_knn-{random_string}.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_FULL_knn-{random_string}.csv")

            no_season_mae = metrics.mean_absolute_error(pred_ew.observation, pred_ew.forcast)
            no_season_mse = metrics.mean_squared_error(pred_ew.observation, pred_ew.forcast)
            no_season_rmse = np.sqrt(no_season_mse) # or mse**(0.5)
            no_season_r2 = metrics.r2_score(pred_ew.observation, pred_ew.forcast)
            no_season_smape = calculate_smape(pred_ew.observation, pred_ew.forcast)            
            mlflow.log_metric("mae", no_season_mae)
            mlflow.log_metric("mse", no_season_mse)
            mlflow.log_metric("rmse", no_season_rmse)
            mlflow.log_metric("r2", no_season_r2)
            mlflow.log_metric("smape", no_season_smape)

            plt = plot_forcast_vs_obs(pred_ew,f"KNN")
            mlflow.log_figure(plt, f"forcast_FULL_knn-{random_string}.png")                      
        
        elif name_model == "Random Forest":
          with mlflow.start_run(run_name=f"RF_predict_FULL") as run:
            for i in range(int(len(self.y_train_list_expanding))):
              (x_train,y_train,x_test,y_test) = (self.x_train_list_expanding[i],self.y_train_list_expanding[i],self.x_test_list_expanding[i],self.y_test_list_expanding[i])

              if i % 80 == 0:
                ########################################################### Hyperparameter tune start ###########################################################
                # Define search space
                search_space = {"n_estimators": hp.quniform("n_estimators", 100, 500, 5),
                                "max_depth": hp.quniform("max_depth", 5, 20, 1),
                                "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
                                "min_samples_split": hp.quniform("min_samples_split", 2, 6, 1)}
                
                model = RandomForestRegressor()

                def objective_rf(params):
                  model = RandomForestRegressor(n_estimators=int(params["n_estimators"]), 
                                                max_depth=int(params["max_depth"]), 
                                                min_samples_leaf=int(params["min_samples_leaf"]),
                                                min_samples_split=int(params["min_samples_split"]))
                  model.fit(x_train, y_train)
                  pred = model.predict(x_train)
                  score = calculate_smape(pred, y_train)
                  # Hyperopt minimizes score, here we minimize smape. 
                  return score

                # Set parallelism (should be order of magnitude smaller than max_evals)
                spark_trials = SparkTrials(parallelism=25)

                with mlflow.start_run(run_name=f"RF_predict_FULL_window-{i}",nested=True):
                    argmin = fmin(fn=objective_rf,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=25,
                                  trials=spark_trials)
                ########################################################### Hyperparameter tune end ###########################################################                     
                argmin = {key: int(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"RF_predict_FULL_window-{i}-chosen",nested=True):
                params_list.append(argmin)
                ew_list.append(model.set_params(**argmin))
                ew_list[i].fit(self.x_train_list_expanding[i],self.y_train_list_expanding[i])
                forcast_list.append(ew_list[i].predict(self.x_test_list_expanding[i]))
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pred_ew.append(temp)
                mlflow.sklearn.log_model(ew_list[i], "rf_model-mindow-{i}")

            random_string = generate_random_string(7)
            forcast_path = f"{experiment_path_}forcast_FULL_rf-{random_string}.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_FULL_rf-{random_string}.csv")

            no_season_mae = metrics.mean_absolute_error(pred_ew.observation, pred_ew.forcast)
            no_season_mse = metrics.mean_squared_error(pred_ew.observation, pred_ew.forcast)
            no_season_rmse = np.sqrt(no_season_mse) # or mse**(0.5)
            no_season_r2 = metrics.r2_score(pred_ew.observation, pred_ew.forcast)
            no_season_smape = calculate_smape(pred_ew.observation, pred_ew.forcast)            
            mlflow.log_metric("mae", no_season_mae)
            mlflow.log_metric("mse", no_season_mse)
            mlflow.log_metric("rmse", no_season_rmse)
            mlflow.log_metric("r2", no_season_r2)
            mlflow.log_metric("smape", no_season_smape)

            plt = plot_forcast_vs_obs(pred_ew,f"Random Forest")
            mlflow.log_figure(plt, f"forcast_FULL_rf-{random_string}.png")


        self.list_of_models[-1].append(ew_list)
        self.list_of_models[-1].append(pred_ew)



# COMMAND ----------

serie_solv = timeseries_solver()

serie_solv.y_train_list_expanding = input_variables[0]
serie_solv.x_train_list_expanding = input_variables[1]
serie_solv.y_test_list_expanding = input_variables[2]
serie_solv.x_test_list_expanding = input_variables[3]

# COMMAND ----------

serie_solv.fit_expanding(Chosen_Model)
