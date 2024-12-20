# Databricks notebook source
dbutils.widgets.dropdown("Chosen Model", "Random Forest",["Linear Regression", "KNN", "Random Forest", "SES" ],label=None)
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing



# COMMAND ----------

# Setting variables/env

# Checking the ammount of windows
# ammount_of_windows("/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/test_set/")
n_windows = ammount_of_windows(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}{decorator}/train_set/")

experiment_name = "/Users/jdeoliveira@microsoft.com/Master-Experiment"

# Set the experiment
mlflow.set_experiment(experiment_name)

train_featured_list = []
test_featured_list = []

for i in range(n_windows):
    test_featured_list.append(pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}{decorator}/test_set_featured/window_{i}.csv", index_col=0))
    train_featured_list.append(pd.read_csv(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/SerieNumber_{SerieNumber}{decorator}/train_set_featured/window_{i}.csv", index_col=0))


input_variables = train_test_convert_2_regressor(train_featured_list,test_featured_list)

experiment_path_ = "/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/"

# COMMAND ----------

Chosen_Model = 'SES'

# COMMAND ----------

try:
    _random_string_ = dbutils.jobs.taskValues.get(taskKey="TS_FeatureEng", key="job_reference")
    print(f"Retrieved from TS_FeatureEng task:{_random_string_}")
except:
    print("No reference found - creating new one")
    _random_string_ = generate_random_string(7)
    print(_random_string_)
    dbutils.fs.mkdirs(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}')
    # dbutils.jobs.taskValues.set(key="job_reference", value=_random_string_)

# COMMAND ----------

_random_string_

# COMMAND ----------

class timeseries_solver:
      def __init__(self, data, _SerieNumber_):

        self.y_train_list_expanding = data[0]
        self.x_train_list_expanding = data[1]
        self.y_test_list_expanding = data[2]
        self.x_test_list_expanding = data[3]

        self.SerieNumber = _SerieNumber_
        self.list_of_models = []

      def fit(self, name_model):


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
          with mlflow.start_run(run_name=f"LR_predict_VEST") as run:
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

                with mlflow.start_run(run_name=f"LR_predict_VEST_window-{i}",nested=True):
                    argmin = fmin(fn=objective_lr,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=4,
                                  trials=spark_trials)
                    
                ########################################################### Hyperparameter tune end ###########################################################
                argmin = {key: bool(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"LR_predict_VEST_window-{i}-chosen",nested=True):
                params_list.append(argmin)
                ew_list.append(model.set_params(**argmin))
                ew_list[i].fit(self.x_train_list_expanding[i],self.y_train_list_expanding[i])
                forcast_list.append(ew_list[i].predict(self.x_test_list_expanding[i]))
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pd.concat([pred_ew, temp], ignore_index=True)
                mlflow.sklearn.log_model(ew_list[i], "lr_model-mindow-{i}")
                mlflow.log_params(argmin)
                
                # Log feature importance
                importance = (pd.DataFrame(list(zip(self.x_train_list_expanding[i].columns, ew_list[i].coef_)), columns=["Feature", "Importance"])
                              .sort_values("Importance", ascending=False))
                
                dbutils.fs.mkdirs(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_VEST_lr')
                importance_path = f"{experiment_path_}Experiments_info/{_random_string_}/importance_VEST_lr/_window-{i}.csv"
                importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "importance_VEST_lr_window-{i}-{_random_string_}.csv")              

            # Log a tag
            mlflow.set_tag("iternal_id", _random_string_)
            mlflow.set_tag("dataset", self.SerieNumber)

            forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_VEST_lr.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_VEST_lr-{_random_string_}.csv")

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

            plt = plot_forcast_vs_obs(pred_ew,f"Linear Regression VEST")
            mlflow.log_figure(plt, f"forcast_VEST_lr-{_random_string_}.png")

        elif name_model == "KNN":
          with mlflow.start_run(run_name=f"KNN_predict_VEST") as run:
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

                with mlflow.start_run(run_name=f"KNN_predict_VEST_window-{i}",nested=True):
                    argmin = fmin(fn=objective_knn,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=25,
                                  trials=spark_trials)
                ########################################################### Hyperparameter tune end ###########################################################    
                argmin = {key: int(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"KNN_predict_VEST_window-{i}-chosen",nested=True):
                params_list.append(argmin)
                ew_list.append(model.set_params(**argmin))
                ew_list[i].fit(self.x_train_list_expanding[i],self.y_train_list_expanding[i])
                forcast_list.append(ew_list[i].predict(self.x_test_list_expanding[i]))
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pd.concat([pred_ew, temp], ignore_index=True)  
                mlflow.sklearn.log_model(ew_list[i], "knn_model-mindow-{i}")
                mlflow.log_params(argmin)

            # Log a tag
            mlflow.set_tag("iternal_id", _random_string_)
            mlflow.set_tag("dataset", self.SerieNumber)

            # random_string = generate_random_string(7)            
            forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_VEST_knn.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_VEST_knn-{_random_string_}.csv")

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

            plt = plot_forcast_vs_obs(pred_ew,f"KNN VEST")
            mlflow.log_figure(plt, f"forcast_VEST_knn-{_random_string_}.png")                      
        
        elif name_model == "Random Forest":
          with mlflow.start_run(run_name=f"RF_predict_VEST") as run:
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

                with mlflow.start_run(run_name=f"RF_predict_VEST_window-{i}",nested=True):
                    argmin = fmin(fn=objective_rf,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=25,
                                  trials=spark_trials)
                ########################################################### Hyperparameter tune end ###########################################################                     
                argmin = {key: int(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"RF_predict_VEST_window-{i}-chosen",nested=True):
                params_list.append(argmin)
                ew_list.append(model.set_params(**argmin))
                ew_list[i].fit(self.x_train_list_expanding[i],self.y_train_list_expanding[i])
                forcast_list.append(ew_list[i].predict(self.x_test_list_expanding[i]))
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pd.concat([pred_ew, temp], ignore_index=True)
                mlflow.sklearn.log_model(ew_list[i], "rf_model-mindow-{i}")
                mlflow.log_params(argmin)

                # Log feature importance
                dbutils.fs.mkdirs(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/Experiments_info/{_random_string_}/importance_VEST_rf')
                importance = (pd.DataFrame(list(zip(self.x_train_list_expanding[i].columns, ew_list[i].feature_importances_)), columns=["Feature", "Importance"])
                              .sort_values("Importance", ascending=False))
                importance_path = f"{experiment_path_}Experiments_info/{_random_string_}/importance_VEST_rf/_window-{i}.csv"
                importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "importance_VEST_rf_window-{i}-{_random_string_}.csv")

            # Log a tag
            mlflow.set_tag("iternal_id", _random_string_)
            mlflow.set_tag("dataset", self.SerieNumber)
     
            forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_VEST_rf.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_VEST_rf-{_random_string_}.csv")

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

            plt = plot_forcast_vs_obs(pred_ew,f"Random Forest VEST")
            mlflow.log_figure(plt, f"forcast_VEST_rf-{_random_string_}.png")

        elif name_model == "ARIMA":
          with mlflow.start_run(run_name=f"ARIMA_predict") as run:
            for i in range(int(len(self.y_train_list_expanding))):
              (x_train,y_train,x_test,y_test) = (self.x_train_list_expanding[i],self.y_train_list_expanding[i],self.x_test_list_expanding[i],self.y_test_list_expanding[i])

              if i % 80 == 0:
                ########################################################### Hyperparameter tune start ###########################################################
                # Define search space
                search_space = {"p": hp.quniform("p", 0, 7, 1),
                                "d": hp.quniform("d", 0, 7, 1),
                                "q": hp.quniform("q", 0, 7, 1)}
                
                model = ARIMA(y_train.values, order=(1, 1, 1),enforce_stationarity=False)


                def objective_arima(params):
                  try:
                    p, d, q = int(params["p"]), int(params["d"]), int(params["q"])

                    train_size = int(len(y_train.values) * 0.8)
                    train, val = y_train.values[:train_size], y_train.values[train_size:]

                    model = ARIMA(train, order=(p, d, q),enforce_stationarity=False)
                    result = model.fit()
                    pred = result.forecast(steps=len(val))
                    score = calculate_smape(pred, val)
                    # Hyperopt minimizes score, here we minimize smape. 
                    return score
                  except:
                      # In case of any issues with model fitting
                      return np.inf

                # Set parallelism (should be order of magnitude smaller than max_evals)
                spark_trials = SparkTrials(parallelism=25)

                with mlflow.start_run(run_name=f"ARIMA_predict_window-{i}",nested=True):
                    argmin = fmin(fn=objective_arima,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=50,
                                  trials=spark_trials)
                ########################################################### Hyperparameter tune end ###########################################################
                trials_results = sorted(spark_trials.trials, key=lambda x: x['result']['loss'])

                for j in range(len(trials_results)):
                  trials_results[j] = trials_results[j]['misc']['vals']
                  trials_results[j] = {key: [int(value[0])] for key, value in trials_results[j].items()}
                  
                argmin = {key: int(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"ARIMA_predict_window-{i}-chosen",nested=True):
                p = argmin['p']
                d = argmin['d']
                q = argmin['q']
                params_list.append(argmin)

                while_i = 0
                # while this logic fails to fit or it predicts a nan, try it again
                while True:
                    try:
                        ew_list.append(ARIMA(y_train.values,order=(p,d,q),enforce_stationarity=False))
                        # Your logic goes here
                        ew_list_fited = ew_list[i].fit()
                        # Check if x is NaN
                        assert not math.isnan(ew_list_fited.forecast(steps=1)), "forecast-window {i} should not be NaN"

                        # If x is NaN, the program will fail here with the specified error message
                        print(f"forecast-window {i}: {ew_list_fited.forecast(steps=1)}")
                        break  # Break out of the loop if the input is successfully converted to an integer
                    except:  # Catch the specific error that occurs when the input can't be converted to an integer

                        print(trials_results)
                        while_i = while_i +1
                        print(f"Didnt work with p:{p}, d:{d}, q:{q}")              
                        p = trials_results[while_i]['p'][0]
                        d = trials_results[while_i]['d'][0]
                        q = trials_results[while_i]['q'][0]
                        
                        print(f"except:   p:{p}, d:{d}, q:{q}")
                        params_list = params_list[:-1]
                        ew_list = ew_list[:-1]                  

                        params_list.append(trials_results[1])

                forcast_list.append(pd.DataFrame({'forcast': ew_list_fited.forecast(steps=1)}))
                forcast_list[-1].index += (i+40)
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pd.concat([pred_ew, temp], ignore_index=True)  
                mlflow.sklearn.log_model(ew_list[i], "ARIMA_model-mindow-{i}")
                mlflow.log_params({'p':p,'d':d,'q':q})

            # Log a tag
            mlflow.set_tag("iternal_id", _random_string_)
            mlflow.set_tag("dataset", self.SerieNumber)

            forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_ARIMA.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_ARIMA-{_random_string_}.csv")
            
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

            plt = plot_forcast_vs_obs(pred_ew,f"ARIMA")
            mlflow.log_figure(plt, f"forcast_ARIMA-{_random_string_}.png")


        elif name_model == "ARIMA":
          with mlflow.start_run(run_name=f"ARIMA_predict") as run:
            for i in range(int(len(self.y_train_list_expanding))):
              (x_train,y_train,x_test,y_test) = (self.x_train_list_expanding[i],self.y_train_list_expanding[i],self.x_test_list_expanding[i],self.y_test_list_expanding[i])

              if i % 80 == 0:
                ########################################################### Hyperparameter tune start ###########################################################
                # Define search space
                search_space = {"p": hp.quniform("p", 0, 7, 1),
                                "d": hp.quniform("d", 0, 7, 1),
                                "q": hp.quniform("q", 0, 7, 1)}
                
                model = ARIMA(y_train.values, order=(1, 1, 1),enforce_stationarity=False)


                def objective_arima(params):
                  try:
                    p, d, q = int(params["p"]), int(params["d"]), int(params["q"])

                    train_size = int(len(y_train.values) * 0.8)
                    train, val = y_train.values[:train_size], y_train.values[train_size:]

                    model = ARIMA(train, order=(p, d, q),enforce_stationarity=False)
                    result = model.fit()
                    pred = result.forecast(steps=len(val))
                    score = calculate_smape(pred, val)
                    # Hyperopt minimizes score, here we minimize smape. 
                    return score
                  except:
                      # In case of any issues with model fitting
                      return np.inf

                # Set parallelism (should be order of magnitude smaller than max_evals)
                spark_trials = SparkTrials(parallelism=25)

                with mlflow.start_run(run_name=f"ARIMA_predict_window-{i}",nested=True):
                    argmin = fmin(fn=objective_arima,
                                  space=search_space,
                                  algo=tpe.suggest,
                                  max_evals=50,
                                  trials=spark_trials)
                ########################################################### Hyperparameter tune end ###########################################################
                trials_results = sorted(spark_trials.trials, key=lambda x: x['result']['loss'])

                for j in range(len(trials_results)):
                  trials_results[j] = trials_results[j]['misc']['vals']
                  trials_results[j] = {key: [int(value[0])] for key, value in trials_results[j].items()}
                  
                argmin = {key: int(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"ARIMA_predict_window-{i}-chosen",nested=True):
                p = argmin['p']
                d = argmin['d']
                q = argmin['q']
                params_list.append(argmin)

                while_i = 0
                # while this logic fails to fit or it predicts a nan, try it again
                while True:
                    try:
                        ew_list.append(ARIMA(y_train.values,order=(p,d,q),enforce_stationarity=False))
                        # Your logic goes here
                        ew_list_fited = ew_list[i].fit()
                        # Check if x is NaN
                        assert not math.isnan(ew_list_fited.forecast(steps=1)), "forecast-window {i} should not be NaN"

                        # If x is NaN, the program will fail here with the specified error message
                        print(f"forecast-window {i}: {ew_list_fited.forecast(steps=1)}")
                        break  # Break out of the loop if the input is successfully converted to an integer
                    except:  # Catch the specific error that occurs when the input can't be converted to an integer

                        print(trials_results)
                        while_i = while_i +1
                        print(f"Didnt work with p:{p}, d:{d}, q:{q}")              
                        p = trials_results[while_i]['p'][0]
                        d = trials_results[while_i]['d'][0]
                        q = trials_results[while_i]['q'][0]
                        
                        print(f"except:   p:{p}, d:{d}, q:{q}")
                        params_list = params_list[:-1]
                        ew_list = ew_list[:-1]                  

                        params_list.append(trials_results[1])

                forcast_list.append(pd.DataFrame({'forcast': ew_list_fited.forecast(steps=1)}))
                forcast_list[-1].index += (i+40)
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pd.concat([pred_ew, temp], ignore_index=True)  
                mlflow.sklearn.log_model(ew_list[i], "ARIMA_model-mindow-{i}")
                mlflow.log_params({'p':p,'d':d,'q':q})

            # Log a tag
            mlflow.set_tag("iternal_id", _random_string_)
            mlflow.set_tag("dataset", self.SerieNumber)

            forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_ARIMA.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_ARIMA-{_random_string_}.csv")
            
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

            plt = plot_forcast_vs_obs(pred_ew,f"ARIMA")
            mlflow.log_figure(plt, f"forcast_ARIMA-{_random_string_}.png")


        elif name_model == "SES":
          with mlflow.start_run(run_name=f"SES_predict") as run:
            for i in range(int(len(self.y_train_list_expanding))):
              (x_train,y_train,x_test,y_test) = (self.x_train_list_expanding[i],self.y_train_list_expanding[i],self.x_test_list_expanding[i],self.y_test_list_expanding[i])

              if i % 80 == 0:
                ########################################################### Hyperparameter tune start ###########################################################
                # Define search space
                search_space = {'smoothing_level': hp.uniform('smoothing_level', 0.01, 1.0)}

                model = ExponentialSmoothing(y_train.values)

                def objective_ses(params):
                    smoothing_level = params['smoothing_level']

                    train_size = int(len(y_train.values) * 0.8)
                    train, val = y_train.values[:train_size], y_train.values[train_size:]
                    
                    model = ExponentialSmoothing(train,trend='add')                          
                    result = model.fit(smoothing_level=smoothing_level)
                    pred = result.forecast(steps=len(val))
                    score = calculate_smape(pred, val)
                    # Hyperopt minimizes score, here we minimize smape. 
                    return score


                # Set parallelism (should be order of magnitude smaller than max_evals)
                spark_trials = SparkTrials(parallelism=25)

                with mlflow.start_run(run_name=f"SES_predict_window-{i}",nested=True):
                    argmin = fmin(fn=objective_ses,
                                    space=search_space,
                                    algo=tpe.suggest,
                                    max_evals=25,
                                    trials=spark_trials)
                ########################################################### Hyperparameter tune end ###########################################################    

                ########################################################### Hyperparameter tune end ###########################################################                     
                argmin = {key: int(value) for key, value in argmin.items()}
              with mlflow.start_run(run_name=f"SES_predict_window-{i}-chosen",nested=True):
                params_list.append(argmin)
                ew_list.append(ExponentialSmoothing(y_train.values,trend='add'))
                ew_list_fited = ew_list[i].fit(**argmin)
                forcast_list.append(pd.DataFrame({'forcast': ew_list_fited.forecast(steps=1)}))
                forcast_list[-1].index += (i+40)
                temp = pd.DataFrame(self.y_test_list_expanding[i])
                temp["forcast"] = forcast_list[i]
                pred_ew = pd.concat([pred_ew, temp], ignore_index=True)
                mlflow.sklearn.log_model(ew_list[i], "SES_model-mindow-{i}")
                mlflow.log_params(argmin)

            # Log a tag
            mlflow.set_tag("iternal_id", _random_string_)
            mlflow.set_tag("dataset", self.SerieNumber)

            forcast_path = f"{experiment_path_}Experiments_info/{_random_string_}/forcast_SES.csv"
            pred_ew.to_csv(forcast_path, mode='x', index=False)
            mlflow.log_artifact(forcast_path, "forcast_SES-{_random_string_}.csv")
            
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

            plt = plot_forcast_vs_obs(pred_ew,f"SES")
            mlflow.log_figure(plt, f"forcast_SES-{_random_string_}.png")


        self.list_of_models[-1].append(ew_list)
        self.list_of_models[-1].append(pred_ew)



# COMMAND ----------

serie_solv = timeseries_solver(input_variables,SerieNumber)

# COMMAND ----------

serie_solv.fit(Chosen_Model)
