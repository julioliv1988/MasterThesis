# Databricks notebook source
import pandas as pd
from ast import literal_eval
from collections import Counter

# COMMAND ----------

def count_occurrences(lst):
    # Count the occurrences using Counter
    occurrence_count = Counter(lst)
    
    # Convert the Counter object to a pandas DataFrame
    df = pd.DataFrame(occurrence_count.items(), columns=['Element', 'Count'])
    
    # Sort the DataFrame by count (optional)
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    return df
def convert_to_list(array_str):
    try:
        return literal_eval(array_str)
    except (ValueError, SyntaxError) as e:
        # Handle cases where conversion fails
        print(f"Error converting {array_str}: {e}")
        return None

def transform_string(s):
    # Remove all occurrences of ":"
    transformed = s.replace(":", "")
    # Add "/" at the beginning
    result = "/" + transformed
    return result

# COMMAND ----------

list_ = [sublist[0][67:-1] for sublist in dbutils.fs.ls(f'/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/')]

serie_names_list = [item for item in list_ if item.startswith("SerieNumber")]

csv_list = []

for _series_ in serie_names_list:

    total_list = [sublist[0] for sublist in dbutils.fs.ls(f"/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/{_series_}/Feature_eng")]

    for itens_ in [transform_string(item) for item in total_list if "summary" in item]:
        csv_list.append(itens_)
# csv_list

# COMMAND ----------

def create_window_array(integer_list):
    return [f'window-{i}-chosen' for i in integer_list]

# COMMAND ----------

# Note tha that this logic relies on the back/foard elimination csvs!!!!!

cuts_dic = {}

for _series_ in serie_names_list:
    cuts_dic[_series_] = {}

    partial_list = [item for item in csv_list if f"{_series_}/" in item]

    partial_list_foward = [item for item in partial_list if f"foward" in item][0]

    cuts_dic[_series_]["df"] = pd.read_csv(partial_list_foward)[["type","Points"]]

    cuts_dic[_series_]["df"]['Points'] = cuts_dic[_series_]["df"]['Points'] - 1

    # Initial crescent list
    crescent_list = cuts_dic[_series_]["df"]['Points'].tolist()

    # Create the list of lists using a list comprehension
    list_of_lists = [list(range(start, end + 1)) for start, end in zip([0] + [x + 1 for x in crescent_list[:-1]], crescent_list)]

    window_points_list = []
    for element in list_of_lists:
        window_points_list.append(create_window_array(element))

    cuts_dic[_series_]["df"]["list_of_points"] = window_points_list

# COMMAND ----------

cuts_dic["SerieNumber_0"]["df"]

# COMMAND ----------

cuts_dic["SerieNumber_1-trend-noise"]["df"]

# COMMAND ----------

ba_fo_elim_dic.keys()

# COMMAND ----------

# MAGIC %md # Imports and definitions

# COMMAND ----------

import pandas as pd
import mlflow

# COMMAND ----------

def aux_filter_string_fn(run_name_like,iternal_id):
    return f""" attributes.run_name like "{run_name_like}" AND tags.iternal_id="{iternal_id}" """

def aux_filter_string_fn_alt(iternal_id):
    return f""" tags.iternal_id="{iternal_id}" """

def filter_string_fn(run_name_like,rootRunId):
    return f""" tags.mlflow.runName like "{run_name_like}" AND tags.mlflow.rootRunId = "{rootRunId}" """

# COMMAND ----------

class mlflow_retrieve_runs:
      def __init__(self):        

        self.dict_experiments = {}
        self.params = {}
        self.general_summary = {}
        self.params["LR_predict_FULL"] = ['params.fit_intercept', 'params.positive']
        self.params["RF_predict_FULL"] = ["params.n_estimators", "params.max_depth", "params.min_samples_leaf", "params.min_samples_split"]
        self.params["KNN_predict_FULL"] = ["params.n_neighbors", "params.leaf_size", "params.p"]
        self.params["ARIMA_predict"] = ["params.p", "params.d", "params.q"]

        self.lean_set_cols = ["tags.mlflow.runName", "tags.iternal_id", "tags.dataset"]
        
      def register_runs(self,run_kind,internal_id,given_name=None):
          
        runs_with_complex_filter = mlflow.search_runs(
            filter_string=aux_filter_string_fn(f"{run_kind}%",internal_id),
            search_all_experiments=True,
            order_by=["start_time"],)
        mlflow_id = runs_with_complex_filter["tags.mlflow.rootRunId"].values[0]

        runs_with_complex_filter = mlflow.search_runs(
        filter_string=filter_string_fn(f"{run_kind}%",mlflow_id),
        search_all_experiments=True,
        order_by=["start_time"], )

        subdic = {}

        subdic[f"df_{run_kind}"] = runs_with_complex_filter
        
        if internal_id in self.dict_experiments:
            self.dict_experiments[internal_id].update(subdic)
        else:
            self.dict_experiments[internal_id] = subdic
            dic_name = {}
            dic_name["given_name"] = given_name
            self.dict_experiments[internal_id].update(dic_name)

        #new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for item in range(len(cuts_dic[self.dict_experiments[internal_id]["given_name"]]["df"]["type"].to_list())):
            type_temp = cuts_dic[self.dict_experiments[internal_id]["given_name"]]["df"]["type"][item]
            df_temp = self.dict_experiments[internal_id][f"df_{run_kind}"]

            substrings = cuts_dic[self.dict_experiments[internal_id]["given_name"]]["df"]["list_of_points"].to_list()[item]

            # substrings
            # Create a filter using str.contains with regex and join with |
            pattern = '|'.join(substrings)

            # Filter the DataFrame
            self.dict_experiments[internal_id][f"df_{run_kind}-{type_temp}"] = df_temp[df_temp['tags.mlflow.runName'].str.contains(pattern, case=False, na=False)]


      def calc_general_summary(self):
        for models_keys in self.params.keys():

            __model_ = models_keys[:-13]
            if __model_ == '':
                __model_ = "ARIMA"
            self.general_summary[__model_] = {}

            self.general_summary[__model_]["all"] = {}
            for params_keys in self.params[models_keys]:
                self.general_summary[__model_]["all"][params_keys] = pd.DataFrame()
                
                for _df_ in [self.dict_experiments[key][f"params_summary_{models_keys}"][params_keys] for key in self.dict_experiments.keys() ]:
                    self.general_summary[__model_]["all"][params_keys] = pd.concat([self.general_summary[__model_]["all"][params_keys], _df_], axis=0, ignore_index=True)
                
                self.general_summary[__model_]["all"][params_keys] = self.general_summary[__model_]["all"][params_keys].sum(axis=0).to_frame().T

                # Sort the columns in ascending order
                try:
                    # Assuming 'df' is your DataFrame
                    self.general_summary[__model_]["all"][params_keys].columns = self.general_summary[__model_]["all"][params_keys].columns.astype(int)  # Convert column names to integers
                    self.general_summary[__model_]["all"][params_keys] = self.general_summary[__model_]["all"][params_keys].reindex(sorted(self.general_summary[__model_]["all"][params_keys].columns), axis=1)  # Reorder columns numerically
                except: pass

                # sorted_columns = sorted(self.general_summary[__model_]["all"][params_keys].columns)
                # # Reorder the DataFrame based on the sorted columns
                # self.general_summary[__model_]["all"][params_keys] = self.general_summary[__model_]["all"][params_keys][sorted_columns]

            self.general_summary[__model_]["trend-season"] = {}
            for params_keys in self.params[models_keys]:
                self.general_summary[__model_]["trend-season"][params_keys] = pd.DataFrame()
                
                for _df_ in [self.dict_experiments[key][f"params_summary_{models_keys}"][params_keys] for key in self.dict_experiments.keys() if "trend-season" in self.dict_experiments[key]["given_name"] ]:
                    self.general_summary[__model_]["trend-season"][params_keys] = pd.concat([self.general_summary[__model_]["trend-season"][params_keys], _df_], axis=0, ignore_index=True)
                
                self.general_summary[__model_]["trend-season"][params_keys] = self.general_summary[__model_]["trend-season"][params_keys].sum(axis=0).to_frame().T
                # Sort the columns in ascending order
                try:
                    # Assuming 'df' is your DataFrame
                    self.general_summary[__model_]["trend-season"][params_keys].columns = self.general_summary[__model_]["trend-season"][params_keys].columns.astype(int)  # Convert column names to integers
                    self.general_summary[__model_]["trend-season"][params_keys] = self.general_summary[__model_]["trend-season"][params_keys].reindex(sorted(self.general_summary[__model_]["trend-season"][params_keys].columns), axis=1)  # Reorder columns numerically
                except: pass

                # # Sort the columns in ascending order
                # sorted_columns = sorted(self.general_summary[__model_]["trend-season"][params_keys].columns)
                # # Reorder the DataFrame based on the sorted columns
                # self.general_summary[__model_]["trend-season"][params_keys] = self.general_summary[__model_]["trend-season"][params_keys][sorted_columns]

            self.general_summary[__model_]["season-noise"] = {}
            for params_keys in self.params[models_keys]:
                self.general_summary[__model_]["season-noise"][params_keys] = pd.DataFrame()
                
                for _df_ in [self.dict_experiments[key][f"params_summary_{models_keys}"][params_keys] for key in self.dict_experiments.keys() if "season-noise" in self.dict_experiments[key]["given_name"] ]:
                    self.general_summary[__model_]["season-noise"][params_keys] = pd.concat([self.general_summary[__model_]["season-noise"][params_keys], _df_], axis=0, ignore_index=True)
                
                self.general_summary[__model_]["season-noise"][params_keys] = self.general_summary[__model_]["season-noise"][params_keys].sum(axis=0).to_frame().T
                
                # Sort the columns in ascending order
                try:
                    # Assuming 'df' is your DataFrame
                    self.general_summary[__model_]["season-noise"][params_keys].columns = self.general_summary[__model_]["season-noise"][params_keys].columns.astype(int)  # Convert column names to integers
                    self.general_summary[__model_]["season-noise"][params_keys] = self.general_summary[__model_]["season-noise"][params_keys].reindex(sorted(self.general_summary[__model_]["season-noise"][params_keys].columns), axis=1)  # Reorder columns numerically
                except: pass

                # # Sort the columns in ascending order
                # sorted_columns = sorted(self.general_summary[__model_]["season-noise"][params_keys].columns)
                # # Reorder the DataFrame based on the sorted columns
                # self.general_summary[__model_]["season-noise"][params_keys] = self.general_summary[__model_]["season-noise"][params_keys][sorted_columns]

            self.general_summary[__model_]["trend-noise"] = {}
            for params_keys in self.params[models_keys]:
                self.general_summary[__model_]["trend-noise"][params_keys] = pd.DataFrame()
                
                for _df_ in [self.dict_experiments[key][f"params_summary_{models_keys}"][params_keys] for key in self.dict_experiments.keys() if "trend-noise" in self.dict_experiments[key]["given_name"] ]:
                    self.general_summary[__model_]["trend-noise"][params_keys] = pd.concat([self.general_summary[__model_]["trend-noise"][params_keys], _df_], axis=0, ignore_index=True)
                
                self.general_summary[__model_]["trend-noise"][params_keys] = self.general_summary[__model_]["trend-noise"][params_keys].sum(axis=0).to_frame().T

                # Sort the columns in ascending order
                try:
                    # Assuming 'df' is your DataFrame
                    self.general_summary[__model_]["trend-noise"][params_keys].columns = self.general_summary[__model_]["trend-noise"][params_keys].columns.astype(int)  # Convert column names to integers
                    self.general_summary[__model_]["trend-noise"][params_keys] = self.general_summary[__model_]["trend-noise"][params_keys].reindex(sorted(self.general_summary[__model_]["trend-noise"][params_keys].columns), axis=1)  # Reorder columns numerically
                except: pass

                # # Sort the columns in ascending order
                # sorted_columns = sorted(self.general_summary[__model_]["trend-noise"][params_keys].columns)
                # # Reorder the DataFrame based on the sorted columns
                # self.general_summary[__model_]["trend-noise"][params_keys] = self.general_summary[__model_]["trend-noise"][params_keys][sorted_columns]
    
            self.general_summary[__model_]["trend-season-noise"] = {}
            for params_keys in self.params[models_keys]:
                self.general_summary[__model_]["trend-season-noise"][params_keys] = pd.DataFrame()
                
                for _df_ in [self.dict_experiments[key][f"params_summary_{models_keys}"][params_keys] for key in self.dict_experiments.keys() if ("trend-noise" not in self.dict_experiments[key]["given_name"]) and ("trend-season" not in self.dict_experiments[key]["given_name"]) and ("season-noise" not in self.dict_experiments[key]["given_name"]) ]:
                    self.general_summary[__model_]["trend-season-noise"][params_keys] = pd.concat([self.general_summary[__model_]["trend-season-noise"][params_keys], _df_], axis=0, ignore_index=True)
                
                self.general_summary[__model_]["trend-season-noise"][params_keys] = self.general_summary[__model_]["trend-season-noise"][params_keys].sum(axis=0).to_frame().T

                # Sort the columns in ascending order
                try:
                    # Assuming 'df' is your DataFrame
                    self.general_summary[__model_]["trend-season-noise"][params_keys].columns = self.general_summary[__model_]["trend-season-noise"][params_keys].columns.astype(int)  # Convert column names to integers
                    self.general_summary[__model_]["trend-season-noise"][params_keys] = self.general_summary[__model_]["trend-season-noise"][params_keys].reindex(sorted(self.general_summary[__model_]["trend-season-noise"][params_keys].columns), axis=1)  # Reorder columns numerically
                except: pass

                # # Sort the columns in ascending order
                # sorted_columns = sorted(self.general_summary[__model_]["trend-season-noise"][params_keys].columns)
                # # Reorder the DataFrame based on the sorted columns
                # self.general_summary[__model_]["trend-season-noise"][params_keys] = self.general_summary[__model_]["trend-season-noise"][params_keys][sorted_columns]


      def display_lean_df(self,run_kind,internal_id,_n_=False):
        
        cols_ = self.lean_set_cols+self.params[run_kind]
        if _n_:
            display(self.dict_experiments[internal_id][f"df_{run_kind}"][cols_].head(_n_))
        else:
            display(self.dict_experiments[internal_id][f"df_{run_kind}"][cols_])

      def params_summary(self,run_kind,internal_id,decorator=""):
        """
        decorator -> ["-No season", "-1 season", "-2 season"]
        given run_kind (e.g.: LR_predict_FULL) and internal_id (e.g.: kDzXmXR)
        it gets all the parameters and count the occurrence of each value
        returns self.dict_experiments[internal_id][f"params_summary_{run_kind}"][_params_]
        """      

        params_list = self.params[run_kind] 
        self.dict_experiments[internal_id][f"params_summary_{run_kind}{decorator}"] = {}
        for _params_ in params_list:
            # print(_params_)
            value_counts = self.dict_experiments[internal_id][f"df_{run_kind}{decorator}"][_params_].value_counts()
            counts_df = pd.DataFrame({'Value': value_counts.index,'Count': value_counts.values})
            # display(counts_df)
        
            # Transpose the DataFrame
            df_transposed = counts_df.T
            # df_transposed.iloc[0] = df_transposed.iloc[0].str.replace('params.', '', regex=False)
            # Set the first row as the header
            df_transposed.columns = df_transposed.iloc[0]
            # Drop the first row as it is now the header
            df_transposed = df_transposed[1:]
            # Optionally, reset the index if needed
            df_transposed.reset_index(drop=True, inplace=True)
            
            self.dict_experiments[internal_id][f"params_summary_{run_kind}{decorator}"][_params_] = df_transposed
        
      def compare_generate(self,internal_id):

        internal_id= internal_id
        runs_with_complex_filter = mlflow.search_runs(
            filter_string=aux_filter_string_fn_alt(internal_id),
            search_all_experiments=True,
            order_by=["start_time"],)

        runs_with_complex_filter = runs_with_complex_filter[["run_id","tags.mlflow.runName","start_time","end_time","metrics.mae","metrics.mse","metrics.r2","metrics.rmse","metrics.smape","tags.iternal_id","tags.dataset"]]

        new = runs_with_complex_filter.astype(str)

        # Transpose the DataFrame
        df_transposed = new.transpose()

        # Set the new index to the original column names
        df_transposed.columns = new.index

        df_transposed.columns = df_transposed.iloc[1]

        df_transposed = df_transposed.rename_axis(' ').reset_index()
        df_transposed = df_transposed.drop(index=1)

        # display(df_transposed)

        subdic = {}

        subdic["compare"] = df_transposed

        if internal_id in self.dict_experiments:
            self.dict_experiments[internal_id].update(subdic)
        else:
            self.dict_experiments[internal_id] = subdic

      def complete_register(self,internal_id,given_name):    
        """
        Does:
        * "self.register_runs" -> (LR_predict_FULL, LR_predict_FULL,KNN_predict_FULL, ARIMA_predict)
        
        * "self.params_summary" (LR_predict_FULL, LR_predict_FULL,KNN_predict_FULL, ARIMA_predict) and variations ["-No season", "-1 season", "-2 season"]
        * "self.compare_generate"

        """
        # register_runs
        self.register_runs("LR_predict_FULL",internal_id,given_name)
        self.register_runs("RF_predict_FULL",internal_id)
        self.register_runs("KNN_predict_FULL",internal_id)
        self.register_runs("ARIMA_predict",internal_id)

        # params_summary
        self.params_summary("LR_predict_FULL",internal_id)
        for _type_ in cuts_dic[_mlflow_retrieve_runs_.dict_experiments[internal_id]["given_name"]]["df"]["type"].to_list():
            _mlflow_retrieve_runs_.params_summary("LR_predict_FULL",internal_id_,f"-{_type_}")
        self.params_summary("RF_predict_FULL",internal_id)
        for _type_ in cuts_dic[_mlflow_retrieve_runs_.dict_experiments[internal_id]["given_name"]]["df"]["type"].to_list():
            _mlflow_retrieve_runs_.params_summary("RF_predict_FULL",internal_id_,f"-{_type_}")
        self.params_summary("KNN_predict_FULL",internal_id)
        for _type_ in cuts_dic[_mlflow_retrieve_runs_.dict_experiments[internal_id]["given_name"]]["df"]["type"].to_list():
            _mlflow_retrieve_runs_.params_summary("KNN_predict_FULL",internal_id_,f"-{_type_}")
        self.params_summary("ARIMA_predict",internal_id)
        for _type_ in cuts_dic[_mlflow_retrieve_runs_.dict_experiments[internal_id]["given_name"]]["df"]["type"].to_list():
            _mlflow_retrieve_runs_.params_summary("ARIMA_predict",internal_id_,f"-{_type_}")

        # compare_generate
        self.compare_generate(internal_id)


      def retrieve_outcome_per_serie(self,internal_id_):
            print("Linear Regression")
            _mlflow_retrieve_runs_.display_lean_df("LR_predict_FULL",internal_id_,20)
            _mlflow_retrieve_runs_.params_summary("LR_predict_FULL",internal_id_)
            for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_LR_predict_FULL"]:
                display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_LR_predict_FULL"][_dfs_])
            print("Random Forest")
            _mlflow_retrieve_runs_.display_lean_df("RF_predict_FULL",internal_id_,20)
            _mlflow_retrieve_runs_.params_summary("RF_predict_FULL",internal_id_)
            for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_RF_predict_FULL"]:
                display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_RF_predict_FULL"][_dfs_])
            print("KNN")
            _mlflow_retrieve_runs_.display_lean_df("KNN_predict_FULL",internal_id_,20)
            _mlflow_retrieve_runs_.params_summary("KNN_predict_FULL",internal_id_)
            for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_KNN_predict_FULL"]:
                display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_KNN_predict_FULL"][_dfs_])
            print("ARIMA")
            _mlflow_retrieve_runs_.display_lean_df("ARIMA_predict",internal_id_,20)
            _mlflow_retrieve_runs_.params_summary("ARIMA_predict",internal_id_)
            for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_ARIMA_predict"]:
                display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_ARIMA_predict"][_dfs_])

            display(_mlflow_retrieve_runs_.dict_experiments[internal_id_]["compare"])

# COMMAND ----------

_mlflow_retrieve_runs_.dict_experiments.keys()

# COMMAND ----------

_mlflow_retrieve_runs_.dict_experiments["kDzXmXR"].keys()

# COMMAND ----------

_mlflow_retrieve_runs_.dict_experiments["1aMpsha"].keys()

# COMMAND ----------

_mlflow_retrieve_runs_.dict_experiments["kDzXmXR"]["given_name"]

# COMMAND ----------

_mlflow_retrieve_runs_ = mlflow_retrieve_runs()
############################################# trend-season-noise #############################################
# Serie 0
internal_id_ = "kDzXmXR"
given_name_ = "SerieNumber_0"

_mlflow_retrieve_runs_.register_runs("LR_predict_FULL",internal_id_,given_name_)
_mlflow_retrieve_runs_.register_runs("RF_predict_FULL",internal_id_)
_mlflow_retrieve_runs_.register_runs("KNN_predict_FULL",internal_id_)
_mlflow_retrieve_runs_.register_runs("ARIMA_predict",internal_id_)
_mlflow_retrieve_runs_.params_summary("LR_predict_FULL",internal_id_)
# _mlflow_retrieve_runs_.params_summary("LR_predict_FULL",internal_id_,"-No season")
for _type_ in cuts_dic[_mlflow_retrieve_runs_.dict_experiments[internal_id_]["given_name"]]["df"]["type"].to_list():
    # print(f"-{_type_}")
    _mlflow_retrieve_runs_.params_summary("LR_predict_FULL",internal_id_,f"-{_type_}")

_mlflow_retrieve_runs_.params_summary("RF_predict_FULL",internal_id_)
_mlflow_retrieve_runs_.params_summary("KNN_predict_FULL",internal_id_)
_mlflow_retrieve_runs_.params_summary("ARIMA_predict",internal_id_)
_mlflow_retrieve_runs_.compare_generate(internal_id_)



# COMMAND ----------

_mlflow_retrieve_runs_ = mlflow_retrieve_runs()
############################################# trend-season-noise #############################################
# Serie 0
internal_id_ = "kDzXmXR"
given_name_ = "SerieNumber_0"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 1
internal_id_ = "u9tveCp"
given_name_ = "SerieNumber_1"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 2
internal_id_ = "X1LUwKn"
given_name_ = "SerieNumber_2"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 3
internal_id_ = "CLGo9rk"
given_name_ = "SerieNumber_3"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 4
internal_id_ = "6tRf7JU"
given_name_ = "SerieNumber_4"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
############################################# trend-season #############################################
# Serie 0
internal_id_ = "xRmAioK"
given_name_ = "SerieNumber_0-trend-season"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 1
internal_id_ = "6KC6QnM"
given_name_ = "SerieNumber_1-trend-season"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 2
internal_id_ = "Ef93YW6"
given_name_ = "SerieNumber_2-trend-season"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 3
internal_id_ = "h94TUJC"
given_name_ = "SerieNumber_3-trend-season"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 4
internal_id_ = "1TqrE9V"
given_name_ = "SerieNumber_4-trend-season"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
############################################# season-noise #############################################
# Serie 0
internal_id_ = "cGJPwDY"
given_name_ = "SerieNumber_0-season-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 1
internal_id_ = "4P9oiEy"
given_name_ = "SerieNumber_1-season-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 2
internal_id_ = "2I2jYTD"
given_name_ = "SerieNumber_2-season-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 3
internal_id_ = "YJViDWK"
given_name_ = "SerieNumber_3-season-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 4
internal_id_ = "Wki1N7U"
given_name_ = "SerieNumber_4-season-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
############################################# trend-noise #############################################
# Serie 0
internal_id_ = "1aMpsha"
given_name_ = "SerieNumber_0-trend-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 1
internal_id_ = "K6muDqb"
given_name_ = "SerieNumber_1-trend-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 2
internal_id_ = "ig7jLMN"
given_name_ = "SerieNumber_2-trend-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 3
internal_id_ = "uqUOXyM"
given_name_ = "SerieNumber_3-trend-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)
# Serie 4
internal_id_ = "cXlFrik"
given_name_ = "SerieNumber_4-trend-noise"
_mlflow_retrieve_runs_.complete_register(internal_id_,given_name_)

# COMMAND ----------

# MAGIC %md # Load

# COMMAND ----------

# MAGIC %md ##trend-season-noise

# COMMAND ----------

# MAGIC %md ###Serie 0 

# COMMAND ----------

# MAGIC %md
# MAGIC kDzXmXR

# COMMAND ----------

# Serie 0 - trend-season-noise
internal_id_ = "kDzXmXR"

# print("Linear Regression")
# _mlflow_retrieve_runs_.display_lean_df("LR_predict_FULL",internal_id_,20)
# _mlflow_retrieve_runs_.params_summary("LR_predict_FULL",internal_id_)
# for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_LR_predict_FULL"]:
#     display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_LR_predict_FULL"][_dfs_])
# print("Random Forest")
# _mlflow_retrieve_runs_.display_lean_df("RF_predict_FULL",internal_id_,20)
# _mlflow_retrieve_runs_.params_summary("RF_predict_FULL",internal_id_)
# for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_RF_predict_FULL"]:
#     display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_RF_predict_FULL"][_dfs_])
# print("KNN")
# _mlflow_retrieve_runs_.display_lean_df("KNN_predict_FULL",internal_id_,20)
# _mlflow_retrieve_runs_.params_summary("KNN_predict_FULL",internal_id_)
# for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_KNN_predict_FULL"]:
#     display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_KNN_predict_FULL"][_dfs_])
# print("ARIMA")
# _mlflow_retrieve_runs_.display_lean_df("ARIMA_predict",internal_id_,20)
# _mlflow_retrieve_runs_.params_summary("ARIMA_predict",internal_id_)
# for _dfs_ in _mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_ARIMA_predict"]:
#     display(_mlflow_retrieve_runs_.dict_experiments[internal_id_][f"params_summary_ARIMA_predict"][_dfs_])

# display(_mlflow_retrieve_runs_.dict_experiments[internal_id_]["compare"])

_mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ### Serie 1

# COMMAND ----------

# MAGIC %md
# MAGIC u9tveCp

# COMMAND ----------

# Serie 1 - trend-season-noise
internal_id_ = "u9tveCp"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ### Serie 2

# COMMAND ----------

# MAGIC %md
# MAGIC X1LUwKn

# COMMAND ----------

# Serie 2 - trend-season-noise
internal_id_ = "X1LUwKn"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ### Serie 3

# COMMAND ----------

# MAGIC %md
# MAGIC CLGo9rk

# COMMAND ----------

# Serie 3 - trend-season-noise
internal_id_ = "CLGo9rk"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ### Serie 4

# COMMAND ----------

# MAGIC %md
# MAGIC 6tRf7JU

# COMMAND ----------

# Serie 4 - trend-season-noise
internal_id_ = "6tRf7JU"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ##trend-season

# COMMAND ----------

# MAGIC %md ###Serie 0

# COMMAND ----------

# MAGIC %md
# MAGIC xRmAioK
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

# Serie 0 - trend-season
internal_id_ = "xRmAioK"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ###Serie 1

# COMMAND ----------

# MAGIC %md
# MAGIC 6KC6QnM
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

internal_id_ = "6KC6QnM"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 2

# COMMAND ----------

# MAGIC %md
# MAGIC Ef93YW6
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

internal_id_ = "Ef93YW6"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 3

# COMMAND ----------

# MAGIC %md
# MAGIC h94TUJC
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

internal_id_ = "h94TUJC"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 4

# COMMAND ----------

# MAGIC %md
# MAGIC MmpnJ41 (antigo)
# MAGIC
# MAGIC 1TqrE9V (novo)

# COMMAND ----------

#https://adb-6905234774964628.8.azuredatabricks.net/jobs/309844446891484/runs/561774085326083?o=6905234774964628

# COMMAND ----------

internal_id_ = "1TqrE9V"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ## season-noise

# COMMAND ----------

# MAGIC %md ###Serie 0

# COMMAND ----------

# MAGIC %md
# MAGIC cGJPwDY
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

internal_id_ = "cGJPwDY"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 1

# COMMAND ----------

# MAGIC %md
# MAGIC 4P9oiEy

# COMMAND ----------

internal_id_ = "4P9oiEy"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 2

# COMMAND ----------

# MAGIC %md
# MAGIC ckGrURE (antigo)
# MAGIC
# MAGIC 2I2jYTD (novo)

# COMMAND ----------

internal_id_ = "2I2jYTD"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 3

# COMMAND ----------

# MAGIC %md
# MAGIC KLUPRil (antigo)
# MAGIC
# MAGIC YJViDWK (novo)

# COMMAND ----------

internal_id_ = "YJViDWK"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md
# MAGIC https://adb-6905234774964628.8.azuredatabricks.net/jobs/309844446891484/runs/653693321998623?o=6905234774964628

# COMMAND ----------

# MAGIC %md ###Serie 4

# COMMAND ----------

# MAGIC %md
# MAGIC rxulmxt (antigo)
# MAGIC
# MAGIC Wki1N7U (novo)

# COMMAND ----------

internal_id_ = "Wki1N7U"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ## trend-noise

# COMMAND ----------

# MAGIC %md ###Serie 0

# COMMAND ----------

# MAGIC %md
# MAGIC 1aMpsha

# COMMAND ----------

internal_id_ = "1aMpsha"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 1

# COMMAND ----------

# MAGIC %md
# MAGIC K6muDqb

# COMMAND ----------

internal_id_ = "K6muDqb"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 2

# COMMAND ----------

# MAGIC %md
# MAGIC ig7jLMN

# COMMAND ----------

internal_id_ = "ig7jLMN"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 3

# COMMAND ----------

# MAGIC %md
# MAGIC bzJCZpi (antigo)
# MAGIC
# MAGIC uqUOXyM (novo)

# COMMAND ----------

internal_id_ = "uqUOXyM"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md ###Serie 4

# COMMAND ----------

# MAGIC %md
# MAGIC cXlFrik
# MAGIC
# MAGIC (ARIMA issue)

# COMMAND ----------

internal_id_ = "cXlFrik"

# _mlflow_retrieve_runs_.retrieve_outcome_per_serie(internal_id_)

# COMMAND ----------

# MAGIC %md # Hypertune Summary

# COMMAND ----------

_mlflow_retrieve_runs_.params.keys()

# COMMAND ----------

_mlflow_retrieve_runs_.dict_experiments["xRmAioK"]["params_summary_LR_predict_FULL"]

# COMMAND ----------

_mlflow_retrieve_runs_.general_summary["LR"]["all"]['params.fit_intercept']

# COMMAND ----------

_mlflow_retrieve_runs_.general_summary["LR"]["trend-season"]['params.fit_intercept']

# COMMAND ----------

_mlflow_retrieve_runs_.general_summary["LR"]["season-noise"]['params.fit_intercept']

# COMMAND ----------

_mlflow_retrieve_runs_.general_summary["LR"]["trend-noise"]['params.fit_intercept']

# COMMAND ----------

df =len(_mlflow_retrieve_runs_.general_summary["LR"]["trend-season-noise"]['params.fit_intercept'].columns)

df

# COMMAND ----------

df.columns = df.columns.astype(int)  # Convert column names to integers
df = df.reindex(sorted(df.columns), axis=1)  # Reorder columns numerically
df

# COMMAND ----------

_mlflow_retrieve_runs_.calc_general_summary()

# COMMAND ----------

import matplotlib.pyplot as plt

for _models_keys_ in _mlflow_retrieve_runs_.general_summary:
    print(f"######################################################################{_models_keys_}######################################################################")
    for _models_keys_2 in _mlflow_retrieve_runs_.general_summary[_models_keys_]:
        print(f"######################################################################{_models_keys_2}######################################################################")

        for params_keys in _mlflow_retrieve_runs_.general_summary[_models_keys_][_models_keys_2]:
            print(f"####{params_keys}####")
            display(_mlflow_retrieve_runs_.general_summary[_models_keys_][_models_keys_2][params_keys])

            if len(_mlflow_retrieve_runs_.general_summary[_models_keys_][_models_keys_2][params_keys].columns) > 25:
                fs_ = 12
                x1 = 12
                x2 = 8
            else: 
                fs_ = 12
                x1 = 12
                x2 = 8

            df = _mlflow_retrieve_runs_.general_summary[_models_keys_][_models_keys_2][params_keys]
            # Since the first row represents the categories, transpose the DataFrame
            df = df.T  # Transpose to switch rows and columns
            df.columns = ['Value']  # Rename the new column
            plt.figure(figsize=(x1, x2))  # Adjust figure size
            ax = df.plot(kind='bar', legend=False,figsize=(x1, x2))

            # Adding labels and title
            plt.xlabel(f"{params_keys}")
            plt.ylabel('Count')
            plt.title(f'{_models_keys_}--{_models_keys_2}')
            


            # Display the chart
            # Rotate the x-axis labels and align them
            plt.xticks(rotation=45, ha='right', fontsize=fs_)

            # Adjust layout for better fit
            plt.tight_layout()

            plt.show()

# COMMAND ----------

for _models_keys_ in _mlflow_retrieve_runs_.general_summary:
    print(f"######################################################################{_models_keys_}######################################################################")
    for _models_keys_2 in _mlflow_retrieve_runs_.general_summary[_models_keys_]:
        print(f"######################################################################{_models_keys_2}######################################################################")

        for params_keys in _mlflow_retrieve_runs_.general_summary[_models_keys_][_models_keys_2]:
            print(f"####{params_keys}####")
            display(_mlflow_retrieve_runs_.general_summary[_models_keys_][_models_keys_2][params_keys])

# COMMAND ----------

# MAGIC %md #Best models

# COMMAND ----------

best_model_name_list = []
worst_model_name_list = []
given_name_list = []
for key_ in _mlflow_retrieve_runs_.dict_experiments.keys():
    best_model_name_list.append(_mlflow_retrieve_runs_.dict_experiments[key_]['compare'].iloc[[7]].drop(columns=_mlflow_retrieve_runs_.dict_experiments[key_]['compare'].columns[0]).astype(float).idxmin(axis=1).values[0])
    worst_model_name_list.append(_mlflow_retrieve_runs_.dict_experiments[key_]['compare'].iloc[[7]].drop(columns=_mlflow_retrieve_runs_.dict_experiments[key_]['compare'].columns[0]).astype(float).idxmax(axis=1).values[0])
    given_name_list.append(_mlflow_retrieve_runs_.dict_experiments[key_]['given_name'])


# List of lists
data = [
best_model_name_list,
worst_model_name_list
]

# Creating DataFrame
df = pd.DataFrame(data, columns=given_name_list)

print("DataFrame from list of lists:")
display(df)


# COMMAND ----------

# tests
# Downloading artifacts
import mlflow

# Download the artifact
artifact_path = mlflow.artifacts.download_artifacts(artifact_uri="dbfs:/databricks/mlflow-tracking/3062746970085784/06c5949c43594a7faf4b5b40273dfda1/artifacts/forcast_ARIMA-xRmAioK.png")
from PIL import Image
image = Image.open(artifact_path)
# image.show()


logged_model = 'runs:/307c58dbe37b48bbaa0592e6c660855b/ARIMA_model-mindow-{i}'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri=logged_model)

# Predict on a Pandas DataFrame.
# import pandas as pd
# loaded_model.predict(test)

# Download the artifact

artifact_path = mlflow.artifacts.download_artifacts(artifact_uri="dbfs:/databricks/mlflow-tracking/3062746970085784/06c5949c43594a7faf4b5b40273dfda1/artifacts/forcast_ARIMA-{_random_string_}.csv/forcast_ARIMA.csv")

# # Read the CSV file
df_retrieved = pd.read_csv(artifact_path)
# display(df_retrieved)
