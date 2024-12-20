# Databricks notebook source
# MAGIC %md # Imports and Functions

# COMMAND ----------

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

# MAGIC %md # Dictionary Definition

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

regressor_list = ["LR features removed", "KNN features removed","RF features removed"]

ba_fo_elim_dic = {}

for _series_ in serie_names_list:
    ba_fo_elim_dic[_series_] = {}

    # print("############### \n\n")
    # print(_series_)
    # len_partial = len(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/{_series_}/Feature_eng/FIR ")

    partial_list = [item for item in csv_list if f"{_series_}/" in item]

    partial_list_back = [item for item in partial_list if "back" in item][0]

    partial_list_foward = [item for item in partial_list if f"foward" in item][0]

    ba_fo_elim_dic[_series_]["fo_df"] = pd.read_csv(partial_list_foward)

    ba_fo_elim_dic[_series_]["ba_df"] = pd.read_csv(partial_list_back)

    ba_fo_elim_dic[_series_]["foward"] = {}

    ba_fo_elim_dic[_series_]["backward"] = {}


    for regresors in regressor_list:

        ba_fo_elim_dic[_series_]["foward"][regresors] = ba_fo_elim_dic[_series_]["fo_df"].set_index("type")[regresors].to_dict()

        for _keys_ in ba_fo_elim_dic[_series_]["foward"][regresors].keys():

            ba_fo_elim_dic[_series_]["foward"][regresors][_keys_] = literal_eval(ba_fo_elim_dic[_series_]["foward"][regresors][_keys_])

        ba_fo_elim_dic[_series_]["backward"][regresors] = ba_fo_elim_dic[_series_]["ba_df"].set_index("type")[regresors].to_dict()

        for _keys_ in ba_fo_elim_dic[_series_]["backward"][regresors].keys():

            ba_fo_elim_dic[_series_]["backward"][regresors][_keys_] = literal_eval(ba_fo_elim_dic[_series_]["backward"][regresors][_keys_])



# COMMAND ----------

ba_fo_elim_dic["SerieNumber_0"]["fo_df"]

# COMMAND ----------

ba_fo_elim_dic["SerieNumber_0"]["foward"]['LR features removed']

# COMMAND ----------

# Checking all the features

df = ba_fo_elim_dic['SerieNumber_0']["fo_df"][["LR features","LR features removed"]]
# Apply the function to each element in the DataFrame
df['LR features'] = df['LR features'].apply(convert_to_list)
df['LR features removed'] = df['LR features removed'].apply(convert_to_list)

df['JoinedArray'] = df['LR features'] + df['LR features removed']

all_features = df["JoinedArray"].iloc[-1]

all_features

# COMMAND ----------

ba_fo_elim_dic["SerieNumber_0"]['foward']["KNN features removed"]

# COMMAND ----------

# MAGIC %md # Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ### All runs

# COMMAND ----------

######################################################Foward######################################################
# _1season_lr_fo_removed
_1season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
        _1season_lr_fo_removed = _1season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['1 season']
# _2season_lr_fo_removed
_2season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '2 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
        _2season_lr_fo_removed = _2season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['2 season']
# _Noseason_lr_fo_removed
_Noseason_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if 'No season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
        _Noseason_lr_fo_removed = _Noseason_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['No season']
# _Noseason_season_lr_fo_removed
_Noseason_season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' not in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
        for _keys_ in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _Noseason_season_lr_fo_removed = _Noseason_season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"][_keys_]

######################################################Backward######################################################
# _1season_lr_ba_removed
_1season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
        _1season_lr_ba_removed = _1season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['1 season']
# _2season_lr_ba_removed
_2season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '2 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
        _2season_lr_ba_removed = _2season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['2 season']
# _Noseason_lr_ba_removed
_Noseason_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if 'No season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
        _Noseason_lr_ba_removed = _Noseason_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['No season']
# _Noseason_season_lr_ba_removed
_Noseason_season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' not in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
        for _keys_ in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _Noseason_season_lr_ba_removed = _Noseason_season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"][_keys_]

print("##################### All runs #####################")
print("######### Foward #########")

print("_1season_lr_fo_removed")
display(count_occurrences(_1season_lr_fo_removed))
print("_2season_lr_fo_removed")
display(count_occurrences(_2season_lr_fo_removed))
print("_Noseason_lr_fo_removed")
display(count_occurrences(_Noseason_lr_fo_removed))
print("_Noseason_season_lr_fo_removed")
display(count_occurrences(_Noseason_season_lr_fo_removed))

print("######### Backward #########")
print("_1season_lr_ba_removed")
display(count_occurrences(_1season_lr_ba_removed))
print("_2season_lr_ba_removed")
display(count_occurrences(_2season_lr_ba_removed))
print("_Noseason_lr_ba_removed")
display(count_occurrences(_Noseason_lr_ba_removed))
print("_Noseason_season_lr_ba_removed")
display(count_occurrences(_Noseason_season_lr_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### season-noise variation

# COMMAND ----------

######################################################Foward######################################################
# _1season_lr_fo_removed
_1season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _1season_lr_fo_removed = _1season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['1 season']
# _2season_lr_fo_removed
_2season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _2season_lr_fo_removed = _2season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['2 season']
# _Noseason_lr_fo_removed
_Noseason_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _Noseason_lr_fo_removed = _Noseason_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['No season']
# # _Noseason_season_lr_fo_removed -> there's no _Noseason_season_lr_fo_removed variation for "season-noise variation"
# _Noseason_season_lr_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "season-noise" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#                 _Noseason_season_lr_fo_removed = _Noseason_season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"][_keys_]

######################################################Backward######################################################
# _1season_lr_ba_removed
_1season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _1season_lr_ba_removed = _1season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['1 season']
# _2season_lr_ba_removed
_2season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _2season_lr_ba_removed = _2season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['2 season']
# _Noseason_lr_ba_removed
_Noseason_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _Noseason_lr_ba_removed = _Noseason_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['No season']
# # _Noseason_season_lr_ba_removed -> there's no _Noseason_season_lr_ba_removed variation for "season-noise variation"
# _Noseason_season_lr_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "season-noise" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#                 _Noseason_season_lr_ba_removed = _Noseason_season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"][_keys_]

print("############### season-noise variation ###############")
print("######### Foward #########")

print("_1season_lr_fo_removed")
display(count_occurrences(_1season_lr_fo_removed))
print("_2season_lr_fo_removed")
display(count_occurrences(_2season_lr_fo_removed))
print("_Noseason_lr_fo_removed")
display(count_occurrences(_Noseason_lr_fo_removed))
# print("_Noseason_season_lr_fo_removed")
# display(count_occurrences(_Noseason_season_lr_fo_removed))

print("######### Backward #########")

print("_1season_lr_ba_removed")
display(count_occurrences(_1season_lr_ba_removed))
print("_2season_lr_ba_removed")
display(count_occurrences(_2season_lr_ba_removed))
print("_Noseason_lr_ba_removed")
display(count_occurrences(_Noseason_lr_ba_removed))
# print("_Noseason_season_lr_ba_removed")
# display(count_occurrences(_Noseason_season_lr_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season variation

# COMMAND ----------

######################################################Foward######################################################
# _1season_lr_fo_removed
_1season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _1season_lr_fo_removed = _1season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['1 season']
# _2season_lr_fo_removed
_2season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _2season_lr_fo_removed = _2season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['2 season']
# _Noseason_lr_fo_removed
_Noseason_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _Noseason_lr_fo_removed = _Noseason_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['No season']
# # _Noseason_season_lr_fo_removed -> there's no _Noseason_season_lr_fo_removed variation for "trend-season variation"
# _Noseason_season_lr_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-season" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#                 _Noseason_season_lr_fo_removed = _Noseason_season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"][_keys_]

######################################################Backward######################################################
# _1season_lr_ba_removed
_1season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _1season_lr_ba_removed = _1season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['1 season']
# _2season_lr_ba_removed
_2season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _2season_lr_ba_removed = _2season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['2 season']
# _Noseason_lr_ba_removed
_Noseason_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _Noseason_lr_ba_removed = _Noseason_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['No season']
# # _Noseason_season_lr_ba_removed -> there's no _Noseason_season_lr_ba_removed variation for "trend-season variation"
# _Noseason_season_lr_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-season" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#                 _Noseason_season_lr_ba_removed = _Noseason_season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"][_keys_]


print("############### trend-season variation ###############")
print("######### Foward #########")
print("_1season_lr_fo_removed")
display(count_occurrences(_1season_lr_fo_removed))
print("_2season_lr_fo_removed")
display(count_occurrences(_2season_lr_fo_removed))
print("_Noseason_lr_fo_removed")
display(count_occurrences(_Noseason_lr_fo_removed))
# print("_Noseason_season_lr_fo_removed")
# display(count_occurrences(_Noseason_season_lr_fo_removed))

print("######### Backward #########")

print("_1season_lr_ba_removed")
display(count_occurrences(_1season_lr_ba_removed))
print("_2season_lr_ba_removed")
display(count_occurrences(_2season_lr_ba_removed))
print("_Noseason_lr_ba_removed")
display(count_occurrences(_Noseason_lr_ba_removed))
# print("_Noseason_season_lr_ba_removed")
# display(count_occurrences(_Noseason_season_lr_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-noise variation

# COMMAND ----------

######################################################Foward######################################################
# # _1season_lr_fo_removed -> there's just _Noseason_season_lr_fo_removed variation for "trend-noise variation"
# _1season_lr_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '1 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#             _1season_lr_fo_removed = _1season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['1 season']
# # _2season_lr_fo_removed
# _2season_lr_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '2 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#             _2season_lr_fo_removed = _2season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['2 season']
# # _Noseason_lr_fo_removed
# _Noseason_lr_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if 'No season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#             _Noseason_lr_fo_removed = _Noseason_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['No season']
# _Noseason_season_lr_fo_removed 
_Noseason_season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-noise" in _series_:
        if '1 season' not in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            for _keys_ in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
                _Noseason_season_lr_fo_removed = _Noseason_season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"][_keys_]

######################################################Backward######################################################
# # _1season_lr_ba_removed -> there's just _Noseason_season_lr_ba_removed variation for "trend-noise variation"
# _1season_lr_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '1 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#             _1season_lr_ba_removed = _1season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['1 season']
# # _2season_lr_ba_removed
# _2season_lr_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '2 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#             _2season_lr_ba_removed = _2season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['2 season']
# # _Noseason_lr_ba_removed
# _Noseason_lr_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if 'No season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#             _Noseason_lr_ba_removed = _Noseason_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['No season']
# _Noseason_season_lr_ba_removed 
_Noseason_season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-noise" in _series_:
        if '1 season' not in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            for _keys_ in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
                _Noseason_season_lr_ba_removed = _Noseason_season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"][_keys_]

print("############### trend-noise variation ###############")
print("######### Foward #########")
# print("_1season_lr_fo_removed")
# display(count_occurrences(_1season_lr_fo_removed))
# print("_2season_lr_fo_removed")
# display(count_occurrences(_2season_lr_fo_removed))
# print("_Noseason_lr_fo_removed")
# display(count_occurrences(_Noseason_lr_fo_removed))
print("_Noseason_season_lr_fo_removed")
display(count_occurrences(_Noseason_season_lr_fo_removed))

print("######### Backward #########")

# print("_1season_lr_ba_removed")
# display(count_occurrences(_1season_lr_ba_removed))
# print("_2season_lr_ba_removed")
# display(count_occurrences(_2season_lr_ba_removed))
# print("_Noseason_lr_ba_removed")
# display(count_occurrences(_Noseason_lr_ba_removed))
print("_Noseason_season_lr_ba_removed")
display(count_occurrences(_Noseason_season_lr_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season-noise (complete)

# COMMAND ----------

######################################################Foward######################################################
# _1season_lr_fo_removed
_1season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _1season_lr_fo_removed = _1season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['1 season']
# _2season_lr_fo_removed
_2season_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _2season_lr_fo_removed = _2season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['2 season']
# _Noseason_lr_fo_removed
_Noseason_lr_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
            _Noseason_lr_fo_removed = _Noseason_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"]['No season']
# # _Noseason_season_lr_fo_removed -> there's no _Noseason_season_lr_fo_removed variation for "season-noise variation"
# _Noseason_season_lr_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["LR features removed"].keys():
#                 _Noseason_season_lr_fo_removed = _Noseason_season_lr_fo_removed + ba_fo_elim_dic[_series_]['foward']["LR features removed"][_keys_]

######################################################Backward######################################################
# _1season_lr_ba_removed
_1season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _1season_lr_ba_removed = _1season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['1 season']
# _2season_lr_ba_removed
_2season_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _2season_lr_ba_removed = _2season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['2 season']
# _Noseason_lr_ba_removed
_Noseason_lr_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
            _Noseason_lr_ba_removed = _Noseason_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"]['No season']
# # _Noseason_season_lr_ba_removed -> there's no _Noseason_season_lr_ba_removed variation for "season-noise variation"
# _Noseason_season_lr_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["LR features removed"].keys():
#                 _Noseason_season_lr_ba_removed = _Noseason_season_lr_ba_removed + ba_fo_elim_dic[_series_]['backward']["LR features removed"][_keys_]


print("############### trend-season-noise (complete) ###############")
print("######### Foward #########")
print("_1season_lr_fo_removed")
display(count_occurrences(_1season_lr_fo_removed))
print("_2season_lr_fo_removed")
display(count_occurrences(_2season_lr_fo_removed))
print("_Noseason_lr_fo_removed")
display(count_occurrences(_Noseason_lr_fo_removed))
# print("_Noseason_season_lr_fo_removed")
# display(count_occurrences(_Noseason_season_lr_fo_removed))

print("######### Backward #########")
print("_1season_lr_ba_removed")
display(count_occurrences(_1season_lr_ba_removed))
print("_2season_lr_ba_removed")
display(count_occurrences(_2season_lr_ba_removed))
print("_Noseason_lr_ba_removed")
display(count_occurrences(_Noseason_lr_ba_removed))
# print("_Noseason_season_lr_ba_removed")
# display(count_occurrences(_Noseason_season_lr_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC ### All runs

# COMMAND ----------

######################################################Foward######################################################
# _1season_rf_fo_removed
_1season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
        _1season_rf_fo_removed = _1season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['1 season']
# _2season_rf_fo_removed
_2season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '2 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
        _2season_rf_fo_removed = _2season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['2 season']
# _Noseason_rf_fo_removed
_Noseason_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if 'No season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
        _Noseason_rf_fo_removed = _Noseason_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['No season']
# _Noseason_season_rf_fo_removed
_Noseason_season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' not in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
        for _keys_ in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _Noseason_season_rf_fo_removed = _Noseason_season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"][_keys_]

######################################################Backward######################################################
# _1season_rf_ba_removed
_1season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
        _1season_rf_ba_removed = _1season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['1 season']
# _2season_rf_ba_removed
_2season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '2 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
        _2season_rf_ba_removed = _2season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['2 season']
# _Noseason_rf_ba_removed
_Noseason_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if 'No season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
        _Noseason_rf_ba_removed = _Noseason_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['No season']
# _Noseason_season_rf_ba_removed
_Noseason_season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' not in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
        for _keys_ in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _Noseason_season_rf_ba_removed = _Noseason_season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"][_keys_]

print("############### All runs ###############")
print("######### Foward #########")
print("_1season_rf_fo_removed")
display(count_occurrences(_1season_rf_fo_removed))
print("_2season_rf_fo_removed")
display(count_occurrences(_2season_rf_fo_removed))
print("_Noseason_rf_fo_removed")
display(count_occurrences(_Noseason_rf_fo_removed))
print("_Noseason_season_rf_fo_removed")
display(count_occurrences(_Noseason_season_rf_fo_removed))

print("######### Backward #########")
print("_1season_rf_ba_removed")
display(count_occurrences(_1season_rf_ba_removed))
print("_2season_rf_ba_removed")
display(count_occurrences(_2season_rf_ba_removed))
print("_Noseason_rf_ba_removed")
display(count_occurrences(_Noseason_rf_ba_removed))
print("_Noseason_season_rf_ba_removed")
display(count_occurrences(_Noseason_season_rf_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### season-noise variation

# COMMAND ----------

######################################################Foward######################################################
# _1season_rf_fo_removed
_1season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _1season_rf_fo_removed = _1season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['1 season']
# _2season_rf_fo_removed
_2season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _2season_rf_fo_removed = _2season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['2 season']
# _Noseason_rf_fo_removed
_Noseason_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _Noseason_rf_fo_removed = _Noseason_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['No season']
# # _Noseason_season_rf_fo_removed -> there's no _Noseason_season_rf_fo_removed variation for "season-noise variation"
# _Noseason_season_rf_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "season-noise" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#                 _Noseason_season_rf_fo_removed = _Noseason_season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"][_keys_]

######################################################Backward######################################################
# _1season_rf_ba_removed
_1season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _1season_rf_ba_removed = _1season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['1 season']
# _2season_rf_ba_removed
_2season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _2season_rf_ba_removed = _2season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['2 season']
# _Noseason_rf_ba_removed
_Noseason_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _Noseason_rf_ba_removed = _Noseason_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['No season']
# # _Noseason_season_rf_ba_removed -> there's no _Noseason_season_rf_ba_removed variation for "season-noise variation"
# _Noseason_season_rf_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "season-noise" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#                 _Noseason_season_rf_ba_removed = _Noseason_season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"][_keys_]

print("############### season-noise variation ###############")
print("######### Foward #########")
print("_1season_rf_fo_removed")
display(count_occurrences(_1season_rf_fo_removed))
print("_2season_rf_fo_removed")
display(count_occurrences(_2season_rf_fo_removed))
print("_Noseason_rf_fo_removed")
display(count_occurrences(_Noseason_rf_fo_removed))
# print("_Noseason_season_rf_fo_removed")
# display(count_occurrences(_Noseason_season_rf_fo_removed))

print("######### Backward #########")
print("_1season_rf_ba_removed")
display(count_occurrences(_1season_rf_ba_removed))
print("_2season_rf_ba_removed")
display(count_occurrences(_2season_rf_ba_removed))
print("_Noseason_rf_ba_removed")
display(count_occurrences(_Noseason_rf_ba_removed))
# print("_Noseason_season_rf_ba_removed")
# display(count_occurrences(_Noseason_season_rf_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season variation

# COMMAND ----------

######################################################Foward######################################################
# _1season_rf_fo_removed
_1season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _1season_rf_fo_removed = _1season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['1 season']
# _2season_rf_fo_removed
_2season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _2season_rf_fo_removed = _2season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['2 season']
# _Noseason_rf_fo_removed
_Noseason_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _Noseason_rf_fo_removed = _Noseason_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['No season']
# # _Noseason_season_rf_fo_removed -> there's no _Noseason_season_rf_fo_removed variation for "trend-season variation"
# _Noseason_season_rf_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-season" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#                 _Noseason_season_rf_fo_removed = _Noseason_season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"][_keys_]

######################################################Backward######################################################
# _1season_rf_ba_removed
_1season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _1season_rf_ba_removed = _1season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['1 season']
# _2season_rf_ba_removed
_2season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _2season_rf_ba_removed = _2season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['2 season']
# _Noseason_rf_ba_removed
_Noseason_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _Noseason_rf_ba_removed = _Noseason_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['No season']
# # _Noseason_season_rf_ba_removed -> there's no _Noseason_season_rf_ba_removed variation for "trend-season variation"
# _Noseason_season_rf_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-season" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#                 _Noseason_season_rf_ba_removed = _Noseason_season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"][_keys_]

print("############### trend-season variation ###############")
print("######### Foward #########")
print("_1season_rf_fo_removed")
display(count_occurrences(_1season_rf_fo_removed))
print("_2season_rf_fo_removed")
display(count_occurrences(_2season_rf_fo_removed))
print("_Noseason_rf_fo_removed")
display(count_occurrences(_Noseason_rf_fo_removed))
# print("_Noseason_season_rf_fo_removed")
# display(count_occurrences(_Noseason_season_rf_fo_removed))

print("######### Backward #########")
print("_1season_rf_ba_removed")
display(count_occurrences(_1season_rf_ba_removed))
print("_2season_rf_ba_removed")
display(count_occurrences(_2season_rf_ba_removed))
print("_Noseason_rf_ba_removed")
display(count_occurrences(_Noseason_rf_ba_removed))
# print("_Noseason_season_rf_ba_removed")
# display(count_occurrences(_Noseason_season_rf_ba_removed))


# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-noise variation

# COMMAND ----------

######################################################Foward######################################################
# # _1season_rf_fo_removed -> there's just _Noseason_season_rf_fo_removed variation for "trend-noise variation"
# _1season_rf_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '1 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#             _1season_rf_fo_removed = _1season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['1 season']
# # _2season_rf_fo_removed
# _2season_rf_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '2 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#             _2season_rf_fo_removed = _2season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['2 season']
# # _Noseason_rf_fo_removed
# _Noseason_rf_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if 'No season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#             _Noseason_rf_fo_removed = _Noseason_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['No season']
# _Noseason_season_rf_fo_removed 
_Noseason_season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-noise" in _series_:
        if '1 season' not in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            for _keys_ in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
                _Noseason_season_rf_fo_removed = _Noseason_season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"][_keys_]

######################################################Backward######################################################
# # _1season_rf_ba_removed -> there's just _Noseason_season_rf_ba_removed variation for "trend-noise variation"
# _1season_rf_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '1 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#             _1season_rf_ba_removed = _1season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['1 season']
# # _2season_rf_ba_removed
# _2season_rf_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '2 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#             _2season_rf_ba_removed = _2season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['2 season']
# # _Noseason_rf_ba_removed
# _Noseason_rf_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if 'No season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#             _Noseason_rf_ba_removed = _Noseason_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['No season']
# _Noseason_season_rf_ba_removed 
_Noseason_season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-noise" in _series_:
        if '1 season' not in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            for _keys_ in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
                _Noseason_season_rf_ba_removed = _Noseason_season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"][_keys_]


print("############### trend-noise variation ###############")
print("######### Foward #########")
# print("_1season_rf_fo_removed")
# display(count_occurrences(_1season_rf_fo_removed))
# print("_2season_rf_fo_removed")
# display(count_occurrences(_2season_rf_fo_removed))
# print("_Noseason_rf_fo_removed")
# display(count_occurrences(_Noseason_rf_fo_removed))
print("_Noseason_season_rf_fo_removed")
display(count_occurrences(_Noseason_season_rf_fo_removed))

print("######### Backward #########")
# print("_1season_rf_ba_removed")
# display(count_occurrences(_1season_rf_ba_removed))
# print("_2season_rf_ba_removed")
# display(count_occurrences(_2season_rf_ba_removed))
# print("_Noseason_rf_ba_removed")
# display(count_occurrences(_Noseason_rf_ba_removed))
print("_Noseason_season_rf_ba_removed")
display(count_occurrences(_Noseason_season_rf_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season-noise (complete)

# COMMAND ----------

######################################################Foward######################################################
# _1season_rf_fo_removed
_1season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _1season_rf_fo_removed = _1season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['1 season']
# _2season_rf_fo_removed
_2season_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _2season_rf_fo_removed = _2season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['2 season']
# _Noseason_rf_fo_removed
_Noseason_rf_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
            _Noseason_rf_fo_removed = _Noseason_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"]['No season']
# # _Noseason_season_rf_fo_removed -> there's no _Noseason_season_rf_fo_removed variation for "season-noise variation"
# _Noseason_season_rf_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["RF features removed"].keys():
#                 _Noseason_season_rf_fo_removed = _Noseason_season_rf_fo_removed + ba_fo_elim_dic[_series_]['foward']["RF features removed"][_keys_]

######################################################Backward######################################################
# _1season_rf_ba_removed
_1season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _1season_rf_ba_removed = _1season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['1 season']
# _2season_rf_ba_removed
_2season_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _2season_rf_ba_removed = _2season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['2 season']
# _Noseason_rf_ba_removed
_Noseason_rf_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
            _Noseason_rf_ba_removed = _Noseason_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"]['No season']
# # _Noseason_season_rf_ba_removed -> there's no _Noseason_season_rf_ba_removed variation for "season-noise variation"
# _Noseason_season_rf_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["RF features removed"].keys():
#                 _Noseason_season_rf_ba_removed = _Noseason_season_rf_ba_removed + ba_fo_elim_dic[_series_]['backward']["RF features removed"][_keys_]


print("############### trend-season-noise (complete) ###############")
print("######### Foward #########")
print("_1season_rf_fo_removed")
display(count_occurrences(_1season_rf_fo_removed))
print("_2season_rf_fo_removed")
display(count_occurrences(_2season_rf_fo_removed))
print("_Noseason_rf_fo_removed")
display(count_occurrences(_Noseason_rf_fo_removed))
# print("_Noseason_season_rf_fo_removed")
# display(count_occurrences(_Noseason_season_rf_fo_removed))

print("######### Backward #########")
print("_1season_rf_ba_removed")
display(count_occurrences(_1season_rf_ba_removed))
print("_2season_rf_ba_removed")
display(count_occurrences(_2season_rf_ba_removed))
print("_Noseason_rf_ba_removed")
display(count_occurrences(_Noseason_rf_ba_removed))
# print("_Noseason_season_rf_ba_removed")
# display(count_occurrences(_Noseason_season_rf_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ## KNN

# COMMAND ----------

# MAGIC %md
# MAGIC ### All runs

# COMMAND ----------

######################################################Foward######################################################
# _1season_knn_fo_removed
_1season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
        _1season_knn_fo_removed = _1season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['1 season']
# _2season_knn_fo_removed
_2season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '2 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
        _2season_knn_fo_removed = _2season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['2 season']
# _Noseason_knn_fo_removed
_Noseason_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if 'No season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
        _Noseason_knn_fo_removed = _Noseason_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['No season']
# _Noseason_season_knn_fo_removed
_Noseason_season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' not in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
        for _keys_ in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _Noseason_season_knn_fo_removed = _Noseason_season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"][_keys_]

######################################################Backward######################################################
# _1season_knn_ba_removed
_1season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
        _1season_knn_ba_removed = _1season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['1 season']
# _2season_knn_ba_removed
_2season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '2 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
        _2season_knn_ba_removed = _2season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['2 season']
# _Noseason_knn_ba_removed
_Noseason_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if 'No season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
        _Noseason_knn_ba_removed = _Noseason_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['No season']
# _Noseason_season_knn_ba_removed
_Noseason_season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if '1 season' not in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
        for _keys_ in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _Noseason_season_knn_ba_removed = _Noseason_season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"][_keys_]

print("############### All runs ###############")
print("######### Foward #########")
print("_1season_knn_fo_removed")
display(count_occurrences(_1season_knn_fo_removed))
print("_2season_knn_fo_removed")
display(count_occurrences(_2season_knn_fo_removed))
print("_Noseason_knn_fo_removed")
display(count_occurrences(_Noseason_knn_fo_removed))
print("_Noseason_season_knn_fo_removed")
display(count_occurrences(_Noseason_season_knn_fo_removed))

print("######### Backward #########")
print("_1season_knn_ba_removed")
display(count_occurrences(_1season_knn_ba_removed))
print("_2season_knn_ba_removed")
display(count_occurrences(_2season_knn_ba_removed))
print("_Noseason_knn_ba_removed")
display(count_occurrences(_Noseason_knn_ba_removed))
print("_Noseason_season_knn_ba_removed")
display(count_occurrences(_Noseason_season_knn_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### season-noise variation

# COMMAND ----------

######################################################Foward######################################################
# _1season_knn_fo_removed
_1season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _1season_knn_fo_removed = _1season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['1 season']
# _2season_knn_fo_removed
_2season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _2season_knn_fo_removed = _2season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['2 season']
# _Noseason_knn_fo_removed
_Noseason_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _Noseason_knn_fo_removed = _Noseason_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['No season']
# # _Noseason_season_knn_fo_removed -> there's no _Noseason_season_knn_fo_removed variation for "season-noise variation"
# _Noseason_season_knn_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "season-noise" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#                 _Noseason_season_knn_fo_removed = _Noseason_season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"][_keys_]

######################################################Backward######################################################
# _1season_knn_ba_removed
_1season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _1season_knn_ba_removed = _1season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['1 season']
# _2season_knn_ba_removed
_2season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _2season_knn_ba_removed = _2season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['2 season']
# _Noseason_knn_ba_removed
_Noseason_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "season-noise" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _Noseason_knn_ba_removed = _Noseason_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['No season']
# # _Noseason_season_knn_ba_removed -> there's no _Noseason_season_knn_ba_removed variation for "season-noise variation"
# _Noseason_season_knn_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "season-noise" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#                 _Noseason_season_knn_ba_removed = _Noseason_season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"][_keys_]

print("############### season-noise variation ###############")
print("######### Foward #########")
print("_1season_knn_fo_removed")
display(count_occurrences(_1season_knn_fo_removed))
print("_2season_knn_fo_removed")
display(count_occurrences(_2season_knn_fo_removed))
print("_Noseason_knn_fo_removed")
display(count_occurrences(_Noseason_knn_fo_removed))
# print("_Noseason_season_knn_fo_removed")
# display(count_occurrences(_Noseason_season_knn_fo_removed))

print("######### Backward #########")
print("_1season_knn_ba_removed")
display(count_occurrences(_1season_knn_ba_removed))
print("_2season_knn_ba_removed")
display(count_occurrences(_2season_knn_ba_removed))
print("_Noseason_knn_ba_removed")
display(count_occurrences(_Noseason_knn_ba_removed))
# print("_Noseason_season_knn_ba_removed")
# display(count_occurrences(_Noseason_season_knn_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season variation

# COMMAND ----------

######################################################Foward######################################################
# _1season_knn_fo_removed
_1season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _1season_knn_fo_removed = _1season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['1 season']
# _2season_knn_fo_removed
_2season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _2season_knn_fo_removed = _2season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['2 season']
# _Noseason_knn_fo_removed
_Noseason_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _Noseason_knn_fo_removed = _Noseason_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['No season']
# # _Noseason_season_knn_fo_removed -> there's no _Noseason_season_knn_fo_removed variation for "trend-season variation"
# _Noseason_season_knn_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-season" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#                 _Noseason_season_knn_fo_removed = _Noseason_season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"][_keys_]

######################################################Backward######################################################
# _1season_knn_ba_removed
_1season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _1season_knn_ba_removed = _1season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['1 season']
# _2season_knn_ba_removed
_2season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _2season_knn_ba_removed = _2season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['2 season']
# _Noseason_knn_ba_removed
_Noseason_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-season" in _series_:
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _Noseason_knn_ba_removed = _Noseason_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['No season']
# # _Noseason_season_knn_ba_removed -> there's no _Noseason_season_knn_ba_removed variation for "trend-season variation"
# _Noseason_season_knn_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-season" in _series_:
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#                 _Noseason_season_knn_ba_removed = _Noseason_season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"][_keys_]

print("############### trend-season variation ###############")
print("######### Foward #########")
print("_1season_knn_fo_removed")
display(count_occurrences(_1season_knn_fo_removed))
print("_2season_knn_fo_removed")
display(count_occurrences(_2season_knn_fo_removed))
print("_Noseason_knn_fo_removed")
display(count_occurrences(_Noseason_knn_fo_removed))
# print("_Noseason_season_knn_fo_removed")
# display(count_occurrences(_Noseason_season_knn_fo_removed))

print("######### Backward #########")
print("_1season_knn_ba_removed")
display(count_occurrences(_1season_knn_ba_removed))
print("_2season_knn_ba_removed")
display(count_occurrences(_2season_knn_ba_removed))
print("_Noseason_knn_ba_removed")
display(count_occurrences(_Noseason_knn_ba_removed))
# print("_Noseason_season_knn_ba_removed")
# display(count_occurrences(_Noseason_season_knn_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-noise variation

# COMMAND ----------

######################################################Foward######################################################
# # _1season_knn_fo_removed -> there's just _Noseason_season_knn_fo_removed variation for "trend-noise variation"
# _1season_knn_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '1 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#             _1season_knn_fo_removed = _1season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['1 season']
# # _2season_knn_fo_removed
# _2season_knn_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '2 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#             _2season_knn_fo_removed = _2season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['2 season']
# # _Noseason_knn_fo_removed
# _Noseason_knn_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if 'No season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#             _Noseason_knn_fo_removed = _Noseason_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['No season']
# _Noseason_season_knn_fo_removed 
_Noseason_season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-noise" in _series_:
        if '1 season' not in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            for _keys_ in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
                _Noseason_season_knn_fo_removed = _Noseason_season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"][_keys_]

######################################################Backward######################################################
# # _1season_knn_ba_removed -> there's just _Noseason_season_knn_ba_removed variation for "trend-noise variation"
# _1season_knn_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '1 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#             _1season_knn_ba_removed = _1season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['1 season']
# # _2season_knn_ba_removed
# _2season_knn_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if '2 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#             _2season_knn_ba_removed = _2season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['2 season']
# # _Noseason_knn_ba_removed
# _Noseason_knn_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if "trend-noise" in _series_:
#         if 'No season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#             _Noseason_knn_ba_removed = _Noseason_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['No season']
# _Noseason_season_knn_ba_removed 
_Noseason_season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if "trend-noise" in _series_:
        if '1 season' not in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            for _keys_ in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
                _Noseason_season_knn_ba_removed = _Noseason_season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"][_keys_]


print("############### trend-noise variation ###############")
print("######### Foward #########")
# print("_1season_knn_fo_removed")
# display(count_occurrences(_1season_knn_fo_removed))
# print("_2season_knn_fo_removed")
# display(count_occurrences(_2season_knn_fo_removed))
# print("_Noseason_knn_fo_removed")
# display(count_occurrences(_Noseason_knn_fo_removed))
print("_Noseason_season_knn_fo_removed")
display(count_occurrences(_Noseason_season_knn_fo_removed))

print("######### Backward #########")
# print("_1season_knn_ba_removed")
# display(count_occurrences(_1season_knn_ba_removed))
# print("_2season_knn_ba_removed")
# display(count_occurrences(_2season_knn_ba_removed))
# print("_Noseason_knn_ba_removed")
# display(count_occurrences(_Noseason_knn_ba_removed))
print("_Noseason_season_knn_ba_removed")
display(count_occurrences(_Noseason_season_knn_ba_removed))

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season-noise (complete)

# COMMAND ----------

######################################################Foward######################################################
# _1season_knn_fo_removed
_1season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '1 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _1season_knn_fo_removed = _1season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['1 season']
# _2season_knn_fo_removed
_2season_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '2 season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _2season_knn_fo_removed = _2season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['2 season']
# _Noseason_knn_fo_removed
_Noseason_knn_fo_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if 'No season' in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
            _Noseason_knn_fo_removed = _Noseason_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"]['No season']
# # _Noseason_season_knn_fo_removed -> there's no _Noseason_season_knn_fo_removed variation for "season-noise variation"
# _Noseason_season_knn_fo_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#     if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
#         if '1 season' not in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['foward']["KNN features removed"].keys():
#                 _Noseason_season_knn_fo_removed = _Noseason_season_knn_fo_removed + ba_fo_elim_dic[_series_]['foward']["KNN features removed"][_keys_]

######################################################Backward######################################################
# _1season_knn_ba_removed
_1season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '1 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _1season_knn_ba_removed = _1season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['1 season']
# _2season_knn_ba_removed
_2season_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if '2 season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _2season_knn_ba_removed = _2season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['2 season']
# _Noseason_knn_ba_removed
_Noseason_knn_ba_removed = []
for _series_ in ba_fo_elim_dic.keys():
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if 'No season' in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
            _Noseason_knn_ba_removed = _Noseason_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"]['No season']
# # _Noseason_season_knn_ba_removed -> there's no _Noseason_season_knn_ba_removed variation for "season-noise variation"
# _Noseason_season_knn_ba_removed = []
# for _series_ in ba_fo_elim_dic.keys():
#    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
#         if '1 season' not in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#             for _keys_ in ba_fo_elim_dic[_series_]['backward']["KNN features removed"].keys():
#                 _Noseason_season_knn_ba_removed = _Noseason_season_knn_ba_removed + ba_fo_elim_dic[_series_]['backward']["KNN features removed"][_keys_]


print("############### trend-season-noise (complete) ###############")
print("######### Foward #########")
print("_1season_knn_fo_removed")
display(count_occurrences(_1season_knn_fo_removed))
print("_2season_knn_fo_removed")
display(count_occurrences(_2season_knn_fo_removed))
print("_Noseason_knn_fo_removed")
display(count_occurrences(_Noseason_knn_fo_removed))
# print("_Noseason_season_knn_fo_removed")
# display(count_occurrences(_Noseason_season_knn_fo_removed))

print("######### Backward #########")
print("_1season_knn_ba_removed")
display(count_occurrences(_1season_knn_ba_removed))
print("_2season_knn_ba_removed")
display(count_occurrences(_2season_knn_ba_removed))
print("_Noseason_knn_ba_removed")
display(count_occurrences(_Noseason_knn_ba_removed))
# print("_Noseason_season_knn_ba_removed")
# display(count_occurrences(_Noseason_season_knn_ba_removed))
