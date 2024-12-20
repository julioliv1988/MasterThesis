# Databricks notebook source
import pandas as pd

# COMMAND ----------

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

    for itens_ in [transform_string(item) for item in total_list if "/Feature_eng/FIR" in item]:
        csv_list.append(itens_)
# csv_list


# COMMAND ----------

regressor_list = ["LR","RF","KNN"]

FIR_dic = {}

for _series_ in serie_names_list:
    FIR_dic[_series_] = {}

    # print("############### \n\n")
    # print(_series_)
    len_partial = len(f"/dbfs/mnt/automated_mounts_sas/0juliostoragetest/julio/master_data/{_series_}/Feature_eng/FIR ")
    partial_list = [item for item in csv_list if f"{_series_}/" in item]
    for regresors in regressor_list:

        # print(f"{regresors} \n")
        internal_list = [item for item in partial_list if f"{regresors}" in item]
        # print(internal_list)
        internal_list_ = [item[len_partial:-len(regresors)-1] for item in internal_list]

        FIR_dic[_series_][regresors] = {}

        for i in internal_list_:
            FIR_dic[_series_][regresors][i] = {}
            FIR_dic[_series_][regresors][i]["full_path"] = [item for item in internal_list if f"{i}" in item][0]
            FIR_dic[_series_][regresors][i]["df"] = pd.read_csv(FIR_dic[_series_][regresors][i]["full_path"])
            FIR_dic[_series_][regresors][i]["rank_df_inverted"] = FIR_dic[_series_][regresors][i]["df"]
            FIR_dic[_series_][regresors][i]["rank_df_inverted"]['rank'] = range(1, len(FIR_dic[_series_][regresors][i]["rank_df_inverted"]) + 1)
            FIR_dic[_series_][regresors][i]["rank_df_inverted"] = FIR_dic[_series_][regresors][i]["rank_df_inverted"].set_index("Feature").T.iloc[2:]



# COMMAND ----------

# FIR_dic["SerieNumber_0"]["LR"]["1 season"]["rank_df_inverted"]

# COMMAND ----------

FIR_dic.keys()

# COMMAND ----------

variation_list = ["season-noise", "trend-noise", "trend-season", "none"]

# COMMAND ----------

[s for s in FIR_dic.keys() if "season-noise" in s]

# COMMAND ----------

FIR_dic[_series_]["LR"].keys()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ### All runs

# COMMAND ----------

# _1season_lr
_1season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "1 season" in FIR_dic[_series_]["LR"]:
        _1season_lr = pd.concat([_1season_lr, FIR_dic[_series_]["LR"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_lr = _1season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_lr = median_1season_lr.rename(columns={median_1season_lr.columns[1]: 'median'})
median_1season_lr = median_1season_lr.sort_values(by=median_1season_lr.columns[1])

# _2season_lr
_2season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "2 season" in FIR_dic[_series_]["LR"]:
        _2season_lr = pd.concat([_2season_lr, FIR_dic[_series_]["LR"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_lr = _2season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_lr = median_2season_lr.rename(columns={median_2season_lr.columns[1]: 'median'})
median_2season_lr = median_2season_lr.sort_values(by=median_2season_lr.columns[1])

# _Noseason_lr
_Noseason_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "No season" in FIR_dic[_series_]["LR"]:
        _Noseason_lr = pd.concat([_Noseason_lr, FIR_dic[_series_]["LR"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_lr = _Noseason_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_lr = median_Noseason_lr.rename(columns={median_Noseason_lr.columns[1]: 'median'})
median_Noseason_lr = median_Noseason_lr.sort_values(by=median_Noseason_lr.columns[1])

# _Noseason_season_lr -> non seasonal runs eg: trend-noise
_Noseason_season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "No season" not in FIR_dic[_series_]["LR"]:
        keys = FIR_dic[_series_]["LR"].keys()
        for key_ in keys:
            _Noseason_season_lr = pd.concat([_Noseason_season_lr, FIR_dic[_series_]["LR"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_season_lr = _Noseason_season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_season_lr = median_Noseason_season_lr.rename(columns={median_Noseason_season_lr.columns[1]: 'median'})
median_Noseason_season_lr = median_Noseason_season_lr.sort_values(by=median_Noseason_season_lr.columns[1])

# COMMAND ----------

print("############### All runs ###############")
# _Noseason_lr.describe()
# _Noseason_season_lr.describe()
# _1season_lr.describe()
# _2season_lr.describe()
print("median_Noseason_season_lr")
display(median_Noseason_season_lr)
print("median_Noseason_lr")
display(median_Noseason_lr)
print("median_1season_lr")
display(median_1season_lr)
print("median_2season_lr")
display(median_2season_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### season-noise variation

# COMMAND ----------

# season-noise variation

# _1season_lr
_1season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "1 season" in FIR_dic[_series_]["LR"]:
            _1season_lr = pd.concat([_1season_lr, FIR_dic[_series_]["LR"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_lr = _1season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_lr = median_1season_lr.rename(columns={median_1season_lr.columns[1]: 'median'})
median_1season_lr = median_1season_lr.sort_values(by=median_1season_lr.columns[1])

# _2season_lr
_2season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "2 season" in FIR_dic[_series_]["LR"]:
            _2season_lr = pd.concat([_2season_lr, FIR_dic[_series_]["LR"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_lr = _2season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_lr = median_2season_lr.rename(columns={median_2season_lr.columns[1]: 'median'})
median_2season_lr = median_2season_lr.sort_values(by=median_2season_lr.columns[1])

# _Noseason_lr
_Noseason_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "No season" in FIR_dic[_series_]["LR"]:
            _Noseason_lr = pd.concat([_Noseason_lr, FIR_dic[_series_]["LR"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_lr = _Noseason_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_lr = median_Noseason_lr.rename(columns={median_Noseason_lr.columns[1]: 'median'})
median_Noseason_lr = median_Noseason_lr.sort_values(by=median_Noseason_lr.columns[1])

# _Noseason_season_lr -> there's no _Noseason_season_lr variation for "season-noise variation"
# _Noseason_season_lr = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "season-noise" in _series_:
#         if "No season" not in FIR_dic[_series_]["LR"]:
#             keys = FIR_dic[_series_]["LR"].keys()
#             for key_ in keys:
#                 _Noseason_season_lr = pd.concat([_Noseason_season_lr, FIR_dic[_series_]["LR"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_lr = _Noseason_season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_lr = median_Noseason_season_lr.rename(columns={median_Noseason_season_lr.columns[1]: 'median'})
# median_Noseason_season_lr = median_Noseason_season_lr.sort_values(by=median_Noseason_season_lr.columns[1])

# COMMAND ----------

print("############### season-noise variation ###############")
# _Noseason_lr.describe()
# _Noseason_season_lr.describe()
# _1season_lr.describe()
# _2season_lr.describe()
# print("median_Noseason_season_lr")
# display(median_Noseason_season_lr)
print("median_Noseason_lr")
display(median_Noseason_lr)
print("median_1season_lr")
display(median_1season_lr)
print("median_2season_lr")
display(median_2season_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season variation

# COMMAND ----------

# trend-season variation

# _1season_lr
_1season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "1 season" in FIR_dic[_series_]["LR"]:
            _1season_lr = pd.concat([_1season_lr, FIR_dic[_series_]["LR"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_lr = _1season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_lr = median_1season_lr.rename(columns={median_1season_lr.columns[1]: 'median'})
median_1season_lr = median_1season_lr.sort_values(by=median_1season_lr.columns[1])

# _2season_lr
_2season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "2 season" in FIR_dic[_series_]["LR"]:
            _2season_lr = pd.concat([_2season_lr, FIR_dic[_series_]["LR"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_lr = _2season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_lr = median_2season_lr.rename(columns={median_2season_lr.columns[1]: 'median'})
median_2season_lr = median_2season_lr.sort_values(by=median_2season_lr.columns[1])

# _Noseason_lr
_Noseason_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "No season" in FIR_dic[_series_]["LR"]:
            _Noseason_lr = pd.concat([_Noseason_lr, FIR_dic[_series_]["LR"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_lr = _Noseason_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_lr = median_Noseason_lr.rename(columns={median_Noseason_lr.columns[1]: 'median'})
median_Noseason_lr = median_Noseason_lr.sort_values(by=median_Noseason_lr.columns[1])

# # _Noseason_season_lr -> there's no _Noseason_season_lr variation for "trend-season variation"
# _Noseason_season_lr = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-season" in _series_:
#         if "No season" not in FIR_dic[_series_]["LR"]:
#             keys = FIR_dic[_series_]["LR"].keys()
#             for key_ in keys:
#                 _Noseason_season_lr = pd.concat([_Noseason_season_lr, FIR_dic[_series_]["LR"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_lr = _Noseason_season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_lr = median_Noseason_season_lr.rename(columns={median_Noseason_season_lr.columns[1]: 'median'})
# median_Noseason_season_lr = median_Noseason_season_lr.sort_values(by=median_Noseason_season_lr.columns[1])

# COMMAND ----------

print("############### trend-season variation ###############")
# _Noseason_lr.describe()
# _Noseason_season_lr.describe()
# _1season_lr.describe()
# _2season_lr.describe()
# print("median_Noseason_season_lr")
# display(median_Noseason_season_lr)
print("median_Noseason_lr")
display(median_Noseason_lr)
print("median_1season_lr")
display(median_1season_lr)
print("median_2season_lr")
display(median_2season_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-noise variation

# COMMAND ----------

# trend-noise variation

# # _1season_lr -> -> there's just _Noseason_season_lr variation for "trend-noise variation"
# _1season_lr = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "1 season" in FIR_dic[_series_]["LR"]:
#             _1season_lr = pd.concat([_1season_lr, FIR_dic[_series_]["LR"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_1season_lr = _1season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_1season_lr = median_1season_lr.rename(columns={median_1season_lr.columns[1]: 'median'})
# median_1season_lr = median_1season_lr.sort_values(by=median_1season_lr.columns[1])

# # _2season_lr
# _2season_lr = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "2 season" in FIR_dic[_series_]["LR"]:
#             _2season_lr = pd.concat([_2season_lr, FIR_dic[_series_]["LR"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_2season_lr = _2season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_2season_lr = median_2season_lr.rename(columns={median_2season_lr.columns[1]: 'median'})
# median_2season_lr = median_2season_lr.sort_values(by=median_2season_lr.columns[1])

# # _Noseason_lr
# _Noseason_lr = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "No season" in FIR_dic[_series_]["LR"]:
#             _Noseason_lr = pd.concat([_Noseason_lr, FIR_dic[_series_]["LR"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_lr = _Noseason_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_lr = median_Noseason_lr.rename(columns={median_Noseason_lr.columns[1]: 'median'})
# median_Noseason_lr = median_Noseason_lr.sort_values(by=median_Noseason_lr.columns[1])

# _Noseason_season_lr
_Noseason_season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-noise" in _series_:
        if "No season" not in FIR_dic[_series_]["LR"]:
            keys = FIR_dic[_series_]["LR"].keys()
            for key_ in keys:
                _Noseason_season_lr = pd.concat([_Noseason_season_lr, FIR_dic[_series_]["LR"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_season_lr = _Noseason_season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_season_lr = median_Noseason_season_lr.rename(columns={median_Noseason_season_lr.columns[1]: 'median'})
median_Noseason_season_lr = median_Noseason_season_lr.sort_values(by=median_Noseason_season_lr.columns[1])

# COMMAND ----------

print("############### trend-noise variation ###############")
# _Noseason_lr.describe()
# _Noseason_season_lr.describe()
# _1season_lr.describe()
# _2season_lr.describe()
print("median_Noseason_season_lr")
display(median_Noseason_season_lr)
# print("median_Noseason_lr")
# display(median_Noseason_lr)
# print("median_1season_lr")
# display(median_1season_lr)
# print("median_2season_lr")
# display(median_2season_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season-noise (complete)

# COMMAND ----------

# trend-season-noise (complete)

# _1season_lr
_1season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "1 season" in FIR_dic[_series_]["LR"]:
            _1season_lr = pd.concat([_1season_lr, FIR_dic[_series_]["LR"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_lr = _1season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_lr = median_1season_lr.rename(columns={median_1season_lr.columns[1]: 'median'})
median_1season_lr = median_1season_lr.sort_values(by=median_1season_lr.columns[1])

# _2season_lr
_2season_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "2 season" in FIR_dic[_series_]["LR"]:
            _2season_lr = pd.concat([_2season_lr, FIR_dic[_series_]["LR"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_lr = _2season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_lr = median_2season_lr.rename(columns={median_2season_lr.columns[1]: 'median'})
median_2season_lr = median_2season_lr.sort_values(by=median_2season_lr.columns[1])

# _Noseason_lr
_Noseason_lr = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "No season" in FIR_dic[_series_]["LR"]:
            _Noseason_lr = pd.concat([_Noseason_lr, FIR_dic[_series_]["LR"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_lr = _Noseason_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_lr = median_Noseason_lr.rename(columns={median_Noseason_lr.columns[1]: 'median'})
median_Noseason_lr = median_Noseason_lr.sort_values(by=median_Noseason_lr.columns[1])

# # _Noseason_season_lr -> there's no _Noseason_season_lr for "trend-season-noise"
# _Noseason_season_lr = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-season" in _series_:
#         if "No season" not in FIR_dic[_series_]["LR"]:
#             keys = FIR_dic[_series_]["LR"].keys()
#             for key_ in keys:
#                 _Noseason_season_lr = pd.concat([_Noseason_season_lr, FIR_dic[_series_]["LR"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_lr = _Noseason_season_lr.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_lr = median_Noseason_season_lr.rename(columns={median_Noseason_season_lr.columns[1]: 'median'})
# median_Noseason_season_lr = median_Noseason_season_lr.sort_values(by=median_Noseason_season_lr.columns[1])

# COMMAND ----------

print("############### trend-season variation ###############")
# _Noseason_lr.describe()
# _Noseason_season_lr.describe()
# _1season_lr.describe()
# _2season_lr.describe()
# print("median_Noseason_season_lr")
# display(median_Noseason_season_lr)
print("median_Noseason_lr")
display(median_Noseason_lr)
print("median_1season_lr")
display(median_1season_lr)
print("median_2season_lr")
display(median_2season_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC ### All runs

# COMMAND ----------

# _1season_rf
_1season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "1 season" in FIR_dic[_series_]["RF"]:
        _1season_rf = pd.concat([_1season_rf, FIR_dic[_series_]["RF"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_rf = _1season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_rf = median_1season_rf.rename(columns={median_1season_rf.columns[1]: 'median'})
median_1season_rf = median_1season_rf.sort_values(by=median_1season_rf.columns[1])

# _2season_rf
_2season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "2 season" in FIR_dic[_series_]["RF"]:
        _2season_rf = pd.concat([_2season_rf, FIR_dic[_series_]["RF"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_rf = _2season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_rf = median_2season_rf.rename(columns={median_2season_rf.columns[1]: 'median'})
median_2season_rf = median_2season_rf.sort_values(by=median_2season_rf.columns[1])

# _Noseason_rf
_Noseason_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "No season" in FIR_dic[_series_]["RF"]:
        _Noseason_rf = pd.concat([_Noseason_rf, FIR_dic[_series_]["RF"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_rf = _Noseason_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_rf = median_Noseason_rf.rename(columns={median_Noseason_rf.columns[1]: 'median'})
median_Noseason_rf = median_Noseason_rf.sort_values(by=median_Noseason_rf.columns[1])

# _Noseason_season_rf -> non seasonal runs eg: trend-noise
_Noseason_season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "No season" not in FIR_dic[_series_]["RF"]:
        keys = FIR_dic[_series_]["RF"].keys()
        for key_ in keys:
            _Noseason_season_rf = pd.concat([_Noseason_season_rf, FIR_dic[_series_]["RF"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_season_rf = _Noseason_season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_season_rf = median_Noseason_season_rf.rename(columns={median_Noseason_season_rf.columns[1]: 'median'})
median_Noseason_season_rf = median_Noseason_season_rf.sort_values(by=median_Noseason_season_rf.columns[1])

# COMMAND ----------

print("############### All runs ###############")
# _Noseason_rf.describe()
# _Noseason_season_rf.describe()
# _1season_rf.describe()
# _2season_rf.describe()
print("median_Noseason_season_rf")
display(median_Noseason_season_rf)
print("median_Noseason_rf")
display(median_Noseason_rf)
print("median_1season_rf")
display(median_1season_rf)
print("median_2season_rf")
display(median_2season_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### season-noise variation

# COMMAND ----------

# season-noise variation

# _1season_rf
_1season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "1 season" in FIR_dic[_series_]["RF"]:
            _1season_rf = pd.concat([_1season_rf, FIR_dic[_series_]["RF"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_rf = _1season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_rf = median_1season_rf.rename(columns={median_1season_rf.columns[1]: 'median'})
median_1season_rf = median_1season_rf.sort_values(by=median_1season_rf.columns[1])

# _2season_rf
_2season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "2 season" in FIR_dic[_series_]["RF"]:
            _2season_rf = pd.concat([_2season_rf, FIR_dic[_series_]["RF"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_rf = _2season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_rf = median_2season_rf.rename(columns={median_2season_rf.columns[1]: 'median'})
median_2season_rf = median_2season_rf.sort_values(by=median_2season_rf.columns[1])

# _Noseason_rf
_Noseason_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "No season" in FIR_dic[_series_]["RF"]:
            _Noseason_rf = pd.concat([_Noseason_rf, FIR_dic[_series_]["RF"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_rf = _Noseason_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_rf = median_Noseason_rf.rename(columns={median_Noseason_rf.columns[1]: 'median'})
median_Noseason_rf = median_Noseason_rf.sort_values(by=median_Noseason_rf.columns[1])

# _Noseason_season_rf -> there's no _Noseason_season_rf variation for "season-noise variation"
# _Noseason_season_rf = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "season-noise" in _series_:
#         if "No season" not in FIR_dic[_series_]["RF"]:
#             keys = FIR_dic[_series_]["RF"].keys()
#             for key_ in keys:
#                 _Noseason_season_rf = pd.concat([_Noseason_season_rf, FIR_dic[_series_]["RF"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_rf = _Noseason_season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_rf = median_Noseason_season_rf.rename(columns={median_Noseason_season_rf.columns[1]: 'median'})
# median_Noseason_season_rf = median_Noseason_season_rf.sort_values(by=median_Noseason_season_rf.columns[1])

# COMMAND ----------

print("############### season-noise variation ###############")
# _Noseason_rf.describe()
# _Noseason_season_rf.describe()
# _1season_rf.describe()
# _2season_rf.describe()
# print("median_Noseason_season_rf")
# display(median_Noseason_season_rf)
print("median_Noseason_rf")
display(median_Noseason_rf)
print("median_1season_rf")
display(median_1season_rf)
print("median_2season_rf")
display(median_2season_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season variation

# COMMAND ----------

# trend-season variation

# _1season_rf
_1season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "1 season" in FIR_dic[_series_]["RF"]:
            _1season_rf = pd.concat([_1season_rf, FIR_dic[_series_]["RF"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_rf = _1season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_rf = median_1season_rf.rename(columns={median_1season_rf.columns[1]: 'median'})
median_1season_rf = median_1season_rf.sort_values(by=median_1season_rf.columns[1])

# _2season_rf
_2season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "2 season" in FIR_dic[_series_]["RF"]:
            _2season_rf = pd.concat([_2season_rf, FIR_dic[_series_]["RF"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_rf = _2season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_rf = median_2season_rf.rename(columns={median_2season_rf.columns[1]: 'median'})
median_2season_rf = median_2season_rf.sort_values(by=median_2season_rf.columns[1])

# _Noseason_rf
_Noseason_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "No season" in FIR_dic[_series_]["RF"]:
            _Noseason_rf = pd.concat([_Noseason_rf, FIR_dic[_series_]["RF"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_rf = _Noseason_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_rf = median_Noseason_rf.rename(columns={median_Noseason_rf.columns[1]: 'median'})
median_Noseason_rf = median_Noseason_rf.sort_values(by=median_Noseason_rf.columns[1])

# # _Noseason_season_rf -> there's no _Noseason_season_rf variation for "trend-season variation"
# _Noseason_season_rf = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-season" in _series_:
#         if "No season" not in FIR_dic[_series_]["RF"]:
#             keys = FIR_dic[_series_]["RF"].keys()
#             for key_ in keys:
#                 _Noseason_season_rf = pd.concat([_Noseason_season_rf, FIR_dic[_series_]["RF"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_rf = _Noseason_season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_rf = median_Noseason_season_rf.rename(columns={median_Noseason_season_rf.columns[1]: 'median'})
# median_Noseason_season_rf = median_Noseason_season_rf.sort_values(by=median_Noseason_season_rf.columns[1])

# COMMAND ----------

print("############### season-noise variation ###############")
# _Noseason_rf.describe()
# _Noseason_season_rf.describe()
# _1season_rf.describe()
# _2season_rf.describe()
# print("median_Noseason_season_rf")
# display(median_Noseason_season_rf)
print("median_Noseason_rf")
display(median_Noseason_rf)
print("median_1season_rf")
display(median_1season_rf)
print("median_2season_rf")
display(median_2season_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-noise variation

# COMMAND ----------

# trend-noise variation

# # _1season_rf -> -> there's just _Noseason_season_rf variation for "trend-noise variation"
# _1season_rf = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "1 season" in FIR_dic[_series_]["RF"]:
#             _1season_rf = pd.concat([_1season_rf, FIR_dic[_series_]["RF"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_1season_rf = _1season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_1season_rf = median_1season_rf.rename(columns={median_1season_rf.columns[1]: 'median'})
# median_1season_rf = median_1season_rf.sort_values(by=median_1season_rf.columns[1])

# # _2season_rf
# _2season_rf = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "2 season" in FIR_dic[_series_]["RF"]:
#             _2season_rf = pd.concat([_2season_rf, FIR_dic[_series_]["RF"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_2season_rf = _2season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_2season_rf = median_2season_rf.rename(columns={median_2season_rf.columns[1]: 'median'})
# median_2season_rf = median_2season_rf.sort_values(by=median_2season_rf.columns[1])

# # _Noseason_rf
# _Noseason_rf = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "No season" in FIR_dic[_series_]["RF"]:
#             _Noseason_rf = pd.concat([_Noseason_rf, FIR_dic[_series_]["RF"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_rf = _Noseason_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_rf = median_Noseason_rf.rename(columns={median_Noseason_rf.columns[1]: 'median'})
# median_Noseason_rf = median_Noseason_rf.sort_values(by=median_Noseason_rf.columns[1])

# _Noseason_season_rf
_Noseason_season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-noise" in _series_:
        if "No season" not in FIR_dic[_series_]["RF"]:
            keys = FIR_dic[_series_]["RF"].keys()
            for key_ in keys:
                _Noseason_season_rf = pd.concat([_Noseason_season_rf, FIR_dic[_series_]["RF"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_season_rf = _Noseason_season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_season_rf = median_Noseason_season_rf.rename(columns={median_Noseason_season_rf.columns[1]: 'median'})
median_Noseason_season_rf = median_Noseason_season_rf.sort_values(by=median_Noseason_season_rf.columns[1])

# COMMAND ----------

print("############### trend-noise variation ###############")
# _Noseason_rf.describe()
# _Noseason_season_rf.describe()
# _1season_rf.describe()
# _2season_rf.describe()
print("median_Noseason_season_rf")
display(median_Noseason_season_rf)
# print("median_Noseason_rf")
# display(median_Noseason_rf)
# print("median_1season_rf")
# display(median_1season_rf)
# print("median_2season_rf")
# display(median_2season_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season-noise (complete)

# COMMAND ----------

# trend-season-noise (complete)

# _1season_rf
_1season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "1 season" in FIR_dic[_series_]["RF"]:
            _1season_rf = pd.concat([_1season_rf, FIR_dic[_series_]["RF"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_rf = _1season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_rf = median_1season_rf.rename(columns={median_1season_rf.columns[1]: 'median'})
median_1season_rf = median_1season_rf.sort_values(by=median_1season_rf.columns[1])

# _2season_rf
_2season_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "2 season" in FIR_dic[_series_]["RF"]:
            _2season_rf = pd.concat([_2season_rf, FIR_dic[_series_]["RF"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_rf = _2season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_rf = median_2season_rf.rename(columns={median_2season_rf.columns[1]: 'median'})
median_2season_rf = median_2season_rf.sort_values(by=median_2season_rf.columns[1])

# _Noseason_rf
_Noseason_rf = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "No season" in FIR_dic[_series_]["RF"]:
            _Noseason_rf = pd.concat([_Noseason_rf, FIR_dic[_series_]["RF"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_rf = _Noseason_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_rf = median_Noseason_rf.rename(columns={median_Noseason_rf.columns[1]: 'median'})
median_Noseason_rf = median_Noseason_rf.sort_values(by=median_Noseason_rf.columns[1])

# # _Noseason_season_rf -> there's no _Noseason_season_rf for "trend-season-noise"
# _Noseason_season_rf = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-season" in _series_:
#         if "No season" not in FIR_dic[_series_]["RF"]:
#             keys = FIR_dic[_series_]["RF"].keys()
#             for key_ in keys:
#                 _Noseason_season_rf = pd.concat([_Noseason_season_rf, FIR_dic[_series_]["RF"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_rf = _Noseason_season_rf.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_rf = median_Noseason_season_rf.rename(columns={median_Noseason_season_rf.columns[1]: 'median'})
# median_Noseason_season_rf = median_Noseason_season_rf.sort_values(by=median_Noseason_season_rf.columns[1])

# COMMAND ----------

print("############### trend-season variation ###############")
# _Noseason_rf.describe()
# _Noseason_season_rf.describe()
# _1season_rf.describe()
# _2season_rf.describe()
# print("median_Noseason_season_rf")
# display(median_Noseason_season_rf)
print("median_Noseason_rf")
display(median_Noseason_rf)
print("median_1season_rf")
display(median_1season_rf)
print("median_2season_rf")
display(median_2season_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## KNN

# COMMAND ----------

# MAGIC %md
# MAGIC ### All runs

# COMMAND ----------

# _1season_knn
_1season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "1 season" in FIR_dic[_series_]["RF"]:
        _1season_knn = pd.concat([_1season_knn, FIR_dic[_series_]["RF"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_knn = _1season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_knn = median_1season_knn.rename(columns={median_1season_knn.columns[1]: 'median'})
median_1season_knn = median_1season_knn.sort_values(by=median_1season_knn.columns[1])

# _2season_knn
_2season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "2 season" in FIR_dic[_series_]["RF"]:
        _2season_knn = pd.concat([_2season_knn, FIR_dic[_series_]["RF"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_knn = _2season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_knn = median_2season_knn.rename(columns={median_2season_knn.columns[1]: 'median'})
median_2season_knn = median_2season_knn.sort_values(by=median_2season_knn.columns[1])

# _Noseason_knn
_Noseason_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "No season" in FIR_dic[_series_]["RF"]:
        _Noseason_knn = pd.concat([_Noseason_knn, FIR_dic[_series_]["RF"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_knn = _Noseason_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_knn = median_Noseason_knn.rename(columns={median_Noseason_knn.columns[1]: 'median'})
median_Noseason_knn = median_Noseason_knn.sort_values(by=median_Noseason_knn.columns[1])

# _Noseason_season_knn -> non seasonal runs eg: trend-noise
_Noseason_season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "No season" not in FIR_dic[_series_]["RF"]:
        keys = FIR_dic[_series_]["RF"].keys()
        for key_ in keys:
            _Noseason_season_knn = pd.concat([_Noseason_season_knn, FIR_dic[_series_]["RF"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_season_knn = _Noseason_season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_season_knn = median_Noseason_season_knn.rename(columns={median_Noseason_season_knn.columns[1]: 'median'})
median_Noseason_season_knn = median_Noseason_season_knn.sort_values(by=median_Noseason_season_knn.columns[1])

# COMMAND ----------

print("############### All runs ###############")
# _Noseason_knn.describe()
# _Noseason_season_knn.describe()
# _1season_knn.describe()
# _2season_knn.describe()
print("median_Noseason_season_knn")
display(median_Noseason_season_knn)
print("median_Noseason_knn")
display(median_Noseason_knn)
print("median_1season_knn")
display(median_1season_knn)
print("median_2season_knn")
display(median_2season_knn)

# COMMAND ----------

# MAGIC %md
# MAGIC ### season-noise variation

# COMMAND ----------

# season-noise variation

# _1season_knn
_1season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "1 season" in FIR_dic[_series_]["KNN"]:
            _1season_knn = pd.concat([_1season_knn, FIR_dic[_series_]["KNN"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_knn = _1season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_knn = median_1season_knn.rename(columns={median_1season_knn.columns[1]: 'median'})
median_1season_knn = median_1season_knn.sort_values(by=median_1season_knn.columns[1])

# _2season_knn
_2season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "2 season" in FIR_dic[_series_]["KNN"]:
            _2season_knn = pd.concat([_2season_knn, FIR_dic[_series_]["KNN"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_knn = _2season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_knn = median_2season_knn.rename(columns={median_2season_knn.columns[1]: 'median'})
median_2season_knn = median_2season_knn.sort_values(by=median_2season_knn.columns[1])

# _Noseason_knn
_Noseason_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "season-noise" in _series_:
        if "No season" in FIR_dic[_series_]["KNN"]:
            _Noseason_knn = pd.concat([_Noseason_knn, FIR_dic[_series_]["KNN"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_knn = _Noseason_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_knn = median_Noseason_knn.rename(columns={median_Noseason_knn.columns[1]: 'median'})
median_Noseason_knn = median_Noseason_knn.sort_values(by=median_Noseason_knn.columns[1])

# _Noseason_season_knn -> there's no _Noseason_season_knn variation for "season-noise variation"
# _Noseason_season_knn = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "season-noise" in _series_:
#         if "No season" not in FIR_dic[_series_]["KNN"]:
#             keys = FIR_dic[_series_]["KNN"].keys()
#             for key_ in keys:
#                 _Noseason_season_knn = pd.concat([_Noseason_season_knn, FIR_dic[_series_]["KNN"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_knn = _Noseason_season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_knn = median_Noseason_season_knn.rename(columns={median_Noseason_season_knn.columns[1]: 'median'})
# median_Noseason_season_knn = median_Noseason_season_knn.sort_values(by=median_Noseason_season_knn.columns[1])

# COMMAND ----------

print("############### season-noise variation ###############")
# _Noseason_knn.describe()
# _Noseason_season_knn.describe()
# _1season_knn.describe()
# _2season_knn.describe()
# print("median_Noseason_season_knn")
# display(median_Noseason_season_knn)
print("median_Noseason_knn")
display(median_Noseason_knn)
print("median_1season_knn")
display(median_1season_knn)
print("median_2season_knn")
display(median_2season_knn)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season variation

# COMMAND ----------

# trend-season variation

# _1season_knn
_1season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "1 season" in FIR_dic[_series_]["KNN"]:
            _1season_knn = pd.concat([_1season_knn, FIR_dic[_series_]["KNN"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_knn = _1season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_knn = median_1season_knn.rename(columns={median_1season_knn.columns[1]: 'median'})
median_1season_knn = median_1season_knn.sort_values(by=median_1season_knn.columns[1])

# _2season_knn
_2season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "2 season" in FIR_dic[_series_]["KNN"]:
            _2season_knn = pd.concat([_2season_knn, FIR_dic[_series_]["KNN"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_knn = _2season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_knn = median_2season_knn.rename(columns={median_2season_knn.columns[1]: 'median'})
median_2season_knn = median_2season_knn.sort_values(by=median_2season_knn.columns[1])

# _Noseason_knn
_Noseason_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-season" in _series_:
        if "No season" in FIR_dic[_series_]["KNN"]:
            _Noseason_knn = pd.concat([_Noseason_knn, FIR_dic[_series_]["KNN"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_knn = _Noseason_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_knn = median_Noseason_knn.rename(columns={median_Noseason_knn.columns[1]: 'median'})
median_Noseason_knn = median_Noseason_knn.sort_values(by=median_Noseason_knn.columns[1])

# # _Noseason_season_knn -> there's no _Noseason_season_knn variation for "trend-season variation"
# _Noseason_season_knn = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-season" in _series_:
#         if "No season" not in FIR_dic[_series_]["KNN"]:
#             keys = FIR_dic[_series_]["KNN"].keys()
#             for key_ in keys:
#                 _Noseason_season_knn = pd.concat([_Noseason_season_knn, FIR_dic[_series_]["KNN"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_knn = _Noseason_season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_knn = median_Noseason_season_knn.rename(columns={median_Noseason_season_knn.columns[1]: 'median'})
# median_Noseason_season_knn = median_Noseason_season_knn.sort_values(by=median_Noseason_season_knn.columns[1])

# COMMAND ----------

print("############### trend-season variation ###############")
# _Noseason_knn.describe()
# _Noseason_season_knn.describe()
# _1season_knn.describe()
# _2season_knn.describe()
# print("median_Noseason_season_knn")
# display(median_Noseason_season_knn)
print("median_Noseason_knn")
display(median_Noseason_knn)
print("median_1season_knn")
display(median_1season_knn)
print("median_2season_knn")
display(median_2season_knn)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-noise variation

# COMMAND ----------

# trend-noise variation

# # _1season_knn -> -> there's just _Noseason_season_knn variation for "trend-noise variation"
# _1season_knn = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "1 season" in FIR_dic[_series_]["KNN"]:
#             _1season_knn = pd.concat([_1season_knn, FIR_dic[_series_]["KNN"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_1season_knn = _1season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_1season_knn = median_1season_knn.rename(columns={median_1season_knn.columns[1]: 'median'})
# median_1season_knn = median_1season_knn.sort_values(by=median_1season_knn.columns[1])

# # _2season_knn
# _2season_knn = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "2 season" in FIR_dic[_series_]["KNN"]:
#             _2season_knn = pd.concat([_2season_knn, FIR_dic[_series_]["KNN"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_2season_knn = _2season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_2season_knn = median_2season_knn.rename(columns={median_2season_knn.columns[1]: 'median'})
# median_2season_knn = median_2season_knn.sort_values(by=median_2season_knn.columns[1])

# # _Noseason_knn
# _Noseason_knn = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-noise" in _series_:
#         if "No season" in FIR_dic[_series_]["KNN"]:
#             _Noseason_knn = pd.concat([_Noseason_knn, FIR_dic[_series_]["KNN"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_knn = _Noseason_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_knn = median_Noseason_knn.rename(columns={median_Noseason_knn.columns[1]: 'median'})
# median_Noseason_knn = median_Noseason_knn.sort_values(by=median_Noseason_knn.columns[1])

# _Noseason_season_knn
_Noseason_season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if "trend-noise" in _series_:
        if "No season" not in FIR_dic[_series_]["KNN"]:
            keys = FIR_dic[_series_]["KNN"].keys()
            for key_ in keys:
                _Noseason_season_knn = pd.concat([_Noseason_season_knn, FIR_dic[_series_]["KNN"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_season_knn = _Noseason_season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_season_knn = median_Noseason_season_knn.rename(columns={median_Noseason_season_knn.columns[1]: 'median'})
median_Noseason_season_knn = median_Noseason_season_knn.sort_values(by=median_Noseason_season_knn.columns[1])

# COMMAND ----------

print("############### trend-noise variation ###############")
# _Noseason_knn.describe()
# _Noseason_season_knn.describe()
# _1season_knn.describe()
# _2season_knn.describe()
print("median_Noseason_season_knn")
display(median_Noseason_season_knn)
# print("median_Noseason_knn")
# display(median_Noseason_knn)
# print("median_1season_knn")
# display(median_1season_knn)
# print("median_2season_knn")
# display(median_2season_knn)

# COMMAND ----------

# MAGIC %md
# MAGIC ### trend-season-noise (complete)

# COMMAND ----------

# trend-season-noise (complete)

# _1season_knn
_1season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "1 season" in FIR_dic[_series_]["KNN"]:
            _1season_knn = pd.concat([_1season_knn, FIR_dic[_series_]["KNN"]["1 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_1season_knn = _1season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_1season_knn = median_1season_knn.rename(columns={median_1season_knn.columns[1]: 'median'})
median_1season_knn = median_1season_knn.sort_values(by=median_1season_knn.columns[1])

# _2season_knn
_2season_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "2 season" in FIR_dic[_series_]["KNN"]:
            _2season_knn = pd.concat([_2season_knn, FIR_dic[_series_]["KNN"]["2 season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_2season_knn = _2season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_2season_knn = median_2season_knn.rename(columns={median_2season_knn.columns[1]: 'median'})
median_2season_knn = median_2season_knn.sort_values(by=median_2season_knn.columns[1])

# _Noseason_knn
_Noseason_knn = pd.DataFrame()
for _series_ in serie_names_list:
    if ("trend-noise" not in _series_) and ("trend-season" not in _series_) and ("season-noise" not in _series_):
        if "No season" in FIR_dic[_series_]["KNN"]:
            _Noseason_knn = pd.concat([_Noseason_knn, FIR_dic[_series_]["KNN"]["No season"]["rank_df_inverted"]], ignore_index=True).fillna(0)

median_Noseason_knn = _Noseason_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
median_Noseason_knn = median_Noseason_knn.rename(columns={median_Noseason_knn.columns[1]: 'median'})
median_Noseason_knn = median_Noseason_knn.sort_values(by=median_Noseason_knn.columns[1])

# # _Noseason_season_knn -> there's no _Noseason_season_knn for "trend-season-noise"
# _Noseason_season_knn = pd.DataFrame()
# for _series_ in serie_names_list:
#     if "trend-season" in _series_:
#         if "No season" not in FIR_dic[_series_]["KNN"]:
#             keys = FIR_dic[_series_]["KNN"].keys()
#             for key_ in keys:
#                 _Noseason_season_knn = pd.concat([_Noseason_season_knn, FIR_dic[_series_]["KNN"][key_]["rank_df_inverted"]], ignore_index=True).fillna(0)

# median_Noseason_season_knn = _Noseason_season_knn.describe().iloc[[5]].T.reset_index().rename(columns={'index': 'Feature'})
# median_Noseason_season_knn = median_Noseason_season_knn.rename(columns={median_Noseason_season_knn.columns[1]: 'median'})
# median_Noseason_season_knn = median_Noseason_season_knn.sort_values(by=median_Noseason_season_knn.columns[1])

# COMMAND ----------

print("############### trend-season variation ###############")
# _Noseason_knn.describe()
# _Noseason_season_knn.describe()
# _1season_knn.describe()
# _2season_knn.describe()
# print("median_Noseason_season_knn")
# display(median_Noseason_season_knn)
print("median_Noseason_knn")
display(median_Noseason_knn)
print("median_1season_knn")
display(median_1season_knn)
print("median_2season_knn")
display(median_2season_knn)
