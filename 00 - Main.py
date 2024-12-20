# Databricks notebook source
dbutils.widgets.dropdown("TS_generation", "False",["True","False"],label=None)

dbutils.widgets.dropdown("TS_generation_insights", "False",["True","False"],label=None)

dbutils.widgets.dropdown("TS_FeatureEng", "False",["True","False"],label=None)

dbutils.widgets.dropdown("TS_FeatureEng_insights", "False",["True","False"],label=None)

dbutils.widgets.dropdown("TS_store_split_raw", "False",["True","False"],label=None)

dbutils.widgets.dropdown("TS_store_feature_eng", "False",["True","False"],label=None)

# COMMAND ----------

TS_generation = dbutils.widgets.get("TS_generation") in ['True']
TS_generation_insights = dbutils.widgets.get("TS_generation_insights") in ['True']
TS_FeatureEng = dbutils.widgets.get("TS_FeatureEng") in ['True']
TS_FeatureEng_insights = dbutils.widgets.get("TS_FeatureEng_insights") in ['True']

TS_store_split_raw = dbutils.widgets.get("TS_store_split_raw") in ['True']
TS_store_feature_eng = dbutils.widgets.get("TS_store_feature_eng") in ['True']

# COMMAND ----------

if TS_generation:
    dbutils.notebook.run("./01a - TS_generation",60)
if TS_generation_insights:
    dbutils.notebook.run("./01b - TS_generation_insights",60)
if TS_FeatureEng:
    dbutils.notebook.run("./02a - TS_FeatureEng",6000,{"TS_store_split_raw": f'{TS_store_split_raw}',"TS_store_feature_eng": f'{TS_store_feature_eng}'})
if TS_FeatureEng_insights:
    dbutils.notebook.run("./02b - TS_FeatureEng_insights",600)


# COMMAND ----------

import requests

def find_job_id(instance, headers, job_name, offset_limit=1000):
    params = {"offset": 0}
    uri = f"{instance}/api/2.1/jobs/list"
    done = False
    job_id = None
    while not done:
        done = True
        res = requests.get(uri, params=params, headers=headers)
        assert res.status_code == 200, f"Job list not returned; {res.content}"
        
        jobs = res.json().get("jobs", [])
        if len(jobs) > 0:
            for job in jobs:
                if job.get("settings", {}).get("name", None) == job_name:
                    job_id = job.get("job_id", None)
                    break

            # if job_id not found; update the offset and try again
            if job_id is None:
                params["offset"] += len(jobs)
                if params["offset"] < offset_limit:
                    done = False
    
    return job_id

def get_job_parameters(job_name, cluster_id, notebook_path):
    params = {
            "name": job_name,
            "tasks": [{"task_key": "webhook_task", 
                       "existing_cluster_id": cluster_id,
                       "notebook_task": {
                           "notebook_path": notebook_path
                       }
                      }]
        }
    return params

def get_create_parameters(job_name, cluster_id, notebook_path):
    api = "api/2.1/jobs/create"
    return api, get_job_parameters(job_name, cluster_id, notebook_path)

def get_reset_parameters(job_name, cluster_id, notebook_path, job_id):
    api = "api/2.1/jobs/reset"
    params = {"job_id": job_id, "new_settings": get_job_parameters(job_name, cluster_id, notebook_path)}
    return api, params

def get_webhook_job(instance, headers, job_name, cluster_id, notebook_path):
    job_id = find_job_id(instance, headers, job_name)
    if job_id is None:
        api, params = get_create_parameters(job_name, cluster_id, notebook_path)
    else:
        api, params = get_reset_parameters(job_name, cluster_id, notebook_path, job_id)
    
    uri = f"{instance}/{api}"
    res = requests.post(uri, headers=headers, json=params)
    assert res.status_code == 200, f"Expected an HTTP 200 response, received {res.status_code}; {res.content}"
    job_id = res.json().get("job_id", job_id)
    return job_id

notebook_path = mlflow.utils.databricks_utils.get_notebook_path().replace("03a-Webhooks-and-Testing", "03b-Webhooks-Job-Demo")

# We can use our utility method for creating a unique 
# database name to help us construct a unique job name.
prefix = DA.unique_name("-")
job_name = f"{prefix}_webhook-job"

# if the Job was created via UI, set it here.
job_id = get_webhook_job(instance, 
                         headers, 
                         job_name,
                         spark.conf.get("spark.databricks.clusterUsageTags.clusterId"),
                         notebook_path)

print(f"Job ID:   {job_id}")
print(f"Job name: {job_name}")
