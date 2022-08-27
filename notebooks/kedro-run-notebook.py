# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install -r ../src/requirements.txt

# COMMAND ----------

import os
if not os.path.isdir("/tmp/databricks-sof-models-churn/logs"):
  os.makedirs("/tmp/databricks-sof-models-churn/logs")
if not os.path.isdir("/dbfs/FileStore/databricks-sof-models-churn/shared/data"):
  os.makedirs("/dbfs/FileStore/databricks-sof-models-churn/shared/data")
  
### for dev env if you are doing multiple runs for your own purpose
#if not os.path.isdir("/dbfs/FileStore/databricks-sof-models-churn/{my_username}/data"):
#  os.makedirs("/dbfs/FileStore/databricks-sof-models-churn/{my_username}/data")

# COMMAND ----------

# DBTITLE 1,Run Pipeline
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import logging
import os
import pandas as pd

## grabbing the job parameters
environment=dbutils.widgets.get("env")
action= dbutils.widgets.get("action")

def date_exception(x):
  if pd.to_datetime(x).date() > pd.datetime.now().date():
    raise Exception(f"Invalid date : {x}. Date cannot be in future ")
  return x

empty_str_to_None = lambda x: None if x == "" else date_exception(x)
end_date_pred= empty_str_to_None(dbutils.widgets.get("end_date_pred"))
end_date_train= empty_str_to_None(dbutils.widgets.get("end_date_train"))
  
extra_params = {"common": {"end_date_pred": end_date_pred,
                           "end_date_train": end_date_train}}
os.chdir('..')

# suppress excessive logging from py4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

#setting project root directory
project_root = os.getcwd()
bootstrap_project(project_root)

if environment in ['prod','stage']:
  if action == 'train':
    with KedroSession.create(project_path=project_root, save_on_close=True, env=environment,extra_params=extra_params) as session:
      session.run(pipeline_name="de")
      session.run(pipeline_name="ptt")
      session.run(pipeline_name="mt")
      session.run(pipeline_name="mv")
  else:
    with KedroSession.create(project_path=project_root, save_on_close=True, env=environment,extra_params=extra_params) as session:
      session.run(pipeline_name="de")
      session.run(pipeline_name="pdn")
elif environment in ['dev']:
  with KedroSession.create(project_path=project_root, save_on_close=True, env=environment,extra_params=extra_params) as session:
    session.run(pipeline_name="de")
    session.run(pipeline_name="ptt")
    session.run(pipeline_name="mt")
    session.run(pipeline_name="mv")
    session.run(pipeline_name="pdn")
else:
  raise Exception(f"environment {environment} not recognized")


# COMMAND ----------

## Manual way of exporting the logs after pipeline run 
from datetime import datetime
from pytz import timezone
import os.path
now = datetime.now(timezone('US/Pacific')).strftime('%Y-%m-%d %H:%M:%S')


if environment=='dev':
  dbutils.fs.cp("file:/tmp/databricks-sof-models-churn/logs/info.log/", f"dbfs:/mnt/sofdlzonelab_data-science-dev/shared/silver/source/data_science/customer_churn/logs/info_{now}.log")
  
if environment=='stage':
  dbutils.fs.cp("file:/tmp/databricks-sof-models-churn/logs/info.log/", f"dbfs:/mnt/sofdlzonelab_main-data/silver/source/data_science/customer_churn/logs/info_{now}.log")
  
if environment=='prod':
  dbutils.fs.cp("file:/tmp/databricks-sof-models-churn/logs/info.log/", f"dbfs:/mnt/sofdlzone2s_main-data/source/data_science/customer_churn/logs/info_{now}.log")
 

