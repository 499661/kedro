# Databricks notebook source
# MAGIC %md
# MAGIC Note Before Running
# MAGIC 
# MAGIC - The python packages only need to be installed if you are using a cluster which does not already have them installed. Note that the clusters `churn-dev` and `churn-staging` already have them installed.
# MAGIC - This notebook is not very stable. It has the following issues:
# MAGIC   - Installing python packages using pip sometimes causes issues. You may see an input error, or errors related to matplotlib. When this happens you can try installing the dependencies again and/or clearing the state and rerunning the notebook.
# MAGIC   - Sometimes converting the data from churn_test_featurized to an array fails because of memory issues. Usually this happens when the function `convert_df_to_arr` is called and the error message mentions an "RPC Error" or a "Memory Error". When this happens you can try running the cell which threw the error again. Alternatively, you can try using a different cluster with more memory or more workers.

# COMMAND ----------

# MAGIC %md
# MAGIC # Install python packages
# MAGIC Need to do this at the very top

# COMMAND ----------

# Installing the required python packages.
# If you are using a cluster on which the python packages are already installed (e.g churn-dev or churn-staging) this is not needed.
# %pip install -r ../src/requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC # Input
# MAGIC Change the parameters for running the notebook here.

# COMMAND ----------

run_id = "3a8dfcc512cd4e7d95140bb809dbc587" # The run_id for the model for which you want to calculate the metrics
churn_test_featurized_path = "/mnt/sofdlzonelab_main-data/silver/source/data_science/customer_churn/input/churn_test_featurized" # the path were the test data is stored on the data lake (unlikely to change).
churn_test_timestamp_as_of = "2022-08-11T06:29:44+00:00" # The timestamp corresponding to the test data that you want to load. You can find this in the model registry.
n_weeks_churn = 13

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC %matplotlib inline

# COMMAND ----------

import logging
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from pyspark.sql import DataFrame
from mlflow.tracking.client import MlflowClient
import sys
sys.path.append("../src")
from sof_models_churn.features.ece_metric_scoring import ChurnPredictionMetrics
from sof_models_churn.features.rnn_model import RNNModel
from sof_models_churn.common import wtte_utils as wtte
from sof_models_churn.common.wtte_weibull import cmf as weibull_cmf
from sof_models_churn.common.mlflow_utils import get_model_from_run_id

# COMMAND ----------

# suppress excessive logging
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model

# COMMAND ----------

prod_model = get_model_from_run_id(run_id, return_sequences=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data

# COMMAND ----------

def convert_df_to_arr(df: DataFrame, n_features_total=16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  '''
  The data is stored on the data lake in delta format. Before feeding it to the model we need to convert it to arrays.
  
  The code is taken from sof_models_churn/pipelines/model_validating/nodes.py
  '''
  arr = [ l[0] for l in df.toPandas().values ] # unpack nested array-list-array structure
  arr = np.stack(arr, axis=0)
  arr = arr.reshape(arr.shape[0], -1, n_features_total + 2 ) # "+2" because y tensor is appended here
  
  x_valid_features = arr[:,:,:-3]
  x_valid_store = arr[:,:,-3]
  y = arr[:,:,-2:]
  
  return x_valid_features, x_valid_store, y

# COMMAND ----------

# Need to specify the version in case the data is overwritten
churn_test_featurized = (
  spark.read.format("delta")
            .option("TimestampAsOf", churn_test_timestamp_as_of)
            .load(churn_test_featurized_path)
)

# COMMAND ----------

x_valid_features, x_valid_store, y = convert_df_to_arr(churn_test_featurized)
print(f"x_valid_features: {x_valid_features.shape}")
print(f"x_valid_store: {x_valid_store.shape}")
print(f"y: {y.shape}")

# COMMAND ----------

predictions = prod_model._model.predict([x_valid_features, x_valid_store], batch_size=2048)
print(f"predictions {predictions.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Churn probabilities and ground truth

# COMMAND ----------

metric_dict = ChurnPredictionMetrics.evaluate_metrics(predictions, y, n_weeks_churn+1, min_box_width=n_weeks_churn,
                                                      consider_last_event=True, dummy_val=0.5)
pred_prob_in_box = metric_dict[n_weeks_churn]["pred_prob_in_box"]
is_in_box = metric_dict[n_weeks_churn]["is_in_box"]

# COMMAND ----------

prob_churn = 1.0 - pred_prob_in_box
churned = (~is_in_box)*1

# COMMAND ----------

print(f"in total {churned.sum()} households churned")

# COMMAND ----------

# MAGIC %md
# MAGIC # Test data "stats"

# COMMAND ----------

print(f"n_rows: {predictions.shape[0]}")
print(f"n_rows after cleaning: {prob_churn.shape[0]}")
print(f"n_rows / test_frac {predictions.shape[0]/0.025}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Probability distribution

# COMMAND ----------

fig, ax = plt.subplots(1,1, figsize=(6,4))
fig.set_tight_layout(True)

ax.hist(prob_churn, 10, color="C0", alpha=0.5, label="model")
ax.hist(churned, 10, color="C1", alpha=0.5, label="ground truth")
ax.set_xlabel("Churn Probability [-]")
ax.set_ylabel("Count [-]")
ax.legend(loc=0)

# COMMAND ----------

# MAGIC %md
# MAGIC # Precision Recall and ROC

# COMMAND ----------

fig, axes = plt.subplots(1,2, figsize=(8,4))
fig.set_tight_layout(True)

axes[0].plot(metric_dict[n_weeks_churn]["recall"], metric_dict[n_weeks_churn]["precision"], '-C0')
axes[0].set_xlabel("Recall [-]")
axes[0].set_ylabel("Precision [-]")

axes[1].plot(metric_dict[n_weeks_churn]["fpr"], metric_dict[n_weeks_churn]["tpr"], '-C0')
axes[1].set_xlabel("FPR [-]")
axes[1].set_ylabel("TPR [-]")
axes[1].plot([0,1],[0,1], "--k")

# COMMAND ----------

# MAGIC %md
# MAGIC # Metrics for business

# COMMAND ----------

def find_bin(val: float, bins: list):
  for i, b in enumerate(bins):
    if b > val:
      break
  return i

# COMMAND ----------

def create_bins(n_bins: int):
  bins = []
  step = 1.0/n_bins
  for i in range(n_bins):
    bins.append(step*(i+1))
  return bins

# COMMAND ----------

def create_aggregate_df(df: pd.DataFrame, n_bins: int):
  bins = create_bins(n_bins)
  df["segment"] = df.apply(lambda x: find_bin(x.prob_churn, bins), axis=1)
  grouped = df.groupby("segment")
  df_agg = grouped.agg(prob_range = ("segment", lambda x: "{:.2f}".format(([0]+bins)[np.unique(x)[0]]) + " - " + "{:.2f}".format(bins[np.unique(x)[0]])),
                       mean_prob = ("prob_churn", lambda x: np.mean(x)),
                       median_prob = ("prob_churn", lambda x: np.median(x)),
                       total_count = ("prob_churn", lambda x: len(x)),
                       churned = ("churned", lambda x: x.sum()))
  df_agg["percent_churned"] = df_agg["churned"] / df_agg["total_count"] * 100
  df_agg["percent_of_total_churned"] = df_agg["churned"] / df_agg["churned"].sum() * 100
  return df_agg.sort_index(ascending=False)

# COMMAND ----------

df = pd.DataFrame(data={"prob_churn": prob_churn,
                        "churned": churned})

# COMMAND ----------

df_agg_4 = create_aggregate_df(df, 4)
df_agg_4 = df_agg_4.rename(index = {3: "At Risk", 2: "High", 1: "Medium", 0: "Low"})
df_agg_4

# COMMAND ----------

df_agg_10 = create_aggregate_df(df, 10)
df_agg_10

# COMMAND ----------

def sanity_check(prob_churn: np.ndarray, churned: np.ndarray, n_bins: int):
  step = 1.0/n_bins

  sum_ = 0
  print("left\tright\ttotal_count\tchurned\t\tpercent_churned")
  for i in range(n_bins):
    left = step*i
    right = step*(i+1)
    if i < n_bins - 1:
      fltr = (prob_churn >= left) & (prob_churn < right)
    else:
      fltr = (prob_churn >= left) & (prob_churn <= right)
    print("{:.2f}\t{:.2f}\t{:10d}\t{:5d}\t{:15.4f}".format(left, right, fltr.sum(), churned[fltr].sum(), churned[fltr].sum()/fltr.sum()*100))
    sum_ += fltr.sum()
  print(f"\nsum(total_count): {sum_}")

# COMMAND ----------

sanity_check(prob_churn, churned, 10)

# COMMAND ----------


