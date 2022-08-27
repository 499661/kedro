# Databricks notebook source
# MAGIC %md
# MAGIC # Install dependencies

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC %matplotlib inline

# COMMAND ----------

!pip install -r ../src/requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC # Model
# MAGIC 
# MAGIC We are loading the production model.

# COMMAND ----------

import sys
sys.path.append("../src")

# COMMAND ----------

from sof_models_churn.features.rnn_model import RNNModel

# COMMAND ----------

model_params = {"div_reg_unique_count": 30, # 30 appears to be the standard 
                "init_alpha": 1.0 # probably does not matter
               }
# Hyperparams can be read off from /dbfs/mnt/sofdlzonelab_data-science-dev/users/rajat_paliwal@saveonfoods.com/churn_models_prod/churn_6/model_summary.txt
hyperparams = {"lstm_1_units": 10,
               "lstm_2_units": 30,
               "lstm_final_units": 5}
prod_model = RNNModel({**model_params, **hyperparams})
prod_model._model.load_weights("/dbfs/mnt/sofdlzonelab_data-science-dev/users/rajat_paliwal@saveonfoods.com/churn_models_prod/churn_6/model/data/model.h5")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data
# MAGIC 
# MAGIC At the moment there is an issue with the dataframes `churn_test` and `churn_train` from the delta lake because they are saved using the append mode.

# COMMAND ----------

from typing import Tuple
import numpy as np
from pyspark.sql import DataFrame

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

# Specify version here for reproducibility?
churn_test_featurized = (
  spark.read.format("delta")
            .option("versionAsOf", 12)
            .load("/mnt/sofdlzonelab_data-science-dev/users/mirko_moeller@saveonfoods.com/source/data_science/customer_churn/input/churn_test_featurized")
)

# COMMAND ----------

x_valid_features, x_valid_store, y = convert_df_to_arr(churn_test_featurized)
print(f"x_valid_features: {x_valid_features.shape}")
print(f"x_valid_store: {x_valid_store.shape}")
print(f"y: {y.shape}")

# COMMAND ----------

predictions = prod_model._model.predict([x_valid_features, x_valid_store])
print(f"predictions {predictions.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate some metrics

# COMMAND ----------

from sof_models_churn.features.ece_metric_scoring import ChurnPredictionMetrics

# COMMAND ----------

def print_info(metrics_dict):
  for bw in metrics_dict:
    tmp = metrics_dict[bw]
    print("box width\t{:02d}\tauc:{:.4f}\tap:{:.4f}\tECE:{:.4f}\t(%in_box:{:.4f}\t obs {:d})".\
                        format(bw, tmp["auc"], tmp["ap"], tmp["ece"], tmp["frac_in_box"],
                               tmp["observations"]))

# COMMAND ----------

metrics_1 = ChurnPredictionMetrics.evaluate_metrics(predictions, y, 13, min_box_width=0, consider_last_event=False) 

# COMMAND ----------

print_info(metrics_1)

# COMMAND ----------

metrics_2 = ChurnPredictionMetrics.evaluate_metrics(predictions, y, 13, min_box_width=0, consider_last_event=True)

# COMMAND ----------

print_info(metrics_2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Look at time to last event

# COMMAND ----------

from sof_models_churn.common import wtte_utils as wtte

# COMMAND ----------

predicted_nan, y_nan = ChurnPredictionMetrics.replace_dummy_value_with_nans(predictions, y)
predicted_left = wtte.right_pad_to_left_pad(predicted_nan)
y_left = wtte.right_pad_to_left_pad(y_nan)

# COMMAND ----------

def print_tte_for_row(row, cutoff):
  t_since_last_event, mask_t = ChurnPredictionMetrics.calc_time_since_last_event(y_left[:,:,0], cutoff)
  print(f"tte array:\t{y_left[row,:cutoff+1,0]}")
  print(f"time since last event:\t{t_since_last_event[row]},\t\tvalid: {mask_t[row]}")

# COMMAND ----------

print_tte_for_row(0, 10)
print()
for i in range(10):
  print_tte_for_row(i, 12)
  print()

# COMMAND ----------

# MAGIC %md
# MAGIC # Look at behavior of wtte.weibull_cmf
# MAGIC 
# MAGIC The result returned by weibull_cmf is different when t is a vector instead of a scalar. This might be due to floating point errors.

# COMMAND ----------

from sof_models_churn.common.wtte_weibull import cmf as weibull_cmf
import matplotlib.pyplot as plt
import sklearn.metrics

# COMMAND ----------

box_width = 10
testset_begin = 12

alpha = predicted_left[:,testset_begin,0]
beta = predicted_left[:,testset_begin,1]
m = ~np.isnan(alpha + beta)
prob_1 = weibull_cmf(a=alpha[m], b=beta[m], t=box_width)
# prob_2 = weibull_cmf(a=alpha[m], b=beta[m], t=box_width - np.zeros(alpha[m].shape[0]))
prob_2 = weibull_cmf(a=alpha[m], b=beta[m], t=np.full(m.sum(), box_width, dtype="float32"))

fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
ax.plot(prob_1 - prob_2)

ax.set_xlabel("Index")
ax.set_ylabel("prob_1 - prob_2")

is_in_box = (y_left[:,testset_begin,0] <= box_width)[m].flatten()
auc_1 = sklearn.metrics.roc_auc_score(~is_in_box, 1.0 - prob_1)
auc_2= sklearn.metrics.roc_auc_score(~is_in_box, 1.0 - prob_2)
print(f"auc_1: {auc_1}\tauc_2: {auc_2}\t difference: {auc_1 - auc_2}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Under The hood weibull_cmf calls cdf which is given by
# MAGIC ```python
# MAGIC def cdf(t, a, b):
# MAGIC     """ Cumulative distribution function.
# MAGIC     :param t: Value
# MAGIC     :param a: Alpha
# MAGIC     :param b: Beta
# MAGIC     :return: `1 - np.exp(-np.power(t / a, b))`
# MAGIC     """
# MAGIC     t = np.double(t)
# MAGIC     return 1 - np.exp(-np.power(t / a, b))
# MAGIC ```
# MAGIC 
# MAGIC Note the call to `np.double()` which converts to `float64`. Below I am showing how arrays and scalars of different type interact in numpy. That is at the root of the discrepancy.

# COMMAND ----------

print(f"np.double(1.0).dtype:\t\t\t{np.double(1.0).dtype}")
print(f"alpha.dtype:\t\t\t\t{alpha.dtype}")
print(f"(np.double(1.0)/alpha[m]).dtype:\t{(np.double(1.0)/alpha[m]).dtype}")
print(f"np.full(m.sum(), 1.0).dtype:\t\t{np.full(m.sum(), 1.0).dtype}")
print(f"(np.full(m.sum(), 1.0)/alpha[m]).dtype:\t{(np.full(m.sum(), 1.0)/alpha[m]).dtype}")

# COMMAND ----------

# The discrepancy also exists when we only use simple numpy functions

prob_1 = np.double(1.0)/alpha[m] # dtype is determined by alpha since np.double(1.0) is not an array
prob_2 = np.full(m.sum(), 1.0)/alpha[m] # dtype is float64

fig, ax = plt.subplots(1,1)
fig.set_tight_layout(True)
ax.plot(prob_1 - prob_2)

ax.set_xlabel("Index")
ax.set_ylabel("prob_1 - prob_2")

print(f"prob_1: {prob_1.dtype}\t\tprob_2: {prob_2.dtype}")

# COMMAND ----------


