# Databricks notebook source
# MAGIC %md
# MAGIC ### This Notebook creates household_aggregates and household_churn table in Gold Layer

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql import Window
from delta.tables import *
import pandas as pd
import json

# COMMAND ----------

spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite", True)
spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.autoCompact", True)

# COMMAND ----------

## setting job params as widgets for easy notebook debugging
dbutils.widgets.text("Environment", "", "")
dbutils.widgets.text("end_date_pred","","")
Environment = dbutils.widgets.get("Environment").lower()
end_date_pred = dbutils.widgets.get("end_date_pred")

# COMMAND ----------

# DBTITLE 1,Check if Environment is set
if not Environment:
  raise Exception("Environment - Mandatory parameter is not passed")
  
if Environment != 'dev' and Environment != 'stage' and Environment != 'prod':
  raise Exception(f"Invalid Environment : {Environment}. Valid values are dev or stage or prod")

# COMMAND ----------

# DBTITLE 1,Check if end_date_pred is valid
if pd.to_datetime(end_date_pred).date() > pd.datetime.now().date():
  raise Exception(f"Invalid date : {end_date_pred}. Date cannot be in future ")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Mount Path
# MAGIC * Initiate to set Silver and Gold destination ADLS paths

# COMMAND ----------

config = open("../conf/config_standalone_notebooks.json")
settings = json.load(config)

householdAnlzdSalesPath = settings[Environment]['GoldMountPath'] + "/source/customer/agg/household_annualized_sales"
householdChurnPath = settings[Environment]['GoldMountPath'] + "/source/customer/fact/household_churn"

retailSalesFactPath = settings[Environment]['GoldReadMountPath'] + "/source/sales/fact/retail_sale_fact/"
itemDimPath =  settings[Environment]['GoldReadMountPath'] + "/source/master/dim/item_dim/"
locationDimPath =   settings[Environment]['GoldReadMountPath'] + "/source/master/dim/location_dim/"
dateDimPath =  settings[Environment]['GoldReadMountPath'] + "/source/master/dim/date_dim"
customerDimPath =  settings[Environment]['GoldReadMountPath'] + "/source/master/dim/customer_dim"
churnOutputPath = settings[Environment]['SilverMountPath'] + "/source/data_science/customer_churn/output/churn_output/"

# COMMAND ----------

rsDF = spark.read.format("delta").load(retailSalesFactPath)
itemDF = spark.read.format("delta").load(itemDimPath)
locationDF = spark.read.format("delta").load(locationDimPath)
dateDF = spark.read.format("delta").load(dateDimPath)
customerDF = spark.read.format("delta").load(customerDimPath)
churnOutputDF = spark.read.format("delta").load(churnOutputPath)

# COMMAND ----------

n_years_lookback = 2

period_length_weeks = 13 # each household's sales are aggregated/"resampled" at this frequency before averaging their sales over time

windowCol = Window.partitionBy('hhn', 'max_week_ending_date', 'prediction_date')

churnOutputFltDF = churnOutputDF\
              .withColumn('max_table_update_dt', f.max('table_update_dt')\
                          .over(windowCol))\
              .where(f.col('table_update_dt') == f.col('max_table_update_dt'))\
              .drop('max_table_update_dt')

if end_date_pred == "":
  max_wed = churnOutputFltDF.select(f.max('max_week_ending_date').alias('max_wed').cast("date")).collect()[0][0]
else:
  max_wed = dateDF.where(f.col("date") == pd.to_datetime(end_date_pred).date())\
                                .select(f.date_add(f.col("week_end_date"), -7))\
                                .collect()[0][0]
baseDateDF = dateDF\
                .filter(f.col('date').between(f.lit(max_wed) - (7*52*n_years_lookback - 1), f.lit(max_wed)))\
                .join(rsDF, dateDF.date == rsDF.fiscal_date)\
                .join(customerDF, rsDF.customer_hk == customerDF.customer_hk)\
                .groupBy('current_household_id')\
                .agg(f.date_sub(f.min('week_end_date'), 7).alias('first_event_trunc'))\
                .withColumn('weeks_since_first_shop_trunc', (f.datediff(f.lit(max_wed), f.col('first_event_trunc'))/7).cast("integer"))\
                .withColumn('period_length_weeks', f.lit(period_length_weeks))\
                .withColumn('max_wed', f.lit(max_wed))\
                .withColumn('n_periods_tenure', 
                            f.ceil((f.col('weeks_since_first_shop_trunc') + 1)/(f.col('period_length_weeks')\
                                   .cast("double")).cast("integer")))\
                .select(f.col('current_household_id').alias('hhn'), 'max_wed' ,'first_event_trunc', 
                        'weeks_since_first_shop_trunc', 'period_length_weeks', 'n_periods_tenure')


# COMMAND ----------

itemFltDF = itemDF\
              .filter(f.col('financial_sale_flag') == 'Y')

custFltDF = customerDF\
                .filter((~f.col('current_card_type').isin('Manager Cards', 'Training Cards')) | (f.col('current_card_type').isNull()))


annualizedSalesDF = rsDF\
                      .join(custFltDF, rsDF.customer_hk == custFltDF.customer_hk)\
                      .join(baseDateDF, custFltDF.current_household_id == baseDateDF.hhn)\
                      .join(itemFltDF, rsDF.item_hk == itemFltDF.item_hk)\
                      .filter((baseDateDF.hhn != 0) &
                             (rsDF.fiscal_date > baseDateDF.first_event_trunc) &
                             (rsDF.fiscal_date <= baseDateDF.max_wed))\
                      .groupBy(custFltDF.current_household_id, baseDateDF.n_periods_tenure, baseDateDF.period_length_weeks, baseDateDF.max_wed)\
                      .agg((f.round(f.coalesce(f.sum(rsDF.merch_sales) / baseDateDF.n_periods_tenure * (52/baseDateDF.period_length_weeks), f.lit(0))/1, 0)*1).alias('annualized_sales'))\
                      .select(custFltDF.current_household_id.alias('hhn'), 'annualized_sales', baseDateDF.max_wed.alias('churn_max_week_ending_date'))

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, householdAnlzdSalesPath)):
  householdAnlzdSalesTable = DeltaTable.forPath(spark, householdAnlzdSalesPath)
  
  householdAnlzdSalesTable.alias("target")\
                      .merge(annualizedSalesDF.alias("source"),
                           "source.hhn = target.hhn AND source.churn_max_week_ending_date = target.churn_max_week_ending_date")\
                      .whenMatchedUpdate(condition = "source.annualized_sales != target.annualized_sales", set = {
                                    "annualized_sales" : "source.annualized_sales" })\
                      .whenNotMatchedInsertAll() \
                      .execute()

else:
  annualizedSalesDF.write.mode("overwrite")\
                    .format("delta")\
                    .partitionBy("churn_max_week_ending_date") \
                    .save(householdAnlzdSalesPath)

# COMMAND ----------

householdAnlzdSalesDF = spark.read.format("delta").load(householdAnlzdSalesPath)

householdChurnDF = householdAnlzdSalesDF\
                    .join(churnOutputFltDF, (householdAnlzdSalesDF.hhn == churnOutputFltDF.hhn) 
                          & (householdAnlzdSalesDF.churn_max_week_ending_date == churnOutputFltDF.max_week_ending_date)
                           & (householdAnlzdSalesDF.hhn != 0))\
                    .withColumn('annualized_sales', f.round(f.col('annualized_sales')))\
                    .withColumn('next_week_shop_prob', f.round(1 - f.exp(-pow(1 / churnOutputFltDF.alpha, churnOutputFltDF.beta)),4))\
                    .withColumn('churn_probability', f.round(churnOutputFltDF.churn_probability, 4))\
                    .withColumn('weeks_since_last_shop', churnOutputFltDF.weeks_since_last_shop)\
                    .withColumn('churn_segment', 
                                  f.when(f.col('churn_probability')==f.lit(1), f.lit('Churned'))
                                  .when(f.col('churn_probability')>=f.lit(0.75), f.lit('At Risk'))
                                  .when(f.col('churn_probability').between(0.5, 0.75), f.lit('High'))
                                  .when(f.col('churn_probability').between(0.25, 0.5), f.lit('Med'))
                                  .otherwise(f.lit('Low')))\
                    .select(householdAnlzdSalesDF.churn_max_week_ending_date.alias('max_week_ending_date'), householdAnlzdSalesDF.hhn, 'churn_probability', 'weeks_since_last_shop', 
                            'next_week_shop_prob', 'annualized_sales', 'churn_segment', 'prediction_date')\
                    .distinct()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automated Test Cases before saving the table

# COMMAND ----------

# DBTITLE 1,Schema Check
## Test 1: verify columns present in the table and ensure all the expected columns should be present
expectedTrgColumns = ["max_week_ending_date", "hhn", "churn_probability", "weeks_since_last_shop", "next_week_shop_prob", "annualized_sales", "churn_segment", "prediction_date"]

trgColumns = householdChurnDF.columns

if trgColumns == expectedTrgColumns:
  resultDF1 = 'Pass'
else:
  resultDF1='Schema Check Fail'

print(resultDF1)

# COMMAND ----------

# DBTITLE 1,Null Check
## Test 2: Checking for Nulls in household_churn table
nullCounts = householdChurnDF\
        .select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in householdChurnDF.columns])\
        .collect()[0]\
        .asDict()
    
if (all(value == 0 for value in nullCounts.values())) == True:
        resultDF2 = 'Pass'
else:
  for key, value in nullCounts.items():
    if (value > 0):
      resultDF2='Null Check Fail'
      
print(resultDF2)

# COMMAND ----------

# DBTITLE 1,Churn Probability Check
## Test 3: probability should be between 0 and 1
churnProbabilityCondition= householdChurnDF\
        .where((f.col('churn_probability') < 0) & (f.col('churn_probability') > 1))

if churnProbabilityCondition.count() > 0:
  resultDF3='Churn Probability Check Fail'
else:
  resultDF3 = 'Pass'

print(resultDF3)

# COMMAND ----------

# DBTITLE 1,Weekly Count of HHNs Check

## Check weekly count of HHNs should not be below the Threshold Value of 2.5M
hhnWeeklyDF = householdChurnDF.groupBy('max_week_ending_date').count()
  
threshold = 2500000

## Data in dev is expected to be less, setting threshold to 1000 for dev
if(Environment == 'dev'):
  threshold = 1000

testConditionDF = hhnWeeklyDF\
    .filter(f.col('count') < threshold)

if testConditionDF.count() > 0:
  ## check if the count is less due to Chistmas Week  
  dateFilterDF = dateDF\
                      .filter(f.date_format("date", "MM-dd") == '12-25').select('date', 'week_end_date')

  filteredCountDF = testConditionDF\
                      .join(dateFilterDF, (testConditionDF.max_week_ending_date == dateFilterDF.week_end_date))\
                      .select('week_end_date')

  testConditionFltDF = testConditionDF.join(filteredCountDF, (testConditionDF.max_week_ending_date == filteredCountDF.week_end_date), "leftanti")

  ## if count is low due to Christmas Week, pass the test 
  if testConditionFltDF.count()>0:
    resultDF4='Threshold Check Fail'
  else:
      resultDF4='Pass'

else:
    resultDF4='Pass'
  
print(resultDF4)

# COMMAND ----------

#job should be failed upon any automated tests fail.
if resultDF1=='Pass' and resultDF2=='Pass' and resultDF3=='Pass' and resultDF4=='Pass':
  print('Success')
else:
  raise Exception ('Unit test validations of Gold Layer Household Churn Table Failed')

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, householdChurnPath)):
  householdChurnTable = DeltaTable.forPath(spark, householdChurnPath)
  
  householdChurnTable.alias("target")\
                      .merge(householdChurnDF.alias("source"),
                           "source.hhn = target.hhn AND source.max_week_ending_date = target.max_week_ending_date")\
                      .whenMatchedUpdate(condition = """source.annualized_sales != target.annualized_sales OR 
                                                        source.next_week_shop_prob != target.next_week_shop_prob OR
                                                        source.churn_probability != target.churn_probability OR
                                                        source.churn_segment != target.churn_segment OR
                                                        source.prediction_date != target.prediction_date""" , 
                                         set = {"annualized_sales" : "source.annualized_sales",
                                               "next_week_shop_prob" :"source.next_week_shop_prob", 
                                               "churn_probability" : "source.churn_probability", 
                                               "churn_segment" : "source.churn_segment",
                                               "prediction_date" : "source.prediction_date"})\
                      .whenNotMatchedInsertAll() \
                      .execute()

else:
  householdChurnDF.write.mode("overwrite")\
                    .format("delta")\
                    .partitionBy("max_week_ending_date")\
                    .save(householdChurnPath)

# COMMAND ----------

import sys
import requests
sys.path.append("../src")
from src.sof_models_churn.common.webhook_utils import pipeline_completion_notification
# adding a temporary webhook to get notified once this job finishes.Once this job will be run using Control-M we will remove this notification

if Environment == 'dev':
  webhook_url = 'https://owfg.webhook.office.com/webhookb2/08a37fb4-e893-4e4f-a8e8-589152fc7bbf@fdb969dd-87c5-4a41-87d6-86f80f4581db/IncomingWebhook/f5756e8ff49c41afadad25e95b6466f8/afe2f6a7-2d13-4f5b-b0fd-a93043a90512'
if Environment == 'stage':
  webhook_url = 'https://owfg.webhook.office.com/webhookb2/08a37fb4-e893-4e4f-a8e8-589152fc7bbf@fdb969dd-87c5-4a41-87d6-86f80f4581db/IncomingWebhook/4d47ccd74e054081895fea08c278cb3c/afe2f6a7-2d13-4f5b-b0fd-a93043a90512'
if Environment == 'prod':
  webhook_url = 'https://owfg.webhook.office.com/webhookb2/08a37fb4-e893-4e4f-a8e8-589152fc7bbf@fdb969dd-87c5-4a41-87d6-86f80f4581db/IncomingWebhook/c4ad2c7ad81f4e0ba1a6db9cf433b64b/afe2f6a7-2d13-4f5b-b0fd-a93043a90512'
title = "Churn pipeline has successfully completed the run."
action= "No action required"
content = " Pipeline completed successfully."
color = "00FF00"
pipeline_completion_notification(webhook_url,content,title,action, color)
