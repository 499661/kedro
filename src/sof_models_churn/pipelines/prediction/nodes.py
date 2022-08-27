import os
import logging
import pandas as pd
import numpy as np
from typing import Any, Tuple, Dict
from datetime import date
from mlflow.tracking import MlflowClient
import pyspark.sql.functions as F
from pyspark.sql.types import ByteType, ShortType
from datetime import date, timedelta
from pyspark.sql import SparkSession, DataFrame
from sof_models_churn.common.mlflow_utils import get_model_version_number, get_prod_model
from sof_models_churn.features.rnn_training_data_prep import WtteRnnDataPrep
from sof_models_churn.common.date_utils import convert_end_date_to_python_date
from sof_models_churn.common.webhook_utils import notify_new_region

logger = logging.getLogger(__name__)

def get_regional_lookup_from_mlflow(params: dict) -> pd.DataFrame:

    client = MlflowClient()
    production_rnn = get_prod_model(params['model_name'],client=client,return_sequences=False)
    run_id = production_rnn.run_id

    #Download artifacts
    local_dir = "/tmp/artifact_downloads"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local_path = client.download_artifacts(run_id, "regional_lookup_dir", local_dir)
    logger.info('Artifacts downloaded in: {}'.format(local_path))
    logger.info('Artifacts: {}'.format(os.listdir(local_path)))

    regional_lookup_path = os.path.join(local_path, 'regional_lookup.csv')
    regional_lookup = pd.read_csv(regional_lookup_path)

    return regional_lookup

def get_week_ending_dates(dim_date : DataFrame, hhn_transact_yearweek: DataFrame, params: dict) -> Tuple[date, date]:
    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    start_week_ending_date, end_week_ending_date = Dataprep.get_week_ending_dates(dim_date, hhn_transact_yearweek,
                                                                                  end_date=params["end_date_pred"],
                                                                                  n_weeks_total=params["n_weeks_total_pred"])
    return start_week_ending_date, end_week_ending_date

def _create_subset_helper(hhn_transact_yearweek: DataFrame,
                          use_lower_half: bool,
                          params:dict) -> Tuple[np.ndarray, Dict[str, Any]]:

    split_frac = 0.5 if params["predict_frac"] is None else params["predict_frac"]/2.0
    split_lower_threshold = 0.0 if use_lower_half else split_frac
    data_prep_params = dict(
        split_frac = split_frac,
        split_lower_threshold = split_lower_threshold,
        end_date = params["end_date_pred"],
        n_weeks_total = params['n_weeks_total_pred'],
        batch_numbers = None
    )
    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    assert len(Dataprep.column_types) == params['n_features_total']
    subset = Dataprep.create_subset(hhn_transact_yearweek, **data_prep_params)
    return subset

def create_subset_1(hhn_transact_yearweek: DataFrame,
                    params:dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    subset_1 = _create_subset_helper(hhn_transact_yearweek,
                                     True,
                                     params)
    return subset_1

def create_subset_2(hhn_transact_yearweek: DataFrame,
                    params:dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    subset_2 = _create_subset_helper(hhn_transact_yearweek,
                                     False,
                                     params)
    return subset_2

def predict_on_subset(subset_1: DataFrame,
                      regional_lookup: pd.DataFrame,
                      start_week_ending_date: date,
                      end_week_ending_date: date,
                      params:dict) -> Tuple[np.ndarray, Dict[str, Any]]:

    data_prep_params = dict(
        prediction = True,
        regional_lookup = regional_lookup,
        default_region = params["default_region"],
        quantile_choice = None,
        validation_steps_cut_frac = None,
        n_testing_timesteps = None,
        to_keep = None,
        first_n_cols_to_rescale = 12,
        mask_value = 0.133713371337,
    )
    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    assert len(Dataprep.column_types) == params['n_features_total']

    logger.info("Generating features for prediction")
    pred_data = Dataprep.prepare_subset_for_model(subset_1,
                                                  start_week_ending_date,
                                                  end_week_ending_date,
                                                  **data_prep_params)

    #adding webhook to notify if new region is added
    if  Dataprep.regions_added:
        logger.info(f"New regions added: {Dataprep.regions_added}")
        region_title = "New regions added"
        region_content = f"{len(Dataprep.regions_added)} new regions has been added. Regions added are {Dataprep.regions_added}"
        region_action = "Please perform model training."
        region_color= "FFA500"
        notify_new_region(params["webhook_url"],region_content,region_title,region_action,region_color)   
    else:
        logger.info("No new region added")

    logger.info("Predicting...")
    client = MlflowClient()
    production_rnn = get_prod_model(params['model_name'],client=client,return_sequences=False)
    preds = production_rnn._model.predict([pred_data['x_features'],
                                           pred_data['x_store']])

    return preds, pred_data

def combine_predictions(preds_1: np.ndarray, pred_data_1: Dict[str, Any],
                        preds_2: np.ndarray, pred_data_2: Dict[str, Any],):
    preds = np.vstack((preds_1, preds_2))
    # Remove the x_features and x_stores arrays from pred_data
    hhn_1 = pred_data_1["lookup_hhn_id"]
    hhn_2 = pred_data_2["lookup_hhn_id"]
    hhn_2[:,0] += preds_1.shape[0]
    pred_data = {"lookup_hhn_id": np.vstack((hhn_1, hhn_2))}
    return preds, pred_data

def prepare_and_write_outputs(dim_date: DataFrame,
                              hhn_transact_yearweek: DataFrame,
                              preds: np.ndarray,
                              pred_data: Dict[str, Any],
                              params: Dict[str, Any]) -> DataFrame:

    client = MlflowClient()
    production_rnn = get_prod_model(params['model_name'],client=client,return_sequences=False)
    production_version= get_model_version_number(params['model_name'],production_rnn.run_id,
                                                client = client, stage = 'Production')

    pandas_df = pd.DataFrame(preds, columns=['alpha','beta'])
    pandas_df['hhn'] = pred_data['lookup_hhn_id'][:,-1]
    pandas_df['prediction_date']= date.today()
    end_date = params['end_date_pred']
    end_date = convert_end_date_to_python_date(end_date)
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(pandas_df)
    
    # Get the most recent 'week_ending_date' prior to 'end_date':
    max_week_ending_date = (dim_date.where(F.col("date") == end_date)
                                .select(F.date_add(F.col("week_end_date"), -7))
                                .toPandas().iloc[0,0])

    # Get each hhn's first+last shop, plus sales & transactions
    hhn_dates = hhn_transact_yearweek.where(
        # Q: Why the params["n_weeks_stats"] -1?
        # A: when n_weeks_stats = 1 the first week_ending_date and the last week_ending_date must be the same.
        F.col("week_ending_date").between(max_week_ending_date - timedelta(weeks=params["n_weeks_stats"] - 1),
                                          max_week_ending_date)
    ).groupBy("hhn").agg(
        F.lit(max_week_ending_date).alias("max_week_ending_date"),
        F.col('hhn'),
        F.max(F.col("week_ending_date")).alias("last_event"),
        F.min(F.col("week_ending_date")).alias("first_event"),
        F.sum(F.col("sales_dollars")).alias("historical_sales"),
        F.sum(F.col("n_transaction")).cast(ShortType()).alias("historical_transactions"),
        F.lit(params["n_weeks_stats"]).cast(ShortType()).alias("historical_kpis_num_weeks"),
    )

    # Merge
    df = df.join(hhn_dates, ['hhn'], 'inner')

    # NOTE: the opsGuru code has additional columns which are commented out
    df = df.withColumn("weeks_since_first_shop",
                        ( F.datediff(F.col("max_week_ending_date"),
                                     F.col("first_event")) / 7 ).cast(ShortType())
            ).withColumn("tenure_4wk_periods", # NOTE: this is not selected below
                        F.ceil((F.datediff(F.col("max_week_ending_date"),
                                           F.col("first_event")) / 7 + 1)/4.0).cast(ShortType())
            ).withColumn("weeks_since_last_shop",
                         ( F.datediff(F.col("max_week_ending_date"),
                                      F.col("last_event")) / 7 ).cast(ShortType())
            ).withColumn("weeks_until_churned",
                         ( F.greatest(F.lit(0), F.lit(params["n_weeks_churn"]) -\
                                      F.datediff(F.col("max_week_ending_date"),
                                                 F.col("last_event") ) / 7) ).cast(ByteType())
            ).select(F.col("prediction_date"),
                    F.col("max_week_ending_date"),
                    F.col("hhn"),
                    F.col("alpha"),
                    F.col("beta"),
                    F.exp(-F.pow(F.col("weeks_until_churned") / F.col("alpha"), F.col("beta")))\
                        .alias("churn_probability"),
                    F.lit(params["n_weeks_churn"]).cast(ByteType()).alias("churn_definition_weeks"),
                    F.col("weeks_until_churned"),
                    F.col("weeks_since_last_shop"),
                    F.col("weeks_since_first_shop"),
                    F.lit(production_version).alias("model_version"),
                    F.current_timestamp().alias("table_update_dt")
        )

    return df




    