import pandas as pd
import numpy as np
import pyspark.sql.functions as f
from typing import Tuple, Dict, Any
from pyspark.sql import DataFrame, SparkSession
from datetime import date
from sof_models_churn.features.rnn_training_data_prep import WtteRnnDataPrep

def create_regional_lookup_table(churn_train: DataFrame) -> pd.DataFrame:
    # In principle we could use hhn_transact_yearweek as the input, but churn_train is
    # a subset of hhn_transact_yearweek and therefore the call to distinct below is
    # faster.
    # TODO: figure out if using hhn_transact_yearweek is more robust. This might be
    # the case, because the subset churn_train may not include all of the regions.
    df = churn_train.select("divisional_regional_rollup").distinct().toPandas()
    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    regional_lookup =  Dataprep.create_regional_lookup(df)
    return regional_lookup

def get_week_ending_dates(dim_date : DataFrame, hhn_transact_yearweek: DataFrame, params: dict) -> Tuple[date, date]:
    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    start_week_ending_date, end_week_ending_date = Dataprep.get_week_ending_dates(dim_date, hhn_transact_yearweek,
                                                                                  end_date=params["end_date_train"],
                                                                                  n_weeks_total=params["n_weeks_total_train"])
    return start_week_ending_date, end_week_ending_date

def create_train_subset(hhn_transact_yearweek: DataFrame, params: dict) -> DataFrame:

    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    assert len(Dataprep.column_types) == params["n_features_total"]

    data_prep_params = dict(
        split_frac = params["train_frac"],
        split_lower_threshold = 0.,
        end_date = params["end_date_train"],
        n_weeks_total = params["n_weeks_total_train"],
        batch_numbers = params["batch_numbers_train"],
    )

    hhn_train_subset =  Dataprep.create_subset(hhn_transact_yearweek, **data_prep_params)
    return hhn_train_subset

def create_test_subset(hhn_transact_yearweek: DataFrame, params: dict) -> DataFrame:

    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    assert len(Dataprep.column_types) == params["n_features_total"]

    data_prep_params = dict(
        split_frac = params["test_frac"],
        split_lower_threshold = 1-params["test_frac"],
        end_date = params["end_date_train"],
        n_weeks_total = params["n_weeks_total_train"],
        batch_numbers = params["batch_numbers_train"],
    )

    hhn_test_subset =  Dataprep.create_subset(hhn_transact_yearweek, **data_prep_params)
    return hhn_test_subset

def create_train_tensors(hhn_train_subset: DataFrame, regional_lookup: pd.DataFrame, start_week_ending_date: date,
                         end_week_ending_date: date, params: dict) -> Dict[str, Any]:

    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    assert len(Dataprep.column_types) == params["n_features_total"]

    data_prep_params = dict(
        prediction = False,
        regional_lookup = regional_lookup,
        default_region = params["default_region"],
        quantile_choice = 0.6,
        n_testing_timesteps = params["n_validation_timesteps"],
        to_keep = params["n_validation_timesteps"],
        first_n_cols_to_rescale  = 12,
        mask_value = 0.133713371337,
    )

    train_data = Dataprep.prepare_subset_for_model(hhn_train_subset,
                                             start_week_ending_date,
                                             end_week_ending_date,
                                             **data_prep_params)
    return train_data

def create_test_tensors(hhn_test_subset: DataFrame, regional_lookup: pd.DataFrame, start_week_ending_date: date,
                        end_week_ending_date: date, params: dict) -> Tuple[Dict[str, Any], DataFrame]:

    spark = SparkSession.builder.getOrCreate()
    Dataprep = WtteRnnDataPrep(spark=spark)
    assert len(Dataprep.column_types) == params["n_features_total"]

    data_prep_params = dict(
        prediction = False,
        regional_lookup = regional_lookup,
        default_region = params["default_region"],
        quantile_choice = None,
        n_testing_timesteps = params["n_testing_timesteps"],
        to_keep = params["n_testing_timesteps"],
        first_n_cols_to_rescale  = 12,
        mask_value = 0.133713371337,
    )

    test_data = Dataprep.prepare_subset_for_model(hhn_test_subset,
                                             start_week_ending_date,
                                             end_week_ending_date,
                                             **data_prep_params)

    # Combine all three x/y tensors into one
    # (x_valid_features/y_valid are 3d; x_valid_store is 2d )
    z =  np.concatenate([test_data['x_valid_features'], 
                        np.expand_dims(test_data['x_valid_store'], axis=-1),
                        test_data['y_valid']
                        ], axis=-1)

    # Convert to single-col dataframe (length=num_households), whose entries are flattened (1d) tensors:
    d1, d2, d3 = z.shape
    z = z.reshape((-1, d2*d3))
    z = pd.DataFrame({'flattened_inputs': list(z)})
    test_set_df = spark.createDataFrame(z)

    return test_data, test_set_df