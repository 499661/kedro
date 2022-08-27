from ast import Continue
import logging
import os
import numpy as np
import tensorflow as tf
from asyncio.log import logger
from typing import Any, Dict, Iterable, Optional, Tuple
from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal
from kedro.config import TemplatedConfigLoader
from kedro.pipeline.node import Node
from kedro.framework.session import get_current_session
from delta.tables import *
from pyspark.sql import SparkSession
from datetime import datetime
from kedro.extras.datasets.spark import SparkDataSet
from sof_models_churn.common.date_utils import add_colon_to_utc_offset

import mlflow
from mlflow.tracking import MlflowClient
from sof_models_churn.common.mlflow_keras_customized import save_model, autolog
from sof_models_churn.features.ece_metric_scoring import ChurnPredictionMetrics
from sof_models_churn.pipelines.model_training.nodes import get_model_from_dict
import sof_models_churn.pipelines.prediction.validation as pdnv
import sof_models_churn.pipelines.data_engineering.validation as dev
from sof_models_churn.common.webhook_utils import model_training_reminder

mlflow.keras.save_model = save_model
mlflow.keras.autolog = autolog

logger = logging.getLogger(__name__)

class ProjectHooks:
    @hook_impl
    def register_config_loader(
        self, conf_paths: Iterable[str], env: str, extra_params: Dict[str, Any]
    ) -> ConfigLoader:
        return TemplatedConfigLoader(conf_paths,
            globals_pattern="*globals.yml" if env in ["dbconnect", "dev"] else None,
            globals_dict=None)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )

def split_node_name(node_name: str)-> Tuple[Any, str]:
    '''
    Split the full node name into namespace and name. It is assumed that the
    full name is of the form namespace.some_name .
    '''
    split = node_name.split(".")
    if len(split) == 2:
        return split[0], split[1]
    else:
        return None, split[0]

class ModelTrainingHooks:
    def __init__(self):
        self.mlflow_run_uuids = {}

    @hook_impl
    def before_node_run(self, node: Node, inputs: Dict) -> Dict:
        namespace, node_name = split_node_name(node.name)
        
        if namespace is not None and node_name == "pretrain_model":
            mlflow.keras.autolog(disable=True)

        if namespace is not None and node_name == "train_model":
            # start a new mlflow run and keep track of the run_uuid
            run = mlflow.start_run()
            self.mlflow_run_uuids[namespace] = run.info.run_uuid

            mlflow.log_params({
                k: inputs['params:common'][k] for k in ['end_date_train','n_weeks_total_train','batch_numbers_train',
                'test_frac','train_frac','n_validation_timesteps','n_testing_timesteps',
                'batch_size','epochs', "metrics_min_box_width", "consider_last_event","t_test_significance_level","t_test_n_folds"]
            })
            mlflow.log_params(
                {'end_week_ending_date': inputs['train_data']['end_week_ending_date'].strftime('%Y-%m-%d'), 
                 'start_week_ending_date': inputs['train_data']['start_week_ending_date'].strftime('%Y-%m-%d')})
            timestamp = datetime.now().astimezone() # defaults to local timezone
            mlflow.log_params({
                "churn_train_timestamp_as_of": add_colon_to_utc_offset(timestamp.strftime("%Y-%m-%dT%H:%M:%S%z"))
            })

            # log regional_lookup as artifact:
            inputs['train_data']['regional_lookup'].to_csv(os.path.join(inputs['params:common']['artifact_root'],'regional_lookup.csv'))
            mlflow.log_artifact(os.path.join(inputs['params:common']['artifact_root'],'regional_lookup.csv'),
                                'regional_lookup_dir' # this is the artifact directory relative to the run root directory)
            )

            mlflow.keras.autolog(log_models=False, skip_params='sample_weight')

            logger.info("mlflow run_uuid: " + str(run.info.run_uuid))

        if namespace is not None and node_name == "evaluate_model":
            # mlflow run has already been created by the node train_model
            run_uuid = self.mlflow_run_uuids[namespace]
            logger.info(f"restarting mlflow run with id {run_uuid}")
            run = mlflow.start_run(run_id=run_uuid)
            timestamp = datetime.now().astimezone() # defaults to local timezone
            mlflow.log_params({
                "churn_test_timestamp_as_of": add_colon_to_utc_offset(timestamp.strftime("%Y-%m-%dT%H:%M:%S%z"))
            })

    @hook_impl
    def after_node_run(self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict:
        namespace, node_name = split_node_name(node.name)

        if namespace is not None and node_name == "train_model":
             
            for key in outputs.keys():
                if key.endswith("trained_model"):

                    model = get_model_from_dict(outputs[key])
                    # Log model manually here as ``mlflow.keras.autolog`` does not expose the ``custom_objects`` kwarg. See https://www.mlflow.org/docs/latest/_modules/mlflow/keras.html#autolog
# comment out temporarily as some errors occur in this part related with mlflow keras customized.
                    mlflow.keras.log_model(model._model,
                                        artifact_path="model",
                                        save_weights_only=True,
                                        save_format="h5"
                            )
                    mlflow.log_params(model._params)

                elif key.endswith('lowest_idx'):
                    mlflow.log_metric("best_epoch__zero-indexed", outputs[key])

            logger.info("ending mlflow run")
            mlflow.end_run()

        if namespace is not None and node_name == "evaluate_model":

            for key in outputs.keys():
                if key.endswith("metric_dict"):
                    metric_dict = outputs[key]

            if len(metric_dict) == 0:
                logger.info(f"No metrics for {namespace}, possibly because the loss became NaN during training")
                mlflow.end_run()
                return

            box_widths = [k for k in metric_dict]
            aucs = [metric_dict[k]["auc"] for k in metric_dict]
            aps = [metric_dict[k]["ap"] for k in metric_dict]
            eces = [metric_dict[k]["ece"] for k in metric_dict]

            for i, bw in enumerate(box_widths):
                mlflow.log_metrics({'AUC_bw{}'.format(bw): aucs[i], 
                                    'AP_bw{}'.format(bw): aps[i], 
                                    'ECE_bw{}'.format(bw): eces[i], 
                                    'box_width_idx{}'.format(i): bw})

            mean_ece = np.mean(eces) 
            mean_auc = np.mean(aucs) 
            mean_ap = np.mean(aps) 
            mlflow.log_metric("mean_ece", mean_ece)
            mlflow.log_metric("mean_auc", mean_auc)
            mlflow.log_metric("mean_ap", mean_ap)

            ece_auc_path = os.path.join(inputs['params:common']['artifact_root'],'ece_auc_plot.png')
            ChurnPredictionMetrics.plot_metrics(metric_dict, save=True, name=ece_auc_path)
            mlflow.log_artifact(ece_auc_path, 'plots')
            
            roc_path = os.path.join(inputs["params:common"]["artifact_root"],"roc_curves_plot.png")
            ChurnPredictionMetrics.plot_roc_curves(metric_dict, save=True, name=roc_path)
            mlflow.log_artifact(roc_path, 'plots')

            pr_path = os.path.join(inputs["params:common"]["artifact_root"],"precision_recall_plot.png")
            ChurnPredictionMetrics.plot_pr_curves(metric_dict, save=True, name=pr_path)
            mlflow.log_artifact(pr_path, 'plots')

            logger.info("ending mlflow run")
            mlflow.end_run()
        
        if node_name =='prepare_prediction_features_and_predict':
            if inputs["params:common"]["environment"] in ["dev","stage","prod"]:
                catalog = get_current_session().load_context().catalog
                filepath = catalog.datasets.churn_train__spark._describe()['filepath']
                spark = SparkSession.builder.getOrCreate()
                # DeltaTable command is not supported on databricks-connect
                churn_train_ovr =  DeltaTable.forPath(spark, filepath)
                churn_train_last_operationDF = churn_train_ovr.history(1)
                last_training_time = churn_train_last_operationDF.collect()[0]['timestamp']
                now_time = datetime.now()
                time_passed = abs((now_time-last_training_time).days)
                
            
            ## it has been decided we should receive notification if there has been no model training for more than last 60 days in production env
                if time_passed >=60 :
                    mt_reminder_title = "Train new model reminder"
                    mt_reminder_content = "No new model has been trained for more than 60 days."
                    mt_action = "Please perform model training."
                    mt_reminder_color = "FFA500"
                    model_training_reminder(inputs["params:common"]["webhook_url"],mt_reminder_content, mt_reminder_title,mt_action,mt_reminder_color)
            else:
                Continue

            
            
    @hook_impl
    def on_pipeline_error( self)->None:
      mlflow.end_run()


class ModelValidatingHooks:

    @hook_impl
    def after_node_run(self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict:

        if node.name=='get_recent_challenger_id':
            
            logger.info("Registering challenger model; run_id='{}' ".format(outputs['challenger_id']))
            name = inputs["params:common"]['model_name']
            description = inputs["params:common"]['model_description']
            run_uri = "runs:/{}/model".format(outputs['challenger_id'])
            mv = mlflow.register_model(run_uri, name)
            logger.info("Name: {}".format(mv.name))
            logger.info("Version: {}".format(mv.version))

            client = MlflowClient()
            client.update_model_version(
                name=mv.name,
                version=mv.version,
                description=description
            )

            logger.info("Challenger model with run_id='{}' has registered. ".format(outputs['challenger_id']))   

        if node.name == "compare_score_and_transition":

            if outputs["comparison_results"]:
                if ~np.isnan(outputs["comparison_results"]['pvalue']):
                    client = MlflowClient()
                    # # log production model test result to challenger run
                    client.log_metric(run_id=inputs['challenger_id'], key="mean_auc_CHAMPION", value=np.mean(outputs["comparison_results"]['mean_auc_production']))
                    client.log_metric(run_id=inputs['challenger_id'], key="mean_auc_CHALLENGER", value=np.mean(outputs["comparison_results"]['mean_auc_challenger']))
                    client.log_metric(run_id=inputs['challenger_id'], key="t_test_pvalue", value=outputs["comparison_results"]['pvalue'])


class DataValidationHooks:

    @hook_impl
    def after_node_run(self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]) ->None:

        if node.name =='prepare_and_write_outputs':

            df = outputs["churn_output@spark"]
            output_table_name = "churn_output"
            n_weeks = inputs["params:common"]["n_weeks_total_pred"]
            n_weeks_churn = inputs["params:common"]["n_weeks_churn"]
            logger.info("Validating data for " + output_table_name + "in progress ...")

            pdnv.null_check(df,output_table_name)
            colstoDrop = ['prediction_date','max_week_ending_date','table_update_dt']
            output_flt_df = df.drop(*colstoDrop)
            pdnv.all_zeroes_check(output_flt_df, output_table_name)
            pdnv.churn_probability_value_check(df, output_table_name)
            pdnv.weeks_until_churned_check(df, output_table_name, n_weeks_churn)
            pdnv.alpha_beta_value_check(df, output_table_name)
            pdnv.wsls_wsfs_value_check(df, output_table_name, n_weeks)

        elif node.name == "create_cx_table":

            cx_transact_yearweek = outputs["cx_transact_yearweek@spark"]
            table = "cx_transact_yearweek"

            dev.count_check(cx_transact_yearweek, table)
            dev.null_check(cx_transact_yearweek, table)
            dev.weekly_sales_check(cx_transact_yearweek, table)
            dev.unit_sales_check(cx_transact_yearweek, table)
            dev.all_zeroes_check(cx_transact_yearweek, table)
            # The checks raise an Exception when they fail. If we get this far all of
            # them have passed.
            logger.info('Data validation check for ' + table + ' completes successfully.')

        elif node.name == "create_hh_table":

            hhn_transact_yearweek = outputs["hhn_transact_yearweek@spark"]
            table = "hhn_transact_yearweek"

            dev.count_check(hhn_transact_yearweek, table)
            dev.null_check(hhn_transact_yearweek, table)
            dev.weekly_sales_check(hhn_transact_yearweek, table)
            dev.unit_sales_check(hhn_transact_yearweek, table)
            hhn_flt_df = hhn_transact_yearweek\
                .drop('week_ending_date')
            dev.all_zeroes_check(hhn_flt_df, table)
            dev.avg_n_transaction_check(hhn_transact_yearweek, table)
            dev.avg_ecomm_transaction_check(hhn_transact_yearweek, table)
            dev.unit_offer_ratio_check(hhn_transact_yearweek, table)
            # The checks raise an Exception when they fail. If we get this far all of
            # them have passed.
            logger.info('Data validation check for ' + table + ' completes successfully.')

class TensorFlowHooks:

    @hook_impl
    def before_pipeline_run(self) -> None:
        tf.keras.utils.disable_interactive_logging()
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) == 0:
            logger.info("No GPU/TPU available.")
        else:
            for gpu in gpus:
                logger.info(f"Available physical device: {gpu.name}, {gpu.device_type}")

class TableHistoryHooks:

    @hook_impl
    def before_node_run(self, node: Node, inputs: Dict) -> Dict:
        keep_history_for_nodes = ["create_train_subset", "create_test_subset", "create_test_tensors"]
        spark = SparkSession.builder.getOrCreate()
        if (node.name in keep_history_for_nodes):
            spark.conf.set("spark.databricks.delta.properties.defaults.logRetentionDuration", "104 weeks")
        else:
            spark.conf.set("spark.databricks.delta.properties.defaults.logRetentionDuration", "30 days")

class DuplicateRowRemovalHooks:
    
    @hook_impl
    def after_node_run(self, node: Node, catalog, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        if node.name == "prepare_and_write_outputs":
            config = catalog.datasets.churn_output__spark._describe()
            df = outputs["churn_output@spark"]
            # The max_week_ending_date is the same for all the rows
            max_week_ending_date = df.select("max_week_ending_date").limit(1).collect()[0][0]

            config["save_args"]["replaceWhere"] = (
                "max_week_ending_date == '{:s}'".format(max_week_ending_date.strftime("%Y-%m-%d"))
            )
            logger.info(f"Replaced ReplaceWhere for churn_output@spark with {str(config['save_args']['replaceWhere'])}")
            dataset = SparkDataSet(config["filepath"],
                                   load_args=config["load_args"],
                                   save_args=config["save_args"],
                                   file_format=config["file_format"],
                                   version=config["version"])
            catalog.add("churn_output@spark", dataset, replace=True)

