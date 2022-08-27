import logging
#from turtle import title
from typing import Any, Dict
from pyspark.sql import DataFrame
import numpy as np
import time

from sklearn.model_selection import KFold
from scipy import stats

import mlflow
from mlflow.tracking import MlflowClient
from sof_models_churn.common.mlflow_utils import read_experiment_runs, get_model_version_number, get_prod_model, get_model_from_run_id
from sof_models_churn.features.ece_metric_scoring import ChurnPredictionMetrics
from sof_models_churn.common.mlflow_keras_customized import save_model, autolog
from sof_models_churn.common.webhook_utils import ttest_result

mlflow.keras.save_model = save_model
mlflow.keras.autolog = autolog

logger = logging.getLogger(__name__)

def get_recent_challenger_id(parameters:dict,mlflow_params:dict) -> str:
        """
        Look at the most recent runs and select the best based on AUC. 
        These are assumed to have been tested on our most recent test set.
        """

        def eligible_run(run):
            """
            Logic for selecting candidate set of runs here.
            To be valid, a run should have started within the last 24hrs AND have a logged, *non-null* value of the comparison metric.
            """
            condition_1 = ('mean_auc' in run.data.metrics)
            condition_2 = (time.time() - run.info.start_time/1e3  # run start_time is in milliseconds; time.time() is in seconds
                           )/(60*60) <= parameters['recent_runs_lookback_hrs']
            condition_3 = condition_1 and (~(np.isnan(run.data.metrics['mean_auc']))) # using short-circuiting property of "and" here

            if condition_1 and condition_2 and condition_3:
                return True
            else:
                return False

        # runs = read_experiment_runs(experiment_id="1") # may be convenient to use this when developing locally
        runs = read_experiment_runs(experiment_name=mlflow_params['mlflow_experiment']) # use this when developping on databricks
        logger.info('There are totally {} runs in the last {} hours'.format(len(runs),parameters['recent_runs_lookback_hrs']))
        valid_runs = filter(eligible_run, runs)
        sorted_runs = sorted(valid_runs, key=lambda x: x.data.metrics['mean_auc'], reverse=True)

        # Best run has highest auc:
        best_run_id = sorted_runs[0].info.run_id
        logger.info('Challenger run_id: {}'.format(best_run_id))

        return best_run_id

# challenger_run_id is needed by a hook
def get_production_and_challenger_score(parameters:dict,test_data_df: DataFrame, challenger_run_id:str ) -> Any:

    client=MlflowClient()
    model_name = parameters['model_name']
    logger.info('*** Fetching Production Model... ***')
    production_rnn = get_prod_model(model_name, client=client, return_sequences=True)

    if production_rnn is None:
        mean_auc_production=[]
        mean_auc_challenger=[]

    else:
        #fetch recent test dataset
        logger.info('*** Fetching test set... ***')
        arr = [ l[0] for l in test_data_df.toPandas().values ] # unpack nested array-list-array structure
        logger.info('list length: {}, shape of elements: {}'.format(len(arr),arr[0].shape))

        arr = np.stack(arr, axis=0)
        arr = arr.reshape(arr.shape[0], -1, parameters['n_features_total'] + 2 ) # "+2" because y tensor is appended here
        logger.info('Shape of test set array, after reshaping: {}'.format(arr.shape))
        x_valid_features = arr[:,:,:-3]
        x_valid_store = arr[:,:,-3]
        y = arr[:,:,-2:]
        logger.info('Final tensor shapes:\n x_valid_features:{}\n x_valid_store:{}\n y:{}'.format(x_valid_features.shape,x_valid_store.shape,y.shape))

        # fetch challenger model
        logger.info('*** Fetching Challenger Model... ***')
        challenger_rnn = get_model_from_run_id(run_id=challenger_run_id,client=client,return_sequences=True) # get the most recent model from 'None' stage.

        logger.info('*** Evaluating models... ***')

        kf = KFold(n_splits=parameters['t_test_n_folds'],random_state=101,shuffle=True)
        mean_auc_production=[]
        mean_auc_challenger=[]
        for _, test_index in kf.split(x_valid_features):
            x_valid_features_sub = x_valid_features[test_index]
            x_valid_store_sub = x_valid_store[test_index]
            y_sub = y[test_index]

            predicted_production = production_rnn._model.predict([x_valid_features_sub, x_valid_store_sub])
            predicted_challenger = challenger_rnn._model.predict([x_valid_features_sub, x_valid_store_sub])

            n_weeks = parameters['n_testing_timesteps']
            consider_last_event = parameters["consider_last_event"]
            min_box_width = parameters["metrics_min_box_width"]

            metric_dict_production = ChurnPredictionMetrics.evaluate_metrics(predicted_production, y_sub, n_steps=n_weeks,
                                                min_box_width=min_box_width,
                                                consider_last_event=consider_last_event)

            metric_dict_challenger = ChurnPredictionMetrics.evaluate_metrics(predicted_challenger, y_sub, n_steps=n_weeks,
                                                min_box_width=min_box_width,
                                                consider_last_event=consider_last_event)
            aucs_production = [metric_dict_production[k]["auc"] for k in metric_dict_production]
            mean_auc_production.append(np.mean(aucs_production)) 
            aucs_challenger = [metric_dict_challenger[k]["auc"] for k in metric_dict_challenger]
            mean_auc_challenger.append(np.mean(aucs_challenger)) 
            
        logger.info('Production model run_id: {}'.format(production_rnn.run_id))
        logger.info('Production model mean_auc 5 folds: {}'.format(mean_auc_production))
        logger.info('Challenger model run_id: {}'.format(challenger_rnn.run_id))
        logger.info('Challenger model mean_auc 5 folds: {}'.format(mean_auc_challenger))

    return mean_auc_production, mean_auc_challenger


def compare_score_and_transition(challenger_id:str, mean_auc_production:list,mean_auc_challenger:list, parameters:dict) -> Dict:

    client = MlflowClient()
    model_name = parameters['model_name']
    # get challenger data
    challenger_version = get_model_version_number(model_name, challenger_id, stage='None')
    if parameters["environment"]=="dev":
        challenger_version_uri ="https://adb-2990036548421646.6.azuredatabricks.net/?o=2990036548421646#mlflow/models/{model_name}/versions/{model_version}".format(model_name=model_name,
        model_version=challenger_version)
    if parameters["environment"]=="stage":
        challenger_version_uri = "https://adb-3386209605122527.7.azuredatabricks.net/?o=3386209605122527#mlflow/models/{model_name}/versions/{model_version}".format(model_name=model_name,
        model_version=challenger_version)
    if parameters["environment"]=="prod":
        challenger_version_uri= "https://adb-3762862571430522.2.azuredatabricks.net/?o=3762862571430522#mlflow/models/{model_name}/versions/{model_version}".format(model_name=model_name,
        model_version=challenger_version)

    if len(mean_auc_production)==0:
        logger.info('Challenger wins by default! Promoting "challenger" to production...')
        ### everytime transitioning, there will be a new version automatically. Is this supposed to be like this?
        client.transition_model_version_stage(name=model_name, version=challenger_version, 
                                                stage="Production", archive_existing_versions=True)
        return {}

    else:
        results = stats.ttest_rel(mean_auc_production, mean_auc_challenger, alternative='less')
        pvalue = results.pvalue

        logger.info('Paired t Test results: \n statistic: {}, pvalue: {}'.format(results.statistic, pvalue))

        if pvalue<=parameters['t_test_significance_level']:

            logger.info('pvalue is less than {} and considered to be statistically significant, so we accept the alternative hypothesis Production model is worse than Challenger model'.format(parameters['t_test_significance_level']))
        
            logger.info('Challenger wins Over existing Production model! Promoting "challenger" to staging...')
            client.transition_model_version_stage(name=model_name, version=challenger_version, 
                                                    stage="Staging")

            client.set_model_version_tag(name=model_name,version= challenger_version, key= "paired ttest", value= "passed")                                        

            if parameters["environment"] in ["stage","prod"]:
                notif_title = "Alert - Challenger model wins over prod model in paired t-test"
                notif_content= f"""<b>Production model </b> is found to be performing <b>worse </b> than the <b>challenger model </b>. Transitioned challenger model : \
                    {model_name} with model_version :{challenger_version} to Staging."""
                notif_action = "Transition <b>challenger </b> model to <b>production </b>"
                color ="FF0000"
                logger.info("Sending webhook notification for promoting models to production")
                ttest_result(parameters['webhook_url'],notif_content,notif_title,challenger_version_uri,notif_action,color)


            # Automate promotion to Prod for end-to-end testing in Development:
            if parameters["environment"] in ["dev", "dbconnect"]:
                logger.info('Challenger wins Over existing Production model! Promoting "challenger" to production...')
                client.transition_model_version_stage(name=parameters['model_name'], version=challenger_version, 
                                                        stage="Production", archive_existing_versions=True)
        else:
            if parameters["environment"] in ["stage","prod"]:
                logger.info('pvalue is greater than {} and not statistically significant, so we reject the alternative hypothesis Production model is worse than Challenger model'.format(parameters['t_test_significance_level']))
                client.transition_model_version_stage(name=parameters['model_name'], version=challenger_version, 
                                                    stage="Archived")
                client.set_model_version_tag(name=model_name,version= challenger_version, key= "paired ttest", value= "did not pass")
                notif_title= "Production model wins over challenger model"
                notif_content = f"""<b>Challenger model </b> is found to be performing worse than the <b>production model </b>. Transitioned challenger model : \
                    {model_name} with model_version :{challenger_version} to <b>Archived </b>"""
                notif_action = "No action needed."
                color = "00FF00"
                ttest_result(parameters['webhook_url'],notif_content,notif_title,challenger_version_uri,notif_action,color)

            

    return {'mean_auc_production':mean_auc_production, 'mean_auc_challenger':mean_auc_challenger, 'pvalue':pvalue}