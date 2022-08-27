from typing import List
import logging
import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from sof_models_churn.common.mlflow_keras_customized import get_model_path
from sof_models_churn.features.rnn_model import RNNModel

logger = logging.getLogger(__name__)
def read_experiment_runs(experiment_id=None,
                         experiment_name=None, 
                         max_results=10000) -> List[Run]:
    client = MlflowClient()
    if experiment_id is not None:
        experiment = client.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = client.get_experiment_by_name(experiment_name)
    else:
        raise Exception(f"``experiment_id`` and ``experiment_name`` cannot both be None in function``read_experiment_runs()``")

    if experiment:
        experiment_id = experiment.experiment_id

        runs = [
            client.get_run(run_id=run_info.run_id)
            for run_info in client.list_run_infos(experiment_id=experiment_id, max_results=max_results)
        ]

        return runs
    else:
        raise Exception(f"Experiment {experiment_name} doesn't exist in tracking server {mlflow.get_registry_uri()}")



def get_model_version_number(model_name, run_id, client=None, stage='None'):
    """
    Looks at latest model versions from specified stage ('None', 'Staging', 'Production'), and returns the version number corresponding to run 'run_id'
    """

    if not client:
        client = MlflowClient()

    latest_versions = client.get_latest_versions(model_name, stages=[stage])
    version_data = list(filter(lambda x: x.run_id==run_id, latest_versions))
    if not version_data:
        raise Exception('Model version with run_id "{}" not found amongst latest versions (looking in stage="{}")'.format(run_id, 'stage'))
    else:
        return version_data[0].version

def get_prod_model(model_name, client=None, return_sequences=True):
    """
    Retrieves the "production" model weights & metadata from the model registry and instantiates the model.
    Also adds a 'run_id' attribute for convenience.

    Kwargs:
        client - optional (instance of mlflow.tracking.MlflowClient)
        return_sequences (bool): parameter for final LSTM layer of model. 
            True: each input time series gets mapped to time series (model output is 3d)
            False: each input time series gets mapped to a single output corresponding to the final timestep (model output is 2d)

    Returns:
        production_rnn (instance of ml_project.model.train.rnn_model.RNNModel class), with extra 'run_id' attribute (str)
    """
    
    if not client:
        client = MlflowClient()
    prod_model = client.get_latest_versions(model_name, stages=['Production'])

    if len(prod_model)>1:
        raise Exception('More than one model registered as "Production" model - add some logic for selecting one of them!')
    elif len(prod_model)==0:
        logger.info('Unable to find model with stage="Production"') 
        return None
    else:
        # Fetch production model
        prod_model = prod_model[0]

        production_run_id = prod_model.run_id
        production_rnn = get_model_from_run_id(run_id=production_run_id, client=client,return_sequences=return_sequences)

    return production_rnn

def get_model_from_run_id(run_id, client=None, return_sequences=True):
    """
    Retrieves the model weights & metadata from the model registry and instantiates the model.
    Also adds a 'run_id' attribute for convenience.

    Kwargs:
        run_id: Unique identifier for the run to get the model parameters and weights.
        client - optional (instance of mlflow.tracking.MlflowClient)
        return_sequences (bool): parameter for final LSTM layer of model. 
            True: each input time series gets mapped to time series (model output is 3d)
            False: each input time series gets mapped to a single output corresponding to the final timestep (model output is 2d)

    Returns:
        rnn_model (instance of ml_project.model.train.rnn_model.RNNModel class), with extra 'run_id' attribute (str)
    """
    
    if not client:
        client = MlflowClient()
    
    run = client.get_run(run_id)
    params = run.data.params
    artifact_uri = run.info.artifact_uri + '/model'
    model_path = get_model_path(artifact_uri)
    model_path = model_path + ".h5"

    def to_num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)

    hyperparams  = {p: to_num(params[p]) for p in ['lstm_1_units', 'lstm_2_units', 'lstm_final_units']}
    model_params = {p: to_num(params[p]) for p in ['div_reg_unique_count', 'init_alpha']}
    model_params['return_sequences'] = return_sequences

    # instantiate model and load weights
    rnn_model = RNNModel({**model_params, **hyperparams})
    # The following may complain that "A checkpoint was restored ... but not all checkpointed values were used." This is likely because we are not loading things like optimizer state, which may have been saved too [check]
    # See https://stackoverflow.com/a/59582698
    
    rnn_model._model.load_weights(model_path)

    rnn_model.run_id = run_id

    return rnn_model