"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from sof_models_churn.pipelines import data_engineering as de
from sof_models_churn.pipelines import prediction as pdn
from sof_models_churn.pipelines import model_training as mt
from sof_models_churn.pipelines import model_validating as mv
from sof_models_churn.pipelines import prepare_train_test_data as ptt
from kedro.pipeline.modular_pipeline import pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_engineering_pipeline = de.create_pipeline()
    prediction_pipeline = pdn.create_pipeline()
    model_training_template_pipeline = mt.create_pipeline()
    model_validating_pipeline = mv.create_pipeline()
    prepare_train_test_pipeline = ptt.create_pipeline()
    model_1_pipeline = pipeline(pipe=model_training_template_pipeline,
                                inputs={"train_data", "test_data"},
                                parameters={"params:override_me": "params:hyperparams_search.hyperparams_1"},
                                namespace="model_1")
    model_2_pipeline = pipeline(pipe=model_training_template_pipeline,
                                inputs={"train_data", "test_data"},
                                parameters={"params:override_me": "params:hyperparams_search.hyperparams_2"},
                                namespace="model_2")
    model_3_pipeline = pipeline(pipe=model_training_template_pipeline,
                                inputs={"train_data", "test_data"},
                                parameters={"params:override_me": "params:hyperparams_search.hyperparams_3"},
                                namespace="model_3")
    model_training_pipeline = model_1_pipeline + model_2_pipeline + model_3_pipeline
    return {
        "de": data_engineering_pipeline,
        "mt": model_training_pipeline, 
        "mv": model_validating_pipeline, 
        "pdn": prediction_pipeline,
        "ptt": prepare_train_test_pipeline
    }
