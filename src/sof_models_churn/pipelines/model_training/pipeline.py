from kedro.pipeline import Pipeline, node
from .nodes import train_model, evaluate_model, create_model, pretrain_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                create_model,
                name="create_model",
                inputs=["train_data", "params:override_me", "params:common"],
                outputs="model",
            ),
            node(
                pretrain_model,
                name="pretrain_model",
                inputs=["model","train_data"],
                outputs="pretrained_model"
            ),
            node(
                train_model,
                name="train_model",
                inputs=["pretrained_model", "train_data", "params:common"],
                outputs=["trained_model", "lowest_idx"]
            ),
            node(
                evaluate_model,
                name="evaluate_model",
                inputs=["trained_model", "test_data", "params:common"],
                outputs="metric_dict"
            ),
        ]
    )