from kedro.pipeline import Pipeline, node
from .nodes import get_production_and_challenger_score, compare_score_and_transition,get_recent_challenger_id

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_recent_challenger_id,
                name="get_recent_challenger_id",
                inputs=["params:common","params:mlflow"],
                outputs="challenger_id"
            ),
            node(
                get_production_and_challenger_score,
                name="get_production_and_challenger_score",
                inputs=["params:common","churn_test_featurized@spark","challenger_id"],
                outputs=["mean_auc_production", "mean_auc_challenger"]
            ),
            node(
                compare_score_and_transition,
                name="compare_score_and_transition",
                inputs=["challenger_id", "mean_auc_production", "mean_auc_challenger","params:common"],
                outputs="comparison_results"
            )
        ]
    )