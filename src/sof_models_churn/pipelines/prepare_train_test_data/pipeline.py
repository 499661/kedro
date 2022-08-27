from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_train_subset, create_train_tensors, create_test_subset,\
    create_test_tensors, get_week_ending_dates, create_regional_lookup_table

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                get_week_ending_dates,
                name="get_week_ending_dates",
                inputs=["dim_date@spark", "hhn_transact_yearweek@spark", "params:common"],
                outputs=["start_week_ending_date", "end_week_ending_date"]
            ),
            node(
                create_train_subset,
                name="create_train_subset",
                inputs=["hhn_transact_yearweek@spark", "params:common"],
                outputs="churn_train@spark"
            ),
            node(
                create_regional_lookup_table,
                name="create_regional_lookup_table",
                inputs=["churn_train@spark"],
                outputs="regional_lookup"
            ),
            node(
                create_train_tensors,
                name="create_train_tensors",
                inputs=["churn_train@spark", "regional_lookup", "start_week_ending_date",
                        "end_week_ending_date", "params:common"],
                outputs="train_data"
            ),
            node(
                create_test_subset,
                name="create_test_subset",
                inputs=["hhn_transact_yearweek@spark", "params:common"],
                outputs="churn_test@spark"
            ),
            node(
                create_test_tensors,
                name="create_test_tensors",
                inputs=["churn_test@spark", "regional_lookup", "start_week_ending_date",
                        "end_week_ending_date", "params:common"],
                outputs=["test_data", "churn_test_featurized@spark"]
            ),
    ]
)
