from kedro.pipeline import Pipeline, node
from .nodes import get_regional_lookup_from_mlflow, get_week_ending_dates, create_subset_1,\
    create_subset_2, predict_on_subset, combine_predictions, prepare_and_write_outputs


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_regional_lookup_from_mlflow,
                name="get_regional_lookup_from_mlflow",
                inputs=["params:common"],
                outputs="regional_lookup_mlflow"
            ),
            node(
                get_week_ending_dates,
                name="get_week_ending_dates",
                inputs=["dim_date@spark", "hhn_transact_yearweek@spark", "params:common"],
                outputs=["start_week_ending_date", "end_week_ending_date"]
            ),
            node(
                create_subset_1,
                name="create_subset_1",
                inputs=["hhn_transact_yearweek@spark", "params:common"],
                outputs="subset_1"
            ),
            node(
                create_subset_2,
                name="create_subset_2",
                inputs=["hhn_transact_yearweek@spark", "params:common"],
                outputs="subset_2"
            ),
            node(
                predict_on_subset,
                name="predict_on_subset_1",
                inputs=["subset_1", "regional_lookup_mlflow", "start_week_ending_date",
                          "end_week_ending_date", "params:common"],
                outputs=["preds_1","pred_data_1"]
            ),
            node(
                predict_on_subset,
                name="predict_on_subset_2",
                inputs=["subset_2", "regional_lookup_mlflow", "start_week_ending_date",
                          "end_week_ending_date", "params:common"],
                outputs=["preds_2","pred_data_2"]
            ),
            node(
                combine_predictions,
                name="combine_predictions",
                inputs=["preds_1","pred_data_1", "preds_2","pred_data_2"],
                outputs=["preds_combined", "pred_data_combined"]
            ),
            node(
                prepare_and_write_outputs,
                name="prepare_and_write_outputs",
                inputs=["dim_date@spark", "hhn_transact_yearweek@spark",
                        "preds_combined","pred_data_combined", "params:common"],
                outputs="churn_output@spark"
            )
        ])
