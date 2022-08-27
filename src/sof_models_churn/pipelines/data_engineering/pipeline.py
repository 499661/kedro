from kedro.pipeline import Pipeline, node
from .nodes import get_min_max_dates, create_cx_table, create_hh_table

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_min_max_dates,
                name="get_min_max_dates",
                inputs=["dim_date@spark", "params:common"],
                outputs="min_max_dates_df",
            ),
            node(
                create_cx_table,
                name="create_cx_table",
                inputs=["dim_location@spark", "dim_product@spark",
                "dim_date@spark", "retail_sale@spark",
                "min_max_dates_df"],
                outputs="cx_transact_yearweek@spark",
            ),
            node(
                create_hh_table,
                name="create_hh_table",
                inputs=["dim_location@spark", "dim_customer@spark", "dim_date@spark",
                    "min_max_dates_df", "cx_transact_yearweek@spark", "params:common"],
                outputs="hhn_transact_yearweek@spark",
            ),
        ]
    )