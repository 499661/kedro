import logging
from pyspark.sql import DataFrame
import pyspark.sql.functions as f
from pyspark.sql.types import DecimalType
from pyspark.sql.window import Window
from sof_models_churn.common.date_utils import convert_end_date_to_python_date

logger = logging.getLogger(__name__)

def get_min_max_dates(dim_date: DataFrame, params: dict) -> DataFrame:
  
    end_date = max(map(convert_end_date_to_python_date,
                       [params["end_date_train"], params["end_date_pred"]]))
    end_date = end_date.strftime("%Y-%m-%d")
    n_weeks = max(params["n_weeks_total_train"], params["n_weeks_total_pred"])

    ## filter to take data upto last week into consideration
    dim_date_filter = dim_date\
        .filter(f.col('date') < end_date)\
        .groupBy('week_end_date')\
        .count()\
        .select('week_end_date', f.col('count').alias('cnt'))

    dim_date_filter = dim_date_filter\
        .filter(f.col('cnt') == 7)\
        .orderBy(f.col('week_end_date')\
        .desc())\
        .limit(n_weeks)

    ## getting min and max end_date
    min_max_dates_df = dim_date_filter\
        .agg(f.min(f.col('week_end_date')).alias('min_week_end_date'),
            f.max(f.col('week_end_date')).alias('max_week_end_date'))

    return min_max_dates_df

def create_cx_table(dim_location: DataFrame, dim_item: DataFrame, dim_date: DataFrame,
                     rs_table: DataFrame, week_ends: DataFrame) -> DataFrame:

    ## joining week_ends with dim_date table for max and min records
    joined_dates_max = week_ends\
        .join(dim_date,
            week_ends.max_week_end_date == dim_date.week_end_date)\
        .groupBy('week_end_date')\
        .agg(f.max(f.col('date')).alias('max_date'),
            f.max(f.col('period')).alias('max_period'),
            f.max(f.col('year')).alias('max_year'))\
        .withColumn('min_date', f.lit(None))\
        .withColumn('min_period', f.lit(None))\
        .withColumn('min_year', f.lit(None))\
        .select('week_end_date', 'min_date', 'max_date',
                'min_period', 'max_period',
                'min_year', 'max_year')

    joined_dates_min = week_ends\
        .join(dim_date,
            week_ends.min_week_end_date == dim_date.week_end_date)\
        .groupBy('week_end_date')\
        .agg(f.min(f.col('date')).alias('min_date'),
            f.min(f.col('period')).alias('min_period'),
            f.min(f.col('year')).alias('min_year'))\
        .withColumn('max_date', f.lit(None))\
        .withColumn('max_period', f.lit(None))\
        .withColumn('max_year', f.lit(None))\
        .select('week_end_date', 'min_date', 'max_date',
                'min_period', 'max_period',
                'min_year', 'max_year')

    joined_table_dates = joined_dates_max.union(joined_dates_min)

    ### getting the min and max date, period and year to key_dates 
    key_dates = joined_table_dates\
        .agg(f.min(f.col('min_date')).alias('min_date'),
            f.max(f.col('max_date')).alias('max_date'),
            f.min(f.col('min_period')).alias('start_period'),
            f.max(f.col('max_period')).alias('end_period'),
            f.min(f.col('min_year')).alias('start_year'),
            f.max(f.col('max_year')).alias('end_year'))

    ## pulling data from retail_sale and joining to key_dates
    rs_ecomm_df = rs_table\
                  .filter(f.col('ecomm_trans_id').isNotNull())\
                  .withColumn('cx', f.col('customer_id'))\
                  .select('cx','transaction_number', 'fiscal_date',
                          'ecomm_trans_id', 'year', 'month')

    rs_ecomm_keydate_join_cond = [rs_ecomm_df.fiscal_date >= key_dates.min_date,
                                    rs_ecomm_df.fiscal_date <= key_dates.max_date]
    
    key_dates_filter_df = rs_ecomm_df\
            .join(key_dates, rs_ecomm_keydate_join_cond)

    rs_ecomm_flt_df = key_dates_filter_df\
            .select('cx', 'transaction_number', 'fiscal_date', 'ecomm_trans_id')


    current_banners_list = ['SOF','UF','PSF']

    ## pulling data from retail_sale, dim_location, dim_item
    rs_df = rs_table\
              .filter(f.col('customer_id') != 0)\
              .withColumn('total_sales_dollars', f.col('merch_sales'))\
              .withColumn('extended_cost', f.col('product_cost'))\
              .withColumn('owt_points', f.col('points_issued'))\
              .withColumn('redemption_points', f.col('points_redeemed'))\
              .withColumn('total_sales_quantity', f.col('unit_count'))\
              .withColumn('total_sales_weight', f.col('weight'))\
              .withColumn('transaction_dt', f.col('fiscal_date'))\
              .withColumn('cost_uom_cd', f.when(f.col('weight') == 0, f.lit('EACH'))
                          .when(f.col('weight') != 0, f.lit('KG')))\
              .withColumn('cx', f.col('customer_id'))\
              .select('fiscal_date','cx','store_number','item_id','offer_id','total_sales_quantity','total_sales_weight','total_sales_dollars',
                          'extended_cost','owt_points','redemption_points','transaction_number','transaction_dt','cost_uom_cd','year')

    loc_table = dim_location\
            .filter((f.col('location_type') == 'ST')
                & (f.col('banner_short_name').isin(current_banners_list))
                & (f.col('is_active') == 1))\
            .withColumn('location_id', f.col('id'))\
            .select('location_id')

    item_table = dim_item\
            .filter(f.col('is_active') == 1)\
            .withColumn('item_id', f.col('id'))

    #convert all column name to lowercase
    for rs_col in rs_df.columns:
            rs_df = rs_df.withColumnRenamed(rs_col, rs_col.lower())

    for item_col in item_table.columns:
            item_table = item_table.withColumnRenamed(item_col, item_col.lower())

    rs_keydate_join_cond = [rs_df.fiscal_date >= key_dates.min_date,
                                rs_df.fiscal_date <= key_dates.max_date]

    hhn = rs_df\
            .join(key_dates, rs_keydate_join_cond)

    ## persist hhn dataframe
    ## comenting this out as this causes memory overload issue on cluster during pipeline run
    #hhn.cache()

    ##	The mdm_subcatergory ids can be from the following list and more:
    ##  1) Coupons
    ##  2) POS support items, training items
    ##  3) Store open special discounts
    ##  4) Charity items 
    ##  5) SOF Gift Cards
    ##  6) Unknown Category
    ##  7) Environment Fee 
    mdm_sub_cat_ids = ['8040','6639','6638','6637','6636','6635','6634']

    cx_transact= hhn\
        .join(loc_table, hhn.store_number == loc_table.location_id)\
        .join(item_table, hhn.item_id == item_table.item_id)\
        .drop(item_table.item_id)\
        .withColumn('sales_dollars_offer', 
                f.when((f.col('offer_id').isNotNull()) & (f.col('offer_id') != f.lit(0)), f.col('total_sales_dollars'))
                .otherwise(f.lit(0)))\
        .withColumn('extended_cost_offer', 
                f.when((f.col('offer_id').isNotNull()) & (f.col('offer_id') != f.lit(0)), f.col('extended_cost'))
                .otherwise(f.lit(0)))\
        .withColumn('units_sales', 
                f.when((((f.col('cost_uom_cd').isNull()) | (f.col('cost_uom_cd') == 'EACH')) & (f.col('total_sales_quantity') != f.lit(0))), f.col('total_sales_quantity'))
                .when(f.col('total_sales_weight') == f.lit(0), f.col('total_sales_quantity'))\
                .otherwise(f.col('total_sales_weight')))\
        .withColumn('units_sales_offer',\
                f.when((f.col('offer_id').isNotNull()) & (f.col('offer_id') != f.lit(0)), f.col('units_sales'))
                .otherwise(f.lit(0)))\
        .withColumn('units_sales_real',
                f.when(((f.col('mdm_sub_cat_id') >= '8044') & (f.col('mdm_sub_cat_id') <= '8104'))
                    | (f.col('mdm_sub_cat_id').isin(mdm_sub_cat_ids))
                        | ((f.col('mdm_sub_cat_id') >= '7431') & (f.col('mdm_sub_cat_id') <= '7440'))
                        | ((f.col('mdm_sub_cat_id') >= '6994') & (f.col('mdm_sub_cat_id') <= '7024'))
                        | (f.col('variant_name').like('%C/C RECORD%')), f.lit(0))
                .otherwise(f.col('units_sales')))\
        .withColumn('units_sales_real_offer', 
                f.when(((f.col('mdm_sub_cat_id') >= '8044') & (f.col('mdm_sub_cat_id') <= '8104'))
                    | (f.col('mdm_sub_cat_id').isin(mdm_sub_cat_ids))
                        | ((f.col('mdm_sub_cat_id') >= '7431') & (f.col('mdm_sub_cat_id') <= '7440'))
                        | ((f.col('mdm_sub_cat_id') >= '6994') & (f.col('mdm_sub_cat_id') <= '7024'))
                        | (f.col('variant_name').like('%C/C RECORD%')), f.lit(0))
                .otherwise(f.col('units_sales_offer')))\
        .withColumn('unique_item_count', 
                f.when(((f.col('mdm_sub_cat_id') >= '8044') & (f.col('mdm_sub_cat_id') <= '8104'))
                    | (f.col('mdm_sub_cat_id').isin(mdm_sub_cat_ids))
                        | ((f.col('mdm_sub_cat_id') >= '7431') & (f.col('mdm_sub_cat_id') <= '7440'))
                        | ((f.col('mdm_sub_cat_id') >= '6994') & (f.col('mdm_sub_cat_id') <= '7024'))
                        | (f.col('variant_name').like('%C/C RECORD%')), f.lit(None))
                .otherwise(f.col('item_id')))\
        .withColumn('unique_item_count_offer', 
                f.when(((f.col('mdm_sub_cat_id') >= '8044') & (f.col('mdm_sub_cat_id') <= '8104'))
                    | (f.col('mdm_sub_cat_id').isin(mdm_sub_cat_ids))
                        | ((f.col('mdm_sub_cat_id') >= '7431') & (f.col('mdm_sub_cat_id') <= '7440'))
                        | ((f.col('mdm_sub_cat_id') >= '6994') & (f.col('mdm_sub_cat_id') <= '7024'))
                        | (f.col('variant_name').like('%C/C RECORD%'))
                        | ((f.col('offer_id').isNull()) | (f.col('offer_id') == f.lit(0))), f.lit(None))
                .otherwise(f.col('item_id')))\
        .withColumn('redeem_count', 
                f.when(f.col('redemption_points') < f.lit(0), f.col('units_sales'))
                .otherwise(f.lit(0)))\
        .groupBy('fiscal_date', 'cx', 'location_id', 'transaction_number')\
        .agg(f.max(f.col('transaction_dt')).alias('transaction_dt'),
            f.coalesce(f.sum(f.col('total_sales_dollars')),f.lit(0)).alias('sales_dollars'),
            f.coalesce(f.sum(f.col('extended_cost')), f.lit(0)).alias('extended_cost'),
            f.coalesce(f.sum(f.col('sales_dollars_offer')), f.lit(0)).alias("sales_dollars_offer"),
            f.coalesce(f.sum(f.col('extended_cost_offer')), f.lit(0)).alias("extended_cost_offer"),
            f.coalesce(f.sum(f.col('units_sales')), f.lit(0)).alias('units_sales'),
            f.coalesce(f.sum(f.col('units_sales_offer')), f.lit(0)).alias('units_sales_offer'),
            f.coalesce(f.sum(f.col('units_sales_real')), f.lit(0)).alias('units_sales_real'),
            f.coalesce(f.sum(f.col('units_sales_real_offer')), f.lit(0)).alias('units_sales_real_offer'),
            f.coalesce(f.countDistinct(f.col('unique_item_count')), f.lit(0)).alias('unique_item_count'),
            f.coalesce(f.countDistinct(f.col('unique_item_count_offer')), f.lit(0)).alias('unique_item_count_offer'),
            f.coalesce(f.sum(f.col('total_sales_quantity')), f.lit(0)).alias('sales_quantity'),
            f.coalesce(f.sum(f.col('total_sales_weight')), f.lit(0)).alias('sales_weight'),
            f.sum(f.coalesce(f.col('owt_points'), f.lit(0))).alias('points_earned'),
            f.sum(-1*f.coalesce(f.col('redemption_points'), f.lit(0))).alias('points_redeemed'),
            f.coalesce(f.sum(f.col('redeem_count')), f.lit(0)).alias('redeem_count'),
            f.count(f.lit(1)).alias('row_count'))

    dim_date_flt = dim_date\
        .withColumnRenamed('Week', 'week_number')\
        .withColumnRenamed('Year', 'year_number')\
        .select('date', 'week_number', 'year_number')

    rs_ecomm_flag_df = rs_ecomm_flt_df\
        .filter('cx != 0')\
        .withColumn('ecomm_flag', f.lit(1))\
        .select('fiscal_date', 'cx', 'transaction_number', 'ecomm_flag')\
        .distinct()


    ctr_rs_ecomm_join_cond = [cx_transact.cx == rs_ecomm_flag_df.cx,
                    cx_transact.fiscal_date == rs_ecomm_flag_df.fiscal_date,
                    cx_transact.transaction_number == rs_ecomm_flag_df.transaction_number]


    cx_transact_yearweek = cx_transact\
        .join(rs_ecomm_flag_df, ctr_rs_ecomm_join_cond, "left")\
        .join(dim_date_flt, cx_transact.fiscal_date == dim_date_flt.date)\
        .drop(rs_ecomm_flag_df.cx)\
        .drop(rs_ecomm_flag_df.transaction_number)\
        .drop(rs_ecomm_flag_df.fiscal_date)\
        .groupBy('cx','week_number','year_number','location_id')\
        .agg(f.count(f.lit(1)).alias('n_transaction'),
            f.sum(f.coalesce(f.col('ecomm_flag'), f.lit(0))).alias('n_ecomm_trans'),
            f.sum(f.col('sales_dollars')).alias('sales_dollars'),
            f.sum(f.col('extended_cost')).alias('extended_cost'),
            f.sum(f.col('sales_dollars_offer')).alias('sales_dollars_offer'),
            f.sum(f.col('units_sales')).alias('units_sales'),
            f.sum(f.col('units_sales_offer')).alias('units_sales_offer'),
            f.sum(f.col('points_earned')).alias('points_earned'),
            f.sum(f.col('points_redeemed')).alias('points_redeemed'),
            f.sum(f.col('redeem_count')).alias('redeem_count'))

    # #hhn.unpersist()

    # This time is wrong, because it does not account for collecting the DataFrame
    # logger.info("Table creation complete. Time taken: {} min".format((time()-t0)/60) )
    return cx_transact_yearweek

def create_hh_table(dim_location: DataFrame, dim_customer: DataFrame, dim_date:
                    DataFrame, week_ends: DataFrame, cx_transact_yearweek: DataFrame,
                    params: dict) -> DataFrame:

    ## joining week_ends with dim_date table for max and min records
    joined_dates_max = week_ends \
        .join(dim_date,
                week_ends.max_week_end_date == dim_date.week_end_date)\
        .groupBy('week_end_date')\
        .agg(f.max(f.col('week')).alias('max_week'),
            f.max(f.col('year')).alias('max_year'))\
        .withColumn('min_week', f.lit(None))\
        .withColumn('min_year', f.lit(None))\
        .select('week_end_date', 'min_week', 'max_week',
                'min_year', 'max_year')

    joined_dates_min = week_ends \
        .join(dim_date,
                week_ends.min_week_end_date == dim_date.week_end_date)\
        .groupBy('week_end_date')\
        .agg(f.min(f.col('week')).alias('min_week'),
            f.min(f.col('year')).alias('min_year'))\
        .withColumn('max_week', f.lit(None))\
        .withColumn('max_year', f.lit(None))\
        .select('week_end_date', 'min_week', 'max_week',
                'min_year', 'max_year')

    joined_table_dates = joined_dates_max.union(joined_dates_min)

    key_dates_hhn = joined_table_dates\
        .agg(f.min(f.col('min_week')).alias('start_week'),
            f.max(f.col('max_week')).alias('end_week'),
            f.min(f.col('min_year')).alias('start_year'),
            f.max(f.col('max_year')).alias('end_year'))

    ## pulling data from cx_transact_yearweek and joining to key_dates
    cxtrnsct_keydatehhn_join_cond = [(cx_transact_yearweek.year_number >= key_dates_hhn.start_year) &
                            (cx_transact_yearweek.year_number <= key_dates_hhn.end_year) 
                        &~ ((cx_transact_yearweek.year_number == key_dates_hhn.start_year) & (cx_transact_yearweek.week_number < key_dates_hhn.start_week))
                            &~ ((cx_transact_yearweek.year_number == key_dates_hhn.end_year) & (cx_transact_yearweek.week_number > key_dates_hhn.end_week))]

    ctr = cx_transact_yearweek\
    .join(key_dates_hhn, cxtrnsct_keydatehhn_join_cond)

    cust_table = dim_customer\
              .filter((f.col('household_id') != 0)
                      & (f.col('is_active') == 1)\
                      & (f.col('deleted_flag') == 0))\
            .select(f.col('ldw_customer_id').alias('cx'), f.col('household_id').alias('hhn'))

    group_table = ctr\
        .join(cust_table,
            ctr.cx == cust_table.cx)\
        .groupBy('hhn', 'week_number', 'year_number', 'location_id')\
        .agg(f.count(f.lit(1)).alias('n_transaction'),
            f.sum(f.col('n_ecomm_trans')).alias('n_ecomm_trans'),
            f.sum(f.col('sales_dollars')).alias('sales_dollars'),
            f.sum(f.col('extended_cost')).alias('extended_cost'),
            f.sum(f.col('sales_dollars_offer')).alias('sales_dollars_offer'),
            f.sum(f.col('units_sales')).alias('units_sales'),
            f.sum(f.col('units_sales_offer')).alias('units_sales_offer'),
            f.sum(f.col('points_earned')).alias('points_earned'),
            f.sum(f.col('points_redeemed')).alias('points_redeemed'),
            f.sum(f.col('redeem_count')).alias('redeem_count'))

    ## persist group_table dataframe
    #group_table.cache()

    ## defining window function filter
    group_table_wndw_fltr = Window().partitionBy('hhn', 'week_number', 'year_number')\
                        .orderBy(f.desc('sales_dollars'), f.asc('location_id'))

    maxsales_location_table = group_table\
    .withColumn('maxsales_location_id', f.first('location_id').over(group_table_wndw_fltr))\
    .select('hhn', 'week_number', 'year_number', 'maxsales_location_id')\
    .distinct()

    final_group = group_table\
        .groupBy('hhn','week_number', 'year_number')\
        .agg(f.sum(f.col('n_transaction')).alias('n_transaction'),
            f.sum(f.coalesce(f.col('n_ecomm_trans'), f.lit(0))).alias('n_ecomm_trans'),
            f.sum(f.col('sales_dollars')).alias('sales_dollars'),
            f.sum(f.col('extended_cost')).alias('extended_cost'),
            f.sum(f.col('units_sales')).alias('units_sales'),
            f.sum(f.col('units_sales_offer')).alias('units_sales_offer'),
            f.sum(f.col('points_earned')).alias('points_earned'),
            f.sum(f.col('points_redeemed')).alias('points_redeemed'),
            f.sum(f.col('redeem_count')).alias('redeem_count'))

    ## using aggregated columns from final_group dataframe to generate unit_offer_ratio 
    final_group = final_group\
        .withColumn('unit_offer_ratio',
                    f.when(f.col('units_sales') <= f.lit(0), f.lit(0))
                    .when(f.col('units_sales_offer') > f.col('units_sales'), f.lit(1.0))
                    .when(f.col('units_sales_offer') <= f.lit(0), f.lit(0))
                    .otherwise((f.lit(1.0)*f.col('units_sales_offer'))/(f.lit(1.0)*f.col('units_sales'))))\
        .select('hhn','week_number', 'year_number',
                'n_transaction','n_ecomm_trans','sales_dollars',
                'extended_cost','units_sales','unit_offer_ratio',
                'points_earned','points_redeemed','redeem_count')

    dim_date_flt = dim_date\
        .select('year', 'week', 'week_end_date')\
        .withColumnRenamed('year', 'year_number')\
        .withColumnRenamed('week', 'week_number')\
        .withColumnRenamed('week_end_date', 'week_ending_date')\
        .distinct()
    
    hh_transactevents_yearweek = final_group.alias('f')\
        .join(dim_date_flt.alias('ddf'),
                (f.col('f.week_number') == f.col('ddf.week_number')) 
                & (f.col('f.year_number') == f.col('ddf.year_number')))\
        .join(maxsales_location_table.alias('mlt'), 
                (f.col('f.hhn') == f.col('mlt.hhn')) 
                & (f.col('f.week_number') == f.col('mlt.week_number'))
                & (f.col('f.year_number') == f.col('mlt.year_number')))\
        .select('f.hhn', 'f.week_number', 'f.year_number', 'ddf.week_ending_date', 'f.n_transaction', 
                'f.n_ecomm_trans', 'mlt.maxsales_location_id', 'f.sales_dollars', 
                'f.extended_cost', 'f.units_sales', 'f.unit_offer_ratio', 
                'f.points_earned', 'f.points_redeemed', 'f.redeem_count')

    ## aggregation of features 
    ## casting two columns to decimal to maintain the schema already present in the delta table
    global_aggregated_features = hh_transactevents_yearweek\
        .groupBy('year_number', 'week_number')\
        .agg(f.percentile_approx('sales_dollars', 0.5).alias('median_sales'),
            f.percentile_approx('units_sales', 0.5).alias('median_units_sales'),
            f.avg('n_transaction' * int(1.0)).cast(DecimalType(27,5)).alias('avg_n_transaction'),
            f.avg('n_ecomm_trans' * int(1.0)).cast(DecimalType(27,5)).alias('avg_n_ecomm_trans'))

    hhn_distinct = hh_transactevents_yearweek\
        .orderBy('hhn')\
        .select('hhn')\
        .distinct()

    ## assigning each unique household with a row number
    unique_hhn_row = hhn_distinct\
        .withColumn('unique_hhn_num', f.row_number().over(Window.orderBy('hhn')) - 1)\
        .orderBy('hhn')

    ## dividing households into batches of 25,000
    partition_index = unique_hhn_row\
        .withColumn('hhn_batch_num', f.floor(f.col('unique_hhn_num')/25000))\
        .select('hhn', 'hhn_batch_num')

    current_banners_list = ["SOF","UF","PSF"]
    
    loc_table =  dim_location\
              .filter((f.col('location_type') == 'ST')
                      & (f.col('banner_short_name').isin(current_banners_list))
                      & (f.col('is_active') == 1))\
              .select(f.col('id').alias('location_id'),'divisional_regional_rollup')

    hhn_transact_yearweek = hh_transactevents_yearweek.alias('a')\
        .join(loc_table, f.col('a.maxsales_location_id') == loc_table.location_id)\
        .join(global_aggregated_features.alias('globaler'), 
                (f.col('a.year_number') == f.col('globaler.year_number'))
                & (f.col('a.week_number') == f.col('globaler.week_number')), 'left')\
        .join(partition_index.alias('parter'), f.col('a.hhn') == f.col('parter.hhn'))\
        .select('a.*', 'divisional_regional_rollup', 'globaler.median_sales', 
                'globaler.median_units_sales', 'globaler.avg_n_transaction',
                'globaler.avg_n_ecomm_trans', 'parter.hhn_batch_num')
    return hhn_transact_yearweek
