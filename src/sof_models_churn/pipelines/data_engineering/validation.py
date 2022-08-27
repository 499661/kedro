from pyspark.sql.functions import *
import logging

logger = logging.getLogger(__name__)

## Test 1: The count of transactions for any given week should not be below 600k
def count_check(df, tableName):
    count_df = df\
        .groupBy('week_number', 'year_number')\
        .agg(count(lit(1)).alias('total_count'))\
        .orderBy('week_number', 'year_number')

    count_condition = count_df\
    .where(col('total_count') < 600000)

    if count_condition.count() > 0:
        raise Exception('Test 1 - ' + tableName + ' count_check Failure: Count of transactions for a week is less than 600k')
    else:
        logger.info('Test 1 - ' + tableName + ' count_check Pass: Count of transactions for every week is above 600k')

## Test 2: Checking for Nulls in cx/hhn weekly table
def null_check(df, tableName):
    null_counts = df\
        .select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])\
        .collect()[0]\
        .asDict()
    
    if (all(value == 0 for value in null_counts.values())) == True:
        logger.info('Test 2 - ' + tableName + ' null_check Success: No Nulls found')
    else:
        for key, value in null_counts.items():
            if (value > 0):
                logger.info('Test 2 - Nulls Found in the ' + tableName + 'for ' + str(key) + ' - ' + str(value))
        raise Exception('Test 2 - ' + tableName + ' null_check Failure: Nulls found in the table')

## Test 3: Total Sales per week should not be below 60M
def weekly_sales_check(df, tableName):
    sales_dollars_weekly_df = df\
        .select('week_number', 'year_number', 'sales_dollars')\
        .groupBy('week_number', 'year_number')\
        .agg(sum(col('sales_dollars')).alias('total_sales_dollars_weekly'))\
        .orderBy('week_number', 'year_number')

    weekly_sales_condition = sales_dollars_weekly_df\
    .where(col('total_sales_dollars_weekly') < 60000000)

    if weekly_sales_condition.count() > 0:
        raise Exception('Test 3 - ' + tableName + ' weekly_sales_check Failure: total_sales_dollars_weekly for a week is less than 60M')
    else:
        logger.info('Test 3 - ' + tableName + ' weekly_sales_check Pass: total_sales_dollars_weekly for a week is more than 60M')

## Test 4: Total Units per week should not be below 15M
def unit_sales_check(df, tableName):
    unit_sales_weekly_df = df\
        .select('week_number', 'year_number', 'units_sales')\
        .groupBy('week_number', 'year_number')\
        .agg(sum(col('units_sales')).alias('total_unit_sales_weekly'))\
        .orderBy('week_number', 'year_number')

    weekly_unit_sales_condition = unit_sales_weekly_df\
        .where(col('total_unit_sales_weekly') < 15000000)

    if weekly_unit_sales_condition.count() > 0:
        raise Exception('Test 4 - ' + tableName + ' unit_sales_check Failure: total_unit_sales_weekly for a week are less than 15M')
    else:
        logger.info('Test 4 - ' + tableName + ' unit_sales_check Pass: total_unit_sales_weekly for a week are more than 15M')

## Test 5: Checking for columns with all Zeroes
def all_zeroes_check(df, tableName):
    zero_counts_df = df\
        .select([count(when(col(c) == 0, c)).alias(c) for c in df.columns])\
        .collect()[0]\
        .asDict()
    total_counts = df.count()

    if (all(value < total_counts for value in zero_counts_df.values())) == True:
        logger.info('Test 5 - ' + tableName + ' all_zeroes_check Suceess: No column has all zeroes')
    else:
        for key, value in zero_counts_df.items():
            if (value == total_counts):
                logger.info('Test 5 - Column: ' + str(key) + ' - has all zeroes in ' + tableName)
        raise Exception('Test 5 - ' + tableName + ' all_zeroes_check Failure: Table has a column with all zeroes')

## Test 6: avg_n_transactions should not be less than 1
def avg_n_transaction_check(df, tableName):
    avg_n_transaction_condition= df\
        .where(col('avg_n_transaction') < 1) 

    if avg_n_transaction_condition.count() > 0:
        raise Exception('Test 6 - ' + tableName + ' avg_n_transaction_check Failure: avg_n_transaction has values less than 1')
    else:
        logger.info('Test 6 - ' + tableName + ' avg_n_transaction_check Pass: avg_n_transaction has no anomalies')

## Test 7: avg_ecomm_transaction should not be equal to zero
def avg_ecomm_transaction_check(df, tableName):
    avg_n_ecomm_trans_condition= df\
        .where(col('avg_n_ecomm_trans') == 0) 

    if avg_n_ecomm_trans_condition.count() > 0:
        raise Exception('Test 7 - ' + tableName + ' avg_ecomm_transaction_check Failure: avg_n_ecomm_trans has values less than 1')
    else:
        logger.info('Test 7 - ' + tableName + ' avg_ecomm_transaction_check Pass: avg_n_ecomm_trans has no anomalies')

## Test 8: unit_offer_ratio should be between 0 and 1
def unit_offer_ratio_check(df, tableName):
    offer_ratio_condition= df\
        .where((col('unit_offer_ratio') < 0) & (col('unit_offer_ratio') > 1))

    if offer_ratio_condition.count() > 0:
        raise Exception('Test 8 - ' + tableName + 'unit_offer_ratio_check Failure: unit_offer_ratio has values less than zero or greater than one')
    else:
        logger.info('Test 8 - ' + tableName + 'unit_offer_ratio_check Pass: unit_offer_ratio has no anomalies')
