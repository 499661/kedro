import logging
from pyspark.sql.functions import *

logger = logging.getLogger(__name__)
## Test 1: Check for nulls in the entire output table
def null_check(df, tableName):
    null_counts = df\
        .select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])\
        .collect()[0]\
        .asDict()
    
    if (all(value == 0 for value in null_counts.values())) == True:
        logger.info('Test 1 - ' + tableName + ' null_check Success: No Nulls found')
    else:
        for key, value in null_counts.items():
            if (value > 0):
                logger.info('Test 1 - Nulls Found in the ' + tableName + 'for ' + str(key) + ' - ' + str(value))
        raise Exception('Test 1 - ' + tableName + ' null_check Failure: Nulls found in the table')

## Test 2: Checking for columns with all zeroes
def all_zeroes_check(df, tableName):
    zero_counts_df = df\
        .select([count(when(col(c) == 0, c)).alias(c) for c in df.columns])\
        .collect()[0]\
        .asDict()
    total_counts = df.count()

    if (all(value < total_counts for value in zero_counts_df.values())) == True:
        logger.info('Test 2 - ' + tableName + ' all_zeroes_check Suceess: No column has all zeroes')
    else:
        for key, value in zero_counts_df.items():
            if (value == total_counts):
                print('Test 2 - Column: ' + str(key) + ' - has all zeroes in ' + tableName)
        raise Exception('Test 2 - ' + tableName + ' all_zeroes_check Failure: Table has a column with all zeroes')

## Test 3: Churn probability value should be between 0 and 1
def churn_probability_value_check(df, tableName):
    probability_value_check = df\
        .where((col('churn_probability')<0) | (col('churn_probability')>1))
    
    if probability_value_check.count() > 0:
        raise Exception('Test 3 -' + tableName + ' churn_probability_value_check Failure : churn_probability has value less than zero or greater than one')
    else:
        logger.info('Test 3 - ' + tableName + ' churn_probability_value_check Pass: churn_probability_value_check has no anomalies')


## Test 4: Number of weeks until churned should be between 0 and n_weeks_churn with 0 and n_weeks_churn included
def weeks_until_churned_check(df, tableName, n_weeks_churn):
    weeks_until_churned_value_check = df\
        .where((col('weeks_until_churned')<0) | (col('weeks_until_churned')>n_weeks_churn))

    if weeks_until_churned_value_check.count() >0:
        raise Exception('Test 4 -' + tableName + f' weeks_until_churned_check Failure : weeks_until_churned has value less than zero or greater than {n_weeks_churn}')
    else:
        logger.info('Test 4 - ' + tableName + ' weeks_until_churned_check Pass: weeks_until_churned column has no anomalies')


## Test 5: Alpha and Beta value should be greater than 0
def alpha_beta_value_check( df, tableName):
    alpha_beta_value_check_condition = df\
        .where((col('alpha')<=0) | (col('beta')<=0))
    
    if alpha_beta_value_check_condition.count() >0:
        raise Exception('Test 5 -' + tableName + ' alpha_beta_value_check Failure : Either alpha or beta column has value equal to zero or less than zero ')
    else:
        logger.info('Test 5 - ' + tableName + ' alpha_beta_value_check Pass: alpha, beta  column has no anomalies')
    

## Test 6: Number of weeks since last shop and number of weeks since first shop should be less than n_weeks with n_weeks included
## WSLS = weeks since last shop , WSFS = weeks since first shop
def wsls_wsfs_value_check(df, tableName, n_weeks):
    wsls_wsfs_value_check_condition = df\
        .where((col('weeks_since_last_shop')>n_weeks)|(col('weeks_since_first_shop')>n_weeks))

    if wsls_wsfs_value_check_condition.count() >0:
        raise Exception('Test 6 -' + tableName + ' wsls_wsfs_value_check Failure : Either weeks_since_last_shop or weeks_since_first_shop has value more than required')
    else:
        logger.info('Test 6 - ' + tableName + ' wsls_wsfs_value_check Pass: weeks_since_last_shop, weeks_since_first_shop has no anomalies')

