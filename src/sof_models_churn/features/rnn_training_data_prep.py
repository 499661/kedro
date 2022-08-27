import numpy as np
import pandas as pd
import sof_models_churn.common.wtte_utils as wtte

from datetime import date, timedelta
from typing import Optional, Union
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
from typing import Tuple, Dict, Any
from sof_models_churn.common.date_utils import convert_end_date_to_python_date
import logging

logger = logging.getLogger(__name__)

class WtteRnnDataPrep():

    def __init__(self, spark=None,
                 n_weeks_total=None):
        self.spark = spark
        self.n_weeks_total = n_weeks_total

        self.columns = ['year_number','week_number','week_ending_date','hhn', 'n_transaction',
                    'n_ecomm_trans', 'maxsales_location_id', 'sales_dollars', 'extended_cost',
                    'units_sales', 'unit_offer_ratio', 'points_earned', 'points_redeemed',
                    'redeem_count', 'divisional_regional_rollup', 'median_sales', 'median_units_sales', 'avg_n_transaction', 'avg_n_ecomm_trans']

        # **NB**  The column order in self.column_types matters(!!), as column rescaling is performed on the "first n " columns. 
        # (See `first_n_cols_to_rescale` kwarg below). This is not great, but will apparently not cause probs as long as we are using python 3.7+ 
        # (see https://stackoverflow.com/a/40007169 ). Consider replacing this with an OrderedDict.
        self.column_types = {'n_transaction':np.int32, 'n_ecomm_trans':np.int32,
                        'sales_dollars':np.float64,'extended_cost':np.float64, 'units_sales':np.float64,
                        'points_earned':np.int64, 'points_redeemed':np.int64, 'redeem_count':np.int64,
                        'avg_n_transaction':np.float64, 'avg_n_ecomm_trans':np.float64, 'median_sales':np.float64,
                        'median_units_sales':np.float64,'unit_offer_ratio':np.float64,'week_in_year_cos':np.float64,
                        'week_in_year_sin':np.float64,'div_reg_index':np.int32}

        self.regions_added = None # updated by preprocessing_A when a new region is added

    def get_week_ending_dates(self, dim_date, hhn_transact_yearweek,
                              end_date=None, n_weeks_total=None):
        """
        KWARGS:
            dim_date (DataFrame): the date dimension

            hhn_transact_yearweek (DataFrame): transaction data by household number

            end_date (str or None): historical end_date for simulating historical predictions. If None, then defaults to today's date (ie. date of actual run)

            n_weeks_total (int): number of weeks of history to use.
        Returns:
            start_week_ending_date (date): the most recent `week_ending_date` prior to `end_date`
            end_week_ending_date (date): the first `week_ending_date`
        """
        if n_weeks_total is None:
            assert self.n_weeks_total is not None, "`n_weeks_total` cannot be `None`"
            n_weeks_total = self.n_weeks_total

        end_date = convert_end_date_to_python_date(end_date)

        # Get the most recent `week_ending_date` prior to `end_date`:
        end_week_ending_date = (dim_date.where(F.col("date") == end_date)
                                        .select(F.date_add(F.col("week_end_date"), -7))
                                        .toPandas().iloc[0,0])

        start_week_ending_date = end_week_ending_date - timedelta(weeks=n_weeks_total - 1)

        # Get max & min weeks that are available in our data (hhn transaction table):
        max_available_transact_week, min_available_transact_week = \
                (
                hhn_transact_yearweek.select(
                    F.max(F.col("week_ending_date")).alias("maxwed"),
                    F.min(F.col("week_ending_date")).alias("minwed")
                    )
                .toPandas().iloc[0,:].to_list()
                )

        logger.info('end_date: {:s}'.format(str(end_date)))
        logger.info('end_week_ending_date: {:s}'.format(str(end_week_ending_date)))
        logger.info('min_available_transact_week: {:s}'.format(str(min_available_transact_week)))
        logger.info('max_available_transact_week: {:s}'.format(str(max_available_transact_week)))
        logger.info('n_weeks_total: {:d}'.format(n_weeks_total))
        assert end_week_ending_date <= max_available_transact_week, \
            "`end_date` is later than latest date in hhn table!"
        # note ``n_weeks_total-1`` here for correct counting of week_ending_dates:
        assert start_week_ending_date >= min_available_transact_week, \
            "There are fewer than `n_weeks` weeks of data available in hhn table!"

        return start_week_ending_date, end_week_ending_date

    def create_subset(self,
                      hhn_transact_yearweek: DataFrame,
                      split_frac: Optional[float] = 0.2,
                      split_lower_threshold: Optional[float] = 0.,
                      end_date: Optional[str] = None,
                      n_weeks_total: int = 104,
                      batch_numbers: Optional[Union[int, list]] = None,
                      ) -> Tuple[DataFrame, date, date]:
        """
        Returns a subset of the dataset for training or testing or prediction.

        KWARGS (general):
            hhn_transact_yearweek (DataFrame): the preprocessed household number table

            end_date (str or None): historical end_date for simulating historical predictions. If None, then defaults to today's date (ie. date of actual run)

            n_weeks_total (int): number of weeks of history to use.

            batch_numbers (int or list of ints): specific hhn batch numbers to use if only a subset is required.

            split_frac (float or None): fraction of total HHN to use for this dataset.

            split_lower_threshold (float or None): number between 0 and 1 indicating lower end of the 'hashed hhn' range to use for subsetting. Eg: if split_frac=0.3 and split_lower_threshold=0.2, then we take all households for which 0.2 < `hashed_hhn`/(2^32) <= 0.5

        Returns:
            filtered_hhn_transact (DataFrame): a subset of the dataset.
        """
        end_date = convert_end_date_to_python_date(end_date)

        hht = hhn_transact_yearweek.where( (F.col("week_ending_date") < end_date) &  
                                           (F.col("week_ending_date") >=  F.date_sub(F.lit(end_date),
                                                                            7*n_weeks_total)) )
        min_week_endingd_date = hht.select(F.min(F.col("week_ending_date"))).collect()[0][0]
        max_week_endingd_date = hht.select(F.max(F.col("week_ending_date"))).collect()[0][0]
        logger.info("end_date: {:s}".format(str(end_date)))
        logger.info("min_week_ending_date: {:s}".format(str(min_week_endingd_date)))
        logger.info("max_week_ending_date: = {:s}".format(str(max_week_endingd_date)))

        if batch_numbers is not None:
            if isinstance(batch_numbers, int): batch_numbers = [batch_numbers]
            hht = hht.where(F.col("hhn_batch_num").isin(batch_numbers))

        hhn_transact = hht.select(
                            "year_number",
                            "week_number",
                            "week_ending_date",
                            "hhn",
                            "n_transaction",
                            "n_ecomm_trans",
                            "maxsales_location_id",
                            "sales_dollars",
                            "extended_cost",
                            "units_sales",
                            "unit_offer_ratio",
                            "points_earned",
                            "points_redeemed",
                            "redeem_count",
                            "divisional_regional_rollup",
                            "median_sales",
                            "median_units_sales",
                            "avg_n_transaction",
                            "avg_n_ecomm_trans"
                        )
        if split_frac is not None:
            # FIXME: what about the "reference_table" mode? Do we ever use this?
            hhn_transact = hhn_transact.where(
                                ( F.crc32(F.col("hhn").cast(StringType())) > split_lower_threshold * pow(2, 32) ) & 
                                ( F.crc32(F.col("hhn").cast(StringType())) <= (split_lower_threshold + split_frac) * pow(2,32))
                            )
        return hhn_transact

    def prepare_subset_for_model(self,
                                 filtered_hhn_transact,
                                 start_week_ending_date,
                                 end_week_ending_date,
                                 prediction: bool = False,
                                 regional_lookup: Optional[pd.DataFrame] = None,
                                 default_region: Optional[str] = "Surrey Central",
                                 quantile_choice: Optional[float] = 0.6,
                                 validation_steps_cut_frac: Optional[float] = None,
                                 n_testing_timesteps: Optional[int] = 4,
                                 to_keep: int = 4,
                                 first_n_cols_to_rescale: int  = 12,
                                 mask_value: float = 0.133713371337,
                                 ) -> Dict[str, Any]:
        """
        Generates a dataset ready for ingestion by the ML model - for training, testing or prediction.

        KWARGS (general):
            filtered_hhn_transact (Dataframe): subset of hhn table (output of self.split_dataset)

            start_week_ending_date (date): the most recent `week_ending_date` prior to `end_date`

            end_week_ending_date (date): the first `week_ending_date`

            prediction (bool): whether data is being prepared for prediction (as opposed to train/test)

            first_n_cols_to_rescale (int): rescale the first n feature columns (using np.log(1+x)).

            mask_value (float): an "unlikely" value to use as a masking indicator to keras

            regional_lookup (pd.DataFrame):
            
            default_region (str): name of divisional_regional_rollup to use when encountering one that is missing from `regional_lookup` (i.e. a new region that didn't exist when model was trained)

        KWARGS REQUIRED FOR Train/Val/Test sets ONLY:
            quantile_choice (float or None): *only for training*. Used to undersample households with shorter purchase histories.

            n_testing_timesteps (int): number of last timesteps to exclude from training samples (held out for testing/validation purposes).

            validation_steps_cut_frac (float): this is an alternative way to specify `n_testing_timesteps`; give the *fraction* of the (max_sequence_length-) timesteps to exclude from training set.

            to_keep (int): the number of held-out timesteps to use for testing/validation (must be <= n_testing_timesteps)

        KWARGS REQUIRED FOR PREDICTION ONLY:

        Returns:
            data_dict (dictionary) containing:
                x_train_features, x_train_store,
                x_valid_features, x_valid_store, 
                x_features, x_store, 
                y_valid, y_train, y,
                events,
                sample_weights_train, sample_weights_valid, 
                seq_lengths, lookup_hhn_id, mask_value, 
                regional_lookup, div_reg_unique_count, 
                init_alpha,
                end_week_ending_date, start_week_ending_date
        """

        df = filtered_hhn_transact.toPandas()
        df = df[self.columns]
        assert len(df)>0, 'Query returned zero rows!'
        logger.info('Unique households: {}'.format(df.hhn.nunique()))
        (min_week_ending_date_name, max_week_ending_date_name) = df.week_ending_date.min(), df.week_ending_date.max()

        # Some preprocessing bundled together (TO DO: needs to be unbundled for readability)
        df, regional_lookup, div_reg_unique_count, lookup_hhn_id = self.preprocessing_A(df, regional_lookup=regional_lookup,
                                                                                        default_region=default_region)

        if quantile_choice is not None:
            df, lookup_hhn_id = self.subsample(df, lookup_hhn_id=lookup_hhn_id, quantile_choice=quantile_choice)
        x  = self.get_feature_matrix(df=df,
                               min_date=min_week_ending_date_name,
                               max_date=max_week_ending_date_name,
                               mask_value=np.nan, # NB: use NaN, otherwise mask value gets subsequently rescaled!
                               feature_cols=list(self.column_types.keys()))
        logger.info("Shape of x: {:s}".format(str(x.shape)))

        if prediction is True:
            if first_n_cols_to_rescale>0:
                x = self.rescale(x, first_n_cols_to_rescale=first_n_cols_to_rescale)

            x[:,:,:][np.isnan(x)] = mask_value
            ### split x into two types of inputs
            x_features = x[:,:,0:15]
            x_store = x[:,:,15:16]

            data_dict = {'x_features':x_features, 'x_store':x_store,
                        'lookup_hhn_id':lookup_hhn_id}

        else:
            events = self.get_events(x)
            if first_n_cols_to_rescale>0:
                x = self.rescale(x, first_n_cols_to_rescale=first_n_cols_to_rescale)

            ### split x into two types of inputs
            x_features = x[:,:,0:15]
            x_store = x[:,:,15:16]

            (x_train,
             y_train,
             events_train) = self.generate_train_tensors(x, events,
                                            frac_timesteps_to_cut=validation_steps_cut_frac,
                                            n_testing_timesteps=n_testing_timesteps)
            logger.info("Shapes of x_train, y_train, events_train: {:s} {:s} {:s}".\
                        format(str(x_train.shape), str(y_train.shape), str(events_train.shape)))

            x, y, events    = self.prep_tensors(x,events)
            logger.info("x, y , events shapes: {:s} {:s} {:s}".format(str(x.shape),
                        str(y.shape), str(events.shape)))

            # More preprocessing bundled together. TO DO: unbundle for readability!
            ( x_train_features, x_train_store, y_train, sample_weights_train,
              x_valid_features, x_valid_store, y_valid, sample_weights_valid,
              x_features, x_store, y, seq_lengths, init_alpha
              ) = self.preprocessing_B( x, y, events,
                                   x_train, y_train, events_train,
                                   mask_value=mask_value, to_keep=to_keep)
            logger.info("Shape of x_train_features: {:s}".format(str(x_train_features.shape)))
            logger.info("Shape of x_valid_features: {:s}".format(str(x_valid_features.shape)))
            logger.info("Shape of y_train: {:s}".format(str(y_train.shape)))
            logger.info("Shape of y_valid: {:s}".format(str(y_valid.shape)))
            logger.info("Shape of sample_weights_train: {:s}".format(str(sample_weights_train.shape)))
            logger.info("Shape of sample_weights_valid: {:s}".format(str(sample_weights_train.shape)))

            data_dict = {'x_train_features':x_train_features,
             'x_train_store':x_train_store, 'x_valid_features':x_valid_features, 'x_valid_store':x_valid_store, 
             'x_features':x_features, 'x_store':x_store, 'y_valid':y_valid, 'y_train':y_train, 'y':y, 'events':events, 
             'sample_weights_train':sample_weights_train, 'sample_weights_valid':sample_weights_valid, 
             'seq_lengths':seq_lengths, 'lookup_hhn_id':lookup_hhn_id, 'mask_value':mask_value, 
             'regional_lookup': regional_lookup, 'div_reg_unique_count':div_reg_unique_count, 
             'init_alpha':init_alpha, 'end_week_ending_date': end_week_ending_date,
             'start_week_ending_date': start_week_ending_date}

        return data_dict

    def prepare_dataset_for_model(self,
                                  dim_date: DataFrame,
                                  hhn_transact_yearweek: DataFrame,
                                  split_frac: Optional[float] = 0.2,
                                  split_lower_threshold: Optional[float] = 0.,
                                  end_date: Optional[str] = None,
                                  n_weeks_total: int = 104,
                                  prediction: bool = False,
                                  regional_lookup: Optional[pd.DataFrame] = None,
                                  default_region: Optional[str] = "Surrey Central",
                                  quantile_choice: Optional[float] = 0.6,
                                  validation_steps_cut_frac: Optional[float] = None,
                                  n_testing_timesteps: Optional[int] = 4,
                                  to_keep: int = 4,
                                  first_n_cols_to_rescale: int  = 12,
                                  mask_value: float = 0.133713371337,
                                  batch_numbers: Optional[Union[int, list]] = None,
                                  ) -> DataFrame:
        """
        Convenience method which combines get_week_ending_dates, create_subset, and prepare_filtered_dataset_for_model.
        """
        start_week_ending_date, end_week_ending_date = self.get_week_ending_dates(dim_date,
                                                                                  hhn_transact_yearweek,
                                                                                  end_date=end_date,
                                                                                  n_weeks_total=n_weeks_total)

        filtered_hhn_transact = self.create_subset(hhn_transact_yearweek,
                                                   split_frac=split_frac,
                                                   split_lower_threshold=split_lower_threshold,
                                                   end_date=end_date,
                                                   n_weeks_total=n_weeks_total,
                                                   batch_numbers=batch_numbers)
        data_dict = self.prepare_subset_for_model(filtered_hhn_transact,
                                                  start_week_ending_date,
                                                  end_week_ending_date,
                                                  prediction=prediction,
                                                  regional_lookup=regional_lookup,
                                                  default_region=default_region,
                                                  quantile_choice=quantile_choice,
                                                  validation_steps_cut_frac=validation_steps_cut_frac,
                                                  n_testing_timesteps=n_testing_timesteps,
                                                  to_keep=to_keep,
                                                  first_n_cols_to_rescale=first_n_cols_to_rescale,
                                                  mask_value=mask_value)
        return data_dict, filtered_hhn_transact

    def create_regional_lookup(self, df):
        # generate in regional_lookup
        logger.info("Creating regional lookup table")
        regional_lookup = (pd.DataFrame(
            df['divisional_regional_rollup'].unique(),
            columns=['divisional_regional_rollup'])
                           .sort_values(by=['divisional_regional_rollup'])
                           .reset_index(drop=True)
                           .reset_index() )
        regional_lookup.columns=['div_reg_index','divisional_regional_rollup']
        # extra index because 'mask_value' effectively adds to embedding dimension:
        regional_lookup['div_reg_index'] = regional_lookup['div_reg_index'] + 1
        return regional_lookup

    def preprocessing_A(self, df, regional_lookup=None, default_region="Surrey Central"):
        logger.info('Running "preprocessing_A" function...')
        ### Clip negative values to 0
        clip_cols = ['sales_dollars', 'extended_cost', 'units_sales', 'points_earned', 'points_redeemed']
        df[clip_cols] = np.clip(df[clip_cols],a_min=0,a_max=None)

        ### Embed week number
        week_in_year = 52
        df['week_in_year_cos'] = np.cos((2*np.pi*df['week_number'].values)/week_in_year)
        df['week_in_year_sin'] = np.sin((2*np.pi*df['week_number'].values)/week_in_year)
        df = df.drop(labels=['year_number','week_number'],axis=1)

        if regional_lookup is None:
            regional_lookup = self.create_regional_lookup(df)

        ### Convert regional rollup to index and check if new regions were added
        df = (df.merge(regional_lookup,
                       on='divisional_regional_rollup',
                       how= 'left')
            )
        fltr_nan =  df["div_reg_index"].isna()
        if fltr_nan.sum() > 0:
            self.regions_added = df[fltr_nan]["divisional_regional_rollup"].unique().tolist()
        df = df.drop(['divisional_regional_rollup'], axis=1)

        # Replace div_reg_index for regions which do not appear in regional_lookup with the default_region's index
        default_region_id = regional_lookup.loc[regional_lookup["divisional_regional_rollup"] == default_region,
                                                "div_reg_index"].iloc[0]
        df.loc[fltr_nan, "div_reg_index"] = default_region_id

        # extra index because 'mask_value' effectively adds to embedding dimension:
        div_reg_unique_count = df['div_reg_index'].value_counts().shape[0] + 1

        ### Create id index based on hhn, smallest to largest
        df_hhn_list = pd.DataFrame(df['hhn'].unique(),columns = ['hhn']).sort_values(by=['hhn']).reset_index(drop=True).reset_index()
        df_hhn_list.columns = ['id','hhn']

        ### Attach hhn index to df
        df = df.merge(df_hhn_list, on = 'hhn', how = 'inner')
        ## Create a lookup for results
        lookup_hhn_id = df_hhn_list.copy().values

        ### Cast types
        for c,dtype in self.column_types.items():
            df[c] = df[c].astype(dtype)

        # time_int seems not to be used?
        df = df.assign(time_int = (pd.to_datetime(df['week_ending_date']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

        return df, regional_lookup, div_reg_unique_count, lookup_hhn_id


    def subsample(self, df, lookup_hhn_id=None, quantile_choice = 0.6):
        """
        Kwargs:
            lookup_hhn_id (2d np.array): two columns corresponding to idx and hhn respectively (this is the mapping from row number of df to hhn - should be subsampled in the same way as df)
        Returns:
            if lookup_hhn_id is None:
                subsampled_df (pd.Dataframe)
            else:
                df_sampled, lookup_hhn_id_sampled (pd.DataFrame, np.array)
        """
        logger.info('Undersampling training data...')
        #### Subsamples very infrequent customers to improve training #####

        df_hhn_count = df.groupby(['hhn']).agg(
            num_active_week = pd.NamedAgg(column = 'hhn', aggfunc='count')
        ).reset_index()

        ### get total active week counts
        df_hhn_count['num_active_week_set4'] = np.floor((df_hhn_count['num_active_week']-1)/4.0)
        #bin_count_samplecutoff = np.floor(df_hhn_count['num_active_week_set4'].value_counts().quantile(.6))

        ### get the .60 quantile to sample for a relatively even sample
        df_setcount = pd.DataFrame(
            df_hhn_count['num_active_week_set4'].value_counts()
            ).reset_index()
        df_setcount.columns = ['num_active_week_set4','hh_count']
        bin_count_samplecutoff = np.floor(df_setcount['hh_count'].quantile(quantile_choice))

        var1 = (df_setcount['hh_count'] > bin_count_samplecutoff)
        bins_need_sampling = df_setcount[var1].copy()
        bins_need_sampling['need_sample_flag'] = 1

        ### flag those that have more than the quantile amount and needs to be sampled from
        df_hhn_count = pd.merge(left = df_hhn_count, right = bins_need_sampling, on =['num_active_week_set4'], how = 'left')
        df_hhn_count['need_sample_flag'] = df_hhn_count['need_sample_flag'].fillna(0).astype(np.int8)

        ### sample from this group
        varneedsample = df_hhn_count['need_sample_flag'] == 1
        sampled_group = (df_hhn_count[varneedsample]
                         .groupby(['num_active_week_set4'])
                         .apply(lambda x: x.sample(int(bin_count_samplecutoff), random_state=123)) )

        keep_only_hhn = pd.concat(
            [pd.DataFrame(sampled_group['hhn']).reset_index(drop=True),
             df_hhn_count[df_hhn_count['need_sample_flag'] == 0][['hhn']]]
            )

        df = pd.merge(left = df , right = keep_only_hhn, how = 'inner', on = ['hhn'])
        if lookup_hhn_id is not None:
            lookup_hhn_df = pd.DataFrame(lookup_hhn_id, columns=['id', 'hhn'])
            lookup_hhn_df = pd.merge(left=lookup_hhn_df,
                                     right=keep_only_hhn, how='inner', on=['hhn'])
            return df, lookup_hhn_df.values
        else:
            return df


    def get_feature_matrix(self, df, feature_cols,
                           min_date, max_date,
                           mask_value=np.nan, verbose=False):
        logger.info('Pivoting and padding df into feature matrix..')

        date_column_index = pd.date_range(start=min_date,end=max_date,freq='7D')

        x_features = []
        for c in feature_cols:
            if verbose:
                logger.info("Formatting {:s}".format(c))
            dfloc = df.pivot_table(columns=['week_ending_date'],
                                   index='id',
                                   values=c)\
                      .reindex(columns=date_column_index)
            # calculate nan shift (find column index of first valid date)
            nan_padding = np.argmin(pd.isnull(dfloc).values,axis=1)
            # Row by row, format data
            data = []
            for ii,row in enumerate(dfloc.itertuples(index=False,name=None)):
                # Pad with zeros on min valid date, then append nans equal to number of non-valid dates before first valid date
                data.append(np.pad(np.nan_to_num(row[nan_padding[ii]:],0),
                                   pad_width=(0,nan_padding[ii]),
                                   mode='constant',
                                   constant_values=mask_value).tolist()) #constant_values=np.nan).tolist())
            x_features.append(np.expand_dims(np.array(data),axis=-1))
        return np.concatenate(x_features,axis=2)


    def drop_n_last_timesteps(self, padded,n_timesteps_to_drop,remove_empty_seqs = True):
        # brutal method: simply right align all tensors and simply cut off the last n_timesteps_to_drop
        """
        Input:
            'padded':  left-aligned array of shape (n_hhn, n_timesteps, n_features), padded on the right with np.nan
            'n_timesteps_to_drop': exactly what it sounds like
        Returns:
            array of shape (n_hhn*, n_timesteps - n_timesteps_to_drop, n_features)
            [*possibly reduced if remove_empty_seqs=True]
        """
        n_timesteps = padded.shape[1]
        padded      = wtte.left_pad_to_right_pad(
                           wtte.right_pad_to_left_pad(
                                padded)[:,:(n_timesteps-n_timesteps_to_drop)]
                           )

        if remove_empty_seqs:
            seq_lengths = wtte.get_padded_seq_lengths(padded)
            padded = padded[seq_lengths>0]
        return padded


    def prep_tensors(self,x,events):
        # 0. calculate time to event and censoring indicators.
        y  = np.ones([events.shape[0],events.shape[1],2])
        y[:,:,0] = wtte.padded_events_to_tte(np.squeeze(events),discrete_time=True)
        y[:,:,1] = wtte.padded_events_to_not_censored(np.squeeze(events),discrete_time=True)

        # 1. Disalign features and targets otherwise truth is leaked.
        # 2. drop first timestep (that we now dont have features for)
        # 3. nan-mask the last timestep of features. (that we now don't have targets for)
        #### Comment added: "3." seems irrelevant here. We are disaligning the targets by appropriate slicing. 
        #### No need for additional masking
        events = events[:,1:,]
        y  = y[:,1:]
        x  = np.roll(x, shift=1, axis=1)[:,1:,] # equivalent to x = x[:-1, ...] ?
        x  = x + 0*np.expand_dims(events,-1)  # this seems pointless
        return x,y,events


    def nanmask_to_keras_mask(self,x,y,mask_value,tte_mask):
        """nanmask to keras mask.
            :param float mask_value: Use some improbable telltale value
                                    (but not nan-causing)
            :param float tte_mask: something that wont NaN the loss-function
        """
        logger.info("Preparing target tensors, masking inputs...")
        # Use some improbable telltale value (but not nan-causing)
        x[:,:,:][np.isnan(x)] = mask_value
        y[:,:,0][np.isnan(y[:,:,0])] = tte_mask  #
        y[:,:,1][np.isnan(y[:,:,1])] = 0.5
        sample_weights = (x[:,:,0]!=mask_value)*1.
        return x,y,sample_weights


    def get_events(self, x):
        logger.info('Creating "events" tensor...')
        # Get events mask (assuming first feature is the event indicator)
        events = (x[:,:,0]>0).copy()
        nanmask = np.isnan(x[:,:,0])
        events = np.where(nanmask, np.nan, events)
        return events


    def rescale(self, x, first_n_cols_to_rescale=12):
        logger.info('Rescaling inputs...')
        x[:,:,:first_n_cols_to_rescale] = np.log(1.0 + x[:,:,:first_n_cols_to_rescale])
        return x


    def generate_train_tensors(self, x, events, frac_timesteps_to_cut=0.11, n_testing_timesteps=None, remove_empty_seqs=True):
        logger.info('Slicing off training tensors...')
        n_timesteps = x.shape[1]

        if frac_timesteps_to_cut is not None:
            ## Hide frac_timesteps_to_cut of the last timesteps and keep them for testing
            n = np.floor(n_timesteps*frac_timesteps_to_cut).astype(int)

            if n_testing_timesteps is not None:
                assert n==n_testing_timesteps, 'Both "frac_timesteps_to_cut" and "n_testing_timesteps" kwargs were passed, but their values are incompatible'
            else:
                n_testing_timesteps = n

        # create training data
        x_train      = self.drop_n_last_timesteps(x,n_testing_timesteps,
                                                  remove_empty_seqs=remove_empty_seqs)
        events_train = self.drop_n_last_timesteps(events,n_testing_timesteps,
                                                  remove_empty_seqs=remove_empty_seqs)

        ## Do the necessary preparations of the tensors.
        x_train,y_train,events_train = self.prep_tensors(x_train,events_train)
        return x_train, y_train, events_train


    def preprocessing_B(self, x, y, events,
                        x_train, y_train, events_train,
                        mask_value=0.133713371337,
                        to_keep=4,
                        remove_empty_seqs=True):
        logger.info('Running preprocessing_B function...')

        seq_lengths = np.count_nonzero(~np.isnan(events), axis=1)
        seq_lengths_train = np.count_nonzero(~np.isnan(events_train), axis=1)

        # Used for initialization of alpha-bias:
        tte_mean_train = np.nanmean(y_train[:,:,0])
        mean_u = np.nanmean(y_train[:,:,1])

        logger.info('x_train size {:.2f} mb'.format(x_train.nbytes*1e-6))

        x_train,y_train,sample_weights_train = self.nanmask_to_keras_mask(
            x_train, y_train, mask_value, tte_mean_train)


        # Set weights to 0s except some non-nan timesteps immediately after testset ends.
        trainset_end = x_train.shape[1]
        # to_keep = 5 # Only eval first _ timesteps of testset. `to_keep=1` even more realistic if we retrain model nightly.
        n_to_drop = x.shape[1]-(trainset_end+to_keep)

        x_valid = self.drop_n_last_timesteps(x.copy(),n_to_drop,
                                             remove_empty_seqs=remove_empty_seqs)
        y_valid = self.drop_n_last_timesteps(y.copy(),n_to_drop,
                                             remove_empty_seqs=remove_empty_seqs)

        sample_weights_valid = y_valid[...,0]*0+1 # 1 if not nan
        sample_weights_valid = wtte.right_pad_to_left_pad(sample_weights_valid)
        sample_weights_valid[:,:trainset_end] = sample_weights_valid[:,:trainset_end]*0
        sample_weights_valid = wtte.left_pad_to_right_pad(sample_weights_valid)
        #timeline_plot(sample_weights_valid,'validation set weights','Greys')

        x_valid,y_valid,_ = self.nanmask_to_keras_mask(x_valid,y_valid,mask_value,tte_mean_train)

        # If there's zero-weight obs the batched model.fit will return NaN loss
        sample_weights_valid[np.isnan(sample_weights_valid)] = 0
        m = sample_weights_valid.sum(1)>0
        sample_weights_valid = np.copy(sample_weights_valid[m])
        x_valid = x_valid[m]
        y_valid = y_valid[m]

        # Initialization value for alpha-bias
        init_alpha = -1.0/np.log(1.0-1.0/(tte_mean_train+1.0) )
        init_alpha = init_alpha/mean_u
        logger.info('init_alpha: {:.6e} mean uncensored train: {:.6e}'.format(init_alpha, mean_u))

        logger.info('sample_weights_train.shape \t {:s}'.format(str(sample_weights_train.shape)))
        logger.info('sample_weights_valid.shape \t {:s}'.format(str(sample_weights_valid.shape))) # note: `to_keep` timesteps longer than train

        ### split x into two types of inputs

        x_train_features = x_train[:,:,0:15]
        x_train_store = np.squeeze(x_train[:,:,15:16])
        x_valid_features = x_valid[:,:,0:15]
        x_valid_store = np.squeeze(x_valid[:,:,15:16])

        x_features = x[:,:,0:15]
        x_store = np.squeeze(x[:,:,15:16])

        return (x_train_features, x_train_store, y_train, sample_weights_train,
                x_valid_features, x_valid_store, y_valid, sample_weights_valid,
                x_features, x_store, y,
                seq_lengths, init_alpha)
