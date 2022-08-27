import numpy as np
import tensorflow as tf
# from tensorflow import cast, int32

import sof_models_churn.common.wtte_utils as wtte

import keras.backend as K
from keras.layers import Dense, LSTM, GRU, Masking, Lambda, Input, Embedding, Concatenate,\
    GaussianDropout, TimeDistributed, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam

#####################################################################################
default_params = {
    "n_features": 15 # x_train_features.shape[-1]
    ,"embedding_out_shape": 5
    ,"init_alpha": 4.466198
    ,"max_beta_value": 5.0
    # Stability heuristic: scale by log-number of pre-output layer inputs:
    ,"scalefactor": 1/np.log(15)
    ,"mask_value": 0.133713371337
    ,"div_reg_unique_count": 29
    ,"recurrent_dropout": 0
    ,"lstm_activation": 'tanh'
    ,"lstm_1_units": 5
    ,"lstm_2_units": 30
    ,"lstm_final_units":5
    ,"bn_momentum": 0.95
    ,"bn_epsilon":0.01
    ,"gaussian_dropout": 0.05
    ,"learning_rate": 0.01
    ,"return_sequences": True
    }

#####################################################################################

class RNNModel():
    """
    Class to construct RNN model for WTTE churn prediction
    """
    # class wide variables common to all instances
    # that keep track of cumulative estimators and rmse
    # so we can plot the results to see how it fares w.r.t
    # rmse
    ece = []
    loss = []
    random_seed = 1

    def __init__(self,
                 params={}
                 ):
        """
        Constructor for the RNN model
        :param params: dictionary containing RNN hyperparameters
        :param return_sequences: bool. Whether model should return 
        full sequence or just final value. Use True for training, 
        False for prediction.
        """
        np.random.seed(RNNModel.random_seed)
        self._params = default_params
        self._params.update(params)

        # construct keras model
        inp_w1 = Input(shape=(None,self._params['n_features']),
                       name='input_1')
        inp_store = Input(shape=(None,),
                          name='store_input')

        mask1 = Masking(mask_value=self._params['mask_value'],
                        input_shape=(None,None))
        w2 = mask1(inp_store)
        w2 = Lambda(lambda t: K.cast(t, "int32"), #cast(t,dtype=int32),
                    name='cast_to_int')(w2)
        w2 = Embedding(input_dim=self._params['div_reg_unique_count'],
                       output_dim=self._params['embedding_out_shape'],
                       input_length=None)(w2)
        w1 = mask1(inp_w1)

        # ## TanH input dense/GRU layer activation squashes outliers -> small generalizability effect
        w1 = Concatenate(axis=-1,name='merge_masked_input')([w1, w2])
        w1 = LSTM(self._params['lstm_1_units'],
                  activation=self._params['lstm_activation'],
                  return_sequences=True,
                  recurrent_dropout=self._params['recurrent_dropout'],
                  unroll=False)(w1)
        w1 = BatchNormalization(axis=-1,
                                momentum=self._params['bn_momentum'],
                                epsilon=self._params['bn_epsilon']
                                )(w1)
        w1 = LSTM(self._params['lstm_2_units'],
                  activation=self._params['lstm_activation'],
                  return_sequences=True,
                  recurrent_dropout=self._params['recurrent_dropout'],
                  unroll=False)(w1)
        w1 = BatchNormalization(axis=-1,
                                momentum=self._params['bn_momentum'],
                                epsilon=self._params['bn_epsilon']
                                )(w1)
        w1 = GaussianDropout(self._params['gaussian_dropout'])(w1) # Recurrent dropout + Batchnorm doesn't jive well. This is alternative.(w1)

        ## GRU or Dense bottleneck-layer before output. Makes interesting embeddings.
        # model.add(TimeDistributed(Dense(5,activation='tanh')))
        w1 = LSTM(self._params['lstm_final_units'],
                  activation=self._params['lstm_activation'],
                  return_sequences=self._params['return_sequences'],
                  recurrent_dropout=self._params['recurrent_dropout'],
                  unroll=False)(w1)

        ## Wtte-RNN part
        if self._params['return_sequences']:
            w1 = TimeDistributed(Dense(2))(w1)
        else:
            w1 = Dense(2)(w1)
        w1 = Lambda(wtte.output_lambda, arguments={
            "init_alpha":self._params['init_alpha'],
            "max_beta_value":self._params['max_beta_value'], ### adjusted from 2.0
            "scalefactor":self._params['scalefactor'], ## 15 seems to work best
                               }
                    )(w1)

        m = Model(inputs = [inp_w1, inp_store],
                  outputs = [w1])

        loss_fun = wtte.loss(kind='discrete',
                             reduce_loss=False,
                             clip_prob=1e-5).loss_function
        m.compile(loss=loss_fun,
                      optimizer=Adam(lr=self._params['learning_rate'],
                                     clipvalue=0.5),
                      sample_weight_mode='temporal',
                      weighted_metrics=[])

        self._model = m
        self._pretrainer =  self._construct_pretrainer(self._model, loss_fun, self._params['mask_value'])

    def _construct_pretrainer(self, main_model, loss_fun=None,
                              mask_value=0.133713371337):
        # define mini-model out of the output layers of main_model. Used for pre-training
        n_input = main_model.layers[-2].input_shape[-1]
        tmp_model = Sequential()
        tmp_model.add(Masking(mask_value=mask_value,
                              input_shape=(None, n_input)))
        tmp_model.add(main_model.layers[-2])
        tmp_model.add(main_model.layers[-1])

        tmp_model.compile(loss=loss_fun, optimizer=Adam(lr=1e-2),
                          sample_weight_mode='temporal')
        return tmp_model

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    @property
    def model(self):
        """
        Getter to return the model created
        :return: handle or instance of the RandomForestReqgressor
        """
        return self._model

    @property
    def params(self):
      """
      Getter for model parameters
      """
      return self._params
