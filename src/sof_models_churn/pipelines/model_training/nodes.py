import logging
from typing import Any, Dict
import numpy as np

from sof_models_churn.features.rnn_model import RNNModel
from sof_models_churn.features.ece_metric_scoring import ChurnPredictionMetrics
from sof_models_churn.common.wtte_utils import WeightWatcher

import keras.backend as K
from keras.callbacks import TerminateOnNaN,ReduceLROnPlateau,EarlyStopping

logger = logging.getLogger(__name__)

def get_model_from_dict(model_weights_and_params: Dict) -> RNNModel:
    params = {k: model_weights_and_params['params'][k] for k in ['div_reg_unique_count',
                                                                 'init_alpha',
                                                                 "lstm_1_units",
                                                                 "lstm_2_units",
                                                                 "lstm_final_units"]}
    weights = model_weights_and_params['weights']
    model = RNNModel.new_instance({**params})
    model._model.set_weights(weights)

    return model

def create_model(train_data: Dict, hyperparams: Dict[str, Any],parameters: Dict[str, Any]) -> Dict:
    '''Initialize the RNN model'''
    model_params = {'div_reg_unique_count': train_data['div_reg_unique_count'],
                            'init_alpha': train_data['init_alpha']}
    # Set value of the fuzz factor used in numeric expressions in Keras
    K.set_epsilon(10**(parameters['keras_fuzz_factor_exponent']))
    model = RNNModel.new_instance({**model_params,**hyperparams})
    params = model.params
    weights = model.model.get_weights()
    model_weights_and_params = {'params':params,'weights':weights}

    return model_weights_and_params

def pretrain_model(model_weights_and_params: dict,train_data: dict) -> Dict:

    """ Smart trick to initialize the alpha and beta efficiently.
                Improves numerical stability and speeds up training.
                Basically pretrain the output layer as a parametric model.
                Create a separate model for output that only takes zero-input.
                Then pretrain the layer
            """

    '''Initialize the RNN model'''
    model = get_model_from_dict(model_weights_and_params)

    pre_nanterminator = TerminateOnNaN()

    # Hack to only train the output layer bias; set its input to zero.
    n_input = model._model.layers[-2].input_shape[-1]
    x_tmp = np.zeros([train_data['x_train_features'].shape[0],
                        train_data['x_train_features'].shape[1],
                        n_input])
    x_tmp[
        train_data['x_train_features'][:,:,0]==model._params['mask_value']
        ] = model._params['mask_value']

    predicted = model._pretrainer.predict(x_tmp[:1,:1]).flatten()
    logger.info('(before  pretrain)\t alpha: {}\t beta : {}'.format(predicted[0],predicted[1]))

    prehistory = model._pretrainer.fit(x_tmp, train_data['y_train'],
                epochs=10, ## change to a smaller value for large dataset
                batch_size=2000,
                verbose=0,
                sample_weight = train_data['sample_weights_train'],
                callbacks=[pre_nanterminator])
    # pretraining_loss_curve = prehistory.history['loss']

    predicted = model._pretrainer.predict(x_tmp[:1,:1]).flatten()
    logger.info('(after  pretrain)\t alpha: {}\t beta : {}'.format(predicted[0],predicted[1]))

    params = model.params
    weights = model.model.get_weights()
    model_weights_and_params = {'params':params,'weights':weights}

    return model_weights_and_params

def train_model(model_weights_and_params: Dict[str,Any], train_data: Dict[str, Any],
                parameters: Dict[str, Any])-> Any:
    fit_params = dict(epochs=parameters['epochs'], batch_size=parameters['batch_size'])
    ##  these params (or a subset) could potentially be shoved in the config file:
    callback_params = {
        'reduce_lr_params':
            dict(monitor='val_loss', factor=0.5, patience=parameters['reduce_lr_patience'], verbose=1, mode='min', min_delta=1e-4, cooldown=5, min_lr=1e-5),
        'early_stopping_params':
            dict(monitor='val_loss',min_delta=0, patience=parameters['early_stop_patience'],
                    verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    }

    model = get_model_from_dict(model_weights_and_params)

    logger.info("*"*50)
    logger.info("Launching training run...")

    nanterminator = TerminateOnNaN()
    weightwatcher = WeightWatcher(per_batch=False,per_epoch= True)
    reduce_lr = ReduceLROnPlateau(**callback_params['reduce_lr_params'])
    earlystop_callback   = EarlyStopping(**callback_params['early_stopping_params'])

    history = model._model.fit(
                [train_data['x_train_features'], train_data['x_train_store']],
                train_data['y_train'],
                verbose=2,
                validation_data=([train_data['x_valid_features'],train_data['x_valid_store']],
                                 train_data['y_valid'],
                                 train_data['sample_weights_valid']),
                sample_weight = train_data['sample_weights_train'],
                callbacks=[nanterminator, weightwatcher, reduce_lr, earlystop_callback],
                **fit_params
                )

    # Compute evaluation metrics
    #   Here we assume that early_stopping was used with "restore_best_weights=True"
    #   so that min(loss) corresponds to the loss for the final model
    #   # to do: add in ECE metric
    lowest_idx = np.argmin(history.history['val_loss'])
    val_loss = history.history['val_loss'][lowest_idx]
    loss = history.history['loss'][lowest_idx]


    # update global class instance variable with values
    model.loss.append(loss)

    logger.info("-" * 100)
    # logger.info("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
    logger.info("Validation loss: {}".format(val_loss))
    logger.info('Training loss  : {}'.format(loss))
    logger.info("*"*50)

    params = model.params
    weights = model._model.get_weights()
    model_weights_and_params = {'params':params,'weights':weights}

    return model_weights_and_params, lowest_idx

def evaluate_model(model_weights_and_params:dict,test_data: dict, parameters:dict) -> Any:

    model = get_model_from_dict(model_weights_and_params)

    n_steps = parameters["n_testing_timesteps"]
    consider_last_event = parameters["consider_last_event"]
    min_box_width = parameters["metrics_min_box_width"]

    # Evaluate on test set:
    predicted = model._model.predict([test_data['x_valid_features'], test_data['x_valid_store']])
    metric_dict = ChurnPredictionMetrics.evaluate_metrics(predicted=predicted,
                                    y=test_data['y_valid'], n_steps=n_steps,
                                    min_box_width=min_box_width,
                                    consider_last_event=consider_last_event)

    return metric_dict