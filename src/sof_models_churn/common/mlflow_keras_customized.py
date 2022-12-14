from mlflow.keras import *
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _get_flavor_configuration
import os
from distutils.version import LooseVersion

####  "Private" variables from mlflow.keras, not picked up by the above import:  ####
# File name to which custom objects cloudpickle is saved - used during save and load
_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.cloudpickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"
_KERAS_SAVE_FORMAT_PATH = "save_format.txt"
# File name to which keras model is saved
_MODEL_SAVE_PATH = "model"
# Conda env subpath when saving/loading model
_CONDA_ENV_SUBPATH = "conda.yaml"



def get_model_path(model_uri, **kwargs):
    """
    :param model_uri: The location, in URI format, of the MLflow model. For example:
                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    keras_model_artifacts_path = os.path.join(
        local_model_path, flavor_conf.get("data", _MODEL_SAVE_PATH)
    )
    model_path = os.path.join(keras_model_artifacts_path, _MODEL_SAVE_PATH)
    return model_path


def save_model(
    keras_model,
    path,
    save_weights_only=True,  # this is the custom piece that we have added (see below)
    conda_env=None,
    mlflow_model=None,
    custom_objects=None,
    keras_module=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    **kwargs
):
    """
    Save a Keras model to a path on the local file system.

    :param keras_model: Keras model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the
                      dependencies contained in :func:`get_default_conda_env()`. If
                      ``None``, the default :func:`get_default_conda_env()` environment is
                      added to the model. The following is an *example* dictionary
                      representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'keras=2.2.4',
                                'tensorflow=1.8.0'
                            ]
                        }
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param keras_module: Keras module to be used to save / load the model
                         (``keras`` or ``tf.keras``). If not provided, MLflow will
                         attempt to infer the Keras module based on the given model.
    :param kwargs: kwargs to pass to ``keras_model.save`` method.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.

    .. code-block:: python
        :caption: Example

        import mlflow
        # Build, compile, and train your model
        keras_model = ...
        keras_model_path = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
        # Save the model as an MLflow Model
        mlflow.keras.save_model(keras_model, keras_model_path)
    """

    if keras_module is None:

        def _is_plain_keras(model):
            try:
                import keras

                if LooseVersion(keras.__version__) < LooseVersion("2.2.0"):
                    import keras.engine

                    return isinstance(model, keras.engine.Model)
                else:
                    # NB: Network is the first parent with save method
                    import keras.engine.network

                    return isinstance(model, keras.engine.network.Network)
            except ImportError:
                return False

        def _is_tf_keras(model):
            try:
                # NB: Network is not exposed in tf.keras, we check for Model instead.
                import tensorflow.keras.models

                return isinstance(model, tensorflow.keras.models.Model)
            except ImportError:
                return False

        if _is_plain_keras(keras_model):
            keras_module = importlib.import_module("keras")
        elif _is_tf_keras(keras_model):
            keras_module = importlib.import_module("tensorflow.keras")
        else:
            raise MlflowException(
                "Unable to infer keras module from the model, please specify "
                "which keras module ('keras' or 'tensorflow.keras') is to be "
                "used to save and load the model."
            )
    elif type(keras_module) == str:
        keras_module = importlib.import_module(keras_module)

    # check if path exists
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))

    # construct new data folder in existing path
    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # save custom objects if there are custom objects
    if custom_objects is not None:
        _save_custom_objects(data_path, custom_objects)

    # save keras module spec to path/data/keras_module.txt
    with open(os.path.join(data_path, _KERAS_MODULE_SPEC_PATH), "w") as f:
        f.write(keras_module.__name__)

    # Use the SavedModel format if `save_format` is unspecified
    save_format = kwargs.get("save_format", "tf")

    # save keras save_format to path/data/save_format.txt
    with open(os.path.join(data_path, _KERAS_SAVE_FORMAT_PATH), "w") as f:
        f.write(save_format)

    #####################################################
    # New code for saving weights only:
    if save_weights_only:
        # save keras model
        # To maintain prior behavior, when the format is HDF5, we save
        # with the h5 file extension. Otherwise, model_path is a directory
        # where the saved_model.pb will be stored (for SavedModel format)
        file_extension = ".h5" if save_format == "h5" else ""
        model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)
        model_path = os.path.join(path, model_subpath) + file_extension
        if path.startswith("/dbfs/"):
            # The Databricks Filesystem uses a FUSE implementation that does not support
            # random writes. It causes an error.
            with tempfile.NamedTemporaryFile(suffix=".h5") as f:
                keras_model.save_weights(f.name, **kwargs)
                f.flush()  # force flush the data
                shutil.copyfile(src=f.name, dst=model_path)
        else:
            # keras_model.save_weights(model_path, **kwargs)
            keras_model.save_weights(model_path)
    else:
    #####################################################
        # save keras model
        # To maintain prior behavior, when the format is HDF5, we save
        # with the h5 file extension. Otherwise, model_path is a directory
        # where the saved_model.pb will be stored (for SavedModel format)
        file_extension = ".h5" if save_format == "h5" else ""
        model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)
        model_path = os.path.join(path, model_subpath) + file_extension
        if path.startswith("/dbfs/"):
            # The Databricks Filesystem uses a FUSE implementation that does not support
            # random writes. It causes an error.
            with tempfile.NamedTemporaryFile(suffix=".h5") as f:
                keras_model.save(f.name, **kwargs)
                f.flush()  # force flush the data
                shutil.copyfile(src=f.name, dst=model_path)
        else:
            keras_model.save(model_path, **kwargs)

    # update flavor info to mlflow_model
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        keras_module=keras_module.__name__,
        keras_version=keras_module.__version__,
        save_format=save_format,
        data=data_subpath,
    )

    # save conda.yaml info to path/conda.yml
    if conda_env is None:
        conda_env = get_default_conda_env(
            include_cloudpickle=custom_objects is not None, keras_module=keras_module
        )
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, _CONDA_ENV_SUBPATH), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # append loader_module, data and env data to mlflow_model
    pyfunc.add_to_model(
        mlflow_model, loader_module="mlflow.keras", data=data_subpath, env=_CONDA_ENV_SUBPATH
    )

    # save mlflow_model to path/MLmodel
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))



########################################################################
#  Autologging  (tweaked to exclude `sample_weight` from logged params)
########################################################################

@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models=True,
    skip_params=None, # This is the custom kwarg that we have added
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    # pylint: disable=E0611
    """
    Enables (or disables) and configures autologging from Keras to MLflow. Autologging captures
    the following information:

    **Metrics** and **Parameters**
     - Training loss; validation loss; user-specified metrics
     - Metrics associated with the ``EarlyStopping`` callbacks: ``stopped_epoch``,
       ``restored_epoch``, ``restore_best_weight``, ``last_epoch``, etc
     - ``fit()`` or ``fit_generator()`` parameters; optimizer name; learning rate; epsilon
     - ``fit()`` or ``fit_generator()`` parameters associated with ``EarlyStopping``: ``min_delta``,
       ``patience``, ``baseline``, ``restore_best_weights``, etc
    **Artifacts**
     - Model summary on training start
     - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Keras model) on training end

    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.keras
        # Build, compile, enable autologging, and train your model
        keras_model = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        # autolog your metrics, parameters, and model
        mlflow.keras.autolog()
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val))

    ``EarlyStopping Integration with Keras AutoLogging``

    MLflow will detect if an ``EarlyStopping`` callback is used in a ``fit()`` or
    ``fit_generator()`` call, and if the ``restore_best_weights`` parameter is set to be ``True``,
    then MLflow will log the metrics associated with the restored model as a final, extra step.
    The epoch of the restored model will also be logged as the metric ``restored_epoch``.
    This allows for easy comparison between the actual metrics of the restored model and
    the metrics of other models.

    If ``restore_best_weights`` is set to be ``False``, then MLflow will not log an additional step.

    Regardless of ``restore_best_weights``, MLflow will also log ``stopped_epoch``,
    which indicates the epoch at which training stopped due to early stopping.

    If training does not end due to early stopping, then ``stopped_epoch`` will be logged as ``0``.

    MLflow will also log the parameters of the ``EarlyStopping`` callback,
    excluding ``mode`` and ``verbose``.

    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the Keras autologging integration. If ``False``,
                    enables the Keras autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      keras that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during Keras
                   autologging. If ``False``, show all events and warnings during Keras
                   autologging.
    """
    import keras

    def getKerasCallback(metrics_logger):
        class __MLflowKerasCallback(keras.callbacks.Callback, metaclass=ExceptionSafeClass):
            """
            Callback for auto-logging metrics and parameters.
            Records available logs after each epoch.
            Records model structural information as params when training begins
            """

            def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
                try_mlflow_log(mlflow.log_param, "num_layers", len(self.model.layers))
                try_mlflow_log(
                    mlflow.log_param, "optimizer_name", type(self.model.optimizer).__name__
                )
                if hasattr(self.model.optimizer, "lr"):
                    lr = (
                        self.model.optimizer.lr
                        if type(self.model.optimizer.lr) is float
                        else keras.backend.eval(self.model.optimizer.lr)
                    )
                    try_mlflow_log(mlflow.log_param, "learning_rate", lr)
                if hasattr(self.model.optimizer, "epsilon"):
                    epsilon = (
                        self.model.optimizer.epsilon
                        if type(self.model.optimizer.epsilon) is float
                        else keras.backend.eval(self.model.optimizer.epsilon)
                    )
                    try_mlflow_log(mlflow.log_param, "epsilon", epsilon)

                sum_list = []
                self.model.summary(print_fn=sum_list.append)
                summary = "\n".join(sum_list)
                tempdir = tempfile.mkdtemp()
                try:
                    summary_file = os.path.join(tempdir, "model_summary.txt")
                    with open(summary_file, "w") as f:
                        f.write(summary)
                    try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
                finally:
                    shutil.rmtree(tempdir)

            def on_epoch_end(self, epoch, logs=None):
                if not logs:
                    return
                metrics_logger.record_metrics(logs, epoch)

            def on_train_end(self, logs=None):
                if log_models:
                    try_mlflow_log(log_model, self.model, artifact_path="model")

            # As of Keras 2.4.0, Keras Callback implementations must define the following
            # methods indicating whether or not the callback overrides functions for
            # batch training/testing/inference
            def _implements_train_batch_hooks(self):
                return False

            def _implements_test_batch_hooks(self):
                return False

            def _implements_predict_batch_hooks(self):
                return False

        return __MLflowKerasCallback()

    def _early_stop_check(callbacks):
        if LooseVersion(keras.__version__) < LooseVersion("2.3.0") or LooseVersion(
            keras.__version__
        ) >= LooseVersion("2.4.0"):
            es_callback = keras.callbacks.EarlyStopping
        else:
            es_callback = keras.callbacks.callbacks.EarlyStopping
        for callback in callbacks:
            if isinstance(callback, es_callback):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            earlystopping_params = {
                "monitor": callback.monitor,
                "min_delta": callback.min_delta,
                "patience": callback.patience,
                "baseline": callback.baseline,
                "restore_best_weights": callback.restore_best_weights,
            }
            try_mlflow_log(mlflow.log_params, earlystopping_params)

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  # pylint: disable=W0703
            return None

    def _log_early_stop_callback_metrics(callback, history, metrics_logger):
        if callback:
            callback_attrs = _get_early_stop_callback_attrs(callback)
            if callback_attrs is None:
                return
            stopped_epoch, restore_best_weights, patience = callback_attrs
            metrics_logger.record_metrics({"stopped_epoch": stopped_epoch})
            # Weights are restored only if early stopping occurs
            if stopped_epoch != 0 and restore_best_weights:
                restored_epoch = stopped_epoch - max(1, patience)
                metrics_logger.record_metrics({"restored_epoch": restored_epoch})
                restored_index = history.epoch.index(restored_epoch)
                restored_metrics = {
                    key: history.history[key][restored_index] for key in history.history.keys()
                }
                # Checking that a metric history exists
                metric_key = next(iter(history.history), None)
                if metric_key is not None:
                    last_epoch = len(history.history[metric_key])
                    metrics_logger.record_metrics(restored_metrics, last_epoch)

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        log_fn_args_as_params(original, args, kwargs, unlogged_params)
        early_stop_callback = None

        # Checking if the 'callback' argument of the function is set
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            mlflowKerasCallback = getKerasCallback(metrics_logger)
            if len(args) > callback_arg_index:
                tmp_list = list(args)
                early_stop_callback = _early_stop_check(tmp_list[callback_arg_index])
                tmp_list[callback_arg_index] += [mlflowKerasCallback]
                args = tuple(tmp_list)
            elif kwargs.get("callbacks"):
                early_stop_callback = _early_stop_check(kwargs["callbacks"])
                kwargs["callbacks"] += [mlflowKerasCallback]
            else:
                kwargs["callbacks"] = [mlflowKerasCallback]

            try_mlflow_log(_log_early_stop_callback_params, early_stop_callback)

            history = original(self, *args, **kwargs)

            try_mlflow_log(
                _log_early_stop_callback_metrics, early_stop_callback, history, metrics_logger
            )

        return history

    ##################################################################################################
    # Modified autolog code
    ##################################################################################################
    unlogged_params_OVERWRITE = ["self", "x", "y", "callbacks", "validation_data", "verbose"]
    if skip_params is not None:
        if not isinstance(skip_params, list): 
            skip_params = [skip_params]
        unlogged_params_OVERWRITE = unlogged_params_OVERWRITE + skip_params
    def fit(original, self, *args, **kwargs):
        # unlogged_params = ["self", "x", "y", "callbacks", "validation_data", "verbose"]
        unlogged_params = unlogged_params_OVERWRITE
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 5)

    def fit_generator(original, self, *args, **kwargs):
        """
        NOTE: `fit_generator()` is deprecated in Keras >= 2.4.0 and simply wraps `fit()`.
        To avoid unintentional creation of nested MLflow runs caused by a patched
        `fit_generator()` method calling a patched `fit()` method, we only patch
        `fit_generator()` in Keras < 2.4.0.
        """
        # unlogged_params = ["self", "generator", "callbacks", "validation_data", "verbose"]
        unlogged_params = unlogged_params_OVERWRITE        
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 4)
    ##################################################################################################

    safe_patch(FLAVOR_NAME, keras.Model, "fit", fit, manage_run=True)
    # `fit_generator()` is deprecated in Keras >= 2.4.0 and simply wraps `fit()`.
    # To avoid unintentional creation of nested MLflow runs caused by a patched
    # `fit_generator()` method calling a patched `fit()` method, we only patch
    # `fit_generator()` in Keras < 2.4.0.
    if LooseVersion(keras.__version__) < LooseVersion("2.4.0"):
        safe_patch(FLAVOR_NAME, keras.Model, "fit_generator", fit_generator, manage_run=True)