import time
from keras.callbacks import Callback
from tensorflow.python.keras import backend as K

class LoggingCallback(Callback):
    """LoggingCallback

    LoggingCallback is a customized keras callback, which could be used to log epoch information during model training using `logging` function.

    Args:
      logger: `logging.Logger` instance. If logger is left as None, it will go back to `print` function.
    """

    def __init__(self, logger=None):
        Callback.__init__(self)
        if logger is None:
            self.print_fcn = print
        else:
            self.print_fcn = logger.info

    def on_epoch_begin(self,epoch,logs={}):
        self.starttime=time.time()
        self.old_lr = float(K.get_value(self.model.optimizer.lr))
        

    def on_epoch_end(self, epoch, logs={}):
        
        # On epoch end, all the callbacks require a `on_epoch_end` action will be executed 
        # by the order they were added in the callback list, 
        # this `LoggingCallback` should be added last, 
        # it makes sure not only the epoch information from logs can be logged by logging function, 
        # but also the callbacks outputs can also be logged. 
        # More about the callbacks execution order refer to 
        # https://stackoverflow.com/questions/60858783/keras-callback-execution-order.

        time_diff=time.time()-self.starttime
        self.new_lr = float(K.get_value(self.model.optimizer.lr))

        msg = "Epoch: {:05d}, time: {} s, {}".format(epoch, time_diff, ", ".join("{}: {}".format(k, v) for k, v in logs.items()))
        
        self.print_fcn(msg)

        if self.model.stop_training:
            self.print_fcn('Epoch: %05d, early stopping' % (epoch+1))

        if self.old_lr > self.new_lr:
            self.print_fcn('Epoch: %05d, ReduceLROnPlateau reducing learning '
                    'rate to %s.' % (epoch+1, self.new_lr))
            