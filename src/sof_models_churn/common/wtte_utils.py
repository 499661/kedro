import numpy as np
import pandas as pd
from six.moves import xrange

from math import log

import warnings
# import numpy as np

from keras import backend as K
from keras.callbacks import Callback

import logging
logger = logging.getLogger(__name__)


#############################################################
# tte_util  (utility fn's used in transforms further down)
#############################################################

def get_is_not_censored(is_event, discrete_time=True):
    """ Calculates non-censoring indicator `u` for one vector.
        :param array is_event: logical or numeric array indicating event.
        :param Boolean discrete_time: if `True`, last observation is conditionally censored.
    """
    n = len(is_event)
    is_not_censored = np.copy(is_event)

    if discrete_time:
        # Last obs is conditionally censored
        event_seen = is_event[-1]
        for i in reversed(xrange(n)):
            if is_event[i] and not event_seen:
                event_seen = is_event[i]
            is_not_censored[i] = event_seen
    else:
        # Last obs is always censored
        event_seen = False
        for i in reversed(xrange(n)):
            is_not_censored[i] = event_seen
            if is_event[i] and not event_seen:
                event_seen = is_event[i]

    return is_not_censored

def get_tte(is_event, discrete_time, t_elapsed=None):
    """ wrapper to calculate *Time To Event* for input vector.
        :param Boolean discrete_time: if `True`, use `get_tte_discrete`. If `False`, use `get_tte_continuous`.
    """
    if discrete_time:
        return get_tte_discrete(is_event, t_elapsed)
    else:
        return get_tte_continuous(is_event, t_elapsed)

def get_tte_discrete(is_event, t_elapsed=None):
    """Calculates discretely measured tte over a vector.
        :param Array is_event: Boolean array
        :param IntArray t_elapsed: integer array with same length as `is_event`. If none, it will use `xrange(len(is_event))`
        :return Array tte: Time-to-event array (discrete version)
        - Caveats
            tte[i] = numb. timesteps to timestep with event
            Step of event has tte = 0 \
           (event happened at time [t,t+1))
            tte[-1]=1 if no event (censored data)
    """
    n = len(is_event)
    tte = np.int32(is_event)
    stepsize = 1
    if t_elapsed is None:
        t_elapsed = xrange(n)

    t_next = t_elapsed[-1] + stepsize
    for i in reversed(xrange(n)):
        if is_event[i]:
            t_next = t_elapsed[i]
        tte[i] = t_next - t_elapsed[i]
    return tte


def get_tte_continuous(is_event, t_elapsed):
    """Calculates time to (pointwise measured) next event over a vector.
        :param Array is_event: Boolean array
        :param IntArray t_elapsed: integer array with same length as `is_event` that supports vectorized subtraction. If none, it will use `xrange(len(is_event))`
        :return Array tte: Time-to-event (continuous version)
        TODO::
            Should support discretely sampled, continuously measured TTE
        .. Caveats::
            tte[i] = time to *next* event at time t[i]
            (t[i] is exactly at event&/or query)
            tte[-1]=0 always
            (since last time is a *point*)
            Last datpoints are right censored.
    """
    n = len(is_event)
    if t_elapsed is None:
        t_elapsed = np.int32(xrange(n))

    t_next = t_elapsed[-1]
    # lazy initialization to autoinit if difftime
    tte = t_elapsed - t_next
    for i in reversed(xrange(n)):
        tte[i] = t_next - t_elapsed[i]
        if is_event[i]:
            t_next = t_elapsed[i]
    return tte


#############################################################
#  Transforms  (wtte.transforms)
#############################################################

def right_pad_to_left_pad(padded):
    """ Change right padded to left padded. """
    return _align_padded(padded, align_right=True)

def left_pad_to_right_pad(padded):
    """ Change left padded to right padded. """
    return _align_padded(padded, align_right=False)

def _align_padded(padded, align_right):
    """ (Internal function) Aligns nan-padded temporal arrays to the right (align_right=True) or left.
    :param Array padded: padded array
    :param align_right: Determines padding orientation (right or left). If `True`, pads to right direction.
    """
    padded = padded.copy()

    seq_lengths = get_padded_seq_lengths(padded)
    if len(padded.shape) == 2:
        # (n_seqs,n_timesteps)
        is_flat = True
        padded = np.expand_dims(padded, -1)
    elif len(padded.shape) == 3:
        # (n_seqs,n_timesteps,n_features)
        is_flat = False
    else:
        # (n_seqs,n_timesteps,...,n_features)
        logger.error('not yet implemented')
        # TODO

    n_seqs = padded.shape[0]
    n_timesteps = padded.shape[1]

    if align_right:
        for i in xrange(n_seqs):
            n = seq_lengths[i]
            if n > 0:
                padded[i, (n_timesteps - n):] = padded[i, :n]
                padded[i, :(n_timesteps - n)] = np.nan
    else:
        for i in xrange(n_seqs):
            n = seq_lengths[i]
            if n > 0:
                padded[i, :n, :] = padded[i, (n_timesteps - n):, :]
                padded[i, n:, :] = np.nan

    if is_flat:
        padded = np.squeeze(padded)

    return padded


def get_padded_seq_lengths(padded):
    """Returns the number of (seq_len) non-nan elements per sequence.
    :param padded: 2d or 3d tensor with dim 2 the time dimension
    """
    if np.isnan(padded).sum() == 0:
        logger.warning("All sequences have full length. Check if NaNs have been accidentally removed.")
    if len(padded.shape) == 2:
        # (n_seqs,n_timesteps)
        seq_lengths = np.count_nonzero(~np.isnan(padded), axis=1)
    elif len(padded.shape) == 3:
        # (n_seqs,n_timesteps,n_features,..)
        seq_lengths = np.count_nonzero(~np.isnan(padded[:, :, 0]), axis=1)
    else:
        logger.error('not yet implemented')
        # TODO

    return seq_lengths


def padded_events_to_tte(events, discrete_time, t_elapsed=None):
    """ computes (right censored) time to event from padded binary events.
    For details see `tte_util.get_tte`
    :param Array events: Events array.
    :param Boolean discrete_time: `True` when applying discrete time scheme.
    :param Array t_elapsed: Elapsed time. Default value is `None`.
    :return Array time_to_events: Time-to-event tensor.
    """
    seq_lengths = get_padded_seq_lengths(events)
    n_seqs = len(events)

    times_to_event = np.zeros_like(events)
    times_to_event[:] = np.nan

    t_seq = None
    for s in xrange(n_seqs):
        n = seq_lengths[s]
        if n > 0:
            event_seq = events[s, :n]
            if t_elapsed is not None:
                t_seq = t_elapsed[s, :n]

            times_to_event[s, :n] = get_tte(is_event=event_seq,
                                            discrete_time=discrete_time,
                                            t_elapsed=t_seq)

    if np.isnan(times_to_event).any():
        times_to_event[np.isnan(events)] = np.nan
    return times_to_event

def padded_events_to_not_censored(events, discrete_time):
    seq_lengths = get_padded_seq_lengths(events)
    n_seqs = events.shape[0]
    is_not_censored = events.copy()

    for i in xrange(n_seqs):
        if seq_lengths[i] > 0:
            is_not_censored[i][:seq_lengths[i]] = get_is_not_censored(
                events[i][:seq_lengths[i]], discrete_time)
    return is_not_censored




#############################################################
# Model-related utilities (loss, callback, etc)
#############################################################

def _keras_split(y_true, y_pred):
    """
        Everything is a hack around the y_true,y_pred paradigm.
    """
    y, u = _keras_unstack_hack(y_true)
    a, b = _keras_unstack_hack(y_pred)

    return y, u, a, b

def _keras_unstack_hack(ab):
    """Implements tf.unstack(y_true_keras, num=2, axis=-1).
       Keras-hack adopted to be compatible with Theano backend.
       :param ab: stacked variables
       :return a, b: unstacked variables
    """
    ndim = len(K.int_shape(ab))
    if ndim == 0:
        logger.error('can not unstack with ndim=0')
    else:
        a = ab[..., 0]
        b = ab[..., 1]
    return a, b

def loglik_discrete(y, u, a, b, epsilon=K.epsilon()):
    hazard0 = K.pow((y + epsilon) / a, b)
    hazard1 = K.pow((y + 1.0) / a, b)

    loglikelihoods = u * \
        K.log(K.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
    return loglikelihoods


def loglik_continuous(y, u, a, b, epsilon=K.epsilon()):
    ya = (y + epsilon) / a
    loglikelihoods = u * (K.log(b) + b * K.log(ya)) - K.pow(ya, b)
    return loglikelihoods


def output_lambda(x, init_alpha=1.0, max_beta_value=5.0, scalefactor=None,
                  alpha_kernel_scalefactor=None):
    """Elementwise (Lambda) computation of alpha and regularized beta.
        - Alpha:
            (activation)
            Exponential units seems to give faster training than
            the original papers softplus units. Makes sense due to logarithmic
            effect of change in alpha.
            (initialization)
            To get faster training and fewer exploding gradients,
            initialize alpha to be around its scale when beta is around 1.0,
            approx the expected value/mean of training tte.
            Because we're lazy we want the correct scale of output built
            into the model so initialize implicitly;
            multiply assumed exp(0)=1 by scale factor `init_alpha`.
        - Beta:
            (activation)
            We want slow changes when beta-> 0 so Softplus made sense in the original
            paper but we get similar effect with sigmoid. It also has nice features.
            (regularization) Use max_beta_value to implicitly regularize the model
            (initialization) Fixed to begin moving slowly around 1.0
        - Usage
            .. code-block:: python
                model.add(TimeDistributed(Dense(2)))
                model.add(Lambda(wtte.output_lambda, arguments={"init_alpha":init_alpha,
                                                        "max_beta_value":2.0
                                                       }))
        :param x: tensor with last dimension having length 2 with x[...,0] = alpha, x[...,1] = beta
        :param init_alpha: initial value of `alpha`. Default value is 1.0.
        :param max_beta_value: maximum beta value. Default value is 5.0.
        :param max_alpha_value: maxumum alpha value. Default is `None`.
        :type x: Array
        :type init_alpha: Float
        :type max_beta_value: Float
        :type max_alpha_value: Float
        :return x: A positive `Tensor` of same shape as input
        :rtype: Array
    """
    if max_beta_value is None or max_beta_value > 3:
        if K.epsilon() > 1e-07 and K.backend() == 'tensorflow':
            # TODO need to think this through lol
            message = "\
            Using tensorflow backend and allowing high `max_beta_value` may lead to\n\
            gradient NaN during training unless `K.epsilon()` is small.\n\
            Call `keras.backend.set_epsilon(1e-08)` to lower epsilon \
            "
            warnings.warn(message)
    if alpha_kernel_scalefactor is not None:
        message = "`alpha_kernel_scalefactor` deprecated in favor of `scalefactor` scaling both.\n Setting `scalefactor = alpha_kernel_scalefactor`"
        warnings.warn(message)
        scalefactor = alpha_kernel_scalefactor

    a, b = _keras_unstack_hack(x)

    if scalefactor is not None:
        # Done after due to theano bug.
        a, b = scalefactor * a, scalefactor * b

    # Implicitly initialize alpha:
    a = init_alpha * K.exp(a)

    if max_beta_value > 1.05:  # some value >>1.0
        # shift to start around 1.0
        # assuming input is around 0.0
        _shift = np.log(max_beta_value - 1.0)

        b = b - _shift

    b = max_beta_value * K.sigmoid(b)

    x = K.stack([a, b], axis=-1)

    return x





class Loss(object):
    """ Creates a keras WTTE-loss function.
        - Usage
            :Example:
            .. code-block:: python
               loss = wtte.Loss(kind='discrete').loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01))
               # And with masking:
               loss = wtte.Loss(kind='discrete',reduce_loss=False).loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01),
                              sample_weight_mode='temporal')
        .. note::
            With masking keras needs to access each loss-contribution individually.
            Therefore we do not sum/reduce down to scalar (dim 1), instead return a
            tensor (with reduce_loss=False).
        :param kind:  One of 'discrete' or 'continuous'
        :param reduce_loss:
        :param clip_prob: Clip likelihood to [log(clip_prob),log(1-clip_prob)]
        :param regularize: Deprecated.
        :param location: Deprecated.
        :param growth: Deprecated.
        :type reduce_loss: Boolean
    """

    def __init__(self,
                 kind,
                 reduce_loss=True,
                 clip_prob=1e-6,
                 regularize=False,
                 location=None,
                 growth=None):

        self.kind = kind
        self.reduce_loss = reduce_loss
        self.clip_prob = clip_prob

        if regularize == True or location is not None or growth is not None:
            raise DeprecationWarning('Directly penalizing beta has been found \
                                      to be unneccessary when using bounded activation \
                                      and clipping of log-likelihood.\
                                      Use this method instead.')

    def loss_function(self, y_true, y_pred):

        y, u, a, b = _keras_split(y_true, y_pred)
        if self.kind == 'discrete':
            loglikelihoods = loglik_discrete(y, u, a, b)
        elif self.kind == 'continuous':
            loglikelihoods = loglik_continuous(y, u, a, b)

        if self.clip_prob is not None:
            loglikelihoods = K.clip(loglikelihoods,
                log(self.clip_prob), log(1 - self.clip_prob))
        if self.reduce_loss:
            loss = -1.0 * K.mean(loglikelihoods, axis=-1)
        else:
            loss = -loglikelihoods

        return loss

# For backwards-compatibility
loss = Loss










class WeightWatcher(Callback):
    """Keras Callback to keep an eye on output layer weights.
        (under dev)
        Usage:
            weightwatcher = WeightWatcher(per_batch=True,per_epoch=False)
            model.fit(...,callbacks=[weightwatcher])
            weightwatcher.plot()
    """

    def __init__(self,
                 per_batch=False,
                 per_epoch=True
                 ):
        self.per_batch = per_batch
        self.per_epoch = per_epoch

    def on_train_begin(self, logs={}):
        self.a_weights_mean = []
        self.b_weights_mean = []
        self.a_weights_min = []
        self.b_weights_min = []
        self.a_weights_max = []
        self.b_weights_max = []
        self.a_bias = []
        self.b_bias = []

    def append_metrics(self):
        # Last two weightlayers in model

        output_weights, output_biases = self.model.get_weights()[-2:]

        a_weights_mean, b_weights_mean = output_weights.mean(0)
        a_weights_min, b_weights_min = output_weights.min(0)
        a_weights_max, b_weights_max = output_weights.max(0)

        a_bias, b_bias = output_biases

        self.a_weights_mean.append(a_weights_mean)
        self.b_weights_mean.append(b_weights_mean)
        self.a_weights_min.append(a_weights_min)
        self.b_weights_min.append(b_weights_min)
        self.a_weights_max.append(a_weights_max)
        self.b_weights_max.append(b_weights_max)
        self.a_bias.append(a_bias)
        self.b_bias.append(b_bias)

    def on_train_end(self, logs={}):
        if self.per_epoch:
            self.append_metrics()
        return

    def on_epoch_begin(self, epoch, logs={}):
        if self.per_epoch:
            self.append_metrics()
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        if self.per_batch:
            self.append_metrics()
        return

    def on_batch_end(self, batch, logs={}):
        if self.per_batch:
            self.append_metrics()
        return

    def plot(self):
        import matplotlib.pyplot as plt

        # Create axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(self.a_bias, color='b')
        ax1.set_xlabel('step')
        ax1.set_ylabel('alpha')

        ax2.plot(self.b_bias, color='r')
        ax2.set_ylabel('beta')

        # Change color of each axis
        def color_y_axis(ax, color):
            """Color your axes."""
            for t in ax.get_yticklabels():
                t.set_color(color)
            return None

        plt.title('biases')
        color_y_axis(ax1, 'b')
        color_y_axis(ax2, 'r')
        plt.show()

        ###############
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(self.a_weights_min, color='blue',
                 linestyle='dotted', label='min', linewidth=2)
        ax1.plot(self.a_weights_mean, color='blue',
                 linestyle='solid', label='mean', linewidth=1)
        ax1.plot(self.a_weights_max, color='blue',
                 linestyle='dotted', label='max', linewidth=2)

        ax1.set_xlabel('step')
        ax1.set_ylabel('alpha')

        ax2.plot(self.b_weights_min, color='red',
                 linestyle='dotted', linewidth=2)
        ax2.plot(self.b_weights_mean, color='red',
                 linestyle='solid', linewidth=1)
        ax2.plot(self.b_weights_max, color='red',
                 linestyle='dotted', linewidth=2)
        ax2.set_ylabel('beta')

        # Change color of each axis
        def color_y_axis(ax, color):
            """Color your axes."""
            for t in ax.get_yticklabels():
                t.set_color(color)
            return None

        plt.title('weights (min,mean,max)')
        color_y_axis(ax1, 'b')
        color_y_axis(ax2, 'r')
        plt.show()
