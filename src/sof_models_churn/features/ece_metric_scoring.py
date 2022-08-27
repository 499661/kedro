import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from six.moves import xrange
from typing import Tuple
from collections import OrderedDict
import logging

import sof_models_churn.common.wtte_utils as wtte
from sof_models_churn.common.wtte_weibull import cmf as weibull_cmf

logger = logging.getLogger(__name__)

class ChurnPredictionMetrics():

    @staticmethod
    def binned_prob(label,prob,K=10):
        """
        Bins the predicted probabilities into K bins, and within each bin calculates
         - average predicted probability p (thus by definition: left_edge < p <= right_edge)
         - mean of the binary label, for samples within this bin
         - fraction of total samples that fall within this bin

        Args:
            label (1d array): binary label, indicating whether event occurred within box or not
            prob (1d array): the model's predicted probability of the event occurring within the box
        Kwargs:
            K (int): number of bins

        Returns:
            expected  (np.array length K): mean predicted probability, per bin
            predicted (np.array length K): mean positive samples, per bin
            frac_in_bin (np.array length K): fraction of total samples occurring in each bin

        Raises:
        """
        a = -1
        predicted = []
        expected = []
        frac_in_bin = []
        for i in range(K):
            b = (i+1)/K
            in_interval = ((a<prob)*(prob<=b))>0
            if in_interval.any():
                expected.append(label[in_interval].mean())
                predicted.append(prob[in_interval].mean())
            else:
                expected.append(0)
                predicted.append(0)
            a = b
            frac_in_bin.append(in_interval.mean())
        expected,predicted,frac_in_bin = [np.array(arr) for arr in [expected,predicted,frac_in_bin]]
        return expected,predicted,frac_in_bin

    @classmethod
    def expected_calibration_error(cls,label,prob,K=10):
        """
        (Called on outputs of binned_prob() function - see above for definition)

        Computes the weighted sum (over bins of p_pred) of differences between predicted p and "actual p" (ie. empirical fraction computed within each bin). Weighting is by fraction of total samples occurring in respective bin.
        """
        # See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/
        expected,predicted,frac_in_bin = cls.binned_prob(label,prob,K)
        return (frac_in_bin*np.abs(expected-predicted)).sum()

    @staticmethod
    def calc_time_since_last_event(tte: np.ndarray, cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            tte: time to event shape (n_samples, n_weeks). Must be left padded.
            cutoff: index corresponding to current week
        Returns:
            time_since_last_event: shape (n_samples,) contains NaNs where the last event cannot be determined.
            mask: boolean array shape (n_samples,) false when the time to last event is not available.

        Note that time_since_last_event has dtype int64 and therefore cannot contain any NaNs. That is why
        we need to return the mask separately.
        """
        # Find the index < cutoff where tte is 0. This is the index corresponding to the last event.
        idx = np.argmax(np.where(tte[:, :cutoff+1] == 0,
                                np.array([range(cutoff+1)]*tte.shape[0]),
                                -np.ones((tte.shape[0], cutoff+1))),
                        axis=1)
        t_since_last_event = cutoff - idx

        # Determine cases where the last event does not exist 
        mask = ~(np.sum(tte[:,:cutoff+1] == 0, axis=1) == 0)
        return t_since_last_event, mask

    @staticmethod
    def replace_dummy_value_with_nans(predicted: np.ndarray, target: np.ndarray, dummy_val=0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            predicted: prediction from RNN shape (n_samples, n_weeks, 2)
            target: observed values shape (n_samples, n_weeks, 2)
            dummy_val: The value in target[...,1] that is indicative of NaNs
        Returns:
            predicted: with dummy values replaced by np.nan
            target: with dummy values replaced by np.nan
        """
        fltr = target[:,:,1] == dummy_val
        predicted_out = predicted.copy()
        predicted_out[fltr, :] = np.nan
        target_out = target.copy()
        target_out[fltr, :] = np.nan
        
        return predicted_out, target_out

    @classmethod
    def evaluate_metrics(cls, predicted, y, n_steps, min_box_width=10, consider_last_event=False, dummy_val=0.5):
        """
        Args:
            predicted (array):
            y (array):
            n_steps (int): number of timesteps (counting back from latest) over which to evaluate the metrics.
            min_box_width (int): smallest number of timesteps for which to evaluate the metrics.
            include_time_to_last_event (bool): whether to take into account the last time a customer shopped when calculating the metrics.

        Returns:
            metric_dict (OrderedDict): containing auc, fpr, tpr, ap, precision, recall, frac_in_box, ece for each box width.

        Raises:
        """
        metric_dict = OrderedDict()

        # Evaluate on timesteps after training set end.
        testset_begin = y.shape[1]-n_steps
        # We can use a box-width 0,...,n_timesteps-trainset_end
        max_box_width = n_steps

        predicted, y = cls.replace_dummy_value_with_nans(predicted, y, dummy_val=dummy_val)
        predicted_tmp = wtte.right_pad_to_left_pad(predicted)
        tte_tmp = wtte.right_pad_to_left_pad(y[...,0])

        # Select datapoints at the first timestep after training set ends.
        alpha      = predicted_tmp[:,testset_begin,0]
        beta       = predicted_tmp[:,testset_begin,1]
        tte        = tte_tmp[:,testset_begin]
        if consider_last_event:
            time_since_last_event,  m_t = cls.calc_time_since_last_event(tte_tmp, testset_begin - 1)
            m = ~np.isnan(tte+alpha+beta) & m_t # nan-mask
        else:
            time_since_last_event = np.zeros(tte.shape)
            m = ~np.isnan(tte+alpha+beta) # nan-mask

        logger.info(f"Total number of observations in validation set: {tte.shape[0]}")

        for box_width in xrange(min(min_box_width, max_box_width-1), max_box_width):
            m_ = m & ((box_width - time_since_last_event) >= 0)
            if m_.sum() == 0:
                continue
            if consider_last_event and (box_width == 0):
                continue
            is_in_box = (tte[m_] <= box_width - time_since_last_event[m_]).flatten()
            pred_prob_in_box  = weibull_cmf(a=alpha[m_],b=beta[m_],t=box_width-time_since_last_event[m_]).flatten()

            # is_in_box measures retention, but we want the positive class to be churn
            ece = cls.expected_calibration_error(~is_in_box, 1.0 - pred_prob_in_box)
            fpr, tpr, _ = metrics.roc_curve(~is_in_box, 1.0 - pred_prob_in_box)
            auc = metrics.auc(fpr,tpr)
            precision, recall, _ = metrics.precision_recall_curve(~is_in_box, 1.0 - pred_prob_in_box)
            ap = metrics.average_precision_score(~is_in_box, 1.0 - pred_prob_in_box)
            logger.info('box width:{:02d}\tauc:{:.4f}\tap:{:.4f}\tECE:{:.4f}\t(frac_in_box:{:.4f}\t{:d} obs)'.\
                        format(box_width, auc, ap, ece, is_in_box.mean(), m_.sum()))
            metric_dict[box_width] = {"auc": auc, "fpr": fpr, "tpr": tpr, "ap": ap, "precision": precision,
                                      "recall": recall, "ece": ece, "frac_in_box": np.mean(is_in_box), 
                                      "observations": m_.sum(), "pred_prob_in_box": pred_prob_in_box,
                                      "is_in_box": is_in_box}

        return metric_dict

    @staticmethod
    def plot_metrics(metric_dict, save=False, name='ece_auc_plot.png'):
        box_widths = [k for k in metric_dict]
        aucs = [metric_dict[k]["auc"] for k in metric_dict]
        aps = [metric_dict[k]["ap"] for k in metric_dict]
        eces = [metric_dict[k]["ece"] for k in metric_dict]

        fig = plt.figure(1)
        plt.plot(box_widths,aucs,label='AUC (high=good)')
        plt.plot(box_widths,aps,label='AP (high=good)')
        plt.plot(box_widths,eces,label ='ECE (low=good)')
        plt.ylim(0,1)
        plt.legend()
        plt.xlabel('box width (predicted timesteps in future)')
        plt.title('Calibration (ECE) & discrimination (AUC) - Model Predictions')
        plt.show()
        if save is True:
            fig.savefig(name)
        plt.close(fig)

    @staticmethod
    def _create_subplot_grid(n_plots):

        # Try if 3 or 4 columns works better
        if (n_plots % 4) == 0:
            n_cols = 4
            n_rows = n_plots // n_cols
        elif (n_plots % 3)  == 0:
            n_cols = 3
            n_rows = n_plots // n_cols
        else:
            # Want to minimize empty subplots
            n_cols = 4 if (n_plots % 4) > (n_plots % 3) else 3
            n_rows = n_plots // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 4*n_rows),
                                 sharex=True, sharey=True)
        if n_rows == 1:
            axes = axes.reshape(-1, n_cols)
        fig.set_tight_layout(True)
        
        return fig, axes

    @classmethod
    def plot_roc_curves(cls, metric_dict, save=False, name="roc_curves_plot.png"):
        fig, axes = cls._create_subplot_grid(len(metric_dict))
        for i in range(axes.shape[1]):
            axes[-1,i].set_xlabel("FPR")
        for i in range(axes.shape[0]):
            axes[i, 0].set_ylabel("TPR")

        axes = axes.flatten()
        for i, bw in enumerate(metric_dict):
            axes[i].plot(metric_dict[bw]["fpr"], metric_dict[bw]["tpr"], '-C0')
            axes[i].plot([0,1], [0,1], '--k')
            axes[i].set_title("BW={:02d}, AUC={:.3f}".format(bw, metric_dict[bw]["auc"]))

        if save:
            fig.savefig(name, dpi=200)
            plt.close(fig)
        else:
            return fig, axes

    @classmethod
    def plot_pr_curves(cls, metric_dict, save=False, name="precision_recall_plot.png"):
        fig, axes = cls._create_subplot_grid(len(metric_dict))
        for i in range(axes.shape[1]):
            axes[-1,i].set_xlabel("Recall")
        for i in range(axes.shape[0]):
            axes[i, 0].set_ylabel("Precision")

        axes = axes.flatten()
        for i, bw in enumerate(metric_dict):
            axes[i].plot(metric_dict[bw]["recall"], metric_dict[bw]["precision"], '-C0')
            axes[i].set_title("BW={:02d}, AP={:.3f}".format(bw, metric_dict[bw]["ap"]))
            axes[i].axhline(1.0 - metric_dict[bw]["frac_in_box"], ls="--", c='k')

        if save:
            fig.savefig(name, dpi=200)
            plt.close(fig)
        else:
            return fig, axes
