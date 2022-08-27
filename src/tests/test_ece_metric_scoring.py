import numpy as np
from sof_models_churn.features.ece_metric_scoring import ChurnPredictionMetrics

class TestEceMetricScoring:

    def setup_method(self, method):
        # time to event (tte) has shape (n_samples, n_weeks, 1)
        # tte is left-padded
        tte = np.array(
            [[np.nan, np.nan, np.nan, 3, 2, 1, 0, 5, 4, 3, 2, 1],
             [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
             [3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0]], dtype="float64"
        )
        self.tte = tte

    def test_calc_time_since_last_event_c4(self):
        t, m = ChurnPredictionMetrics.calc_time_since_last_event(self.tte, 4)
        assert np.array_equal(t, np.array([4, 4, 1], dtype="float64"))
        assert np.array_equal(m, np.array([False, False, True], dtype="float64"))

    def test_calc_time_since_last_event_c5(self):
        t, m = ChurnPredictionMetrics.calc_time_since_last_event(self.tte, 5)
        assert np.array_equal(t, np.array([5, 5, 2], dtype="float64"))
        assert np.array_equal(m, np.array([False, False, True], dtype="float64"))
        
    def test_calc_time_since_last_event_c6(self):
        t, m = ChurnPredictionMetrics.calc_time_since_last_event(self.tte, 6)
        assert np.array_equal(t, np.array([0, 6, 3], dtype="float64"))
        assert np.array_equal(m, np.array([True, False, True], dtype="float64"))
        
    def test_calc_time_since_last_event_c7(self):
        t, m = ChurnPredictionMetrics.calc_time_since_last_event(self.tte, 7)
        assert np.array_equal(t, np.array([1, 7, 0], dtype="float64"))
        assert np.array_equal(m, np.array([True, False, True], dtype="float64"))
        
    def test_calc_time_since_last_event_c8(self):
        t, m = ChurnPredictionMetrics.calc_time_since_last_event(self.tte, 8)
        assert np.array_equal(t, np.array([2, 8, 1], dtype="float64"))
        assert np.array_equal(m, np.array([True, False, True], dtype="float64"))
        
    def test_calc_time_since_last_event_c10(self):
        t, m = ChurnPredictionMetrics.calc_time_since_last_event(self.tte, 10)
        assert np.array_equal(t, np.array([4, 10, 3], dtype="float64"))
        assert np.array_equal(m, np.array([True, False, True], dtype="float64"))

    def test_calc_time_since_last_event_c11(self):
        t, m = ChurnPredictionMetrics.calc_time_since_last_event(self.tte, 11)
        assert np.array_equal(t, np.array([5, 0, 0], dtype="float64"))
        assert np.array_equal(m, np.array([True, True, True], dtype="float64"))

    def test_replace_dummy_values_with_nans(self):
        # The shape of y and p is (n_samples, n_weeks, 2)
        # They are right padded, but NaNs have been replaced with dummy values
        y0 = np.array(
            [[3, 2, 1, 0, 2, 1, 1.2, 1.2, 1.2],
             [1, 1, 1, 1, 0, 0, 0.5, 0.5, 0.5]], dtype="float64"
        )
        y1 = np.array(
            [[3, 2, 1, 0, 2, 1, 0, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="float64"
        )
        y2 = np.array(
            [[3, 2, 1, 0, 2, 1, 0, 2, 1],
             [1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype="float64"
        )
        y = np.stack((y0.T, y1.T, y2.T), axis=0)

        p0 = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8],
             [10, 11, 12, 13, 14, 15, 16, 17, 18]], dtype="float64"
        )
        p = np.stack((p0.T, p0.T, p0.T), axis=0)

        p_r, y_r = ChurnPredictionMetrics.replace_dummy_value_with_nans(p, y)
        assert np.array_equal(y_r[0,:,0], np.array([3, 2, 1, 0, 2, 1, np.nan, np.nan, np.nan]), equal_nan=True)
        assert np.array_equal(y_r[0,:,1], np.array([1, 1, 1, 1, 0, 0, np.nan, np.nan, np.nan]), equal_nan=True)
        assert np.array_equal(y_r[1,:,:], y1.T, equal_nan=True)
        assert np.array_equal(y_r[2,:,:], y2.T, equal_nan=True)

        assert np.array_equal(p_r[0,:,0], np.array([0, 1, 2, 3, 4, 5, np.nan, np.nan, np.nan]), equal_nan=True)
        assert np.array_equal(p_r[0,:,1], np.array([10, 11, 12, 13, 14, 15, np.nan, np.nan, np.nan]), equal_nan=True)
        assert np.array_equal(p_r[1,:,:], p0.T, equal_nan=True)
        assert np.array_equal(p_r[2,:,:], p0.T, equal_nan=True)

