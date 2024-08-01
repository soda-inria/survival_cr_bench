import numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from hazardous.utils import make_recarray

class SksurvBoosting(GradientBoostingSurvivalAnalysis):
    
    def fit(self, X, y):
        y = make_recarray(y)
        return super().fit(X, y)

    def predict_cumulative_incidence(self, X, time_grid):
        y_surv = self.predict_survival_function(X, return_array=True)

        time_indices = np.searchsorted(self.unique_times_, time_grid)
        time_indices = np.clip(time_indices, 0, self.unique_times_.shape[0] - 1)
        y_surv = y_surv[:, time_indices][None, :, :]
        y_pred = 1 - y_surv
        y_pred = np.concatenate([y_surv, y_pred], axis=0)

        return y_pred
    
    def score(self, X, y):
        y = make_recarray(y)
        return super().score(X, y)