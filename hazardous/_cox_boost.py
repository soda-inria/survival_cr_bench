import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.utils import check_random_state

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector

from hazardous.utils import check_y_survival, get_n_events, make_time_grid
from hazardous.metrics._brier_score import (
    integrated_brier_score_incidence_oracle,
    integrated_brier_score_incidence
)

pandas2ri.activate()


class CoxBoost(BaseEstimator):
    """
    Boosting for high-dimensional time-to-event data with competing risks.

    Parameters
    ----------
    stepno: int, default=100
        Number of boosting steps.
    
    penalty: int, default=9
        Penalty value for the update of an individual element of the parameter vector
        in each boosting step.
        
    criterion: str, {'score', 'pscore'}, default='score'
        Indicates the criterion to be used for selection in each boosting step.
        - score: fit using the un-penalized score statistics.
        - pscore: fit using the penalized score statistics.
    
    scheme: str, {'linear', 'sigmoid'}
        Scheme for changing step sizes (via stepsize.factor).
        - "linear" corresponds to the scheme described in Binder and Schumacher (2009b)
        - "sigmoid" employs a sigmoid shape.
    
    max_samples: int, default=10_000
        The trainset size after subsampling. CoxBoost scale badly and might crashes
        when the number of rows > 50k.

    References
    ----------
    https://rdrr.io/cran/CoxBoost/man/CoxBoost.html
    """

    def __init__(
        self,
        stepno=100,
        penalty=9,
        criterion="score",
        scheme="linear",
        verbose=True,
        max_samples=1_000,
        random_state=None,
    ):
        self.stepno = stepno
        self.penalty = penalty
        self.criterion = criterion
        self.scheme = scheme
        self.verbose = verbose
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        
        coxboost = importr('CoxBoost')

        if X.shape[0] > self.max_samples:
            rng = check_random_state(self.random_state)
            sample_indices = rng.choice(
                np.arange(X.shape[0]),
                size=self.max_samples,
                replace=False,
            )
            X, y = X.iloc[sample_indices], y.iloc[sample_indices]

        X_r = check_features(X)
        event, duration = check_y_survival(y)
        
        # Used for prediction
        self.time_grid_ = make_time_grid(duration, n_steps=100)
        self.y_train_ = y.copy()

        n_events = get_n_events(y["event"])
        self.event_ids_ = list(range(1, n_events+1))

        # Fit one CoxBoost for each event of interest.
        # CoxBoost only fits the hazard sub-distribution for the event 1,
        # considering all event > 1 as competing event.
        # Crucially, CoxBoost doesn't predict the CIF for competing events.
        # So, we must fit one CoxBoost model for each event to predict CIFs
        # for all events.
        self.estimators_ = {}

        iter_ = self.event_ids_
        if self.verbose:
            iter_ = tqdm(iter_)

        for event_id in iter_:
            event_swapped = event.copy()

            if event_id != 1:
                event_swapped[(event == event_id)] = 1
                event_swapped[(event == 1)] = event_id
    
            time_r = IntVector(duration)
            status_r = IntVector(event_swapped)
    
            self.estimators_[event_id] = coxboost.CoxBoost(
                time_r,
                status_r,
                X_r,
                stepno=self.stepno,
                penalty=self.penalty,
                cmprsk="csh",
                **{"sf.scheme": self.scheme},
            )
        
        return self

    def predict_cumulative_incidence(self, X, times=None):
        
        check_is_fitted(self, "estimators_")
        coxboost = importr('CoxBoost')

        X_r = check_features(X)

        if times is None:
            times = self.time_grid_
        times = FloatVector(times)

        y_proba = []
        for event_id in self.event_ids_:
            estimator = self.estimators_[event_id]
            predictions = coxboost.predict_CoxBoost(
                estimator,
                newdata=X_r,
                type="CIF",
                times=times,
            )
            y_proba.append(
                np.array(predictions)[:, None, :]
            )

        y_proba = np.concatenate(y_proba, axis=1)
        surv_proba = (1 - y_proba.sum(axis=1))[:, None, :]
        y_proba = np.concatenate([surv_proba, y_proba], axis=1)
        
        # For consistency with this benchmark project. Should be removed elsewhere.
        y_proba = y_proba.swapaxes(0, 1)

        return y_proba

    def score(self, X, y, scale_censoring=None, shape_censoring=None):

        predicted_curves = self.predict_cumulative_incidence(X)
        ibs_events = []

        for event_id in self.event_ids_:

            predicted_curves_for_event = predicted_curves[event_id, :, :]

            if scale_censoring is not None and shape_censoring is not None:
                ibs_event = integrated_brier_score_incidence_oracle(
                    y_train=self.y_train_,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.time_grid_,
                    shape_censoring=shape_censoring,
                    scale_censoring=scale_censoring,
                    event_of_interest=event_id,
                )

            else:
                ibs_event = integrated_brier_score_incidence(
                    y_train=self.y_train_,
                    y_test=y,
                    y_pred=predicted_curves_for_event,
                    times=self.time_grid_,
                    event_of_interest=event_id,
                )

            ibs_events.append(ibs_event)

        return -np.mean(ibs_events)


def check_features(X):
    n_samples, n_features = X.shape
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    X_r = ro.r.matrix(
        FloatVector(X.flatten()), nrow=n_samples, ncol=n_features
    )
    return X_r