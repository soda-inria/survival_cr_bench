from numbers import Integral
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_scalar


def make_recarray(y):
    event = y["event"].values
    duration = y["duration"].values
    return np.array(
        [(event[i], duration[i]) for i in range(y.shape[0])],
        dtype=[("e", bool), ("t", float)],
    )


def check_y_survival(y):
    """Convert DataFrame and dictionnary to record array."""
    y_keys = ["event", "duration"]

    if isinstance(y, np.ndarray) and not isinstance(y.dtype, Iterable):
        # assumes event, then duration
        return y[:, 0], y[:, 1]

    if (
        isinstance(y, np.ndarray)
        and sorted(y.dtype.names, reverse=True) == y_keys
        or isinstance(y, dict)
        and sorted(y, reverse=True) == y_keys
    ):
        return np.ravel(y["event"]), np.ravel(y["duration"])

    elif isinstance(y, pd.DataFrame) and sorted(y.columns, reverse=True) == y_keys:
        return y["event"].values, y["duration"].values

    else:
        raise ValueError(
            "y must be a record array, a pandas DataFrame, or a dict "
            "whose dtypes, keys or columns are 'event' and 'duration'. "
            f"Got:\n{repr(y)}"
        )


def check_event_of_interest(k):
    """`event_of_interest` must be the string 'any' or a positive integer."""
    check_scalar(k, "event_of_interest", target_type=(str, Integral))
    not_str_any = isinstance(k, str) and k != "any"
    not_positive = isinstance(k, int) and k < 1
    if not_str_any or not_positive:
        raise ValueError(
            "event_of_interest must be a strictly positive integer or 'any', "
            f"got: event_of_interest={k}"
        )
    return


class CumulativeIncidencePipeline(Pipeline):
    def predict_cumulative_incidence(self, X, times=None):
        Xt = X
        for _, _, transformer in self._iter(with_final=False):
            Xt = transformer.transform(Xt)
        return self.steps[-1][1].predict_cumulative_incidence(Xt, times)


def get_n_events(event):
    """Fetch the number of distinct competing events.

    Parameters
    ----------
    event : pd.Series of shape (n_samples,)
        Binary or multiclass events.

    Returns
    -------
    n_events : int
        The number of events, without accounting for the censoring 0.
    """
    event_ids = event.unique()
    has_censoring = int(0 in event_ids)
    return len(event_ids) - has_censoring


def make_time_grid(duration, n_steps=20):
    t_min, t_max = duration.min(), duration.max()
    return np.linspace(t_min, t_max, n_steps)


class SurvStratifiedShuffleSplit(StratifiedShuffleSplit):
    def split(self, X, y, groups=None):
        event, _ = check_y_survival(y)
        event = event.astype(int)
        return super().split(X, event, groups)


class SurvStratifiedSingleSplit(SurvStratifiedShuffleSplit):
    def split(self, X, y, groups=None):
        train, test = next(iter(super().split(X, y, groups)))
        yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1
