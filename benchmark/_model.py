from hazardous.survtrace._model import SurvTRACE
from hazardous._deep_hit import _DeepHit
from hazardous.survtrace._encoder import SurvFeatureEncoder
from hazardous.utils import (
    SurvStratifiedSingleSplit,
    SurvStratifiedShuffleSplit,
    CumulativeIncidencePipeline,
)
from hazardous._gb_multi_incidence import GBMultiIncidence
from hazardous._aalen_johansen import AalenJohansenEstimator


def init_gbmi(show_progressbar=True, n_times=1, loss="inll", **model_params):

    return CumulativeIncidencePipeline(
        [
            ("encoder", SurvFeatureEncoder()),
            (
                "estimator",
                GBMultiIncidence(
                    loss=loss,
                    show_progressbar=show_progressbar,
                    n_times=n_times,
                    **model_params,
                ),
            ),
        ]
    )


def init_survtrace(lr=1e-3, batch_size=128, max_epochs=20, **model_params):
    return SurvTRACE(
        batch_size=batch_size,
        lr=lr,
        optimizer__weight_decay=0,
        max_epochs=max_epochs,
        **model_params,
    )


def init_deephit(
    num_nodes_shared=[64, 64],
    num_nodes_indiv=[32],
    verbose=True,
    num_durations=10,
    batch_norm=True,
    dropout=None,
    **model_params,
):
    return _DeepHit(
        num_nodes_shared=num_nodes_shared,
        num_nodes_indiv=num_nodes_indiv,
        verbose=verbose,
        num_durations=num_durations,
        batch_norm=batch_norm,
        dropout=dropout,
        **model_params
    )


def init_aalen_johansen(calculate_variance=False, seed=0):
    return AalenJohansenEstimator(calculate_variance=calculate_variance, seed=seed)


def init_fine_and_gray():
    # This import is shielded inside a function call because R and cmprsk need
    # to be installed to import FineGrayEstimator.
    from hazardous._fine_and_gray import FineGrayEstimator
    return CumulativeIncidencePipeline(
        [("encoder", SurvFeatureEncoder()), ("estimator", FineGrayEstimator())]
    )


INIT_MODEL_FUNCS = {
    "gbmi": init_gbmi,
    "survtrace": init_survtrace,
    "deephit": init_deephit,
    "fine_and_gray": init_fine_and_gray,
    "aalen_johansen": init_aalen_johansen,
    # TODO: "init_pchazard"
    # TODO: "init_random_survival_forest"
}
