from hazardous.survtrace._model import SurvTRACE
from hazardous.survtrace._encoder import SurvFeatureEncoder
from hazardous._deep_hit import DeepHitEstimator
from hazardous._gb_multi_incidence import GBMultiIncidence
from hazardous._aalen_johansen import AalenJohansenEstimator
from hazardous._sksurv_boosting import SksurvBoosting
from hazardous.utils import CumulativeIncidencePipeline


def init_gbmi(
    random_state=None,
    **model_params,
):
    return CumulativeIncidencePipeline(
        [
            ("encoder", SurvFeatureEncoder()),
            (
                "estimator",
                GBMultiIncidence(
                    loss="inll",
                    show_progressbar=True,
                    n_times=1,
                ),
            ),
        ]
    ).set_params(**model_params)


def init_survtrace(
    lr=1e-3,
    batch_size=128,
    max_epochs=100,
    random_state=None,
    **model_params,
):
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
    random_state=None,
    **model_params,
):
    return DeepHitEstimator(
        num_nodes_shared=num_nodes_shared,
        num_nodes_indiv=num_nodes_indiv,
        verbose=verbose,
        num_durations=num_durations,
        batch_norm=batch_norm,
        dropout=dropout,
        **model_params,
    )


def init_aalen_johansen(calculate_variance=False, random_state=None):
    return AalenJohansenEstimator(
        calculate_variance=calculate_variance, seed=random_state
    )


def init_fine_and_gray(random_state=None, **model_params):
    # This import is shielded inside a function call because R and cmprsk need
    # to be installed to import FineGrayEstimator.
    from hazardous._fine_and_gray import FineGrayEstimator

    return CumulativeIncidencePipeline(
        [
            ("encoder", SurvFeatureEncoder()),
            ("estimator", FineGrayEstimator(random_state=random_state)),
        ]
    ).set_params(**model_params)


def init_random_survival_forest(random_state=None, **model_params):
    # This import is shielded inside a function call because R and
    # randomForestSRC need to be installed to import RSFEstimator.
    from hazardous._rsf import RSFEstimator

    return CumulativeIncidencePipeline(
        [
            ("encoder", SurvFeatureEncoder()),
            ("estimator", RSFEstimator(random_state=random_state))
        ]
    ).set_params(**model_params)


def init_sksurv_boosting(random_state=None, **model_params):
    return CumulativeIncidencePipeline(
        [
            ("encoder", SurvFeatureEncoder()),
            ("estimator", SksurvBoosting(random_state=random_state))
        ]
    ).set_params(**model_params)


INIT_MODEL_FUNCS = {
    "gbmi": init_gbmi,
    "survtrace": init_survtrace,
    "deephit": init_deephit,
    "fine_and_gray": init_fine_and_gray,
    "aalen_johansen": init_aalen_johansen,
    "random_survival_forest": init_random_survival_forest,
    "sksurv_boosting": init_sksurv_boosting,
    # TODO: "init_pchazard"
}
