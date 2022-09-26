import warnings
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error

from al_experiments.experiment import Experiment
from al_experiments.stopping import WilcoxonSignificance


def total_amount_runtime_target_solver(
    y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
    cluster_boundaries: pd.DataFrame, target_solver: str,
    experiment: Experiment
) -> float:
    """ Returns the total amount of runtime that has been used so far within the selected experiments.

    Args:
        y_sampled (np.ndarray): Which instances are sampled.
        runtimes_df (pd.DataFrame): The runtimes.
        cluster_boundaries (pd.DataFrame): The cluster boundaries.
        target_solver (str): The target solver.
        experiment (Experiment): The experiment configuration.

    Returns:
        float: The amount of runtime sampled.
    """

    return (
        np.sum(runtimes_df.loc[
            y_sampled == 1,
            target_solver
        ].replace(np.inf, 5000))
        / np.sum(runtimes_df.loc[:, target_solver].replace(np.inf, 5000))
    )


def mean_wilcoxon_pvalue_predicted_labels(
    clustering: pd.DataFrame, pred: np.ndarray,
    y: np.ndarray, y_sampled: np.ndarray,
    target_solver: str, experiment: Experiment,
) -> float:
    """ Returns the mean Wilcoxon p-value of the predicted
    labels of the target solvers with each other known solver.

    Args:
        clustering (pd.DataFrame): The cluster labels.
        pred (np.ndarray): The predictions.
        y (np.ndarray): The ground-truth labels.
        y_sampled (np.ndarray): Which instances are sampled.

    Returns:
        float: The mean Wilcoxon p-value.
    """

    if isinstance(experiment.stopping, WilcoxonSignificance):
        t = experiment.stopping.observed_thresh
    else:
        t = 0.2

    if np.count_nonzero(y_sampled) / y_sampled.shape[0] <= t:
        other_solver_labels = clustering.iloc[
            y_sampled == 1,
            clustering.columns != target_solver
        ]
        target_labels = y[y_sampled == 1]
    else:
        other_solver_labels = clustering.iloc[
            :, clustering.columns != target_solver
        ]
        target_labels = np.where(y_sampled == 1, y, pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.mean(np.apply_along_axis(
            lambda col: (
                0.0
                if np.all(col == target_labels)
                else max(0, wilcoxon(col, target_labels).pvalue
                            )),
            0,
            other_solver_labels
        ))
    return result


def root_mean_squared_prediction_loss(
    pred: np.ndarray, actual: np.ndarray
) -> float:
    """ Returns the root-mean-squared-error of the predicted labels.

    Args:
        pred (np.ndarray): The predictions.
        actual (np.ndarray): The actual true labels.

    Returns:
        float: The root-mean-squared-error.
    """

    return mean_squared_error(actual, pred, squared=False)
