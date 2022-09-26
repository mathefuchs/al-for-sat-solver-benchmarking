from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier


class StoppingPredicate(ABC):
    """ Base class representing the stopping criterion. """

    @abstractmethod
    def get_params(self) -> str:
        return ""

    @abstractmethod
    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment,
        ranking_history: List[np.ndarray], wilcoxon_history: List[float],
    ) -> bool:
        """ Determines when to stop iterating.

        Args:
            y_sampled (np.ndarray): The already sampled instances.
            runtimes_df (pd.DataFrame): The runtimes.
            clustering (pd.DataFrame): The labels of known solvers.
            cluster_boundaries (pd.DataFrame): The cluster boundaries.
            timeout_clf (StackingClassifier): The time-out predictor.
            label_clf (StackingClassifier): The non-time-out label predictor.
            x (np.ndarray): The input features.
            y (np.ndarray): The target labels.
            target_solver (str): The target solver.
            experiment (Experiment): The experiment configuration.

        Returns:
            bool: Whether to stop adding more instances.
        """

        pass


class FixedSubsetSize(StoppingPredicate):
    """ Stop after a fixed amount of instances is in the subset. """

    def __init__(self, subset_size: float) -> None:
        """ Initializes this stopping predicate.

        Args:
            subset_size (float): The desired subset size.
        """

        self.subset_size = subset_size

    def get_params(self) -> str:
        return f"fixed_{self.subset_size:.2f}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment,
        ranking_history: List[np.ndarray], wilcoxon_history: List[float],
    ) -> bool:
        """ Determines when to stop iterating.

        Args:
            y_sampled (np.ndarray): The already sampled instances.
            runtimes_df (pd.DataFrame): The runtimes.
            clustering (pd.DataFrame): The labels of known solvers.
            cluster_boundaries (pd.DataFrame): The cluster boundaries.
            timeout_clf (StackingClassifier): The time-out predictor.
            label_clf (StackingClassifier): The non-time-out label predictor.
            x (np.ndarray): The input features.
            y (np.ndarray): The target labels.
            target_solver (str): The target solver.
            experiment (Experiment): The experiment configuration.

        Returns:
            bool: Whether to stop adding more instances.
        """

        return np.count_nonzero(y_sampled) / y_sampled.shape[0] >= self.subset_size


class RankingConverged(StoppingPredicate):
    """ Stop after the ranking does not change over some time. """

    def __init__(self, convergence_duration: float, min_amount: float) -> None:
        self.convergence_duration = convergence_duration
        self.min_amount = min_amount

    def get_params(self) -> str:
        return f"rnk_conv_{self.convergence_duration:.2f}_{self.min_amount:.2f}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment,
        ranking_history: List[np.ndarray], wilcoxon_history: List[float],
    ) -> bool:
        """ Determines when to stop iterating.

        Args:
            y_sampled (np.ndarray): The already sampled instances.
            runtimes_df (pd.DataFrame): The runtimes.
            clustering (pd.DataFrame): The labels of known solvers.
            cluster_boundaries (pd.DataFrame): The cluster boundaries.
            timeout_clf (StackingClassifier): The time-out predictor.
            label_clf (StackingClassifier): The non-time-out label predictor.
            x (np.ndarray): The input features.
            y (np.ndarray): The target labels.
            target_solver (str): The target solver.
            experiment (Experiment): The experiment configuration.

        Returns:
            bool: Whether to stop adding more instances.
        """

        con_num_instances = int(self.convergence_duration * y_sampled.shape[0])

        if (
            len(ranking_history) < con_num_instances or
            np.count_nonzero(y_sampled) / y_sampled.shape[0] < self.min_amount
        ):
            return False
        else:
            recent = ranking_history[-1]
            for i in range(2, con_num_instances + 1):
                if not np.all(recent == ranking_history[-i]):
                    return False
            return True


class WilcoxonSignificance(StoppingPredicate):  # TODO
    """ Stop after we are confident enough to distinguish solvers. """

    def __init__(
        self, observed_thresh: float, p_val_thresh: float, ema: float
    ) -> None:
        self.observed_thresh = observed_thresh
        self.p_val_thresh = p_val_thresh
        self.ema = ema

    def get_params(self) -> str:
        return f"wil_{self.observed_thresh:.2f}_{self.p_val_thresh:.2f}_{self.ema:.2f}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment,
        ranking_history: List[np.ndarray], wilcoxon_history: List[float],
    ) -> bool:
        """ Determines when to stop iterating.

        Args:
            y_sampled (np.ndarray): The already sampled instances.
            runtimes_df (pd.DataFrame): The runtimes.
            clustering (pd.DataFrame): The labels of known solvers.
            cluster_boundaries (pd.DataFrame): The cluster boundaries.
            timeout_clf (StackingClassifier): The time-out predictor.
            label_clf (StackingClassifier): The non-time-out label predictor.
            x (np.ndarray): The input features.
            y (np.ndarray): The target labels.
            target_solver (str): The target solver.
            experiment (Experiment): The experiment configuration.

        Returns:
            bool: Whether to stop adding more instances.
        """

        if len(wilcoxon_history) == 0 or np.count_nonzero(y_sampled) / y_sampled.shape[0] < self.observed_thresh:
            return False
        else:
            ema_val = wilcoxon_history[0]
            for v in wilcoxon_history[1:]:
                ema_val = self.ema * v + (1 - self.ema) * ema_val
            return ema_val <= self.p_val_thresh
