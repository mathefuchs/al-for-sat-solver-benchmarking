from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier


class RankingFunctor(ABC):
    """ Base class for ranking decision making. """

    def __init__(self, needs_predictions) -> None:
        self.needs_predictions = needs_predictions

    @abstractmethod
    def get_params(self) -> str:
        return ""

    @abstractmethod
    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, pred: np.ndarray,
        experiment, pred_history: List[np.ndarray]
    ) -> np.ndarray:
        """ Calculates how the target solver compares against known solvers.

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
            pred (np.ndarray): The predictions.
            experiment (Experiment): The experiment configuration.

        Returns:
            np.ndarray: Binary vector indicating whether the
            target solver is better than the respective solver
        """

        pass


class PartialObservationBasedRanking(RankingFunctor):
    """ Ranking based on partially observed runtimes. """

    def __init__(self) -> None:
        super().__init__(False)

    def get_params(self) -> str:
        return "par2_obs"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, pred: np.ndarray,
        experiment, pred_history: List[np.ndarray]
    ) -> np.ndarray:
        """ Calculates how the target solver compares against known solvers.

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
            pred (np.ndarray): The predictions.
            experiment (Experiment): The experiment configuration.

        Returns:
            np.ndarray: Binary vector indicating whether the
            target solver is better than the respective solver
        """

        other_sampled_runtimes = np.mean(runtimes_df.loc[
            y_sampled == 1,
            runtimes_df.columns != target_solver
        ].replace([np.inf], experiment.par * 5000), axis=0).to_numpy()

        target_runtimes = runtimes_df.loc[
            y_sampled == 1,
            target_solver
        ].replace([np.inf], experiment.par * 5000).to_numpy()

        target_sampled_runtimes = np.mean(target_runtimes, axis=0)
        # Binary vector with 1 representing that the target solver is better than the respective solver
        return other_sampled_runtimes <= target_sampled_runtimes


class PredictedClusterLabelRanking(RankingFunctor):
    """ Ranking based on the predicted cluster labels. """

    def __init__(
        self, use_hist: bool, only_observed_thresh: float,
        num_hist: int, fallback_thresh: float,
    ) -> None:
        super().__init__(True)
        self.use_hist = use_hist
        self.only_observed_thresh = only_observed_thresh
        self.num_hist = num_hist
        self.fallback_thresh = fallback_thresh

    def get_params(self) -> str:
        return f"lbl_pred_{self.use_hist}_{self.only_observed_thresh:.2f}_{self.num_hist}_{self.fallback_thresh:.2f}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, pred: np.ndarray,
        experiment, pred_history: List[np.ndarray]
    ) -> np.ndarray:
        """ Calculates how the target solver compares against known solvers.

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
            pred (np.ndarray): The predictions.
            experiment (Experiment): The experiment configuration.

        Returns:
            np.ndarray: Binary vector indicating whether the
            target solver is better than the respective solver
        """

        n = self.num_hist
        if self.use_hist and len(pred_history) >= n:
            count_0 = np.where(pred_history[-1] == 0, 1, 0)
            for i in range(2, n + 1):
                count_0 = count_0 + np.where(pred_history[-i] == 0, 1, 0)

            count_1 = np.where(pred_history[-1] == 1, 1, 0)
            for i in range(2, n + 1):
                count_1 = count_1 + np.where(pred_history[-i] == 1, 1, 0)

            count_2 = n - (count_0 + count_1)

            pred_to_use = np.where(
                (count_2 >= count_1) & (count_2 >= count_0),
                2,
                np.where((count_1 >= count_2) & (count_1 >= count_0), 1, 0)
            )
        else:
            pred_to_use = pred

        other_label_mean = np.mean(np.where(
            clustering == 2, 2 * experiment.label_ranking_timeout_weight, clustering
        ), axis=0)
        target_label_mean = np.mean(np.where(
            pred_to_use == 2, 2 * experiment.label_ranking_timeout_weight, pred_to_use
        ))
        label_ranking = other_label_mean <= target_label_mean
        partial_obs_ranking = PartialObservationBasedRanking()(
            y_sampled, runtimes_df, clustering, cluster_boundaries,
            timeout_clf, label_clf, x, y, target_solver, pred,
            experiment, pred_history,
        )
        return np.where(
            np.abs(other_label_mean - target_label_mean) <= self.fallback_thresh,
            partial_obs_ranking,
            label_ranking,
        )


class PartialObservedLabelRanking(RankingFunctor):
    """ Ranking based on partially observed labels. """

    def __init__(self, fallback_thresh: float) -> None:
        super().__init__(False)
        self.fallback_thresh = fallback_thresh

    def get_params(self) -> str:
        return f"lbl_obs_{self.fallback_thresh:.2f}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, pred: np.ndarray,
        experiment, pred_history: List[np.ndarray]
    ) -> np.ndarray:
        """ Calculates how the target solver compares against known solvers.

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
            pred (np.ndarray): The predictions.
            experiment (Experiment): The experiment configuration.

        Returns:
            np.ndarray: Binary vector indicating whether the
            target solver is better than the respective solver
        """

        observed = y[y_sampled == 1]
        other_label_mean = np.mean(np.where(
            clustering == 2, 2 * experiment.label_ranking_timeout_weight, clustering
        ), axis=0)
        target_label_mean = np.mean(np.where(
            observed == 2, 2 * experiment.label_ranking_timeout_weight, observed
        ))
        label_ranking = other_label_mean <= target_label_mean
        partial_obs_ranking = PartialObservationBasedRanking()(
            y_sampled, runtimes_df, clustering, cluster_boundaries,
            timeout_clf, label_clf, x, y, target_solver, pred,
            experiment, pred_history,
        )
        return np.where(
            np.abs(other_label_mean - target_label_mean) <= self.fallback_thresh,
            partial_obs_ranking,
            label_ranking,
        )


class PartialObservedLabelAndPredictionRanking(RankingFunctor):
    """ Ranking based on partially observed labels. """

    def __init__(
        self, use_hist: bool, only_observed_thresh: float,
        num_hist: int, fallback_thresh: float
    ) -> None:
        super().__init__(True)
        self.use_hist = use_hist
        self.only_observed_thresh = only_observed_thresh
        self.num_hist = num_hist
        self.fallback_thresh = fallback_thresh

    def get_params(self) -> str:
        return f"lbl_obs_pred_{self.use_hist}_{self.only_observed_thresh:.2f}_{self.num_hist}_{self.fallback_thresh:.2f}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, pred: np.ndarray,
        experiment, pred_history: List[np.ndarray]
    ) -> np.ndarray:
        """ Calculates how the target solver compares against known solvers.

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
            pred (np.ndarray): The predictions.
            experiment (Experiment): The experiment configuration.

        Returns:
            np.ndarray: Binary vector indicating whether the
            target solver is better than the respective solver
        """

        if np.count_nonzero(y_sampled) / y_sampled.shape[0] <= self.only_observed_thresh:
            observed = y[y_sampled == 1]
            other_label_mean = np.mean(np.where(
                clustering == 2, 2 * experiment.label_ranking_timeout_weight, clustering
            ), axis=0)
            target_label_mean = np.mean(np.where(
                observed == 2, 2 * experiment.label_ranking_timeout_weight, observed
            ))
            label_ranking = other_label_mean <= target_label_mean
            partial_obs_ranking = PartialObservationBasedRanking()(
                y_sampled, runtimes_df, clustering, cluster_boundaries,
                timeout_clf, label_clf, x, y, target_solver, pred,
                experiment, pred_history,
            )
            return np.where(
                np.abs(other_label_mean - target_label_mean) <= 0.05,
                partial_obs_ranking,
                label_ranking,
            )
        else:
            n = self.num_hist
            if self.use_hist and len(pred_history) >= n:
                count_0 = np.where(pred_history[-1] == 0, 1, 0)
                for i in range(2, n + 1):
                    count_0 = count_0 + np.where(pred_history[-i] == 0, 1, 0)

                count_1 = np.where(pred_history[-1] == 1, 1, 0)
                for i in range(2, n + 1):
                    count_1 = count_1 + np.where(pred_history[-i] == 1, 1, 0)

                count_2 = n - (count_0 + count_1)

                pred_to_use = np.where(
                    (count_2 >= count_1) & (count_2 >= count_0),
                    2,
                    np.where((count_1 >= count_2) & (count_1 >= count_0), 1, 0)
                )
            else:
                pred_to_use = pred

            other_label_mean = np.mean(np.where(
                clustering == 2, 2 * experiment.label_ranking_timeout_weight, clustering
            ), axis=0)
            target_label_mean = np.mean(np.where(
                pred_to_use == 2, 2 * experiment.label_ranking_timeout_weight, pred_to_use
            ))
            label_ranking = other_label_mean <= target_label_mean
            partial_obs_ranking = PartialObservationBasedRanking()(
                y_sampled, runtimes_df, clustering, cluster_boundaries,
                timeout_clf, label_clf, x, y, target_solver, pred,
                experiment, pred_history,
            )
            return np.where(
                np.abs(other_label_mean - target_label_mean) <= self.fallback_thresh,
                partial_obs_ranking,
                label_ranking,
            )
