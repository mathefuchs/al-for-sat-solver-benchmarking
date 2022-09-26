from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from scipy.stats import entropy
from sklearn.ensemble import StackingClassifier


class SelectionFunctor(ABC):
    """ Base class for instance selection making. """

    def __init__(self, model_based_selection_threshold: float) -> None:
        self.model_based_selection_threshold = model_based_selection_threshold

    @abstractmethod
    def get_params(self) -> str:
        return ""

    @abstractmethod
    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        pass


class RandomSampling(SelectionFunctor):
    """ Random sampling. """

    def __init__(self, batch: int = 1) -> None:
        super().__init__(0.0)
        self.batch = batch

    def get_params(self) -> str:
        return f"random_{self.batch}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        for _ in range(self.batch):
            if np.count_nonzero(y_sampled) == y_sampled.shape[0]:
                break

            sampled_index = np.random.choice(
                np.arange(0, y_sampled.shape[0])[y_sampled == 0])
            y_sampled[sampled_index] = 1


class FastRuntimeSampling(SelectionFunctor):
    """ Select-fast-instances-first sampling. """

    def __init__(self) -> None:
        super().__init__(0.0)

    def get_params(self) -> str:
        return "fast"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        if np.count_nonzero(y_sampled) % 2 == 0:
            sampled_index = np.random.choice(
                np.arange(0, y_sampled.shape[0])[y_sampled == 0])
            y_sampled[sampled_index] = 1
        else:
            # Available runtimes
            runtimes = np.mean(np.log1p(runtimes_df.loc[:, (runtimes_df.columns != f"{target_solver}")].replace(
                [np.inf], experiment.par * 5000
            )), axis=1)
            best_score_index = np.argmin(runtimes[y_sampled == 0])
            y_sampled_index = np.arange(0, y_sampled.shape[0])[
                y_sampled == 0][best_score_index]
            y_sampled[y_sampled_index] = 1


class VarianceRuntimePerRuntimeSampling(SelectionFunctor):
    """ Selects instances with highest runtime variance per runtime. """

    def __init__(self) -> None:
        super().__init__(0.0)

    def get_params(self) -> str:
        return "var"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        # Available runtimes
        log_par2_runtimes = np.log1p(
            runtimes_df.loc[:, (runtimes_df.columns != f"{target_solver}")].replace(
                [np.inf], experiment.par * 5000
            )
        )
        variance_score = (
            np.var(log_par2_runtimes, axis=1) /
            (np.mean(log_par2_runtimes, axis=1) + 1)
        )

        if np.count_nonzero(y_sampled) % 2 == 0:
            sampled_index = np.random.choice(
                np.arange(0, y_sampled.shape[0])[y_sampled == 0])
            y_sampled[sampled_index] = 1
        else:
            best_score_index = np.argmin(variance_score[y_sampled == 0])
            y_sampled_index = np.arange(0, y_sampled.shape[0])[
                y_sampled == 0][best_score_index]
            y_sampled[y_sampled_index] = 1


class ModelUncertaintySampling(SelectionFunctor):
    """ Sampling based on the uncertainty of the model. """

    def __init__(
        self, model_based_selection_threshold: float, time_scaling: bool
    ) -> None:
        super().__init__(model_based_selection_threshold)
        self.time_scaling = time_scaling

    def get_params(self) -> str:
        return f"unc_{self.model_based_selection_threshold:.2f}_{self.time_scaling}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        # Compute log variance
        log_par2_runtimes = np.log1p(
            runtimes_df.loc[:, (runtimes_df.columns != f"{target_solver}")].replace(
                [np.inf], experiment.par * 5000)
        )

        # Solely use variance-based sampling for first decisions
        if np.count_nonzero(y_sampled) / y_sampled.shape[0] < self.model_based_selection_threshold:
            # Pick best score
            sampled_index = np.random.choice(
                np.arange(0, y_sampled.shape[0])[y_sampled == 0])
            y_sampled[sampled_index] = 1
            # print("Selected", y_sampled_index, "with score", variance_score[y_sampled_index])
        # Simple uncertainty
        else:
            try:
                timeout_pred_class_probabilities = np.sum(
                    np.abs(timeout_clf.transform(x) - 0.5), axis=1)
                label_pred_class_probabilities = np.sum(
                    np.abs(label_clf.transform(x) - 0.5), axis=1)
                if self.time_scaling:
                    score = (
                        (timeout_pred_class_probabilities +
                         label_pred_class_probabilities)
                        * np.mean(log_par2_runtimes, axis=1)
                    )
                else:
                    score = (
                        timeout_pred_class_probabilities +
                        label_pred_class_probabilities
                    )

                # Pick best score
                best_score_index = np.argmin(score[y_sampled == 0])
                y_sampled_index = np.arange(0, y_sampled.shape[0])[
                    y_sampled == 0][best_score_index]
                # print(np.count_nonzero(y_sampled), y_sampled[y_sampled_index])
                y_sampled[y_sampled_index] = 1
                # print("Selected", y_sampled_index, "with score", score[y_sampled_index])
            except (ValueError, AttributeError):
                # No properly trained model present, use fallback
                # if np.count_nonzero(y_sampled) % 2 == 0:
                sampled_index = np.random.choice(
                    np.arange(0, y_sampled.shape[0])[y_sampled == 0])
                y_sampled[sampled_index] = 1
                # else:
                #     best_score_index = np.argmax(
                #         variance_score[y_sampled == 0])
                #     y_sampled_index = np.arange(0, y_sampled.shape[0])[
                #         y_sampled == 0][best_score_index]
                #     y_sampled[y_sampled_index] = 1


class ModelBasedInformationGainSampling(SelectionFunctor):
    """ Sampling based on Model Information-Gain Metric. """

    def __init__(
        self, model_based_selection_threshold: float, time_scaling: bool
    ) -> None:
        super().__init__(model_based_selection_threshold)
        self.time_scaling = time_scaling

    def get_params(self) -> str:
        return f"ig_{self.model_based_selection_threshold:.2f}_{self.time_scaling}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        # Compute log variance
        log_par2_runtimes = np.log1p(
            runtimes_df.loc[:, (runtimes_df.columns != f"{target_solver}")].replace(
                [np.inf], experiment.par * 5000)
        )

        # Compute number of solvers per label and instance
        labels = clustering.loc[:, (clustering.columns != f"{target_solver}")]
        num_labels = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=3),
            axis=1,
            arr=labels
        )

        # Solely use variance-based sampling for first decisions
        if np.count_nonzero(y_sampled) / y_sampled.shape[0] < self.model_based_selection_threshold:
            # Pick best score
            sampled_index = np.random.choice(
                np.arange(0, y_sampled.shape[0])[y_sampled == 0])
            y_sampled[sampled_index] = 1
            # print("Selected", y_sampled_index, "with score", variance_score[y_sampled_index])
        else:
            try:
                timeout_pred = np.mean(timeout_clf.transform(x), axis=1)
                label_pred = np.mean(label_clf.transform(x), axis=1)
                label_0 = (1 - timeout_pred) * (1 - label_pred)
                label_1 = (1 - timeout_pred) * label_pred
                label_2 = timeout_pred

                # Information-gain
                information_gain = np.zeros(num_labels.shape[0])
                for i, label_counts in enumerate(num_labels):
                    label_prob_if_0 = (
                        label_counts + np.array([1, 0, 0])) / (labels.shape[1] + 1)
                    label_prob_if_1 = (
                        label_counts + np.array([0, 1, 0])) / (labels.shape[1] + 1)
                    label_prob_if_2 = (
                        label_counts + np.array([0, 0, 1])) / (labels.shape[1] + 1)
                    entropy_before = entropy(label_counts / labels.shape[1])
                    split_entropy = (
                        label_0[i] * entropy(label_prob_if_0)
                        + label_1[i] * entropy(label_prob_if_1)
                        + label_2[i] * entropy(label_prob_if_2)
                    )
                    information_gain[i] = split_entropy - entropy_before

                # Score
                if self.time_scaling:
                    score = (
                        information_gain /
                            (np.mean(log_par2_runtimes, axis=1) + 1)
                    )
                else:
                    score = information_gain

                # Select instance with most information-gain
                best_score_index = np.argmax(score[y_sampled == 0])
                y_sampled_index = np.arange(0, y_sampled.shape[0])[
                    y_sampled == 0][best_score_index]
                # print(np.count_nonzero(y_sampled), y_sampled[y_sampled_index])
                y_sampled[y_sampled_index] = 1
                # print("Selected", y_sampled_index, "with score", score[y_sampled_index])
            except (ValueError, AttributeError) as e:
                # No properly trained model present, use fallback
                sampled_index = np.random.choice(
                    np.arange(0, y_sampled.shape[0])[y_sampled == 0])
                y_sampled[sampled_index] = 1


class VotingEntropyBasedCommitteeDisagreementSampling(SelectionFunctor):
    """ Sampling based on Voting-Entropy-based Committee Disagreement. """

    def __init__(
        self, model_based_selection_threshold: float, time_scaling: bool
    ) -> None:
        super().__init__(model_based_selection_threshold)
        self.time_scaling = time_scaling

    def get_params(self) -> str:
        return f"voting_{self.model_based_selection_threshold:.2f}_{self.time_scaling}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        # Compute log variance
        log_par2_runtimes = np.log1p(
            runtimes_df.loc[:, (runtimes_df.columns != f"{target_solver}")].replace(
                [np.inf], experiment.par * 5000)
        )
        variance_score = (
            np.var(log_par2_runtimes, axis=1) /
            (np.mean(log_par2_runtimes, axis=1) + 1)
        )

        # Solely use variance-based sampling for first decisions
        if np.count_nonzero(y_sampled) / y_sampled.shape[0] < self.model_based_selection_threshold:
            # Pick best score
            best_score_index = np.argmax(variance_score[y_sampled == 0])
            y_sampled_index = np.arange(0, y_sampled.shape[0])[
                y_sampled == 0][best_score_index]
            # print(np.count_nonzero(y_sampled), y_sampled[y_sampled_index])
            y_sampled[y_sampled_index] = 1
            # print("Selected", y_sampled_index, "with score", variance_score[y_sampled_index])
        else:
            try:
                timeout_pred_count = np.count_nonzero(
                    timeout_clf.transform(x) >= 0.5, axis=1).reshape((-1, 1))
                label_pred_count = np.count_nonzero(
                    label_clf.transform(x) >= 0.5, axis=1).reshape((-1, 1))
                timeout_entropy = entropy(np.hstack([
                    timeout_pred_count / 2,
                    (2 - timeout_pred_count) / 2,
                ]), axis=1)
                label_entropy = entropy(np.hstack([
                    label_pred_count / 2,
                    (2 - label_pred_count) / 2,
                ]), axis=1)
                score = (
                    (timeout_entropy + label_entropy)
                    / np.mean(log_par2_runtimes, axis=1)
                )

                # Pick best score
                best_score_index = np.argmax(score[y_sampled == 0])
                y_sampled_index = np.arange(0, y_sampled.shape[0])[
                    y_sampled == 0][best_score_index]
                # print(np.count_nonzero(y_sampled), y_sampled[y_sampled_index])
                y_sampled[y_sampled_index] = 1
                # print("Selected", y_sampled_index, "with score", score[y_sampled_index])
            except (ValueError, AttributeError) as e:
                # No properly trained model present, use fallback
                if np.count_nonzero(y_sampled) % 2 == 0:
                    sampled_index = np.random.choice(
                        np.arange(0, y_sampled.shape[0])[y_sampled == 0])
                    y_sampled[sampled_index] = 1
                else:
                    best_score_index = np.argmax(
                        variance_score[y_sampled == 0])
                    y_sampled_index = np.arange(0, y_sampled.shape[0])[
                        y_sampled == 0][best_score_index]
                    y_sampled[y_sampled_index] = 1


class EntropyBasedUncertaintySampling(SelectionFunctor):
    """ Entropy-based Uncertainty Sampling. """

    def __init__(
        self, model_based_selection_threshold: float, time_scaling: bool
    ) -> None:
        super().__init__(model_based_selection_threshold)
        self.time_scaling = time_scaling

    def get_params(self) -> str:
        return f"entr_unc_{self.model_based_selection_threshold:.2f}_{self.time_scaling}"

    def __call__(
        self, y_sampled: np.ndarray, runtimes_df: pd.DataFrame,
        clustering: pd.DataFrame, cluster_boundaries: pd.DataFrame,
        timeout_clf: StackingClassifier, label_clf: StackingClassifier,
        x: np.ndarray, y: np.ndarray, target_solver: str, experiment
    ) -> None:
        """ Selects another instance in 'y_sampled'.

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
        """

        # Compute log variance
        log_par2_runtimes = np.log1p(
            runtimes_df.loc[:, (runtimes_df.columns != f"{target_solver}")].replace(
                [np.inf], experiment.par * 5000)
        )
        variance_score = (
            np.var(log_par2_runtimes, axis=1) /
            (np.mean(log_par2_runtimes, axis=1) + 1)
        )

        # Solely use variance-based sampling for first decisions
        if np.count_nonzero(y_sampled) / y_sampled.shape[0] < self.model_based_selection_threshold:
            # Pick best score
            best_score_index = np.argmax(variance_score[y_sampled == 0])
            y_sampled_index = np.arange(0, y_sampled.shape[0])[
                y_sampled == 0][best_score_index]
            # print(np.count_nonzero(y_sampled), y_sampled[y_sampled_index])
            y_sampled[y_sampled_index] = 1
            # print("Selected", y_sampled_index, "with score", variance_score[y_sampled_index])
        else:
            try:
                timeout_pred_class_probabilities = np.mean(
                    timeout_clf.transform(x), axis=1).reshape((-1, 1))
                timeout_entropy = entropy(np.hstack([
                    timeout_pred_class_probabilities,
                    (1 - timeout_pred_class_probabilities),
                ]), axis=1)
                label_pred_class_probabilities = np.mean(
                    label_clf.transform(x), axis=1).reshape((-1, 1))
                label_entropy = entropy(np.hstack([
                    label_pred_class_probabilities,
                    (1 - label_pred_class_probabilities),
                ]), axis=1)
                score = (
                    (timeout_entropy + label_entropy)
                    / np.mean(log_par2_runtimes, axis=1)
                )

                # Pick best score
                best_score_index = np.argmax(score[y_sampled == 0])
                y_sampled_index = np.arange(0, y_sampled.shape[0])[
                    y_sampled == 0][best_score_index]
                # print(np.count_nonzero(y_sampled), y_sampled[y_sampled_index])
                y_sampled[y_sampled_index] = 1
                # print("Selected", y_sampled_index, "with score", score[y_sampled_index])
            except (ValueError, AttributeError):
                # No properly trained model present, use fallback
                if np.count_nonzero(y_sampled) % 2 == 0:
                    sampled_index = np.random.choice(
                        np.arange(0, y_sampled.shape[0])[y_sampled == 0])
                    y_sampled[sampled_index] = 1
                else:
                    best_score_index = np.argmax(
                        variance_score[y_sampled == 0])
                    y_sampled_index = np.arange(0, y_sampled.shape[0])[
                        y_sampled == 0][best_score_index]
                    y_sampled[y_sampled_index] = 1
