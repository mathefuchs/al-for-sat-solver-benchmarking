from al_experiments.selection import (
    SelectionFunctor,
    RandomSampling,
    ModelUncertaintySampling,
    ModelBasedInformationGainSampling,
    VotingEntropyBasedCommitteeDisagreementSampling,
    EntropyBasedUncertaintySampling,
    FastRuntimeSampling,
    VarianceRuntimePerRuntimeSampling,
)
from al_experiments.stopping import StoppingPredicate, FixedSubsetSize
from al_experiments.ranking import (
    RankingFunctor,
    PartialObservationBasedRanking,
    PredictedClusterLabelRanking,
    PartialObservedLabelRanking,
    PartialObservedLabelAndPredictionRanking,
)


class Experiment:
    """ Configuration class for a single experiment. """

    def __init__(
        self, key: str, description: str,
        selection: SelectionFunctor, stopping: StoppingPredicate, ranking: RankingFunctor,
        repetitions: int = 1, par: float = 2,
        instance_filter: str = "sat_202x_submission_hashes", only_hashes: bool = True,
    ):
        """ Initializes an experiment configuration.

        Args:
            key (str): The key of the experiment.
            description (str): The description.
            selection (SelectionFunctor): The selection functor.
            stopping (StoppingPredicate): The stopping predicate.
            ranking (RankingFunctor): The ranking functor.
            repetitions (int, optional): The number of repetitions. Defaults to 1.
            par (float, optional): The PAR value. Defaults to 2.
            instance_filter (str, optional): Which instances to use. Defaults to "sat_202x_submission_hashes".
        """

        self.key = key
        self.description = description
        self.selection = selection
        self.stopping = stopping
        self.ranking = ranking
        self.results_location = f"../al-for-sat-solver-benchmarking-data/pickled-data/end_to_end/{key}"
        self.repetitions = repetitions
        self.par = par
        self.instance_filter_prefix = instance_filter
        self.instance_filter = f"../al-for-sat-solver-benchmarking-data/pickled-data/instance_filters/{instance_filter}.pkl"
        self.label_ranking_timeout_weight = 2.0
        self.only_hashes = only_hashes

    @classmethod
    def simple_baseline(
        cls,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Simple baseline configuration.

        Args:
            instance_filter (str): Which instances to use.
            repetitions (int): Repetitions of random sampling.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_random_fixed_observed",
            description=(
                "Random selection with fixed subset size stopping "
                "criterion and ranking based on observed runtimes."
            ),
            selection=RandomSampling(),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservationBasedRanking(),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def simple_baseline_batch(
        cls,
        instance_filter: str,
        repetitions: int,
        batch: int,
    ) -> "Experiment":
        """ Simple baseline configuration.

        Args:
            instance_filter (str): Which instances to use.
            repetitions (int): Repetitions of random sampling.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_random_fixed_observed_batch",
            description=(
                "Random selection with fixed subset size stopping "
                "criterion and ranking based on observed runtimes."
            ),
            selection=RandomSampling(batch),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservationBasedRanking(),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def fast_runtime_baseline(
        cls,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Baseline selecting the fastest non-sampled experiment.

        Args:
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_fast_runtime_baseline",
            description=(
                "Iteratively select the instances with the fastest runtime."
            ),
            selection=FastRuntimeSampling(),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservationBasedRanking(),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def variance_per_runtime_observed_baseline(
        cls,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Baseline selecting the highest runtime variance per runtime experiment.

        Args:
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_variance_per_runtime_observed_baseline",
            description=(
                "Iteratively select the instances with the highest variance per runtime."
            ),
            selection=VarianceRuntimePerRuntimeSampling(),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservationBasedRanking(),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def partial_observed_pred_labels(
        cls,
        instance_filter: str,
        repetitions: int,
        batch: int,
    ) -> "Experiment":
        """ Ranking based on partially observed labels.

        Args:
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_partial_observed_pred_labels",
            description=(
                "Ranking based on partially observed labels."
            ),
            selection=RandomSampling(batch),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservedLabelRanking(0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def partial_observed_and_predictions(
        cls,
        instance_filter: str,
        repetitions: int,
        batch: int,
    ) -> "Experiment":
        """ Ranking based on partially observed labels and predictions.

        Args:
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_partial_observed_and_predictions",
            description=(
                "Ranking based on partially observed labels and predictions."
            ),
            selection=RandomSampling(batch),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservedLabelAndPredictionRanking(
                False, 0.2, 10, 0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def uncertainty_fixed_subset_observed(
        cls,
        model_based_selection_threshold: float,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Uncertainty-based sampling, fixed subset,
        partially observed runtimes ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_uncertainty_{model_based_selection_threshold:.2f}_fixed_subset_observed",
            description=(
                "Uncertainty-based sampling, "
                "fixed subset, "
                "partially observed runtimes ranking."
            ),
            selection=ModelUncertaintySampling(
                model_based_selection_threshold, True),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservationBasedRanking(),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def uncertainty_fixed_subset_pred_labels(
        cls,
        model_based_selection_threshold: float,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Uncertainty-based sampling, fixed subset,
        predicted labels ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_uncertainty_{model_based_selection_threshold:.2f}_fixed_subset_pred_labels",
            description=(
                "Uncertainty-based sampling, "
                "fixed subset, "
                "predicted labels ranking."
            ),
            selection=ModelUncertaintySampling(
                model_based_selection_threshold, True),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PredictedClusterLabelRanking(False, 0.2, 10, 0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def uncertainty_partial_pred_labels(
        cls,
        model_based_selection_threshold: float,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Uncertainty-based sampling, fixed subset,
        predicted labels ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_uncertainty_{model_based_selection_threshold:.2f}_partial_pred_labels",
            description=(
                "Uncertainty-based sampling, "
                "fixed subset, "
                "predicted labels ranking."
            ),
            selection=ModelUncertaintySampling(
                model_based_selection_threshold, True),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservedLabelAndPredictionRanking(
                True, 0.2, 10, 0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def uncertainty_partial_pred_labels_no_time_scaling(
        cls,
        model_based_selection_threshold: float,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Uncertainty-based sampling, fixed subset,
        predicted labels ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_uncertainty_{model_based_selection_threshold:.2f}_partial_pred_labels_no_time_scaling",
            description=(
                "Uncertainty-based sampling, "
                "fixed subset, "
                "predicted labels ranking."
            ),
            selection=ModelUncertaintySampling(
                model_based_selection_threshold, False),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PartialObservedLabelAndPredictionRanking(
                True, 0.2, 10, 0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def model_ig_fixed_subset_pred_labels(
        cls,
        model_based_selection_threshold: float,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Model-Information-Gain sampling, fixed subset,
        predicted labels ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_model_ig_{model_based_selection_threshold:.2f}_fixed_subset_pred_labels",
            description=(
                "Model-Information-Gain-based sampling, "
                "fixed subset, "
                "predicted labels ranking."
            ),
            selection=ModelBasedInformationGainSampling(
                model_based_selection_threshold, True),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PredictedClusterLabelRanking(False, 0.2, 10, 0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def voting_entropy_fixed_subset_pred_labels(
        cls,
        model_based_selection_threshold: float,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Voting-Entropy-Committee-based sampling, fixed subset,
        predicted labels ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_voting_entropy_{model_based_selection_threshold:.2f}_fixed_subset_pred_labels",
            description=(
                "Voting-Entropy-Committee-based sampling, "
                "fixed subset, "
                "predicted labels ranking."
            ),
            selection=VotingEntropyBasedCommitteeDisagreementSampling(
                model_based_selection_threshold, True),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PredictedClusterLabelRanking(False, 0.2, 10, 0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def entropy_uncertainty_fixed_subset_pred_labels(
        cls,
        model_based_selection_threshold: float,
        instance_filter: str,
        repetitions: int,
    ) -> "Experiment":
        """ Entropy-based Uncertainty sampling, fixed subset,
        predicted labels ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=f"{instance_filter}_entropy_uncertainty_{model_based_selection_threshold:.2f}_fixed_subset_pred_labels",
            description=(
                "Entropy-based Uncertainty sampling, "
                "fixed subset, "
                "predicted labels ranking."
            ),
            selection=EntropyBasedUncertaintySampling(
                model_based_selection_threshold, True),
            stopping=FixedSubsetSize(0.95),  # TODO
            ranking=PredictedClusterLabelRanking(False, 0.2, 10, 0.05),
            instance_filter=instance_filter,
            repetitions=repetitions,
        )

    @classmethod
    def custom(
        cls,
        instance_filter: str,
        only_hashes: bool,
        repetitions: int,
        selection: SelectionFunctor,
        stopping: StoppingPredicate,
        ranking: RankingFunctor,
    ) -> "Experiment":
        """ Entropy-based Uncertainty sampling, fixed subset,
        predicted labels ranking.

        Args:
            model_based_selection_threshold (float): When to switch to the model-based
            selection criterion if supported
            instance_filter (str): Which instances to use.

        Returns:
            Experiment: The experiment configuration.
        """

        return cls(
            key=(
                f"custom_{instance_filter}"
                f"_{selection.get_params()}"
                f"_{stopping.get_params()}"
                f"_{ranking.get_params()}"
            ),
            description=(
                "Custom experiment."
            ),
            selection=selection,
            stopping=stopping,
            ranking=ranking,
            instance_filter=instance_filter,
            only_hashes=only_hashes,
            repetitions=repetitions,
        )
