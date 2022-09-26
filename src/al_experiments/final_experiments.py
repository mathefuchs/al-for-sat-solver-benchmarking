from typing import List

from al_experiments.experiment import Experiment
from al_experiments.selection import (
    RandomSampling,
    SelectionFunctor,
    ModelUncertaintySampling,
    ModelBasedInformationGainSampling,
)
from al_experiments.ranking import (
    RankingFunctor,
    PartialObservationBasedRanking,
    PredictedClusterLabelRanking,
)
from al_experiments.stopping import (
    StoppingPredicate,
    FixedSubsetSize,
    RankingConverged,
    WilcoxonSignificance,
)


# # Selection strategies (9)
# selection_strategies: List[SelectionFunctor] = []
# selection_strategies.append(RandomSampling())
# for f in [0.0, 0.05, 0.1, 0.15, 0.2]:
#     selection_strategies.append(ModelUncertaintySampling(
#         f, True
#     ))
#     selection_strategies.append(ModelUncertaintySampling(
#         f, False
#     ))
#     selection_strategies.append(ModelBasedInformationGainSampling(
#         f, True,
#     ))
#     selection_strategies.append(ModelBasedInformationGainSampling(
#         f, False,
#     ))

# # Ranking strategies (15)
# ranking_strategies: List[RankingFunctor] = []
# ranking_strategies.append(PartialObservationBasedRanking())
# for fallback_thresh in [0.01, 0.05, 0.1]:
#     # ranking_strategies.append(PartialObservedLabelRanking(fallback_thresh))
#     for num_hist in [1, 10, 20, 30, 40]:
#         ranking_strategies.append(PredictedClusterLabelRanking(
#             True, 0.0, num_hist, fallback_thresh
#         ))
#         # for obs in [0.0, 0.1, 0.2]:
#         #     # ranking_strategies.append(PredictedClusterLabelRanking(
#         #     #     False, obs, 1, fallback_thresh
#         #     # ))
#         #     # ranking_strategies.append(PartialObservedLabelAndPredictionRanking(
#         #     #     False, obs, 1, fallback_thresh
#         #     # ))
#         #     ranking_strategies.append(PartialObservedLabelAndPredictionRanking(
#         #         True, obs, num_hist, fallback_thresh
#         #     ))


# # Stopping criteria (8)
# stopping_criteria: List[StoppingPredicate] = []
# for f in [0.1, 0.2]:
#     stopping_criteria.append(FixedSubsetSize(f))
# for conv in [0.01, 0.02]:
#     for m in [0.02, 0.08, 0.1, 0.12]:
#         stopping_criteria.append(RankingConverged(conv, m))
# for obs in [0.02, 0.08, 0.1, 0.12]:
#     for p_val in [0.05]:
#         for ema in [0.1, 0.7]:
#             stopping_criteria.append(WilcoxonSignificance(obs, p_val, ema))

# # All experiments
# all_experiments: List[Experiment] = []
# for selection in selection_strategies:
#     for ranking in ranking_strategies:
#         for stopping in stopping_criteria:
#             for filter, only_hashes in [
#                 # ("sat_202x_submission_hashes", True),
#                 # ("main_sat_2020_hashes", True),
#                 # ("main_sat_2021_hashes", True),
#                 # ("runtimes_sc2020", False),
#                 # ("runtimes_sc2021", False),
#                 ("anni_train", False),
#             ]:
#                 all_experiments.append(Experiment.custom(
#                     filter, only_hashes, 1,
#                     selection,
#                     stopping,
#                     ranking,
#                 ))

# Run best models on complete dataset
all_experiments: List[Experiment] = []
# delta = 0.00
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0, True),
    RankingConverged(0.01, 0.02),
    PartialObservationBasedRanking(),
))
# delta = 0.05
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0, True),
    RankingConverged(0.02, 0.08),
    PredictedClusterLabelRanking(True, 0.0, 10, 0.01),
))
# delta = 0.10
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0, True),
    RankingConverged(0.02, 0.1),
    PredictedClusterLabelRanking(True, 0.0, 20, 0.05),
))
# delta = 0.15 .. 0.50
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelBasedInformationGainSampling(0, True),
    RankingConverged(0.01, 0.1),
    PredictedClusterLabelRanking(True, 0.0, 10, 0.05),
))
# delta = 0.55
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0.05, True),
    RankingConverged(0.01, 0.12),
    PredictedClusterLabelRanking(True, 0.0, 10, 0.01),
))
# delta = 0.55 .. 0.75
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0.0, False),
    RankingConverged(0.01, 0.02),
    PredictedClusterLabelRanking(True, 0.0, 20, 0.1),
))
# delta = 0.80
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0.0, False),
    RankingConverged(0.01, 0.12),
    PredictedClusterLabelRanking(True, 0.0, 30, 0.05),
))
# delta = 0.85
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0.05, False),
    RankingConverged(0.02, 0.1),
    PredictedClusterLabelRanking(True, 0.0, 10, 0.05),
))
# delta = 0.90
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0.1, False),
    WilcoxonSignificance(0.1, 0.05, 0.1),
    PredictedClusterLabelRanking(True, 0.0, 40, 0.1),
))
# delta = 0.95
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0.0, False),
    WilcoxonSignificance(0.08, 0.05, 0.1),
    PartialObservationBasedRanking(),
))
# delta = 1.00
all_experiments.append(Experiment.custom(
    "anni_final", False, 1,
    ModelUncertaintySampling(0.15, False),
    WilcoxonSignificance(0.08, 0.05, 0.1),
    PartialObservationBasedRanking(),
))
