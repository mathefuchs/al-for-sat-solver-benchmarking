import numpy as np
import pandas as pd
import logging
import sys
import time
import random
import os
import pickle
import warnings

from typing import List, Tuple
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import StackingClassifier, RandomForestClassifier

from al_experiments.config import DEBUG
from al_experiments.clustering import merge_closest_intervals, assign_labels_from_label_boundaries
from al_experiments.evaluation import (
    total_amount_runtime_target_solver,
    mean_wilcoxon_pvalue_predicted_labels,
)
from al_experiments.final_experiments import all_experiments
from al_experiments.experiment import Experiment
from al_experiments.helper import push_notification


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("./run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def run_e2e_experiment(
    i: int, experiment: Experiment, target_solver: str
) -> Tuple[Tuple[int, int, float, float, float, float], np.ndarray]:
    """ Runs a single end-to-end experiment.

    Args:
        i (int): The number of the experiment.
        experiment (Experiment): The experiment configuration.
        target_solver (str): The target solver.

    Returns:
        Tuple[int, int, float, float, float, float]: The performance summary.
    """

    # Features
    with open("../al-for-sat-solver-benchmarking-data/pickled-data/base_features_df.pkl", "rb") as file:
        base_features_df: pd.DataFrame = pickle.load(file).copy()

    if experiment.only_hashes:
        # Load full runtimes of 16 solvers and filter to competition instances
        with open("../al-for-sat-solver-benchmarking-data/pickled-data/runtimes_df.pkl", "rb") as file:
            runtimes_df: pd.DataFrame = pickle.load(file).copy()
        with open(experiment.instance_filter, "rb") as file:
            instance_filter: pd.Series = pickle.load(file).copy()
    else:
        # Load dedicated competition dataset with more solvers
        with open(f"../al-for-sat-solver-benchmarking-data/pickled-data/{experiment.instance_filter_prefix}_df.pkl", "rb") as file:
            runtimes_df = pickle.load(file).copy()
        instance_filter = runtimes_df.index.copy()

    # Notify of experiment start
    len_solvers = len(list(runtimes_df.columns))
    if DEBUG:
        logging.info(
            f'({i+1}/{len_solvers}) - (0/2) Starting experiment '
            f'"{experiment.key}" for target solver "{target_solver}".'
        )

    # Filter instances
    base_features_df = base_features_df.loc[instance_filter].copy()
    runtimes_df = runtimes_df.loc[instance_filter].copy()
    if not np.all(base_features_df.index == instance_filter) \
            or not np.all(runtimes_df.index == instance_filter):
        logging.error("Filtering instances failed.")

    # Feature log-normalization
    lognorm_base_features_df: pd.DataFrame = (
        (np.log1p(base_features_df) - np.log1p(base_features_df).mean())
        / np.log1p(base_features_df).std()
    ).iloc[:, :-1].copy()

    # Train and target solver
    solvers = list(runtimes_df.columns)
    num_test_set_solvers = 0  # hold-out some solvers
    test_solvers = list(np.random.choice(
        solvers, size=num_test_set_solvers, replace=False))
    train_solvers = solvers.copy()
    for s in test_solvers:
        train_solvers.remove(s)
    train_solvers.remove(target_solver)

    # Cluster based on solvers in training set
    clustering, cluster_boundaries = merge_closest_intervals(
        train_solvers, runtimes_df)
    target_labels = assign_labels_from_label_boundaries(
        cluster_boundaries, runtimes_df, target_solver,
    )
    clustering = clustering.copy()  # Reduces fragmentation

    # Log-normalize data based on only solvers in training set
    lognorm_runtimes_sep_timeout_df = runtimes_df[train_solvers].replace(
        [-np.inf, np.inf, np.nan], 0).astype(dtype=np.int8).copy()

    for idx, (_, instance) in enumerate(list(runtimes_df[train_solvers].iterrows())):
        if not np.all(np.isinf(instance)):
            sorted_rt = np.sort(instance.replace([np.inf], np.nan).dropna())
            if sorted_rt.shape[0] != 1 and np.unique(sorted_rt).shape[0] > 1:
                log_sorted = np.log1p(sorted_rt)
                lognormal_sorted = (
                    log_sorted - np.mean(log_sorted)) / np.std(log_sorted)
            else:
                lognormal_sorted = np.array([0.0])
            transform_map = {orig_val: trans_val for orig_val,
                             trans_val in zip(sorted_rt, lognormal_sorted)}
            transform_map[np.inf] = 0.0
        else:
            transform_map = {np.inf: 0.0}

        for solver_idx, runtime in enumerate(instance):
            lognorm_runtimes_sep_timeout_df.iloc[idx, solver_idx] = float(
                transform_map[runtime])

    for col in train_solvers:
        # Add is_timeout column for every solver
        lognorm_runtimes_sep_timeout_df[f"{col}_istimeout"] = np.where(
            np.isinf(runtimes_df[col]), 1, 0)

    # Base and runtime data
    features = pd.concat([
        lognorm_base_features_df,
        lognorm_runtimes_sep_timeout_df
    ], axis=1)
    features = features.copy()  # Reduce fragmentation

    # Add one-hot encoded clustering data
    for solver in train_solvers:
        for lbl in range(2):
            features[f"{solver}_is_{lbl}"] = np.where(
                clustering[solver] == lbl, 1, 0)

    # True Ranking for Evaluation (binary vector indicating against which solvers it is better)
    other_sampled_runtimes: np.ndarray = np.mean(runtimes_df.loc[
        :, runtimes_df.columns != target_solver
    ].replace([np.inf], experiment.par * 5000), axis=0).to_numpy()
    target_sampled_runtimes: np.ndarray = np.mean(runtimes_df.loc[
        :, runtimes_df.columns == target_solver
    ].replace([np.inf], experiment.par * 5000), axis=0).to_numpy()
    true_par2_ranking = other_sampled_runtimes <= target_sampled_runtimes

    # True label-induced ranking
    other_cluster_labels = clustering.loc[
        :, clustering.columns != target_solver
    ]
    other_label_scores: np.ndarray = np.mean(np.where(
        other_cluster_labels == 2,
        2 * experiment.label_ranking_timeout_weight,
        other_cluster_labels
    ), axis=0)
    target_label_score: float = np.mean(np.where(
        target_labels == 2,
        2 * experiment.label_ranking_timeout_weight,
        target_labels
    ))
    true_label_ranking = other_label_scores <= target_label_score

    # True ranking score
    true_par2_runtimes: np.ndarray = np.mean(
        runtimes_df.replace([np.inf], experiment.par * 5000), axis=0
    ).to_numpy()
    true_par2_runtimes_list = enumerate(list(true_par2_runtimes))
    true_par2_runtimes = np.array(
        list(sorted(true_par2_runtimes_list, key=lambda x: (x[1], x[0]))))

    # Remove all columns belonging to the target solver
    # (Avoid data leakage)
    x: np.ndarray = features.loc[:, (
        (features.columns != f"{target_solver}") &
        (features.columns != f"{target_solver}_istimeout") &
        (features.columns != f"{target_solver}_is_0") &
        (features.columns != f"{target_solver}_is_1")
    )].to_numpy()
    y = target_labels
    y_sampled = np.zeros_like(y)

    # Two-level model for discrete runtime prediction
    # Fixed timeout prediction model
    timeout_clf = StackingClassifier(
        estimators=[
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0, tol=0)),
            ("rf", RandomForestClassifier(
                criterion="entropy", class_weight="balanced")),
        ],
        final_estimator=DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=5)
    )

    # Discrete label prediction
    label_clf = StackingClassifier(
        estimators=[
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0, tol=0)),
            ("rf", RandomForestClassifier(criterion="entropy", class_weight="balanced"))
        ],
        final_estimator=DecisionTreeClassifier(
            criterion="gini", splitter="best", max_depth=5)
    )

    # Keep history
    pred_history: List[np.ndarray] = []
    ranking_history: List[np.ndarray] = []
    wilcoxon_history: List[float] = []

    # Add samples until stopping criterion reached
    logging.info(
        f'({i+1}/{len_solvers}) - (1/2) Entering main AL '
        f'loop of experiment "{experiment.key}" '
        f'for target solver "{target_solver}".'
    )
    while not experiment.stopping(
        y_sampled, runtimes_df, clustering, cluster_boundaries,
        timeout_clf, label_clf, x, y, target_solver, experiment,
        ranking_history, wilcoxon_history,
    ) and np.count_nonzero(y_sampled) < y_sampled.shape[0]:
        # Select sample
        sampled_instances_before = int(np.count_nonzero(y_sampled))
        experiment.selection(
            y_sampled, runtimes_df, clustering, cluster_boundaries,
            timeout_clf, label_clf, x, y, target_solver, experiment
        )
        if sampled_instances_before >= int(np.count_nonzero(y_sampled)):
            logging.error(
                f'({i+1}/{len_solvers}) - (FATAL) Sampling did not '
                f'succeed to select next instance in experiment '
                f'"{experiment.key}" for target solver "{target_solver}".'
            )
            break

        # Update p
        p = np.count_nonzero(y_sampled) / y_sampled.shape[0]

        # Split which data is marked selected and which not
        x_train: np.ndarray = x[y_sampled == 1]
        x_test: np.ndarray = x[y_sampled == 0]
        y_train: np.ndarray = y[y_sampled == 1]
        y_test: np.ndarray = y[y_sampled == 0]

        # Permute sampled data
        train_perm = np.random.permutation(x_train.shape[0])
        x_train = x_train[train_perm].copy()
        y_train = y_train[train_perm].copy()
        test_perm = np.random.permutation(x_test.shape[0])
        x_test = x_test[test_perm].copy()
        y_test = y_test[test_perm].copy()

        # Separate training data for each level of the model
        y_train_timeout = np.where(y_train == 2, 1, 0)
        non_timeout = (y_train != 2)
        x_train_non_timeout = x_train[non_timeout]
        y_train_non_timeout = y_train[non_timeout]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Train timeout predictor
                timeout_clf.fit(x_train, y_train_timeout)

                # Train discrete label predictor
                label_clf.fit(x_train_non_timeout, y_train_non_timeout)
        except ValueError:
            if not experiment.ranking.needs_predictions:
                current_ranking = experiment.ranking(
                    y_sampled, runtimes_df, clustering, cluster_boundaries,
                    timeout_clf, label_clf, x, y, target_solver,
                    np.array([]), experiment, pred_history
                )
                ranking_history.append(current_ranking)
                amount_correct_par2_ranking = max(
                    0, (
                        np.count_nonzero(current_ranking == true_par2_ranking) /
                        true_par2_ranking.shape[0]
                    )
                )
                amount_correct_label_ranking = max(
                    0, (
                        np.count_nonzero(current_ranking == true_label_ranking) /
                        true_label_ranking.shape[0]
                    )
                )
            else:
                amount_correct_par2_ranking = 0.0
                amount_correct_label_ranking = 0.0

            perf_tuple = (
                i,
                np.count_nonzero(y_sampled),
                total_amount_runtime_target_solver(
                    y_sampled, runtimes_df, cluster_boundaries,
                    target_solver, experiment,
                ),
                amount_correct_par2_ranking,
                amount_correct_label_ranking,
                0.0,
            )
            continue

        # Predict test set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            is_timeout_pred = timeout_clf.predict(x)
            label_pred = label_clf.predict(x)
        pred = np.where(is_timeout_pred == 1, 2, label_pred)
        pred_history.append(pred)

        # Ranking evaluation
        current_ranking = experiment.ranking(
            y_sampled, runtimes_df, clustering, cluster_boundaries,
            timeout_clf, label_clf, x, y, target_solver, pred,
            experiment, pred_history,
        )
        ranking_history.append(current_ranking)
        amount_correct_par2_ranking = max(
            0, (
                np.count_nonzero(current_ranking == true_par2_ranking) /
                true_par2_ranking.shape[0]
            )
        )
        amount_correct_label_ranking = max(
            0, (
                np.count_nonzero(current_ranking == true_label_ranking) /
                true_label_ranking.shape[0]
            )
        )

        w = mean_wilcoxon_pvalue_predicted_labels(
            clustering, pred, y, y_sampled, target_solver, experiment
        )
        wilcoxon_history.append(w)

        # Spearman correlation
        current_ranking_score: float = np.mean(np.where(
            pred == 2,
            2 * experiment.label_ranking_timeout_weight,
            pred,
        ))
        current_ranking_scores = np.zeros_like(true_par2_runtimes[:, 1])
        for j in range(len_solvers):
            scores_idx = np.where(true_par2_runtimes[:, 0] == j)[0][0]
            if j == i:  # Target solver
                current_ranking_scores[scores_idx] = current_ranking_score
            elif j > i:
                current_ranking_scores[scores_idx] = other_label_scores[j - 1]
            else:
                current_ranking_scores[scores_idx] = other_label_scores[j]
        spearman_correlation = spearmanr(
            true_par2_runtimes[:, 1], current_ranking_scores)[0]

        perf_tuple = (
            i,
            np.count_nonzero(y_sampled),
            total_amount_runtime_target_solver(
                y_sampled, runtimes_df, cluster_boundaries,
                target_solver, experiment,
            ),
            amount_correct_par2_ranking,
            amount_correct_label_ranking,
            spearman_correlation,
        )
        if DEBUG:
            logging.info((
                "Solver: {}, "
                "Number Instances: {}, "
                "Amount Runtime: {:.6f}, "
                "Accuracy PAR2 Ranking: {:.6f}, "
                "Accuracy Label Ranking: {:.6f}, "
                "Spearman correlation: {:.6f}"
            ).format(*perf_tuple))

    logging.info(
        f'({i+1}/{len_solvers}) - (2/2) Finished AL '
        f'experiment "{experiment.key}" '
        f'for target solver "{target_solver}".'
    )

    return (perf_tuple, y_sampled)


if __name__ == "__main__":
    # Notify experiment start
    push_notification("Starting experiments.")
    rand_int = random.randint(1, 100)

    for i_exp, experiment in enumerate(all_experiments):
        # Skip if present
        time.sleep(random.random() * 5)
        if os.path.exists(f"{experiment.results_location}_{rand_int:03}.csv"):
            continue
        else:
            with open(f"{experiment.results_location}_{rand_int:03}.csv", "w") as lock_file:
                lock_file.write("lock")

        # Retrieve column list
        if experiment.only_hashes:
            with open("../al-for-sat-solver-benchmarking-data/pickled-data/runtimes_df.pkl", "rb") as file:
                runtimes_df = pickle.load(file).copy()
        else:
            with open(f"../al-for-sat-solver-benchmarking-data/pickled-data/{experiment.instance_filter_prefix}_df.pkl", "rb") as file:
                runtimes_df = pickle.load(file).copy()
        column_list = list(runtimes_df.columns)
        del runtimes_df

        # Run cross-validation with each solver as target once
        if DEBUG:
            solver_results = [
                run_e2e_experiment(i, experiment, target_solver)
                for i, target_solver in enumerate(column_list)
            ]
        else:
            solver_results = Parallel(n_jobs=-1)(
                delayed(run_e2e_experiment)(
                    i, experiment, target_solver
                )
                for i, target_solver in enumerate(column_list)
                for _ in range(experiment.repetitions)
            )

        # Store results
        res_df = pd.DataFrame.from_records(
            [t for t, _ in solver_results],
            columns=[
                "solver", "num_instances", "amount_runtime",
                "par2_ranking_acc", "label_ranking_acc",
                "spearman"
            ],
            index="solver",
        )
        res_df.to_csv(f"{experiment.results_location}_{rand_int:03}.csv")

        # Store sampled instances
        for i, (_, y_sampled) in enumerate(solver_results):
            with open(f"../al-for-sat-solver-benchmarking-data/pickled-data/end_to_end/{experiment.key}_y_sampled_{i:02d}.pkl", "wb") as wfile:
                pickle.dump(y_sampled.copy(), wfile)

        if len(all_experiments) <= 50 or i_exp % 100 == 0:
            push_notification(
                f"{(i_exp+1)}/{len(all_experiments)} = {(i_exp/len(all_experiments)):.2f} ({experiment.key})."
            )
