from typing import List, Tuple

import numpy as np
import pandas as pd


def min_log_dist(i: List[float], j: List[float]) -> float:
    """ Single-link logarithmic distance.

    Args:
        i (float): The first interval.
        j (float): The second interval.

    Returns:
        float: The single-link logarithmic distance.
    """

    fst = np.log1p(i)
    snd = np.log1p(j)
    return np.min(snd) - np.max(fst)


def merge_closest_intervals(
    train_solvers: List[str], runtimes_df: pd.DataFrame, n_intervals=2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Hierarchical merging of the runtimes into clusters.

    Args:
        train_solvers (List[str]): The list of known solvers.
        runtimes_df (pd.DataFrame): The runtimes to cluster.
        n_intervals (int, optional): The number of labels to
        cluster the runtime into. Defaults to 2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of cluster labels
        and the boundaries between the labels.
    """

    merge_runtimes_df = runtimes_df[train_solvers].replace(
        [-np.inf, np.inf, np.nan], 0).astype(dtype=np.int8).copy()
    boundaries = {}

    for idx, (pd_index, instance) in enumerate(list(runtimes_df[train_solvers].iterrows())):
        if np.all(np.isinf(instance)):
            merge_runtimes_df.iloc[idx, :] = n_intervals

            # All solvers have time-out
            decision_boundaries = [
                np.nan,
                np.nan,
            ]
        else:
            actual_n_intervals = min(n_intervals, np.unique(instance).shape[0])
            unique_vals = list(np.sort(np.unique(instance)))
            if unique_vals[-1] == np.inf:
                del unique_vals[-1]
            intervals = [[i] for i in unique_vals]

            # Function to find the smallest distance between neighboring intervals
            def find_smallest_dist_neighbors(intervals):
                min_val = -1
                min_idx = -1
                idx = 0
                for i, j in zip(intervals[:-1], intervals[1:]):
                    val = min_log_dist(i, j)
                    if min_val == -1 or val < min_val:
                        min_val = val
                        min_idx = idx
                    idx += 1
                return min_idx, min_val

            # Repeat merging until desired number of intervals reached
            def merge_recursive(intervals):
                if len(intervals) <= actual_n_intervals:
                    return intervals
                else:
                    min_idx, min_val = find_smallest_dist_neighbors(intervals)
                    intervals[min_idx] += intervals[min_idx + 1]
                    del intervals[min_idx + 1]
                    return merge_recursive(intervals)

            # Compute clustering
            intervals = merge_recursive(intervals)
            curr_cluster_id = 0
            value_cluster_map = {np.inf: n_intervals}
            for interval in intervals:
                for value in interval:
                    value_cluster_map[value] = curr_cluster_id
                curr_cluster_id += 1

            # Compute boundaries of cluster labels
            sorted_labels = sorted(value_cluster_map.items())
            decision_boundaries = []
            for ((val1, int1), (val2, int2)) in zip(sorted_labels[:-1], sorted_labels[1:]):
                if int1 + 1 == int2:
                    # All cluster labels exist, use mean log distance as boundary
                    log_val1 = np.log1p(val1)
                    log_val2 = np.log1p(val2) if not np.isinf(
                        val2) else np.log1p(5000.0)
                    log_center = log_val1 + (log_val2 - log_val1) / 2
                    decision_boundaries.append(np.exp(log_center) - 1)
                elif int1 + 2 == int2:
                    # Only one solver does not time-out, log-divide space for boundaries
                    log_val1 = np.log1p(val1)
                    log_val2 = np.log1p(val2) if not np.isinf(
                        val2) else np.log1p(5000.0)
                    log_center = log_val1 + (log_val2 - log_val1) / 3
                    decision_boundaries.append(np.exp(log_center) - 1)
                    log_center = log_val1 + 2 * (log_val2 - log_val1) / 3
                    decision_boundaries.append(np.exp(log_center) - 1)

            # Output clusters for row
            merge_runtimes_df.iloc[idx, :] = instance.apply(
                lambda i: value_cluster_map[i])

        # Set decision boundary range
        boundaries[pd_index] = decision_boundaries

    return merge_runtimes_df, pd.DataFrame.from_dict(boundaries).transpose()


def assign_labels_from_label_boundaries(
    cluster_boundaries: pd.DataFrame, runtimes_df: pd.DataFrame,
    target_solver: str,
) -> np.ndarray:
    """ Assigns labels to the target solver runtimes based on the cluster boundaries.

    Args:
        cluster_boundaries (pd.DataFrame): The cluster label boundaries.
        runtimes_df (pd.DataFrame): The runtimes.
        target_solver (str): The target solver.

    Returns:
        np.ndarray: An array of labels.
    """

    target_labels = np.zeros(runtimes_df.shape[0], dtype=int)

    for i, ((_, (label1_boundary, label2_boundary)), (_, target_runtime)) in enumerate(zip(
        cluster_boundaries.iterrows(), runtimes_df[target_solver].items()
    )):
        if np.isnan(label1_boundary) and np.isnan(label2_boundary):
            # First solver that solves instance
            target_labels[i] = 2 if np.isinf(target_runtime) else 0
        elif target_runtime <= label1_boundary:
            # Top-tier solver
            target_labels[i] = 0
        else:
            # Put time-outs in bottom-tier
            target_labels[i] = 1 if not np.isinf(target_runtime) else 2

    return target_labels
