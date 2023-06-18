#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import io
import time
import csv
import os
from pathlib import Path
from math import sqrt
from itertools import permutations
from typing import Sequence, Union, Iterator

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
import pandas as pd
import scipy
import scipy.spatial as spatial
from scipy.special import betaln, expit, gammaln
from scipy.sparse import csr_matrix
from unidecode import unidecode


EPS = np.finfo(float).eps


PathLike = Union[str, Path]
"""Convenience type for cases where `str` or `Path` are acceptable types."""



def encode_cluster(cluster: NDArray[bool]) -> str:
    """Format the given cluster as a compact bit-string."""
    cluster_s = cluster.astype(int).astype(str)
    return ''.join(cluster_s)


def decode_cluster(cluster_str: str) -> NDArray[bool]:
    """Read a bit-string and parse it into an area array."""
    return np.array(list(cluster_str)).astype(int).astype(bool)


def format_cluster_columns(clusters: NDArray[bool]) -> str:
    """Format the given array of clusters as tab separated strings."""
    clusters_encoded = map(encode_cluster, clusters)
    return '\t'.join(clusters_encoded)


def parse_cluster_columns(clusters_encoded: str) -> NDArray[bool]:
    """Read tab-separated area encodings into a two-dimensional area array."""
    clusters_decoded = map(decode_cluster, clusters_encoded.split('\t'))
    return np.array(list(clusters_decoded))


def compute_distance(a, b):
    """ This function computes the Euclidean distance between two points a and b

    Args:
        a (list): The x and y coordinates of a point in a metric CRS.
        b (list): The x and y coordinates of a point in a metric CRS.

    Returns:
        float: Distance between a and b
    """

    a = np.asarray(a)
    b = np.asarray(b)
    ab = b-a
    dist = sqrt(ab[0]**2 + ab[1]**2)

    return dist


def bounding_box(points):
    """ This function retrieves the bounding box for a set of 2-dimensional input points

    Args:
        points (numpy.array): Point tuples (x,y) for which the bounding box is computed
    Returns:
        (dict): the bounding box of the points
    """
    x = [x[0] for x in points]
    y = [x[1] for x in points]
    box = {'x_max': max(x),
           'y_max': max(y),
           'x_min': min(x),
           'y_min': min(y)}

    return box


def get_neighbours(cluster, already_in_cluster, adjacency_matrix):
    """This function returns the neighbourhood of a cluster as given in the adjacency_matrix, excluding sites already
    belonging to this or any other cluster.

    Args:
        cluster (np.array): The current cluster (boolean array)
        already_in_cluster (np.array): All sites already assigned to a cluster (boolean array)
        adjacency_matrix (np.array): The adjacency matrix of the sites (boolean)

    Returns:
        np.array: The neighborhood of the cluster (boolean array)
    """

    # Get all neighbors of the current zone, excluding all vertices that are already in a zone

    neighbours = np.logical_and(adjacency_matrix.dot(cluster), ~already_in_cluster)
    return neighbours


def compute_delaunay(locations):
    """Computes the Delaunay triangulation between a set of point locations

    Args:
        locations (np.array): a set of locations
            shape (n_sites, n_spatial_dims = 2)
    Returns:
        (np.array) sparse matrix of Delaunay triangulation
            shape (n_edges, n_edges)
    """
    n = len(locations)

    if n < 4:
        # scipy's Delaunay triangulation fails for <3. Return a fully connected graph:
        return csr_matrix(1-np.eye(n, dtype=int))

    delaunay = spatial.Delaunay(locations, qhull_options="QJ Pp")

    indptr, indices = delaunay.vertex_neighbor_vertices
    data = np.ones_like(indices)

    return csr_matrix((data, indices, indptr), shape=(n, n))


def gabriel_graph_from_delaunay(delaunay, locations):
    delaunay = delaunay.toarray()
    # converting delaunay graph to boolean array denoting whether points are connected
    delaunay = delaunay > 0

    # Delaunay indices and locations
    delaunay_connections = []
    delaunay_locations = []

    for index, connected in np.ndenumerate(delaunay):
        if connected:
            # getting indices of points in area
            i1, i2 = index[0], index[1]
            if [i2, i1] not in delaunay_connections:
                delaunay_connections.append([i1, i2])
                delaunay_locations.append(locations[[*[i1, i2]]])
    delaunay_connections = np.sort(np.asarray(delaunay_connections), axis=1)
    delaunay_locations = np.asarray(delaunay_locations)

    # Find the midpoint on all Delaunay edges
    m = (delaunay_locations[:, 0, :] + delaunay_locations[:, 1, :]) / 2

    # Find the radius sphere between each pair of nodes
    r = np.sqrt(np.sum((delaunay_locations[:, 0, :] - delaunay_locations[:, 1, :]) ** 2, axis=1)) / 2

    # Use the kd-tree function in Scipy's spatial module
    tree = spatial.cKDTree(locations)
    # Find the nearest point for each midpoint
    n = tree.query(x=m, k=1)[0]
    # If nearest point to m is at a distance r, then the edge is a Gabriel edge
    g = n >= r * 0.999  # The factor is to avoid precision errors in the distances

    return delaunay_connections[g]


def n_smallest_distances(a, n, return_idx: bool):
    """ This function finds the n smallest distances in a distance matrix

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=False)
    array([2, 3, 4])

    >>> n_smallest_distances([
    ... [0, 2, 3, 4],
    ... [2, 0, 5, 6],
    ... [3, 5, 0, 7],
    ... [4, 6, 7, 0]], 3, return_idx=True)
    (array([1, 2, 3]), array([0, 0, 0]))

    Args:
        a (np.array): The distane matrix
        n (int): The number of distances to return
        return_idx (bool): return the indices of the points (True) or rather the distances (False)

    Returns:
        (np.array): the n_smallest distances
    or
        (np.array, np.array): the indices between which the distances are smallest
    """
    a_tril = np.tril(a)
    a_nn = a_tril[np.nonzero(a_tril)]
    smallest_n = np.sort(a_nn)[: n]
    a_idx = np.isin(a_tril, smallest_n)

    if return_idx:
        return np.where(a_idx)
    else:
        return smallest_n


def clusters_autosimilarity(cluster, t):
    """
    This function computes the similarity of consecutive cluster in a chain
    Args:
        cluster (list): cluster
        t (integer): lag between consecutive cluster in the chain

    Returns:
        (float) : mean similarity between cluster in the chain with lag t
    """
    z = np.asarray(cluster)
    z = z[:, 0, :]
    unions = np.maximum(z[t:], z[:-t])
    intersections = np.minimum(z[t:], z[:-t])
    sim_norm = np.sum(intersections, axis=1) / np.sum(unions, axis=1)

    return np.mean(sim_norm)


def range_like(a):
    """Return a list of incrementing integers (range) with same length as `a`."""
    return list(range(len(a)))


# Encoding
def encode_states(features_raw, feature_states):
    # Define shapes
    n_states, n_features = feature_states.shape
    features_bin_shape = features_raw.shape + (n_states,)
    n_sites, _ = features_raw.shape
    assert n_features == _

    # Initialize arrays and counts
    features_bin = np.zeros(features_bin_shape, dtype=int)
    applicable_states = np.zeros((n_features, n_states), dtype=bool)
    state_names = []
    na_number = 0

    # Binary vectors used for encoding
    one_hot = np.eye(n_states)

    for f_idx in range(n_features):
        f_name = feature_states.columns[f_idx]
        f_states = feature_states[f_name]

        # Define applicable states for feature f
        applicable_states[f_idx] = ~f_states.isna()

        # Define external and internal state names
        s_ext = f_states.dropna().to_list()
        s_int = range_like(s_ext)
        state_names.append(s_ext)

        # Map external to internal states for feature f
        ext_to_int = dict(zip(s_ext, s_int))
        f_raw = features_raw[f_name]
        f_enc = f_raw.map(ext_to_int)
        if not (set(f_raw.dropna()).issubset(set(s_ext))):
            print(set(f_raw.dropna()) - set(s_ext))
            print(s_ext)
        assert set(f_raw.dropna()).issubset(set(s_ext))  # All states should map to an encoding

        # Binarize features
        f_applicable = ~f_enc.isna().to_numpy()
        f_enc_applicable = f_enc[f_applicable].astype(int)

        features_bin[f_applicable, f_idx] = one_hot[f_enc_applicable]

        # Count NA
        na_number += np.count_nonzero(f_enc.isna())

    features = {
        'values': features_bin.astype(bool),
        'states': applicable_states,
        'state_names': state_names
    }

    return features, na_number


def normalize_str(s: str) -> str:
    if pd.isna(s):
        return s
    return str.strip(unidecode(s))


def read_data_csv(csv_path: PathLike | io.StringIO) -> pd.DataFrame:
    na_values = ["", " ", "\t", "  "]
    data: pd.DataFrame = pd.read_csv(csv_path, na_values=na_values, keep_default_na=False, dtype=str)
    data.columns = [unidecode(c) for c in data.columns]
    return data.applymap(normalize_str)


def read_costs_from_csv(file: str, logger=None):
    """This is a helper function to read the cost matrix from a csv file
        Args:
            file: file location of the csv file
            logger: Logger objects for printing info message.

        Returns:
            pd.DataFrame: cost matrix
        """

    data = pd.read_csv(file, dtype=str, index_col=0)
    if logger:
        logger.info(f"Geographical cost matrix read from {file}.")
    return data


def write_languages_to_csv(features, sites, families, file):
    """This is a helper function to export features as a csv file
    Args:
        features (np.array): features
            shape: (n_sites, n_features, n_categories)
        sites (dict): sites with unique id
        families (np.array): families
            shape: (n_families, n_sites)
        file(str): output csv file
    """
    families = families.transpose(1, 0)

    with open(file, mode='w', encoding='utf-8') as csv_file:
        f_names = list(range(features.shape[1]))
        csv_names = ['f' + str(f) for f in f_names]
        csv_names = ["name", "x", "y", "family"] + csv_names
        writer = csv.writer(csv_file)
        writer.writerow(csv_names)

        for i in sites['id']:
            # name
            name = "site_" + str(i)
            # location
            x, y = sites['locations'][i]
            # features
            f = np.where(features[i] == 1)[1].tolist()
            # family
            fam = np.where(families[i] == 1)[0].tolist()
            if not fam:
                fam = ""
            else:
                fam = "family_" + str(fam[0])
            writer.writerow([name] + [x] + [y] + [fam] + f)


def write_feature_occurrence_to_csv(occurrence, categories, file):
    """This is a helper function to export the occurrence of features in families or globally to a csv
    Args:
        occurrence: the occurrence of each feature, either as a relative frequency or counts
        categories: the possible categories per feature
        file(str): output csv file
    """

    with open(file, mode='w', encoding='utf-8') as csv_file:
        features = list(range(occurrence.shape[0]))
        feature_names = ['f' + str(f) for f in features]
        cats = list(range(occurrence.shape[1]))
        cat_names = ['cat' + str(c) for c in cats]
        csv_names = ["feature"] + cat_names
        writer = csv.writer(csv_file)
        writer.writerow(csv_names)

        for f in range(len(feature_names)):
            # feature name
            f_name = feature_names[f]
            # frequencies
            p = occurrence[f, :].tolist()
            idx = categories[f]
            for i in range(len(p)):
                if i not in idx:
                    p[i] = ""
            writer.writerow([f_name] + p)


def read_feature_occurrence_from_csv(file, feature_states_file):
    """This is a helper function to import the occurrence of features in families (or globally) from a csv
        Args:
            file(str): path to the csv file containing feature-state counts
            feature_states_file (str): path to csv file containing features and states

        Returns:
            np.array :
            The occurrence of each feature, either as relative frequencies or counts, together with feature
            and category names
    """

    # Load data and feature states
    counts_raw = pd.read_csv(file, index_col='feature')
    feature_states = pd.read_csv(feature_states_file, dtype=str)
    n_states, n_features = feature_states.shape

    # Check that features match
    assert set(counts_raw.index) == set(feature_states.columns)

    # Replace NAs by 0.
    counts_raw[counts_raw.isna()] = 0.

    # Collect feature and state names
    feature_names = {'external': feature_states.columns.to_list(),
                     'internal': list(range(n_features))}
    state_names = {'external': [[] for _ in range(n_features)],
                   'internal': [[] for _ in range(n_features)]}

    # Align state columns with feature_states file
    counts = np.zeros((n_features, n_states))
    for f_idx in range(n_features):
        f_name = feature_states.columns[f_idx]         # Feature order is given by ´feature_states_file´
        for s_idx in range(n_states):
            s_name = feature_states[f_name][s_idx]     # States order is given by ´feature_states_file´
            if pd.isnull(s_name):                      # ...same for applicable states per feature
                continue

            counts[f_idx, s_idx] = counts_raw.loc[f_name, s_name]

            state_names['external'][f_idx].append(s_name)
            state_names['internal'][f_idx].append(s_idx)

    # # Sanity check
    # Are the data count data?
    if not all(float(y).is_integer() for y in np.nditer(counts)):
        out = f"The data in {file} must be count data."
        raise ValueError(out)

    return counts.astype(int), feature_names, state_names


def touch(fname):
    """Create an empty file at path `fname`."""
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()


def mkpath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isdir(path):
        touch(path)


def linear_rescale(value, old_min, old_max, new_min, new_max):
    """
    Function to linear rescale a number to a new range

    Args:
         value (float): number to rescale
         old_min (float): old minimum of value range
         old_max (float): old maximum of value range
         new_min (float): new minimum of value range
         new_max (float): new maximum of vlaue range
    """

    return (new_max - new_min) / (old_max - old_min) * (value - old_max) + old_max


def normalize(x, axis=-1):
    """Normalize ´x´ s.t. the last axis sums up to 1.

    Args:
        x (np.array): Array to be normalized.
        axis (int): The axis to be normalized (will sum up to 1).

    Returns:
         np.array: x with normalized s.t. the last axis sums to 1.

    == Usage ===
    >>> normalize(np.ones((2, 4)))
    array([[0.25, 0.25, 0.25, 0.25],
           [0.25, 0.25, 0.25, 0.25]])
    >>> normalize(np.ones((2, 4)), axis=0)
    array([[0.5, 0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5, 0.5]])
    """
    assert np.all(np.sum(x, axis=axis) > 0)
    return x / np.sum(x, axis=axis, keepdims=True)


def mle_weights(samples):
    """Compute the maximum likelihood estimate for categorical samples.

    Args:
        samples (np.array):
    Returns:
        np.array: the MLE for the probability vector in the categorical distribution.
    """
    counts = np.sum(samples, axis=0)
    return normalize(counts)


def log_binom(
    n: int | NDArray[int],
    k: int | NDArray[int]
) -> float | NDArray[float]:
    """Compute the logarithm of (n choose k), i.e. the binomial coefficient of `n` and `k`.

    Args:
        n: Population size.
        k: Sample size.
    Returns:
        double: log(n choose k)

    == Usage ===
    >>> log_binom(10, np.arange(3))
    array([0.        , 2.30258509, 3.80666249])
    >>> log_binom(np.arange(1, 4), 1)
    array([0.        , 0.69314718, 1.09861229])
    """
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def log_multinom(n: int, ks: Sequence[int]) -> float:
    """Compute the logarithm of (n choose k1,k2,...), i.e. the multinomial coefficient of
    `n` and the integers in the list `ks`. The sum of the sample sizes (the numbers in
     `ks`) may not exceed the population size (`n`).

    Args:
        n: Population size.
        ks: Sample sizes

    Returns:
        The log multinomial coefficient: log(n choose k1,k2,...)

    == Usage ===
    >>> log_multinom(5, [1,1,1,1])  # == log(5!)
    4.787491742782046
    >>> log_multinom(13, [4])  # == log_binom(13, 4)
    6.572282542694008
    >>> log_multinom(13, [3, 2])  # == log_binom(13, 3) + log_binom(10, 2)
    9.462654300590172
    """
    ks = np.asarray(ks)
    # assert np.all(ks >= 0)
    # assert np.sum(ks) <= n

    # Simple special case
    if np.sum(ks) == 0:
        return 0.

    # Filter out 0-samples
    ks = ks[ks > 0]

    log_i = np.log(1 + np.arange(n))
    log_i_cumsum = np.cumsum(log_i)

    # Count all permutations of the total population
    m = np.sum(log_i)

    # Subtract all permutation within the samples (with sample sizes specified in `ks`).
    m -= np.sum(log_i_cumsum[ks-1])

    # If there are is a remainder in the population, that was not assigned to any of the
    # samples, subtract all permutations of the remainder population.
    rest = n - np.sum(ks)
    if rest > 0:
        m -= log_i_cumsum[rest-1]

    return m


def decompose_config_path(config_path: PathLike) -> (Path, Path):
    """Extract the base directory of `config_path` and return the path itself as an absolute path."""
    abs_config_path = Path(config_path).absolute()
    base_directory = abs_config_path.parent
    return base_directory, abs_config_path


def fix_relative_path(path: PathLike, base_directory: PathLike) -> Path:
    """Make sure that the provided path is either absolute or relative to the config file directory.

    Args:
        path: The original path (absolute or relative).
        base_directory: The base directory

    Returns:
        The fixed path.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return base_directory / path


def timeit(units='s'):
    SECONDS_PER_UNIT = {
        'h': 3600.,
        'm': 60.,
        's': 1.,
        'ms': 1E-3,
        'µs': 1E-6,
        'ns': 1E-9
    }
    unit_scaler = SECONDS_PER_UNIT[units]

    def timeit_decorator(func):

        def timed_func(*args, **kwargs):


            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            passed = (end - start) / unit_scaler

            print(f'Runtime {func.__name__}: {passed:.2f}{units}')

            return result

        return timed_func

    return timeit_decorator


def get_permutations(n: int) -> Iterator[tuple[int]]:
    return permutations(range(n))


def get_best_permutation(
        areas: NDArray[bool],  # shape = (n_areas, n_sites)
        prev_area_sum: NDArray[int],  # shape = (n_areas, n_sites)
) -> NDArray[int]:
    """Return a permutation of areas that would align the areas in the new sample with previous ones."""
    cluster_agreement_matrix = np.matmul(prev_area_sum, areas.T)
    return linear_sum_assignment(cluster_agreement_matrix, maximize=True)[1]


if scipy.__version__ >= '1.8.0':
    log_expit = scipy.special.log_expit
else:
    def log_expit(*args, **kwargs):
        return np.log(expit(*args, **kwargs))


def set_defaults(cfg: dict, default_cfg: dict):
    """Iterate through a recursive config dictionary and set all fields that are not
    present in cfg to the default values from default_cfg.

    == Usage ===
    >>> set_defaults(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}
    >>> set_defaults(cfg={0:0, 1:1, 2:2},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: 1, 2: 2}
    """
    for key in default_cfg:
        if key not in cfg:
            # Field ´key´ is not defined in cfg -> use default
            cfg[key] = default_cfg[key]

        else:
            # Field ´key´ is defined in cfg
            # -> update recursively if the field is a dictionary
            if isinstance(default_cfg[key], dict) and isinstance(cfg[key], dict):
                set_defaults(cfg[key], default_cfg[key])

    return cfg


def update_recursive(cfg: dict, new_cfg: dict):
    """Iterate through a recursive config dictionary and update cfg in all fields that are specified in new_cfg.

    == Usage ===
    >>> update_recursive(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {2: 2, 1: 1}}
    >>> update_recursive(cfg={0:0, 1:1, 2:2},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {1: 1, 2: 2}}
    """
    for key in new_cfg:
        if (key in cfg) and isinstance(new_cfg[key], dict) and isinstance(cfg[key], dict):
            # Both dictionaries have another layer -> update recursively
            update_recursive(cfg[key], new_cfg[key])
        else:
            cfg[key] = new_cfg[key]

    return cfg


def iter_items_recursive(cfg: dict, loc=tuple()):
    """Recursively iterate through all key-value pairs in ´cfg´ and sub-dictionaries.

    Args:
        cfg (dict): Config dictionary, potentially containing sub-dictionaries.
        loc (tuple): Specifies the sequene of keys that lead to the current sub-dictionary.
    Yields:
        tuple: key-value pairs of the bottom level dictionaries

    == Usage ===
    >>> list(iter_items_recursive({0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}))
    [(0, 0, ()), (1, 0, (1,)), (2, 1, (2,)), (1, 1, (2,))]
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from iter_items_recursive(value, loc + (key, ))
        else:
            yield key, value, loc


def categorical_log_probability(x: NDArray[bool], p: NDArray[float]) -> NDArray[float]:
    """Compute the log-probability of observations `x` under categorical distribution `p`.

    Args:
        x: observations in one-hot encoding. Each column (on last axis) contains exactly one 1.
            shape: (*distr_shape, n_categories)
        p: probability of each state in each dimension of the distribution. Last axis is
                normalised to one.
            shape: (*distr_shape, n_categories)

    Returns:
        The log-probability for each observation elementwise.
            shape: distr_shape

    """
    return np.log(np.sum(x*p, axis=-1))


def dirichlet_multinomial_logpdf(
    counts: NDArray[int],        # shape: (n_features, n_states)
    a: NDArray[float],      # shape: (n_features, n_states)
) -> NDArray[float]:        # shape: (n_features)
    """Calculate log-probability of DirichletMultinomial distribution for given Dirichlet
    concentration parameter `a` and multinomial observations ´counts´.

    Dirichlet-multinomial distribution:
        https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    Reference implementation (pymc3):
        https://github.com/pymc-devs/pymc/blob/main/pymc/distributions/multivariate.py

    == Usage ===
    >>> dirichlet_multinomial_logpdf(counts=np.array([2, 1, 0, 0]), a=np.array([1., 1., 0., 0.]))
    -1.386294361303224
    """
    # Only apply to
    # valid = a > 0
    # counts = counts[valid]
    # a = a[valid]
    a = a + 1e-12      # TODO Find a better way to fix 0s in a (and still use broadcasting)

    n = counts.sum(axis=-1)
    sum_a = a.sum(axis=-1)
    const = (gammaln(n + 1) + gammaln(sum_a)) - gammaln(n + sum_a)
    series = gammaln(counts + a) - (gammaln(counts + 1) + gammaln(a))
    return const + series.sum(axis=-1)


def dirichlet_categorical_logpdf(
    counts: NDArray[int],   # shape: (n_features, n_states)
    a: NDArray[float],      # shape: (n_features, n_states)
) -> NDArray[float]:        # shape: (n_features)
    """Calculate log-probability of DirichletMultinomial distribution for given Dirichlet
    concentration parameter `a` and multinomial observations ´counts´.

    Dirichlet-multinomial distribution:
        https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    Reference implementation (pymc3):
        https://github.com/pymc-devs/pymc/blob/main/pymc/distributions/multivariate.py

    == Usage ===
    >>> dirichlet_multinomial_logpdf(counts=np.array([2, 1, 0, 0]), a=np.array([1., 1., 0., 0.]))
    -1.386294361303224
    """
    a = a + 1e-12  # TODO Find a better way to fix 0s in a (and still use broadcasting)

    n = counts.sum(axis=-1)
    sum_a = a.sum(axis=-1)
    const = gammaln(sum_a) - gammaln(n + sum_a)
    series = gammaln(counts + a) - gammaln(a)
    return const + series.sum(axis=-1)


def get_along_axis(a: NDArray, index: int, axis: int):
    """Get the index-th entry in the axis-th dimension of array a.
    Examples:
        >>> get_along_axis(a=np.arange(6).reshape((2,3)), index=2, axis=1)
        array([2, 5])
    """
    I = [slice(None)] * a.ndim
    I[axis] = index
    return a[tuple(I)]


def pmf_categorical_with_replacement(idxs: list[int], p: NDArray[float]):
    prob = 0
    for idxs_perm in map(list, permutations(idxs)):
        prob += np.prod(p[idxs_perm]) / np.prod(1-np.cumsum(p[idxs_perm][:-1]))
    return prob


def read_geo_cost_matrix(object_names: Sequence[str], file: PathLike, logger=None) -> NDArray[float]:
    """ This is a helper function to import the geographical cost matrix.

    Args:
        object_names: the names of the objects or languages (external and internal)
        file: path to the file location

    Returns:
        The symmetric cost matrix between objects.
    """
    costs = read_costs_from_csv(file, logger=logger)
    assert set(costs.columns) == set(object_names)

    # Sort the data by object names
    sorted_costs = costs.loc[object_names, object_names]

    cost_matrix = np.asarray(sorted_costs).astype(float)

    # Check if matrix is symmetric, if not make symmetric
    if not np.allclose(cost_matrix, cost_matrix.T):
        cost_matrix = (cost_matrix + cost_matrix.T)/2
        if logger:
            logger.info("The cost matrix is not symmetric. It was made symmetric by "
                        "averaging the original costs in the upper and lower triangle.")
    return cost_matrix


if __name__ == "__main__":
    import doctest
    doctest.testmod()


def min_and_max_with_padding(x, pad=0.05):
    lower = np.min(x)
    upper = np.max(x)
    diff = upper - lower
    return lower - pad * diff, upper + pad * diff


def reproject_locations(locations, data_proj, map_proj):
    if data_proj == map_proj:
        return locations
    loc = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*locations.T), crs=data_proj)
    loc_re = loc.to_crs(map_proj).geometry
    return np.array([loc_re.x, loc_re.y]).T
