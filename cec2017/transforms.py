# cec2017.transforms
# Author: Duncan Tilley
# Contains rotation, shift and shuffle data loaded from data.pkl.
# Note that these correspond to the many .txt files provided along with the
# original implementation and should be used for final benchmark results.

import numpy as np
import pickle
import os

with open(os.path.join(os.path.dirname(__file__), 'data.pkl'), 'rb') as _pkl_file:
    _pkl = pickle.load(_pkl_file)

# Each has shape (20, N, N) containing an N-dimensional rotation matrix
# for functions f1 to f20
rotations = {
    2: _pkl['M_D2'],
    10: _pkl['M_D10'],
    20: _pkl['M_D20'],
    30: _pkl['M_D30'],
    50: _pkl['M_D50'],
    100: _pkl['M_D100']
}

# Each has shape (10, 10, N, N) containing 10 N-dimensional rotation matrices
# for functions f21 to f30
rotations_cf = {
    2: _pkl['M_cf_d2'],
    10: _pkl['M_cf_D10'],
    20: _pkl['M_cf_D20'],
    30: _pkl['M_cf_D30'],
    50: _pkl['M_cf_D50'],
    100: _pkl['M_cf_D100']
}

# Shape (20, 100)
# Contains 100-dimension shift vectors for functions f1 to f20
shifts = _pkl['shift']

# Shape (10, 10, 100)
# Contains 10 100-dimension shift vectors for functions f21 to f30
shifts_cf = _pkl['shift_cf']

# Each has shape (10, N) containing N-dimensional permutations for functions f11
# to f20 (note: the original were 1-indexed, these are 0-indexed)
shuffles = {
    10: _pkl['shuffle_D10'],
    30: _pkl['shuffle_D30'],
    50: _pkl['shuffle_D50'],
    100: _pkl['shuffle_D100']
}

# Each has shape (2, 10, N) containing 10 N-dimensional permutations for
# functions f29 and f30 (note: the original were 1-indexed, these are 0-indexed)
shuffles_cf = {
    10: _pkl['shuffle_cf_D10'],
    30: _pkl['shuffle_cf_D30'],
    50: _pkl['shuffle_cf_D50'],
    100: _pkl['shuffle_cf_D100']
}


def shift_rotate(x: np.ndarray, shift: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Apply the shift and rotation to vector x along its second axis.

    Args:
        x (np.ndarray):
            (M, N) array of M N-dimensional vectors.
        shift (np.ndarray):
            Array of size N providing the shift.
        rotation (np.ndarray):
            (N, N) array providing the rotation matrix.

    Returns:
        (M, N) array of M shifted and rotated N-dimensional vectors.
    """
    shifted = np.expand_dims(x - np.expand_dims(shift, 0), -1)
    x_transformed = np.matmul(np.expand_dims(rotation, 0), shifted)
    return x_transformed[:, :, 0]


def shuffle_and_partition(x, shuffle, partitions):
    """
    First applies the given permutation, then splits x into partitions given
    the percentages.

    Args:
        x (array): Input vector.
        shuffle (array): Shuffle vector.
        partitions (list): List of percentages. Assumed to add up to 1.0.

    Returns:
        (list of arrays): The partitions of x after shuffling.
    """
    nx = x.shape[1]

    # shuffle
    xs = np.zeros_like(x)
    for i in range(0, nx):
        xs[:, i] = x[:, shuffle[i]]
    # and partition
    parts = []
    start, end = 0, 0
    for p in partitions[:-1]:
        end = start + int(np.ceil(p * nx))
        parts.append(xs[:, start:end])
        start = end
    parts.append(xs[:, end:])
    return parts
