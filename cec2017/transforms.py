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
