# cec2017.basic
# Author: Duncan Tilley
# Simple function definitions, f1 to f10

from . import basic
from . import transforms

def f1(x, rotation=None, shift=None):
    """
    Shifted and Rotated Bent Cigar Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][0]
    if shift is None:
        shift = transforms.shifts[0][:nx]
    x_transformed = x # TODO still have to use the transformations
    return basic.bent_cigar(x_transformed)
