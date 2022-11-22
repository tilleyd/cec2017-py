# cec2017.simple
# Author: Duncan Tilley
# Simple function definitions, f1 to f10

from . import basic
from . import transforms

import numpy as np


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
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][0]
    if shift is None:
        shift = transforms.shifts[0][:nx]

    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.bent_cigar(x_transformed) + 100.0


def f2(x, rotation=None, shift=None):
    """
    (Deprecated) Shifted and Rotated Sum of Different Power Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    if 'warned' not in f2.__dict__:
        f2.warned = True
        print('WARNING: f2 has been deprecated from the CEC 2017 benchmark suite')

    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][1]
    if shift is None:
        shift = transforms.shifts[1][:nx]
    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.sum_diff_pow(x_transformed) + 200.0


def f3(x, rotation=None, shift=None):
    """
    Shifted and Rotated Zakharov Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][2]
    if shift is None:
        shift = transforms.shifts[2][:nx]
    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.zakharov(x_transformed) + 300.0


def f4(x, rotation=None, shift=None):
    """
    Shifted and Rotated Rosenbrock's Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][3]
    if shift is None:
        shift = transforms.shifts[3][:nx]
    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.rosenbrock(x_transformed) + 400.0


def f5(x, rotation=None, shift=None):
    """
    Shifted and Rotated Rastrigin's Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][4]
    if shift is None:
        shift = transforms.shifts[4][:nx]
    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.rastrigin(x_transformed) + 500.0


def f6(x, rotation=None, shift=None):
    """
    Shifted and Rotated Schaffer's F7 Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][5]
    if shift is None:
        shift = transforms.shifts[5][:nx]
    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.schaffers_f7(x_transformed) + 600.0


def f7(x, rotation=None, shift=None):
    """
    Shifted and Rotated Lunacek Bi-Rastrigin's Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][6]
    if shift is None:
        shift = transforms.shifts[6][:nx]
    # pass the shift and rotation directly to the function
    return basic.lunacek_bi_rastrigin(x, shift, rotation) + 700.0


def f8(x, rotation=None, shift=None):
    """
    Shifted and Rotated Non-Continuous Rastrigin’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][7]
    if shift is None:
        shift = transforms.shifts[7][:nx]
    # pass the shift and rotation directly to the function
    return basic.non_cont_rastrigin(x, shift, rotation) + 800.0


def f9(x, rotation=None, shift=None):
    """
    Shifted and Rotated Levy Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][8]
    if shift is None:
        shift = transforms.shifts[8][:nx]
    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.levy(x_transformed) + 900.0


def f10(x, rotation=None, shift=None):
    """
    Shifted and Rotated Schwefel’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    x = np.array(x)
    nx = x.shape[1]

    if rotation is None:
        rotation = transforms.rotations[nx][9]
    if shift is None:
        shift = transforms.shifts[9][:nx]
    x_transformed = transforms.shift_rotate(x, shift, rotation)
    return basic.modified_schwefel(x_transformed) + 1000.0


all_functions = [
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10
]
