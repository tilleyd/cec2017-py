# cec2017.basic
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
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][0]
    if shift is None:
        shift = transforms.shifts[0][:nx]
    x_transformed = np.matmul(rotation, x - shift)
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

    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][1]
    if shift is None:
        shift = transforms.shifts[1][:nx]
    x_transformed = np.matmul(rotation, x - shift)
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
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][2]
    if shift is None:
        shift = transforms.shifts[2][:nx]
    x_transformed = np.matmul(rotation, x - shift)
    return basic.zakharov(x_transformed) + 300.0

def f4(x, rotation=None, shift=None):
    """
    Shifted and Rotated Rosenbrock’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][3]
    if shift is None:
        shift = transforms.shifts[3][:nx]
    x_transformed = np.matmul(rotation, 0.02048 * (x - shift)) + 1.0
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
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][4]
    if shift is None:
        shift = transforms.shifts[4][:nx]
    # Note: the 0.0512 shrinking is omitted in the problem definitions but is present in the provided code
    x_transformed = np.matmul(rotation, 0.0512 * (x - shift))
    return basic.rastrigin(x_transformed) + 500.0

def f6(x, rotation=None, shift=None):
    """
    Shifted and Rotated Schaffer’s F7 Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][5]
    if shift is None:
        shift = transforms.shifts[5][:nx]
    x_transformed = np.matmul(rotation, 0.005 * (x - shift))
    return basic.schaffers_f7(x_transformed) + 600.0

def f7(x, rotation=None, shift=None):
    """
    Shifted and Rotated Lunacek Bi-Rastrigin’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][6]
    if shift is None:
        shift = transforms.shifts[6][:nx]
    x_transformed = np.matmul(rotation, 6.0 * (x - shift))
    return basic.lunacek_bi_rastrigin(x_transformed) + 700.0

def f8(x, rotation=None, shift=None):
    """
    Shifted and Rotated Lunacek Bi-Rastrigin’s Function

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][7]
    if shift is None:
        shift = transforms.shifts[7][:nx]
    x_transformed = np.matmul(rotation, 0.0512 * (x - shift))
    return basic.non_cont_rotated_rastrigin(x_transformed) + 800.0

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
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][8]
    if shift is None:
        shift = transforms.shifts[8][:nx]
    x_transformed = np.matmul(rotation, 0.0512 * (x - shift))
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
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][9]
    if shift is None:
        shift = transforms.shifts[9][:nx]
    x_transformed = np.matmul(rotation, 10.0 * (x - shift))
    return basic.modified_schwefel(x_transformed) + 1000.0
