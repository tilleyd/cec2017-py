# cec2017.hybrid
# Author: Duncan Tilley
# Hybrid function definitions, f11 to f20

from . import basic
from . import transforms

import numpy as np

def _shuffle_and_partition(x, shuffle, partitions):
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
    nx = len(x)
    # shuffle
    xs = np.zeros(x.shape)
    for i in range(0, nx):
        xs[i] = x[shuffle[i]]
    # and partition
    parts = []
    start, end = 0, 0
    for p in partitions[:-1]:
        end = start + int(np.ceil(p * nx))
        parts.append(xs[start:end])
        start = end
    parts.append(xs[end:])
    return parts

def f11(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 1 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][10]
    if shift is None:
        shift = transforms.shifts[10][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][0]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.4, 0.4])

    y = basic.zakharov(x_parts[0])
    y += basic.rosenbrock(x_parts[1])
    y += basic.rastrigin(x_parts[2])
    return y + 1100.0

def f12(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 2 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][11]
    if shift is None:
        shift = transforms.shifts[11][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][1]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.3, 0.3, 0.4])

    y = basic.high_conditioned_elliptic(x_parts[0])
    y += basic.modified_schwefel(x_parts[1])
    y += basic.bent_cigar(x_parts[2])
    return y + 1200.0

def f13(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 3 (N=3)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][12]
    if shift is None:
        shift = transforms.shifts[12][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][2]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.3, 0.3, 0.4])

    y = basic.bent_cigar(x_parts[0])
    y += basic.rosenbrock(x_parts[1])
    y += basic.lunacek_bi_rastrigin(x_parts[2])
    return y + 1300.0

def f14(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 4 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][13]
    if shift is None:
        shift = transforms.shifts[13][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][3]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.4])

    y = basic.high_conditioned_elliptic(x_parts[0])
    y += basic.ackley(x_parts[1])
    y += basic.schaffers_f7(x_parts[2])
    y += basic.rastrigin(x_parts[3])
    return y + 1400.0

def f15(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 5 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][14]
    if shift is None:
        shift = transforms.shifts[14][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][4]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.3, 0.3])

    y = basic.bent_cigar(x_parts[0])
    y += basic.h_g_bat(x_parts[1])
    y += basic.rastrigin(x_parts[2])
    y += basic.rosenbrock(x_parts[3])
    return y + 1500.0

def f16(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 6 (N=4)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][15]
    if shift is None:
        shift = transforms.shifts[15][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][5]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.3, 0.3])

    y = basic.expanded_schaffers_f6(x_parts[0])
    y += basic.h_g_bat(x_parts[1])
    y += basic.rosenbrock(x_parts[2])
    y += basic.modified_schwefel(x_parts[3])
    return y + 1600.0

def f17(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 7 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][16]
    if shift is None:
        shift = transforms.shifts[16][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][6]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.1, 0.2, 0.2, 0.2, 0.3])

    y = basic.katsuura(x_parts[0])
    y += basic.ackley(x_parts[1])
    y += basic.expanded_griewanks_plus_rosenbrock(x_parts[2])
    y += basic.modified_schwefel(x_parts[3])
    y += basic.rastrigin(x_parts[4])
    return y + 1700.0

def f18(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 8 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][17]
    if shift is None:
        shift = transforms.shifts[17][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][7]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])

    y = basic.high_conditioned_elliptic(x_parts[0])
    y += basic.ackley(x_parts[1])
    y += basic.rastrigin(x_parts[2])
    y += basic.h_g_bat(x_parts[3])
    y += basic.discus(x_parts[4])
    return y + 1800.0

def f19(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 9 (N=5)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][18]
    if shift is None:
        shift = transforms.shifts[18][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][8]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.2, 0.2, 0.2, 0.2, 0.2])

    y = basic.bent_cigar(x_parts[0])
    y += basic.rastrigin(x_parts[1])
    y += basic.expanded_griewanks_plus_rosenbrock(x_parts[2])
    y += basic.weierstrass(x_parts[3])
    y += basic.expanded_schaffers_f6(x_parts[4])
    return y + 1900.0

def f20(x, rotation=None, shift=None, shuffle=None):
    """
    Hybrid Function 10 (N=6)

    Args:
        x (array): Input vector of dimension 2, 10, 20, 30, 50 or 100.
        rotation (matrix): Optional rotation matrix. If None (default), the
            official matrix from the benchmark suite will be used.
        shift (array): Optional shift vector. If None (default), the official
            vector from the benchmark suite will be used.
        shuffle (array): Optionbal shuffle vector. If None (default), the
            official permutation vector from the benchmark suite will be used.
    """
    nx = len(x)
    if rotation is None:
        rotation = transforms.rotations[nx][19]
    if shift is None:
        shift = transforms.shifts[19][:nx]
    if shuffle is None:
        shuffle = transforms.shuffles[nx][9]

    x_transformed = np.matmul(rotation, x - shift)
    x_parts = _shuffle_and_partition(x_transformed, shuffle, [0.1, 0.1, 0.2, 0.2, 0.2, 0.2])

    y = basic.happy_cat(x_parts[0])
    y += basic.katsuura(x_parts[1])
    y += basic.ackley(x_parts[2])
    y += basic.rastrigin(x_parts[3])
    y += basic.modified_schwefel(x_parts[4])
    y += basic.schaffers_f7(x_parts[5])
    return y + 2000.0

all_functions = [
    f11,
    f12,
    f13,
    f14,
    f15,
    f16,
    f17,
    f18,
    f19,
    f20
]
