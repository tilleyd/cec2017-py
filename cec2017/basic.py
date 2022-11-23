# cec2017.basic
# Author: Duncan Tilley
# Basic function definitions

from typing import Optional
import numpy as np


def bent_cigar(x: np.ndarray) -> np.ndarray:
    sm = np.sum(x[:, 1:] * x[:, 1:], axis=1)
    sm = sm * 10e6
    return x[:, 0]*x[:, 0] + sm


def sum_diff_pow(x: np.ndarray) -> np.ndarray:
    i = np.expand_dims(np.arange(x.shape[1]) + 1, 0)
    x_pow = np.power(np.abs(x), i)
    return np.sum(x_pow, axis=1)


def zakharov(x: np.ndarray) -> np.ndarray:
    # NOTE: the i+1 term is not in the CEC function definitions, but is in the
    # code and in any definition you find online
    i = np.expand_dims(np.arange(x.shape[1]) + 1, 0)
    sm = np.sum(i * x, axis=1)
    sms = np.sum(x * x, axis=1)
    sm = 0.5 * sm
    sm = sm * sm
    return sms + sm + (sm * sm)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    x = 0.02048 * x + 1.0
    t1 = x[:, :-1] * x[:, :-1] - x[:, 1:]
    t1 = 100 * t1 * t1
    t2 = x[:, :-1] - 1
    t2 = t2 * t2
    return np.sum(t1 + t2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    # NOTE: the 0.0512 shrinking is omitted in the problem definitions but is
    # present in the provided code
    x = 0.0512 * x
    cs = np.cos(2 * np.pi * x)
    xs = x*x - 10*cs + 10
    return np.sum(xs, axis=1)


def expanded_schaffers_f6(x: np.ndarray) -> np.ndarray:
    t = x[:, :-1]*x[:, :-1] + x[:, 1:]*x[:, 1:]
    t1 = np.sin(np.sqrt(t))
    t1 = t1*t1 - 0.5
    t2 = 1 + 0.001*t
    t2 = t2*t2
    return np.sum(0.5 + t1/t2, axis=1)


def lunacek_bi_rastrigin(
    x: np.ndarray,
    shift: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    # a special case; we need the shift vector and rotation matrix
    nx = x.shape[1]
    if shift is None:
        shift = np.zeros((1, nx))
    else:
        shift = np.expand_dims(shift, 0)

    # calculate the coefficients
    mu0 = 2.5
    s = 1 - 1 / (2 * ((nx+20)**0.5) - 8.2)
    mu1 = -((mu0*mu0-1)/s)**0.5

    # shift and scale
    y = 0.1 * (x - shift)

    tmpx = 2 * y
    tmpx[:, shift[0] < 0] *= -1

    z = tmpx.copy()
    tmpx = tmpx + mu0

    t1 = tmpx - mu0
    t1 = t1 * t1
    t1 = np.sum(t1, axis=1)
    t2 = tmpx - mu1
    t2 = s * t2 * t2
    t2 = np.sum(t2, axis=1) + nx

    if rotation is None:
        y = z
    else:
        y = np.matmul(
            np.expand_dims(rotation, 0),
            np.expand_dims(z, -1),
        )[:, :, 0]

    y = np.cos(2.0*np.pi*y)
    t = np.sum(y, axis=1)

    r = t1
    r[t1 >= t2] = t2[t1 >= t2]
    return r + 10.0*(nx-t)


def non_cont_rastrigin(
    x: np.ndarray,
    shift: Optional[np.ndarray] = None,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    # a special case; we need the shift vector and rotation matrix
    nx = x.shape[1]
    if shift is None:
        shift = np.zeros((1, nx))
    else:
        shift = np.expand_dims(shift, 0)
    shifted = x - shift

    sm = 0.0
    x = x.copy()
    mask = np.abs(shifted) > 0.5
    x[mask] = (shift + np.floor(2*shifted+0.5) * 0.5)[mask]

    # for i in range(0, nx):
    #     if abs(x[i]-shift[i]) > 0.5:
    #         x[i] = shift[i] + np.floor(2*(x[i]-shift[i])+0.5)/2

    z = 0.0512 * shifted
    if rotation is not None:
        z = np.matmul(
            np.expand_dims(rotation, 0),
            np.expand_dims(z, -1),
        )[:, :, 0]

    sm = z*z - 10*np.cos(2*np.pi*z) + 10
    sm = np.sum(sm, axis=1)
    # for i in range(0, nx):
    #     sm += (z[i]*z[i] - 10.0*np.cos(2.0*np.pi*z[i]) + 10.0)
    return sm


def levy(x: np.ndarray) -> np.ndarray:
    # NOTE: the function definitions state to scale by 5.12/100, but the code
    # doesn't do this, and the example graph in the definitions correspond to
    # the version without scaling
    # x = 0.0512 * x
    w = 1.0 + 0.25*(x - 1.0)

    term1 = (np.sin(np.pi*w[:, 0]))**2
    term3 = ((w[:, -1] - 1)**2) * (1 + ((np.sin(2*np.pi*w[:, -1]))**2))

    sm = 0.0

    wi = w[:, :-1]
    newv = ((wi - 1)**2) * (1 + 10*((np.sin(np.pi*wi+1))**2))
    sm = np.sum(newv, axis=1)

    return term1 + sm + term3


def modified_schwefel(x: np.ndarray) -> np.ndarray:
    nx = x.shape[1]
    x = 10.0 * x # scale to search range

    z = x + 420.9687462275036
    mask1 = z < -500
    mask2 = z > 500
    sm = z * np.sin(np.sqrt(np.abs(z)))

    zm = np.mod(np.abs(z), 500)
    zm[mask1] = (zm[mask1] - 500)
    zm[mask2] = (500 - zm[mask2])
    t = z + 500
    t[mask2] = z[mask2] - 500
    t = t*t

    mask1_or_2 = np.logical_or(mask1, mask2)
    sm[mask1_or_2] = (zm * np.sin(np.sqrt(np.abs(zm))) - t / (10_000*nx))[mask1_or_2]
    return 418.9829*nx - np.sum(sm, axis=1)


def high_conditioned_elliptic(x: np.ndarray) -> np.ndarray:
    factor = 6 / (x.shape[1] - 1)
    i = np.expand_dims(np.arange(x.shape[1]), 0)
    sm = x*x * 10**(i * factor)
    return np.sum(sm, axis=1)


def discus(x: np.ndarray) -> np.ndarray:
    sm0 = 1e+6*x[:, 0]*x[:, 0]
    sm = np.sum(x[:, 1:]*x[:, 1:], axis=1)
    return sm0 + sm


def ackley(x: np.ndarray) -> np.ndarray:
    smsq = np.sum(x*x, axis=1)
    smcs = np.sum(np.cos((2*np.pi)*x), axis=1)
    inx = 1/x.shape[1]
    return -20*np.exp(-0.2*np.sqrt(inx*smsq)) - np.exp(inx*smcs) + 20 + np.e


def weierstrass(x: np.ndarray) -> np.ndarray:
    x = 0.005 * x
    k = np.arange(start=0, stop=21, step=1)
    k = np.expand_dims(np.expand_dims(k, 0), 0)
    ak = 0.5**k
    bk = np.pi * (3**k)

    kcs = ak * np.cos(2*(np.expand_dims(x, -1) + 0.5)*bk)  # shape (M, nx, 21)
    ksm = np.sum(kcs, axis=2)
    sm = np.sum(ksm, axis=1)

    kcs = ak * np.cos(bk)
    ksm = np.sum(kcs)
    return sm - x.shape[1]*ksm


def griewank(x: np.ndarray) -> np.ndarray:
    nx = x.shape[1]
    x = 6.0 * x
    factor = 1/4000
    d = np.expand_dims(np.arange(start=1, stop=nx + 1), 0)
    cs = np.cos(x / d)
    sm = np.sum(factor*x*x, axis=1)
    pd = np.prod(np.cos(x / d), axis=1)
    return sm - pd + 1


def katsuura(x: np.ndarray) -> np.ndarray:
    x = 0.05 * x
    nx = x.shape[1]
    pw = 10/(nx**1.2)
    prd = 1.0
    tj = 2**np.arange(start=1, stop=33, step=1)
    tj = np.expand_dims(np.expand_dims(tj, 0), 0)
    tjx = tj*np.expand_dims(x, -1)  # shape (M, nx, 32)
    t = np.abs(tjx - np.round(tjx)) / tj
    tsm = np.sum(t, axis=2)

    i = np.arange(nx) + 1
    prd = np.prod((1 + i*tsm)**pw, axis=1)
    df = 10/(nx*nx)
    return df*prd - df


def happy_cat(x: np.ndarray) -> np.ndarray:
    x = (0.05 * x) - 1
    nx = x.shape[1]
    sm = np.sum(x, axis=1)
    smsq = np.sum(x*x, axis=1)
    return (np.abs(smsq - nx))**0.25 + (0.5*smsq + sm)/nx + 0.5


def h_g_bat(x: np.ndarray) -> np.ndarray:
    x = (0.05 * x) - 1
    nx = x.shape[1]
    sm = np.sum(x, axis=1)
    smsq = np.sum(x*x, axis=1)
    return (np.abs(smsq*smsq - sm*sm))**0.5 + (0.5*smsq + sm)/nx + 0.5


def expanded_griewanks_plus_rosenbrock(x: np.ndarray) -> np.ndarray:
    x = (0.05 * x) + 1

    tmp1 = x[:, :-1]*x[:, :-1] - x[:, 1:]
    tmp2 = x[:, :-1] - 1.0
    temp = 100*tmp1*tmp1 + tmp2*tmp2
    sm = (temp*temp)/4000 - np.cos(temp) + 1

    tmp1 = x[:, -1:]*x[:, -1:] - x[:, 0:1]
    tmp2 = x[:, -1:] - 1
    temp = 100*tmp1*tmp1 + tmp2*tmp2
    sm = sm + (temp*temp)/4000 - np.cos(temp) + 1

    return np.sum(sm, axis=1)


def schaffers_f7(x: np.ndarray) -> np.ndarray:
    nx = x.shape[1]
    # NOTE: the function definitions state to scale by 0.5/100, but the code
    # doesn't do this, and the example graph in the definitions correspond to
    # the version without scaling
    # x = 0.005 * x
    sm = 0.0
    si = np.sqrt(x[:, :-1]*x[:, :-1] + x[:, 1:]*x[:, 1:])
    tmp = np.sin(50*(np.power(si, 0.2)))
    # NOTE: the original code has this error here (tmp shouldn't be squared)
    # that I'm keeping for consistency.
    sm = np.sqrt(si) * (tmp*tmp + 1)
    sm = np.sum(sm, axis=1)
    sm = (sm*sm) / (nx*nx - 2*nx + 1)
    return sm


all_functions = [
    bent_cigar,
    sum_diff_pow,
    zakharov,
    rosenbrock,
    rastrigin,
    expanded_schaffers_f6,
    lunacek_bi_rastrigin,
    non_cont_rastrigin,
    levy,
    modified_schwefel,
    high_conditioned_elliptic,
    discus,
    ackley,
    weierstrass,
    griewank,
    katsuura,
    happy_cat,
    h_g_bat,
    expanded_griewanks_plus_rosenbrock,
    schaffers_f7
]
