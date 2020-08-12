# cec2017.basic
# Author: Duncan Tilley
# Basic function definitions

import numpy as np

def bent_cigar(x):
    sm = 0.0
    for i in range(1, len(x)):
        sm += x[i]*x[i]
    sm *= 10e6
    return x[0]*x[0] + sm

def sum_diff_pow(x):
    sm = 0.0
    for i in range(0, len(x)):
        sm += (abs(x[i])) ** (i+1)
    return sm

def zakharov(x):
    sms = 0.0
    sm = 0.0
    for i in range(0, len(x)):
        sms += x[i]*x[i]
        sm += x[i]
    sm = 0.5 * sm
    sm = sm * sm
    return sms + sm + (sm * sm)

def rosenbrock(x):
    sm = 0
    for i in range(0, len(x)-1):
        t1 = x[i]*x[i] - x[i+1]
        t1 = 100*t1*t1
        t2 = x[i] - 1
        t2 = t2*t2
        sm += t1 + t2
    return sm

def rastrigin(x):
    tpi = 2.0 * np.pi
    sm = 0.0
    cs = np.cos(tpi*x)
    for i in range(0, len(x)):
        sm += x[i]*x[i] - 10*cs[i]
    return sm + 10*len(x)

def expanded_schaffers_f6(x):
    sm = 0.0
    for i in range(0, len(x)-1):
        t = x[i]*x[i] + x[i+1]*x[i+1]
        t1 = np.sin(np.sqrt(t))
        t1 = t1*t1 - 0.5
        t2 = 1 + 0.001*t
        t2 = t2*t2
        sm += 0.5 + t1/t2
    return sm

def lunacek_bi_rastrigin(x, o=None):
    if o == None:
        o = np.zeros(x.shape)

    x = (x - o) * 0.1 # calculate y
    nx = len(x)
    s = 1.0 - 1.0 / (2.0*np.sqrt(nx+20) - 8.2)
    mu0 = 2.5
    mu1 = -np.sqrt((mu0*mu0-1)/s)

    after = 0.0
    t1 = 0.0
    t2 = 0.0
    for i in range(0, nx):
        z = (-2 if o[i] < 0.0 else 2) * x[i]
        after += np.cos(2.0*np.pi*z)
        t1 += z*z
        t = z + mu0 - mu1
        t2 += t*t

    t2 *= s
    t2 += nx

    r = t1 if t1 < t2 else t2

    return r + 10.0*(nx-after)

def non_cont_rotated_rastrigin(x, o=None):
    if o == None:
        o = np.zeros(x.shape)

    x = 0.0512*(x-o)

    sm = 0.0
    for i in range(0, len(x)):
        t = x[i] if abs(x[i]) <= 0.5 else round(2*x[i])*0.5
        sm += t*t - 10*np.cos(2*np.pi*t) + 10
    return sm

def levy(x):
    # do first term and first summation together
    w = 1 + (x[0] - 1) * 0.25
    t1 = np.sin(np.pi*w)
    t1 = t1*t1
    t = w - 1
    t = t*t
    sn = np.sin(np.pi*w + 1)
    sn = sn*sn
    t1 += t * (1 + 10 * sn)

    # rest of summation
    t2 = 0
    for i in range(1, len(x) - 1):
        w = 1 + (x[i] - 1) * 0.25
        t = w - 1
        t = t*t
        sn = np.sin(np.pi*w + 1)
        sn = sn*sn
        t2 += t * (1 + 10 * sn)

    # last term
    w = 1 + (x[-1] - 1) * 0.25
    t = w - 1
    t = t*t
    sn = np.sin(2*np.pi*w)
    sn = sn*sn
    t3 = t * (1 + sn)
    return t1 + t2 + t3

def modified_schwefel(x):
    nx = len(x)
    sm = 0.0
    for i in range(0, nx):
        z = x[i] + 420.9687462275036
        if z < -500:
            zm = (abs(z) % 500) - 500
            t = z + 500
            t = t*t
            sm += zm * np.sin(np.sqrt(abs(zm))) - t / (10000*nx)
        elif z > 500:
            zm = 500 - (z % 500)
            t = z - 500
            t = t*t
            sm += zm * np.sin(np.sqrt(abs(zm))) - t / (10000*nx)
        else:
            sm += z * np.sin(np.sqrt(abs(z)))

    return 418.9829*nx - sm

def high_conditioned_elliptic(x):
    factor = 1 / (len(x) - 1)
    sm = 0.0
    for i in range(0, len(x)):
        sm += x[i]*x[i] * 10e+6**(i*factor)
    return sm

def discus(x):
    sm = 10e+6*x[0]*x[0]
    for i in range(1, len(x)):
        sm += x[i]*x[i]
    return sm

def ackley(x):
    smsq = 0.0
    smcs = 0.0
    cs = np.cos((2*np.pi)*x)
    for i in range(0, len(x)):
        smsq += x[i]*x[i]
        smcs += cs[i]
    inx = 1/len(x)
    return -20*np.exp(-0.2*np.sqrt(inx*smsq)) - np.exp(inx*smcs) + 20 + np.e

def weierstrass(x):
    k = np.arange(start=0, stop=21, step=1)
    ak = 0.5**k
    bk = np.pi * (3**k)
    sm = 0.0
    for i in range(0, len(x)):
        kcs = ak * np.cos(2*(x[i]+0.5)*bk)
        ksm = 0.0
        for j in range(0, 21):
            ksm += kcs[j]
        sm += ksm
    kcs = ak * np.cos(bk)
    ksm = 0.0
    for j in range(0, 21):
        ksm += kcs[j]
    return sm - len(x)*ksm

def griewank(x):
    factor = 1/4000
    cs = np.cos(x / np.arange(start=1, stop=len(x)+1))
    sm = 0.0
    pd = 1.0
    for i in range(0, len(x)):
        sm += factor*x[i]*x[i]
        pd *= cs[i]
    return sm - pd + 1

def katsuura(x):
    nx = len(x)
    pw = 10/(nx**1.2)
    prd = 1.0
    tj = 2**np.arange(start=1, stop=33, step=1)
    for i in range(0, nx):
        tjx = tj*x[i]
        t = np.abs(tjx - np.round(tjx)) / tj
        tsm = 0.0
        for j in range(0, 32):
            tsm += t[j]
        prd *= (1+ (i+1)*tsm)**pw
    df = 10/(nx*nx)
    return df*prd - df

def happy_cat(x):
    nx = len(x)
    sm = 0.0
    smsq = 0.0
    for i in range(0, nx):
        sm += x[i]
        smsq += x[i]*x[i]
    return (abs(smsq - nx))**0.25 + (0.5*smsq + sm)/nx + 0.5

def h_g_bat(x):
    nx = len(x)
    sm = 0.0
    smsq = 0.0
    for i in range(0, nx):
        sm += x[i]
        smsq += x[i]*x[i]
    return (abs(smsq*smsq - sm*sm))**0.5 + (0.5*smsq + sm)/nx + 0.5

def expanded_griewanks_plus_rosenbrock(x):
    sm = 0.0
    for i in range(0, len(x) - 1):
        # rosenbrok
        t1 = x[i]*x[i] - x[i+1]
        t1 = 100 * t1*t1
        t2 = x[i] - 1
        t2 = t2*t2
        y = t1 + t2
        # griewank
        sm += y*y/4000 - np.cos(y) + 1
    t1 = x[-1]*x[-1] - x[0]
    t1 = 100 * t1*t1
    t2 = x[-1] - 1
    t2 = t2*t2
    y = t1 + t2
    sm += y*y/4000 - np.cos(y) + 1
    return sm

def schaffers_f7(x):
    nxm = len(x)-1
    sm = 0.0
    for i in range(0, nxm):
        s = (x[i]*x[i] + x[i+1]*x[i+1])**0.5
        sm += s**0.5 * (np.sin(50*s**0.2) + 1)
    sm = (1/nxm)*sm
    sm = sm*sm
    return sm

all_functions = [
    bent_cigar,
    sum_diff_pow,
    zakharov,
    rosenbrock,
    rastrigin,
    expanded_schaffers_f6,
    lunacek_bi_rastrigin,
    non_cont_rotated_rastrigin,
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
