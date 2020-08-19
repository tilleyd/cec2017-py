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
        # Note: the i+1 term is not in the CEC function definitions, but is
        # in the code and in any definition you find online
        sm += (i+1)*x[i]
    sm = 0.5 * sm
    sm = sm * sm
    return sms + sm + (sm * sm)

def rosenbrock(x):
    x = 0.02048 * x + 1.0
    sm = 0
    for i in range(0, len(x)-1):
        t1 = x[i]*x[i] - x[i+1]
        t1 = 100*t1*t1
        t2 = x[i] - 1
        t2 = t2*t2
        sm += t1 + t2
    return sm

def rastrigin(x):
    # Note: the 0.0512 shrinking is omitted in the problem definitions but is
    # present in the provided code
    x = 0.0512 * x
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

def lunacek_bi_rastrigin(x, shift=None, rotation=None):
    # a special case; we need the shift vector and rotation matrix
    nx = len(x)
    if shift is None:
        shift = np.zeros(nx)

    # calculate the coefficients
    mu0=2.5
    tmpx = np.zeros(nx)
    s = 1 - 1 / (2 * ((nx+20)**0.5) - 8.2)
    mu1 = -((mu0*mu0-1)/s)**0.5

    # shift and scale
    y = 0.1 * (x - shift)

    for i in range(0, nx):
        tmpx[i] = 2*y[i]
        if shift[i] < 0.0:
            tmpx[i] *= -1.0

    z = tmpx.copy()
    tmpx = tmpx + mu0

    t1=0.0
    t2=0.0
    for i in range(0, nx):
        t = tmpx[i]-mu0
        t1 += t*t
        t = tmpx[i]-mu1
        t2 += t*t
    t2 *= s
    t2 += nx

    y = z if rotation is None else np.matmul(rotation, z)

    t = 0.0
    y = np.cos(2.0*np.pi*y)
    for i in range(0, nx):
        t += y[i]

    r = t1 if t1 < t2 else t2
    return r + 10.0*(nx-t)

def non_cont_rastrigin(x, shift=None, rotation=None):
    # a special case; we need the shift vector and rotation matrix
    if shift is None:
        shift = np.zeros(x.shape)

    nx = len(x)
    sm = 0.0
    for i in range(0, nx):
        if abs(x[i]-shift[i]) > 0.5:
            x[i] = shift[i] + np.floor(2*(x[i]-shift[i])+0.5)/2

    z = 0.0512 * (x - shift)
    z = z if rotation is None else np.matmul(rotation, z)

    for i in range(0, nx):
        sm += (z[i]*z[i] - 10.0*np.cos(2.0*np.pi*z[i]) + 10.0)
    return sm

def levy(x):
    # Note: the function definitions state to scale by 5.12/100, but the code
    # doesn't do this, and the example graph in the definitions correspond to
    # the version without scaling
    # x = 0.0512 * x
    nx = len(x)
    w = 1.0 + 0.25*(x - 1.0)

    term1 = (np.sin(np.pi*w[0]))**2
    term3 = ((w[nx-1] - 1)**2) * (1 + ((np.sin(2*np.pi*w[nx-1]))**2))

    sm = 0.0

    for i in range(0, nx-1):
        wi = w[i]
        newv = ((wi-1)**2) * (1 + 10*((np.sin(np.pi*wi+1))**2))
        sm += newv

    return term1 + sm + term3

def modified_schwefel(x):
    nx = len(x)
    x = 10.0 * x # scale to search range
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
    factor = 6 / (len(x) - 1)
    sm = 0.0
    for i in range(0, len(x)):
        sm += x[i]*x[i] * 10**(i*factor)
    return sm

def discus(x):
    sm = 1e+6*x[0]*x[0]
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
    x = 0.005 * x
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
    x = 6.0 * x
    factor = 1/4000
    cs = np.cos(x / np.arange(start=1, stop=len(x)+1))
    sm = 0.0
    pd = 1.0
    for i in range(0, len(x)):
        sm += factor*x[i]*x[i]
        pd *= cs[i]
    return sm - pd + 1

def katsuura(x):
    x = 0.05 * x
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
    x = (0.05 * x) - 1
    nx = len(x)
    sm = 0.0
    smsq = 0.0
    for i in range(0, nx):
        sm += x[i]
        smsq += x[i]*x[i]
    return (abs(smsq - nx))**0.25 + (0.5*smsq + sm)/nx + 0.5

def h_g_bat(x):
    x = (0.05 * x) - 1
    nx = len(x)
    sm = 0.0
    smsq = 0.0
    for i in range(0, nx):
        sm += x[i]
        smsq += x[i]*x[i]
    return (abs(smsq*smsq - sm*sm))**0.5 + (0.5*smsq + sm)/nx + 0.5

def expanded_griewanks_plus_rosenbrock(x):
    x = (0.05 * x) + 1

    sm = 0.0
    for i in range(0, len(x)-1):
        tmp1 = x[i]*x[i]-x[i+1]
        tmp2 = x[i] - 1.0
        temp = 100*tmp1*tmp1 + tmp2*tmp2
        sm += (temp*temp)/4000.0 - np.cos(temp) + 1
        tmp1 = x[-1]*x[-1] - x[0]
        tmp2 = x[-1] - 1
        temp = 100.0*tmp1*tmp1 + tmp2*tmp2
        sm += (temp*temp)/4000.0 - np.cos(temp) + 1.0
    return sm

def schaffers_f7(x):
    nx = len(x)
    # Note: the function definitions state to scale by 0.5/100, but the code
    # doesn't do this, and the example graph in the definitions correspond to
    # the version without scaling
    # x = 0.005 * x
    sm = 0.0
    for i in range(0, nx-1):
        si = (x[i]*x[i] + x[i+1]*x[i+1])**0.5
        tmp = np.sin(50.0*(si**0.2))
        # Note: the original code has this error here (tmp shouldn't be squared)
        # that I'm keeping for consistency.
        sm += (si**0.5) * (tmp*tmp + 1)
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
