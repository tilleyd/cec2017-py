#!/usr/bin/python3

import numpy as np
# Accepted dimensions are 2, 10, 20, 30, 50 or 100
# (f11 - f20 and f29 - f30 not defined for D = 2)
D = 10

# evaluate a specific function a few times
import cec2017.functions as functions
f = functions.f5
for i in range(0, 10):
    x = np.random.uniform(low=-100, high=100, size=D)
    y = f(x)
    print('%s( %.1f, %.1f, ... ) = %.2f' %(f.__name__, x[0], x[1], y))

# or evaluate each function once
for f in functions.all_functions:
    x = np.random.uniform(low=-100, high=100, size=D)
    y = f(x)
    print('%s( %.1f, %.1f, ... ) = %.2f' %(f.__name__, x[0], x[1], y))

# or all hybrid functions (or basic, simple or composite functions...)
import cec2017.simple as simple # cec2017.basic cec2017.hybrid cec2017.composite
for f in simple.all_functions: # f1 to f10
    x = np.random.uniform(low=-100, high=100, size=D)
    y = f(x)
    print('%s( %.1f, %.1f, ... ) = %.2f' %(f.__name__, x[0], x[1], y))

# make a surface plot of f27
import cec2017.utils as utils
utils.surface_plot(functions.f27, points=120)
# or of f14 (not defined for D=2, so specify D=10)
utils.surface_plot(functions.f14, points=120, dimension=10)
# or even a base function like Ackley
import cec2017.basic as basic
utils.surface_plot(basic.ackley, points=120)
