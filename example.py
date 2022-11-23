#!/usr/bin/python3

import numpy as np

# evaluate a specific function a few times with one sample
import cec2017.functions as functions

f = functions.f5
dimension = 30
for i in range(0, 10):
    x = np.random.uniform(low=-100, high=100, size=dimension)
    y = f([x])[0]
    print(f"f5({x[0]:.2f}, {x[1]:.2f}, ...) = {y:.2f}")

# or with a population (i.e. multiple samples)
f = functions.f3
samples = 3
dimension = 30
for i in range(0, 10):
    x = np.random.uniform(low=-100, high=100, size=(samples, dimension))
    y = f(x)
    for i in range(samples):
        print(f"f5({x[i, 0]:.2f}, {x[i, 1]:.2f}, ...) = {y[i]:.2f}")

# or evaluate each function once
samples = 3
dimension = 50
for f in functions.all_functions:
    x = np.random.uniform(-100, 100, size=(samples, dimension))
    val = f(x)
    for i in range(samples):
        print(f"{f.__name__}({x[i, 0]:.2f}, {x[i, 1]:.2f}, ...) = {y[i]:.2f}")

# or all hybrid functions (or basic, simple or composite functions...)
import cec2017.simple as simple # cec2017.basic cec2017.hybrid cec2017.composite
samples = 3
dimension = 50
for f in simple.all_functions: # f1 to f10
    x = np.random.uniform(low=-100, high=100, size=(samples, dimension))
    y = f(x)
    for i in range(samples):
        print(f"{f.__name__}({x[i, 0]:.2f}, {x[i, 1]:.2f}, ...) = {y[i]:.2f}")

# make a surface plot of f27
import cec2017.utils as utils
utils.surface_plot(functions.f27, points=120)
# or of f14 (not defined for D=2, so specify D=10)
utils.surface_plot(functions.f14, points=120, dimension=10)
# or even a base function like Ackley
import cec2017.basic as basic
utils.surface_plot(basic.ackley, points=120)
