#!/usr/bin/python3

import cec2017.basic as basic
import cec2017.simple as simple
import cec2017.utils as utils

utils.surface_plot(simple.f1)

# time functions
for f in basic.all_functions:
    print('%s : %.3f' %(f.__name__, utils.time(f, points=100)))
