#!/usr/bin/python3

import cec2017.basic as basic
import cec2017.utils as utils

utils.surface_plot(basic.schaffers_f7)

# time functions
for f in basic.all_functions:
    print('%s : %.3f' %(f.__name__, utils.time(f, points=100)))
