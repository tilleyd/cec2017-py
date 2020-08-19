# CEC 2017 Python

Python 3 module containing a native implementation of the CEC 2017 benchmark functions (single objective optimization). The implementation is adapted from Awad's original C implementation, available on [Suganthan's GitHub repo](https://github.com/P-N-Suganthan/CEC2017-BoundContrained), along with the problem definitions [1].

Although there are wrappers for the C code, this module is easier to use, natively supports numpy arrays and is (_much_) more readable than the C implementation.

During the implementation of this module, a few differences between the problem definitions and the original code were picked up on. In these cases, the implementation always follows whatever the original code did to remain compatible, and are marked with `Note:` comments.

As per the problem definitions, functions are defined for 10, 30, 50 and 100 dimensions, with functions `f1` to `f10` and `f21` to `f28` also being defined for 2 and 20 dimensions. If you provide custom rotation matrices (and shuffles, where applicable) you can however use arbitrary dimensions. Below are some surface plots for functions `f1` to `f10` over 2 dimensions.

![Function Surface Plots](extra/plots.jpg)

> \[1\] _Awad, N. H., Ali, M. Z., Suganthan, P. N., Liang, J. J., & Qu, B. Y. (2016). Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective Bound Constrained Real-Parameter Numerical Optimization._

## Features

- Native implementation of all CEC 2017 single objective functions
- Pre-defined rotations, shifts and shuffles for 2, 10, 20, 30, 50 and 100 dimensions
- Allows custom rotations, shifts and shuffles
- Convenient surface plot utility
- Easy access to basic functions f1 to f19 (e.g. Ackley, Discus, etc.)

## Installation

```
git clone https://github.com/tilleyd/cec2017-py
cd cec2017-py
python3 setup.py install
```

## Usage

Below is a simple example of executing either a single function or all functions. See [example.py](example.py) for more advanced use cases.

```py
# Using only f5:
from cec2017.functions import f5
x = np.random.uniform(-100, 100, size=50)
val = f5(x)
print('f5(x) = %.6f' %val)

# Using all functions:
from cec2017.functions import all_functions
for f in all_functions:
    x = np.random.uniform(-100, 100, size=50)
    val = f(x)
    print('%s(x) = %.6f' %( f.__name__, val ))
```

## License

Copyright &copy; 2020 Duncan Tilley
See the [license notice](LICENSE.txt) for full details.

## Issues

If you see any issues or possible improvements, please open an issue or feel free to make a pull request.
