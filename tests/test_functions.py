import numpy as np


def test_simple_functions():
    from cec2017.simple import all_functions

    assert len(all_functions) == 10

    M = 2
    dims = [2, 10, 20, 30, 50, 100]

    for i, f in enumerate(all_functions):
        assert f.__name__ == f"f{1 + i}"
        for d in dims:
            x = np.zeros((M, d))
            y = f(x)
            assert y.shape == (M,)

    pass


def test_hybrid_functions():
    from cec2017.hybrid import all_functions

    assert len(all_functions) == 10

    M = 2
    dims = [10, 30, 50, 100]

    for i, f in enumerate(all_functions):
        assert f.__name__ == f"f{11 + i}"
        for d in dims:
            x = np.zeros((M, d))
            y = f(x)
            assert y.shape == (M,)


def test_composition_functions():
    from cec2017.composition import all_functions

    assert len(all_functions) == 10

    M = 2
    dims = [10, 30, 50, 100]

    for i, f in enumerate(all_functions):
        assert f.__name__ == f"f{21 + i}"
        for d in dims:
            x = np.zeros((M, d))
            y = f(x)
            assert y.shape == (M,)


def test_all_functions():
    from cec2017.functions import all_functions

    assert len(all_functions) == 30

    M = 2
    D = 10
    x = np.zeros((M, D))

    for i, f in enumerate(all_functions):
        assert f.__name__ == f"f{1 + i}"
        y = f(x)
        assert y.shape == (M,)


def test_list_inputs():
    from cec2017.functions import all_functions

    M = 2
    D = 10
    x = [0.0] * D
    x = [x] * M
    # functions should allow
    for f in all_functions:
        y = f(x)
        assert y.shape == (M,)
