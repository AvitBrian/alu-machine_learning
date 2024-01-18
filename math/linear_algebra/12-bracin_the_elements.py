#!/usr/bin/env python3
#!/usr/bin/env python3
import numpy as np


def np_elementwise(mat1, mat2):
    """ Matrix Operations

    Args:
        mat1 (_type_): Matrix 1
        mat2 (_type_): Matrix 2

    Returns:
        _type_: different Operations
    """
    addition = mat1 + mat2
    substraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return addition, substraction, multiplication, division
