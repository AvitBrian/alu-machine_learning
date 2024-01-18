#!/usr/bin/env python3
#!/usr/bin/env python3
import numpy as np


def np_elementwise(mat1, mat2):
    addition = mat1 + mat2
    substraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return addition, substraction, multiplication, division
