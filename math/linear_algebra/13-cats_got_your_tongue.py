#!/usr/bin/env python3
import numpy as np
"""_Cancatinating with NumPy_
"""

def np_cat(mat1, mat2, axis=0):
    """_summary_

    Args:
        mat1 (_type_): _Matrix 1_
        mat2 (_type_): _matrix 2_
        axis (int, optional): _The axis to be used_. Defaults to 0.

    Returns:
        _type_: _Concantination of two matrices_
    """    
    return np.concatenate((mat1, mat2), axis=axis)
