#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None

    new_mat = []
    for i in range(len(mat1)):
        temp_row = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat2)):
                sum += mat1[i][k] * mat2[k][j]
            temp_row.append(sum)
        new_mat.append(temp_row)

    return new_mat
