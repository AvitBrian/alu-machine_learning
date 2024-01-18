 #!/usr/bin/env python3
def cat_matrices2D(mat1, mat2, axis=0):
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        result_matrix = []
        for row in mat1:
            result_matrix.append(row.copy())
        for row in mat2:
            result_matrix.append(row.copy())

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        result_matrix = []
        for i in range(len(mat1)):
            result_matrix.append(mat1[i].copy() + mat2[i])

    return result_matrix
