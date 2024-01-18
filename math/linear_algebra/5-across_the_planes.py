 #!/usr/bin/env python3
def add_matrices2D(mat1, mat2):

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
    sum_matrix = []
    for row1, row2 in zip(mat1, mat2):
        sum_row = []
        for a, b in zip(row1, row2):
            sum_row.append(a + b)
        sum_matrix.append(sum_row)

    return sum_matrix