def multiply_matrices(a, b):
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])

    result = [[0] * cols_b for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]

    return result

matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6], [7, 8]]

print("Result:", multiply_matrices(matrix_a, matrix_b))
