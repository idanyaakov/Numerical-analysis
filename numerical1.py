# Aviram ben ishay - 208186171
# Idan yaakov - 207468554
# GIT: https://github.com/Aviram96/Numerical-analysis/tree/main

import numpy as np


def is_invertible(matrix):
    # Check if matrix is nXn and its determinant
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")
    if np.linalg.det(matrix) == 0:
        raise ValueError("Matrix is singular, cannot find its inverse.")


def find_norm(matrix):
    # computes the condition number of a 2 matrix
    max_norm = 0
    for i in range(len(matrix)):
        temp_norm = 0
        for j in range(len(matrix)):
            temp_norm += abs(matrix[i][j])
        if max_norm < temp_norm:
            max_norm = temp_norm
    return max_norm


def calc_cond(normA, norm_inverseOf_A):
    # Calculates the cond of the two matrix's
    cond_number = normA * norm_inverseOf_A
    return cond_number


def gaussian_inverse(matrix):
    # computes the inverse of a given matrix using Gaussian elimination
    # create a new matrix that describes the matrix with the unit matrix next to it
    n = matrix.shape[0]
    augmented_matrix = np.hstack((matrix, np.eye(n)))

    # Forward elimination
    for i in range(n):
        if augmented_matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / augmented_matrix[i, i]
            augmented_matrix[i] = augmented_matrix[i] * scalar

        # Make all rows below this one 0 in the current column
        for j in range(i + 1, n):
            row_factor = augmented_matrix[j, i]
            augmented_matrix[j] = augmented_matrix[j] - row_factor * augmented_matrix[i]

    # Backward elimination
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            row_factor = augmented_matrix[j, i]
            augmented_matrix[j] = augmented_matrix[j] - row_factor * augmented_matrix[i]

    # Extract the inverse matrix from the augmented matrix
    inverse_matrix = augmented_matrix[:, n:]
    return inverse_matrix


if __name__ == '__main__':
    # Example matrix
    A = np.array([[1, -1,  -2], [2, -3, -5], [-1, 3, 5]])
    try:
        # try to inverse matrix A find the norma and print both, print norma of A,print cond of A and inverse ofA
        is_invertible(A)
        A_inverse = gaussian_inverse(A)
        print(f"\nMatrix A:\n {A}")
        print(f"\ninverse of A:\n {A_inverse}")
        print(f"\n||A||: {find_norm(A)}")
        print(f"\ninverse of ||A||: {find_norm(A_inverse)}")
        print(f"\nCOND = {calc_cond(find_norm(A), find_norm(A_inverse))}")

    except ValueError as e:
        print(str(e))
