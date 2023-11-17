import numpy as np

def load_triangular_matrix(file_path):
    with open(file_path, 'r') as file:
        # Skip the first row
        lines = file.readlines()[1:]
        matrix = [list(map(int, line.split())) for line in lines]
    return matrix

def make_symmetric(matrix):
    n = len(matrix)
    symmetric_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):  # Only iterate up to the diagonal (inclusive)
            symmetric_matrix[i][j] = symmetric_matrix[j][i] = matrix[i][j]

    return symmetric_matrix


file_path = 'test.txt'
triangular_matrix = load_triangular_matrix(file_path)
symmetric_matrix = make_symmetric(triangular_matrix)


symmetric_matrix_np = np.array(symmetric_matrix)

print("Triangular Matrix (skipping the first row):")
for row in triangular_matrix:
    print(row)

print("\nSymmetric Matrix as NumPy array:")
print(symmetric_matrix_np)

