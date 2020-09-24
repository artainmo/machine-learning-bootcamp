import numpy as np

def linear_regression_cost_function(predicted_values, expected_values):
    lenght = len(predicted_values)
    matrix = np.subtract(predicted_values, expected_values)
    return matrix.dot(matrix) / (2 * lenght)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(linear_regression_cost_function(X, Y))
