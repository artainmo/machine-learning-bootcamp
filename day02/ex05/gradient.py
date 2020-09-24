import numpy as np

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))


def gradient(x_values, expected_values, theta):
    lenght = x_values.shape[0]
    x_values = add_intercept(x_values)
    predicted_values = x_values.dot(theta)
    predicted_expected_difference = np.subtract(predicted_values, expected_values)
    return x_values.transpose().dot(predicted_expected_difference) / lenght


x = np.array([[ -6, -7, -9],
            [ 13, -2, 14],
            [ -7, 14, -1],
            [ -8, -4, 6],
            [ -5, -9, 6],
            [ 1, -5, 11],
            [ 9,-11, 8]])
y = np.array([2, 14, -13, 5, 12, 4, -19])
theta1 = np.array([3,0.5,-6, 2])
print(gradient(x, y, theta1))
