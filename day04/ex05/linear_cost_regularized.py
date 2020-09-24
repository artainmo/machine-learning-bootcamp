import numpy as np

def l2_regularization(theta):
    theta[0] = 0
    return theta.dot(theta)


def regularized_cost_(expected_values, predicted_values, theta, lambda_):
    matrix = np.subtract(predicted_values, expected_values)
    matrix = matrix.dot(matrix) + (lambda_ * l2_regularization(theta))
    return matrix / (2 * predicted_values.shape[0])


y = np.array([2, 14, -13, 5, 12, 4, -19])
y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20])
theta = np.array([1, 2.5, 1.5, -0.9])
print(regularized_cost_(y, y_hat, theta, .9))
