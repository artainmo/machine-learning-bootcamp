import numpy as np

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def theta_0(theta):
    theta[0] = 0
    return theta

def regularized_linear_gradient(expected_values, x_values, theta, lambda_):
    lenght = x_values.shape[0]
    x_values = add_intercept(x_values)
    res = x_values.transpose().dot(np.subtract(x_values.dot(theta), expected_values)) + (lambda_ * theta_0(theta))
    return res / lenght

x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[-8,-4, 6],
[-5,-9, 6],
[ 1, -5, 11],
[9,-11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])
print(regularized_linear_gradient(y, x, theta, 1))
