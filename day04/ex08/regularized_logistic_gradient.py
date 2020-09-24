import numpy as np

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def theta_0(theta):
    theta[0] = 0
    return theta

def sigmoid_(x):
    return 1 / (1 + (np.exp(x * -1)))

def regularized_logistic_gradient(expected_values, x_values, theta, lambda_):
    lenght = x_values.shape[0]
    x_values = add_intercept(x_values)
    res = x_values.transpose().dot(np.subtract(sigmoid_(x_values.dot(theta)), expected_values)) + (lambda_ * theta_0(theta))
    return res / lenght

x = np.array([[0, 2, 3, 4],
[2, 4, 5, 5],
[1, 3, 2, 7]])
y = np.array([[0], [1], [1]])
theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(regularized_logistic_gradient(y, x, theta, 0.5))
