#Fit a machine learning model to a dataset: Theta parameters converge so minimal cost
import numpy as np


def add_intercept(x_values):
    res = []
    for x in x_values:
        res.append([1, x])
    return np.array(res)


def get_gradient(x_values, expected_values, theta):
    lenght = len(x_values)
    x_values = add_intercept(x_values)
    predicted_values = x_values.dot(theta)
    predicted_expected_difference = np.subtract(predicted_values, expected_values)
    return x_values.transpose().dot(predicted_expected_difference) / lenght


def fit(x_values, expected_values, theta=[1, 1], alpha=0.1, max_iter=1000):
    for x in range(0,max_iter):
        gradient = get_gradient(x_values, expected_values, theta)
        theta[0] = theta[0] - (alpha*gradient[0])
        theta[1] = theta[1] - (alpha*gradient[1])
    return theta

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta = np.array([1, 1])
theta = fit(x, y, alpha=5e-8, max_iter = 1500000)
print(theta[0])
print(theta[1])
