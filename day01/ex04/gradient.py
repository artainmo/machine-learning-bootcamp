import numpy as np

def add_intercept(x_values):
    res = []
    for x in x_values:
        res.append([1, x])
    return np.array(res)


def gradient(x_values, expected_values, theta):
    lenght = len(x_values)
    x_values = add_intercept(x_values)
    predicted_values = x_values.dot(theta)
    predicted_expected_difference = np.subtract(predicted_values, expected_values)
    return x_values.transpose().dot(predicted_expected_difference) / lenght

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta1 = np.array([2, 0.7])
print(gradient(x, y, theta1))
