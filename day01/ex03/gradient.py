import numpy as np

def simple_gradient(x_values, expected_values, theta):
    summation = 0
    gradient = []
    lenght = len(x_values)
    for x, y in zip(x_values, expected_values):
        summation += ((theta[0] + theta[1]*x) - y)
    gradient.append((summation / lenght)*-1 + theta[0])
    summation = 0
    for x, y in zip(x_values, expected_values):
        summation += (((theta[0] + theta[1]*x) - y)*x)
    gradient.append((summation / lenght)*-1 + theta[1])
    return gradient


x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta = np.array([1, -0.4])
print(simple_gradient(x, y, theta))
