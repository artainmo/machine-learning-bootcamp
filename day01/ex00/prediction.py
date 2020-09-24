import numpy as np

def add_intercept(x_values):
    res = []
    for x in x_values:
        res.append([1, x])
    return np.array(res)

def predict(x_values, theta):
    x_values = add_intercept(x_values)
    return x_values.dot(theta)


x = np.arange(1,6)
theta2 = np.array([0, 1])
print(predict(x, theta2))
