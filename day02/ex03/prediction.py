import numpy as np

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def predict(input_variables, theta):
    predicted_values = []
    input_variables = add_intercept(input_variables)
    return input_variables.dot(theta)


x = np.arange(1,13).reshape((4,3))
theta1 = np.array([-1.5, 0.6, 2.3, 1.98])
print(predict(x, theta1))
