import numpy as np

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def sigmoid_(x):
    return 1 / (1 + (np.exp(x * -1)))

def logistic_predict(input_variables, theta):
    predicted_values = []
    input_variables = add_intercept(input_variables)
    return sigmoid_(input_variables.dot(theta))


x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(logistic_predict(x3, theta3))
