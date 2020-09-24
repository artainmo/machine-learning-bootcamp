import numpy as np

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def sigmoid_(x):
    return 1 / (1 + (np.exp(x * -1)))

def logistic_predict(input_variables, theta):
    predicted_values = []
    input_variables = add_intercept(input_variables)
    return sigmoid_(input_variables.dot(theta))
