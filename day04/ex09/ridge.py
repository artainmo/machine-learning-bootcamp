import numpy as np
import matplotlib.pyplot as mpl


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

def add_polynomial_features(x, power):
    power = range(1, power + 1)
    init = x
    for pow in power:
        x = np.column_stack((x, init ** pow))
    return x

class MyRidge:
    def __init__(self, thetas, alpha=0.001, n_cycle=1000, lambda_=0.5):
        self.thetas = thetas
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.lambda = lambda_

    def fit_(self, x_values, expected_values, alpha=0.1, max_iter=1000):
        for x in range(0,max_iter):
            gradient = self.regularized_linear_gradient(x_values, expected_values, self.theta, self.lambda)
            self.theta = self.theta - (alpha*gradient)
        return self.theta

    def regularized_cost_(expected_values, predicted_values, theta, lambda_):
        matrix = np.subtract(predicted_values, expected_values)
        matrix = matrix.dot(matrix) + (lambda_ * l2_regularization(theta))
        return matrix / (2 * predicted_values.shape[0])

    def predict_(self, input_variables):
        predicted_values = []
        input_variables = add_intercept(input_variables)
        return input_variables.dot(self.theta)

    def plot(self, x_values, expected_values, cost):
        mpl.plot(x_values, self.theta[1]*x_values + self.theta[0], color="orange")
        mpl.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
        mpl.title("Cost: " + str(cost))
        mpl.show()
