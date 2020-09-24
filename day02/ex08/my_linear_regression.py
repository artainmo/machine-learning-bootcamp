import numpy as np
import matplotlib.pyplot as mpl

class My_linear_regression:
    def __init__(self, theta=[1,1], alpha=0.001, n_cycles=1000):
        self.theta = theta
        self.alpha = alpha
        self.n_cycles = n_cycles


    def add_intercept(self, x_values):
        return np.column_stack((np.full(x_values.shape[0], 1), x_values))


    def get_gradient(self, x_values, expected_values):
        lenght = x_values.shape[0]
        x_values = self.add_intercept(x_values)
        predicted_values = x_values.dot(self.theta)
        predicted_expected_difference = np.subtract(predicted_values, expected_values)
        return x_values.transpose().dot(predicted_expected_difference) / lenght

    def fit(self, x_values, expected_values, alpha=0.1, max_iter=1000):
        for x in range(0,max_iter):
            gradient = self.get_gradient(x_values, expected_values)
            self.theta = self.theta - (alpha*gradient)
        return self.theta

    def predict(self, input_variables):
        predicted_values = []
        input_variables = self.add_intercept(input_variables)
        return input_variables.dot(self.theta)


    def cost_function(self, predicted_values, expected_values):
        lenght = len(predicted_values)
        matrix = np.subtract(predicted_values, expected_values)
        return matrix.dot(matrix) / (2 * lenght)


    def plot(self, x_values, expected_values, cost):
        mpl.plot(x_values, self.theta[1]*x_values + self.theta[0], color="orange")
        mpl.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
        mpl.title("Cost: " + str(cost))
        mpl.show()
