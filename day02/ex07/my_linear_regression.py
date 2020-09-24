import numpy as np
import matplotlib.pyplot as mpl

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

class My_linear_regression:
    def __init__(self, theta=[1,1], alpha=0.001, n_cycles=1000):
        self.theta = theta
        self.alpha = alpha
        self.n_cycles = n_cycles

    def fit_(self, x_values, expected_values, alpha=0.1, max_iter=1000):
        for x in range(0,max_iter):
            gradient = self.get_gradient(x_values, expected_values)
            self.theta = self.theta - (alpha*gradient)
        return self.theta

    def predict_(self, input_variables):
        predicted_values = []
        input_variables = add_intercept(input_variables)
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
