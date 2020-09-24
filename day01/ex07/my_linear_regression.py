import numpy as np
import matplotlib.pyplot as mpl

class My_linear_regression:
    def __init__(self, theta=[1,1], alpha=0.001, n_cycles=1000):
        self.theta = theta
        self.alpha = alpha
        self.n_cycles = n_cycles

    def add_intercept(self, x_values):
        res = []
        for x in x_values:
            res.append([1, x])
        return np.array(res)

    def predict(self, x_values):
        x_values = self.add_intercept(x_values)
        return x_values.dot(self.theta).reshape(1,-1)[0]

    def cost_function(self, predicted_values, expected_values):
        lenght = len(predicted_values)
        matrix = np.subtract(predicted_values, expected_values)
        return matrix.dot(matrix) / (2 * lenght)

    def gradient(self, x_values, expected_values):
        lenght = len(x_values)
        x_values = self.add_intercept(x_values)
        predicted_values = x_values.dot(self.theta)
        predicted_expected_difference = np.subtract(predicted_values, expected_values)
        return x_values.transpose().dot(predicted_expected_difference) / lenght

    def fit(self, x_values, expected_values, theta=[1, 1], alpha=0.1, max_iter=1000):
        for x in range(0,max_iter):
            gradient = self.get_gradient(x_values, expected_values, self.theta)
            self.theta[0] = self.theta[0] - (alpha*gradient[0])
            self.theta[1] = self.theta[1] - (alpha*gradient[1])
        return theta

    def plot(self, x_values, expected_values, cost):
        mpl.plot(x_values, self.theta[1]*x_values + self.theta[0], color="orange")
        mpl.plot(x_values, self.theta[1]*x_values + self.theta[0], color="orange")
        mpl.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
        mpl.title("Cost: " + str(cost))
        mpl.show()
