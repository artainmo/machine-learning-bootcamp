import numpy as np
import matplotlib.pyplot as mpl

def l2_regularization(theta):
    theta[0] = 0
    return theta.transpose().dot(theta)

def split_x_y(set):
    set = set.transpose()
    set_x = set[:-1]
    set_y = set[-1:]
    return (set_x.transpose(), set_y.transpose())

def data_spliter(x, y, proportion=0.8):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return split_x_y(training_set) + split_x_y(test_set)

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def theta_0(theta):
    theta[0] = 0
    return theta

def sigmoid_(x):
    return 1 / (1 + (np.exp(x * -1)))

def regularized_logistic_gradient(expected_values, x_values, theta, lambda_):
    lenght = x_values.shape[0]
    x_values = add_intercept(x_values)
    res = x_values.transpose().dot(np.subtract(sigmoid_(x_values.dot(theta)), expected_values)) + (lambda_ * theta_0(theta))
    return res / lenght

def add_polynomial_features(x, power):
    power = range(1, power + 1)
    init = x
    for pow in power:
        x = np.column_stack((x, init ** pow))
    return x

class MyLogisticRegression():
    def __init__(self, thetas, alpha=0.0001, n_cycle=1000000, lambda_=0.5):
        self.theta = thetas
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.lambda_ = lambda_

    def fit_(self, x_values, expected_values):
        for x in range(0,self.n_cycle):
            gradient = regularized_logistic_gradient(expected_values, x_values, self.theta, self.lambda_)
            self.theta = self.theta - (self.alpha*gradient)
        return self.theta

    def cost_(expected_values, predicted_values):
        one = np.ones(predicted_values.shape)
        summation = expected_values.dot(np.log(predicted_values)) + (np.subtract(one, expected_values).dot(np.log(np.subtract(one, predicted_values))))
        summation = (summation / predicted_values.shape[0] * -1)
        return summation + ((self.lambda_ / (2 * predicted_values.shape[0])) * l2_regularization(self.theta))

    def predict_(self, input_variables):
        predicted_values = []
        input_variables = add_intercept(input_variables)
        return sigmoid_(input_variables.dot(self.theta))

    def plot_(self, x_values, expected_values, cost):
        mpl.plot(x_values, self.theta[1]*x_values + self.theta[0], color="orange")
        mpl.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
        mpl.title("Cost: " + str(cost))
        mpl.show()
