import numpy as np

def split_x_y(set):
    set = set.transpose()
    set_x = set[:-1]
    set_y = set[-1:]
    return (set_x.transpose(), set_y.transpose())

def data_spliter(x, y, proportion):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    data = split_x_y(training_set) + split_x_y(test_set)
    return {"x_train" : data[0],
            "y_train" : data[1],
            "x_test" : data[2],
            "y_test" : data[3]}


def descriminate_classes(predicted_values, class_):
    i = 0
    while i < predicted_values.shape[0]:
        if predicted_values[i] == class_:
            predicted_values[i] = 1
        else:
            predicted_values[i] = 0
        i += 1
    return predicted_values


def minmax_normalization(x_values):
    i = 0
    l = 0
    max = np.max(x_values)
    min = np.min(x_values)
    range = max - min
    while i < x_values.shape[0]:
        l = 0
        while l < x_values.shape[1]:
            x_values[i][l] = (x_values[i][l] - min) / range
            l += 1
        i += 1
    return x_values


def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def sigmoid_(x):
    return 1 / (1 + (np.exp(np.multiply(x, -1))))

def logistic_predict(input_variables, theta):
    predicted_values = []
    input_variables = add_intercept(input_variables)
    return sigmoid_(input_variables.dot(theta))

def logistic_gradient(input_variables, expected_values, theta):
    lenght = input_variables.shape[0]
    predicted_values = logistic_predict(input_variables, theta)
    predicted_expected_difference = np.subtract(predicted_values, expected_values)
    input_variables = add_intercept(input_variables)
    return input_variables.transpose().dot(predicted_expected_difference) / lenght

class MyLogisticRegression():
    def __init__(self, theta, alpha=0.001, n_cycle=10000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = theta

    def predict_(self, input_variables):
        predicted_values = []
        input_variables = add_intercept(input_variables)
        return sigmoid_(input_variables.dot(self.theta))

    def fit_(self, x_values, expected_values):
        for x in range(0,self.max_iter):
            gradient = logistic_gradient(x_values, expected_values, self.theta)
            self.theta = np.subtract(self.theta, (self.alpha*gradient))
        return self.theta

    def cost_(self, expected_values, predicted_values, eps=1e-15):
        summation = 0
        for switch, predicted_value in zip(expected_values, predicted_values):
            summation += (switch * np.log(predicted_value + eps) + (1 - switch) * np.log(1 - predicted_value + eps))
        return summation / predicted_values.shape[0] * -1
