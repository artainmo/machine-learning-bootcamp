def add_intercept(self, x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def sigmoid_(self, x):
    return 1 / (1 + (np.exp(x * -1)))

def logistic_gradient(self, input_variables, expected_values, theta):
    lenght = input_variables.shape[0]
    predicted_values = logistic_predict(input_variables, theta)
    predicted_expected_difference = np.subtract(predicted_values, expected_values)
    input_variables = add_intercept(input_variables)
    return input_variables.transpose().dot(predicted_expected_difference) / lenght

class MyLogisticRegression():
    def __init__(self, theta, alpha=0.001, n_cycle=1000):
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
            self.theta = self.theta - (self.alpha*gradient)
        return self.theta

    def cost_(expected_values, predicted_values, eps=1e-15):
        summation = 0
        for switch, predicted_value in zip(expected_values, predicted_values):
            summation += (switch * np.log(predicted_value + eps) + (1 - switch) * np.log(1 - predicted_value + eps))
        return
