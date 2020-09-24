import numpy as np

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))


def get_gradient(x_values, expected_values, theta):
    lenght = x_values.shape[0]
    x_values = add_intercept(x_values)
    predicted_values = x_values.dot(theta)
    predicted_expected_difference = np.subtract(predicted_values, expected_values)
    return x_values.transpose().dot(predicted_expected_difference) / lenght

def fit(x_values, expected_values, theta=[1, 1], alpha=0.1, max_iter=1000):
    for x in range(0,max_iter):
        gradient = get_gradient(x_values, expected_values, theta)
        theta = theta - (alpha*gradient)
    return theta

x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])
theta2 = fit(x, y, theta, alpha = 0.0005, max_iter=42000)
print(theta2)
