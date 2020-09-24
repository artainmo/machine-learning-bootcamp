import numpy as np

def l2_regularization(theta):
    theta[0] = 0
    return theta.dot(theta)

def regularized_logistic_cost_(expected_values, predicted_values, theta, lambda_):
    one = np.ones(predicted_values.shape)
    summation = expected_values.dot(np.log(predicted_values)) + (np.subtract(one, expected_values).dot(np.log(np.subtract(one, predicted_values))))
    summation = (summation / predicted_values.shape[0] * -1)
    return summation + ((lambda_ / (2 * predicted_values.shape[0])) * l2_regularization(theta))

y = np.array([1, 1, 0, 0, 1, 1, 0])
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
theta = np.array([1, 2.5, 1.5, -0.9])
print(regularized_logistic_cost_(y, y_hat, theta, .9))
