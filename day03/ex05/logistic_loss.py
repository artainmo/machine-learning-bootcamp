import numpy as np
from logistic_predict import *

def log_loss_(expected_values, predicted_values, eps=1e-15):
    summation = 0
    for switch, predicted_value in zip(expected_values, predicted_values):
        summation += (switch * np.log(predicted_value + eps) + (1 - switch) * np.log(1 - predicted_value + eps))
    return summation / predicted_values.shape[0] * -1


y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
y_hat3 = logistic_predict(x3, theta3)
print(log_loss_(y3, y_hat3))
