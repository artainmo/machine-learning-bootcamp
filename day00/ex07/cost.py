import numpy as np

def cost_function(predicted_values, expected_values):
    i = 0
    summation = 0
    lenght = len(predicted_values)
    while i < lenght:
        summation += (predicted_values[i] - expected_values[i]) ** 2
        i += 1
    return summation / (2 * lenght)



# y_hat = np.array([2.0, 6.0, 10.0, 14.0, 18.0])
# y = np.array([2., 7., 12. ,17., 22.])
# print(cost_(y, y_hat))
