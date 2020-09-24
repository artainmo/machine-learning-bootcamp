import numpy as np

def simple_predict(input_variables, theta):
    predicted_values = []
    for variables in input_variables:
        predicted_value = theta[0]
        i = 1
        for values in variables:
            predicted_value += (theta[i] * values)
            i += 1
        predicted_values.append(predicted_value)
    return predicted_values


x = np.arange(1,13).reshape((4,3))
theta1 = np.array([0, 1, 0, 0])
print(simple_predict(x, theta1))
