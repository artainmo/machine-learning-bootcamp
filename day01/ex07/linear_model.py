import numpy as np
import pandas as pd
import matplotlib as mpl
from my_linear_regression import My_linear_regression as MLR

data = pd.read_csv("../resources/are_blue_pills_magics.csv")
x_pill = np.array(data["Micrograms"])
expected_values = np.array(data["Score"])
print(x_pill)
print(expected_values)
linear_model1 = MLR(np.array([[89.0], [-8]]))
predicted_values_model1 = linear_model1.predict(x_pill)
print(predicted_values_model1)
cost = linear_model1.cost_function(predicted_values_model1, expected_values)
linear_model1.plot(x_pill, expected_values, cost)
