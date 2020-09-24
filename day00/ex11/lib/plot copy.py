import numpy as np
import matplotlib.pyplot as mpl

def plot(x_values, expected_values, theta, cost):
    if isinstance(x_values, list) == True:
        x_values = np.arraexpected_values(x_values)
    if isinstance(expected_values, list) == True:
        expected_values = np.arraexpected_values(expected_values)
    mpl.plot(x_values, theta[1]*x_values + theta[0], color="orange")
    mpl.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
    mpl.title("Cost: " + str(cost))
    mpl.show()

 
# x = np.arange(1,6)
# expected_values = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
# theta1 = np.array([4.5, -0.2])
# plot(x, expected_values, theta1, 3.02)
