import numpy as np
import matplotlib.pyplot as mpl

def plot(x_values, y, theta):
    if isinstance(x_values, list) == True:
        x_values = np.array(x_values)
    if isinstance(y, list) == True:
        y = np.array(y)
    image = mpl.plot(x_values, theta[1]*x_values + theta[0], color="orange")
    image = mpl.plot(x_values, y, linestyle="",marker="o", color="blue")
    mpl.show(image)


# x = np.arange(1,6)
# y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
# theta1 = np.array([4.5, -0.2])
# plot(x, y, theta1)
