from lib.tools import *
from lib.matrix import *
from lib.vector import *
import numpy as np

def predict_(x, theta):
    x = Matrix(add_intercept(x))
    return x.vector_multiplication(Vector(theta))





x_values = np.array([0., 1., 2. ,3., 4.])
theta = [2., 4.]
print(predict_(x_values, theta))
