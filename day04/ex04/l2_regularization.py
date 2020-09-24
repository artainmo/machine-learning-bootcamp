import numpy as np

def l2_regularization(theta):
    theta[0] = 0
    return theta.dot(theta)

x = np.array([2, 14, -13, 5, 12, 4, -19])
y = np.array([3,0.5,-6])
print(l2_regularization(x))
