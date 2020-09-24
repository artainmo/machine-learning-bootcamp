import numpy as np

def add_polynomial_features(x, power):
    power = range(1, power + 1)
    init = x
    for pow in power:
        x = np.column_stack((x, init ** pow))
    return x



x = np.arange(1,11).reshape(5, 2)
print(add_polynomial_features(x, 3))
