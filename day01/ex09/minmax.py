import numpy as np


def minmax(x_values):
    new = []
    max = np.max(x_values)
    min = np.min(x_values)
    range = max - min
    for x in x_values:
        new.append((x - min) / range)
    return new

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(minmax(X))
