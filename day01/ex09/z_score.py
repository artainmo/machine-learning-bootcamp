import statistics as stat
import numpy as np

def z_score(x_values):
    mean = np.array(stat.mean_(x_values))
    standard_deviation = np.array(stat.standard_deviation_(x_values))
    return (x_values - mean) / standard_deviation

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(z_score(X))
