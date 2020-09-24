import statistics as stat
import numpy as np

def column_z_score(x_values):
    mean = np.array(stat.mean_(x_values))
    standard_deviation = np.array(stat.standard_deviation_(x_values))
    return (x_values - mean) / standard_deviation

def z_score(x_values):
    x_values = x_values.transpose()
    new = []
    for column in x_values:
        new.append(column_z_score(column))
    return np.array(new).transpose()
