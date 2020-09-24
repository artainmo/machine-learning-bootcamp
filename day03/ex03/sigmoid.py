import numpy as np
import math

def sigmoid_(x):
    x = x.astype(np.float)
    return (1 / (1 + (np.exp(x * -1))))

x = np.array([[-4.322, 2.5555, 0.34],[-3.7, 4.7, np.nan]], dtype=object)
print(sigmoid_(x))
