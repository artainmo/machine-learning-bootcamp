import numpy as np
import pandas as pd
from my_logistic_regression import *
from other_metrics import *
import matplotlib.pyplot as mpl

x = np.array(pd.read_csv("../resources/solar_system_census.csv")[["height", "weight", "bone_density"]])
x = add_polynomial_features(x, 3)
y = np.array(pd.read_csv("../resources/solar_system_census_planets.csv")[["Origin"]])
data = data_spliter(x, y)

i = 0
cost = []
lambdas = []
while i <= 1:
    RLR = MyLogisticRegression(np.ones(((x.shape[1] + 1, 1))), lambda_=i, alpha=1e-13, n_cycle=1000000)
    RLR.fit_(minmax_normalization(data[0]), descriminate_classes(data[1], 0))
    cost.append(f1_score_(RLR.predict_(minmax_normalization(data[2])), descriminate_classes(data[3], 0)))
    lambdas.append(i)
    i += 0.1
print(cost)

mpl.plot(lambdas, cost)
mpl.show()
