import pandas as pd
import numpy as np
from my_logistic_regression import *


def multiclass_classification(data, class_):
    solar = MyLogisticRegression(np.ones((data["x_train"].shape[1] + 1, 1)))
    solar.cost_(descriminate_classes(data["y_test"], class_), solar.predict_(minmax_normalization(data["x_test"])))
    solar.fit_(minmax_normalization(data["x_train"]), descriminate_classes(data["y_train"], class_))
    solar.cost_(descriminate_classes(data['y_test'], class_), solar.predict_(minmax_normalization(data["x_test"])))
    return solar.predict_(minmax_normalization(data["x_test"]))[0]

data1 = pd.read_csv("../resources/solar_system_census.csv")[["bone_density", "height", "weight"]]
data2 = pd.read_csv("../resources/solar_system_census_planets.csv")[["Origin"]]
data = data_spliter(data1, data2, 0.9)
zipcodes = [0,1,2,3]
probability = 0
highest = 0
for zipcode in zipcodes:
    probability = multiclass_classification(data, zipcode)
    if probability > highest:
        highest = probability
        code = zipcode
print(highest)
print(code)
