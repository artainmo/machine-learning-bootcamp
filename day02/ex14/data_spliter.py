import pandas as pd
import numpy as np
from my_linear_regression import My_linear_regression as MLR
from polynomial_train import *
from norm import *

def split_x_y(set):
    set = set.transpose()
    set_x = set[:-1]
    set_y = set[-1:]
    return (set_x.transpose(), set_y.transpose())

def data_spliter(x, y, proportion):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return split_x_y(training_set) + split_x_y(test_set)



data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data["Sell_price"])
nX = z_score(X)
data = data_spliter(nX, Y, 0.8)
d = {
    "x_train" : data[0],
    "y_train" : data[1],
    "x_test" : data[2],
    "y_test" : data[3]
    }
LR = MLR(np.array([1.0, 1.0, 1.0, 1.0]))
print(LR.cost_function(LR.predict(d["x_train"]), d["y_train"]))
LR.fit(d["x_train"], d["y_train"], alpha = 1e-5, max_iter = 5000000)
print(LR.cost_function(LR.predict(d["x_train"]), d["y_train"]))

print(LR.cost_function(LR.predict(d["x_test"]), d["y_test"]))
