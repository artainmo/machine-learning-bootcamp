import pandas as pd
import numpy as np
from my_linear_regression import My_linear_regression as MLR

data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data["Sell_price"])
LR = MLR(np.array([1.0, 1.0, 1.0, 1.0]))
print(LR.cost_function(LR.predict(X), Y))
LR.fit(X, Y, alpha = 1e-6, max_iter = 60000000)
print(LR.cost_function(LR.predict(X), Y))
# LR.plot(X, Y, LR_age.linear_regression_cost_function(LR_age.predict(X), Y))
