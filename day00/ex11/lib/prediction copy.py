def simple_predict(x_values, theta):
    y = []
    for x in x_values:
        y.append(theta[0] + (theta[1] * x))
    return y

# x_values = [1, 2, 3 ,4, 5]
# theta = [5, 3]
# print(simple_predict(x_values, theta))
