def add_intercept(x_values):
    res = []
    for x in x_values:
        res.append([1, x])
    return res
