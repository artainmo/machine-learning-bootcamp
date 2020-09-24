def add_intercept(x_values):
    if isinstance(x_values, list) == False or isinstance(x_values[0], list) == True:
        print("ERROR input add intercept function")
        return
    res = []
    for x in x_values:
        res.append([1, x])
    return res

def add_intercept(x_values):
    res = []
    for x in x_values:
        res.append([1, x])
    return np.array(res)
