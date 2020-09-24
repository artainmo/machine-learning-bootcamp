def mean(list):
    res = 0
    for values in list:
        res += float(values)
    return res / len(values)

def median(list):
    i = 0
    lenght = len(list)
    if lenght % 2 != 0:
        lenght /= 2
        while i < lenght:
            i += 1
        return list[i]
    else:
        lenght /= 2
        while i < lenght:
            i += 1
        return mean([list[i - 1], list[i]])

def quartiles(list):
    i = 0
    length = len(list)
    if lenght % 2 == 0:
        lenght /= 2
        while i < lenght:
            i += 1
        return (median(list[0:i]), median(list[i + 1:lenght]))
    else:
        lenght /= 2
        while i < lenght:
            i += 1
        return (median(list[0:i - 1]), median(list[i + 1:lenght]))

def variance(list):
    res = 0
    mean = mean(list)
    for value in list:
        res += ((value - mean)**2)
    res /= len(list)
    return res

def standard_deviation(list):
    return variance(list)**0.5 
