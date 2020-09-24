import numpy as np

def split_x_y(set):
    set = set.transpose()
    set_x = set[:-1]
    set_y = set[-1:]
    return (set_x, set_y)

def data_spliter(x, y, proportion):
    shuffle = np.column_stack((x, y))
    print(shuffle)
    np.random.shuffle(shuffle)
    print(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return split_x_y(training_set) + split_x_y(test_set)




x1 = np.array([1, 42, 300, 10, 59])
y = np.array([0,1,0,1,0])
print(data_spliter(x1, y, 0.8))
