from AdaBoost import *


feature = np.array([
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
])

label = np.array([
    [1],
    [1],
    [1],
    [-1],
    [-1],
    [-1],
    [1],
    [1],
    [1],
    [-1],
])

test = np.array([
    [2],
    [2],
    [6],
    [4],
])

a = adaBoost(feature, label)
a.train()
print a.prediction(test)