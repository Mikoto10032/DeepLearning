import numpy as np
from logistic_regression import *



feature = np.array([
    [3,3],
    [4,3],
    [1,1],
])

label = np.array([
    [1],
    [1],
    [0],
])

test_point = np.array([
    [+5,+5],
    [-1,-1],
])

a = LogisticRegression(feature, label)
a.train()
f = a.prediction(test_point)
print  f