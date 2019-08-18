import numpy as np
from support_vector_machine import *



x = np.array([
    [3,3],
    [4,3],
    [1,1],
    [7,7],
    [10,10],
])

y = np.array([
    [1],
    [1],
    [-1],
    [1],
    [1],
])


test = np.array([
    [1,2],
    [7,6],
    [-1,3],
])

a = supprot_vector_machine(x, y)
a.fit()
print a.prediction(x)
print a.prediction(test)