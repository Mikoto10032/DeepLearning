import numpy as np
from naive_bayes import *

feature = np.array([
    [1, 'S'],
    [1, 'M'],
    [1, 'M'],
    [1, 'S'],
    [1, 'S'],
    [2, 'S'],
    [2, 'M'],
    [2, 'M'],
    [2, 'L'],
    [2, 'L'],
    [3, 'L'],
    [3, 'M'],
    [3, 'M'],
    [3, 'L'],
    [3, 'L'],
])

label = np.array([
    [-1],
    [-1],
    [1],
    [1],
    [-1],
    [-1],
    [-1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [-1],
])

a = naive_bayes(feature, label)
print a.prediction([2,'S'])

