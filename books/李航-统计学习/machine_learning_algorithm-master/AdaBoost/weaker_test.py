from weaker_classifier import *

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
])

d=np.array([
    [0.007143],
    [0.07143],
    [0.07143],
    [0.07143],
    [0.07143],
    [0.07143],
    [0.16667],
    [0.16667],
    [0.16667],
    [0.07143],
])
pp = []
a = weake_classifier(feature, label)
b = weake_classifier(feature, label,d)
a.train()
print a.__str__()
print a.prediction(test)
b.train()
print b.__str__()
