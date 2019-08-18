import numpy as np
from perceptron import *
import matplotlib.pyplot as plt

feature = np.array([
    [3,3],
    [4,3],
    [1,1],
])

label = np.array([
    [1],
    [1],
    [-1],
])

a = perceptron(feature, label)
a.train()

test_point = np.array([
    [+5,+5],
    [-1,-1]])
print a.prediction(test_point)

w,b = a.get_wandb()


#########plot##############################
def plot_function(samples, labels, w, b):
    index = 0
    for i in samples:
        if labels[index] == 1:
            s = 'x'
        else:
            s = 'o'
        plt.scatter(i[0], i[1], marker=s)
        index += 1
    xData = np.linspace(0, 5, 100)
    yData = (-b - w[0] * xData)/pow(w[1],2)
    plt.plot(xData, yData, color='r', label='sample data')

    plt.show()

plot_function(feature, label, w, b)
plot_function(test_point, a.prediction(test_point), w, b)