"""
    @ jetou
    @ logistic_regression binary
    @ date 2017 11 03

"""

import numpy as np
import random
import math

class LogisticRegression:
    def __init__(self, feature, label, step=0.4, max_iteration=5000):
        self.feature = np.array(feature).transpose()
        self.label = np.array(label).transpose()
        self.learning_step = step
        self.max_iteration = max_iteration

    def compute(self, x):
        wx = sum([self.w[j] * x[j] for j in xrange(self.w.shape[0])])
        exp_wx = math.exp(wx)

        p1 = exp_wx / (1 + exp_wx)
        p0 = 1 / (1 + exp_wx)

        if p1 > p0:
            return 1
        else:
            return 0

    def train(self):
        self.w = np.zeros((self.feature.shape[0],1))

        correct = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, self.label.shape[1] - 1)
            x = list(self.feature[:,index])
            x.append(1.0)
            y = self.label[:,index]

            if y == self.compute(x):
                correct += 1
                if correct > self.max_iteration:
                    break
                continue

            time+=1
            correct = 0

            wx = sum([self.w[j] * x[j] for j in xrange(self.w.shape[0])])
            exp_wx = math.exp(wx)

            for i in xrange(self.w.shape[0]):
                self.w[i] -= self.learning_step * (-y * x[i] + float(x[i] * exp_wx) / float(1 + exp_wx))

    def prediction(self,features):
        labels = []

        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.compute(x))

        return labels



