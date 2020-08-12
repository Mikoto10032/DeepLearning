"""
    @ jetou
    @ cart decision_tree
    @ date 2017 10 31

"""

import numpy as np

class naive_bayes:
    def __init__(self, feature, label):
        self.feature = feature.transpose()
        self.label = label.transpose().flatten(1)
        self.positive = np.count_nonzero(self.label == 1) * 1.0
        self.negative = np.count_nonzero(self.label == -1) * 1.0

    def train(self):
        positive_dict = {}
        negative_dict = {}
        for i in self.feature:
            unqiue = set(i)
            for j in unqiue:
                positive_dict[j] = np.count_nonzero(self.label[i==j]==1) / self.positive
                negative_dict[j] = np.count_nonzero(self.label[i==j]==-1) / self.negative

        return positive_dict, negative_dict

    def prediction(self,  pre_feature):

        positive_chance = self.positive / self.label.shape[0]
        negative_chance = self.negative / self.label.shape[0]
        positive_dict, negative_dict = self.train()
        for i in pre_feature:
            i = str(i)
            positive_chance *= positive_dict[i]
            negative_chance *= negative_dict[i]

        if positive_chance > negative_chance:
            return 1
        else:
            return -1

