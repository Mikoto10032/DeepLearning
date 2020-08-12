"""
    @ jetou
    @ cart decision_tree
    @ date 2017 10 29

"""
import numpy as np

class tree:
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def Gini(self, dataset):

        data_set = set(dataset)
        sum = 0.0
        data_length = dataset.shape[0] * 1.0
        for i in data_set:
            sum += (np.count_nonzero(dataset == i) / data_length) ** 2

        return 1.0 - sum

    def cmpgini(self, feature, label):
        label = np.array(label).flatten(1)
        unique = set(feature)
        min = 1
        count_feature = np.shape(feature)[0]
        for i in unique:
            c1 = np.count_nonzero(feature == i) * self.Gini(label[feature == i]) / count_feature
            c2 = np.count_nonzero(feature != i) * self.Gini(label[feature != i]) / count_feature
            if min > (c1 + c2):
                min = c1 + c2
                result = i
        return result, min

    def maketree(self, feature, label):
        label = label.flatten(1)
        train_feature = feature.transpose()

        min = 1.0
        opt_feature = 0
        opt_feature_val = 0

        if np.unique(label).size == 1:
            return label[0]

        for i in range(len(train_feature)):
            result, p = self.cmpgini(train_feature[i], label)
            if p < min:
                min = p
                opt_feature = i
                opt_feature_val = result

        if min == 1.0:
            return label

        left = []
        right = []

        left = self.maketree(train_feature.transpose()[train_feature.transpose()[:, opt_feature] != opt_feature_val],
                        label[train_feature.transpose()[:, opt_feature] != opt_feature_val])
        #
        right = self.maketree(train_feature.transpose()[train_feature.transpose()[:, opt_feature] == opt_feature_val],
                         label[train_feature.transpose()[:, opt_feature] == opt_feature_val])


        return [(opt_feature, opt_feature_val), left, right]

    def train(self):
        self.train_result = self.maketree(self.feature, self.label)

    def prediction(self, Mat):
        Mat = np.array(Mat).transpose()
        result = np.zeros((Mat.shape[0], 1))
        for i in range(Mat.shape[0]):
            tree = self.train_result
            while self.isLeaf(tree) == False:
                feature, val = tree[0]
                if Mat[i][feature] == val:

                    tree = self.getRight(tree)
                else:
                    tree = self.getLeft(tree)

            result[i] = tree

        return result

    def isLeaf(self, tree):
        if isinstance(tree, list):
            return False
        else:
            return True

    def getLeft(self, tree):
        assert isinstance(tree, list)
        return tree[1]

    def getRight(self, tree):
        assert isinstance(tree, list)
        return tree[2]




