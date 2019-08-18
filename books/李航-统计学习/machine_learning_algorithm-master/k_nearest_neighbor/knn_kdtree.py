import numpy as np


import numpy as np

class Node:
    def __init__(self, data, lchild=None, rchild=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild

    def sort(self, dataSet, axis):
        sortDataSet = dataSet[:]
        m, n = np.shape(sortDataSet)
        for i in range(m):
            for j in range(0, m - i - 1):
                if (sortDataSet[j][axis] > sortDataSet[j + 1][axis]):
                    temp = sortDataSet[j]
                    sortDataSet[j] = sortDataSet[j + 1]
                    sortDataSet[j + 1] = temp
        return sortDataSet

    def create(self, dataSet, depth):
        if len(dataSet) > 0:
            m, n = np.shape(dataSet)
            midIndex = len(dataSet)/2
            axis = depth % n
            sortedDataSet = self.sort(dataSet, axis)
            node = Node(sortedDataSet[midIndex])

            leftDataSet = sortedDataSet[: midIndex]
            rightDataSet = sortedDataSet[midIndex + 1:]
            node.lchild = self.create(leftDataSet, depth + 1)
            node.rchild = self.create(rightDataSet, depth + 1)
            return node
        else:
            return None

    def preOrder(self, node):
        if node != None:
            self.preOrder(node.lchild)
            self.preOrder(node.rchild)

    def search(self, tree, x):
        self.nearestPoint = None
        self.nearestValue = 0

        def travel(node, depth=0):
            if node != None:
                n = len(x)
                axis = depth % n
                if x[axis] < node.data[axis]:
                    travel(node.lchild, depth + 1)
                else:
                    travel(node.rchild, depth + 1)


                distNodeAndX = self.dist(x, node.data)
                if (self.nearestPoint == None):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                elif (self.nearestValue > distNodeAndX):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX

                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):
                    if x[axis] < node.data[axis]:
                        travel(node.rchild, depth + 1)
                    else:
                        travel(node.lchild, depth + 1)

        travel(tree)
        return self.nearestPoint

    def dist(self, x1, x2):
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5
