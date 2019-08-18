import numpy as np
from knn_kdtree import *

dataSet = [[2, 3],
           [5, 4],
           [9, 6],
           [4, 7],
           [8, 1],
           [7, 2]]
x = [3, 4.5]
kdtree = Node(dataSet)
tree = kdtree.create(dataSet,0)
kdtree.preOrder(tree)
print kdtree.search(tree, x)