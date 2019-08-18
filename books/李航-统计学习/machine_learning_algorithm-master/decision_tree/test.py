import numpy
from cart_Classification_tree import *
train_feature = np.array([
['teenager',        'no',   'no',   0.0],
['teenager',        'no',   'no',   1.0],
['teenager',        'yes',  'no',   1.0],
['teenager',        'yes',  'yes',  0.0],
['teenager',        'no',   'no',   0.0],
['senior citizen',  'no',   'no',   0.0],
['senior citizen',  'no',   'no',   1.0],
['senior citizen',  'yes',  'yes',  1.0],
['senior citizen',  'no',   'yes',  2.0],
['senior citizen',  'no',   'yes',  2.0],
['old pepple',      'no',   'yes',  2.0],
['old pepple',      'no',   'yes',  1.0],
['old pepple',      'yes',  'no',   1.0],
['old pepple',      'yes',  'no',   2.0],
['old pepple',      'no',   'no',   0.0],
])

Tag = np.array([
[-1],
[-1],
[+1],
[+1],
[-1],
[-1],
[-1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[+1],
[-1],
]).transpose()

a = tree(train_feature, Tag)
a.train()
s = a.prediction(np.array([
    ['teenager',        'no',   'no',   0],
    ['old pepple',      'yes',  'no',   2],
    ]).transpose())
print s