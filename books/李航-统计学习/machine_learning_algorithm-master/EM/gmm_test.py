from Gmm import *

sigma = np.matrix([[30, 0], [0, 30]])
MU = [[40, 20], [5, 35]]
a = EmGMM(sigma, 2, 1000, MU, 0.001)

a.train()