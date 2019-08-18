# -*- coding: utf-8 -*-

"""
    @ jetou
    @ Gaussian misture model
    @ date 2017 11 27

"""
# Reference http://blog.csdn.net/jinping_shi/article/details/59613054


import numpy as np
import math
import copy

class EmGMM:
    def __init__(self, sigma, k, N, MU, epsilon):
        """
        k is the number of Gaussian distribution
        N is the number of feature
        sigma is variance
        """
        self.k = k
        self.N = N
        self.epsilon = epsilon
        self.sigma = sigma
        self.MU = np.matrix(MU)
        self.alpha = [0.5, 0.5]

    def init_data(self):
        self.X = np.matrix(np.zeros((self.N, 2)))
        self.Mu = np.random.random(self.k)
        self.Expectations = np.zeros((self.N, self.k))
        for i in xrange(self.N):
            if np.random.random(1) > 0.5:
                self.X[i,:] = np.random.multivariate_normal(self.MU.tolist()[0], self.sigma, 1)
            else:
                self.X[i,:] = np.random.multivariate_normal(self.MU.tolist()[1], self.sigma, 1)

    def e_step(self):
        for i in range(self.N):
            Denom = 0
            Numer = [0.0] * self.k
            for j in range (self.k):
                Numer[j] = self.alpha[j] * math.exp(-(self.X[i,:] - self.MU[j,:]) * self.sigma.I * np.transpose(self.X[i,:] - self.MU[j,:])) \
                           / np.sqrt(np.linalg.det(self.sigma))
                Denom += Numer[j]
            for j in range(0, self.k):
                self.Expectations[i, j] = Numer[j] / Denom

    def m_step(self):
        for j in xrange(0, self.k):
            Numer = 0
            Denom = 0
            sabi = 0
            for i in xrange(0, self.N):
                Numer += self.Expectations[i, j] * self.X[i, :]
                Denom += self.Expectations[i, j]
            self.MU[j, :] = Numer / Denom
            self.alpha[j] = Denom / self.N
            for i in xrange(0, self.N):
                sabi += self.Expectations[i, j] * np.square((self.X[i, :] - self.MU[j, :]))
            self.sigma[j, :]=  sabi / Denom

    def train(self, inter=1000):
        self.init_data()
        for i in range(inter):
            error = 0
            err_alpha = 0
            err_sigma = 0
            old_mu = copy.deepcopy(self.MU)
            old_alpha = copy.deepcopy(self.alpha)
            old_sigma = copy.deepcopy(self.sigma)
            self.e_step()
            self.m_step()
            print "The number of iterations", i
            print "Location parameters: mu", self.MU
            print "variance: sigma", self.sigma
            print "Selected probability: alpha", self.alpha
            for j in range(self.k):
                error += (abs(old_mu[j, 0] - self.MU[j, 0]) + abs(old_mu[j, 1] - self.MU[j, 1]))
                err_sigma += (abs(old_sigma[j, 0] - self.sigma[j, 0]) + abs(old_sigma[j, 1] - self.sigma[j, 1]))
                err_alpha += abs(old_alpha[j] - self.alpha[j])
            if (error <= self.epsilon) and (err_sigma <= self.epsilon) and (err_alpha <= self.epsilon):
                print error, err_alpha, err_sigma
                break

