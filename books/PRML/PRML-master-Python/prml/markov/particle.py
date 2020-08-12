import numpy as np
from scipy.misc import logsumexp
from scipy.spatial.distance import cdist
from .state_space_model import StateSpaceModel


class Particle(StateSpaceModel):
    """
    A class to perform particle filtering, smoothing

    z_1 ~ p(z_1)\n
    z_n ~ p(z_n|z_n-1)\n
    x_n ~ p(x_n|z_n)

    Parameters
    ----------
    init_particle : (n_particle, ndim_hidden)
        initial hidden state
    sampler : callable (particles)
        function to sample particles at current step given previous state
    nll : callable (observation, particles)
        function to compute negative log likelihood for each particle

    Attribute
    ---------
    hidden_state : list of (n_paticle, ndim_hidden) np.ndarray
        list of particles
    """

    def __init__(self, init_particle, system, cov_system, nll, pdf=None):
        """
        construct state space model to perform particle filtering or smoothing

        Parameters
        ----------
        init_particle : (n_particle, ndim_hidden) np.ndarray
            initial hidden state
        system : (ndim_hidden, ndim_hidden) np.ndarray
            system matrix aka transition matrix
        cov_system : (ndim_hidden, ndim_hidden) np.ndarray
            covariance matrix of process noise
        nll : callable (observation, particles)
            function to compute negative log likelihood for each particle

        Attribute
        ---------
        particle : list of (n_paticle, ndim_hidden) np.ndarray
            list of particles at each step
        weight : list of (n_particle,) np.ndarray
            list of importance of each particle at each step
        n_particle : int
            number of particles at each step
        """
        self.particle = [init_particle]
        self.n_particle, self.ndim_hidden = init_particle.shape
        self.weight = [np.ones(self.n_particle) / self.n_particle]
        self.system = system
        self.cov_system = cov_system
        self.nll = nll
        self.smoothed_until = -1

    def resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=self.weight[-1])
        return self.particle[-1][index]

    def predict(self):
        predicted = self.resample() @ self.system.T
        predicted += np.random.multivariate_normal(np.zeros(self.ndim_hidden), self.cov_system, self.n_particle)
        self.particle.append(predicted)
        self.weight.append(np.ones(self.n_particle) / self.n_particle)
        return predicted, self.weight[-1]

    def weigh(self, observed):
        logit = -self.nll(observed, self.particle[-1])
        logit -= logsumexp(logit)
        self.weight[-1] = np.exp(logit)

    def filter(self, observed):
        self.weigh(observed)
        return self.particle[-1], self.weight[-1]

    def filtering(self, observed_sequence):
        mean = []
        cov = []
        for obs in observed_sequence:
            self.predict()
            p, w = self.filter(obs)
            mean.append(np.average(p, axis=0, weights=w))
            cov.append(np.cov(p, rowvar=False, aweights=w))
        return np.asarray(mean), np.asarray(cov)

    def transition_probability(self, particle, particle_prev):
        dist = cdist(
            particle,
            particle_prev @ self.system.T,
            "mahalanobis",
            VI=np.linalg.inv(self.cov_system))
        matrix = np.exp(-0.5 * np.square(dist))
        matrix /= np.sum(matrix, axis=1, keepdims=True)
        matrix[np.isnan(matrix)] = 1 / self.n_particle
        return matrix

    def smooth(self):
        particle_next = self.particle[self.smoothed_until]
        weight_next = self.weight[self.smoothed_until]

        self.smoothed_until -= 1
        particle = self.particle[self.smoothed_until]
        weight = self.weight[self.smoothed_until]
        matrix = self.transition_probability(particle_next, particle).T
        weight *= matrix @ weight_next / (weight @ matrix)
        weight /= np.sum(weight, keepdims=True)

    def smoothing(self, observed_sequence:np.ndarray=None):
        if observed_sequence is not None:
            self.filtering(observed_sequence)
        while self.smoothed_until != -len(self.particle):
            self.smooth()
        mean = []
        cov = []
        for p, w in zip(self.particle, self.weight):
            mean.append(np.average(p, axis=0, weights=w))
            cov.append(np.cov(p, rowvar=False, aweights=w))
        return np.asarray(mean), np.asarray(cov)
