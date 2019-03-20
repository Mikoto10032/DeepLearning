import numpy as np
from prml.rv.multivariate_gaussian import MultivariateGaussian as Gaussian
from prml.markov.state_space_model import StateSpaceModel


class Kalman(StateSpaceModel):
    """
    A class to perform kalman filtering or smoothing
    z : internal state
    x : observation

    z_1 ~ N(z_1|mu_0, P_0)\n
    z_n ~ N(z_n|A z_n-1, P)\n
    x_n ~ N(x_n|C z_n, S)

    Parameters
    ----------
    system : (Dz, Dz) np.ndarray
        system matrix aka transition matrix (A)
    cov_system : (Dz, Dz) np.ndarray
        covariance matrix of process noise
    measure : (Dx, Dz) np.ndarray
        measurement matrix aka observation matrix (C)
    cov_measure : (Dx, Dx) np.ndarray
        covariance matrix of measurement noise
    mu0 : (Dz,) np.ndarray
        mean parameter of initial hidden variable
    P0 : (Dz, Dz) np.ndarray
        covariance parameter of initial hidden variable

    Attributes
    ----------
    Dz : int
        dimensionality of hidden variable
    Dx : int
        dimensionality of observed variable
    """


    def __init__(self, system, cov_system, measure, cov_measure, mu0, P0):
        """
        construct Kalman model

        z_1 ~ N(z_1|mu_0, P_0)\n
        z_n ~ N(z_n|A z_n-1, P)\n
        x_n ~ N(x_n|C z_n, S)

        Parameters
        ----------
        system : (Dz, Dz) np.ndarray
            system matrix aka transition matrix (A)
        cov_system : (Dz, Dz) np.ndarray
            covariance matrix of process noise
        measure : (Dx, Dz) np.ndarray
            measurement matrix aka observation matrix (C)
        cov_measure : (Dx, Dx) np.ndarray
            covariance matrix of measurement noise
        mu0 : (Dz,) np.ndarray
            mean parameter of initial hidden variable
        P0 : (Dz, Dz) np.ndarray
            covariance parameter of initial hidden variable

        Attributes
        ----------
        hidden_mean : list of (Dz,) np.ndarray
            list of mean of hidden state starting from the given hidden state
        hidden_cov : list of (Dz, Dz) np.ndarray
            list of covariance of hidden state starting from the given hidden state
        """
        self.system = system
        self.cov_system = cov_system
        self.measure = measure
        self.cov_measure = cov_measure

        self.hidden_mean = [mu0]
        self.hidden_cov = [P0]
        self.hidden_cov_predicted = [None]

        self.smoothed_until = -1
        self.smoothing_gain = [None]

    def predict(self):
        """
        predict hidden state at current step given estimate at previous step

        Returns
        -------
        tuple ((Dz,) np.ndarray, (Dz, Dz) np.ndarray)
            tuple of mean and covariance of the estimate at current step
        """
        mu_prev, cov_prev = self.hidden_mean[-1], self.hidden_cov[-1]
        mu = self.system @ mu_prev
        cov = self.system @ cov_prev @ self.system.T + self.cov_system
        self.hidden_mean.append(mu)
        self.hidden_cov.append(cov)
        self.hidden_cov_predicted.append(np.copy(cov))
        return mu, cov

    def filter(self, observed):
        """
        bayesian update of current estimate given current observation

        Parameters
        ----------
        observed : (Dx,) np.ndarray
            current observation

        Returns
        -------
        tuple ((Dz,) np.ndarray, (Dz, Dz) np.ndarray)
            tuple of mean and covariance of the updated estimate
        """
        mu, cov = self.hidden_mean[-1], self.hidden_cov[-1]
        innovation = observed - self.measure @ mu
        cov_innovation = self.cov_measure + self.measure @ cov @ self.measure.T
        kalman_gain = np.linalg.solve(cov_innovation, self.measure @ cov).T
        mu += kalman_gain @ innovation
        cov -= kalman_gain @ self.measure @ cov
        return mu, cov

    def filtering(self, observed_sequence):
        """
        perform kalman filtering given observed sequence

        Parameters
        ----------
        observed_sequence : (T, Dx) np.ndarray
            sequence of observations

        Returns
        -------
        tuple ((T, Dz) np.ndarray, (T, Dz, Dz) np.ndarray)
            seuquence of mean and covariance of hidden variable at each time step
        """
        for obs in observed_sequence:
            self.predict()
            self.filter(obs)
        mean_sequence = np.asarray(self.hidden_mean[1:])
        cov_sequence = np.asarray(self.hidden_cov[1:])
        return mean_sequence, cov_sequence

    def smooth(self):
        """
        bayesian update of current estimate with future observations
        """
        mean_smoothed_next = self.hidden_mean[self.smoothed_until]
        cov_smoothed_next = self.hidden_cov[self.smoothed_until]
        cov_pred_next = self.hidden_cov_predicted[self.smoothed_until]

        self.smoothed_until -= 1
        mean = self.hidden_mean[self.smoothed_until]
        cov = self.hidden_cov[self.smoothed_until]
        gain = np.linalg.solve(cov_pred_next, self.system @ cov).T
        mean += gain @ (mean_smoothed_next - self.system @ mean)
        cov += gain @ (cov_smoothed_next - cov_pred_next) @ gain.T
        self.smoothing_gain.insert(0, gain)

    def smoothing(self, observed_sequence:np.ndarray=None):
        """
        perform Kalman smoothing (given observed sequence)

        Parameters
        ----------
        observed_sequence : (T, Dx) np.ndarray, optional
            sequence of observation
            run Kalman filter if given (the default is None)

        Returns
        -------
        tuple ((T, Dz) np.ndarray, (T, Dz, Dz) np.ndarray)
            sequence of mean and covariance of hidden variable at each time step
        """
        if observed_sequence is not None:
            self.filtering(observed_sequence)
        while self.smoothed_until != -len(self.hidden_mean):
            self.smooth()
        mean_sequence = np.asarray(self.hidden_mean[1:])
        cov_sequence = np.asarray(self.hidden_cov[1:])
        return mean_sequence, cov_sequence

    def update_parameter(self, observation_sequence):
        """
        maximization step of EM algorithm
        """
        mu0 = self.hidden_mean[1]
        P0 = self.hidden_cov[1]

        Ezn = np.asarray(self.hidden_mean)
        Eznzn = np.asarray(self.hidden_cov) + Ezn[..., None] * Ezn[:, None, :]
        Eznzn_1 = np.einsum("nij,nkj->nik", self.hidden_cov[2:], self.smoothing_gain[1:-1]) + Ezn[2:, :, None] * Ezn[1:-1, None, :]
        self.system = np.linalg.solve(np.sum(Eznzn[2:], axis=0), np.sum(Eznzn_1, axis=0).T).T
        self.cov_system = np.mean(
            Eznzn[2:]
            - np.einsum("ij,nkj->nik", self.system, Eznzn_1)
            - np.einsum("nij,kj->nik", Eznzn_1, self.system)
            + np.einsum("ij,njk,lk->nil", self.system, Eznzn[1:-1], self.system),
            axis=0
        )
        self.measure = np.linalg.solve(
            np.sum(Eznzn[1:], axis=0),
            np.sum(np.einsum("ni,nj->nij", Ezn[1:], observation_sequence), axis=0)
        ).T
        self.cov_measure = np.mean(
            np.einsum("ni,nj->nij", observation_sequence, observation_sequence)
            - np.einsum("ij,nj,nk->nik", self.measure, Ezn[1:], observation_sequence)
            - np.einsum("ni,nj,kj->nik", observation_sequence, Ezn[1:], self.measure)
            + np.einsum("ij,njk,lk->nil", self.measure, Eznzn[1:], self.measure),
            axis=0
        )
        return self.system, self.cov_system, self.measure, self.cov_measure, mu0, P0

    def fit(self, sequence, max_iter=10):
        for _ in range(max_iter):
            kalman_smoother(self, sequence)
            param = self.update_parameter(sequence)
            self.__init__(*param)
        return kalman_smoother(self, sequence)


def kalman_filter(kalman:Kalman, observed_sequence:np.ndarray)->tuple:
    """
    perform kalman filtering given Kalman model and observed sequence

    Parameters
    ----------
    kalman : Kalman
        Kalman model
    observed_sequence : (T, Dx) np.ndarray
        sequence of observations

    Returns
    -------
    tuple ((T, Dz) np.ndarray, (T, Dz, Dz) np.ndarray)
        seuquence of mean and covariance of hidden variable at each time step
    """
    for obs in observed_sequence:
        kalman.predict()
        kalman.filter(obs)
    mean_sequence = np.asarray(kalman.hidden_mean[1:])
    cov_sequence = np.asarray(kalman.hidden_cov[1:])
    return mean_sequence, cov_sequence


def kalman_smoother(kalman:Kalman, observed_sequence:np.ndarray=None):
    """
    perform Kalman smoothing given Kalman model (and observed sequence)

    Parameters
    ----------
    kalman : Kalman
        Kalman model
    observed_sequence : (T, Dx) np.ndarray, optional
        sequence of observation
        run Kalman filter if given (the default is None)

    Returns
    -------
    tuple ((T, Dz) np.ndarray, (T, Dz, Dz) np.ndarray)
        seuqnce of mean and covariance of hidden variable at each time step
    """

    if observed_sequence is not None:
        kalman_filter(kalman, observed_sequence)
    while kalman.smoothed_until != -len(kalman.hidden_mean):
        kalman.smooth()
    mean_sequence = np.asarray(kalman.hidden_mean[1:])
    cov_sequence = np.asarray(kalman.hidden_cov[1:])
    return mean_sequence, cov_sequence
