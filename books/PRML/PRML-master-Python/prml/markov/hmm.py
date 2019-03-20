import numpy as np


class HiddenMarkovModel(object):
    """
    Base class of Hidden Markov models
    """

    def __init__(self, initial_proba, transition_proba):
        """
        construct hidden markov model

        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray
            initial probability of each hidden state
        transition_proba : (n_hidden, n_hidden) np.ndarray
            transition probability matrix
            (i, j) component denotes the transition probability from i-th to j-th hidden state

        Attribute
        ---------
        n_hidden : int
            number of hidden state
        """
        self.n_hidden = initial_proba.size
        self.initial_proba = initial_proba
        self.transition_proba = transition_proba

    def fit(self, seq, iter_max=100):
        """
        perform EM algorithm to estimate parameter of emission model and hidden variables

        Parameters
        ----------
        seq : (N, ndim) np.ndarray
            observed sequence
        iter_max : int
            maximum number of EM steps

        Returns
        -------
        posterior : (N, n_hidden) np.ndarray
            posterior distribution of each latent variable
        """
        params = np.hstack(
            (self.initial_proba.ravel(), self.transition_proba.ravel()))
        for i in range(iter_max):
            p_hidden, p_transition = self.expect(seq)
            self.maximize(seq, p_hidden, p_transition)
            params_new = np.hstack(
                (self.initial_proba.ravel(), self.transition_proba.ravel()))
            if np.allclose(params, params_new):
                break
            else:
                params = params_new
        return self.forward_backward(seq)

    def expect(self, seq):
        """
        estimate posterior distributions of hidden states and
        transition probability between adjacent latent variables

        Parameters
        ----------
        seq : (N, ndim) np.ndarray
            observed sequence

        Returns
        -------
        p_hidden : (N, n_hidden) np.ndarray
            posterior distribution of each hidden variable
        p_transition : (N - 1, n_hidden, n_hidden) np.ndarray
            posterior transition probability between adjacent latent variables
        """
        likelihood = self.likelihood(seq)

        f = self.initial_proba * likelihood[0]
        constant = [f.sum()]
        forward = [f / f.sum()]
        for like in likelihood[1:]:
            f = forward[-1] @ self.transition_proba * like
            constant.append(f.sum())
            forward.append(f / f.sum())
        forward = np.asarray(forward)
        constant = np.asarray(constant)

        backward = [np.ones(self.n_hidden)]
        for like, c in zip(likelihood[-1:0:-1], constant[-1:0:-1]):
            backward.insert(0, self.transition_proba @ (like * backward[0]) / c)
        backward = np.asarray(backward)

        p_hidden = forward * backward
        p_transition = self.transition_proba * likelihood[1:, None, :] * backward[1:, None, :] * forward[:-1, :, None]
        return p_hidden, p_transition

    def forward_backward(self, seq):
        """
        estimate posterior distributions of hidden states

        Parameters
        ----------
        seq : (N, ndim) np.ndarray
            observed sequence

        Returns
        -------
        posterior : (N, n_hidden) np.ndarray
            posterior distribution of hidden states
        """
        likelihood = self.likelihood(seq)

        f = self.initial_proba * likelihood[0]
        constant = [f.sum()]
        forward = [f / f.sum()]
        for like in likelihood[1:]:
            f = forward[-1] @ self.transition_proba * like
            constant.append(f.sum())
            forward.append(f / f.sum())

        backward = [np.ones(self.n_hidden)]
        for like, c in zip(likelihood[-1:0:-1], constant[-1:0:-1]):
            backward.insert(0, self.transition_proba @ (like * backward[0]) / c)

        forward = np.asarray(forward)
        backward = np.asarray(backward)
        posterior = forward * backward
        return posterior

    def filtering(self, seq):
        """
        bayesian filtering

        Parameters
        ----------
        seq : (N, ndim) np.ndarray
            observed sequence

        Returns
        -------
        posterior : (N, n_hidden) np.ndarray
            posterior distributions of each latent variables
        """
        likelihood = self.likelihood(seq)
        p = self.initial_proba * likelihood[0]
        posterior = [p / np.sum(p)]
        for like in likelihood[1:]:
            p = posterior[-1] @ self.transition_proba * like
            posterior.append(p / np.sum(p))
        posterior = np.asarray(posterior)
        return posterior

    def viterbi(self, seq):
        """
        viterbi algorithm (a.k.a. max-sum algorithm)

        Parameters
        ----------
        seq : (N, ndim) np.ndarray
            observed sequence

        Returns
        -------
        seq_hid : (N,) np.ndarray
            the most probable sequence of hidden variables
        """
        nll = -np.log(self.likelihood(seq))
        cost_total = nll[0]
        from_list = []
        for i in range(1, len(seq)):
            cost_temp = cost_total[:, None] - np.log(self.transition_proba) + nll[i]
            cost_total = np.min(cost_temp, axis=0)
            index = np.argmin(cost_temp, axis=0)
            from_list.append(index)
        seq_hid = [np.argmin(cost_total)]
        for source in from_list[::-1]:
            seq_hid.insert(0, source[seq_hid[0]])
        return seq_hid
