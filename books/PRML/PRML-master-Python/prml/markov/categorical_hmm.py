import numpy as np
from .hmm import HiddenMarkovModel


class CategoricalHMM(HiddenMarkovModel):
    """
    Hidden Markov Model with categorical emission model
    """

    def __init__(self, initial_proba, transition_proba, means):
        """
        construct hidden markov model with categorical emission model

        Parameters
        ----------
        initial_proba : (n_hidden,) np.ndarray
            probability of initial latent state
        transition_proba : (n_hidden, n_hidden) np.ndarray
            transition probability matrix
            (i, j) component denotes the transition probability from i-th to j-th hidden state
        means : (n_hidden, ndim) np.ndarray
            mean parameters of categorical distribution

        Returns
        -------
        ndim : int
            number of observation categories
        n_hidden : int
            number of hidden states
        """
        assert initial_proba.size == transition_proba.shape[0] == transition_proba.shape[1] == means.shape[0]
        assert np.allclose(means.sum(axis=1), 1)
        super().__init__(initial_proba, transition_proba)
        self.ndim = means.shape[1]
        self.means = means

    def draw(self, n=100):
        """
        draw random sequence from this model

        Parameters
        ----------
        n : int
            length of the random sequence

        Returns
        -------
        seq : (n,) np.ndarray
            generated random sequence
        """
        hidden_state = np.random.choice(self.n_hidden, p=self.initial_proba)
        seq = []
        while len(seq) < n:
            seq.append(np.random.choice(self.ndim, p=self.means[hidden_state]))
            hidden_state = np.random.choice(self.n_hidden, p=self.transition_proba[hidden_state])
        return np.asarray(seq)

    def likelihood(self, X):
        return self.means[X]

    def maximize(self, seq, p_hidden, p_transition):
        self.initial_proba = p_hidden[0] / np.sum(p_hidden[0])
        self.transition_proba = np.sum(p_transition, axis=0) / np.sum(p_transition, axis=(0, 2))
        x = p_hidden[:, None, :] * (np.eye(self.ndim)[seq])[:, :, None]
        self.means = np.sum(x, axis=0) / np.sum(p_hidden, axis=0)
