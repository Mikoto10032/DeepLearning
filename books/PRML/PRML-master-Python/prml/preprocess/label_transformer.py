import numpy as np


class LabelTransformer(object):
    """
    Label encoder decoder

    Attributes
    ----------
    n_classes : int
        number of classes, K
    """

    def __init__(self, n_classes:int=None):
        self.n_classes = n_classes

    @property
    def n_classes(self):
        return self.__n_classes

    @n_classes.setter
    def n_classes(self, K):
        self.__n_classes = K
        self.__encoder = None if K is None else np.eye(K)

    @property
    def encoder(self):
        return self.__encoder

    def encode(self, class_indices:np.ndarray):
        """
        encode class index into one-of-k code

        Parameters
        ----------
        class_indices : (N,) np.ndarray
            non-negative class index
            elements must be integer in [0, n_classes)

        Returns
        -------
        (N, K) np.ndarray
            one-of-k encoding of input
        """
        if self.n_classes is None:
            self.n_classes = np.max(class_indices) + 1

        return self.encoder[class_indices]

    def decode(self, onehot:np.ndarray):
        """
        decode one-of-k code into class index

        Parameters
        ----------
        onehot : (N, K) np.ndarray
            one-of-k code

        Returns
        -------
        (N,) np.ndarray
            class index
        """

        return np.argmax(onehot, axis=1)
