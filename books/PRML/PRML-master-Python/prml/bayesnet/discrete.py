import numpy as np
from prml.bayesnet.probability_function import ProbabilityFunction
from prml.bayesnet.random_variable import RandomVariable


class DiscreteVariable(RandomVariable):
    """
    Discrete random variable
    """

    def __init__(self, n_class:int):
        """
        intialize a discrete random variable

        parameters
        ----------
        n_class : int
            number of classes

        Attributes
        ----------
        parent : DiscreteProbability, optional
            parent node this variable came out from
        message_from : dict
            dictionary of message from neighbor node and itself
        child : list of DiscreteProbability
            probability function this variable is conditioning
        proba : np.ndarray
            current estimate
        """
        self.n_class = n_class
        self.parent = []
        self.message_from = {self: np.ones(n_class)}
        self.child = []
        self.is_observed = False

    def __repr__(self):
        string = f"DiscreteVariable("
        if self.is_observed:
            string += f"observed={self.proba})"
        else:
            string += f"proba={self.proba})"
        return string

    def add_parent(self, parent):
        self.parent.append(parent)

    def add_child(self, child):
        self.child.append(child)
        self.message_from[child] = np.ones(self.n_class)

    @property
    def proba(self):
        return self.posterior

    def receive_message(self, message, giver, proprange):
        self.message_from[giver] = message
        self.summarize_message()
        self.send_message(proprange, exclude=giver)

    def summarize_message(self):
        if self.is_observed:
            self.prior = self.message_from[self]
            self.likelihood = self.prior
            self.posterior = self.prior
            return

        self.prior = np.ones(self.n_class)
        for func in self.parent:
            self.prior *= self.message_from[func]
        self.prior /= np.sum(self.prior, keepdims=True)

        self.likelihood = np.copy(self.message_from[self])
        for func in self.child:
            self.likelihood *= self.message_from[func]

        self.posterior = self.prior * self.likelihood
        self.posterior /= self.posterior.sum()

    def send_message(self, proprange=-1, exclude=None):
        for func in self.parent:
            if func is not exclude:
                func.receive_message(self.likelihood, self, proprange)
        for func in self.child:
            if func is not exclude:
                func.receive_message(self.prior, self, proprange)

    def observe(self, data:int, proprange=-1):
        """
        set observed data of this variable

        Parameters
        ----------
        data : int
            observed data of this variable
            This must be smaller than n_class and must be non-negative
        propagate : int, optional
            Range to propagate the observation effect to the other random variable using belief propagation alg.
            If proprange=1, the effect only propagate to the neighboring random variables.
            Default is -1, which is infinite range.
        """
        assert(0 <= data < self.n_class)
        self.is_observed = True
        self.receive_message(np.eye(self.n_class)[data], self, proprange=proprange)


class DiscreteProbability(ProbabilityFunction):
    """
    Discrete probability function
    """

    def __init__(self, table, *condition, out=None, name=None):
        """
        initialize discrete probability function

        Parameters
        ----------
        table : (K, ...) np.ndarray or array-like
            probability table
            If a discrete variable A is conditioned with B and C,
            table[a,b,c] give probability of A=a when B=b and C=c.
            Thus, the sum along the first axis should equal to 1.
            If a table is 1 dimensional, the variable is not conditioned.
        condition : tuple of DiscreteVariable, optional
            parent node, discrete variable this function is conidtioned by
            len(condition) should equal to (table.ndim - 1)
            (Default is (), which means no condition)
        out : DiscreteVariable or list of DiscreteVariable, optional
            output of this discrete probability function
            Default is None which construct a new output instance
        name : str
            name of this discrete probability function
        """
        self.table = np.asarray(table)
        self.condition = condition
        if condition:
            for var in condition:
                var.add_child(self)
        self.message_from = {var: var.prior for var in condition}

        if out is None:
            self.out = [DiscreteVariable(len(table))]
        elif isinstance(out, DiscreteVariable):
            self.out = [out]
        else:
            self.out = out

        for i, random_variable in enumerate(self.out):
            random_variable.add_parent(self)
            self.message_from[random_variable] = np.ones(np.size(self.table, i))

        for random_variable in self.out:
            self.send_message_to(random_variable, proprange=0)

        self.name = name

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return super().__repr__()

    def receive_message(self, message, giver, proprange):
        self.message_from[giver] = message
        if proprange:
            self.send_message(proprange, exclude=giver)

    @staticmethod
    def expand_dims(x, ndim, axis):
        shape = [-1 if i == axis else 1 for i in range(ndim)]
        return x.reshape(*shape)

    def compute_message_to(self, destination):
        proba = np.copy(self.table)
        for i, random_variable in enumerate(self.out):
            if random_variable is destination:
                index = i
                continue
            message = self.message_from[random_variable]
            proba *= self.expand_dims(message, proba.ndim, i)
        for i, random_variable in enumerate(self.condition, len(self.out)):
            if random_variable is destination:
                index = i
                continue
            message = self.message_from[random_variable]
            proba *= self.expand_dims(message, proba.ndim, i)
        axis = list(range(proba.ndim))
        axis.remove(index)
        message = np.sum(proba, axis=tuple(axis))
        message /= np.sum(message, keepdims=True)
        return message

    def send_message_to(self, destination, proprange=-1):
        message = self.compute_message_to(destination)
        destination.receive_message(message, self, proprange)

    def send_message(self, proprange, exclude=None):
        proprange = proprange - 1

        for random_variable in self.out:
            if random_variable is not exclude:
                self.send_message_to(random_variable, proprange)

        if proprange == 0: return

        for random_variable in self.condition:
            if random_variable is not exclude:
                self.send_message_to(random_variable, proprange - 1)


def discrete(table, *condition, out=None, name=None):
    """
    discrete probability function

    Parameters
    ----------
    table : (K, ...) np.ndarray or array-like
        probability table
        If a discrete variable A is conditioned with B and C,
        table[a,b,c] give probability of A=a when B=b and C=c.
        Thus, the sum along the first axis should equal to 1.
        If a table is 1 dimensional, the variable is not conditioned.
    condition : tuple of DiscreteVariable, optional
        parent node, discrete variable this function is conidtioned by
        len(condition) should equal to (table.ndim - 1)
        (Default is (), which means no condition)
    out : DiscreteVariable, optional
        output of this discrete probability function
        Default is None which construct a new output instance
    name : str
        name of the discrete probability function

    Returns
    -------
    DiscreteVariable
        output discrete random variable of discrete probability function
    """
    function = DiscreteProbability(table, *condition, out=out, name=name)
    if len(function.out) == 1:
        return function.out[0]
    else:
        return function.out
