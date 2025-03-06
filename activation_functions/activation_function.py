import abc


class ActivationFunction(abc.ABC):
    """
    Abstract class for activation functions

    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def evaluate(self, x):
        pass

    @abc.abstractmethod
    def derivative(self, x):
        pass
