import numpy as np
from .activation_function import ActivationFunction


class LeakyReLU(ActivationFunction):
    """
    Leaky ReLU activation function

    """

    def __init__(self, alpha=0.01):
        """
        Initialize the Leaky ReLU function with a threshold value.

        Parameters:
        - alpha (float): Threshold value for the Leaky ReLU function.

        Returns:
        - None
        """
        self.alpha = alpha

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Leaky ReLU function at the given input.

        Parameters:
        - x (np.ndarray): Input array.

        Returns:
        - np.ndarray: Output array after applying the Leaky ReLU function.
        """

        return np.piecewise(x, [x <= 0, x > 0], [lambda x: self.alpha * x, lambda x: x])

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the derivative of the Leaky ReLU function at the given input.

        Parameters:
        - x (np.ndarray): Input array.

        Returns:
        - np.ndarray: Output array after applying the derivative of the Leaky ReLU function.
        """

        return np.piecewise(x, [x <= 0, x > 0], [self.alpha, 1])
