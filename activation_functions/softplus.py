import numpy as np
from .sigmoid import Sigmoid
from .activation_function import ActivationFunction


class Softplus(ActivationFunction):
    def __init__(self, threshold: float = 100):
        """
            Initialize the Softplus function with a threshold value.


            Parameters:
            - threshold (float): Threshold value for the Softplus function.

            Returns:
            - None
        """
        self.threshold = threshold

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
            Evaluate the Softplus function at the given input.


            Parameters:
            - x (np.ndarray): Input array.

            Returns:
            - np.ndarray: Output array after applying the Softplus function.
        """

        # result = np.piecewise(x, [x <= self.threshold, x > self.threshold],
        #                       [
        #                           lambda x_p: np.log1p(np.exp(
        #                               -np.clip(np.abs(x_p), np.finfo(float).eps, np.abs(x_p), dtype=np.float64))
        #                           ) + np.maximum(x_p, 0),
        #                           lambda x_p: x_p])
        #

        result = np.log1p(np.exp(x))

        return result

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
            Evaluate the derivative of the Softplus function at the given input.

            Parameters:
            - x (np.ndarray): Input array.

            Returns:
            - np.ndarray: Output array after applying the derivative of the Softplus function.
        """
        return Sigmoid().evaluate(x)
