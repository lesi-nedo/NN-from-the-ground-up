import numpy as np
from .activation_function import ActivationFunction



def x_less_than_zero(x: np.ndarray) -> np.ndarray:
    """
        Numerically stable version of the Sigmoid function for x < 0.

        Parameters:
        - x (np.ndarray): Input array.

        Returns:
        - np.ndarray: Output array after applying the Sigmoid function for x < 0.
    """
    z = np.exp(x)
    return z / (1 + z)


class Sigmoid(ActivationFunction):
    """
        A class to represent the Sigmoid function and its derivative with numerical stability.
    """

    def __init__(self):
        self.evaluated_value = None

    def x_greater_than_zero(self, x: np.ndarray) -> np.ndarray:
        """
            Numerically stable version of the Sigmoid function for x >= 0.

            Parameters:
            - x (np.ndarray): Input array.

            Returns:
            - np.ndarray: Output array after applying the Sigmoid function for x >= 0.
        """
        z = np.exp(-x)
        self.evaluated_value = 1 / (1 + z)
        return self.evaluated_value

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
            Numerically stable version of the Sigmoid function.

            Parameters:
            - x (np.ndarray): Input array.

            Returns:
            - np.ndarray: Output array after applying the Sigmoid function.
        """
        return np.piecewise(x, [x >= 0, x < 0], [self.x_greater_than_zero, x_less_than_zero])

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
            Evaluate the derivative of the Sigmoid function at the given input.

            Parameters:
            - x(np.ndarray): Input array.

            Returns:
            - np.ndarray: Output array after applying the derivative of the Sigmoid function.
        """
        sigmoid_x = self.evaluate(x)
        return sigmoid_x * (1 - sigmoid_x)
