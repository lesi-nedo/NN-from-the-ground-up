import numpy as np
from typing import Optional

from activation_functions import Sigmoid


class BCELogitsLoss:

    def __init__(self):
        self.y_prob = None

    def forward(self, y_true: np.ndarray, input: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
            It merges the activation function and the loss function in one class. It returns the average binary
            cross-entropy loss of the sigmoid activation values.

            Parameters:
            - y_true (np.ndarray): True labels, e.g.: [0, 1, 0, 1, 0] where 1 is True, 0 False.
            - input (np.ndarray): Values that are coming from the previous layer.
            - weights (Optional[np.ndarray]): Weights that influence the individual loss elements.

            Returns:
            - float: The average binary cross-entropy loss.
        """
        # Ensure the inputs are NumPy arrays
        y_true = np.array(y_true, dtype=np.float64)
        input = np.array(input, dtype=np.float64)

        y_pred = Sigmoid().evaluate(input)

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("The true labels and predicted probabilities have different size.")

        eps = np.finfo(float).eps

        self.y_prob = np.clip(y_pred, eps, 1 - eps)

        loss = np.add(np.multiply(y_true, np.log(self.y_prob)),
                      np.multiply(1.0 - y_true, np.log(1.0 - self.y_prob))) * -1

        if weights is not None:
            weights = np.array(weights, dtype=np.float64)
            loss = np.multiply(loss, weights)

        return float(np.mean(loss))

    def backward(self, y_true: np.ndarray, y_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Compute the gradient of the binary cross-entropy loss with respect to the predicted probabilities. Since
            the class computes the mean(y_true * log(sigmoid(input)) + (1 - y_true) * log(1 - sigmoid(input))),
            the derivative of the loss with respect to the input to the sigmoid is y_pred*(1 - y_pred).
            The derivation of the loss with to the input to the binary cross entropy is
            (y_pred - y_true)/y_pred*(1 - y_pred). --->
            [(y_pred - y_true) / (y_pred * (1 - y_pred))] * (y_pred * (1 - y_pred)) = (y_pred - y_true)
            See https://math.stackexchange.com/q/3220477 for more details.

            Parameters:
            - y_true (np.ndarray): The true labels.
            - y_prob (Optional[np.ndarray]): The predicted values. If not provided, uses the last computed values from
              the forward method.

            Return:
            - np.ndarray: The partial derivative of the log binary cross entropy and the sigmoid function.
        """
        if y_prob is None:
            if self.y_prob is None:
                raise ValueError("The forward method must be called before calling the backward method.")
            y_prob = self.y_prob

        return (y_prob.flatten() - y_true)[:, np.newaxis]

    def predict(self, y_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Predict the probabilities of the input data.

            Parameters:
            - y_prob (Optional[np.ndarray]): The predicted values. If not provided, uses the last computed values from
              the forward method.

            Returns
            - np.ndarray: The predicted values.
        """
        if y_prob is not None:
            return (y_prob > 0.5) * 1.0

        if self.y_prob is None:
            raise ValueError("The forward method must be called before calling the predict method.")

        return (self.y_prob > 0.5) * 1.0
