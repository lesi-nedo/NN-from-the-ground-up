import numpy as np
from typing import Optional

class MAELoss:

    def __init__(self):
        self.y_prob = None

    def forward(self, y_true: np.ndarray, input: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """
            Calculate the Mean Absolute Error (MAE) loss.

            Parameters:
            - y_true (np.ndarray): True labels, e.g.: [3.0, 1.0, 2.0, 4.0].
            - input (np.ndarray): Values that are coming from the previous layer.
            - weights (Optional[np.ndarray]): Weights that influence the individual loss elements.

            Returns:
            - float: The mean absolute error loss.
        """
        # Ensure the inputs are NumPy arrays
        y_true = np.array(y_true, dtype=np.float64)
        input = np.array(input, dtype=np.float64)

        loss = np.abs(y_true - input)

        self.y_prob = input

        if weights is not None:
            weights = np.array(weights, dtype=np.float64)
            loss = np.multiply(loss, weights)

        return np.mean(loss)

    def backward(self, y_true: np.ndarray, y_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Compute the gradient of the mean absolute error loss with respect to the predicted values.

            Parameters:
            - y_true (np.ndarray): The true labels.
            - y_prob (Optional[np.ndarray]): The predicted values. If not provided, uses the last computed values from
              the forward method.

            Returns
            - np.ndarray: The gradient of the mean absolute error loss with respect to the predicted values.
        """
        if y_prob is None:
            if self.y_prob is None:
                raise ValueError("The forward method must be called before calling the backward method.")
            y_prob = self.y_prob

        return np.sign(y_prob - y_true)

    def predict(self, y_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """
            Predict the values based on the predicted values.

            Parameters:
            - y_prob (Optional[np.ndarray]): The predicted values. If not provided, uses the last computed values from
              the forward method.

            Returns:
            - np.ndarray: The predicted values.
        """
        if y_prob is not None:
            return y_prob

        if self.y_prob is None:
            raise ValueError("The forward method must be called before calling the predict method.")

        return self.y_prob
