import numpy as np
from typing import Optional
import warnings

class Dense:
    def __init__(
            self, input_dim, units, activation=None, batch_size=1, L2_weight_regularization=0,
            L2_biases_regularization=0,
            L1_weight_regularization=0, L1_biases_regularization=0, weights_initializer='xavier', weights_mean=0.0
    ):
        self.nabla_b = None
        self.nabla_w = None
        self.output_data = None
        self.input_data = None
        self.linear_output = None
        self.iterations = 0
        self.grad_clip = 0
        self.L2_W_reg = L2_weight_regularization
        self.L2_B_reg = L2_biases_regularization
        self.L1_W_reg = L1_weight_regularization
        self.L1_B_reg = L1_biases_regularization
        self.input_dim = input_dim
        self.units = units
        self.weights = None
        self.biases = None
        self.activation_function = activation.evaluate if activation else None
        self.activation_derivative = activation.derivative if activation else None
        self.__batch_size = batch_size
        self.prev_VW = None
        self.prev_VB = None
        self._grad_clip = 0
        self.weights_initializer = weights_initializer
        self.weights_mean = weights_mean

    def initialize(
            self, weights: Optional[np.ndarray] = None, biases: Optional[np.ndarray] = None
    ) -> None:
        """
            Initialize the weights and biases of the layer.

            Parameters:
            - weights (Optional[np.ndarray]): The weight matrix for the layer. If not provided, it is initialized with
              random values.
            - biases (Optional[np.ndarray]): The bias vector for the layer. If not provided, it is initialized to zeros.
            - weights_mean (float): The mean value for the weights initialization, used in the CUP initialization.

            Returns:
            - None
        """
        if weights is None:
            if self.weights_initializer == 'default':
                # Default weight initialization: uniform distribution between -0.2 and 0.2
                self.weights = np.random.uniform(-0.2, 0.2, size=(self.input_dim, self.units))
            elif self.weights_initializer == 'larger':
                # Larger weight initialization: uniform distribution between -5 and 5
                self.weights = np.random.uniform(-5, 5, size=(self.input_dim, self.units))
            elif self.weights_initializer == 'xavier':
                # Xavier initialization: normal distribution based on the number of input and output units
                self.weights = np.random.normal(
                    loc=self.weights_mean, scale=np.sqrt(6) / np.sqrt(self.units + self.input_dim),
                    size=(self.input_dim, self.units)
                )
            elif self.weights_initializer == 'he':
                # He initialization: normal distribution based on the number of input units
                self.weights = np.random.normal(loc=self.weights_mean, scale=np.sqrt(2) / np.sqrt(self.input_dim),
                                                size=(self.input_dim, self.units))
            elif self.weights_initializer == 'CUP':
                # CUP initialization: normal distribution, shifted by the mean and scaled by input dimension
                self.weights = np.random.normal(loc=self.weights_mean, scale=2 / np.sqrt(self.input_dim),
                                                size=(self.input_dim, self.units))
            else:
                # Default fallback if an unknown initializer is specified
                raise ValueError(f"Unknown weight initializer: {self.weights_initializer}")
        else:
            self.weights = weights

        if biases is None:
            self.biases = np.zeros((1, self.units))
        else:
            self.biases = biases
        self.weights = self.weights.astype(np.float64)
        self.biases = self.biases.astype(np.float64)
        self.prev_VB = np.zeros_like(self.biases, dtype=np.float64)
        self.prev_VW = np.zeros_like(self.weights, dtype=np.float64)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
            Perform the forward pass of the layer.

            Parameters:
            - input_data (np.ndarray): The input data to the layer, typically a 2D array where each row is a data sample
              and each column is a feature.

            Returns:
            - np.ndarray: The output data after applying the linear transformation and the activation function (if any).
        """
        self.input_data = input_data
        self.linear_output = np.dot(input_data, self.weights) + self.biases

        if self.activation_function:
            self.output_data = self.activation_function(self.linear_output)
        else:
            self.output_data = self.linear_output

        return self.output_data

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
            Perform the backward pass to compute the derivatives of the weights and biases.

            Parameters:
            - output_gradient (np.ndarray): The gradient of the loss with respect to the output of this layer, often
              denoted as W^{l+1}delta^{l+1}.

            Returns:
            - np.ndarray: The gradient of the loss with respect to the input of this layer, often denoted as
              W^{l}delta^{l}.
        """

        # Apply the derivative of the activation function if it is provided
        if self.activation_derivative:
            derv = self.activation_derivative(self.linear_output)
            output_gradient = np.multiply(output_gradient, derv)

        # Compute the gradient of the weights: input_data.T * output_gradient
        weights_gradient = np.dot(self.input_data.T, output_gradient)

        # Compute the gradient of the input data to propagate it to the previous layer
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Store the computed gradients for weights and biases
        self.nabla_w = weights_gradient
        self.nabla_b = np.sum(output_gradient, axis=0, keepdims=True)

        # Apply L2 regularization (weight decay) to the gradients if specified
        if self.L2_B_reg > 0 or self.L2_W_reg > 0:
            self.nabla_w += 2 * self.L2_W_reg * self.weights
            self.nabla_b += 2 * self.L2_B_reg * self.biases

        # Apply L1 regularization (sparsity constraint) to the gradients if specified
        if self.L1_B_reg > 0 or self.L1_W_reg > 0:
            dl1_dw = np.where(self.weights >= 0, 1, -1)
            dl1_db = np.where(self.biases >= 0, 1, -1)
            self.nabla_w += self.L1_W_reg * dl1_dw
            self.nabla_b += self.L1_B_reg * dl1_db

        # Apply gradient clipping to prevent the gradients from exploding
        if self.grad_clip > 0:
            norm_w = np.linalg.norm(self.nabla_w)
            norm_b = np.linalg.norm(self.nabla_b)
            if norm_w > self.grad_clip:
                self.nabla_w = self.nabla_w * self.grad_clip / norm_w
            if norm_b > self.grad_clip:
                self.nabla_b = self.nabla_b * self.grad_clip / norm_b

        # Return the gradient of the loss with respect to the input of this layer
        return input_gradient

    def zero_grad(self) -> None:
        """
            Resets the gradient values.

            Returns:
            - None
        """
        if self.nabla_b is not None and self.nabla_w is not None:
            self.nabla_w = np.zeros_like(self.nabla_w).astype(np.float64)
            self.nabla_b = np.zeros_like(self.nabla_b).astype(np.float64)
            self.prev_VB = np.zeros_like(self.biases).astype(np.float64)
            self.prev_VW = np.zeros_like(self.weights).astype(np.float64)

    def compute_regularization_term(self) -> float:
        """
            Compute the regularization term for the layer.

            Returns:
            - float: The regularization term for the layer.
        """
        regularization_value = 0

        if self.L2_B_reg > 0 or self.L2_W_reg > 0:
            regularization_value += self.L2_W_reg * np.sum(self.weights * self.weights)
            regularization_value += self.L2_B_reg * np.sum(self.biases * self.biases)

        if self.L1_B_reg > 0 or self.L1_W_reg > 0:
            regularization_value += self.L1_W_reg * np.sum(np.abs(self.weights))
            regularization_value += self.L1_B_reg * np.sum(np.abs(self.biases))

        return regularization_value

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.__batch_size = batch_size

    @property
    def L2_weight_regularization(self):
        return self.L2_W_reg

    @L2_weight_regularization.setter
    def L2_weight_regularization(self, L2_weight_regularization):
        self.L2_W_reg = L2_weight_regularization

    @property
    def L2_biases_regularization(self):
        return self.L2_B_reg

    @L2_biases_regularization.setter
    def L2_biases_regularization(self, L2_biases_regularization):
        self.L2_B_reg = L2_biases_regularization

    @property
    def grad_clip(self):
        return self._grad_clip

    @grad_clip.setter
    def grad_clip(self, grad_clip):
        self._grad_clip = grad_clip
