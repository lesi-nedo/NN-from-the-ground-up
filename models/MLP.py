from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import cast, Any
from typing import Callable, Dict, Optional, Tuple
import math
import warnings

from activation_functions import ActivationFunction
from layers import Dense


class BaseML(ABC):
    """Base class for the Multi-Layer Perceptron (MLP) models.
    """

    @abstractmethod
    def __init__(
            self, units: list[int | list], activation: ActivationFunction | None,
            weights_initializer: str, weights_mean: float, l1: bool | float | int = False,
            l2: bool | float | int = False, reg_lambda: float = .0,
            grad_clip: float = .0
    ) -> None:
        """
            Initialize the neural network.

            Parameters:
            - grad_clip (float): The threshold for gradient clipping. Gradients exceeding this value will be clipped.

            Returns:
            - None
        """
        self.optimizer = None
        self.loss = None
        self.batch_size = None
        self.layers = []
        self.grad_clip = grad_clip
        self.num_layers = 0
        self.l1 = l1
        self.l2 = l2

        input_layer_size, hidden_layers_sizes, output_layer_size = units

        # Add the input layer and hidden layers
        prev_hidden_layer_size = input_layer_size
        for hidden_layer_size in hidden_layers_sizes:
            hidden_layer_size = math.floor(hidden_layer_size * prev_hidden_layer_size)

            self.add(Dense(
                input_dim=prev_hidden_layer_size,
                units=hidden_layer_size,
                activation=activation,
                weights_initializer=weights_initializer,
                weights_mean=weights_mean,
                L2_weight_regularization=reg_lambda if l2 else .0,
                L1_weight_regularization=reg_lambda if l1 else .0)
            )

            prev_hidden_layer_size = hidden_layer_size

        # Add the output layer
        self.add(Dense(
            input_dim=prev_hidden_layer_size,
            units=output_layer_size,
            weights_initializer=weights_initializer,
            weights_mean=weights_mean,
            L2_weight_regularization=reg_lambda if l2 else .0,
            L1_weight_regularization=reg_lambda if l1 else .0)
        )

    @abstractmethod
    def _print_metrics(self, average_loss: float, average_accuracy: float, epoch: int) -> None:
        pass

    @abstractmethod
    def _compute_pred(self, target: np.ndarray, output: np.ndarray, verbose: bool = True) -> Tuple[float, float]:
        pass

    @abstractmethod
    def _compute_on_batch(self, y_batch: np.ndarray, output: np.ndarray) -> Tuple[float, np.ndarray, float]:
        pass

    def add(self, layer: Dense, weights: Optional[np.ndarray] = None, biases: Optional[np.ndarray] = None) -> None:
        """
            Add a layer to the network and initialize its weights and biases.

            Parameters:
            - layer (Dense): The layer to be added to the network. It should have an `initialize` method.
            - weights (Optional[np.ndarray]): Optional initial weights for the layer.
            - biases (Optional[np.ndarray]): Optional initial biases for the layer.

            Returns:
            - None
        """
        # Set gradient clipping for the layer if specified
        if self.grad_clip > 0:
            layer.grad_clip = self.grad_clip

        # Increment the layer count and add the layer to the network
        self.num_layers += 1
        self.layers.append(layer)

        # Initialize the layer with the provided weights and biases
        layer.initialize(weights=weights, biases=biases)

    def predict(self, x: npt.ArrayLike, target: npt.ArrayLike, verbose: bool = True) -> tuple[float, float]:
        """
            Perform prediction on the given input data and compute the loss and accuracy.

            Parameters:
            - x (np.ndarray): Input data for the prediction.
            - target (np.ndarray): The true target values to compute the loss and accuracy.
            - verbose (bool): If True, additional information may be printed (e.g., for debugging).

            Returns:
            - tuple[float, float]: The computed loss and accuracy of the predictions.
        """
        # Forward pass through all layers
        output = np.array(x)
        target = np.array(target)
        for layer in self.layers:
            output = layer.forward(output)

        # Compute the loss and accuracy
        loss, accuracy = self._compute_pred(target, output, verbose)
        return loss, accuracy

    def _compute_single_batch(
            self,
            i: int,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            epoch_loss: float,
            epoch_accuracy: float,
            num_batches: int
    ) -> Tuple[float, float, int]:
        """
            Compute the loss and accuracy for a single batch of data.

            Parameters:
            - i (int): The starting index of the batch.
            - x (np.ndarray): The input data.
            - y (np.ndarray): The target values.
            - batch_size (int): The size of the batch.
            - epoch_loss (float): The accumulated loss for the epoch.
            - epoch_accuracy (float): The accumulated accuracy for the epoch.
            - num_batches (int): The number of batches processed so far.

            Returns:
            - Tuple[float, float, int]: Updated epoch loss, epoch accuracy, and number of batches.
        """
        # Extract the current batch
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        # Perform training on the current batch
        loss, accuracy = self.train_on_batch(x_batch, y_batch)

        # Update and return the accumulated metrics
        return epoch_loss + loss, epoch_accuracy + accuracy, num_batches + 1

    def fit(
            self,
            x: npt.ArrayLike,
            y: npt.ArrayLike,
            epochs: int = 1,
            batch_size: int = 32,
            val_set: Optional[Tuple[npt.ArrayLike, npt.ArrayLike]] = None,
            early_stopping: Optional[Callable[[float], None]] = None,
            verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
            Train the neural network on the provided dataset.

            Parameters:
            - x (np.ndarray): The input data.
            - y (np.ndarray): The target values.
            - epochs (int): The number of epochs to train the network.
            - batch_size (int): The size of each batch.
            - val_set (Optional[Tuple[np.ndarray, np.ndarray]]): A tuple of validation data (inputs and targets).
            - early_stopping (Optional[Callable[[float], None]]): A callback function for early stopping.
            - verbose (bool): If True, print training progress.

            Returns:
            - Dict[str, list]: A dictionary containing lists of training and validation losses and accuracies.
        """
        history: Dict[str, list] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Convert input data to numpy arrays
        x = np.array(x)
        y = np.array(y)
        x_len = len(x)
        train_steps = x_len // batch_size
        self.batch_size = batch_size

        if early_stopping:
            assert val_set is not None, "Validation set is required for early stopping"
        epoch_stop = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            epoch_stop = epoch

            for i in range(train_steps):
                # Compute loss and accuracy for the current batch
                epoch_loss, epoch_accuracy, num_batches = self._compute_single_batch(
                    i, x, y, batch_size, epoch_loss, epoch_accuracy, num_batches
                )

            # Handle the remaining data if not perfectly divisible by batch_size
            if train_steps * batch_size < x_len:
                self.batch_size = x_len - train_steps * batch_size
                i = train_steps * batch_size
                epoch_loss, epoch_accuracy, num_batches = self._compute_single_batch(
                    i, x, y, self.batch_size, epoch_loss, epoch_accuracy, num_batches
                )

            # Calculate average loss and accuracy for the epoch
            average_loss = epoch_loss / num_batches
            history['train_loss'].append(average_loss)
            average_accuracy = epoch_accuracy / num_batches
            history['train_accuracy'].append(average_accuracy)

            # Print metrics every 10 epochs if verbose is True
            if not (epoch + 1) % 10 and verbose:
                self._print_metrics(average_loss, average_accuracy, epoch_stop)

            # Evaluate on validation set if provided
            if val_set is not None:
                val_loss, val_accuracy = self.predict(val_set[0], val_set[1], verbose=verbose)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                # Apply early stopping if callback is provided
                if early_stopping:
                    early_stopping(val_loss)
                    if getattr(early_stopping, 'early_stop', False):
                        if verbose:
                            print(f"Early stopping at epoch: {epoch_stop}")
                        break

            # Update layers
            for layer in self.layers:
                layer.iterations += 1
                layer.zero_grad()
        if verbose:
            self._print_metrics(history['train_loss'][-1], history['train_accuracy'][-1], epoch_stop)

        res = {
            'train_loss': np.array(history['train_loss']),
            'train_accuracy': np.array(history['train_accuracy']),
            'val_loss': np.array(history['val_loss']),
            'val_accuracy': np.array(history['val_accuracy']),
            'epoch_stop': epoch_stop
        }

        return res

    def train_on_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[float, float]:
        """
            Perform a training step on a single batch of data.

            Parameters:
            - x_batch (np.ndarray): A batch of input data.
            - y_batch (np.ndarray): The corresponding batch of target values.

            Returns:
            - Tuple[float, float]: The loss and accuracy for the batch.
        """
        # Forward pass through all layers
        output = x_batch
        for layer in self.layers:
            layer.batch_size = self.batch_size
            output = layer.forward(output)
        # Compute the loss, gradients, and accuracy for the batch
        loss, gradients, accuracy = self._compute_on_batch(y_batch, output)

        # Backward pass through all layers to compute gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)

        # Update weights of all layers using the optimizer
        for layer in self.layers:
            self.optimizer.update(layer)

        return loss, accuracy

    def compile(self, loss, optimizer) -> None:
        """
            Set the loss function and optimizer for the neural network.

            Parameters:
            - loss: The loss function to be used for training. Must be provided.
            - optimizer: The optimizer to be used for updating weights. Must be provided.

            Raises:
            - Exception: If either the optimizer or the loss function is None.
        """
        if loss is None:
            raise Exception("Loss function should not be None")

        if optimizer is None:
            raise Exception("Optimizer should not be None")

        # Set the loss function and optimizer
        self.loss = loss
        self.optimizer = optimizer

    def save(self, path):
        pass

    def load(self, path):
        pass

    def summary(self):
        pass


class MLPClassifier(BaseML):
    """
        Multi-Layer Perceptron (MLP) classifier model.

        Inherits from BaseML and provides methods for training and evaluating an MLP classifier.
    """

    def __init__(
            self, units: list[int | list], activation: ActivationFunction | None, weights_initializer: str,
            weights_mean: float, l1: float = .0, l2: float = .0, reg_lambda: float = .0,
            grad_clip: float = .0
    ) -> None:
        """
            Initialize the MLPClassifier.

            Parameters:
            - grad_clip (float): The gradient clipping value to prevent exploding gradients. Default is 0 (no clipping).

            Calls the constructor of the base class BaseML.
        """
        super().__init__(units=units, activation=activation, weights_initializer=weights_initializer,
                         weights_mean=weights_mean, l1=l1, l2=l2, reg_lambda=reg_lambda, grad_clip=grad_clip)

    def _print_metrics(self, average_loss: float, average_accuracy: float, epoch: int) -> None:
        """
            Print the metrics for a given epoch.

            Parameters:
            - average_loss (float): The average loss for the epoch.
            - average_accuracy (float): The average accuracy for the epoch.
            - epoch (int): The current epoch number.
        """
        print(f'epoch: {epoch + 1}, ' +
              f'Accuracy: {average_accuracy:.3f}, ' +
              f'loss: {average_loss:.3f}, '
              )

    def _compute_pred(self, target: np.ndarray, output: np.ndarray, verbose: bool = True) -> Tuple[float, float]:
        """
            Compute the loss and accuracy of the model based on predictions.

            Parameters:
            - target (np.ndarray): The true target values.
            - output (np.ndarray): The model's output predictions.
            - verbose (bool): If True, prints the evaluated accuracy and loss.

            Returns:
            - Tuple[float, float]: The loss and accuracy of the model.
        """
        # Compute loss using the loss function
        loss = self.loss.forward(target, output.flatten())
        # Make predictions based on the model output
        predictions = self.loss.predict(self.loss.y_prob)

        # Calculate accuracy
        accuracy = np.mean(predictions.flatten() == target)

        if verbose:
            print(f'\nEvaluated Accuracy: {accuracy}')
            print(f'Evaluated Loss: {loss}\n')

        return float(loss), float(accuracy)

    def _compute_on_batch(self, y_batch: np.ndarray, output: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
            Compute the loss, gradients, and accuracy for a single batch.

            Parameters:
            - y_batch (np.ndarray): The true target values for the batch.
            - output (np.ndarray): The model's output predictions for the batch.

            Returns:
            - Tuple[float, np.ndarray, float]: The loss, gradients, and accuracy for the batch.
        """
        # Compute the loss for the batch
        loss = self.loss.forward(y_batch, output.flatten())

        # Compute gradients based on the loss
        gradients = self.loss.backward(y_batch, self.loss.y_prob)

        # Make predictions and compute accuracy
        predictions = self.loss.predict(self.loss.y_prob)
        accuracy = np.mean(predictions.flatten() == y_batch)

        return float(loss), cast(np.ndarray, gradients), float(accuracy)


class MLPRegressor(BaseML):
    """
        Multi-Layer Perceptron (MLP) regressor model.

        Inherits from BaseML and provides methods for training, evaluating, and predicting using an MLP regressor.
    """

    def __init__(
            self, units: list[int | Any], activation: ActivationFunction | None, weights_initializer: str,
            weights_mean: float, l1: float = .0, l2: float = .0, reg_lambda: float = .0, grad_clip: float = .0
    ) -> None:
        """
            Initialize the MLPRegressor.

            Parameters:
            - grad_clip (float): The gradient clipping value to prevent exploding gradients. Default is 0 (no clipping).

            Calls the constructor of the base class BaseML.
        """
        super().__init__(units=units, activation=activation, weights_initializer=weights_initializer,
                         weights_mean=weights_mean, l1=l1, l2=l2, reg_lambda=reg_lambda, grad_clip=grad_clip)

    def _print_metrics(self, average_loss: float, average_accuracy: float, epoch: int) -> None:
        """
            Print the metrics for a given epoch.

            Parameters:
            - average_loss (float): The average loss for the epoch.
            - average_accuracy (float): The average Mean Euclidean Error (MEE) for the epoch.
            - epoch (int): The current epoch number.
        """
        print(f'epoch: {epoch + 1}, ' +
              f'MEE: {average_accuracy:.3f}, ' +
              f'loss: {average_loss:.3f}, '
              )

    def _compute_MEE(self, target: np.ndarray, output: np.ndarray) -> float:
        """
            Compute the Mean Euclidean Error (MEE) of the model based on predictions.

            Parameters:
            - target (np.ndarray): The true target values.
            - output (np.ndarray): The model's output predictions.

            Returns:
            - float: The MEE of the model.
        """
        # Make predictions based on the model output

        # Compute Mean Euclidean Error (MEE)
        #
        # sqrt((o_x - p_x) ^ 2) + sqrt((o_y - p_y) ^ 2) + sqrt((o_z - p_z) ^ 2) MEE x + MEE y + MEE z then mean
        with warnings.catch_warnings(record=True) as w:
            accuracy = np.linalg.norm(target - output, axis=1).mean()
            if len(w) > 0:
                pass
                # print(f"Could not compute MEE for target: {target} and output: {output} difference {target - output}")
        # sqrt((o_x - p_x)^2 + (o_y - p_y)^2 + (o_z - p_z)^2) MEE of x, y, z, which is 2 norm
        # accuracy = np.linalg.norm(target - output)

        return float(accuracy)

    def _compute_pred(self, target: np.ndarray, output: np.ndarray, verbose: bool = True) -> Tuple[float, float]:
        """
            Compute the loss and Mean Euclidean Error (MEE) of the model based on predictions.

            Parameters:
            - target (np.ndarray): The true target values.
            - output (np.ndarray): The model's output predictions.
            - verbose (bool): If True, prints the evaluated MEE and loss.

            Returns:
            - Tuple[float, float]: The loss and MEE of the model.
        """
        # Compute loss using the loss function
        loss = self.loss.forward(target, output)

        # Make predictions based on the model output
        predictions = self.loss.predict(self.loss.y_prob)

        # Compute Mean Euclidean Error (MEE)
        accuracy = self._compute_MEE(target, predictions)

        if verbose:
            print(f'\nEvaluated MEE: {accuracy}')
            print(f'Evaluated Loss: {loss}\n')

        return loss, accuracy

    def _compute_on_batch(self, y_batch: np.ndarray, output: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
            Compute the loss, gradients, and Mean Euclidean Error (MEE) for a single batch.

            Parameters:
            - y_batch (np.ndarray): The true target values for the batch.
            - output (np.ndarray): The model's output predictions for the batch.

            Returns:
            - Tuple[float, np.ndarray, float]: The loss, gradients, and MEE for the batch.
        """
        # Compute the loss for the batch
        loss = self.loss.forward(y_batch, output)

        # Compute gradients based on the loss
        gradients = self.loss.backward(y_batch, self.loss.y_prob)

        # Make predictions and compute MEE
        predictions = self.loss.predict(self.loss.y_prob)

        accuracy = self._compute_MEE(y_batch, predictions)

        return loss, gradients, accuracy

    def evaluate(self, x: npt.ArrayLike) -> np.ndarray:
        """
            Compute the output of the model for the given input data.

            Parameters:
            - x (np.ndarray): The input data for evaluation.

            Returns:
            - np.ndarray: The output predictions of the model.
        """
        output = np.array(x)

        # Forward pass through all layers
        for layer in self.layers:
            output = layer.forward(output)

        return output
