import numpy as np
from layers import Dense


class SGD:
    """
        Stochastic Gradient Descent algorithm with Nesterov momentum and learning rate decay.
    """

    def __init__(self, learning_rate=1.0, decay=.0, momentum=.0, nesterov=True, min_learning_rate=1e-8):
        if momentum < 0 or momentum > 1.0:
            raise Exception("Momentum should be in the range [0, 1]")
        if decay < 0 or decay > 1.0:
            raise Exception("Decay value should be in the range [0, 1]")

        self.learning_rate = learning_rate
        self.decay = decay
        self.c_lr = learning_rate
        self.momentum = momentum
        self.min_learning_rate = min_learning_rate
        self.nesterov = nesterov

    def update(self, layer: Dense) -> None:
        """
            Compute and apply the update step for the weights and biases using Nesterov momentum.

            Parameters:
            - layer (Dense): The layer that contains the weights, biases, and gradients to be updated.

            Returns:
            - None
        """

        # Apply learning rate decay if specified
        if self.decay:
            # Adjust the current learning rate based on the decay rate and number of iterations
            self.c_lr = self.learning_rate * (1.0 / (1 + self.decay * layer.iterations))

        # Scale the learning rate by the batch size
        scaled_lr = self.c_lr / layer.batch_size

        # Ensure the scaled learning rate does not fall below the minimum learning rate
        if self.min_learning_rate > scaled_lr:
            self.decay = 0
            scaled_lr = self.min_learning_rate

        # Calculate the weight and bias updates based on the gradients
        weight_updates = -scaled_lr * layer.nabla_w
        bias_updates = -scaled_lr * layer.nabla_b

        # Apply momentum if specified
        if self.momentum:

            # Initialize previous velocity matrices if they don't exist
            if not hasattr(layer, "prev_VW"):
                layer.prev_VW = np.zeros_like(layer.weights)
            if not hasattr(layer, "prev_VB"):
                layer.prev_VB = np.zeros_like(layer.biases)

            # Apply Nesterov momentum if enabled
            if self.nesterov:
                # Compute velocity with momentum
                VW = self.momentum * layer.prev_VW + weight_updates
                VB = self.momentum * layer.prev_VB + bias_updates

                # Adjust the weight and bias updates using Nesterov's method
                weight_updates = -self.momentum * layer.prev_VW + (1 + self.momentum) * VW
                bias_updates = -self.momentum * layer.prev_VB + (1 + self.momentum) * VB
            else:
                # Standard momentum update
                VW = self.momentum * layer.prev_VW + layer.nabla_w
                VB = self.momentum * layer.prev_VB + layer.nabla_b

                # Scale the velocity to update weights and biases
                weight_updates = -scaled_lr * VW
                bias_updates = -scaled_lr * VB

            # Store the current velocity for the next iteration
            layer.prev_VW = VW
            layer.prev_VB = VB

        # Ensure batch_size is set (useful if layer is improperly initialized)
        if not hasattr(layer, "batch_size"):
            layer.batch_size = 1

        # Apply the computed updates to the layer's weights and biases

        layer.weights += weight_updates
        layer.biases += bias_updates
