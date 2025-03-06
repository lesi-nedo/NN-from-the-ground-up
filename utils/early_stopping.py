import numpy as np


class EarlyStopping:
    """
        Early stopping mechanism to halt training when validation loss does not improve.
    """

    def __init__(self, patience: int = 4, delta: float = 0, verbose: bool = False) -> None:
        """
            Initialize the EarlyStopping instance.

            Parameters:
            - patience (int): Number of epochs to wait for an improvement before stopping training.
            - delta (float): Minimum change required in validation loss to be considered an improvement.
            - verbose (bool): If True, enables printing of additional information about the stopping process.

            Returns:
            - None
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.verbose = verbose

    def __call__(self, val_loss: float) -> None:
        """
            Update the early stopping criteria based on the current validation loss.

            Parameters:
            - val_loss (float): The current validation loss to evaluate.

            Returns:
            - None
        """
        score = -val_loss

        if self.best_score is None:
            # First evaluation; set initial best score
            self.best_score = score

        elif score < self.best_score + self.delta:
            # No significant improvement in validation loss
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # Stop training if patience is exceeded
                self.early_stop = True
        else:
            # Improvement found; reset the counter and update best score
            self.best_score = score
            self.counter = 0
