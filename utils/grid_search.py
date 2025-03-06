import ast
import itertools
import signal
import numpy as np
import os
import psutil

from multiprocessing import cpu_count
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, List, Any
from contextlib import redirect_stdout

from utils import EarlyStopping
from models import MLPClassifier, MLPRegressor


def kill_child_processes(parent_pid: int, sig: int = signal.SIGTERM) -> None:
    """
        Kill all child processes of a given parent process.

        Parameters:
        - parent_pid (int): The PID of the parent process.
        - sig (int): The signal to send to the children processes.

        Returns:
        - None
    """
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


class GridSearch:
    """
        Perform grid search for hyperparameter tuning of a neural network model.
    """

    def __init__(self, objects: Dict[str, Any], param_grid: Dict[str, str]) -> None:
        """
            Initialize the GridSearch instance with the model, training data, and parameter grid.

            Parameters:
            - objects (Dict[str, Any]): Stores the model, training data, loss function, optimizer,
              and optional initialization function.
            - param_grid (Dict[str, str]): Stores the grid of hyperparameters for tuning.

            Returns:
            - None
        """
        self.objects = objects
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = 0 if objects['model'] == MLPClassifier else np.inf
        self.test_loss = np.inf
        self.val_loss = 0
        self.test_accuracy = 0 if objects['model'] == MLPClassifier else np.inf

    def _parallel_comp(
            self, len_train:int, args: Tuple[int, float, int, int, float, float, float, float, bool, List[int], int, str, float]
    ) -> Tuple[float, float, float, float, Dict[str, Any]]:
        """
            Perform model training and evaluation for a given set of hyperparameters.

            Parameters:
            - args (Tuple): Tuple containing hyperparameter values to evaluate.

            Returns:
            - Tuple[float, Dict[str, Any]]: Tuple containing the validation accuracy/loss and the corresponding
              hyperparameters.
        """
        (random_state, learning_rate, batch_size_perc, epochs, grad_clip, reg_lambda, decay, momentum, nesterov,
         hidden_layer_sizes, l2_l1, weights_initializer, weights_mean) = args

        np.random.seed(random_state)
        model = self.objects['model'](
            units=[self.objects['X_train'].shape[1], hidden_layer_sizes, self.objects['nn_output_size']],
            # [input_size, tuple of percentages for hidden layer sizes relative to the previous layer, output_size]
            activation=self.objects['activation'](),
            weights_initializer=weights_initializer,
            weights_mean=weights_mean,
            l1=not l2_l1,
            l2=l2_l1,
            reg_lambda=reg_lambda,
            grad_clip=grad_clip
        )

        model.compile(
            loss=self.objects['loss'](),
            optimizer=self.objects['learning_alg'](
                learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov
            )
        )
        batch_size = round(batch_size_perc * len_train)
        early_stopping_instance = EarlyStopping(patience=int(self.objects['patience'])) if self.objects[
            'early_stopping'] else None
        history = model.fit(
            self.objects['X_train'], self.objects['y_train'], epochs=epochs, batch_size=batch_size,
            val_set=(self.objects['X_val'], self.objects['y_val']),
            early_stopping=early_stopping_instance, verbose=False
        )

        layers_units = [self.objects['X_train'].shape[1]]

        for layer in model.layers:
            if hasattr(layer, 'units'):
                layers_units.append(layer.units)

        val_loss, val_accuracy = history['val_loss'][-1], history['val_accuracy'][-1]
        test_loss, test_accuracy = history['train_loss'][-1], history['train_accuracy'][-1]
        best_params: dict[str, Any] = {
            'learning_rate': learning_rate,
            'batch_size': batch_size, 'batch_size_perc': batch_size_perc, 'epochs': epochs, 'grad_clip': grad_clip,
            'reg_lambda': reg_lambda, 'decay': decay, 'momentum': momentum, 'nesterov': nesterov,
            'num_linear_layers': len(hidden_layer_sizes) + 1, 'hidden_layers_sizes': hidden_layer_sizes,
            "L2": bool(l2_l1), "L1": not bool(l2_l1), "weights_initializer": weights_initializer,
            "weights_mean": weights_mean, "random_state": random_state, 'epoch_stop': history['epoch_stop'],
            "layers_units": layers_units
        }
        return float(test_loss), float(val_loss), float(test_accuracy), float(val_accuracy), best_params

    def print_results(self) -> None:
        print(f"\nNew best train loss: {self.test_loss:.4f}")
        print(f"\nNew best validation loss: {self.val_loss:.4f}")
        print(f"\nNew best train accuracy: {self.test_accuracy:.4f}")
        print(f"\nNew best validation accuracy: {self.best_score:.4f}")
        print(f"\nNew best hyperparameters: {self.best_params}\n")

    def search(self) -> Dict[str, Any]:
        """
            Perform grid search to find the best hyperparameters.

            Returns:
            - Dict[str, Any]: Dictionary containing the best hyperparameters found.
        """
        all_hyp = [
            ast.literal_eval(self.param_grid['grid_random_states']),
            ast.literal_eval(self.param_grid['grid_learning_rates']),
            ast.literal_eval(self.param_grid['grid_batch_sizes_perc']),
            ast.literal_eval(self.param_grid['grid_epochs_list']),
            ast.literal_eval(self.param_grid['grid_grad_clips']),
            ast.literal_eval(self.param_grid['grid_reg_lambdas']),
            ast.literal_eval(self.param_grid['grid_decays']),
            ast.literal_eval(self.param_grid['grid_momentums']),
            ast.literal_eval(self.param_grid['grid_nesterovs']),
            ast.literal_eval(self.param_grid['grid_hidden_layers_sizes']),
            ast.literal_eval(self.param_grid['grid_l2_l1']),
            ast.literal_eval(self.param_grid['grid_weights_initializer'])
        ]
        if self.param_grid.get('grid_weights_mean', None) is not None:
            all_hyp.append(ast.literal_eval(self.param_grid['grid_weights_mean']))
        else:
            all_hyp.append([0.0])
        grid = itertools.product(*all_hyp)

        def comp_mlp_classifier(val_accuracy_parm, best_score):
            return val_accuracy_parm > best_score

        def comp_mlp_regression(val_accuracy_parm, best_score):
            return val_accuracy_parm < best_score

        func_comp = {MLPClassifier: comp_mlp_classifier, MLPRegressor: comp_mlp_regression}
        num_cpus = cpu_count()
        perc_using = 0.1
        print(f"Number of CPUs: {num_cpus}")
        max_worker = 2  # int(num_cpus * perc_using)
        print(f"Spanning {max_worker} process for parallel computation")
        len_train = len(self.objects['X_train'])
        with ProcessPoolExecutor(max_workers=max_worker) as executor:
            print("Starting Grid Search, this might take a while.\n")

            futures = [executor.submit(self._parallel_comp, len_train, all_hyp) for all_hyp in grid]

            for future in tqdm(as_completed(futures), total=len(futures)):
                test_loss, val_loss, test_accuracy, val_accuracy, params = future.result()

                if func_comp[self.objects['model']](val_accuracy, self.best_score):
                    self.best_score = val_accuracy
                    self.best_params = params
                    self.test_loss = test_loss
                    self.val_loss = val_loss
                    self.test_accuracy = test_accuracy
                    if self.objects.get('file_name', None) is not None:
                        with open(self.objects['file_name'], 'a') as f:
                            with redirect_stdout(f):
                                self.print_results()
                    else:
                        self.print_results()
                    if self.objects['threshold'] is not None \
                            and func_comp[self.objects['model']](self.best_score, self.objects['threshold']):
                        print(f"Threshold reached: {self.objects['threshold']}")
                        kill_child_processes(os.getpid())
                        executor.shutdown(wait=False)
                        break
        return self.best_params
