from .early_stopping import EarlyStopping
from .parse_conf_file import parse_conf_file, write_to_csv, load_monk_configuration, load_cup_configuration
from .grid_search import GridSearch

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'EarlyStopping',
    'GridSearch',
    'parse_conf_file',
    'write_to_csv',
    'split_into_train_test_set',
    'load_monk_configuration',
    'load_cup_configuration',
    'style',
    'show_monk_plots',
    'kfold_indices',
    'show_cup_plots_tr_vl',
    'show_cup_plots_ts',
    'what_to_run',
    'choices'
]


def show_monk_plots(history_train, dataset_name):
    # Metrics Plot
    plt.rcParams.update({'font.size': 14})
    plt.plot(history_train['train_loss'][3:], label='Training Loss')
    plt.plot(history_train['val_loss'][3:], label='Test Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name)
    plt.legend()
    plt.show()

    plt.plot(history_train['train_accuracy'][3:], label='Training Accuracy')
    plt.plot(history_train['val_accuracy'][3:], label='Test Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(dataset_name)
    plt.legend()
    plt.show()

    plt.close()


def show_cup_plots_tr_vl(history_train):
    # Metrics Plot
    plt.rcParams.update({'font.size': 14})
    plt.plot(history_train['train_loss'][3:], label='Training Loss')
    if len(history_train['val_loss']) > 0:
        plt.plot(history_train['val_loss'][3:], label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('CUP')
    plt.legend()
    plt.show()

    plt.plot(history_train['train_accuracy'][3:], label='Training MEE')
    if len(history_train['val_accuracy']) > 0:
        plt.plot(history_train['val_accuracy'][3:], label='Validation MEE', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('MEE')
    plt.title('CUP')
    plt.legend()
    plt.show()

    plt.close()


def show_cup_plots_ts(history_train):
    # Metrics Plot
    plt.rcParams.update({'font.size': 14})
    plt.plot(history_train['train_loss'][3:], label='Internal Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('CUP')
    plt.legend()
    plt.show()

    plt.plot(history_train['train_accuracy'][3:], label='Internal Test MEE')
    plt.xlabel('Epochs')
    plt.ylabel('MEE')
    plt.title('CUP')
    plt.legend()
    plt.show()

    plt.close()


def split_into_train_test_set(X, y, test_size, random_state):
    X = np.array(X)
    y = np.array(y)
    if 1 <= test_size <= 100:
        test_size = test_size / 100
    elif 1 > test_size > 0:
        test_size = test_size
    else:
        raise ValueError("test_size should be between 1 and 100 or 0 and 1")
    np.random.seed(random_state)
    test_size = int(test_size * X.shape[0])
    train_size = X.shape[0] - test_size
    indeces = np.random.permutation(X.shape[0])
    test_indecies = indeces[:test_size]
    train_indecies = indeces[test_size:]
    X_train, X_test = X[train_indecies, :], X[test_indecies, :]
    y_train, y_test = y[train_indecies], y[test_indecies]
    return X_train, X_test, y_train, y_test


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def kfold_indices(data, k_fold):
    fold_size = len(data) // k_fold
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    folds = []
    for i in range(k_fold):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    print(f"Fold size: {fold_size}")
    return folds


def what_to_run(term_input=None):
    if term_input is None:
        print(f"{style.GREEN}What do you want to run?{style.RESET}")
        print(f"{style.BLUE}1{style.RESET} - Run Grid Search")
        print(f"{style.BLUE}2{style.RESET} - Run One-time Training (with hyperparameters in config file)")
        print(f"{style.BLUE}3{style.RESET} - Run Training with Internal Test (with hyperparameters in config file)")
        print(f"{style.BLUE}4{style.RESET} - Run Final Training (with best hyperparameters in config file, no validation "
              f"set, no internal test set)")

        choice = input(f"{style.GREEN}Enter your choice:{style.RESET} ")
    else:
        choice = term_input
    try:
        choice = int(choice)
        if choice not in [1, 2, 3, 4]:
            raise ValueError
    except ValueError:
        print(f"{style.RED}Invalid input. Please enter a number between 1 and 3.{style.RESET}")
        exit()
    return choice


class choices():
    GRID_SEARCH = 1  # Run Grid Search
    ONE_TIME_TRAINING_AND_VALIDATION = 2  # Run normal training with validation and validation set
    TRAINING_AND_TEST = 3 # Run normal training, then assess the model with internal test set
    FINAL_TRAINING = 4  # Run final training with the best hyperparameters, no validation set, no internal test set
