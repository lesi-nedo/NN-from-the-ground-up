from activation_functions import Softplus, LeakyReLU
from datasets import read_monk_dataset, select_monk_dataset

from loss_functions import BCELogitsLoss
from models import MLPClassifier
from optimizer import SGD
from utils import EarlyStopping, GridSearch, load_monk_configuration, show_monk_plots
from typing import cast

import numpy as np

if __name__ == '__main__':
    dataset_name = select_monk_dataset()

    settings = load_monk_configuration(dataset_name)

    random_seed = settings['random_state']

    X_train, y_train, X_test, y_test = read_monk_dataset(
        'datasets/monks/' + dataset_name, shuffle=True, random_state=random_seed
    )

    # Perform grid search
    params = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_test,
        'y_val': y_test,
        'activation': Softplus,
        'loss': BCELogitsLoss,
        'model': MLPClassifier,
        'learning_alg': SGD,
        'nn_output_size': 1,
        'threshold': 0.9999,
        'early_stopping': settings['early_stopping'],
        'patience': settings['patience']
    }

    grid_search = GridSearch(params, settings)
    best_params = grid_search.search()

    # best_params = {
    #     'learning_rate': 0.025,
    #     'batch_size': 60,
    #     'epochs': 55,
    #     'grad_clip': 2.5,
    #     'reg_lambda': 0.0,
    #     'decay': 0.009,
    #     'momentum': 0.9,
    #     'nesterov': True,
    #     'hidden_layers_sizes': [0.8, 0.2],
    #     'L1': False,
    #     'L2': True,
    #     'weights_initializer': 'xavier',
    #     'random_state': 125,
    #     'weights_mean': 0.0
    # }

    np.random.seed(best_params['random_state'])
    model = MLPClassifier(
        units=[X_train.shape[1], cast(list, best_params['hidden_layers_sizes']), 1],
        # [input_size, tuple of percentages for hidden layer sizes relative to the previous layer, output_size]
        activation=Softplus(),
        weights_initializer=best_params['weights_initializer'],
        weights_mean=best_params['weights_mean'],
        l1=best_params['L1'],
        l2=best_params['L2'],
        reg_lambda=best_params['reg_lambda'],
        grad_clip=best_params['grad_clip']
    )

    # We are minimizing the log binary cross entropy loss; more info:
    # https://www.cs.tufts.edu/comp/135/2023f/slides/day09_logistic_regression.pdf

    model.compile(
        loss=BCELogitsLoss(),
        optimizer=SGD(
            learning_rate=best_params['learning_rate'],
            decay=best_params['decay'],
            momentum=best_params['momentum'],
            nesterov=best_params['nesterov']
        )
    )

    history_train = model.fit(
        X_train,
        y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        val_set=(X_test, y_test),
        early_stopping=EarlyStopping(patience=settings['patience']) if settings['early_stopping'] else None,
        verbose=True
    )
    # loss, accuracy = model.predict(X_test.to_numpy(), y_test.to_numpy())

    print(f"Train loss: {history_train['train_loss'][-1]} \n")
    print(f"Validation loss: {history_train['val_loss'][-1]} \n")
    print(f"Train accuracy: {history_train['train_accuracy'][-1]} \n")
    print(f"Validation accuracy: {history_train['val_accuracy'][-1]} \n")

    print(f"Best hyperparameters: {best_params}")

    show_monk_plots(history_train, dataset_name)
