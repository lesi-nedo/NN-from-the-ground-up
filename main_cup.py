from activation_functions import Softplus, LeakyReLU
from datasets import read_cup_dataset, save_cup_predictions

from loss_functions import MAELoss, MSELoss

from models.MLP import MLPRegressor
from optimizer import SGD
from utils import (GridSearch, EarlyStopping, load_cup_configuration, split_into_train_test_set,
                   show_cup_plots_tr_vl, show_cup_plots_ts, what_to_run, choices, kfold_indices)

import numpy as np
import ast
import sys

if __name__ == '__main__':
    n = len(sys.argv)
    file_name = None
    choice = None
    print(f"Arguments count: {n}")
    if n == 2:
        choice = what_to_run(sys.argv[1])
    elif n == 3:
        choice = what_to_run(sys.argv[1])
        file_name = sys.argv[2]
    elif n > 3:
        print("Invalid number of arguments")
        print("Usage: python main_cup.py [<choice>] [<file_name>]")
        print("Choices: 0 - Grid Search, 1 - One time training and validation, 2 - Training and test, 3 - Final "
              "training")
        print("File name: Name of the file to save the grid search results")
        sys.exit(1)
    else:
        choice = what_to_run()
    # Load CUP dataset configuration
    settings = load_cup_configuration()
    seed_sett = settings['random_state']

    # Read CUP dataset
    X_train_org, y_train_org, X_test_blind = read_cup_dataset(
        'datasets/cup/ml-cup-23', shuffle=True, random_state=seed_sett
    )

    # Split the dataset into training and testing sets, in a deterministic way. So, to never
    # train and validate the model on test set.
    X_train, X_test, y_train, y_test = split_into_train_test_set(
        X_train_org, y_train_org, test_size=settings['test_size'], random_state=seed_sett
    )

    np.random.seed(seed_sett)

    # Perform grid search if enabled
    if choice == choices.GRID_SEARCH:
        X_train, X_val, y_train, y_val = split_into_train_test_set(
            X_train, y_train, test_size=settings['validation_size'], random_state=seed_sett
        )
        print(f"Of which {len(X_train)} are training samples, {len(X_val)} are validation samples")

        params = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'activation': LeakyReLU,
            'loss': MSELoss,
            'model': MLPRegressor,
            'learning_alg': SGD,
            'nn_output_size': 3,
            'threshold': 0.1,
            'early_stopping': settings['early_stopping'],
            'patience': settings['patience'],
            'file_name': file_name
        }

        grid_search = GridSearch(params, settings)
        best_params = grid_search.search()
        print(f"Best hyperparameters: {best_params}")
    else:
        model = MLPRegressor(
            units=[X_train_org.shape[1], ast.literal_eval(settings['hidden_layers_sizes']), 3],
            # [input_size, tuple of percentages for hidden layer sizes relative to the previous layer, output_size]
            activation=LeakyReLU(),
            weights_initializer=ast.literal_eval(settings['weights_initializer']),
            weights_mean=settings['weights_mean'],
            l1=not settings['l2_l1'],
            l2=settings['l2_l1'],
            reg_lambda=settings['reg_lambda'],
            grad_clip=settings['grad_clip']
        )

        model.compile(
            loss=MSELoss(),
            optimizer=SGD(
                learning_rate=settings['learning_rate'],
                decay=settings['decay'],
                momentum=settings['momentum'],
                nesterov=settings['nesterov']
            )
        )
        history_train = None
        if choice == choices.ONE_TIME_TRAINING_AND_VALIDATION:

            X_train, X_val, y_train, y_val = split_into_train_test_set(
                X_train, y_train, test_size=settings['validation_size'], random_state=seed_sett
            )

            len_train = len(X_train)

            print(f"Of which {len(X_train)} are training samples, {len(X_val)} are validation samples")

            history_train = model.fit(
                X_train,
                y_train,
                val_set=(X_val, y_val),
                early_stopping=EarlyStopping(patience=settings['patience']) if settings['early_stopping'] else False,
                epochs=settings['epochs'],
                batch_size=round(len_train * settings['batch_size_perc']),
                verbose=True
            )
            show_cup_plots_tr_vl(history_train)

        elif choice == choices.TRAINING_AND_TEST:
            len_train = len(X_train)
            history_train = model.fit(
                X_train,
                y_train,
                epochs=settings['epochs'],
                batch_size=round(len_train * settings['batch_size_perc']),
                verbose=True
            )

            test_loss, test_accuracy = model.predict(X_test, y_test, verbose=False)

            print(f"Internal Test set loss: {test_loss}\n")
            print(f"Internal Test set accuracy: {test_accuracy}\n")
            show_cup_plots_ts(history_train)

        elif choice == choices.FINAL_TRAINING:
            len_train = len(X_train_org)
            history_train = model.fit(
                X_train_org,
                y_train_org,
                epochs=settings['epochs'],
                batch_size=round(len_train * settings['batch_size_perc']),
                verbose=True
            )

            predictions = model.evaluate(X_test_blind)
            print(f"Predictions: {predictions}")

            save_cup_predictions(predictions, "DimAndDistant")

