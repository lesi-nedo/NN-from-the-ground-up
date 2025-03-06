# Neural Network Implementation for ML Exam

This repository contains a custom implementation of neural networks for classification and regression tasks, created as part of a Machine Learning exam.

## Project Overview

The project implements Multi-Layer Perceptron (MLP) models from scratch without using deep learning frameworks like TensorFlow or PyTorch. It includes all the necessary components for training and evaluating neural networks:

- Custom MLP models for classification and regression
- Various activation functions
- Multiple loss functions (MSE, MAE, Binary Cross-Entropy)
- SGD optimizer with momentum and Nesterov acceleration
- Utilities for hyperparameter tuning and early stopping

## Project Structure

```
ml-exam-nn-2024/
├── activation_functions/     # Activation functions implementation (Softplus, Sigmoid, etc.)
├── configs/                  # Configuration files for different datasets
├── layers/                   # Neural network layer implementations (Dense)
├── loss_functions/           # Loss function implementations (MSE, MAE, BCE)
├── models/                   # Neural network model implementations
│   └── MLP.py                # Multi-Layer Perceptron base class and specialized models
├── optimizer/                # Optimization algorithms
│   └── SGD.py                # Stochastic Gradient Descent with momentum
└── utils/                    # Utility functions
    ├── early_stopping.py     # Early stopping implementation
    ├── grid_search.py        # Hyperparameter search implementation
    └── parse_conf_file.py    # Configuration file parser
```

## Features

- **MLPClassifier**: Neural network for classification tasks
- **MLPRegressor**: Neural network for regression tasks
- **Customizable architectures**: Flexible layer configurations
- **Regularization**: L1 and L2 regularization to prevent overfitting
- **Early stopping**: Prevent overfitting through validation-based stopping
- **Grid search**: Hyperparameter optimization through exhaustive search
- **Weight initialization**: Multiple initialization strategies including Xavier and He initialization
- **Gradient clipping**: Prevent exploding gradients during training

## Usage

### Basic Model Training

```python
from models import MLPClassifier
from activation_functions import Softplus
from loss_functions import BCELogitsLoss
from optimizer import SGD

# Create model
model = MLPClassifier(
    units=[input_size, [0.5, 0.3], output_size],  # Input size, hidden layers as percentage, output size
    activation=Softplus,
    weights_initializer='xavier',
    weights_mean=0.0,
    l2=True,
    reg_lambda=0.001,
    grad_clip=5.0
)

# Compile model
model.compile(
    loss=BCELogitsLoss,
    optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
)

# Train model
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32,
    val_set=(X_val, y_val),
    early_stopping=EarlyStopping(patience=10)
)
```

### Hyperparameter Tuning

```python
from utils import GridSearch

# Define parameter grid
param_grid = {
    'grid_learning_rates': '[0.01, 0.001]',
    'grid_batch_sizes_perc': '[0.1, 0.2]',
    'grid_epochs_list': '[100]',
    'grid_hidden_layers_sizes': '[[0.5, 0.3], [0.7, 0.5]]',
    # ... other hyperparameters
}

# Create grid search
gs = GridSearch(
    objects={
        'model': MLPClassifier,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'activation': Softplus,
        'loss': BCELogitsLoss,
        'learning_alg': SGD,
        'nn_output_size': output_size,
        'patience': 10,
        'early_stopping': True,
        'threshold': 0.95
    },
    param_grid=param_grid
)

# Find best parameters
best_params = gs.search()
```

## Requirements

- NumPy
- tqdm
- psutil
- Python 3.8+