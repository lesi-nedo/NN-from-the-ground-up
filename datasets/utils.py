import pandas as pd
from typing import Tuple
from utils import style, write_to_csv


def select_monk_dataset():
    number = input(f"{style.GREEN}Which monk dataset do you want to use? {style.BLUE}(1, 2, 3){style.RESET}: ")
    try:
        number = int(number)
        if number not in [1, 2, 3]:
            raise ValueError
    except ValueError:
        print(f"{style.RED}Invalid input. Please enter a number between 1 and 3.{style.RESET}")
        exit()
    dataset = f'monks-{number}'
    return dataset


def save_cup_predictions(predictions, team_name):
    answer = input(f"{style.GREEN}Save predictions to csv file? {style.BLUE}[yes/no]\n{style.RESET}")
    if answer.lower() == 'yes':
        write_to_csv(f'./{team_name}_ML-cup23-ts.csv', 'Oleksiy Nedobiychuk',
                     team_name, predictions)
        print(f"{style.GREEN}Predictions saved to predictions.csv{style.RESET}")
    else:
        print(f"{style.RED}Predictions not saved{style.RESET}")


def one_of_k_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
        Applies one-hot encoding to the DataFrame, converting categorical columns into binary columns.

        Parameters:
        - df (pd.DataFrame): DataFrame with categorical data to be one-hot encoded.

        Returns:
        - pd.DataFrame: DataFrame with one-hot encoded columns, all entries as floats.
    """
    # Convert all entries in the DataFrame to strings
    newdf = df.map(str)

    # Apply one-hot encoding to the DataFrame
    # Converts categorical columns into binary columns with appropriate prefixes
    newdf = pd.get_dummies(newdf, columns=df.columns)

    # Convert all entries in the DataFrame to float
    newdf = newdf.map(float)

    return newdf


def read_monk_dataset(base_path: str, shuffle: bool = False, random_state: int = 42) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
        Reads and preprocesses the MONK dataset from the given base path.

        Parameters:
        - base_path (str): The base path to the dataset files (excluding '.train' and '.test' extensions).
        - shuffle (bool): If True, shuffles the training data.
        - random_state (int): Random seed for shuffling the training data.
        Returns:
        - Tuple containing:
            - train_examples (pd.DataFrame): DataFrame with preprocessed training examples.
            - train_labels (pd.Series): Series with rescaled training labels.
            - test_examples (pd.DataFrame): DataFrame with preprocessed test examples.
            - test_labels (pd.Series): Series with rescaled test labels.
    """

    # Construct file paths for training and test data
    train_path = base_path + '.train'
    test_path = base_path + '.test'

    # Read training and test data
    train_data = pd.read_csv(train_path, sep=" ", header=None)
    test_data = pd.read_csv(test_path, sep=" ", header=None)

    if shuffle:
        train_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_data = test_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Extract examples and labels
    train_examples = one_of_k_encode(train_data.iloc[:, 2:8])
    train_labels = train_data.iloc[:, 1]

    test_examples = one_of_k_encode(test_data.iloc[:, 2:8])
    test_labels = test_data.iloc[:, 1]

    print(
        f"Train examples: {train_examples.shape} - Train labels: {train_labels.shape} - "
        f"Test examples: {test_examples.shape} - Test labels: {test_labels.shape}")

    return train_examples, train_labels.squeeze(), test_examples, test_labels.squeeze()


def read_cup_dataset(base_path: str, shuffle: bool = False, random_state: int = 42) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame]:
    """
        Reads and preprocesses the CUP dataset from the given base path.

        Parameters:
        - base_path (str): The base path to the dataset files (excluding '.train' and '.test' extensions).
        - shuffle (bool): If True, shuffles the training data.
        - random_state (int): Random seed for shuffling the training data.

        Returns:
        - Tuple containing:
            - train_examples (pd.DataFrame): DataFrame with preprocessed training examples.
            - train_labels (pd.Series): Series with rescaled training labels.
            - test_examples (pd.DataFrame): DataFrame with preprocessed test examples.
    """

    # Construct file paths for training and test data
    train_path = base_path + '.train.csv'
    test_path = base_path + '.test.csv'

    # Read training data
    train_data = pd.read_csv(train_path, header=None, sep=",")
    test_data = pd.read_csv(test_path, header=None, sep=",")
    if shuffle:
        train_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_examples = train_data.iloc[:, 1:-3]

    train_labels = train_data.iloc[:, -3:]

    # Read test data
    test_examples = test_data.iloc[:, 1:]

    print(
        f"Train examples: {train_examples.shape} - Train labels: {train_labels.shape} - "
        f"Test examples: {test_examples.shape}")

    return train_examples, train_labels.squeeze(), test_examples
