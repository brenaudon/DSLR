"""
This script trains a logistic regression model. It can use various gradient descent methods.

The script reads a training CSV file, loads the feature variables and target variable,
trains the logistic regression model using the specified gradient descent method,
and saves the learned parameters to a file.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - sys
    - os
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of the input logits.

    @param logits: The input logits.
    @type  logits: np.ndarray

    @return: The softmax probabilities.
    @rtype:  np.ndarray
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability adjustment
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def compute_cost(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the cost for logistic regression.

    @param y: The true labels.
    @type  y: np.ndarray
    @param y_pred: The predicted labels.
    @type  y_pred: np.ndarray

    @return: The computed cost.
    @rtype:  float
    """
    m = y.shape[0]
    return -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def compute_gradient(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the gradient for logistic regression.

    @param X: The input features.
    @type  X: np.ndarray
    @param y: The true labels.
    @type  y: np.ndarray
    @param y_pred: The predicted labels.
    @type  y_pred: np.ndarray

    @return: The computed gradient.
    @rtype:  np.ndarray
    """
    m = X.shape[0]
    return 1/m * np.dot(X.T, (y_pred - y))


def gradient_descent(X: np.ndarray, y: np.ndarray, num_classes: int, learning_rate: float = 0.1, num_iterations: int = 1000) -> tuple[np.ndarray, list[float]]:
    """
    Perform gradient descent to learn the model parameters.

    @param X: The input features.
    @type  X: np.ndarray
    @param y: The true labels.
    @type  y: np.ndarray
    @param num_classes: The number of classes.
    @type  num_classes: int
    @param learning_rate: The learning rate for gradient descent.
    @type  learning_rate: float
    @param num_iterations: The number of iterations for gradient descent.
    @type  num_iterations: int

    @return: The learned parameters and the cost history.
    @rtype:  tuple[np.ndarray, list[float]]
    """
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, num_classes)  # Random initialization
    cost_history = []

    # Convert y to one-hot encoding
    y_onehot = np.eye(num_classes)[y]

    for _ in range(num_iterations):
        # Forward pass
        logits = np.dot(X, theta)
        y_pred = softmax(logits)

        # Compute cost
        cost = compute_cost(y_onehot, y_pred)
        cost_history.append(cost)

        # Compute gradient
        gradient = compute_gradient(X, y_onehot, y_pred)

        # Update parameters
        theta -= learning_rate * gradient

    return theta, cost_history


# BONUS ---------------------------------------------------------------------------

def stochastic_gradient_descent(X: np.ndarray, y: np.ndarray, num_classes: int, learning_rate: float = 0.1, num_iterations: int = 1000) -> tuple[np.ndarray, list[float]]:
    """
    Perform stochastic gradient descent (based on one random element for each iteration) to learn the model parameters.

    @param X: The input features.
    @type  X: np.ndarray
    @param y: The true labels.
    @type  y: np.ndarray
    @param num_classes: The number of classes.
    @type  num_classes: int
    @param learning_rate: The learning rate for gradient descent.
    @type  learning_rate: float
    @param num_iterations: The number of iterations for gradient descent.
    @type  num_iterations: int

    @return: The learned parameters and the cost history.
    @rtype:  tuple[np.ndarray, list[float]]
    """
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, num_classes)  # Random initialization
    cost_history = []

    # Convert y to one-hot encoding
    y_onehot = np.eye(num_classes)[y]

    for _ in range(num_iterations):
        # Randomly select an index
        idx = np.random.randint(n_samples)
        X_rand = X[idx:idx+1]
        y_rand = y_onehot[idx:idx+1]

        # Forward pass
        logits = np.dot(X_rand, theta)
        y_pred = softmax(logits)

        # Compute cost
        cost = compute_cost(y_rand, y_pred)
        cost_history.append(cost)

        # Compute gradient
        gradient = compute_gradient(X_rand, y_rand, y_pred)

        # Update parameters
        theta -= learning_rate * gradient

    return theta, cost_history


def mini_batch_gradient_descent(X: np.ndarray, y: np.ndarray, num_classes: int, learning_rate: float = 0.1, num_iterations: int = 1000, batch_size: int = 3) -> tuple[np.ndarray, list[float]]:
    """
    Perform mini-batch gradient descent (based on batch_size random elements for each iteration) to learn the model parameters.

    @param X: The input features.
    @type  X: np.ndarray
    @param y: The true labels.
    @type  y: np.ndarray
    @param num_classes: The number of classes.
    @type  num_classes: int
    @param learning_rate: The learning rate for gradient descent.
    @type  learning_rate: float
    @param num_iterations: The number of iterations for gradient descent.
    @type  num_iterations: int
    @param batch_size: The size of each mini-batch.
    @type  batch_size: int

    @return: The learned parameters and the cost history.
    @rtype:  tuple[np.ndarray, list[float]]
    """
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, num_classes)  # Random initialization
    cost_history = []

    # Convert y to one-hot encoding
    y_onehot = np.eye(num_classes)[y]

    for i in range(num_iterations):
        X_batch = np.zeros((batch_size, n_features))
        y_batch = np.zeros((batch_size, num_classes))
        for j in range(batch_size):
            # Randomly select an index
            idx = np.random.randint(n_samples)
            X_rand = X[idx:idx+1]
            y_rand = y_onehot[idx:idx+1]

            X_batch[j] = X_rand
            y_batch[j] = y_rand

        # Forward pass
        logits = np.dot(X_batch, theta)
        y_pred = softmax(logits)

        # Compute cost
        cost = compute_cost(y_batch, y_pred)
        cost_history.append(cost)

        # Compute gradient
        gradient = compute_gradient(X_batch, y_batch, y_pred)

        # Update parameters
        theta -= learning_rate * gradient

    return theta, cost_history


def fractionned_batch_gradient_descent(X: np.ndarray, y: np.ndarray, num_classes: int, learning_rate: float = 0.1, num_iterations: int = 1000, fraction_size: int = 4) -> tuple[np.ndarray, list[float]]:
    """
    Perform fractionned batch gradient descent (based on a fraction of the dataset for each iteration) to learn the model parameters.

    @param X: The input features.
    @type  X: np.ndarray
    @param y: The true labels.
    @type  y: np.ndarray
    @param num_classes: The number of classes.
    @type  num_classes: int
    @param learning_rate: The learning rate for gradient descent.
    @type  learning_rate: float
    @param num_iterations: The number of iterations for gradient descent.
    @type  num_iterations: int
    @param fraction_size: The size of each fraction.
    @type  fraction_size: int

    @return: The learned parameters and the cost history.
    @rtype:  tuple[np.ndarray, list[float]]
    """
    n_samples, n_features = X.shape
    theta = np.random.randn(n_features, num_classes)  # Random initialization
    cost_history = []

    # Convert y to one-hot encoding
    y_onehot = np.eye(num_classes)[y]

    for i in range(num_iterations):
        # Random offset between 0 and batch_fraction so we don't always take the same fraction of the dataset
        offset = np.random.randint(fraction_size)
        X_batch = X[offset::fraction_size]
        y_batch = y_onehot[offset::fraction_size]

        # Forward pass
        logits = np.dot(X_batch, theta)
        y_pred = softmax(logits)

        # Compute cost
        cost = compute_cost(y_batch, y_pred)
        cost_history.append(cost)

        # Compute gradient
        gradient = compute_gradient(X_batch, y_batch, y_pred)

        # Update parameters
        theta -= learning_rate * gradient

    return theta, cost_history

# ---------------------------------------------------------------------------------

def main():
    """Main function to read a CSV file, train a logistic regression model, and print the learned parameters.

    This function reads a CSV file specified as a command-line argument,
    trains a logistic regression model using gradient descent,
    and prints the learned parameters.
    """
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py <path_to_dataset_train> [method]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'gd'

    # Load the dataset
    dataset_train = pd.read_csv(dataset_path)

    # Construct the path to config.csv relative to the script's location
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config.csv')

    # Read the list of courses from config.csv
    config = pd.read_csv(config_path)
    courses = config['courses'].tolist()

    # Extract the feature variables (columns listed in config.csv)
    X = dataset_train[courses]

    # Fill missing values with the mean of the column
    X = X.fillna(X.mean())

    # Normalize features for better gradient descent performance
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Extract the target variable (second column)
    y = dataset_train.iloc[:, 1]

    # Define the mapping
    house_mapping = {
        'Gryffindor': 0,
        'Slytherin': 1,
        'Ravenclaw': 2,
        'Hufflepuff': 3
    }

    y = list(map(lambda house: house_mapping[house], y))

    num_classes = len(np.unique(y))  # Number of houses (classes)

    # Train the model
    if method == 'gd':
        learning_rate = 0.1
        num_iterations = 1500
        theta, cost_history = gradient_descent(X, y, num_classes, learning_rate, num_iterations)
    elif method == 'sgd':
        learning_rate = 0.1
        num_iterations = 1500
        theta, cost_history = stochastic_gradient_descent(X, y, num_classes, learning_rate, num_iterations)
    elif method == 'mbgd':
        learning_rate = 0.1
        num_iterations = 1500
        batch_size = 3
        theta, cost_history = mini_batch_gradient_descent(X, y, num_classes, learning_rate, num_iterations, batch_size)
    elif method == 'fbgd':
        learning_rate = 0.1
        num_iterations = 1500
        fraction_size = 4
        theta, cost_history = fractionned_batch_gradient_descent(X, y, num_classes, learning_rate, num_iterations, fraction_size)
    else:
        print("Invalid method. Use 'gd' for gradient descent, 'sgd' for stochastic gradient descent, or 'mbgd' for mini-batch gradient descent.")
        return

    # Plot cost history
    plt.plot(cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost History over Iterations')
    plt.show()

    # Save the learned parameters to a file
    theta_path = os.path.join(script_dir, 'theta.csv')
    np.savetxt(theta_path, theta, delimiter=',')


if __name__ == '__main__':
    main()