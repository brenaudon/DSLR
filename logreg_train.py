import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Initialize softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability adjustment
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Cost function for logistic regression
def compute_cost(y, y_pred):
    m = y.shape[0]
    return -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Gradient calculation for logistic regression
def compute_gradient(X, y, y_pred):
    m = X.shape[0]
    return 1/m * np.dot(X.T, (y_pred - y))

# Gradient descent
def gradient_descent(X, y, num_classes, learning_rate=0.1, num_iterations=1000):
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
# Stochastic gradient descent
def stochastic_gradient_descent(X, y, num_classes, learning_rate=0.1, num_iterations=1000):
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

# Mini-batch gradient descent
def mini_batch_gradient_descent(X, y, num_classes, learning_rate=0.1, num_iterations=1000, batch_size=3):
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

# Fractionned batch gradient descent
def fractionned_batch_gradient_descent(X, y, num_classes, learning_rate=0.1, num_iterations=1000, fraction_size=4):
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
        method = 'gd'
    else:
        method = sys.argv[1]

    # Load the dataset
    dataset_train = pd.read_csv('datasets/dataset_train.csv')

    # Read the list of courses from config.csv
    config = pd.read_csv('config.csv')
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
    np.savetxt('theta.csv', theta, delimiter=',')


if __name__ == '__main__':
    main()