"""
This script predict class labels using a trained logistic regression model.

The script includes a function to predict class labels based on input features and trained weights.

Dependencies:
    - numpy
    - pandas
    - logreg_train.py
"""

import numpy as np
import pandas as pd
from logreg_train import softmax

def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Predict the class labels for the given input features using the trained weights.

    This function calculates the logits (raw, unnormalized scores output by a model before applying any activation function like softmax)
    by performing a dot product between the input features and the trained weights,
    applies the softmax function to obtain the probabilities, and then uses np.argmax to determine the predicted class labels.

    @param X: The input features.
    @type  X: np.ndarray
    @param theta: The trained weights.
    @type  theta: np.ndarray

    @return: The predicted class labels.
    @rtype:  np.ndarray
    """
    logits = np.dot(X, theta)
    probabilities = softmax(logits)
    return np.argmax(probabilities, axis=1)

def main():
    """Main function to read a test CSV file, load trained weights, make predictions, and save them to a file."""
    # Load the test dataset
    dataset_test = pd.read_csv('datasets/dataset_test.csv')

    # Read the list of courses from config.csv
    config = pd.read_csv('config.csv')
    courses = config['courses'].tolist()

    # Extract the feature variables (columns listed in config.csv)
    X_test = dataset_test[courses]

    # Fill missing values with the mean of the column
    X_test = X_test.fillna(X_test.mean())

    # Normalize features for better gradient descent performance
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

    # Load the trained weights
    theta = np.loadtxt('theta.csv', delimiter=',')

    # Make predictions
    y_pred = predict(X_test, theta)

    # Define the reverse mapping
    house_mapping = {
        0: 'Gryffindor',
        1: 'Slytherin',
        2: 'Ravenclaw',
        3: 'Hufflepuff'
    }

    # Map predictions to house names
    y_pred_houses = [house_mapping[pred] for pred in y_pred]

    # Create the output DataFrame
    output_df = pd.DataFrame({
        'Index': np.arange(len(y_pred_houses)),
        'Hogwarts House': y_pred_houses
    })

    # Save the predictions to houses.csv
    output_df.to_csv('houses.csv', index=False)

if __name__ == '__main__':
    main()