"""
This script calculates the training accuracy of a logistic regression model.

The script reads a training CSV file, loads the trained weights, makes predictions using the logistic regression model, and calculates the accuracy of the predictions.

Dependencies:
    - numpy
    - pandas
    - logreg_predict.py
    - os
    - sys
"""

import numpy as np
import pandas as pd
from logreg_predict import predict
import os
import sys

def main():
    """Main function to read the training CSV file, load trained weights, make predictions, and calculate accuracy."""
    if len(sys.argv) < 2:
        print("Usage: python precision.py <path_to_dataset_train>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Load the training dataset
    dataset_train = pd.read_csv(dataset_path)

    # Construct the path to config.csv relative to the script's location
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config.csv')

    # Read the list of courses from config.csv
    config = pd.read_csv(config_path)
    courses = config['courses'].tolist()

    # Extract the feature variables (columns listed in config.csv)
    X_train = dataset_train[courses]

    # Fill missing values with the mean of the column
    X_train = X_train.fillna(X_train.mean())

    # Normalize features for better gradient descent performance
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)

    # Extract the target variable (second column)
    y_train = dataset_train.iloc[:, 1]

    # Define the mapping
    house_mapping = {
        'Gryffindor': 0,
        'Slytherin': 1,
        'Ravenclaw': 2,
        'Hufflepuff': 3
    }

    y_train = list(map(lambda house: house_mapping[house], y_train))

    # Load the trained weights
    theta_path = os.path.join(script_dir, 'theta.csv')
    theta = np.loadtxt(theta_path, delimiter=',')

    # Make predictions
    y_pred = predict(X_train, theta)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_train)
    print(f"Training accuracy: {accuracy}")

if __name__ == '__main__':
    main()