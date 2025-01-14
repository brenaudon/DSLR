import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from itertools import combinations

def find_most_correlated_features(df):
    correlation_matrix = df.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    most_correlated = upper_triangle.stack().idxmax()
    return most_correlated

def plot_scatter(df, feature1, feature2):
    houses = df['Hogwarts House'].unique()
    colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}

    plt.figure(figsize=(10, 6))
    for house in houses:
        house_data = df[df['Hogwarts House'] == house]
        plt.scatter(house_data[feature1], house_data[feature2], alpha=0.5, label=house, color=colors[house])
    plt.title(f'Scatter Plot of {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_scatter_pairs(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Index'])
    features = numeric_df.columns
    num_features = len(features)
    num_plots = num_features * (num_features - 1) // 2
    num_cols = 10
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    houses = df['Hogwarts House'].unique()
    colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}

    for ax, (feature1, feature2) in zip(axes, combinations(features, 2)):
        for house in houses:
            house_data = df[df['Hogwarts House'] == house]
            ax.scatter(house_data[feature1], house_data[feature2], alpha=0.5, s=3, label=house, color=colors[house])
        ax.set_title(f'{feature1} vs {feature2}', fontsize=5)
        ax.set_xlabel(feature1, fontsize=5)
        ax.set_ylabel(feature2, fontsize=5)
        ax.grid(True)

    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.subplots_adjust(hspace=1)
    plt.show()

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python scatter_plot.py <path_to_csv> [--all]")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)

    if len(sys.argv) == 3 and sys.argv[2] == '--all':
        plot_all_scatter_pairs(df)
    else:
        numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Index'])
        feature1, feature2 = find_most_correlated_features(numeric_df)
        plot_scatter(df, feature1, feature2)

if __name__ == '__main__':
    main()