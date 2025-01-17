"""
This script generates pair plots of course scores by Hogwarts House.

The script reads a CSV file containing student data, including their Hogwarts House and scores in various courses. It then generates pair plots for all numeric features, showing the distribution and relationships between scores for each house.

Dependencies:
    - pandas
    - seaborn
    - matplotlib
    - sys
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def plot_pairplot(df: pd.DataFrame):
    """Plot pair plots of course scores by Hogwarts House.

    This function generates pair plots for all numeric features, showing the distribution
    and relationships between scores for each Hogwarts House.

    @param df: The DataFrame containing student data.
    @type  df: pd.DataFrame
    """
    # Drop non-numeric columns and the 'Index' column if it exists
    numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Index'], errors='ignore')

    # Add the 'Hogwarts House' column back to the numeric dataframe for coloring
    numeric_df['Hogwarts House'] = df['Hogwarts House']

    # Create the pair plot
    g = sns.pairplot(numeric_df, hue='Hogwarts House', palette={'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}, plot_kws={'s': 3}, dropna=True)

    # Adjust the subplot parameters to give more space for the legend and column titles
    g.fig.subplots_adjust(bottom=0.05, left=0.1, right=0.93, top=0.95, hspace=0.2)

    # Adjust the y-axis labels
    for ax in g.axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel(), rotation=0, ha='right', fontsize=7)

    g.fig.suptitle("Pair Plot of Hogwarts House Data")  # Add a title to the plot
    plt.show()

def main():
    """Main function to read a CSV file and generate pair plots.

    This function reads a CSV file specified as a command-line argument,
    generates pair plots for each numeric column in the DataFrame,
    and displays the results.
    """
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    plot_pairplot(df)

if __name__ == '__main__':
    main()