import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def plot_pairplot(df: pd.DataFrame):
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
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    plot_pairplot(df)

if __name__ == '__main__':
    main()