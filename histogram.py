import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_histograms(df, courses):
    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    colors = {'Gryffindor': 'red', 'Slytherin': 'green', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue'}

    num_courses = len(courses)
    num_cols = 4
    num_rows = (num_courses + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows))

    for i, course in enumerate(courses):
        ax = axes[i // num_cols, i % num_cols]
        for house in houses:
            house_data = df[df['Hogwarts House'] == house][course].dropna()
            ax.hist(house_data, bins=20, alpha=0.5, label=house, color=colors[house])
        ax.set_title(f'Histogram of {course} Scores by Hogwarts House', fontsize=10)
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j // num_cols, j % num_cols])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust the space between histograms
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)

    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

    available_courses = [course for course in courses if course in df.columns]
    plot_histograms(df, available_courses)

if __name__ == '__main__':
    main()