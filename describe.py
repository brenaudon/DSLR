"""
This script provides functions to generate descriptive statistics for a DataFrame.

The script includes custom implementations for calculating various statistics.

Dependencies:
    - pandas
    - sys
"""

import pandas as pd
import sys

pd.options.display.float_format = '{:.6f}'.format

def custom_count(serie: list[float | int]) -> float:
    """Return the count of non-NaN values in the series.

    This function iterates through the series and counts the number of
    non-NaN values.

    @param serie: The series of values to count.
    @type  serie: list of float or int

    @return: The count of non-NaN values.
    @rtype:  float
    """
    count = 0
    for value in serie:
        if value == value:  # Check for NaN values
            count += 1
    return count

def custom_sum(serie: list[float | int]) -> float:
    """Return the sum of non-NaN values in the series.

    This function iterates through the series and sums the non-NaN values.

    @param serie: The series of values to sum.
    @type  serie: list of float or int

    @return: The sum of non-NaN values.
    @rtype:  float
    """
    total = 0
    for value in serie:
        if value == value:  # Check for NaN values
            total += value
    return total

def custom_mean(serie: list[float | int]) -> float:
    """Return the mean of non-NaN values in the series.

    This function calculates the mean by dividing the sum of non-NaN values
    by their count.

    @param serie: The series of values to calculate the mean.
    @type  serie: list of float or int

    @return: The mean of non-NaN values.
    @rtype:  float
    """
    return custom_sum(serie) / custom_count(serie)

def custom_var(serie: list[float | int]) -> float:
    """Return the variance of non-NaN values in the series.

    This function calculates the variance by summing the squared differences
    from the mean and dividing by the count minus one.

    @param serie: The series of values to calculate the variance.
    @type  serie: list of float or int

    @return: The variance of non-NaN values.
    @rtype:  float
    """
    mean = custom_mean(serie)
    variance = 0
    count = 0
    for value in serie:
        if value == value:  # Check for NaN values
            variance += (value - mean) ** 2
            count += 1
    variance /= (count - 1)
    return variance

def custom_min(serie: list[float | int]) -> float:
    """Return the minimum of non-NaN values in the series.

    This function iterates through the series and finds the minimum non-NaN value.

    @param serie: The series of values to find the minimum.
    @type  serie: list of float or int

    @return: The minimum of non-NaN values.
    @rtype:  float
    """
    min_value = float('inf')
    for value in serie:
        if value == value:  # Check for NaN values
            if value < min_value:
                min_value = value
    return min_value

def custom_quantile(serie: list[float | int], q: float) -> float:
    """Return the q-th quantile of non-NaN values in the series.

    This function calculates the quantile by sorting the non-NaN values and
    finding the appropriate value based on the quantile.

    @param serie: The series of values to calculate the quantile.
    @type  serie: list of float or int
    @param q: The quantile to calculate (between 0 and 1).
    @type  q: float

    @return: The q-th quantile of non-NaN values.
    @rtype:  float
    """
    sorted_series = sorted(value for value in serie if value == value)  # Remove NaN values and sort
    n = len(sorted_series)
    index = q * (n - 1)
    lower = int(index)
    upper = lower + 1
    weight = index - lower
    if upper < n:
        return sorted_series[lower] * (1 - weight) + sorted_series[upper] * weight
    else:
        return sorted_series[lower]

def custom_max(serie: list[float | int]) -> float:
    """Return the maximum of non-NaN values in the series.

    This function iterates through the series and finds the maximum non-NaN value.

    @param serie: The series of values to find the maximum.
    @type  serie: list of float or int

    @return: The maximum of non-NaN values.
    @rtype:  float
    """
    max_value = float('-inf')
    for value in serie:
        if value == value:  # Check for NaN values
            if value > max_value:
                max_value = value
    return max_value

def custom_mad(serie: list[float | int]) -> float:
    """Return the mean absolute deviation of non-NaN values in the series.

    This function calculates the mean absolute deviation by summing the absolute
    deviations from the mean and dividing by the count.

    @param serie: The series of values to calculate the mean absolute deviation.
    @type  serie: list of float or int

    @return: The mean absolute deviation of non-NaN values.
    @rtype:  float
    """
    mean = custom_mean(serie)
    count = custom_count(serie)
    total_deviation = 0
    for value in serie:
        if value == value:  # Check for NaN values
            total_deviation += abs(value - mean)
    mad = total_deviation / count
    return mad

def custom_skew(serie: list[float] | list[int]) -> float:
    """Return the skewness of non-NaN values in the series.

    This function calculates the skewness by summing the cubed deviations from
    the mean, normalized by the standard deviation, and dividing by the count.

    @param serie: The series of values to calculate the skewness.
    @type  serie: list of float or int

    @return: The skewness of non-NaN values.
    @rtype:  float
    """
    mean = custom_mean(serie)
    std_dev = custom_var(serie) ** 0.5
    count = custom_count(serie)
    skewness = 0
    for value in serie:
        if value == value:  # Check for NaN values
            skewness += ((value - mean) / std_dev) ** 3
    skewness /= count
    return skewness

def custom_kurt(serie: list[float | int]) -> float:
    """Return the kurtosis of non-NaN values in the series.

    This function calculates the kurtosis by summing the quartic deviations from
    the mean, normalized by the standard deviation, and dividing by the count,
    then subtracting 3 to get the excess kurtosis.

    @param serie: The series of values to calculate the kurtosis.
    @type  serie: list of float or int

    @return: The kurtosis of non-NaN values.
    @rtype:  float
    """
    mean = custom_mean(serie)
    std_dev = custom_var(serie) ** 0.5
    count = custom_count(serie)
    kurtosis = 0
    for value in serie:
        if value == value:  # Check for NaN values
            kurtosis += ((value - mean) / std_dev) ** 4
    kurtosis = (kurtosis / count) - 3  # Excess kurtosis
    return kurtosis

def describe(df: pd.DataFrame):
    """Generate descriptive statistics for a DataFrame.

    This function calculates various descriptive statistics for each numeric
    column in the DataFrame, including count, mean, standard deviation, min,
    25th percentile, median, 75th percentile, max, range, interquartile range,
    variance, mean absolute deviation, skewness, and kurtosis.

    @param df: The DataFrame to describe.
    @type  df: pd.DataFrame

    @return: A DataFrame containing the descriptive statistics.
    @rtype:  pd.DataFrame
    """
    stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    summary = pd.DataFrame(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'iqr', 'var', 'mad', 'skew', 'kurt'])

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            summary[column] = [
                custom_count(df[column].to_list()), #count
                custom_mean(df[column].to_list()), #mean
                custom_var(df[column].to_list()) ** 0.5, #standard deviation
                custom_min(df[column].to_list()), #min
                custom_quantile(df[column].to_list(), 0.25), #25%
                custom_quantile(df[column].to_list(), 0.5), #50%
                custom_quantile(df[column].to_list(), 0.75), #75%
                custom_max(df[column].to_list()), #max
                custom_max(df[column].to_list()) - custom_min(df[column].to_list()), #range
                custom_quantile(df[column].to_list(), 0.75) - custom_quantile(df[column].to_list(), 0.25), #interquartile range
                custom_var(df[column].to_list()), #variance
                custom_mad(df[column].to_list()), #mean absolute deviation
                custom_skew(df[column].to_list()), #skewness
                custom_kurt(df[column].to_list()) #kurtosis
            ]

    print(summary)

def main():
    """Main function to read a CSV file and generate descriptive statistics.

    This function reads a CSV file specified as a command-line argument,
    generates descriptive statistics for each numeric column in the DataFrame,
    and prints the results.
    """
    if len(sys.argv) != 2:
        print("Usage: python describe.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    describe(df)

if __name__ == '__main__':
    main()
