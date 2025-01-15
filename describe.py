import pandas as pd
import sys

pd.options.display.float_format = '{:.6f}'.format

def custom_count(serie: list[float | int]) -> float:
    count = 0
    for value in serie:
        if value == value:  # Check for NaN values
            count += 1
    return count

def custom_sum(serie: list[float | int]) -> float:
    total = 0
    for value in serie:
        if value == value:  # Check for NaN values
            total += value
    return total

def custom_mean(serie: list[float | int]) -> float:
    return custom_sum(serie) / custom_count(serie)

def custom_var(serie: list[float | int]) -> float:
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
    min_value = float('inf')
    for value in serie:
        if value == value:  # Check for NaN values
            if value < min_value:
                min_value = value
    return min_value

def custom_quantile(serie: list[float | int], q: float) -> float:
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
    max_value = float('-inf')
    for value in serie:
        if value == value:  # Check for NaN values
            if value > max_value:
                max_value = value
    return max_value

def custom_mad(serie: list[float | int]) -> float:
    mean = custom_mean(serie)
    count = custom_count(serie)
    total_deviation = 0
    for value in serie:
        if value == value:  # Check for NaN values
            total_deviation += abs(value - mean)
    mad = total_deviation / count
    return mad

def custom_skew(serie: list[float] | list[int]) -> float:
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
    if len(sys.argv) != 2:
        print("Usage: python describe.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    describe(df)

if __name__ == '__main__':
    main()
