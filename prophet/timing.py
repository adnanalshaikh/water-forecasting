import pandas as pd

x = pd.read_csv('../results/run_adaptive.csv')
t = [x.iloc[i]['exec_time'] for i in range(29)]

# Example time series (Pandas Series)
s = pd.Series(t)  

# Compute statistics
stats = {
    'median': s.median(),
    'mean': s.mean(),
    'stdev': s.std(ddof=1),  # Sample standard deviation (Bessel's correction)
    'IQ25': s.quantile(0.25),
    'IQ75': s.quantile(0.75),
    'IQR': s.quantile(0.75) - s.quantile(0.25),  # Interquartile Range
}

print(stats)


