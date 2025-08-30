import pandas as pd

# 635   150 50
# 462   200 40   0 rgs
# 559   200 40   auto
# 490   200 40   auto 
# 532   200 40   auto
# 535   200 40   1
# 478   200 40   1
# 460   200 40   1

# 519   200 40   2
# 506   200 40   2
# 518   200 40   2

# 528   200 40   3
# 562   200 40   3
# 563   200 40   3 

# 530   200 40   4
# 522   200 49   5

 #517   200 40   6
# 606   200 40   6
# 546   200 40   6

# 582   200 40   7
# 589   200 40   7


x = pd.read_csv('../results/exp31_auto_regs_200_40_time.csv')
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


