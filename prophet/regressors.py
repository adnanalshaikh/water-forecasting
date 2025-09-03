import pandas as pd
import numpy as np

from pandas.tseries.offsets import MonthEnd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from data_loader import get_humidity_data, get_rainfall_data, get_temp_data

def add_fridays_per_month_reg(df, which_regressors=None):
    # Make a copy    
    if which_regressors is not None and not which_regressors.get('fridays_per_month_regressor', True):
        return df  # Skip this regressor if explicitly set to False
    
    df_copy = df.copy()
    
    # Create comprehensive date range
    start_date = df_copy['ds'].min()
    end_date = df_copy['ds'].max() + MonthEnd(1)  # Extra buffer
    month_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Count Fridays per month
    fridays_per_month = month_dates[month_dates.dayofweek == 4].to_period('M').value_counts().sort_index()
    
    # Ensure all months have a value
    all_months = df_copy['ds'].dt.to_period('M').unique()
    for month in all_months:
        if month not in fridays_per_month.index:
            fridays_per_month[month] = 0
    
    # Create regressor and fill any NaNs
    df_copy['fridays_per_month_regressor'] = df_copy['ds'].dt.to_period('M').map(fridays_per_month)
    df_copy['fridays_per_month_regressor'] = df_copy['fridays_per_month_regressor'].fillna(0)
    
    return df_copy

def add_monthly_holidays_reg(df, which_regressors=None):
    """
    Create monthly holiday indicators for a DataFrame with monthly data.
    
    Parameters:
    - df: DataFrame with 'ds' column (datetime)
    - which_regressors: Dictionary with regressor names as keys and boolean values
    
    Returns:
    - DataFrame with additional holiday indicator columns
    """
    # Check if any holiday regressors should be included
    if which_regressors is not None:
        ramadan_enabled = which_regressors.get('ramadan_month_regressor', True)
        eid_alfitr_enabled = which_regressors.get('eid_alfitr_month_regressor', True)
        eid_aladha_enabled = which_regressors.get('eid_aladha_month_regressor', True)
        
        # If all are disabled, return the original dataframe
        if not (ramadan_enabled or eid_alfitr_enabled or eid_aladha_enabled):
            return df
    else:
        # If which_regressors not provided, enable all
        ramadan_enabled = eid_alfitr_enabled = eid_aladha_enabled = True
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Extract year and month
    year = df_copy['ds'].dt.year
    month = df_copy['ds'].dt.month
    
    # Define holiday months for each year
    ramadan_months = {
        2019: 5, 2020: 4, 2021: 4, 2022: 4, 2023: 3, 2024: 3, 2025: 2
    }
    
    eid_alfitr_months = {
        2019: 6, 2020: 5, 2021: 5, 2022: 5, 2023: 4, 2024: 4, 2025: 3
    }
    
    eid_aladha_months = {
        2019: 8, 2020: 7, 2021: 7, 2022: 7, 2023: 6, 2024: 6, 2025: 5
    }
    
    # Add each regressor if enabled
    if ramadan_enabled:
        df_copy['ramadan_month_regressor'] = 0
        for yr, mo in ramadan_months.items():
            mask = (year == yr) & (month == mo)
            df_copy.loc[mask, 'ramadan_month_regressor'] = 1
    
    if eid_alfitr_enabled:
        df_copy['eid_alfitr_month_regressor'] = 0
        for yr, mo in eid_alfitr_months.items():
            mask = (year == yr) & (month == mo)
            df_copy.loc[mask, 'eid_alfitr_month_regressor'] = 1
    
    if eid_aladha_enabled:
        df_copy['eid_aladha_month_regressor'] = 0
        for yr, mo in eid_aladha_months.items():
            mask = (year == yr) & (month == mo)
            df_copy.loc[mask, 'eid_aladha_month_regressor'] = 1
    
    return df_copy

def add_temperature_reg(df, which_regressors=None):
    """
    Add temperature-related regressors to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'ds' column
    - which_regressors: Dictionary with regressor names as keys and boolean values
    
    Returns:
    - DataFrame with added temperature regressors
    """
    # Check if temperature regressors should be included
    if which_regressors is not None:
        temp_enabled = which_regressors.get('temperature_regressor', True)
        anomaly_enabled = which_regressors.get('temp_anomaly_regressor', True)
        
        # If both are disabled, return the original dataframe
        if not (temp_enabled or anomaly_enabled):
            return df
    else:
        # If which_regressors not provided, enable all
        temp_enabled = anomaly_enabled = True
    
    df_copy = df.copy()
    
    # Get temperature data
    temp_data = get_temp_data()
    
    # Calculate monthly temperature averages
    monthly_avgs = {}
    for month in range(1, 13):
        month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                      'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][month-1]
        if month_name in temp_data:
            values = [v for v in temp_data[month_name] if v is not None]
            if values:
                monthly_avgs[month] = sum(values) / len(values)
    
    # Add temperature values if enabled
    if temp_enabled:
        temps = []
        for date in df_copy['ds']:
            year, month = date.year, date.month
            
            temp_value = None
            # Try to get actual temperature first
            month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][month-1]
            if month_name in temp_data:
                year_idx = year - 2019  # Base year
                if 0 <= year_idx < len(temp_data[month_name]):
                    temp_value = temp_data[month_name][year_idx]
            
            # If no actual value, use monthly average
            if temp_value is None and month in monthly_avgs:
                temp_value = monthly_avgs[month]
                
            temps.append(temp_value)
        
        df_copy['temperature_regressor'] = temps
        df_copy['temperature_regressor'] = df_copy['temperature_regressor'].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Add temperature anomaly if enabled
    if anomaly_enabled and (temp_enabled or 'temperature_regressor' in df_copy.columns):
        # If temperature regressor was not just added but exists in the df
        if not temp_enabled and 'temperature_regressor' in df_copy.columns:
            pass  # Use existing temperature column
        elif not temp_enabled:
            # Need to temporarily calculate temperature values for anomaly calculation
            temps = []
            for date in df_copy['ds']:
                year, month = date.year, date.month
                
                temp_value = None
                month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][month-1]
                if month_name in temp_data:
                    year_idx = year - 2019
                    if 0 <= year_idx < len(temp_data[month_name]):
                        temp_value = temp_data[month_name][year_idx]
                
                if temp_value is None and month in monthly_avgs:
                    temp_value = monthly_avgs[month]
                    
                temps.append(temp_value)
            
            temp_values = pd.Series(temps, index=df_copy.index).fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            temp_values = df_copy['temperature_regressor']
        
        # Calculate anomalies
        temp_anomalies = []
        for i, (_, row) in enumerate(df_copy.iterrows()):
            month = row['ds'].month
            if month in monthly_avgs:
                if temp_enabled:
                    temp_value = row['temperature_regressor']
                else:
                    temp_value = temp_values.iloc[i]
                
                temp_anomalies.append(temp_value - monthly_avgs[month])
            else:
                temp_anomalies.append(0)  # Default to no anomaly
        
        df_copy['temp_anomaly_regressor'] = temp_anomalies
        df_copy['temp_anomaly_regressor'] = df_copy['temp_anomaly_regressor'].fillna(0)
    
    return df_copy

def add_humidity_rainfall_reg(df, which_regressors=None):
    """Add humidity and rainfall regressors to the DataFrame."""
    if which_regressors is not None:
        humidity_enabled = which_regressors.get('humidity_regressor', True)
        rainfall_enabled = which_regressors.get('rainfall_regressor', True)
        
        if not (humidity_enabled or rainfall_enabled):
            return df
    else:
        humidity_enabled = True
        rainfall_enabled = True
    
    df_copy = df.copy()
    
    # Add humidity data if enabled
    if humidity_enabled:
        # Get humidity data (already has averages and filled missing values)
        humidity_data = get_humidity_data()
        
        # Add humidity values based on dates
        humidities = []
        for date in df_copy['ds']:
            year, month = date.year, date.month
            
            # Get the month name
            month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][month-1]
            
            # Get the year index (assuming data starts from 2019)
            year_idx = year - 2019
            
            # Get the humidity value if available, otherwise use the last available value
            if 0 <= year_idx < len(humidity_data[month_name]):
                humidity_value = humidity_data[month_name][year_idx]
            else:
                # If we're forecasting beyond our data, use the last available value
                humidity_value = humidity_data[month_name][-1]
                
            humidities.append(humidity_value)
        
        df_copy['humidity_regressor'] = humidities
    
    # Add rainfall data if enabled  
    if rainfall_enabled and 'get_rainfall_data' in globals():
        # Similar implementation to humidity
        rainfall_data = get_rainfall_data()

        rainfall = []
        for date in df_copy['ds']:
            year, month = date.year, date.month

            month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][month-1]
            
            # Get the year index (assuming data starts from 2019)
            year_idx = year - 2019
            if 0 <= year_idx < len(rainfall_data[month_name]):
                rainfall_value = rainfall_data[month_name][year_idx]
            else:
                # If we're forecasting beyond our data, use the last available value
                rainfall_value = rainfall_data[month_name][-1]
                
            rainfall.append(rainfall_value)
        
        df_copy['rainfall_regressor'] = rainfall
        
    return df_copy

def add_monthly_seasonality_reg(df, which_regressors=None):
    """
    Add Fourier terms for monthly seasonality to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'ds' column
    - which_regressors: Dictionary with regressor names as keys and boolean values
    
    Returns:
    - DataFrame with added monthly seasonality regressors
    """
    # Check if any seasonality regressors should be included
    if which_regressors is not None:
        sin_enabled = which_regressors.get('month_sin_regressor', True)
        cos_enabled = which_regressors.get('month_cos_regressor', True)
        sin2_enabled = which_regressors.get('month_sin2_regressor', True)
        cos2_enabled = which_regressors.get('month_cos2_regressor', True)
        
        # If all are disabled, return the original dataframe
        if not (sin_enabled or cos_enabled or sin2_enabled or cos2_enabled):
            return df
    else:
        # If which_regressors not provided, enable all
        sin_enabled = cos_enabled = sin2_enabled = cos2_enabled = True
    
    df_copy = df.copy()
    
    # Extract month
    month = df_copy['ds'].dt.month
    
    # Add first harmonic terms if enabled
    if sin_enabled:
        df_copy['month_sin_regressor'] = np.sin(2 * np.pi * month/12)
    
    if cos_enabled:
        df_copy['month_cos_regressor'] = np.cos(2 * np.pi * month/12)
    
    # Add second harmonic terms if enabled
    if sin2_enabled:
        df_copy['month_sin2_regressor'] = np.sin(4 * np.pi * month/12)
    
    if cos2_enabled:
        df_copy['month_cos2_regressor'] = np.cos(4 * np.pi * month/12)
    
    return df_copy

def add_seasonal_indicators_reg(df, which_regressors=None):
    """
    Add seasonal indicator variables to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'ds' column
    - which_regressors: Dictionary with regressor names as keys and boolean values
    
    Returns:
    - DataFrame with added seasonal indicators
    """
    # Check if any seasonal indicators should be included
    if which_regressors is not None:
        winter_enabled = which_regressors.get('winter_regressor', True)
        spring_enabled = which_regressors.get('spring_regressor', True)
        summer_enabled = which_regressors.get('summer_regressor', True)
        fall_enabled = which_regressors.get('fall_regressor', True)
        
        # If all are disabled, return the original dataframe
        if not (winter_enabled or spring_enabled or summer_enabled or fall_enabled):
            return df
    else:
        # If which_regressors not provided, enable all
        winter_enabled = spring_enabled = summer_enabled = fall_enabled = True
    
    df_copy = df.copy()
    
    # Define seasons based on month
    seasons = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    season = df_copy['ds'].dt.month.map(seasons)
    
    # Add only the enabled indicator variables
    if winter_enabled:
        df_copy['winter_regressor'] = (season == 'winter').astype(int)
    
    if spring_enabled:
        df_copy['spring_regressor'] = (season == 'spring').astype(int)
    
    if summer_enabled:
        df_copy['summer_regressor'] = (season == 'summer').astype(int)
    
    if fall_enabled:
        df_copy['fall_regressor'] = (season == 'fall').astype(int)
    
    return df_copy

def add_moving_avg_reg(df, which_regressors=None, mode='train'):
    """
    Add moving average regressors to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'ds' and 'y' columns
    - which_regressors: Dictionary with regressor names as keys and boolean values
    - mode: 'train' to include y-dependent regressors, 'forecast' to exclude them
    
    Returns:
    - DataFrame with added moving average regressors
    """
    # Check if we're in forecast mode - never add y-dependent regressors in forecast mode
    if mode != 'train' or 'y' not in df.columns:
        return df
    
    # Check if any moving average regressors should be included
    if which_regressors is not None:
        ma3_enabled = which_regressors.get('y_ma3_regressor', True)
        ma6_enabled = which_regressors.get('y_ma6_regressor', True)
        ma12_enabled = which_regressors.get('y_ma12_regressor', True)
        ewma3_enabled = which_regressors.get('y_ewma3_regressor', True)
        ewma6_enabled = which_regressors.get('y_ewma6_regressor', True)
        
        # If all are disabled, return the original dataframe
        if not (ma3_enabled or ma6_enabled or ma12_enabled or ewma3_enabled or ewma6_enabled):
            return df
    else:
        # If which_regressors not provided, enable all
        ma3_enabled = ma6_enabled = ma12_enabled = ewma3_enabled = ewma6_enabled = True
    
    df_copy = df.copy()
    
    # Add only the enabled moving averages
    if ma3_enabled and len(df_copy) >= 3:
        df_copy['y_ma3_regressor'] = df_copy['y'].rolling(window=3, min_periods=1).mean()
    
    if ma6_enabled and len(df_copy) >= 6:
        df_copy['y_ma6_regressor'] = df_copy['y'].rolling(window=6, min_periods=1).mean()
    
    if ma12_enabled and len(df_copy) >= 12:
        df_copy['y_ma12_regressor'] = df_copy['y'].rolling(window=12, min_periods=1).mean()
    
    # Add only the enabled exponentially weighted moving averages
    if ewma3_enabled and len(df_copy) >= 3:
        df_copy['y_ewma3_regressor'] = df_copy['y'].ewm(span=3, adjust=False).mean()
    
    if ewma6_enabled and len(df_copy) >= 6:
        df_copy['y_ewma6_regressor'] = df_copy['y'].ewm(span=6, adjust=False).mean()
        
    # Fill any NaN values in the regressors
    for col in df_copy.columns:
        if col.endswith('_regressor'):
            df_copy[col] = df_copy[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_copy

def add_std_reg(df, which_regressors=None, mode='train'):
    """
    Add standard deviation based regressors to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'ds' and 'y' columns
    - which_regressors: Dictionary with regressor names as keys and boolean values
    - mode: 'train' to include y-dependent regressors, 'forecast' to exclude them
    
    Returns:
    - DataFrame with added standard deviation regressors
    """
    # Check if we're in forecast mode - never add y-dependent regressors in forecast mode
    if mode != 'train' or 'y' not in df.columns:
        return df
    
    # Check if any standard deviation regressors should be included
    if which_regressors is not None:
        std3_enabled = which_regressors.get('y_std3_regressor', True)
        std6_enabled = which_regressors.get('y_std6_regressor', True)
        std12_enabled = which_regressors.get('y_std12_regressor', True)
        
        # If all are disabled, return the original dataframe
        if not (std3_enabled or std6_enabled or std12_enabled):
            return df
    else:
        # If which_regressors not provided, enable all
        std3_enabled = std6_enabled = std12_enabled = True
    
    df_copy = df.copy()
    
    # Add only the enabled standard deviation regressors
    if std3_enabled and len(df_copy) >= 3:
        df_copy['y_std3_regressor'] = df_copy['y'].rolling(window=3, min_periods=1).std()
    
    if std6_enabled and len(df_copy) >= 6:
        df_copy['y_std6_regressor'] = df_copy['y'].rolling(window=6, min_periods=1).std()
    
    if std12_enabled and len(df_copy) >= 12:
        df_copy['y_std12_regressor'] = df_copy['y'].rolling(window=12, min_periods=1).std()
    
    # Fill any NaN values in the regressors
    for col in df_copy.columns:
        if col.endswith('_regressor'):
            df_copy[col] = df_copy[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_copy

def add_time_trend_reg(df, which_regressors=None):
    """
    Add time trend regressors to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'ds' column
    - which_regressors: Dictionary with regressor names as keys and boolean values
    
    Returns:
    - DataFrame with added time trend regressors
    """
    # Check if any time trend regressors should be included
    if which_regressors is not None:
        linear_enabled = which_regressors.get('years_since_start_regressor', True)
        power_1_5_enabled = which_regressors.get('years_since_start_power1_5_regressor', True)
        quadratic_enabled = which_regressors.get('years_since_start_squared_regressor', True)
        
        # If all are disabled, return the original dataframe
        if not (linear_enabled or power_1_5_enabled or quadratic_enabled):
            return df
    else:
        # If which_regressors not provided, enable all
        linear_enabled = power_1_5_enabled = quadratic_enabled = True
    
    df_copy = df.copy()
    
    # Add linear trend if enabled
    if linear_enabled:
        min_year = df_copy['ds'].dt.year.min()
        df_copy['years_since_start_regressor'] = df_copy['ds'].dt.year - min_year
    
    # Calculate base variable for other powers
    years_since_start = df_copy['ds'].dt.year - df_copy['ds'].dt.year.min()
    
    # Add power 1.5 trend if enabled
    if power_1_5_enabled:
        df_copy['years_since_start_power1_5_regressor'] = years_since_start ** 1.5
    
    # Add quadratic trend if enabled
    if quadratic_enabled:
        df_copy['years_since_start_squared_regressor'] = years_since_start ** 2
    
    return df_copy

def add_advanced_time_trend_reg(df, which_regressors=None):
    """
    Add advanced time trend regressors to the DataFrame.
    
    Parameters:
    - df: DataFrame with 'ds' column
    - which_regressors: Dictionary with regressor names as keys and boolean values
    
    Returns:
    - DataFrame with added time trend regressors
    """
    import numpy as np
    
    # Check which regressors to include
    if which_regressors is None:
        which_regressors = {
            'years_since_start_regressor': True,
            'years_since_start_squared_regressor': True,
            'years_since_start_power1_5_regressor': True,
            'years_since_start_log_regressor': True,
            'years_since_halfway_regressor': True,
            'multi_year_cycle_regressor': True,
            'cumulative_rainfall_regressor': True,
            'temp_trend_interaction_regressor': True
        }
    
    df_copy = df.copy()
    
    # Calculate base years since start
    min_year = df_copy['ds'].dt.year.min()
    years_since_start = df_copy['ds'].dt.year + (df_copy['ds'].dt.month - 1) / 12 - min_year
    
    # 1. Basic time trend (linear)
    if which_regressors.get('years_since_start_regressor', False):
        df_copy['years_since_start_regressor'] = years_since_start
    
    # 2. Squared time trend (quadratic)
    if which_regressors.get('years_since_start_squared_regressor', False):
        df_copy['years_since_start_squared_regressor'] = years_since_start ** 2
    
    # 3. Power 1.5 time trend (between linear and quadratic)
    if which_regressors.get('years_since_start_power1_5_regressor', False):
        df_copy['years_since_start_power1_5_regressor'] = years_since_start ** 1.5
    
    # 4. Log-transformed time trend (diminishing returns)
    if which_regressors.get('years_since_start_log_regressor', False):
        # Add small constant to avoid log(0)
        df_copy['years_since_start_log_regressor'] = np.log1p(years_since_start)
    
    # 5. Piecewise linear trend (changes after midpoint)
    if which_regressors.get('years_since_halfway_regressor', False):
        # Calculate the midpoint of the dataset's time range
        max_year = df_copy['ds'].dt.year.max() + (df_copy['ds'].dt.month.max() - 1) / 12
        midpoint = (min_year + max_year) / 2
        
        # Create regressor that's 0 before midpoint, then increases linearly
        years_since_halfway = np.maximum(0, years_since_start - (midpoint - min_year))
        df_copy['years_since_halfway_regressor'] = years_since_halfway
    
    # 6. Multi-year cycle (e.g., 3-year cycle)
    if which_regressors.get('multi_year_cycle_regressor', False):
        # Create 3-year cyclical pattern using sine function
        cycle_period = 3  # 3-year cycle
        df_copy['multi_year_cycle_regressor'] = np.sin(2 * np.pi * years_since_start / cycle_period)
    
    # 7. Cumulative rainfall (if rainfall data is available)
    if which_regressors.get('cumulative_rainfall_regressor', False) and 'rainfall_regressor' in df_copy.columns:
        # Create a 12-month rolling sum of rainfall
        if len(df_copy) >= 12:
            df_copy['cumulative_rainfall_regressor'] = df_copy['rainfall_regressor'].rolling(window=12, min_periods=1).sum()
            # Fill any NaN values
            df_copy['cumulative_rainfall_regressor'] = df_copy['cumulative_rainfall_regressor'].fillna(method='bfill').fillna(0)
    
    # 8. Temperature-trend interaction
    if which_regressors.get('temp_trend_interaction_regressor', False) and 'temperature_regressor' in df_copy.columns:
        # Create interaction between temperature and years since start
        if 'years_since_start_regressor' not in df_copy.columns:
            df_copy['years_since_start_regressor'] = years_since_start
        
        df_copy['temp_trend_interaction_regressor'] = df_copy['temperature_regressor'] * df_copy['years_since_start_regressor']
    
    return df_copy

def get_regressor_names(df, which_regressors=None, mode='train'):
    """
    Get the names of regressor columns based on configuration.
    
    Parameters:
    - df: DataFrame with 'ds' column (used to determine available data points)
    - which_regressors: Dictionary with regressor types as keys and booleans as values
    - mode: 'train' to include y-dependent regressors, 'forecast' to exclude them
    
    Returns:
    - List of regressor column names
    """
    # Generate regressors based on configuration
    df_with_regressors = add_regressors(df, which_regressors, mode)
    
    # Extract regressor column names
    regressor_cols = [col for col in df_with_regressors.columns if col.endswith('_regressor')]
    
    return regressor_cols

def add_regressors(df, which_regressors=None, mode='train'):
    """
    Add multiple types of regressors to a DataFrame based on configuration.
    
    Parameters:
    - df: DataFrame with 'ds' column and optionally 'y' column
    - which_regressors: Dictionary with regressor names as keys and booleans as values
                        e.g., {'fridays_per_month_regressor': True, 'temperature_regressor': False}
    - mode: 'train' to include y-dependent regressors, 'forecast' to exclude them
    
    Returns:
    - DataFrame with added regressors based on configuration
    """
    df_copy = df.copy()
    
    # Default configuration: include all regressors by setting which_regressors to None
    # Each individual function will handle the None case by enabling all its regressors
    
    # If mode is 'forecast', ensure all y-dependent regressors are excluded
    if mode == 'forecast' and which_regressors is not None:
        for key in list(which_regressors.keys()):
            if key.startswith('y_ma') or key.startswith('y_ewma') or key.startswith('y_std'):
                which_regressors[key] = False
    
    # Apply each regressor function, passing the which_regressors dictionary to each
    df_copy = add_fridays_per_month_reg(df_copy, which_regressors)
    df_copy = add_monthly_holidays_reg(df_copy, which_regressors)
    df_copy = add_temperature_reg(df_copy, which_regressors)
    df_copy = add_monthly_seasonality_reg(df_copy, which_regressors)
    df_copy = add_seasonal_indicators_reg(df_copy, which_regressors)
    df_copy = add_moving_avg_reg(df_copy, which_regressors, mode)
    df_copy = add_std_reg(df_copy, which_regressors, mode)
    df_copy = add_time_trend_reg(df_copy, which_regressors)
    df_copy = add_humidity_rainfall_reg(df_copy, which_regressors )
    df_copy = add_advanced_time_trend_reg(df_copy, which_regressors )
    
    return df_copy

def select_regressors_with_pearson(df1):
    area_df = df1.copy()
    #area_df = add_regressors(df, mode='forecast')
    regressors = [col for col in area_df.columns if col.endswith('_regressor')]
    
    correlations = {}
    for reg in regressors:
        corr = area_df['y'].corr(area_df[reg])
        correlations[reg] = abs(corr)  # Use absolute correlation value
    
    return correlations

def select_regressors_with_permutation(df1):
    df = df1.copy()
    #df = add_regressors(df, mode='forecast')
    
    # Get regressor columns
    regressors = [col for col in df.columns if col.endswith('_regressor')]
    X = df[regressors]
    y = df['y']

    model = Ridge()
    model.fit(X, y)
    
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_importance = dict(zip(regressors, result.importances_mean))
    
    #threshold = 0.01  # Adjust as needed
    #selected_regressors = {reg: True for reg, imp in perm_importance.items() if imp > threshold}
    return perm_importance #selected_regressors #which_regressors

def select_regressors_with_random_forest(df1):
    df = df1.copy()
    #df = add_regressors(df, mode='forecast')
    regressors = [col for col in df.columns if col.endswith('_regressor')]
    X = df[regressors]
    y = df['y']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = dict(zip(regressors, rf.feature_importances_))
    #threshold = 0.0001  # Adjust based on your data
    #selected_regressors = {reg: True for reg, imp in importances.items() if imp > threshold}
    return importances #selected_regressors #which_regressors

def select_area_specific_regressors_nonlinear(df):
    df_copy = df.copy()
    #df_copy = add_regressors(df_copy, mode='forecast')
    regressors = [col for col in df_copy.columns if col.endswith('_regressor')]
    X = df_copy[regressors]
    y = df_copy['y']

    mi_scores = mutual_info_regression(X, y)
    mi_dict = dict(zip(regressors, mi_scores))
    
    #threshold = 0.1  # Adjust based on your data
    #selected_regressors = {reg: True for reg, score in mi_dict.items() if score > threshold}
    
    return mi_dict #selected_regressors #which_regressors

def regressors_importants (df, config):

    d1 = select_area_specific_regressors_nonlinear(df)
    d2 = select_regressors_with_random_forest(df)
    d3 = select_regressors_with_permutation(df)
    
    d3_sum = sum(d3.values())
    if d3_sum > 0:  # Avoid division by zero
        d3_normalized = {k: v/d3_sum for k, v in d3.items()}
    else:
        d3_normalized = d3
    
    d3 = d3_normalized 
    combined_dict = {
        key: [d1[key], d2[key], d3[key]] for key in d1  # Assumes all dicts have the same keys
        }
    
    avg_dict = {
        key: sum(values) / len(values) 
        for key, values in combined_dict.items()
    }
    ranked_descending = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
    print (ranked_descending)
    print()

def regressors_importants_weighted(df, top_regressors = 5):
    d1 = select_area_specific_regressors_nonlinear(df)  # Mutual Information
    d2 = select_regressors_with_random_forest(df)       # Random Forest
    d3 = select_regressors_with_permutation(df)         # Permutation Importance
    d4 = select_regressors_with_pearson(df)             # Pearson Correlation
    
    d3_sum = sum(d3.values())
    if d3_sum > 0:
        d3_normalized = {k: v/d3_sum for k, v in d3.items()}
    else:
        d3_normalized = d3
    
    weights = {
        'mutual_info': 0.3,      # Non-linear 0.3
        'random_forest': 0.45,    # Non-linear 0.4
        'permutation': 0.25,      # Non-linear 0.25
        'pearson': 0.0           # Linear only - lower weight 0.05
    }
    
    # Get all unique keys
    all_keys = set(d1.keys()) | set(d2.keys()) | set(d3_normalized.keys()) | set(d4.keys())
    
    # Calculate weighted average importance
    weighted_avg = {}
    for key in all_keys:
        weighted_avg[key] = (
            weights['mutual_info'] * d1.get(key, 0) +
            weights['random_forest'] * d2.get(key, 0) +
            weights['permutation'] * d3_normalized.get(key, 0) +
            weights['pearson'] * d4.get(key, 0)
        )
        
    ranked_descending = sorted(weighted_avg.items(), key=lambda x: x[1], reverse=True)
    print(ranked_descending)

    top_regressor_names = [name for name, _ in ranked_descending[:top_regressors]]
    print ()
    print(top_regressor_names)
    all_regressor_cols = [col for col in df.columns if col.endswith('_regressor')]
    regressors_to_drop = [col for col in all_regressor_cols if col not in top_regressor_names]
    df_filtered = df.drop(columns=regressors_to_drop)
    top_reg = top_regressor_names.copy()
    return df_filtered, top_regressor_names