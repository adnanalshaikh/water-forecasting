import pandas as pd

def load_time_series(filepath='../data/combined_water_data.csv', area_id=3):
    """
    Load and prepare time series data for a specific area.
    
    Parameters:
    - filepath: Path to the CSV file
    - area_id: ID of the area to analyze
    
    Returns:
    - DataFrame with columns 'ds' (dates) and 'y' (target variable)
    """   
    df = load_data(filepath)
    ts_df, missing_count = get_time_series(df, area_id=area_id)
    #if ts_df is None:
    #    return None, missing_count
    
    ts_df['Date'] = pd.to_datetime(ts_df['Date'])
    ts_df = ts_df.rename(columns={'Date': 'ds', 'WaterConsumption': 'y'})
    
    return ts_df, missing_count

def load_data(filepath='../data/combined_water_data.csv', date_column='Date'):
    """
    Load the raw data from CSV file.
    
    Parameters:
    - filepath: Path to the CSV file
    - date_column: Name of the date column
    
    Returns:
    - DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(filepath, parse_dates=[date_column])
        df.sort_values(by=['AreaID', 'Date'], inplace=True)        
    except FileNotFoundError:
        print(f"ERROR: The file {filepath} was not found.")
        return None
    
    return df[['AreaID', 'Date', 'WaterConsumption']]

def get_time_series(df, area_id=3, set_index=False, config = None):
    """
    Extract time series data for a specific area.
    
    Parameters:
    - df: DataFrame with the raw data
    - area_id: ID of the area to extract
    - set_index: Whether to set the date as index
    
    Returns:
    - DataFrame with the time series for the specified area
    """
    if df is None:
        print("ERROR: Data not loaded.")
        return None
    
    ts_df = df[df['AreaID'] == area_id][['Date', 'WaterConsumption']].copy()
    ts_df, missing_count = handle_missing_values(ts_df, config)

    if ts_df is None:
        return None, missing_count
    
    if set_index:
        return ts_df.set_index('Date').sort_index(), missing_count
    else:
        return ts_df.sort_values('Date').reset_index(drop=True), missing_count    

def handle_missing_values(df, config):
    """
    Handle NaN values in the 'WaterConsumption' column based on a threshold.
    
    Parameters:
    - df: DataFrame with columns 'Date' and 'WaterConsumption'
    - method: Imputation method ('linear' for linear interpolation or 'ffill' for forward fill)
    - threshold: Maximum allowed number of NaN values
                If None, handle all missing values regardless of count
                If a number, only handle if missing values are less than or equal to this threshold
    
    Returns:
    - DataFrame with handled missing values or None if NaN count > threshold
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    method = config.get('impute_method', 'ffill') if config is not None else 'ffill'
    threshold = config.get('impute_thresh', 3) if config is not None else None
    # Count the number of NaNs in the 'WaterConsumption' column
    nan_count = df_copy['WaterConsumption'].isna().sum()
    
    # If threshold is not None and NaN count exceeds the threshold, return None
    if threshold is not None and nan_count > threshold:
        return None, nan_count
    
    # If there are NaNs, handle them according to the specified method
    if nan_count > 0:
        if method == 'linear':
            # Use linear interpolation
            df_copy['WaterConsumption'] = df_copy['WaterConsumption'].interpolate(method='linear')
            
            # Handle any remaining NaNs at the beginning or end
            df_copy['WaterConsumption'] = df_copy['WaterConsumption'].fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'ffill':
            # Try forward-fill first
            df_copy['WaterConsumption'] = df_copy['WaterConsumption'].ffill()
            
            # Then backward-fill for any remaining NaNs at the beginning
            df_copy['WaterConsumption'] = df_copy['WaterConsumption'].bfill()
    
    return df_copy, nan_count

def get_rainfall_data():
    """
    Return the temperature data for Nablus station.
    
    Returns:
    - Dictionary with months as keys and temperature lists as values
    """
    # Temperature data (from Nablus station)
    temp_data = {
        'JAN': [187.7, 278.6, 174.1, 126.1, 107.7, 266],
        'FEB': [177.8, 144.3, 233.7, 97.6, 154.8, 142],
        'MAR': [122.5, 152, 32.6, 126.9, 97.7, 76],
        'APR': [17.5, 52.8, 13.3, 23, 88.9, 12.8],
        'MAY': [0, 21, 0, 16.3, .1, 14.3],
        'JUN': [0, 0, 0, 0, 6.5, 0],
        'JUL': [0, 0, 0, 0, .9, 0],
        'AUG': [0, 0, 0, 0, 0, 0],
        'SEP': [0, 0, 7.6, 0, 7, 0],
        'OCT': [10.1, 0, 1.3, 3.3, 6.3, 7.6],
        'NOV': [3.2, 137.1, 39.2, 61.6, 162.2, 46.9],
        'DEC': [260.8, 62.4, 179.3, 47.9, 156.7, 130]
    }
    return temp_data

def get_temp_data():
    """
    Return the temperature data for Nablus station.
    
    Returns:
    - Dictionary with months as keys and temperature lists as values
    """
    # Temperature data (from Nablus station)
    temp_data = {
        'JAN': [9.9, 9.6, 12.4, None, 12.4, 11.9],
        'FEB': [10.9, 11.0, 12.8, None, 10.5, 11.7],
        'MAR': [12.2, 13.6, 13.4, None, 15.2, 14.9],
        'APR': [16.1, None, 19.2, None, 17.8, 19.8],
        'MAY': [23.7, 26.7, 23.1, None, 21.4, 20.9],
        'JUN': [25.3, 23.0, 23.3, None, 23.3, 27.0],
        'JUL': [26.2, 26.4, 26.5, None, 27.2, 27.3],
        'AUG': [26.3, 25.8, 27.3, None, 26.2, 26.4],
        'SEP': [24.5, 27.0, 24.2, None, 25.4, 24.3],
        'OCT': [22.1, 24.0, 21.7, 21.7, 22.7, 20.9],
        'NOV': [18.5, 16.9, 18.9, None, 18.4, 16.2],
        'DEC': [12.6, 13.3, 14.7, None, 14.5, 12.7]
    }
    return temp_data

def get_humidity_data():
    """
    Return the humidity data for Nablus station.
    
    Returns:
    - Dictionary with months as keys and humidity lists as values
    """
    # Humidity data (from Nablus station)
    humidity_data = {
        'JAN': [82, 91, 77, None, 83, 78],
        'FEB': [84, 91, 81, None, 84, 76],
        'MAR': [86, 85, 82, None, 76, 63],
        'APR': [74, None, 64, None, 72, 60],
        'MAY': [55, 48, 64, None, 60, 59],
        'JUN': [70, 75, 76, None, 61, 54],
        'JUL': [73, 78, 78, None, 51, 59],
        'AUG': [77, 85, 74, None, 68, 67],
        'SEP': [82, 82, 84, None, 62, 70],
        'OCT': [80, 75, 79, 74, 62, 61],
        'NOV': [58, 87, 67, None, 66, 72],
        'DEC': [81, 84, 76, None, 72, 64]
    }
    
    # Calculate monthly humidity averages
    monthly_avgs = {}
    for month in range(1, 13):
        month_name = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                      'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][month-1]
        if month_name in humidity_data:
            values = [v for v in humidity_data[month_name] if v is not None]
            if values:
                monthly_avgs[month] = sum(values) / len(values)
    
    # Fill in None values with monthly averages
    for month_name, values in humidity_data.items():
        month_idx = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'].index(month_name) + 1
        
        for i in range(len(values)):
            if values[i] is None:
                humidity_data[month_name][i] = monthly_avgs[month_idx]
    
    return humidity_data


if __name__ == '__main__':
    results = []
    
    for i in range(29):
        d, _ = load_time_series(filepath='../data/combined_water_data.csv', area_id=i+1) 
        # Get min and max values from the time series (assuming column 1 contains values)
        min_val = d.min()[1]
        max_val = d.max()[1]
        
        # Append results as a dictionary
        results.append({
            'area_id': i+1,
            'min_value': min_val,
            'max_value': max_val
        })
    
    # Convert the list of dictionaries to a DataFrame
    df_results = pd.DataFrame(results)
    
    # Display the DataFrame
    print(df_results)
    
    non_zero_min = df_results['min_value'][df_results['min_value'] != 0].min()
    print(non_zero_min)
    
    