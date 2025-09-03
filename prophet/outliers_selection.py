from data_loader import *
from scipy.stats import zscore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def print_options():
    pd.set_option("display.width", 1000)  # Set arbitrarily large
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 10)
    pd.set_option("display.expand_frame_repr", False)  # Critical: Prevent wrapping
    pd.set_option('display.float_format', '{:.2f}'.format)
    
def detect_outliers_zscore(df, target_col='y', threshold=3):
    """
    Detect outliers using the Z-score method.
    
    Parameters:
    - df: DataFrame containing the time series data
    - target_col: Name of the target column
    - threshold: Z-score threshold for identifying outliers
    
    Returns:
    - df: DataFrame with an additional 'outlier' column
    """
    df['zscore'] = zscore(df[target_col])
    df['outlier'] = df['zscore'].abs() > threshold
    return df

def detect_outliers_iqr(df, target_col='y', factor=1.5):
    """
    Detect outliers using the IQR method.
    
    Parameters:
    - df: DataFrame containing the time series data
    - target_col: Name of the target column
    - factor: Factor to multiply the IQR
    
    Returns:
    - df: DataFrame with an additional 'outlier' column
    """
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    df['outlier'] = (df[target_col] < (Q1 - factor * IQR)) | (df[target_col] > (Q3 + factor * IQR))
    return df

def detect_outliers_moving_window(df, target_col='y', window=12, threshold=3):
    df['rolling_mean'] = df[target_col].rolling(window=window, center=True, min_periods=1).mean()
    df['rolling_std'] = df[target_col].rolling(window=window, center=True, min_periods=1).std()
    df['outlier'] = (df[target_col] < (df['rolling_mean'] - threshold * df['rolling_std'])) | \
                    (df[target_col] > (df['rolling_mean'] + threshold * df['rolling_std']))
    return df

def cap_outliers(df, target_col='y'):
    """
    Cap outliers using the IQR method.
    
    Parameters:
    - df: DataFrame containing the time series data
    - target_col: Name of the target column
    
    Returns:
    - DataFrame with outliers capped
    """
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    df[target_col] = df[target_col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    return df

def interpolate_outliers(df, target_col='y'):
    df_copy = df.copy()
    
    # Mark outliers as NaN
    df_copy.loc[df['outlier'], target_col] = np.nan
    
    # First try linear interpolation (works for gaps between valid values)
    df_copy[target_col] = df_copy[target_col].interpolate(method='linear')
    
    # If there are still NaN values at the beginning or end, use different methods
    if df_copy[target_col].isna().any():
        # For beginning NaNs: use next valid values or seasonal patterns if enough data
        if df_copy[target_col].iloc[:12].isna().any():
            # If we have at least 24 months of data, try seasonal filling
            if len(df_copy) >= 24:
                # For each missing month, look at the same month in the next year
                for i in range(min(12, len(df_copy))):
                    if pd.isna(df_copy[target_col].iloc[i]):
                        seasonal_idx = i + 12  # Same month, next year
                        if seasonal_idx < len(df_copy) and not pd.isna(df_copy[target_col].iloc[seasonal_idx]):
                            df_copy[target_col].iloc[i] = df_copy[target_col].iloc[seasonal_idx]
            
            # After trying seasonal filling, do forward fill for any remaining NaNs
            df_copy[target_col] = df_copy[target_col].fillna(method='bfill')
        
        # For ending NaNs, use previous valid values
        if df_copy[target_col].iloc[-12:].isna().any():
            df_copy[target_col] = df_copy[target_col].fillna(method='ffill')
    
    return df_copy

def smooth_outliers(df, target_col='y', outlier_col='outlier', window=3):
    df_copy = df.copy()

    # Create a smoothed version of the target column
    smoothed_series = (
        df_copy[target_col]
        .rolling(window=window, center=True)
        .mean()
        .fillna(method='bfill')
        .fillna(method='ffill')
    )

    # Replace only the outlier values with smoothed values
    df_copy.loc[df_copy[outlier_col], target_col] = smoothed_series[df_copy[outlier_col]]

    return df_copy

def remove_outliers(df, target_col='y'):
    """Remove outliers by dropping rows marked as outliers"""
    return df[~df['outlier']].copy()

def preprocess_with_outliers(df, config):
    """
    Detect and handle outliers in time series data.
    """
    processed_df = df.copy()
    method = config['outlier_detection']
    treatment = config['outlier_treatment']
    
    # Apply detection method
    if method == 'zscore':
        processed_df = detect_outliers_zscore(processed_df)
    elif method == 'iqr':
        processed_df = detect_outliers_iqr(processed_df)
    elif method == 'moving':
        processed_df = detect_outliers_moving_window(processed_df)
    else:
        # If no detection method, assume no outliers
        processed_df['outlier'] = False
    
    processed_df['original_y'] = processed_df['y'].copy()
    outlier_count = processed_df['outlier'].sum() if 'outlier' in processed_df.columns else 0
    
    # Only apply treatment if outliers were found
    if outlier_count > 0:
        if treatment == 'cap':
            processed_df = cap_outliers(processed_df)
        elif treatment == 'interpolate':
            processed_df = interpolate_outliers(processed_df)
        elif treatment == 'smooth':
            processed_df = smooth_outliers(processed_df)
        # Note: removed 'remove' option since function doesn't exist
    
    # Prepare outlier information
    outlier_info = {
        'detection_method': method if outlier_count > 0 else None,
        'treatment_method': treatment if outlier_count > 0 else None,
        'count': outlier_count,
        'percentage': (outlier_count / len(df)) * 100 if len(df) > 0 else 0
    }
    
    return processed_df, outlier_info

def calculate_seasonality_preservation(df_original, df_treated):
    """
    Calculate how well outlier treatment preserves seasonal patterns
    """
    import pandas as pd
    from scipy.stats import pearsonr
    
    # Monthly seasonal patterns
    orig_monthly = df_original.groupby(df_original['ds'].dt.month)['y'].mean()
    treat_monthly = df_treated.groupby(df_treated['ds'].dt.month)['y'].mean()
    
    # Correlation between seasonal patterns
    seasonal_corr, _ = pearsonr(orig_monthly, treat_monthly)
    seasonal_corr = max(0, seasonal_corr)  # Ensure non-negative
    
    # Seasonal amplitude preservation
    orig_amplitude = orig_monthly.max() - orig_monthly.min()
    treat_amplitude = treat_monthly.max() - treat_monthly.min()
    
    if orig_amplitude > 0:
        amplitude_ratio = min(treat_amplitude, orig_amplitude) / max(treat_amplitude, orig_amplitude)
    else:
        amplitude_ratio = 1.0
    
    # Combined seasonal score (0-1, higher is better)
    seasonal_score = (seasonal_corr + amplitude_ratio) / 2
    
    return seasonal_score

def calculate_trend_preservation(df_original, df_treated):
    """
    Calculate how well outlier treatment preserves trend patterns
    """
    from scipy.stats import linregress
    import numpy as np
    
    # Calculate linear trends
    x_orig = np.arange(len(df_original))
    x_treat = np.arange(len(df_treated))
    
    orig_trend = linregress(x_orig, df_original['y']).slope
    treat_trend = linregress(x_treat, df_treated['y']).slope
    
    # Trend direction and magnitude preservation
    if abs(orig_trend) < 0.01:  # Essentially no trend
        trend_score = 1.0 if abs(treat_trend) < 0.1 else 0.8
    else:
        # How well does treated trend match original
        trend_ratio = treat_trend / orig_trend
        trend_score = max(0, 1 - abs(1 - trend_ratio))
    
    # Overall level preservation (mean values)
    orig_mean = df_original['y'].mean()
    treat_mean = df_treated['y'].mean()
    
    if orig_mean > 0:
        level_ratio = min(treat_mean, orig_mean) / max(treat_mean, orig_mean)
    else:
        level_ratio = 1.0
    
    # Combined trend score (0-1, higher is better)
    combined_trend = (trend_score + level_ratio) / 2
    
    return combined_trend

def calculate_data_quality_score(df_original, df_treated, outlier_info):
    """
    Calculate data quality metrics
    """
    # Data retention (penalize excessive removal)
    retention_ratio = len(df_treated) / len(df_original)
    
    # Outlier detection rate (moderate is good, too high/low is suspicious)
    outlier_rate = outlier_info['count'] / len(df_original)
    
    # Optimal outlier rate is 2-8% for water consumption data
    if 0.02 <= outlier_rate <= 0.08:
        outlier_score = 1.0
    elif outlier_rate < 0.02:
        outlier_score = 0.8  # Maybe too conservative
    elif outlier_rate <= 0.15:
        outlier_score = 0.6  # Somewhat high but acceptable
    else:
        outlier_score = 0.3  # Too aggressive
    
    # Variance preservation (avoid over-smoothing)
    orig_cv = df_original['y'].std() / df_original['y'].mean()
    treat_cv = df_treated['y'].std() / df_treated['y'].mean()
    
    if orig_cv > 0:
        variance_ratio = min(treat_cv, orig_cv) / max(treat_cv, orig_cv)
    else:
        variance_ratio = 1.0
    
    # Combined quality score
    quality_score = (retention_ratio + outlier_score + variance_ratio) / 3
    
    return quality_score

def evaluate_outlier_method_statistical(df_original, df_treated, outlier_info):
    """
    Complete statistical evaluation of outlier treatment method
    """
    seasonality_score = calculate_seasonality_preservation(df_original, df_treated)
    trend_score = calculate_trend_preservation(df_original, df_treated)
    quality_score = calculate_data_quality_score(df_original, df_treated, outlier_info)
    
    # Weighted combination (emphasize seasonality for water demand)
    weights = {
        'seasonality': 0.4,  # Most important for water forecasting
        'trend': 0.35,       # Important for long-term planning
        'quality': 0.25      # Data integrity check
    }
    
    combined_score = (
        weights['seasonality'] * seasonality_score +
        weights['trend'] * trend_score +
        weights['quality'] * quality_score
    )
    
    return {
        'seasonality_score': seasonality_score,
        'trend_score': trend_score,
        'quality_score': quality_score,
        'combined_score': combined_score
    }

def select_optimal_outlier_method_statistical(df, area_id):
    """
    Select optimal outlier method using statistical criteria only
    """
    detection_methods = ['zscore', 'iqr', 'moving']
    treatment_methods = ['cap', 'interpolate', 'smooth']
    
    # FIRST: Check if ANY method finds outliers
    outliers_found_anywhere = False
    
    for detect_method in detection_methods:
        # Quick check for outliers with each detection method
        temp_df = df.copy()
        if detect_method == 'zscore':
            temp_df = detect_outliers_zscore(temp_df)
        elif detect_method == 'iqr':
            temp_df = detect_outliers_iqr(temp_df)
        elif detect_method == 'moving':
            temp_df = detect_outliers_moving_window(temp_df)
        
        if temp_df['outlier'].sum() > 0:
            outliers_found_anywhere = True
            break
    
    # If NO outliers found by ANY method, return 'none'
    if not outliers_found_anywhere:
        return (None, None), [{
            'area_id': area_id,
            'detection': None,
            'treatment': None,
            'outliers_found': 0,
            'outlier_percentage': 0.0,
            'seasonality_score': 1.0,  # Perfect preservation
            'trend_score': 1.0,        # Perfect preservation
            'quality_score': 1.0,      # Perfect quality
            'combined_score': 1.0      # Perfect score
        }]
    
    # If outliers ARE found, proceed with normal optimization
    best_score = 0
    best_combination = None
    results = []
    
    for detect_method in detection_methods:
        for treat_method in treatment_methods:
            try:
                config = {
                    'outlier_detection': detect_method,
                    'outlier_treatment': treat_method
                }
                
                processed_df, outlier_info = preprocess_with_outliers(df.copy(), config)
                
                # Skip if this method found no outliers
                if outlier_info['count'] == 0:
                    continue
                
                # Statistical evaluation
                scores = evaluate_outlier_method_statistical(df, processed_df, outlier_info)
                
                results.append({
                    'area_id': area_id,
                    'detection': detect_method,
                    'treatment': treat_method,
                    'outliers_found': outlier_info['count'],
                    'outlier_percentage': outlier_info['percentage'],
                    **scores
                })
                
                if scores['combined_score'] > best_score:
                    best_score = scores['combined_score']
                    best_combination = (detect_method, treat_method)
                    
            except Exception as e:
                print(f"Error with {detect_method}-{treat_method} for area {area_id}: {str(e)}")
                continue
    
    return best_combination, results

def analyze_outlier_methods_across_areas(filepath, area_ids):
    """
    Comprehensive outlier method analysis across all areas
    """
    all_results = []
    optimal_methods = {}
    
    for area_id in area_ids:
        print(f"Analyzing outlier methods for Area {area_id}...")
        
        df, _ = load_time_series(filepath, area_id)
        if df is None:
            continue
            
        best_combo, area_results = select_optimal_outlier_method_statistical(df, area_id)
        optimal_methods[area_id] = best_combo
        all_results.extend(area_results)
    
    # Create comprehensive analysis
    results_df = pd.DataFrame(all_results)
    
    # Summary statistics
    summary = {
        'best_detection_overall': results_df.groupby('detection')['combined_score'].mean().idxmax(),
        'best_treatment_overall': results_df.groupby('treatment')['combined_score'].mean().idxmax(),
        'method_frequency': results_df.groupby(['detection', 'treatment']).size().sort_values(ascending=False),
        'average_scores_by_method': results_df.groupby(['detection', 'treatment'])[
            ['seasonality_score', 'trend_score', 'quality_score', 'combined_score']
        ].mean().round(3)
    }
    
    return results_df, optimal_methods, summary

def create_preprocessing_results_figure(results_df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Combined Score by Area (sorted)
    sorted_data = results_df.sort_values('combined_score', ascending=True)
    colors = ['red' if score < 0.8 else 'orange' if score < 0.9 else 'green' 
              for score in sorted_data['combined_score']]
    
    ax1.barh(range(len(sorted_data)), sorted_data['combined_score'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(sorted_data)))
    ax1.set_yticklabels([f"Area {id}" for id in sorted_data['area_id']], fontsize=8)
    ax1.set_xlabel('Combined Effectiveness Score')
    ax1.set_title('Preprocessing Effectiveness by Area')
    ax1.axvline(x=0.9, color='black', linestyle='--', alpha=0.5, label='Excellent (0.9)')
    ax1.legend()
    
    # 2. Detection-Treatment Method Distribution
    method_combos = results_df.apply(lambda x: f"{x['detection']}-{x['treatment']}", axis=1)
    combo_counts = method_combos.value_counts()
    
    ax2.pie(combo_counts.values, labels=combo_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution of Optimal Method Combinations')
    
    # 3. Outlier Percentage vs Effectiveness Score
    scatter_colors = results_df['detection'].map({None: 'blue', 'iqr': 'orange', 'zscore': 'green'})
    ax3.scatter(results_df['outlier_percentage'], results_df['combined_score'], 
               c=scatter_colors, alpha=0.7, s=60)
    
    # Annotate challenging areas
    for idx, row in results_df.iterrows():
        if row['combined_score'] < 0.8 or row['outlier_percentage'] > 10:
            ax3.annotate(f"Area {row['area_id']}", 
                        (row['outlier_percentage'], row['combined_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Outlier Percentage (%)')
    ax3.set_ylabel('Combined Effectiveness Score')
    ax3.set_title('Outlier Rate vs Preprocessing Effectiveness')
    
    # Create legend for detection methods
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='No Detection'),
                      Patch(facecolor='orange', label='IQR'),
                      Patch(facecolor='green', label='Z-score')]
    ax3.legend(handles=legend_elements)
    
    # 4. Score Components Comparison
    score_cols = ['seasonality_score', 'trend_score', 'quality_score']
    x_pos = np.arange(len(score_cols))
    
    means = [results_df[col].mean() for col in score_cols]
    stds = [results_df[col].std() for col in score_cols]
    
    ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
           color=['skyblue', 'lightcoral', 'lightgreen'])
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Seasonality', 'Trend', 'Quality'])
    ax4.set_ylabel('Average Preservation Score')
    ax4.set_title('Average Preservation Scores by Component')
    ax4.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax4.text(i, mean + std + 0.02, f'{mean:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_method_effectiveness_figure(results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by method combinations and show effectiveness
    method_stats = results_df.groupby(['detection', 'treatment']).agg({
        'combined_score': ['mean', 'count'],
        'outlier_percentage': 'mean'
    }).round(3)
    
    method_stats.columns = ['avg_score', 'count', 'avg_outliers']
    method_stats = method_stats.reset_index()
    method_stats['method_combo'] = method_stats['detection'] + '-' + method_stats['treatment']
    
    # Effectiveness by method
    ax1.barh(method_stats['method_combo'], method_stats['avg_score'], 
             color='steelblue', alpha=0.7)
    ax1.set_xlabel('Average Effectiveness Score')
    ax1.set_title('Method Effectiveness Comparison')
    ax1.axvline(x=0.9, color='red', linestyle='--', label='Excellent Threshold')
    ax1.legend()
    
    # Method usage frequency
    ax2.bar(method_stats['method_combo'], method_stats['count'], 
           color='darkorange', alpha=0.7)
    ax2.set_ylabel('Number of Areas')
    ax2.set_title('Method Usage Frequency')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
def detect_and_visualize_outliers(df, area_id):
    """
    Detect outliers using multiple methods and visualize them.
    Parameters:
    - df: DataFrame with 'ds' and 'y' columns
    - area_id: ID of the area for visualization title
    
    Returns:
    - Dictionary with outlier information
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Make a copy of the data
    df_copy = df.copy()
    
    # Detect outliers using multiple methods
    df_zscore = detect_outliers_zscore(df_copy.copy(), threshold=3.0)
    df_iqr = detect_outliers_iqr(df_copy.copy(), factor=1.5)
    df_moving = detect_outliers_moving_window(df_copy.copy(), window=12, threshold=3.0)
    
    # Count outliers for each method
    zscore_count = df_zscore['outlier'].sum() if 'outlier' in df_zscore.columns else 0
    iqr_count = df_iqr['outlier'].sum() if 'outlier' in df_iqr.columns else 0
    moving_count = df_moving['outlier'].sum() if 'outlier' in df_moving.columns else 0
    
    # Create a visualization comparing all methods
    plt.figure(figsize=(15, 10))
    
    # Plot the original data
    plt.subplot(4, 1, 1)
    plt.plot(df_copy['ds'], df_copy['y'], 'b-', label='Original Data')
    plt.title(f'Area {area_id}: Original Time Series')
    plt.ylabel('Water Consumption')
    plt.grid(True, alpha=0.3)
    
    # Plot Z-score method
    plt.subplot(4, 1, 2)
    plt.plot(df_copy['ds'], df_copy['y'], 'b-', alpha=0.5)
    if zscore_count > 0:
        outliers = df_zscore[df_zscore['outlier']]
        plt.scatter(outliers['ds'], outliers['y'], color='red', s=30)
    plt.title(f'Z-Score Method (threshold=3.0): {zscore_count} outliers')
    plt.ylabel('Water Consumption')
    plt.grid(True, alpha=0.3)
    
    # Plot IQR method
    plt.subplot(4, 1, 3)
    plt.plot(df_copy['ds'], df_copy['y'], 'b-', alpha=0.5)
    if iqr_count > 0:
        outliers = df_iqr[df_iqr['outlier']]
        plt.scatter(outliers['ds'], outliers['y'], color='red', s=30)
    plt.title(f'IQR Method (factor=1.5): {iqr_count} outliers')
    plt.ylabel('Water Consumption')
    plt.grid(True, alpha=0.3)
    
    # Plot Moving Window method
    plt.subplot(4, 1, 4)
    plt.plot(df_copy['ds'], df_copy['y'], 'b-', alpha=0.5)
    if moving_count > 0:
        outliers = df_moving[df_moving['outlier']]
        plt.scatter(outliers['ds'], outliers['y'], color='red', s=30)
    plt.title(f'Moving Window Method (window=12, threshold=3.0): {moving_count} outliers')
    plt.xlabel('Date')
    plt.ylabel('Water Consumption')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'outlier_detection_area_{area_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create treated versions of the data
    df_capped = cap_outliers(df_moving.copy())
    
    # Create interpolated version (using moving window outliers)
    df_interp = df_moving.copy()
    if 'outlier' in df_interp.columns and df_interp['outlier'].any():
        df_interp.loc[df_interp['outlier'], 'y'] = np.nan
        df_interp['y'] = df_interp['y'].interpolate(method='linear')
    
    # Visualize the original vs treated data
    plt.figure(figsize=(12, 6))
    plt.plot(df_copy['ds'], df_copy['y'], 'b-', alpha=0.7, label='Original')
    plt.plot(df_capped['ds'], df_capped['y'], 'g-', label='Capped')
    plt.plot(df_interp['ds'], df_interp['y'], 'r-', label='Interpolated')
    
    plt.title(f'Area {area_id}: Original vs Treated Data')
    plt.xlabel('Date')
    plt.ylabel('Water Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outlier_treatment_area_{area_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'zscore': {'count': zscore_count, 'data': df_zscore},
        'iqr': {'count': iqr_count, 'data': df_iqr},
        'moving': {'count': moving_count, 'data': df_moving},
        'capped': {'data': df_capped},
        'interpolated': {'data': df_interp}
    }
if __name__ == '__main__':
    print_options()
    results_df, optimal_methods, summary =  analyze_outlier_methods_across_areas('../data/combined_water_data.csv', list(range(1, 30)))
    optimal_df = results_df.loc[results_df.groupby('area_id')['combined_score'].idxmax()]
    filtered_df = optimal_df[optimal_df['detection'].notna() & optimal_df['treatment'].notna()]
    create_preprocessing_results_figure(filtered_df)
    create_method_effectiveness_figure(filtered_df)
    print(f"Optimal methods per area: {len(filtered_df)} rows")  # Should be 29



