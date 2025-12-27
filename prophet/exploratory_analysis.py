import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - NO windows!

import matplotlib.pyplot as plt

import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import antropy as ant  # Add this import at the top with other imports
from statsmodels.tsa.seasonal import STL   
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

#from prophet_forecasting_reg import * 
import warnings
warnings.filterwarnings('ignore')
from data_loader import load_time_series

def print_options():
    pd.set_option("display.width", 1000)  # Set arbitrarily large
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 10)
    pd.set_option("display.expand_frame_repr", False)  # Critical: Prevent wrapping
    pd.set_option('display.float_format', '{:.2f}'.format)
    
def calculate_component_variance_contribution(df, area_id):
    """Calculate variance contribution of decomposition components"""
    ts = df.set_index('ds')['y']
    decomposition = seasonal_decompose(ts, model='additive', period=12)
    
    total_var = ts.var()
    trend_var = decomposition.trend.dropna().var()
    seasonal_var = decomposition.seasonal.var()
    residual_var = decomposition.resid.dropna().var()
    
    return {
        'area_id': area_id,
        'total_variance': total_var,
        'trend_contribution': trend_var / total_var * 100,
        'seasonal_contribution': seasonal_var / total_var * 100,
        'residual_contribution': residual_var / total_var * 100,
        'explained_variance': (trend_var + seasonal_var) / total_var * 100
    }

def analyze_seasonal_patterns(df, area_id):
    """Detailed seasonal pattern analysis"""
    df['month'] = df['ds'].dt.month
    monthly_means = df.groupby('month')['y'].mean()
    
    # Calculate seasonal metrics
    peak_month = monthly_means.idxmax()
    trough_month = monthly_means.idxmin()
    seasonal_amplitude = monthly_means.max() - monthly_means.min()
    seasonal_strength = seasonal_amplitude / df['y'].mean()
    
    # Seasonal consistency (CV of monthly means across years)
    yearly_monthly = df.pivot_table(values='y', index=df['ds'].dt.year, 
                                   columns=df['ds'].dt.month, aggfunc='mean')
    seasonal_consistency = {}
    for month in range(1, 13):
        if month in yearly_monthly.columns:
            seasonal_consistency[month] = yearly_monthly[month].std() / yearly_monthly[month].mean()
    
    return {
        'area_id': area_id,
        'peak_month': peak_month,
        'trough_month': trough_month,
        'seasonal_amplitude': seasonal_amplitude,
        'seasonal_strength': seasonal_strength,
        'summer_winter_ratio': monthly_means[[6,7,8]].mean() / monthly_means[[12,1,2]].mean(),
        'seasonal_consistency_cv': np.mean(list(seasonal_consistency.values()))
    }

def perform_stationarity_tests(df, area_id):
    """Test for stationarity and calculate predictability metrics"""
    ts = df['y'].values
    
    # ADF test
    adf_result = adfuller(ts, autolag='AIC')
    is_stationary = adf_result[1] < 0.05
    
    # First difference ADF test
    diff_ts = np.diff(ts)
    adf_diff_result = adfuller(diff_ts, autolag='AIC')
    is_diff_stationary = adf_diff_result[1] < 0.05
    
    # Calculate ACF at lag 1 for predictability
    from statsmodels.tsa.stattools import acf
    acf_values = acf(ts, nlags=12, fft=False)
    lag1_autocorr = acf_values[1]
    
    return {
        'area_id': area_id,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'is_stationary': is_stationary,
        'is_diff_stationary': is_diff_stationary,
        'lag1_autocorr': lag1_autocorr,
        'predictability_score': abs(lag1_autocorr)  # Higher = more predictable
    }

def classify_forecasting_difficulty(area_stats):
    
    """Classify areas by forecasting difficulty based on multiple criteria"""
    cv = area_stats['cv']
    skewness = abs(area_stats['skewness'])
    seasonal_strength = area_stats.get('seasonal_strength', 0)
    predictability = area_stats.get('predictability_score', 0)
    
    # Define scoring criteria
    score = 0
    
    # CV contribution (0-3 points, lower is better)
    if cv <= 0.15:
        score += 3
    elif cv <= 0.25:
        score += 2
    elif cv <= 0.35:
        score += 1
    
    # Skewness contribution (0-2 points, lower is better)
    if skewness <= 0.5:
        score += 2
    elif skewness <= 1.0:
        score += 1
    
    # Seasonal strength (0-2 points, higher is better)
    if seasonal_strength >= 0.2:
        score += 2
    elif seasonal_strength >= 0.1:
        score += 1
    
    # Predictability (0-3 points, higher is better)
    if predictability >= 0.7:
        score += 3
    elif predictability >= 0.5:
        score += 2
    elif predictability >= 0.3:
        score += 1
    
    # Classify based on total score (0-10)
    if score >= 8:
        return 'High Predictability'
    elif score >= 5:
        return 'Moderate Predictability'
    else:
        return 'Challenging'

def calculate_trend_strength(series, period=12):
    """Calculate trend strength using STL decomposition"""
    stl = STL(series, period=period).fit()
    return max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))

def calculate_sample_entropy(series):
    """Calculate sample entropy (measure of unpredictability)"""
    return ant.sample_entropy(series)

def calculate_rolling_volatility(series, window=12):
    """Calculate mean rolling standard deviation (local volatility)"""
    return series.rolling(window=window).std().mean()

def create_comprehensive_eda_table():
    """Create comprehensive EDA summary table for all areas (now with trend/entropy/volatility)"""
    filepath = '../data/combined_water_data.csv'
    results = []
    
    for area_id in range(1, 30):
        df, missing_count = load_time_series(filepath, area_id)
        if df is not None:
            # Basic stats (existing)
            basic_stats = {
                'area_id': area_id,
                'n_obs': len(df),
                'missing_count': missing_count,
                'mean': df['y'].mean(),
                'std': df['y'].std(),
                'cv': df['y'].std() / df['y'].mean(),
                'skewness': df['y'].skew(),
                'min_value': df['y'].min(),
                'max_value': df['y'].max()
            }
            
            # === NEW FEATURES ADDED HERE ===
            ts = df.set_index('ds')['y']
            
            # Trend strength
            trend_strength = calculate_trend_strength(ts)
            
            # Sample entropy
            sample_entropy = calculate_sample_entropy(ts.values)
            
            # Rolling volatility
            rolling_std = calculate_rolling_volatility(ts)
            
            # Component analysis (existing)
            component_stats = calculate_component_variance_contribution(df, area_id)
            
            # Seasonal analysis (existing)
            seasonal_stats = analyze_seasonal_patterns(df, area_id)
            
            # Stationarity analysis (existing)
            stationarity_stats = perform_stationarity_tests(df, area_id)
            
            # Combine all stats (including new features)
            combined_stats = {
                **basic_stats,
                **component_stats,
                **seasonal_stats,
                **stationarity_stats,
                'trend_strength': trend_strength,        # New
                'sample_entropy': sample_entropy,        # New
                'rolling_volatility': rolling_std        # New
            }
            
            # Add difficulty classification (existing)
            combined_stats['forecasting_difficulty'] = classify_forecasting_difficulty(combined_stats)
            
            results.append(combined_stats)
    
    return pd.DataFrame(results)

def create_comprehensive_eda_table1():
    """Create comprehensive EDA summary table for all areas"""
    filepath = '../data/combined_water_data.csv'
    results = []
    
    for area_id in range(1, 30):
        df, missing_count = load_time_series(filepath, area_id)
        if df is not None:
            # Basic stats
            basic_stats = {
                'area_id': area_id,
                'n_obs': len(df),
                'missing_count': missing_count,
                'mean': df['y'].mean(),
                'std': df['y'].std(),
                'cv': df['y'].std() / df['y'].mean(),
                'skewness': df['y'].skew(),
                'min_value': df['y'].min(),
                'max_value': df['y'].max()
            }
            
            # Component analysis
            component_stats = calculate_component_variance_contribution(df, area_id)
            
            # Seasonal analysis
            seasonal_stats = analyze_seasonal_patterns(df, area_id)
            
            # Stationarity analysis
            stationarity_stats = perform_stationarity_tests(df, area_id)
            
            # Combine all stats
            combined_stats = {**basic_stats, **component_stats, **seasonal_stats, **stationarity_stats}
            
            # Add difficulty classification
            combined_stats['forecasting_difficulty'] = classify_forecasting_difficulty(combined_stats)
            
            results.append(combined_stats)
    
    return pd.DataFrame(results)

def create_correlation_heatmap(filepath='../data/combined_water_data.csv'):
    """Create correlation heatmap between areas"""
    # Load all area data
    area_data = {}
    for area_id in range(1, 30):
        df, _ = load_time_series(filepath, area_id)
        if df is not None:
            area_data[f'Area_{area_id}'] = df.set_index('ds')['y']
    
    # Combine into single dataframe
    correlation_df = pd.DataFrame(area_data)
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Inter-Area Water Consumption Correlations')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def analyze_weather_correlations(df_with_weather, area_id):
    """Analyze correlations between consumption and weather variables"""
    correlations = {}
    weather_vars = ['temperature_regressor', 'humidity_regressor', 'rainfall_regressor']
    
    for var in weather_vars:
        if var in df_with_weather.columns:
            corr = df_with_weather['y'].corr(df_with_weather[var])
            correlations[var.replace('_regressor', '')] = corr
    
    return correlations

def create_predictability_ranking_table(comprehensive_stats):
    """Create a ranking table for forecasting predictability"""
    ranking_df = comprehensive_stats[['area_id', 'cv', 'seasonal_strength', 
                                     'predictability_score', 'forecasting_difficulty']].copy()
    
    # Create composite predictability score
    ranking_df['composite_score'] = (
        (1 - ranking_df['cv']) * 0.4 +  # Lower CV is better
        ranking_df['seasonal_strength'] * 0.3 +  # Higher seasonal strength is better  
        ranking_df['predictability_score'] * 0.3  # Higher autocorr is better
    )
    
    ranking_df = ranking_df.sort_values('composite_score', ascending=False)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)
    
    return ranking_df

def decomposition_analysis():
    #############################################################################
    ################ New run 8/7/2025  ##########################################
    #############################################################################
    # First, run your comprehensive EDA analysis
    print_options()
    comprehensive_stats = create_comprehensive_eda_table()
    
    # Generate variance decomposition summary statistics
    variance_stats = comprehensive_stats[['area_id', 'trend_contribution', 'seasonal_contribution', 
                                        'residual_contribution', 'explained_variance']]
    
    print("=== VARIANCE DECOMPOSITION SUMMARY ===")
    print("\nOverall Statistics:")
    print(f"Seasonal contribution - Mean: {variance_stats['seasonal_contribution'].mean():.1f}%, "
          f"Range: {variance_stats['seasonal_contribution'].min():.1f}%-{variance_stats['seasonal_contribution'].max():.1f}%")
    print(f"Trend contribution - Mean: {variance_stats['trend_contribution'].mean():.1f}%, "
          f"Range: {variance_stats['trend_contribution'].min():.1f}%-{variance_stats['trend_contribution'].max():.1f}%")
    print(f"Residual contribution - Mean: {variance_stats['residual_contribution'].mean():.1f}%, "
          f"Range: {variance_stats['residual_contribution'].min():.1f}%-{variance_stats['residual_contribution'].max():.1f}%")
    
    # Categorize areas by explained variance
    high_predictability = variance_stats[variance_stats['explained_variance'] > 80]
    moderate_predictability = variance_stats[(variance_stats['explained_variance'] >= 60) & 
                                            (variance_stats['explained_variance'] <= 80)]
    low_predictability = variance_stats[variance_stats['explained_variance'] < 60]
    
    print(f"\nArea Categories by Explained Variance:")
    print(f"High predictability (>80%): {len(high_predictability)} areas")
    print(f"Moderate predictability (60-80%): {len(moderate_predictability)} areas")
    print(f"Low predictability (<60%): {len(low_predictability)} areas")
    
    # Find dominant patterns
    trend_dominated = variance_stats[variance_stats['trend_contribution'] > 40]
    seasonal_dominated = variance_stats[variance_stats['seasonal_contribution'] > 40]
    noise_dominated = variance_stats[variance_stats['residual_contribution'] > 50]
    
    print(f"\nDominant Component Patterns:")
    print(f"Trend-dominated (>40% trend): {len(trend_dominated)} areas")
    print(f"Seasonal-dominated (>40% seasonal): {len(seasonal_dominated)} areas")
    print(f"Noise-dominated (>50% residual): {len(noise_dominated)} areas")
    
    # Find specific examples
    best_area = variance_stats.loc[variance_stats['explained_variance'].idxmax()]
    worst_area = variance_stats.loc[variance_stats['explained_variance'].idxmin()]
    most_seasonal = variance_stats.loc[variance_stats['seasonal_contribution'].idxmax()]
    
    print(f"\nSpecific Examples:")
    print(f"Most predictable area: {best_area['area_id']} (explained variance: {best_area['explained_variance']:.1f}%)")
    print(f"Most challenging area: {worst_area['area_id']} (explained variance: {worst_area['explained_variance']:.1f}%)")
    print(f"Most seasonal area: {most_seasonal['area_id']} (seasonal contribution: {most_seasonal['seasonal_contribution']:.1f}%)")
    
    # Display the full table for reference
    print(f"\nFull Variance Decomposition Table:")
    print(variance_stats.round(1))
    
    ################ visulization ##############################################
    # 1. Stacked bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Stacked bar chart
    areas = variance_stats['area_id']
    ax1.bar(areas, variance_stats['trend_contribution'], label='Trend', alpha=0.8)
    ax1.bar(areas, variance_stats['seasonal_contribution'], 
            bottom=variance_stats['trend_contribution'], label='Seasonal', alpha=0.8)
    ax1.bar(areas, variance_stats['residual_contribution'], 
            bottom=variance_stats['trend_contribution'] + variance_stats['seasonal_contribution'], 
            label='Residual', alpha=0.8)
    ax1.set_xlabel('Area ID')
    ax1.set_ylabel('Variance Contribution (%)')
    #ax1.set_title('Variance Decomposition by Area')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Box plot of contributions
    components_data = pd.melt(variance_stats[['trend_contribution', 'seasonal_contribution', 'residual_contribution']], 
                             var_name='Component', value_name='Contribution')
    
    label_map = {
        'trend_contribution': 'Trend',
        'seasonal_contribution': 'Seasonal',
        'residual_contribution': 'Residual'
    }
    
    components_data['Component'] = components_data['Component'].map(label_map)
    sns.boxplot(data=components_data, x='Component', y='Contribution', ax=ax2)
    #ax2.set_title('Distribution of Variance Contributions')
    ax2.set_ylabel('Contribution (%)')
    
    plt.tight_layout()

    output_dir = Path("../results/eda")
    output_dir.mkdir(parents=True, exist_ok=True)  # Creates 'eda' if missing (and parents if needed)
    output_path = output_dir / "variance_decomposition.png"  # Construct full path
    plt.savefig(output_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Figure saved to: {output_path}")    
    return comprehensive_stats

def seasonal_characteristics_dist(comprehensive_stats):
    ##############################################################################
    ##############################################################################
    ###############Seasonal Characteristics Distribution
    # Extract seasonal characteristics from  comprehensive_stats
    seasonal_stats = comprehensive_stats[['area_id', 'seasonal_amplitude', 'seasonal_strength', 
                                        'cv', 'peak_month', 'trough_month', 'summer_winter_ratio']]
    
    print("=== SEASONAL CHARACTERISTICS SUMMARY ===")
    print("\nDescriptive Statistics:")
    print(f"Seasonal Amplitude - Mean: {seasonal_stats['seasonal_amplitude'].mean():.1f} m³/month, "
          f"Range: {seasonal_stats['seasonal_amplitude'].min():.1f}-{seasonal_stats['seasonal_amplitude'].max():.1f} m³/month")
    print(f"Seasonal Strength - Mean: {seasonal_stats['seasonal_strength'].mean():.3f}, "
          f"Range: {seasonal_stats['seasonal_strength'].min():.3f}-{seasonal_stats['seasonal_strength'].max():.3f}")
    print(f"Coefficient of Variation - Mean: {seasonal_stats['cv'].mean():.3f}, "
          f"Range: {seasonal_stats['cv'].min():.3f}-{seasonal_stats['cv'].max():.3f}")
    
    # Categorize areas by seasonal strength
    high_seasonal = seasonal_stats[seasonal_stats['seasonal_strength'] > 0.3]
    medium_seasonal = seasonal_stats[(seasonal_stats['seasonal_strength'] >= 0.15) & 
                                    (seasonal_stats['seasonal_strength'] <= 0.3)]
    low_seasonal = seasonal_stats[seasonal_stats['seasonal_strength'] < 0.15]
    
    print(f"\nSeasonal Strength Categories:")
    print(f"High seasonality (>0.3): {len(high_seasonal)} areas - {list(high_seasonal['area_id'].values)}")
    print(f"Medium seasonality (0.15-0.3): {len(medium_seasonal)} areas - {list(medium_seasonal['area_id'].values)}")
    print(f"Low seasonality (<0.15): {len(low_seasonal)} areas - {list(low_seasonal['area_id'].values)}")
    
    # Peak/trough month analysis
    print(f"\nPeak Month Distribution:")
    peak_counts = seasonal_stats['peak_month'].value_counts().sort_index()
    print(peak_counts)
    
    print(f"\nTrough Month Distribution:")
    trough_counts = seasonal_stats['trough_month'].value_counts().sort_index()
    print(trough_counts)
    
    # Summer/Winter ratio analysis
    print(f"\nSummer/Winter Ratio - Mean: {seasonal_stats['summer_winter_ratio'].mean():.2f}, "
          f"Range: {seasonal_stats['summer_winter_ratio'].min():.2f}-{seasonal_stats['summer_winter_ratio'].max():.2f}")
    
    # Find specific examples
    highest_amplitude = seasonal_stats.loc[seasonal_stats['seasonal_amplitude'].idxmax()]
    lowest_amplitude = seasonal_stats.loc[seasonal_stats['seasonal_amplitude'].idxmin()]
    strongest_seasonal = seasonal_stats.loc[seasonal_stats['seasonal_strength'].idxmax()]
    
    print(f"\nSpecific Examples:")
    print(f"Highest amplitude: Area {highest_amplitude['area_id']} ({highest_amplitude['seasonal_amplitude']:.1f} m³/month)")
    print(f"Lowest amplitude: Area {lowest_amplitude['area_id']} ({lowest_amplitude['seasonal_amplitude']:.1f} m³/month)")
    print(f"Strongest seasonality: Area {strongest_seasonal['area_id']} (strength: {strongest_seasonal['seasonal_strength']:.3f})")
    
    # Display full table
    print(f"\nFull Seasonal Characteristics Table:")
    print(seasonal_stats.round(3))
    
    ################################
    
    # Create seasonal characteristics visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Seasonal strength histogram
    ax1.hist(seasonal_stats['seasonal_strength'], bins=10, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Seasonal Strength')
    ax1.set_ylabel('Frequency')
    #ax1.set_title('Distribution of Seasonal Strength Across Areas')
    ax1.axvline(seasonal_stats['seasonal_strength'].mean(), color='red', linestyle='--', 
               label=f'Mean: {seasonal_stats["seasonal_strength"].mean():.3f}')
    ax1.legend()
    
    # 2. Seasonal amplitude vs CV scatter plot
    ax2.scatter(seasonal_stats['cv'], seasonal_stats['seasonal_amplitude']/1000, alpha=0.7)
    ax2.set_xlabel('Coefficient of Variation')
    ax2.set_ylabel('Seasonal Amplitude (km³/m)')
    #ax2.set_title('Seasonal Amplitude vs Variability')
    for i, area in enumerate(seasonal_stats['area_id']):
        ax2.annotate(f'{area}', (seasonal_stats.iloc[i]['cv'], seasonal_stats.iloc[i]['seasonal_amplitude']/1000), 
                    fontsize=8, alpha=0.7)
    
    # 3. Peak month distribution
    peak_counts.plot(kind='bar', ax=ax3, alpha=0.7)
    ax3.set_xlabel('Peak Month')
    ax3.set_ylabel('Number of Areas')
    #ax3.set_title('Distribution of Peak Consumption Months')
    ax3.tick_params(axis='x', rotation=0)
    
    # 4. Box plot of seasonal characteristics
    seasonal_data = pd.melt(seasonal_stats[['seasonal_strength', 'cv']], 
                           var_name='Metric', value_name='Value')
    
    label_map = {
    'High Predictability': 'High',
    'Moderate Predictability': 'Moderate',  # Fix typo ("Moderate")
    'Challenging': 'Low'
    }
    
    #seasonal_data['Component'] = components_data['Component'].map(label_map)
    sns.boxplot(data=seasonal_data, x='Metric', y='Value', ax=ax4)
    
    #ax4.set_title('Distribution of Seasonal Metrics')
    # Save the figure
    
    plt.show()   
    plt.tight_layout()
    output_dir = Path("../results/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "seasonality_dist.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=600, facecolor='white')
    plt.close()
    print(f"Figure saved to: {output_path}")
    return seasonal_stats

def stationary_and_autocorrelation (comprehensive_stats):
    # Extract stationarity and autocorrelation data from comprehensive_stats
    stationarity_stats = comprehensive_stats[['area_id', 'adf_statistic', 'adf_pvalue', 
                                            'is_stationary', 'is_diff_stationary', 
                                            'lag1_autocorr', 'predictability_score']]
    
    print("=== STATIONARITY AND AUTOCORRELATION SUMMARY ===")
    
    # ADF test summary
    stationary_count = stationarity_stats['is_stationary'].sum()
    non_stationary_count = len(stationarity_stats) - stationary_count
    diff_stationary_count = stationarity_stats['is_diff_stationary'].sum()
    
    print(f"\nStationarity Test Results:")
    print(f"Stationary areas (ADF p-value < 0.05): {stationary_count} out of {len(stationarity_stats)} ({stationary_count/len(stationarity_stats)*100:.1f}%)")
    print(f"Non-stationary areas: {non_stationary_count} out of {len(stationarity_stats)} ({non_stationary_count/len(stationarity_stats)*100:.1f}%)")
    print(f"First-difference stationary: {diff_stationary_count} out of {len(stationarity_stats)} ({diff_stationary_count/len(stationarity_stats)*100:.1f}%)")
    
    # Autocorrelation summary
    print(f"\nAutocorrelation Statistics:")
    print(f"Lag-1 Autocorrelation - Mean: {stationarity_stats['lag1_autocorr'].mean():.3f}, "
          f"Range: {stationarity_stats['lag1_autocorr'].min():.3f} to {stationarity_stats['lag1_autocorr'].max():.3f}")
    print(f"Predictability Score - Mean: {stationarity_stats['predictability_score'].mean():.3f}, "
          f"Range: {stationarity_stats['predictability_score'].min():.3f} to {stationarity_stats['predictability_score'].max():.3f}")
    
    # Categorize by autocorrelation strength
    high_autocorr = stationarity_stats[stationarity_stats['predictability_score'] > 0.7]
    medium_autocorr = stationarity_stats[(stationarity_stats['predictability_score'] >= 0.4) & 
                                       (stationarity_stats['predictability_score'] <= 0.7)]
    low_autocorr = stationarity_stats[stationarity_stats['predictability_score'] < 0.4]
    
    print(f"\nAutocorrelation Categories:")
    print(f"High predictability (>0.7): {len(high_autocorr)} areas - {list(high_autocorr['area_id'].values)}")
    print(f"Medium predictability (0.4-0.7): {len(medium_autocorr)} areas - {list(medium_autocorr['area_id'].values)}")
    print(f"Low predictability (<0.4): {len(low_autocorr)} areas - {list(low_autocorr['area_id'].values)}")
    
    # Relationship between stationarity and other characteristics
    print(f"\nStationarity vs Characteristics:")
    stationary_areas = stationarity_stats[stationarity_stats['is_stationary'] == True]
    non_stationary_areas = stationarity_stats[stationarity_stats['is_stationary'] == False]
    
    if len(stationary_areas) > 0:
        print(f"Stationary areas - Mean autocorr: {stationary_areas['lag1_autocorr'].mean():.3f}")
    if len(non_stationary_areas) > 0:
        print(f"Non-stationary areas - Mean autocorr: {non_stationary_areas['lag1_autocorr'].mean():.3f}")
    
    # Find specific examples
    highest_autocorr = stationarity_stats.loc[stationarity_stats['predictability_score'].idxmax()]
    lowest_autocorr = stationarity_stats.loc[stationarity_stats['predictability_score'].idxmin()]
    most_significant_adf = stationarity_stats.loc[stationarity_stats['adf_pvalue'].idxmin()]
    
    print(f"\nSpecific Examples:")
    print(f"Highest predictability: Area {highest_autocorr['area_id']} (autocorr: {highest_autocorr['lag1_autocorr']:.3f})")
    print(f"Lowest predictability: Area {lowest_autocorr['area_id']} (autocorr: {lowest_autocorr['lag1_autocorr']:.3f})")
    print(f"Most stationary: Area {most_significant_adf['area_id']} (p-value: {most_significant_adf['adf_pvalue']:.4f})")
    
    # Display full table
    print(f"\nFull Stationarity and Autocorrelation Table:")
    display_cols = ['area_id', 'adf_pvalue', 'is_stationary', 'lag1_autocorr', 'predictability_score']
    print(stationarity_stats[display_cols].round(3))
    
    ######## visulization ####################
    
    # Create stationarity and autocorrelation visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Stationarity distribution (pie chart)
    stationary_counts = [stationary_count, non_stationary_count]
    labels = ['Stationary', 'Non-Stationary']
    ax1.pie(stationary_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    #ax1.set_title('Distribution of Stationarity Test Results')
    
    # 2. Autocorrelation distribution (histogram)
    ax2.hist(stationarity_stats['lag1_autocorr'], bins=10, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Lag-1 Autocorrelation')
    ax2.set_ylabel('Frequency')
    #ax2.set_title('Distribution of Lag-1 Autocorrelation')
    ax2.axvline(stationarity_stats['lag1_autocorr'].mean(), color='red', linestyle='--', 
               label=f'Mean: {stationarity_stats["lag1_autocorr"].mean():.3f}')
    ax2.legend()
    
    # 3. ADF test p-values
    ax3.scatter(range(len(stationarity_stats)), stationarity_stats['adf_pvalue'], alpha=0.7)
    ax3.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
    ax3.set_xlabel('Area ID')
    ax3.set_ylabel('ADF Test P-value')
    #ax3.set_title('ADF Test P-values by Area')
    ax3.legend()
    ax3.set_yscale('log')
    
    # 4. Stationarity vs Autocorrelation
    stationary_mask = stationarity_stats['is_stationary']
    ax4.scatter(stationarity_stats.loc[stationary_mask, 'lag1_autocorr'], 
               stationarity_stats.loc[stationary_mask, 'adf_pvalue'], 
               alpha=0.7, label='Stationary', s=60)
    ax4.scatter(stationarity_stats.loc[~stationary_mask, 'lag1_autocorr'], 
               stationarity_stats.loc[~stationary_mask, 'adf_pvalue'], 
               alpha=0.7, label='Non-Stationary', s=60)
    ax4.set_xlabel('Lag-1 Autocorrelation')
    ax4.set_ylabel('ADF P-value')
    #ax4.set_title('Stationarity vs Autocorrelation')
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    ax4.set_yscale('log')
    ax4.legend()
    
    plt.tight_layout()
    # Save the figure
    output_dir = Path("../results/eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "autocorrelation-stationarity.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"Figure saved to: {output_path}")
    return stationarity_stats

def test_cv_weights(difficulty_stats):
    weight_combinations = [
        (0.4, 0.3, 0.3),  # Current
        (0.5, 0.25, 0.25),  # Increase CV
        (0.6, 0.2, 0.2),   # Higher CV emphasis
        (0.7, 0.15, 0.15)  # Strong CV emphasis
    ]
    
    for cv_w, ss_w, ps_w in weight_combinations:
        new_score = ((1 - difficulty_stats['cv']) * cv_w + 
                    difficulty_stats['seasonal_strength'] * ss_w + 
                    difficulty_stats['predictability_score'] * ps_w)
        
        print(f"\nWeights: CV={cv_w}, SS={ss_w}, PS={ps_w}")
        print(f"Score range: {new_score.min():.3f} - {new_score.max():.3f}")
        print(f"Standard deviation: {new_score.std():.3f}")
        
        # Check how many would exceed 0.7 threshold
        easy_count = (new_score > 0.7).sum()
        print(f"Areas >0.7 threshold: {easy_count}")

def forcasting_difficulty():
    # Extract forecasting difficulty data from comprehensive_stats
    comprehensive_stats = create_comprehensive_eda_table()
    # Extract forecasting difficulty data from comprehensive_stats
    difficulty_stats = comprehensive_stats[['area_id', 'cv', 'seasonal_strength', 'predictability_score', 
                                          'forecasting_difficulty']].copy()
    
    #test_cv_weights(difficulty_stats)
    
    # Calculate the composite score manually based on your create_predictability_ranking_table function
    difficulty_stats['composite_score'] = (
        (1 - difficulty_stats['cv']) * 0.7 +  # Lower CV is better
        difficulty_stats['seasonal_strength'] * 0.15 +  # Higher seasonal strength is better  
        difficulty_stats['predictability_score'] * 0.15  # Higher autocorr is better
    )
    
    print("=== FORECASTING DIFFICULTY CLASSIFICATION OUTCOMES ===")
    
    # Distribution across difficulty categories
    difficulty_counts = difficulty_stats['forecasting_difficulty'].value_counts()
    print(f"\nDistribution of Areas Across Difficulty Categories:")
    for category, count in difficulty_counts.items():
        percentage = count / len(difficulty_stats) * 100
        areas_in_category = difficulty_stats[difficulty_stats['forecasting_difficulty'] == category]['area_id'].tolist()
        print(f"{category}: {count} areas ({percentage:.1f}%) - Areas: {areas_in_category}")
    
    # Composite score statistics
    print(f"\nComposite Score Distribution:")
    print(f"Mean: {difficulty_stats['composite_score'].mean():.3f}")
    print(f"Range: {difficulty_stats['composite_score'].min():.3f} - {difficulty_stats['composite_score'].max():.3f}")
    print(f"Standard Deviation: {difficulty_stats['composite_score'].std():.3f}")
    
    # Characteristics by difficulty level
    print(f"\nArea Characteristics by Difficulty Level:")
    for category in difficulty_counts.index:
        subset = difficulty_stats[difficulty_stats['forecasting_difficulty'] == category]
        print(f"\n{category} Areas (n={len(subset)}):")
        print(f"  Mean CV: {subset['cv'].mean():.3f} (Range: {subset['cv'].min():.3f}-{subset['cv'].max():.3f})")
        print(f"  Mean Seasonal Strength: {subset['seasonal_strength'].mean():.3f} (Range: {subset['seasonal_strength'].min():.3f}-{subset['seasonal_strength'].max():.3f})")
        print(f"  Mean Predictability: {subset['predictability_score'].mean():.3f} (Range: {subset['predictability_score'].min():.3f}-{subset['predictability_score'].max():.3f})")
        print(f"  Mean Composite Score: {subset['composite_score'].mean():.3f} (Range: {subset['composite_score'].min():.3f}-{subset['composite_score'].max():.3f})")
    
    # Find specific examples
    highest_score = difficulty_stats.loc[difficulty_stats['composite_score'].idxmax()]
    lowest_score = difficulty_stats.loc[difficulty_stats['composite_score'].idxmin()]
    
    print(f"\nSpecific Examples:")
    print(f"Highest composite score: Area {highest_score['area_id']} - {highest_score['forecasting_difficulty']} "
          f"(Score: {highest_score['composite_score']:.3f})")
    print(f"  CV: {highest_score['cv']:.3f}, Seasonal Strength: {highest_score['seasonal_strength']:.3f}, "
          f"Predictability: {highest_score['predictability_score']:.3f}")
    
    print(f"Lowest composite score: Area {lowest_score['area_id']} - {lowest_score['forecasting_difficulty']} "
          f"(Score: {lowest_score['composite_score']:.3f})")
    print(f"  CV: {lowest_score['cv']:.3f}, Seasonal Strength: {lowest_score['seasonal_strength']:.3f}, "
          f"Predictability: {lowest_score['predictability_score']:.3f}")
    
    # Check what your actual difficulty categories are
    print(f"\nActual Difficulty Categories Found:")
    print(difficulty_counts)
    
    # Full classification table sorted by composite score
    print(f"\nFull Forecasting Difficulty Classification Table:")
    display_cols = ['area_id', 'cv', 'seasonal_strength', 'predictability_score', 'composite_score', 'forecasting_difficulty']
    print(difficulty_stats[display_cols].round(3).sort_values('composite_score', ascending=False))
    
    ############# visulization 
    
    # Create forecasting difficulty visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Distribution pie chart - improved with explode and cleaner labels
    explode = (0.05, 0.05, 0.05)  # Slightly separate slices
    short_labels = {'High Predictability': 'High', 
                    'Moderate Predictability': 'Moderate',
                    'Challenging': 'Low'}
    difficulty_counts.rename(index=short_labels).plot(
        kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90,
        explode=explode, shadow=True, textprops={'fontsize': 10}
    )
    ax1.set_ylabel('')  # Remove redundant ylabel
    
    # 2. Composite score histogram - improved formatting
    ax2.hist(difficulty_stats['composite_score'], bins=15, alpha=0.7, 
             edgecolor='black', color='skyblue')
    ax2.axvline(x=0.4, color='red', linestyle='--', linewidth=1.5, 
                label='Low/Moderate (0.4)')
    ax2.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5, 
                label='Moderate/High (0.7)')
    ax2.set_xlabel('Composite Score')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # 3. Score components by difficulty category - improved with short labels
    categories = difficulty_stats['forecasting_difficulty'].map(short_labels).unique()
    metrics = ['cv', 'seasonal_strength', 'predictability_score']
    metric_labels = ['CV', 'Seasonality', 'Predictability']  # Cleaner names
    x_pos = np.arange(len(categories))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = [difficulty_stats[difficulty_stats['forecasting_difficulty'].map(short_labels) == cat][metric].mean() 
                 for cat in categories]
        ax3.bar(x_pos + i*width, means, width, label=label, alpha=0.8)
    
    ax3.set_xlabel('Difficulty Level')
    ax3.set_ylabel('Mean Score')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(categories)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. Scatter plot - improved with consistent styling
    colors = {'High': 'green', 'Moderate': 'orange', 'Low': 'red'}
    for category in difficulty_stats['forecasting_difficulty'].map(short_labels).unique():
        subset = difficulty_stats[difficulty_stats['forecasting_difficulty'].map(short_labels) == category]
        ax4.scatter(subset['cv'], subset['composite_score'], 
                   c=colors[category], label=category, alpha=0.7, s=60)
        
        # Annotate examples more cleanly
        for idx, row in subset.head(2).iterrows():
            ax4.annotate(row['area_id'], (row['cv'], row['composite_score']), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    ax4.set_xlabel('Coefficient of Variation (CV)')
    ax4.set_ylabel('Composite Score')
    ax4.legend(title='Difficulty')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    
    # Save the figure

    import os
    output_path = RESULTS_DIR / "eda"
    output_path.mkdir(parents=True, exist_ok=True)
    # But add debug check
    print(f"EDA directory exists: {output_path.exists()}")
    print(f"EDA directory is writable: {os.access(output_path, os.W_OK)}")

    output_path = output_path / "forecasting_difficulty.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=600, facecolor='white')
    plt.close()
    print(f"Figure saved to: {output_path}")

def generate_eda_figures():
    """
    Generate exploratory data analysis figures.
    
    Produces:
    - Figure X: Time series decomposition examples
    - Figure Y: Seasonal characteristics distribution
    - Figure Z: Stationarity and autocorrelation analysis
    - Table X: Summary statistics
    
    Output directory: figures/eda/
    """
    print("\n" + "="*70)
    print("STEP 0: Exploratory Data Analysis")
    print("        (Data characteristics, seasonality, stationarity)")
    print("="*70)
    
    print("\n[1/3] Running decomposition analysis...")
    comprehensive_stats = decomposition_analysis()
    
    print("\n[2/3] Analyzing seasonal characteristics...")
    seasonal_stats = seasonal_characteristics_dist(comprehensive_stats)
    
    print("\n[3/3] Testing stationarity and autocorrelation...")
    stationarity_stats = stationary_and_autocorrelation(comprehensive_stats)
    
    print("\n Step 0 complete: EDA figures generated")
    
    return {
        'comprehensive_stats': comprehensive_stats,
        'seasonal_stats': seasonal_stats,
        'stationarity_stats': stationarity_stats
    }


if __name__ == '__main__':
    # For standalone testing
    print_options() 
    generate_eda_figures()


