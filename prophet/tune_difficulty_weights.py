import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from exploratory_analysis import  create_comprehensive_eda_table

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
EDA_DIR = RESULTS_DIR / 'eda'
EDA_DIR.mkdir(exist_ok=True, parents=True)

def optimal_weights_r2_weighted(comprehensive_stats, performance_df):
    """
    Use linear regression to find optimal weights using R²-weighted averaging across metrics
    """  
    # Add area_id to performance_df if not present (assuming same order)
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    # Merge with performance data
    final_df = comprehensive_stats.merge(performance_df_with_id, on='area_id')
    
    # Prepare features and targets
    X = final_df[['cv', 'seasonal_strength', 'lag1_autocorr']].copy()
    y_mape = final_df['mape'].copy()
    y_rmse = final_df['rmse'].copy()
    y_mase = final_df['mase'].copy()
    
    results = {}
    individual_weights = {}
    r2_values = {}
    
    for target, y in [('mape', y_mape), ('rmse', y_rmse), ('mase', y_mase)]:
        # Standardize features for comparable coefficients
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Method 1: Sklearn for basic results
        reg = LinearRegression()
        reg.fit(X_scaled, y)
        
        # Method 2: Statsmodels for detailed statistics
        X_scaled_with_const = sm.add_constant(X_scaled_df)
        model = sm.OLS(y, X_scaled_with_const).fit()
        
        # Calculate weights (normalize to sum to 1)
        raw_weights = np.abs(reg.coef_)  # Use absolute values
        normalized_weights = raw_weights / raw_weights.sum()
        
        # Store individual results
        r2_score_val = r2_score(y, reg.predict(X_scaled))
        individual_weights[target] = normalized_weights
        r2_values[target] = r2_score_val
        
        # Store results
        results[target] = {
            'raw_coefficients': reg.coef_,
            'normalized_weights': normalized_weights,
            'r2_score': r2_score_val,
            'statsmodel_summary': model.summary(),
            'feature_names': X.columns.tolist(),
            'intercept': reg.intercept_
        }
        
        print(f"\n=== {target.upper()} Results ===")
        print(f"R² Score: {r2_score_val:.3f}")
        print("Raw Coefficients:")
        for name, coef in zip(X.columns, reg.coef_):
            print(f"  {name}: {coef:.3f}")
        print("Individual Weights (normalized):")
        for name, weight in zip(X.columns, normalized_weights):
            print(f"  {name}: {weight:.3f}")
    
    # Calculate R²-weighted average weights
    total_r2 = sum(r2_values.values())
    
    if total_r2 > 0:  # Avoid division by zero
        r2_weighted_weights = np.zeros(3)  # cv, seasonal_strength, lag1_autocorr
        
        for target in ['mape', 'rmse', 'mase']:
            weight_contribution = r2_values[target] / total_r2
            r2_weighted_weights += individual_weights[target] * weight_contribution
        
        print(f"\n{'='*60}")
        print("R²-WEIGHTED OPTIMAL WEIGHTS")
        print(f"{'='*60}")
        print("R² values:")
        for target, r2_val in r2_values.items():
            print(f"  {target.upper()}: {r2_val:.3f} (weight: {r2_val/total_r2:.3f})")
        
        print(f"\nR²-weighted optimal weights:")
        feature_names = ['cv', 'seasonal_strength', 'lag1_autocorr']
        for name, weight in zip(feature_names, r2_weighted_weights):
            print(f"  {name}: {weight:.3f}")
        
        # Store the optimal weights
        results['r2_weighted_optimal'] = {
            'weights': r2_weighted_weights,
            'feature_names': feature_names,
            'r2_values': r2_values,
            'total_r2': total_r2
        }
    
    return results

def test_r2_weighted_composite_score(comprehensive_stats, performance_df, regression_results):
    """
    Test the statistical significance of the R²-weighted composite score
    """
    
    # Get the R²-weighted optimal weights
    optimal_weights = regression_results['r2_weighted_optimal']['weights']
    
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = comprehensive_stats.merge(performance_df_with_id, on='area_id')
    
    # Standardize components
    scaler = StandardScaler()
    components = final_df[['cv', 'seasonal_strength', 'lag1_autocorr']].copy()
    components_scaled = scaler.fit_transform(components)
    
    # Calculate composite score with R²-weighted optimal weights
    composite_score = np.dot(components_scaled, optimal_weights)
    
    print(f"\n{'='*60}")
    print(f"R²-WEIGHTED COMPOSITE SCORE VALIDATION")
    print(f"Weights: CV={optimal_weights[0]:.3f}, SS={optimal_weights[1]:.3f}, AC={optimal_weights[2]:.3f}")
    print(f"{'='*60}")
    
    results = {}
    
    for target in ['mape', 'rmse', 'mase']:
        y = final_df[target].values
        
        # Calculate correlation and p-value
        correlation, p_value = pearsonr(composite_score, y)
        
        results[target] = {
            'correlation': correlation,
            'p_value': p_value,
            'composite_score': composite_score
        }
        
        # Significance stars
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        elif p_value < 0.10:
            sig = "+"
        else:
            sig = ""
        
        print(f"\n{target.upper()}:")
        print(f"  Correlation: r = {correlation:.3f}{sig}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.001:
            print(f"  Significance: Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"  Significance: Very significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"  Significance: Significant (p < 0.05)")
        elif p_value < 0.10:
            print(f"  Significance: Marginally significant (p < 0.10)")
        else:
            print(f"  Significance: Not significant (p ≥ 0.10)")
    
    return results, optimal_weights

def compare_weight_schemes(comprehensive_stats, performance_df, optimal_weights):
    """
    Compare different weighting schemes including R²-weighted optimal
    """
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = comprehensive_stats.merge(performance_df_with_id, on='area_id')
    
    # Standardize components
    scaler = StandardScaler()
    components = final_df[['cv', 'seasonal_strength', 'lag1_autocorr']].copy()
    components_scaled = scaler.fit_transform(components)
    
    # Different weight schemes to test (removed fixed vector approach)
    weight_schemes = {
        'R²-Weighted Optimal': tuple(optimal_weights),
        'Current (Manual)': (0.400, 0.300, 0.300),
        'Best Manual': (0.300, 0.500, 0.200),
        'Equal Weights': (0.333, 0.333, 0.333),
        'CV Dominant': (0.600, 0.200, 0.200),
        'SS Dominant': (0.200, 0.600, 0.200)
    }
    
    print(f"\n{'='*80}")
    print(f"COMPARISON OF WEIGHT SCHEMES")
    print(f"{'='*80}")
    print(f"{'Scheme':<20} {'MAPE r':<10} {'MAPE p':<10} {'RMSE r':<10} {'RMSE p':<10} {'MASE r':<10} {'MASE p':<10}")
    print("-" * 80)
    
    comparison_results = {}
    
    for scheme_name, (w_cv, w_ss, w_ac) in weight_schemes.items():
        # Calculate composite score
        composite = (w_cv * components_scaled[:, 0] + 
                    w_ss * components_scaled[:, 1] + 
                    w_ac * components_scaled[:, 2])
        
        scheme_results = {}
        correlations = []
        
        for target in ['mape', 'rmse', 'mase']:
            y = final_df[target].values
            correlation, p_value = pearsonr(composite, y)
            scheme_results[target] = {'correlation': correlation, 'p_value': p_value}
            correlations.append(abs(correlation))
        
        scheme_results['avg_abs_correlation'] = np.mean(correlations)
        comparison_results[scheme_name] = scheme_results
        
        # Print results
        print(f"{scheme_name:<20} "
              f"{scheme_results['mape']['correlation']:>9.3f} "
              f"{scheme_results['mape']['p_value']:>9.3f} "
              f"{scheme_results['rmse']['correlation']:>9.3f} "
              f"{scheme_results['rmse']['p_value']:>9.3f} "
              f"{scheme_results['mase']['correlation']:>9.3f} "
              f"{scheme_results['mase']['p_value']:>9.3f}")
    
    # Find best performing scheme
    best_scheme = max(comparison_results.items(), 
                     key=lambda x: x[1]['avg_abs_correlation'])
    
    print(f"\nBest performing scheme: {best_scheme[0]}")
    print(f"Average |correlation|: {best_scheme[1]['avg_abs_correlation']:.3f}")
    
    return comparison_results

def fitting_curve(comprehensive_stats, performance_df, optimal_weights, 
                  save_path=None, dpi=600):
    """
    Create fitting curve with the final difficulty score equation displayed
    """
   
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = comprehensive_stats.merge(performance_df_with_id, on='area_id')
    
    # Standardize and calculate optimal composite score
    scaler = StandardScaler()
    components_scaled = scaler.fit_transform(final_df[['cv', 'seasonal_strength', 'lag1_autocorr']])
    
    optimal_composite = np.dot(components_scaled, optimal_weights)
    optimal_corr, optimal_p = pearsonr(optimal_composite, final_df['mape'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(optimal_composite, final_df['mape'], alpha=0.7, s=60, 
               color='#2E8B57', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(optimal_composite, final_df['mape'], 1)
    p = np.poly1d(z)
    ax.plot(optimal_composite, p(optimal_composite), "r--", alpha=0.8, linewidth=2)
    
    # Create the difficulty score equation text
    equation_text = (f'Difficulty Score = {optimal_weights[0]:.3f} × CV + '
                    f'{optimal_weights[1]:.3f} × SS + '
                    f'{optimal_weights[2]:.3f} × AC')
    
    # Add statistics and equation
    stats_text = (f'r = {optimal_corr:.3f}***\n'
                 f'p = {optimal_p:.6f}\n'
                 f'R² = {optimal_corr**2:.3f}\n\n'
                 f'{equation_text}')
    
    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.8),
            fontsize=10, family='monospace')
    
    ax.set_xlabel('Difficulty Score')
    ax.set_ylabel('MAPE (%)')
    ax.grid(alpha=0.3)
    #ax.set_title('Difficulty Score vs. Forecasting Performance\n(R²-Weighted Optimal Weights)', 
    #            fontweight='bold')
    
    plt.tight_layout()

    if save_path:
        output_path = save_path / 'r2_weighted_fitting.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi, facecolor='white')
        plt.close()
        print(f"Figure saved to: {output_path}")
    
    plt.show()
    return fig

def weight_comparison_r2(comprehensive_stats, performance_df, 
                                    optimal_weights, save_path=None, dpi=300):
    """
    Create comparison of weight schemes including R²-weighted optimal
    """
   
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = comprehensive_stats.merge(performance_df_with_id, on='area_id')
    
    # Standardize and calculate correlations
    scaler = StandardScaler()
    components_scaled = scaler.fit_transform(final_df[['cv', 'seasonal_strength', 'lag1_autocorr']])
    
    weight_schemes = {
        'R²-Weighted Optimal': tuple(optimal_weights),
        'SS Dominant': (0.200, 0.600, 0.200),
        'Best Manual': (0.300, 0.500, 0.200),
        'Current Manual': (0.400, 0.300, 0.300),
        'Equal Weights': (0.333, 0.333, 0.333),
        'CV Dominant': (0.600, 0.200, 0.200)
    }
    
    # Calculate all correlations
    results = {}
    for scheme_name, weights in weight_schemes.items():
        composite = np.dot(components_scaled, weights)
        mape_corr, mape_p = pearsonr(composite, final_df['mape'])
        
        if mape_p < 0.001:
            sig = "***"
        elif mape_p < 0.01:
            sig = "**"
        elif mape_p < 0.05:
            sig = "*"
        else:
            sig = ""
        
        results[scheme_name] = {
            'correlation': mape_corr,
            'p_value': mape_p,
            'significance': sig
        }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Bar chart of correlations
    schemes = list(results.keys())
    correlations = [results[s]['correlation'] for s in schemes]
    significances = [results[s]['significance'] for s in schemes]
    
    # Sort by correlation
    sorted_data = sorted(zip(schemes, correlations, significances), 
                        key=lambda x: x[1], reverse=True)
    schemes_sorted, corr_sorted, sig_sorted = zip(*sorted_data)
    
    colors = ['#2E8B57' if 'Optimal' in s else '#4682B4' if 'SS' in s else 
             '#DAA520' if 'Manual' in s else '#708090' for s in schemes_sorted]
    
    bars = ax1.bar(range(len(schemes_sorted)), corr_sorted, color=colors, alpha=0.8)
    
    # Add significance annotations
    for i, (bar, sig) in enumerate(zip(bars, sig_sorted)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, sig,
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_xticks(range(len(schemes_sorted)))
    ax1.set_xticklabels(schemes_sorted, rotation=45, ha='right')
    ax1.set_ylabel('MAPE Correlation', fontweight='bold')
    ax1.set_title('Weight Scheme Performance Comparison', fontweight='bold', fontsize=14)
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel B: Optimal score scatter plot
    optimal_composite = np.dot(components_scaled, optimal_weights)
    optimal_corr, optimal_p = pearsonr(optimal_composite, final_df['mape'])
    
    ax2.scatter(optimal_composite, final_df['mape'], alpha=0.7, s=60, 
               color='#2E8B57', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(optimal_composite, final_df['mape'], 1)
    p = np.poly1d(z)
    ax2.plot(optimal_composite, p(optimal_composite), "r--", alpha=0.8, linewidth=2)
    
    # Create the equation text
    equation_text = (f'Score = {optimal_weights[0]:.3f}×CV + '
                    f'{optimal_weights[1]:.3f}×SS + '
                    f'{optimal_weights[2]:.3f}×AC')
    
    # Add statistics with equation
    stats_text = (f'r = {optimal_corr:.3f}***\n'
                 f'p = {optimal_p:.6f}\n'
                 f'R² = {optimal_corr**2:.3f}\n\n'
                 f'{equation_text}')
    
    ax2.text(0.05, 0.95, stats_text, 
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9, family='monospace')
    
    ax2.set_xlabel('R²-Weighted Optimal Difficulty Score', fontweight='bold')
    ax2.set_ylabel('MAPE (%)', fontweight='bold')
    ax2.set_title('Optimal Score vs. Forecasting Performance', fontweight='bold', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        output_dir = Path("../results/eda")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / save_path
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi, facecolor='white')
        plt.close()
        print(f"Figure saved to: {output_path}")
    
    plt.show()
    return fig

### ===================================================================
### finding difficulty thresholds 
#######################################################################

def calculate_difficulty_scores(comprehensive_stats, optimal_weights):
    """
    Calculate difficulty scores for all areas using optimal weights
    """    
    # Standardize components
    scaler = StandardScaler()
    components = comprehensive_stats[['cv', 'seasonal_strength', 'lag1_autocorr']].copy()
    components_scaled = scaler.fit_transform(components)
    
    # Calculate composite scores
    difficulty_scores = np.dot(components_scaled, optimal_weights)
    
    comprehensive_stats['difficulty_score'] = difficulty_scores
    return comprehensive_stats

def method_1_quantile_based(difficulty_scores, method='tertiles'):
    """
    Method 1: Quantile-based thresholds (IQR or tertiles)
    """
    if method == 'iqr':
        # IQR approach: Low < 25th, High > 75th, Moderate = middle 50%
        threshold_low = np.percentile(difficulty_scores, 25)
        threshold_high = np.percentile(difficulty_scores, 75)
        
        low_areas = len(difficulty_scores[difficulty_scores < threshold_low])
        moderate_areas = len(difficulty_scores[(difficulty_scores >= threshold_low) & 
                                            (difficulty_scores <= threshold_high)])
        high_areas = len(difficulty_scores[difficulty_scores > threshold_high])
        
        print(f"IQR Method:")
        print(f"  Low threshold: {threshold_low:.3f}")
        print(f"  High threshold: {threshold_high:.3f}")
        print(f"  Distribution: Low={low_areas} ({low_areas/len(difficulty_scores)*100:.1f}%), "
              f"Moderate={moderate_areas} ({moderate_areas/len(difficulty_scores)*100:.1f}%), "
              f"High={high_areas} ({high_areas/len(difficulty_scores)*100:.1f}%)")
        
    elif method == 'tertiles':
        # Tertiles: Equal 33.3% splits
        threshold_low = np.percentile(difficulty_scores, 33.33)
        threshold_high = np.percentile(difficulty_scores, 66.67)
        
        low_areas = len(difficulty_scores[difficulty_scores < threshold_low])
        moderate_areas = len(difficulty_scores[(difficulty_scores >= threshold_low) & 
                                            (difficulty_scores <= threshold_high)])
        high_areas = len(difficulty_scores[difficulty_scores > threshold_high])
        
        print(f"Tertiles Method:")
        print(f"  Low threshold: {threshold_low:.3f}")
        print(f"  High threshold: {threshold_high:.3f}")
        print(f"  Distribution: Low={low_areas} ({low_areas/len(difficulty_scores)*100:.1f}%), "
              f"Moderate={moderate_areas} ({moderate_areas/len(difficulty_scores)*100:.1f}%), "
              f"High={high_areas} ({high_areas/len(difficulty_scores)*100:.1f}%)")
    
    return threshold_low, threshold_high

def method_2_performance_based(comprehensive_stats, performance_df, optimal_weights):
    """
    Method 2: Performance-based thresholds using MAPE differences
    """
    # Calculate difficulty scores
    merged_scores = calculate_difficulty_scores(comprehensive_stats, optimal_weights)
    
    # Add performance data
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = merged_scores.merge(performance_df_with_id, on='area_id')
    
    # Sort by difficulty score
    final_df = final_df.sort_values('difficulty_score')
    
    # Test different threshold combinations to maximize MAPE separation
    scores = final_df['difficulty_score'].values
    mape_values = final_df['mape'].values
    
    best_separation = 0
    best_thresholds = None
    best_performance = None
    
    # Generate candidate thresholds
    score_range = scores.max() - scores.min()
    candidates = np.linspace(scores.min() + 0.1*score_range, 
                           scores.max() - 0.1*score_range, 20)
    
    for t1 in candidates:
        for t2 in candidates:
            if t2 <= t1:
                continue
                
            # Create groups
            low_mask = scores < t1
            moderate_mask = (scores >= t1) & (scores < t2)
            high_mask = scores >= t2
            
            # Ensure minimum group sizes
            if sum(low_mask) < 3 or sum(moderate_mask) < 3 or sum(high_mask) < 3:
                continue
            
            # Calculate group means
            low_mape = mape_values[low_mask].mean()
            moderate_mape = mape_values[moderate_mask].mean()
            high_mape = mape_values[high_mask].mean()
            
            # Calculate separation metric
            separation = (high_mape - low_mape) + abs(moderate_mape - (low_mape + high_mape)/2)
            
            if separation > best_separation:
                best_separation = separation
                best_thresholds = (t1, t2)
                best_performance = {
                    'low_mape': low_mape,
                    'moderate_mape': moderate_mape,
                    'high_mape': high_mape,
                    'low_count': sum(low_mask),
                    'moderate_count': sum(moderate_mask),
                    'high_count': sum(high_mask)
                }
    
    print(f"Performance-Based Method:")
    print(f"  Low threshold: {best_thresholds[0]:.3f}")
    print(f"  High threshold: {best_thresholds[1]:.3f}")
    print(f"  MAPE Performance:")
    print(f"    Low difficulty: {best_performance['low_mape']:.2f}% "
          f"(n={best_performance['low_count']})")
    print(f"    Moderate difficulty: {best_performance['moderate_mape']:.2f}% "
          f"(n={best_performance['moderate_count']})")
    print(f"    High difficulty: {best_performance['high_mape']:.2f}% "
          f"(n={best_performance['high_count']})")
    print(f"  MAPE Range: {best_performance['high_mape'] - best_performance['low_mape']:.2f}%")
    
    return best_thresholds, best_performance

def method_3_standard_deviation_based(difficulty_scores):
    """
    Method 3: Standard deviation-based thresholds
    """
    mean_score = np.mean(difficulty_scores)
    std_score = np.std(difficulty_scores)
    
    # ±0.5 standard deviations
    threshold_low = mean_score - 0.5 * std_score
    threshold_high = mean_score + 0.5 * std_score
    
    low_areas = len(difficulty_scores[difficulty_scores < threshold_low])
    moderate_areas = len(difficulty_scores[(difficulty_scores >= threshold_low) & 
                                         (difficulty_scores <= threshold_high)])
    high_areas = len(difficulty_scores[difficulty_scores > threshold_high])
    
    print(f"Standard Deviation Method (±0.5σ):")
    print(f"  Mean: {mean_score:.3f}, Std: {std_score:.3f}")
    print(f"  Low threshold: {threshold_low:.3f}")
    print(f"  High threshold: {threshold_high:.3f}")
    print(f"  Distribution: Low={low_areas} ({low_areas/len(difficulty_scores)*100:.1f}%), "
          f"Moderate={moderate_areas} ({moderate_areas/len(difficulty_scores)*100:.1f}%), "
          f"High={high_areas} ({high_areas/len(difficulty_scores)*100:.1f}%)")
    
    return threshold_low, threshold_high

def method_4_natural_breaks_jenks(difficulty_scores, n_classes=3):
    """
    Method 4: Natural breaks (Jenks) optimization
    """
    try:
        import jenkspy
        breaks = jenkspy.jenks_breaks(difficulty_scores, n_classes=n_classes)
        
        threshold_low = breaks[1]  # First break
        threshold_high = breaks[2]  # Second break
        
        low_areas = len(difficulty_scores[difficulty_scores < threshold_low])
        moderate_areas = len(difficulty_scores[(difficulty_scores >= threshold_low) & 
                                             (difficulty_scores < threshold_high)])
        high_areas = len(difficulty_scores[difficulty_scores >= threshold_high])
        
        print(f"Natural Breaks (Jenks) Method:")
        print(f"  Low threshold: {threshold_low:.3f}")
        print(f"  High threshold: {threshold_high:.3f}")
        print(f"  Distribution: Low={low_areas} ({low_areas/len(difficulty_scores)*100:.1f}%), "
              f"Moderate={moderate_areas} ({moderate_areas/len(difficulty_scores)*100:.1f}%), "
              f"High={high_areas} ({high_areas/len(difficulty_scores)*100:.1f}%)")
        
        return threshold_low, threshold_high
        
    except ImportError:
        print("Jenks method requires 'jenkspy' package. Install with: pip install jenkspy")
        return None, None

def compare_threshold_methods(comprehensive_stats, performance_df, optimal_weights):
    """
    Compare all threshold setting methods
    """
    # Calculate difficulty scores
    merged_scores = calculate_difficulty_scores(comprehensive_stats, optimal_weights)
    difficulty_scores = merged_scores['difficulty_score'].values
    
    print("="*80)
    print("COMPARISON OF THRESHOLD SETTING METHODS")
    print("="*80)
    
    # Method 1a: IQR
    t1_iqr, t2_iqr = method_1_quantile_based(difficulty_scores, method='iqr')
    
    print()
    # Method 1b: Tertiles  
    t1_ter, t2_ter = method_1_quantile_based(difficulty_scores, method='tertiles')
    
    print()
    # Method 2: Performance-based
    (t1_perf, t2_perf), perf_results = method_2_performance_based(
        comprehensive_stats, performance_df, optimal_weights)
    
    print()
    # Method 3: Standard deviation
    t1_std, t2_std = method_3_standard_deviation_based(difficulty_scores)
    
    print()
    # Method 4: Natural breaks (if available)
    t1_jenks, t2_jenks = method_4_natural_breaks_jenks(difficulty_scores)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("THRESHOLD SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Low Threshold':<15} {'High Threshold':<15} {'MAPE Range':<12}")
    print("-" * 80)
    
    methods_results = {
        'IQR (25%/75%)': (t1_iqr, t2_iqr),
        'Tertiles (33%/67%)': (t1_ter, t2_ter),
        'Performance-based': (t1_perf, t2_perf),
        'Std Dev (±0.5σ)': (t1_std, t2_std),
    }
    
    if t1_jenks is not None:
        methods_results['Natural Breaks'] = (t1_jenks, t2_jenks)
    
    # Calculate MAPE ranges for each method
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = merged_scores.merge(performance_df_with_id, on='area_id')
    
    for method_name, (t1, t2) in methods_results.items():
        if method_name == 'Performance-based':
            mape_range = perf_results['high_mape'] - perf_results['low_mape']
        else:
            # Calculate MAPE range for other methods
            low_mask = final_df['difficulty_score'] < t1
            high_mask = final_df['difficulty_score'] >= t2
            
            if sum(low_mask) > 0 and sum(high_mask) > 0:
                low_mape = final_df[low_mask]['mape'].mean()
                high_mape = final_df[high_mask]['mape'].mean()
                mape_range = high_mape - low_mape
            else:
                mape_range = 0
        
        print(f"{method_name:<20} {t1:<15.3f} {t2:<15.3f} {mape_range:<12.2f}%")
    
    return methods_results, perf_results

def visualize_threshold_comparison(difficulty_df, seasonal_df, performance_df, 
                                 optimal_weights, methods_results):
    """
    Create visualization comparing different threshold methods
    """
    # Calculate difficulty scores and merge with performance
    merged_scores = calculate_difficulty_scores(comprehensive_stats, optimal_weights)
    
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = merged_scores.merge(performance_df_with_id, on='area_id')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    methods_to_plot = ['IQR (25%/75%)', 'Tertiles (33%/67%)', 'Performance-based', 
                      'Std Dev (±0.5σ)']
    
    for i, method_name in enumerate(methods_to_plot):
        if method_name in methods_results:
            t1, t2 = methods_results[method_name]
            
            # Create difficulty categories
            categories = []
            colors = []
            for score in final_df['difficulty_score']:
                if score < t1:
                    categories.append('Low')
                    colors.append('#2E8B57')  # Green
                elif score < t2:
                    categories.append('Moderate') 
                    colors.append('#DAA520')  # Gold
                else:
                    categories.append('High')
                    colors.append('#DC143C')  # Red
            
            # Scatter plot
            ax = axes[i]
            for cat, color in [('Low', '#2E8B57'), ('Moderate', '#DAA520'), ('High', '#DC143C')]:
                mask = np.array(categories) == cat
                if sum(mask) > 0:
                    ax.scatter(final_df['difficulty_score'][mask], final_df['mape'][mask], 
                             c=color, label=cat, alpha=0.7, s=60)
            
            # Add threshold lines
            ax.axvline(x=t1, color='black', linestyle='--', alpha=0.7, label=f'T1={t1:.2f}')
            ax.axvline(x=t2, color='black', linestyle=':', alpha=0.7, label=f'T2={t2:.2f}')
            
            ax.set_xlabel('Difficulty Score')
            ax.set_ylabel('MAPE (%)')
            ax.set_title(f'{method_name}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    # Remove empty subplots
    for j in range(len(methods_to_plot), len(axes)):
        fig.delaxes(axes[j])
    
#    plt.tight_layout()
    plt.suptitle('Comparison of Threshold Setting Methods', y=1.02, fontsize=16, fontweight='bold')
    plt.show()
    
    return fig

#####################

def create_classification_dataframe(comprehensive_stats, performance_df, optimal_weights, methods_results):
    """
    Create comprehensive dataframe with difficulty scores and all classification methods
    """
    # Calculate difficulty scores
    merged_scores = calculate_difficulty_scores(comprehensive_stats, optimal_weights)
    
    # Add performance data
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    # Merge with performance metrics
    final_df = merged_scores.merge(performance_df_with_id, on='area_id')
    
    # Function to classify based on thresholds
    def classify_difficulty(scores, threshold_low, threshold_high):
        classifications = []
        for score in scores:
            if score < threshold_low:
                classifications.append('Low')
            elif score < threshold_high:
                classifications.append('Moderate')
            else:
                classifications.append('High')
        return classifications
    
    # Apply all classification methods
    scores = final_df['difficulty_score'].values
    
    # IQR Classification
    if 'IQR (25%/75%)' in methods_results:
        t1_iqr, t2_iqr = methods_results['IQR (25%/75%)']
        final_df['iqr_class'] = classify_difficulty(scores, t1_iqr, t2_iqr)
    
    # Tertiles Classification
    if 'Tertiles (33%/67%)' in methods_results:
        t1_ter, t2_ter = methods_results['Tertiles (33%/67%)']
        final_df['tertile_class'] = classify_difficulty(scores, t1_ter, t2_ter)
    
    # Performance-based Classification
    if 'Performance-based' in methods_results:
        t1_perf, t2_perf = methods_results['Performance-based']
        final_df['performance_class'] = classify_difficulty(scores, t1_perf, t2_perf)
    
    # Standard Deviation Classification
    if 'Std Dev (±0.5σ)' in methods_results:
        t1_std, t2_std = methods_results['Std Dev (±0.5σ)']
        final_df['std_class'] = classify_difficulty(scores, t1_std, t2_std)
    
    # Natural Breaks Classification (if available)
    if 'Natural Breaks' in methods_results:
        t1_jenks, t2_jenks = methods_results['Natural Breaks']
        final_df['jenks_class'] = classify_difficulty(scores, t1_jenks, t2_jenks)
    
    # Add threshold values as metadata
    final_df.attrs['thresholds'] = methods_results
    
    # Sort by difficulty score for better visualization
    final_df = final_df.sort_values('difficulty_score').reset_index(drop=True)
    
    print("Classification DataFrame Created:")
    #print(f"Shape: {final_df.shape}")
    #print(f"Columns: {list(final_df.columns)}")
    
    return final_df

def analyze_classification_agreement(classification_df):
    """
    Analyze agreement between different classification methods
    """
    # Get classification columns
    class_columns = [col for col in classification_df.columns if col.endswith('_class')]
    
    if len(class_columns) < 2:
        print("Need at least 2 classification methods for agreement analysis")
        return None
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION METHOD AGREEMENT ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate agreement matrix
    agreement_matrix = pd.DataFrame(index=class_columns, columns=class_columns)
    
    for i, method1 in enumerate(class_columns):
        for j, method2 in enumerate(class_columns):
            if i <= j:  # Only calculate upper triangle
                if i == j:
                    agreement_matrix.loc[method1, method2] = 1.0
                else:
                    # Calculate percentage agreement
                    agreement = (classification_df[method1] == classification_df[method2]).mean()
                    agreement_matrix.loc[method1, method2] = agreement
                    agreement_matrix.loc[method2, method1] = agreement
    
    print("Pairwise Agreement Matrix (% areas with same classification):")
    print(agreement_matrix.round(3))
    
    # Show detailed comparison
    print(f"\nDetailed Classification Comparison:")
    print("-" * 60)
    
    for i, method1 in enumerate(class_columns):
        for j, method2 in enumerate(class_columns[i+1:], i+1):
            agreement_pct = agreement_matrix.loc[method1, method2] * 100
            
            # Create crosstab
            crosstab = pd.crosstab(classification_df[method1], classification_df[method2], 
                                 margins=True, margins_name="Total")
            
            print(f"\n{method1} vs {method2} (Agreement: {agreement_pct:.1f}%):")
            print(crosstab)
    
    return agreement_matrix

def create_classification_summary_table(classification_df):
    """
    Create summary table showing distribution of areas across methods
    """
    # Get classification columns
    class_columns = [col for col in classification_df.columns if col.endswith('_class')]
    
    if not class_columns:
        print("No classification columns found")
        return None
    
    # Create summary table
    summary_data = []
    
    for method in class_columns:
        value_counts = classification_df[method].value_counts()
        
        # Ensure we have all three categories
        low_count = value_counts.get('Low', 0)
        moderate_count = value_counts.get('Moderate', 0)
        high_count = value_counts.get('High', 0)
        total = len(classification_df)
        
        summary_data.append({
            'Method': method.replace('_class', '').replace('_', ' ').title(),
            'Low_Count': low_count,
            'Low_Pct': f"{low_count/total*100:.1f}%",
            'Moderate_Count': moderate_count,
            'Moderate_Pct': f"{moderate_count/total*100:.1f}%", 
            'High_Count': high_count,
            'High_Pct': f"{high_count/total*100:.1f}%",
            'Total': total
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n{'='*80}")
    print("CLASSIFICATION DISTRIBUTION SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    
    return summary_df

def export_classification_results(classification_df, save_path="difficulty_classifications.csv"):
    """
    Export the classification results to CSV
    """
    save_path = EDA_DIR / save_path
    
    # Select relevant columns for export
    export_columns = ['area_id', 'difficulty_score']
    
    # Add all classification columns
    class_columns = [col for col in classification_df.columns if col.endswith('_class')]
    export_columns.extend(class_columns)
    
    # Add performance metrics
    performance_columns = ['mape', 'rmse', 'mase']
    export_columns.extend([col for col in performance_columns if col in classification_df.columns])
    
    # Create export dataframe
    export_df = classification_df[export_columns].copy()
    
    # Add metadata as comment in CSV
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# Difficulty Score Classifications\n")
        f.write("# Generated using R²-weighted optimal weights\n")
        f.write("# Classification Methods:\n")
        f.write("#   iqr_class: IQR method (25th/75th percentiles)\n")
        f.write("#   tertile_class: Tertiles method (33rd/67th percentiles)\n")
        f.write("#   performance_class: Performance-based optimization\n")
        f.write("#   std_class: Standard deviation method (±0.5σ)\n")
        if 'jenks_class' in class_columns:
            f.write("#   jenks_class: Natural breaks (Jenks) method\n")
        f.write("#\n")
        
        # Write the actual data
        export_df.to_csv(f, index=False)
    
    print(f"\nClassification results exported to: {save_path}")
    print(f"Exported {len(export_df)} areas with {len(export_columns)} columns")
    
    return export_df

def check_multicollinearity(difficulty_df, seasonal_df):
    """
    Comprehensive multicollinearity analysis for your difficulty score predictors
    """
    
    # Merge the data (same as in your code)
    merged_df = difficulty_df[['area_id', 'lag1_autocorr']].merge(
        seasonal_df[['area_id', 'cv', 'seasonal_strength']], 
        on='area_id'
    )
    
    # Select predictor variables
    predictors = merged_df[['cv', 'seasonal_strength', 'lag1_autocorr']]
    
    print("="*70)
    print("MULTICOLLINEARITY DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    # 1. Correlation Matrix
    print("\n1. CORRELATION MATRIX:")
    print("-" * 40)
    corr_matrix = predictors.corr()
    print(corr_matrix.round(3))
    
    # 2. Correlation significance tests
    print("\n2. CORRELATION SIGNIFICANCE TESTS:")
    print("-" * 40)
    predictor_names = ['cv', 'seasonal_strength', 'lag1_autocorr']
    
    for i, var1 in enumerate(predictor_names):
        for j, var2 in enumerate(predictor_names[i+1:], i+1):
            corr, p_value = pearsonr(predictors[var1], predictors[var2])
            
            # Significance stars
            if p_value < 0.001:
                sig = "***"
            elif p_value < 0.01:
                sig = "**"
            elif p_value < 0.05:
                sig = "*"
            elif p_value < 0.10:
                sig = "+"
            else:
                sig = ""
            
            print(f"{var1} vs {var2}: r = {corr:.3f}{sig} (p = {p_value:.3f})")
    
    # 3. Variance Inflation Factors (VIF)
    print("\n3. VARIANCE INFLATION FACTORS (VIF):")
    print("-" * 40)
    print("Rule of thumb: VIF > 5 indicates multicollinearity concern")
    print("                VIF > 10 indicates serious multicollinearity")
    
    # Calculate VIF for each predictor
    vif_data = pd.DataFrame()
    vif_data["Predictor"] = predictors.columns
    vif_data["VIF"] = [variance_inflation_factor(predictors.values, i) 
                       for i in range(len(predictors.columns))]
    
    print(vif_data.round(3))
    
    # 4. Interpretation
    print("\n4. INTERPRETATION:")
    print("-" * 40)
    
    # Check for high correlations
    high_corr_pairs = []
    for i, var1 in enumerate(predictor_names):
        for j, var2 in enumerate(predictor_names[i+1:], i+1):
            corr_val = abs(corr_matrix.loc[var1, var2])
            if corr_val > 0.7:
                high_corr_pairs.append((var1, var2, corr_val))
    
    if high_corr_pairs:
        print("HIGH CORRELATIONS DETECTED (|r| > 0.7):")
        for var1, var2, corr_val in high_corr_pairs:
            print(f"  {var1} & {var2}: |r| = {corr_val:.3f}")
        print("  → This explains why regression coefficients have wide confidence intervals!")
    else:
        print("No extremely high correlations (|r| > 0.7) detected.")
    
    # Check VIF values
    high_vif = vif_data[vif_data['VIF'] > 5]
    if not high_vif.empty:
        print(f"\nHIGH VIF VALUES DETECTED (VIF > 5):")
        for _, row in high_vif.iterrows():
            print(f"  {row['Predictor']}: VIF = {row['VIF']:.3f}")
        print("  → This confirms multicollinearity issues!")
    else:
        print("\nNo severe multicollinearity detected based on VIF.")
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS:")
    print("-" * 40)
    
    max_vif = vif_data['VIF'].max()
    max_corr = corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()
    
    if max_vif > 10 or max_corr > 0.8:
        print("SEVERE multicollinearity detected. Consider:")
        print("  • Using only the most significant predictor (seasonal_strength)")
        print("  • Principal Component Analysis (PCA)")
        print("  • Ridge regression to handle multicollinearity")
        
    elif max_vif > 5 or max_corr > 0.6:
        print("MODERATE multicollinearity detected. Consider:")
        print("  • Being cautious about interpreting individual coefficients")
        print("  • Using domain knowledge for weights instead of regression")
        print("  • Focusing on overall model performance rather than individual predictors")
        
    else:
        print("Multicollinearity is not a major concern.")
        print("The wide confidence intervals are likely due to:")
        print("  • Small sample size (n=29)")
        print("  • Weak true relationships between predictors and outcome")
    
    # 6. Create visualization
    create_multicollinearity_plots(predictors, corr_matrix)
    
    return corr_matrix, vif_data

def create_multicollinearity_plots(predictors, corr_matrix):
    """
    Create visualization for multicollinearity analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, mask=mask, cbar_kws={"shrink": .8}, ax=ax1)
    ax1.set_title('Predictor Correlation Matrix', fontweight='bold')
    
    # Plot 2: Scatter plot matrix for the highest correlation pair
    # Find the highest correlation pair (excluding diagonal)
    corr_vals = corr_matrix.values
    np.fill_diagonal(corr_vals, 0)  # Remove diagonal
    max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_vals)), corr_vals.shape)
    
    var1 = corr_matrix.index[max_corr_idx[0]]
    var2 = corr_matrix.columns[max_corr_idx[1]]
    corr_val = corr_vals[max_corr_idx]
    
    ax2.scatter(predictors[var1], predictors[var2], alpha=0.7, s=60)
    ax2.set_xlabel(var1.replace('_', ' ').title())
    ax2.set_ylabel(var2.replace('_', ' ').title())
    ax2.set_title(f'Highest Correlation Pair\n{var1} vs {var2} (r = {corr_val:.3f})', 
                 fontweight='bold')
    
    # Add trend line
    z = np.polyfit(predictors[var1], predictors[var2], 1)
    p = np.poly1d(z)
    ax2.plot(predictors[var1], p(predictors[var1]), "r--", alpha=0.8)
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def alternative_weight_strategies(corr_matrix, regression_results):
    """
    Suggest alternative weighting strategies given multicollinearity issues
    """
    print("\n" + "="*70)
    print("ALTERNATIVE WEIGHT STRATEGIES")
    print("="*70)
    
    print("\nGiven the multicollinearity and non-significant coefficients:")
    print("\n1. SINGLE PREDICTOR APPROACH:")
    print("   Use only seasonal_strength (the only significant predictor)")
    print("   Difficulty_Score = seasonal_strength")
    
    print("\n2. DOMAIN-KNOWLEDGE WEIGHTS:")
    print("   Based on forecasting theory:")
    print("   • CV (variability): 40% weight")
    print("   • Seasonal strength: 35% weight") 
    print("   • Autocorrelation: 25% weight")
    
    print("\n3. EQUAL WEIGHTS (SIMPLEST):")
    print("   Difficulty_Score = (CV + Seasonal_Strength + Autocorr) / 3")
    
    print("\n4. CORRELATION-BASED WEIGHTS:")
    print("   Weight predictors by their correlation with performance metrics")
    
    print("\n5. REGULARIZED REGRESSION:")
    print("   Use Ridge or Lasso regression to handle multicollinearity")
    
    return None

# Complete diagnostic function
def multicollinearity_diagnostic(comprehensive_stats, performance_df):
    """
    Complete diagnostic including relationship with outcome variable
    """    
    if 'area_id' not in performance_df.columns:
        performance_df_with_id = performance_df.copy()
        performance_df_with_id['area_id'] = range(1, len(performance_df) + 1)
    else:
        performance_df_with_id = performance_df.copy()
    
    final_df = comprehensive_stats.merge(performance_df_with_id, on='area_id')
    
    # Run multicollinearity check
    corr_matrix, vif_data = check_multicollinearity(difficulty_df, difficulty_df)
    
    # Check individual predictor-outcome correlations
    print("\n" + "="*70)
    print("INDIVIDUAL PREDICTOR-OUTCOME CORRELATIONS")
    print("="*70)
    
    predictors = ['cv', 'seasonal_strength', 'lag1_autocorr']
    outcomes = ['mape', 'rmse', 'mase']
    
    for outcome in outcomes:
        print(f"\n{outcome.upper()} Correlations:")
        for predictor in predictors:
            corr, p_val = pearsonr(final_df[predictor], final_df[outcome])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {predictor}: r = {corr:.3f}{sig} (p = {p_val:.3f})")
    
    # Suggest alternatives
    alternative_weight_strategies(corr_matrix, None)
    
    return final_df, corr_matrix, vif_data

    
if __name__ == '__main__':
    
    # Analyze the optimized results
    run = 'exp2_adaptive_reg'
    input_path = RESULTS_DIR / f'{run}.csv'
    
    save_dir = RESULTS_DIR / f'{run}_figures'
    save_dir.mkdir(exist_ok=True, parents=True)
    
    comprehensive_stats = create_comprehensive_eda_table() 
    performance_df = pd.read_csv (input_path)
       
    # Find R²-weighted optimal weights
    regression_results = optimal_weights_r2_weighted(comprehensive_stats, performance_df)
    
    # Test the R²-weighted composite score
    validation_results, optimal_weights = test_r2_weighted_composite_score(comprehensive_stats, performance_df, 
                                                                           regression_results)    
    # Create fitting curve with equation
    fig1 = fitting_curve(comprehensive_stats, performance_df, optimal_weights, 
                                            save_path=save_dir)


    methods_results, perf_results = compare_threshold_methods( comprehensive_stats, performance_df, 
                                                              optimal_weights)

    classification_df = create_classification_dataframe(comprehensive_stats, performance_df, 
                                                        optimal_weights, methods_results)
    
    summary_table = create_classification_summary_table(classification_df)
    export_df = export_classification_results(classification_df)
