import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

results_dir = Path(__file__).parent.parent / 'results'
results_dir.mkdir(exist_ok=True, parents=True)

def analyze_regressor_optimization(input_path, save_dir, output_prefix, plot_res='low'):
    """
    Analyze the results of regressor optimization from experimental runs
    """
    # Load the results
    df = pd.read_csv(input_path)
    
    print("="*80)
    print("REGRESSOR OPTIMIZATION ANALYSIS")
    print("="*80)
    print(f"\nData Source: {input_path}")
    print(f"Total Areas Analyzed: {len(df)}")
    
    # 1. Analyze k values (number of regressors selected)
    print("\n1. NUMBER OF REGRESSORS (k) ANALYSIS:")
    print("-" * 50)
    
    # Extract number of regressors for each area
    k_values = []
    regressor_lists = []
    
    for idx, row in df.iterrows():
        # Try to parse regressors from the dataframe
        if 'regressors' in df.columns and pd.notna(row['regressors']):
            try:
                # Handle string representation of list
                regressors = eval(row['regressors']) if isinstance(row['regressors'], str) else row['regressors']
                if isinstance(regressors, list):
                    k_values.append(len(regressors))
                    regressor_lists.append(regressors)
            except:
                pass
    
    if k_values:
        k_counter = Counter(k_values)
        print("\nDistribution of k values:")
        for k in sorted(k_counter.keys()):
            pct = (k_counter[k] / len(k_values)) * 100
            print(f"  k={k}: {k_counter[k]} areas ({pct:.1f}%)")
        
        print("\nStatistics:")
        print(f"  Mean k: {np.mean(k_values):.2f}")
        print(f"  Median k: {np.median(k_values):.1f}")
        print(f"  Mode k: {max(k_counter, key=k_counter.get)}")
        
        # Create k distribution plot
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        bars = plt.bar(k_counter.keys(), k_counter.values(), alpha=0.7, color='steelblue')
        plt.xlabel('Number of Regressors (k)')
        plt.ylabel('Number of Areas')
        #plt.title('Distribution of Optimal k Values')
        plt.xticks(sorted(k_counter.keys()))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 2. Analyze which regressors are most frequently selected
    print("\n\n2. MOST FREQUENTLY SELECTED REGRESSORS:")
    print("-" * 50)
    
    all_regressors = []
    for reg_list in regressor_lists:
        all_regressors.extend(reg_list)
    
    if all_regressors:
        regressor_counter = Counter(all_regressors)
        total_areas = len(regressor_lists)
        
        print("\nTop 10 Most Selected Regressors:")
        for i, (reg, count) in enumerate(regressor_counter.most_common(10), 1):
            pct = (count / total_areas) * 100
            print(f"  {i}. {reg}: {count} times ({pct:.1f}% of areas)")
        
        # Create regressor frequency plot
        plt.subplot(1, 2, 2)
        top_regressors = regressor_counter.most_common(10)
        reg_names = [r[0].replace('_regressor', '') for r in top_regressors]
        reg_counts = [r[1] for r in top_regressors]
        
        bars = plt.barh(reg_names, reg_counts, alpha=0.7, color='darkgreen')
        plt.xlabel('Number of Times Selected')
        #plt.title('Top 10 Most Frequently Selected Regressors')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center')
        
        plt.tight_layout() 
        
        if plot_res == 'high':
            for ext in ['png', 'pdf', 'tif']:
                output_path = save_dir / f'top_k_regressors.{ext}'
                plt.savefig(output_path, dpi=600, bbox_inches='tight')
        elif plot_res == 'low':
            output_path = save_dir / 'top_k_regressors.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    # 3. Analyze regressor combinations
    print("\n\n3. REGRESSOR COMBINATION ANALYSIS:")
    print("-" * 50)
    
    # Find most common combinations
    combination_counter = Counter([tuple(sorted(reg_list)) for reg_list in regressor_lists])
    
    print("\nTop 5 Most Common Regressor Combinations:")
    for i, (combo, count) in enumerate(combination_counter.most_common(5), 1):
        pct = (count / len(regressor_lists)) * 100
        combo_str = ', '.join([r.replace('_regressor', '') for r in combo])
        print(f"\n  {i}. [{combo_str}]")
        print(f"     Used in {count} areas ({pct:.1f}%)")
    
    # 4. Analyze performance by k value
    print("\n\n4. PERFORMANCE BY NUMBER OF REGRESSORS:")
    print("-" * 50)
    
    if 'mape' in df.columns:
        # Create a mapping of area to k value
        area_k_map = {}
        for idx, row in df.iterrows():
            area_id = row.get('area_id', idx)
            if idx < len(k_values):
                area_k_map[area_id] = k_values[idx]
        
        # Group performance by k
        k_performance = defaultdict(list)
        for idx, row in df.iterrows():
            area_id = row.get('area_id', idx)
            if area_id in area_k_map and pd.notna(row['mape']):
                k_performance[area_k_map[area_id]].append(row['mape'])
        
        print("\nMean MAPE by k value:")
        k_mape_means = {}
        for k in sorted(k_performance.keys()):
            mean_mape = np.mean(k_performance[k])
            std_mape = np.std(k_performance[k])
            k_mape_means[k] = mean_mape
            print(f"  k={k}: {mean_mape:.2f}% (±{std_mape:.2f}%)")
        
        # Create performance by k plot
        '''
        plt.figure(figsize=(10, 6))
        
        # Box plot
        plt.subplot(1, 2, 1)
        data_for_plot = [k_performance[k] for k in sorted(k_performance.keys())]
        labels_for_plot = [f'k={k}' for k in sorted(k_performance.keys())]
        
        bp = plt.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        plt.ylabel('MAPE (%)')
        #plt.title('MAPE Distribution by Number of Regressors')
        plt.grid(True, alpha=0.3)
        
        # Mean MAPE trend
        plt.subplot(1, 2, 2)
        k_vals = sorted(k_mape_means.keys())
        mape_vals = [k_mape_means[k] for k in k_vals]
        
        plt.plot(k_vals, mape_vals, 'o-', linewidth=2, markersize=8, color='darkred')
        plt.xlabel('Number of Regressors (k)')
        plt.ylabel('Mean MAPE (%)')
        #plt.title('Mean MAPE vs Number of Regressors')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_vals)
        
        # Add value labels
        for k, mape in zip(k_vals, mape_vals):
            plt.text(k, mape + 0.2, f'{mape:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_by_k_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        '''
    # 5. Regressor importance patterns
    print("\n\n5. REGRESSOR SELECTION PATTERNS:")
    print("-" * 50)
    
    # Analyze which regressors tend to be selected together
    cooccurrence = defaultdict(int)
    for reg_list in regressor_lists:
        for i in range(len(reg_list)):
            for j in range(i+1, len(reg_list)):
                pair = tuple(sorted([reg_list[i], reg_list[j]]))
                cooccurrence[pair] += 1
    
    print("\nTop 10 Regressor Pairs (frequently selected together):")
    for i, (pair, count) in enumerate(sorted(cooccurrence.items(), 
                                           key=lambda x: x[1], reverse=True)[:10], 1):
        pct = (count / len(regressor_lists)) * 100
        pair_str = ' + '.join([r.replace('_regressor', '') for r in pair])
        print(f"  {i}. {pair_str}: {count} times ({pct:.1f}%)")
    
    # 6. Summary insights
    print("\n\n6. KEY INSIGHTS:")
    print("-" * 50)
    
    if k_values and regressor_counter:
        print(f"\n• Optimal k range: {min(k_values)} to {max(k_values)} regressors")
        print(f"• Most common k value: {max(k_counter, key=k_counter.get)} regressors")
        
        top_reg = regressor_counter.most_common(1)[0]
        print(f"• Most valuable regressor: {top_reg[0]} (selected in {(top_reg[1]/total_areas)*100:.0f}% of areas)")
        
        if k_mape_means:
            best_k = min(k_mape_means, key=k_mape_means.get)
            print(f"• Best performing k value: k={best_k} (mean MAPE: {k_mape_means[best_k]:.2f}%)")
    
    return {
        'k_distribution': dict(k_counter) if 'k_counter' in locals() else {},
        'regressor_frequency': dict(regressor_counter) if 'regressor_counter' in locals() else {},
        'k_performance': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                         for k, v in k_performance.items()} if 'k_performance' in locals() else {}
    }

def compare_with_baseline(optimized_filepath, baseline_filepath=None):
    """
    Compare optimized results with baseline (if available)
    """
    if baseline_filepath:
        print("\n\n" + "="*80)
        print("COMPARISON WITH BASELINE")
        print("="*80)
        
        opt_df = pd.read_csv(optimized_filepath)
        base_df = pd.read_csv(baseline_filepath)
        
        # Compare overall metrics
        opt_mape = opt_df['mape'].mean()
        base_mape = base_df['mape'].mean()
        improvement = ((base_mape - opt_mape) / base_mape) * 100
        
        print(f"\nOverall Performance Improvement:")
        print(f"  Baseline Mean MAPE: {base_mape:.2f}%")
        print(f"  Optimized Mean MAPE: {opt_mape:.2f}%")
        print(f"  Improvement: {improvement:.1f}%")
        
        # Area-by-area improvement
        if 'area_id' in opt_df.columns and 'area_id' in base_df.columns:
            merged = pd.merge(opt_df[['area_id', 'mape']], 
                            base_df[['area_id', 'mape']], 
                            on='area_id', suffixes=('_opt', '_base'))
            
            merged['improvement'] = ((merged['mape_base'] - merged['mape_opt']) / 
                                   merged['mape_base']) * 100
            
            print(f"\nAreas with Greatest Improvement:")
            top_improved = merged.nlargest(5, 'improvement')
            for _, row in top_improved.iterrows():
                print(f"  Area {int(row['area_id'])}: "
                      f"{row['mape_base']:.1f}% → {row['mape_opt']:.1f}% "
                      f"({row['improvement']:.1f}% better)")


def generate_regressor_selection_analysis(run_name='run_adaptive', results_dir='../results/', 
                                          output_dir='../figures/', plot_res='low'):
    """
    Analyze regressor selection patterns from adaptive optimization.
    
    Produces:
    - Figure 7 (left): Distribution of k values across 29 areas
    - Figure 7 (right): Frequency of each regressor being selected
    
    Shows which regressors are most important and how many regressors
    each area typically needs for optimal performance.
    
    Args:
        run_name: Name of the run to analyze (e.g., 'run_adaptive')
        results_dir: Directory containing the CSV results
        output_dir: Directory to save figures
        plot_res: Plot resolution ('low', 'medium', 'high')
    
    Input file:
        {results_dir}/{run_name}.csv (must have 'selected_regressors' column)
    
    Output:
        {output_dir}/regressor_selection_{run_name}.png
        Summary statistics printed to console
    
    Returns:
        Dictionary with analysis results
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    input_path = results_path / f'{run_name}.csv'
    
    # Check that input file exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing required input file: {input_path}\n"
            f"Please run Step 1 (generate_adaptive_forecasts) first."
        )
    
    # Create output directory for detailed figures
    save_dir = results_path / f'{run_name}_figures'
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"  Input: {input_path}")
    print(f"  Output: {results_path}/regressor_selection_{run_name}.png")
    

    
    # Run the analysis
    results = analyze_regressor_optimization(
        input_path, 
        save_dir, 
        run_name, 
        plot_res=plot_res
    )
    
    return results


if __name__ == "__main__":
    # For standalone testing
    print("Analyzing regressor selection patterns...")
    generate_regressor_selection_analysis(run_name='run_adaptive')




