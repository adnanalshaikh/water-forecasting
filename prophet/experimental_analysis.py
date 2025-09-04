## remove area 20 as outlier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import os
from scipy.stats import ttest_rel
import sys 

RESULTS_PATH = Path(__file__).parent.parent / 'results'


def print_options():
    pd.set_option("display.width", 1000)  # Set arbitrarily large
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 10)
    pd.set_option("display.expand_frame_repr", False)  # Critical: Prevent wrapping
    pd.set_option('display.float_format', '{:.2f}'.format)
    
class ExperimentalAnalysis:
    """
    Framework for analyzing multiple experimental runs
    """
    def __init__(self, run_files):
        """
        Initialize with list of experiment log files
        
        Parameters:
        - run_files: List of tuples (run_name, filepath)
        """
        self.run_files = run_files
        self.combined_results = None
        self.load_all_runs() 
        self.run_stats = None
        
    def load_all_runs(self):
        """Load and combine all experimental runs"""
        all_data = []
        
        for run_name, filepath in self.run_files:
            try:
                df = pd.read_csv(filepath)
                #df = df[df['area_id'] != 20] # remove area 20 as outlier 
                df['run_name'] = run_name
                df['run_type'] = self._categorize_run(run_name)
                all_data.append(df)
                print(f"Loaded {len(df)} areas from {run_name}")
            except FileNotFoundError:
                print(f"Warning: {filepath} not found")
        
        if all_data:
            self.combined_results = pd.concat(all_data, ignore_index=True)
            print(f"Total combined data: {len(self.combined_results)} experiments")
        else:
            self.combined_results = pd.DataFrame()  
    
    def _categorize_run(self, run_name):
        """Categorize run type for analysis"""
        if 'baseline' in run_name.lower() or 'run_1' in run_name:
            return 'Baseline'
        elif 'feature' in run_name.lower():
            return 'Feature Analysis'
        elif 'outlier' in run_name.lower():
            return 'Outlier Method'
        elif 'hyper' in run_name.lower() or 'search' in run_name.lower():
            return 'Hyperparameter'
        elif 'minimal' in run_name.lower():
            return 'Ablation'
        else:
            return 'Other'
    
    def compare_runs_statistical(self):
        """Statistical comparison between runs."""
        if self.combined_results.empty:
            print("No data to compare.")
            return None
    
        # Define the aggregation with custom names for clarity
        agg_dict = {
            'mape': ['mean', 'std', 'median', 'min', 'max', 'count',
                     ('q25', lambda x: x.quantile(0.25)),
                     ('q75', lambda x: x.quantile(0.75)),
                     ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))],
            'mase': ['mean', 'std', 'median', 'min', 'max', 'count',
                     ('q25', lambda x: x.quantile(0.25)),
                     ('q75', lambda x: x.quantile(0.75)),
                     ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))],
            'mase_seas': ['mean', 'std', 'median', 
                          ('q25', lambda x: x.quantile(0.25)),
                          ('q75', lambda x: x.quantile(0.75)),
                          ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))]
        }
        
        # Compute stats
        run_stats = self.combined_results.groupby('run_name').agg(agg_dict).round(3)
        
        # Flatten column names to get 'mape_q25' format
        run_stats.columns = [f'{col[0]}_{col[1]}' for col in run_stats.columns]
    
        print("STATISTICAL COMPARISON ACROSS RUNS:")
        print("=" * 60)
        print(run_stats)
    
        runs = self.combined_results['run_name'].unique()
    
        if len(runs) > 1:
            print(f"\nPAIRWISE T-TESTS (MAPE):")
            print("-" * 60)
    
            # Identify baseline run
            baseline_run = next((r for r in runs if 'baseline' in r.lower() or 'run_1' in r.lower()), runs[0])
            baseline_data = self.combined_results[self.combined_results['run_name'] == baseline_run]['mape'].reset_index(drop=True)
    
            for run in runs:
                if run == baseline_run:
                    continue
    
                run_data = self.combined_results[self.combined_results['run_name'] == run]['mape'].reset_index(drop=True)
    
                # Ensure equal lengths for paired test (e.g., per fold or group)
                if len(baseline_data) == len(run_data):
                    t_stat, p_val = stats.ttest_rel(baseline_data, run_data)
                    improvement = (baseline_data.mean() - run_data.mean()) / baseline_data.mean() * 100
    
                    print(f"{run:25} | Δ: {improvement:+5.1f}% | p-value: {p_val:.4f}")
                else:
                    print(f"{run:25} | Skipped: unequal sample size with baseline")
        
        self.run_stats = run_stats;
    
        return run_stats
    
    def create_boxplots(self, save_dir= RESULTS_PATH, 
                        fig_name_prefix = 'config_perf_comp',
                        save_plot_mode = 'low'):
        
        """Create box plot MAPE and MASE boxplots with significance stars."""
        os.makedirs(save_dir, exist_ok=True)
        if self.combined_results.empty:
            return
    
        # Update plot style
        plt.rcParams.update({
            "font.family": "serif",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14
        })
    
        # Set custom run order: base-0 first, adaptive last
        all_runs = self.combined_results['run_name'].unique()
        desired_order = ['base-0'] + sorted([r for r in all_runs if r not in ('base-0', 'adaptive')]) + ['adaptive']
        self.combined_results['run_name'] = pd.Categorical(
            self.combined_results['run_name'], categories=desired_order, ordered=True
        )
        runs = desired_order
    
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Significance testing vs. base-0
        baseline = self.combined_results[self.combined_results['run_name'] == 'base-0']['mape'].values
        p_values = {}
        for run in runs:
            if run == 'base-0':
                continue
            current = self.combined_results[self.combined_results['run_name'] == run]['mape'].values
            _, p_val = ttest_rel(baseline, current)
            p_values[run] = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.0506 else ''
    
        # MAPE boxplot
        ax1 = axes[0]
        self.combined_results.boxplot(column='mape', by='run_name', ax=ax1,
                                      boxprops=dict(linewidth=1.5),
                                      medianprops=dict(linewidth=2.0),
                                      whiskerprops=dict(linewidth=1.5),
                                      capprops=dict(linewidth=1.5),
                                      flierprops=dict(marker='o', markersize=4, markerfacecolor='black'))
        ax1.set_xlabel('')
        ax1.set_ylabel('MAPE (\%)', fontsize=16)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_title('')
        ax1.get_figure().suptitle('')
        ax1.grid(True, alpha=0.16)
    
        # Add significance stars just above the x-axis
        for i, run in enumerate(runs):
            if run != 'base-0' and p_values[run]:
                ax1.text(i + 1, ax1.get_ylim()[0] + 0.004, p_values[run],
                         ha='center', va='bottom', fontsize=14, color='black')
    
        # MASE boxplot
        ax2 = axes[1]
        self.combined_results.boxplot(column='mase', by='run_name', ax=ax2,
                                      boxprops=dict(linewidth=1.5),
                                      medianprops=dict(linewidth=2.0),
                                      whiskerprops=dict(linewidth=1.5),
                                      capprops=dict(linewidth=1.5),
                                      flierprops=dict(marker='o', markersize=4, markerfacecolor='black'))
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.2)
        ax2.set_xlabel('')
        ax2.set_ylabel('MASE', fontsize=16)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_title('')
        ax2.get_figure().suptitle('')
        ax2.grid(True, alpha=0.16)
    
        # Save figure in multiple formats
        plt.tight_layout()
        
        if save_plot_mode in ['low', 'high']:
            if save_plot_mode == 'high':
                for ext in ['png', 'pdf', 'tif']:
                    fig.savefig(os.path.join(save_dir, f'{fig_name_prefix}.{ext}'), dpi=600, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(save_dir, f'{fig_name_prefix}.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig)
    
    def create_comparison_visualizations(self, save_dir='comparison_results'):
        """Create comprehensive comparison visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if self.combined_results.empty:
            return
        
        # 1. Performance Distribution Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experimental Runs Comparison', fontsize=16, fontweight='bold')
        
        # MAPE box plots
        ax1 = axes[0, 0]
        self.combined_results.boxplot(column='mape', by='run_name', ax=ax1)
        ax1.set_title('MAPE Distribution by Run')
        ax1.set_ylabel('MAPE (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # MASE box plots  
        ax2 = axes[0, 1]
        self.combined_results.boxplot(column='mase', by='run_name', ax=ax2)
        ax2.set_title('MASE Distribution by Run')
        ax2.set_ylabel('MASE')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Naive baseline')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # Success rate by run
        ax3 = axes[1, 0]
        success_rates = self.combined_results.groupby('run_name').apply(
            lambda x: (x['mape'] < 15).mean() * 100
        )
        success_rates.plot(kind='bar', ax=ax3, color='lightgreen', alpha=0.7)
        ax3.set_title('Success Rate by Run (MAPE < 15%)')
        ax3.set_ylabel('Success Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        #ax3.grid(False, alpha=0.3)
        
        # Scatter plot: Run comparison
        ax4 = axes[1, 1]
        runs = self.combined_results['run_name'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(runs)))
        
        for i, run in enumerate(runs):
            run_data = self.combined_results[self.combined_results['run_name'] == run]
            ax4.scatter(run_data['area_id'], run_data['mape'], 
                       label=run, alpha=0.7, color=colors[i])
        
        ax4.set_xlabel('Area ID')
        ax4.set_ylabel('MAPE (%)')
        ax4.set_title('MAPE by Area Across Runs')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.1)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/runs_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Impact Analysis (if applicable)
        self._analyze_feature_impact(save_dir)
        
        # 3. Hyperparameter Analysis (if applicable)
        self._analyze_hyperparameter_impact(save_dir)  
    
    def _analyze_feature_impact(self, save_dir):
        """Analyze impact of different feature configurations"""
        feature_runs = self.combined_results[
            self.combined_results['run_type'].isin(['Baseline', 'Feature Analysis', 'Ablation'])
        ]
        
        if len(feature_runs) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Group by feature configuration and calculate mean performance
        feature_performance = feature_runs.groupby('run_name').agg({
            'mape': ['mean', 'std', 'count'],
            'mase': 'mean'
        })
        
        # Create grouped bar chart
        x_pos = np.arange(len(feature_performance))
        means = feature_performance[('mape', 'mean')]
        stds = feature_performance[('mape', 'std')]
        
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=['blue', 'green', 'red', 'orange'][:len(means)])
        
        plt.xlabel('Feature Configuration')
        plt.ylabel('Mean MAPE (%)')
        plt.title('Feature Configuration Impact on Performance')
        plt.xticks(x_pos, feature_performance.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
                    f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()  
        
    def _analyze_hyperparameter_impact(self, save_dir):
        """Analyze hyperparameter optimization impact"""
        hyper_runs = self.combined_results[
            self.combined_results['run_type'].isin(['Baseline', 'Hyperparameter'])
        ]
        
        if len(hyper_runs) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Trials vs Performance
        if 'n_trials' in hyper_runs.columns:
            ax1 = axes[0]
            scatter = ax1.scatter(hyper_runs['n_trials'], hyper_runs['mape'],
                                 c=hyper_runs['mase'], cmap='viridis', alpha=0.7)
            ax1.set_xlabel('Number of Optimization Trials')
            ax1.set_ylabel('MAPE (%)')
            ax1.set_title('Optimization Effort vs Performance')
            ax1.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('MASE')
        
        # Parameter space visualization
        ax2 = axes[1]
        if all(col in hyper_runs.columns for col in ['changepoint_prior_scale', 'seasonality_prior_scale']):
            scatter2 = ax2.scatter(hyper_runs['changepoint_prior_scale'], 
                                  hyper_runs['seasonality_prior_scale'],
                                  c=hyper_runs['mape'], cmap='RdYlGn_r', alpha=0.7)
            ax2.set_xlabel('Changepoint Prior Scale')
            ax2.set_ylabel('Seasonality Prior Scale')
            ax2.set_title('Hyperparameter Space (Color = MAPE)')
            ax2.grid(True, alpha=0.3)
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('MAPE (%)')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()  
        
    def generate_experimental_summary(self, out=sys.stdout):
        """Generate comprehensive experimental summary"""
        if self.combined_results.empty:
            return
        
        summary_lines = [
            "=" * 80,
            "EXPERIMENTAL RUNS SUMMARY",
            "=" * 80,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "RUNS ANALYZED:",
        ]
        
        for run_name, filepath in self.run_files:
            run_data = self.combined_results[self.combined_results['run_name'] == run_name]
            if len(run_data) > 0:
                mean_mape = run_data['mape'].mean()
                mean_mase = run_data['mase'].mean()
                summary_lines.append(
                    f"  • {run_name}: {len(run_data)} areas | "
                    f"MAPE: {mean_mape:.1f}% | MASE: {mean_mase:.2f}"
                )
        
        # Best performing run
        best_run = self.combined_results.groupby('run_name')['mape'].mean().idxmin()
        best_mape = self.combined_results.groupby('run_name')['mape'].mean().min()
        
        summary_lines.extend([
            "",
            "FINDINGS:",
            f"  • Best performing configuration: {best_run} (MAPE: {best_mape:.1f}%)",
            f"  • Total experiments conducted: {len(self.combined_results)}",
            f"  • Configuration variations tested: {len(self.run_files)}",
        ])
        
        # Write and display
        summary_text = '\n'.join(summary_lines)
        print(summary_text)
        out.write(summary_text)
        
        return summary_text

    def print_all_run_stats(self, baseline_run='base-0', out = sys.stdout):
        """
        Print comprehensive statistics using already computed run_stats and p-values.
        This function uses the existing self.run_stats from compare_runs_statistical().
        """
        if self.run_stats is None:
            print("No statistics computed. Run compare_runs_statistical() first.")
            return
  
        out.write("=" * 60 + "\n")
        out.write("COMPREHENSIVE RUN STATISTICS\n")
        out.write("=" * 60 + "\n")
        
        # Get all metrics from run_stats columns
        metrics = []
        for col in self.run_stats.columns:
            metric = col.split('_')[0]
            if metric not in metrics:
                metrics.append(metric)
        
        # Get baseline data for delta calculations
        baseline_means = {}
        baseline_medians = {}
        if baseline_run in self.run_stats.index:
            for metric in metrics:
                baseline_means[metric] = self.run_stats.loc[baseline_run, f'{metric}_mean']
                baseline_medians[metric] = self.run_stats.loc[baseline_run, f'{metric}_median']
        
        for metric in metrics:
            out.write(f"\n{metric.upper()} STATISTICS:\n")
            out.write("-" * 60 + "\n")
            
            # Prepare data for this metric
            display_data = []
            
            for run_name in self.run_stats.index:
                row = {
                    'Run': run_name,
                    'Count': int(self.run_stats.loc[run_name, f'{metric}_count']) if f'{metric}_count' in self.run_stats.columns else 'N/A',
                    'Mean': self.run_stats.loc[run_name, f'{metric}_mean'],
                    'Std': self.run_stats.loc[run_name, f'{metric}_std'],
                    'Median': self.run_stats.loc[run_name, f'{metric}_median'],
                    'Q25': self.run_stats.loc[run_name, f'{metric}_q25'],
                    'Q75': self.run_stats.loc[run_name, f'{metric}_q75'],
                    'IQR': self.run_stats.loc[run_name, f'{metric}_iqr'],
                    'Min': self.run_stats.loc[run_name, f'{metric}_min'] if f'{metric}_min' in self.run_stats.columns else 'N/A',
                    'Max': self.run_stats.loc[run_name, f'{metric}_max'] if f'{metric}_max' in self.run_stats.columns else 'N/A'
                }
                
                # Calculate deltas vs baseline
                if run_name != baseline_run and metric in baseline_means:
                    # Delta calculations (improvement for lower-is-better metrics)
                    delta_mean = (baseline_means[metric] - row['Mean']) / baseline_means[metric] * 100
                    delta_median = (baseline_medians[metric] - row['Median']) / baseline_medians[metric] * 100
                    
                    row['Delta_Mean%'] = f"{delta_mean:+.2f}%"
                    row['Delta_Median%'] = f"{delta_median:+.2f}%"
                else:
                    row['Delta_Mean%'] = "0.00%" if run_name == baseline_run else "N/A"
                    row['Delta_Median%'] = "0.00%" if run_name == baseline_run else "N/A"
                
                display_data.append(row)
            
            # Create DataFrame for display
            df_display = pd.DataFrame(display_data)
            
            # Format numeric columns
            numeric_cols = ['Mean', 'Std', 'Median', 'Q25', 'Q75', 'IQR']
            for col in numeric_cols:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            
            # Format Min/Max if available
            for col in ['Min', 'Max']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) and x != 'N/A' else "N/A")
            
            out.write(df_display.to_string(index=False) + "\n")
            
            # Print summary
            best_run_mean = self.run_stats[f'{metric}_mean'].idxmin()
            best_run_median = self.run_stats[f'{metric}_median'].idxmin()
            best_mean_val = self.run_stats.loc[best_run_mean, f'{metric}_mean']
            best_median_val = self.run_stats.loc[best_run_median, f'{metric}_median']
            
            out.write(f"\nSUMMARY for {metric.upper()}:\n")
            out.write(f"  • Best Mean Performance: {best_run_mean} ({best_mean_val:.3f})\n")
            out.write(f"  • Best Median Performance: {best_run_median} ({best_median_val:.3f})\n")
            
            # Show improvement over baseline
            if baseline_run in self.run_stats.index and metric in baseline_means:
                mean_improvement = (baseline_means[metric] - best_mean_val) / baseline_means[metric] * 100
                median_improvement = (baseline_medians[metric] - best_median_val) / baseline_medians[metric] * 100
                out.write(f"  • Best Mean Improvement over {baseline_run}: {mean_improvement:.2f}%\n")
                out.write(f"  • Best Median Improvement over {baseline_run}: {median_improvement:.2f}%\n")
        
        # Print P-values section if available from the existing method
        out.write(f"\nSTATISTICAL SIGNIFICANCE (vs {baseline_run}):\n")
        out.write("-" * 50+"\n")
        
        # Re-run the p-value calculations that were already in compare_runs_statistical
        runs = self.combined_results['run_name'].unique()
        if len(runs) > 1:
            from scipy import stats
            
            baseline_data = self.combined_results[self.combined_results['run_name'] == baseline_run]['mape'].reset_index(drop=True)
            
            for run in runs:
                if run == baseline_run:
                    out.write(f"{run:25} | Baseline Reference\n")
                    continue
                
                run_data = self.combined_results[self.combined_results['run_name'] == run]['mape'].reset_index(drop=True)
                
                if len(baseline_data) == len(run_data):
                    t_stat, p_val = stats.ttest_rel(baseline_data, run_data)
                    improvement = (baseline_data.mean() - run_data.mean()) / baseline_data.mean() * 100
                    
                    # Significance stars
                    if p_val < 0.001:
                        sig_stars = '***'
                    elif p_val < 0.01:
                        sig_stars = '**'
                    elif p_val < 0.05:
                        sig_stars = '*'
                    else:
                        sig_stars = ''
                    
                    out.write(f"{run:25} | Delta: {improvement:+5.1f}% | p-value: {p_val:.4f} {sig_stars}\n")
                else:
                    out.write(f"{run:25} | Skipped: unequal sample size with baseline\n")
        
        out.write("\n" + "=" * 60 + "\n")
        out.write("LEGEND:\n")
        out.write("  Delta%: Percentage improvement over baseline (positive = better for MAPE/MASE)\n")
        out.write("  Significance: *** p<0.001, ** p<0.01, * p<0.05\n")
        out.write("  IQR: Interquartile Range (Q75 - Q25)\n")
        out.write("=" * 60 + "\n")


def run_experimental_analysis(run_configs):
    """
    A function to analyze multiple experimental runs
    
    Parameters:
    - run_configs: List of tuples (run_name, log_filepath)
    """
    analyzer = ExperimentalAnalysis(run_configs)
    
    if analyzer.combined_results.empty:
        print("No data to analyze")
        return None
    
    # Perform all analyses
    analyzer.compare_runs_statistical()
    analyzer.create_boxplots()
    analyzer.generate_experimental_summary()
    
    stat_output_path = RESULTS_PATH / "runs_statistics.txt"
    with open(stat_output_path, "w") as f:
        analyzer.print_all_run_stats(baseline_run='base-0', out = f)
        analyzer.generate_experimental_summary(out=f)
    
    return analyzer

def experimental_analysis():
    """Example of how to analyze your experimental runs"""
    folder = RESULTS_PATH
    run_configs = [
        ('base-0', f'{folder}/exp1_base_no_reg.csv'),
        ('adaptive', f'{folder}/exp2_adaptive_reg.csv'),
        ('fixed-1', f'{folder}/exp3_fixed_regs_1.csv'),
        ('fixed-2', f'{folder}/exp4_fixed_regs_2.csv'),
        ('fixed-3', f'{folder}/exp5_fixed_regs_3.csv'),
        ('fixed-4', f'{folder}/exp6_fixed_regs_4.csv'),
        ('fixed-5', f'{folder}/exp7_fixed_regs_5.csv'),
        ('fixed-6', f'{folder}/exp8_fixed_regs_6.csv'),
        ('fixed-7', f'{folder}/exp9_fixed_regs_7.csv'),
    ]
    
    return run_experimental_analysis(run_configs)

if __name__ == "__main__":
    print_options()
    x = experimental_analysis()
