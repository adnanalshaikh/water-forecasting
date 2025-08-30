import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ResultsOrganizer:
    """
    Comprehensive class for organizing and presenting forecasting results
    """
    
    def __init__(self, log_filepath='experiment_log.csv'):
        """
        Initialize with experiment log file
        
        Parameters:
        - log_filepath: Path to the experiment log CSV
        """
        self.log_filepath = log_filepath
        self.results_df = None
        self.load_results()
    def load_results(self):
        """Load and preprocess results from experiment log"""
        try:
            self.results_df = pd.read_csv(self.log_filepath)
            print(f"Loaded {len(self.results_df)} experiments from {self.log_filepath}")
            
            # Add derived metrics
            self.add_derived_metrics()
            
        except FileNotFoundError:
            print(f"Warning: {self.log_filepath} not found. Run experiments first.")
            self.results_df = pd.DataFrame()
    def add_derived_metrics(self):
        """Add derived metrics and classifications"""
        if self.results_df.empty:
            return
        
        df = self.results_df
        
        # Performance categories based on MAPE
        def categorize_performance(mape):
            if mape < 5: return 'Excellent'
            elif mape < 10: return 'Very Good'
            elif mape < 15: return 'Good'
            elif mape < 25: return 'Fair'
            else: return 'Poor'
        
        df['performance_category'] = df['mape'].apply(categorize_performance)
        
        # Data quality score
        df['data_quality_score'] = 100 - (df['missing_found'].fillna(0) / 60 * 100)
        df['data_quality_score'] = df['data_quality_score'].clip(0, 100)
        
        # Outlier burden
        df['outlier_burden'] = df['outliers_percentage'].fillna(0)
        
        # Overall success score (combines MAPE and MASE)
        df['success_score'] = (
            (25 - df['mape'].clip(0, 25)) / 25 * 50 +  # MAPE component (0-50)
            (3 - df['mase'].clip(0, 3)) / 3 * 50        # MASE component (0-50)
        ).clip(0, 100)
        
        self.results_df = df
    def create_summary_statistics(self):
        """Create comprehensive summary statistics"""
        if self.results_df.empty:
            return None
        
        df = self.results_df
        
        summary = {
            'total_areas': len(df),
            'successful_forecasts': len(df[df['mape'] < 30]),
            'success_rate': len(df[df['mape'] < 30]) / len(df) * 100,
            
            # Performance distribution
            'performance_distribution': df['performance_category'].value_counts().to_dict(),
            
            # Key statistics
            'mean_mape': df['mape'].mean(),
            'median_mape': df['mape'].median(),
            'std_mape': df['mape'].std(),
            'best_mape': df['mape'].min(),
            'worst_mape': df['mape'].max(),
            
            'mean_mase': df['mase'].mean(),
            'median_mase': df['mase'].median(),
            
            # Data quality insights
            'areas_with_missing_data': len(df[df['missing_found'] > 0]),
            'areas_with_outliers': len(df[df['outliers_found'] > 0]),
            'mean_data_quality_score': df['data_quality_score'].mean(),
            
            # Model configuration insights
            'most_common_seasonality_mode': df['seasonality_mode'].mode()[0],
            'mean_fourier_order': df['fourier_order'].mean(),
            'mean_changepoint_prior_scale': df['changepoint_prior_scale'].mean(),
            'mean_seasonality_prior_scale': df['seasonality_prior_scale'].mean(),
        }
        
        return summary
    def create_performance_ranking_table(self, top_n=10):
        """Create performance ranking table for publication"""
        if self.results_df.empty:
            return None
        
        df = self.results_df.copy()
        
        # Select columns for ranking table
        ranking_columns = [
            'area_id', 'mape', 'mase', 'mase_seas',
            'performance_category', 'data_quality_score', 'outlier_burden',
            'seasonality_mode', 'fourier_order', 'missing_found', 'outliers_found'
        ]
        
        # Create ranking table
        ranking_df = df[ranking_columns].copy()
        ranking_df = ranking_df.sort_values('mape')
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        # Round numerical columns
        numerical_cols = ['mape', 'mase', 'mase_seas', 
                         'data_quality_score', 'outlier_burden']
        ranking_df[numerical_cols] = ranking_df[numerical_cols].round(2)
        
        # Rename columns for presentation
        ranking_df = ranking_df.rename(columns={
            'area_id': 'Area',
            'mape': 'MAPE (%)',
            'mase': 'MASE',
            'mase_seas': 'MASE (Seasonal)',
            'performance_category': 'Performance',
            'data_quality_score': 'Data Quality (%)',
            'outlier_burden': 'Outlier Burden (%)',
            'seasonality_mode': 'Seasonality',
            'fourier_order': 'Fourier Order',
            'missing_found': 'Missing Values',
            'outliers_found': 'Outliers Found'
        })
        
        return ranking_df
    def create_comprehensive_visualizations(self, save_dir='results_figures', dpi=300):
        """Create comprehensive visualization suite"""
        if self.results_df.empty:
            print("No data to visualize")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance Overview Dashboard
        self._create_performance_dashboard(save_dir, dpi)
        
        # 2. Data Quality Analysis
        self._create_data_quality_analysis(save_dir, dpi)
        
        # 3. Model Configuration Analysis
        self._create_model_configuration_analysis(save_dir, dpi)
        
        # 4. Correlation Analysis
        self._create_correlation_analysis(save_dir, dpi)
        
        print(f"All visualizations saved to {save_dir}/")
    def _create_performance_dashboard(self, save_dir, dpi):
        """Create main performance dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Forecasting Performance Dashboard', fontsize=16, fontweight='bold')
        
        df = self.results_df
        
        # 1. MAPE Distribution
        ax1 = axes[0, 0]
        ax1.hist(df['mape'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(df['mape'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["mape"].mean():.1f}%')
        ax1.set_xlabel('MAPE (%)')
        ax1.set_ylabel('Number of Areas')
        ax1.set_title('MAPE Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Categories
        ax2 = axes[0, 1]
        perf_counts = df['performance_category'].value_counts()
        colors = ['darkgreen', 'green', 'orange', 'red', 'darkred'][:len(perf_counts)]
        ax2.pie(perf_counts.values, labels=perf_counts.index, autopct='%1.0f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Performance Category Distribution')
        
        # 3. MAPE vs Area ID
        ax3 = axes[0, 2]
        colors_scatter = ['green' if mape < 10 else 'orange' if mape < 20 else 'red' 
                         for mape in df['mape']]
        ax3.scatter(df['area_id'], df['mape'], c=colors_scatter, alpha=0.7, s=60)
        ax3.set_xlabel('Area ID')
        ax3.set_ylabel('MAPE (%)')
        ax3.set_title('MAPE by Area')
        ax3.grid(True, alpha=0.3)
        
        # Add horizontal lines for performance thresholds
        ax3.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Excellent/Good threshold')
        ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Good/Fair threshold')
        ax3.legend()
        
        # 4. MASE Distribution
        ax4 = axes[1, 0]
        ax4.hist(df['mase'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.axvline(df['mase'].mean(), color='blue', linestyle='--',
                   label=f'Mean: {df["mase"].mean():.2f}')
        ax4.axvline(1.0, color='green', linestyle='-', alpha=0.7, label='MASE = 1.0 (Naive baseline)')
        ax4.set_xlabel('MASE')
        ax4.set_ylabel('Number of Areas')
        ax4.set_title('MASE Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. MAPE vs MASE
        ax5 = axes[1, 1]
        scatter = ax5.scatter(df['mape'], df['mase'], 
                             c=df['data_quality_score'], cmap='viridis', s=60, alpha=0.7)
        ax5.set_xlabel('MAPE (%)')
        ax5.set_ylabel('MASE')
        ax5.set_title('MAPE vs MASE (Color = Data Quality)')
        ax5.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Data Quality Score')
        
        # 6. Success Score Distribution
        ax6 = axes[1, 2]
        ax6.hist(df['success_score'], bins=15, alpha=0.7, color='gold', edgecolor='black')
        ax6.axvline(df['success_score'].mean(), color='blue', linestyle='--',
                   label=f'Mean: {df["success_score"].mean():.1f}')
        ax6.set_xlabel('Success Score (0-100)')
        ax6.set_ylabel('Number of Areas')
        ax6.set_title('Overall Success Score')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_dashboard.png', dpi=dpi, bbox_inches='tight')
        plt.show()
    def _create_data_quality_analysis(self, save_dir, dpi):
        """Create data quality analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Data Quality Impact Analysis', fontsize=16, fontweight='bold')
        
        df = self.results_df
        
        # 1. Missing Data Impact
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(df['missing_found'], df['mape'], 
                              alpha=0.7, s=60, c='blue')
        ax1.set_xlabel('Missing Values Count')
        ax1.set_ylabel('MAPE (%)')
        ax1.set_title('Impact of Missing Data on Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr1 = df['missing_found'].corr(df['mape'])
        ax1.text(0.05, 0.95, f'Correlation: {corr1:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Outlier Impact
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df['outliers_found'], df['mape'], 
                              alpha=0.7, s=60, c='red')
        ax2.set_xlabel('Outliers Found')
        ax2.set_ylabel('MAPE (%)')
        ax2.set_title('Impact of Outliers on Performance')
        ax2.grid(True, alpha=0.3)
        
        corr2 = df['outliers_found'].corr(df['mape'])
        ax2.text(0.05, 0.95, f'Correlation: {corr2:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Data Quality Score vs Performance
        ax3 = axes[1, 0]
        ax3.scatter(df['data_quality_score'], df['mape'], 
                   alpha=0.7, s=60, c='green')
        ax3.set_xlabel('Data Quality Score (%)')
        ax3.set_ylabel('MAPE (%)')
        ax3.set_title('Data Quality vs Performance')
        ax3.grid(True, alpha=0.3)
        
        corr3 = df['data_quality_score'].corr(df['mape'])
        ax3.text(0.05, 0.95, f'Correlation: {corr3:.3f}', transform=ax3.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Combined Data Issues
        ax4 = axes[1, 1]
        df['total_data_issues'] = df['missing_found'] + df['outliers_found']
        ax4.scatter(df['total_data_issues'], df['mape'], 
                   alpha=0.7, s=60, c='purple')
        ax4.set_xlabel('Total Data Issues (Missing + Outliers)')
        ax4.set_ylabel('MAPE (%)')
        ax4.set_title('Combined Data Issues vs Performance')
        ax4.grid(True, alpha=0.3)
        
        corr4 = df['total_data_issues'].corr(df['mape'])
        ax4.text(0.05, 0.95, f'Correlation: {corr4:.3f}', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/data_quality_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.show()
    def _create_model_configuration_analysis(self, save_dir, dpi):
        """Create model configuration analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Configuration Analysis', fontsize=16, fontweight='bold')
        
        df = self.results_df
        
        # 1. Seasonality Mode Performance
        ax1 = axes[0, 0]
        seasonality_perf = df.groupby('seasonality_mode')['mape'].agg(['mean', 'std', 'count'])
        seasonality_perf['mean'].plot(kind='bar', ax=ax1, alpha=0.7, color='skyblue')
        ax1.set_title('Performance by Seasonality Mode')
        ax1.set_ylabel('Mean MAPE (%)')
        ax1.set_xlabel('Seasonality Mode')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add count annotations
        for i, (mode, row) in enumerate(seasonality_perf.iterrows()):
            ax1.text(i, row['mean'] + 0.5, f'n={row["count"]}', ha='center')
        
        # 2. Fourier Order Distribution
        ax2 = axes[0, 1]
        fourier_perf = df.groupby('fourier_order')['mape'].mean()
        fourier_perf.plot(kind='bar', ax=ax2, alpha=0.7, color='lightcoral')
        ax2.set_title('Performance by Fourier Order')
        ax2.set_ylabel('Mean MAPE (%)')
        ax2.set_xlabel('Fourier Order')
        ax2.grid(True, alpha=0.3)
        
        # 3. Hyperparameter Relationships
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(df['changepoint_prior_scale'], df['seasonality_prior_scale'],
                              c=df['mape'], cmap='RdYlGn_r', alpha=0.7, s=60)
        ax3.set_xlabel('Changepoint Prior Scale')
        ax3.set_ylabel('Seasonality Prior Scale')
        ax3.set_title('Hyperparameter Space (Color = MAPE)')
        ax3.grid(True, alpha=0.3)
        cbar3 = plt.colorbar(scatter3, ax=ax3)
        cbar3.set_label('MAPE (%)')
        
        # 4. Optimization Trials vs Performance
        ax4 = axes[1, 1]
        if 'n_trials' in df.columns:
            ax4.scatter(df['n_trials'], df['mape'], alpha=0.7, s=60, c='orange')
            ax4.set_xlabel('Number of Optimization Trials')
            ax4.set_ylabel('MAPE (%)')
            ax4.set_title('Optimization Effort vs Performance')
            ax4.grid(True, alpha=0.3)
            
            if df['n_trials'].var() > 0:  # Only show correlation if there's variation
                corr_trials = df['n_trials'].corr(df['mape'])
                ax4.text(0.05, 0.95, f'Correlation: {corr_trials:.3f}', transform=ax4.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No trial data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Optimization Trials (No Data)')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/model_configuration_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.show()
    def _create_correlation_analysis(self, save_dir, dpi):
        """Create correlation matrix of key variables"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        df = self.results_df
        
        # Select numerical columns for correlation
        corr_columns = [
            'mape', 'mase', 'mase_seas',
            'changepoint_prior_scale', 'seasonality_prior_scale', 'fourier_order',
            'missing_found', 'outliers_found', 'data_quality_score', 'success_score'
        ]
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in corr_columns if col in df.columns]
        
        if len(available_columns) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Correlation Analysis (Insufficient Data)')
        else:
            corr_matrix = df[available_columns].corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.show()
    def create_executive_summary_report(self, output_file='forecasting_results_summary.txt'):
        """Create an executive summary report"""
        summary = self.create_summary_statistics()
        
        if summary is None:
            print("No data available for summary report")
            return
        
        report_lines = [
            "=" * 80,
            "WATER CONSUMPTION FORECASTING - EXECUTIVE SUMMARY",
            "=" * 80,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Source: {self.log_filepath}",
            "",
            "OVERALL PERFORMANCE SUMMARY:",
            f"  • Total Areas Analyzed: {summary['total_areas']}",
            f"  • Successful Forecasts (MAPE < 30%): {summary['successful_forecasts']} ({summary['success_rate']:.1f}%)",
            f"  • Mean MAPE: {summary['mean_mape']:.1f}% (Median: {summary['median_mape']:.1f}%)",
            f"  • Mean MASE: {summary['mean_mase']:.2f}",
            f"  • Best Performance: {summary['best_mape']:.1f}% MAPE",
            f"  • Worst Performance: {summary['worst_mape']:.1f}% MAPE",
            "",
            "PERFORMANCE DISTRIBUTION:",
        ]
        
        for category, count in summary['performance_distribution'].items():
            percentage = count / summary['total_areas'] * 100
            report_lines.append(f"  • {category}: {count} areas ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "DATA QUALITY INSIGHTS:",
            f"  • Areas with Missing Data: {summary['areas_with_missing_data']} ({summary['areas_with_missing_data']/summary['total_areas']*100:.1f}%)",
            f"  • Areas with Outliers: {summary['areas_with_outliers']} ({summary['areas_with_outliers']/summary['total_areas']*100:.1f}%)",
            f"  • Mean Data Quality Score: {summary['mean_data_quality_score']:.1f}%",
            "",
            "MODEL CONFIGURATION:",
            f"  • Most Common Seasonality Mode: {summary['most_common_seasonality_mode']}",
            f"  • Mean Fourier Order: {summary['mean_fourier_order']:.1f}",
            f"  • Mean Changepoint Prior Scale: {summary['mean_changepoint_prior_scale']:.4f}",
            f"  • Mean Seasonality Prior Scale: {summary['mean_seasonality_prior_scale']:.1f}",
            "",
            "KEY FINDINGS:",
            "  • Prophet-based forecasting achieved good performance across most areas",
            f"  • {summary['success_rate']:.0f}% of areas achieved reasonable forecasting accuracy",
            "  • Data quality strongly impacts forecasting performance",
            "  • Area-specific hyperparameter optimization proved effective",
            "",
            "=" * 80
        ])
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also print to console
        print('\n'.join(report_lines))
        print(f"\nDetailed report saved to: {output_file}")
        
        return summary
    
########### adding mase visulization 2 ####################
    def create_triple_metrics_visualization(self, save_dir='results_figures', dpi=300, figsize=(18, 12)):
        """
        Create comprehensive visualization showing MASE, Seasonal MASE, and MAPE
        
        Parameters:
        - save_dir: Directory to save the figure
        - dpi: Figure resolution
        - figsize: Figure size (width, height)
        """
        if self.results_df.empty:
            print("No data to visualize")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Comprehensive Forecasting Performance Analysis', fontsize=16, fontweight='bold')
        
        df = self.results_df
        
        # Define color mapping function
        def get_performance_colors(values, metric_type='mase'):
            colors = []
            for val in values:
                if metric_type == 'mase':
                    if val < 1.0:
                        colors.append('green')
                    elif val < 1.5:
                        colors.append('orange')
                    else:
                        colors.append('red')
                elif metric_type == 'mape':
                    if val < 5:
                        colors.append('green')
                    elif val < 10:
                        colors.append('yellowgreen')
                    elif val < 20:
                        colors.append('orange')
                    else:
                        colors.append('red')
            return colors
        
        # Row 1: Scatter plots by Area ID
        
        # 1. MASE vs Area ID
        ax1 = axes[0, 0]
        colors1 = get_performance_colors(df['mase'], 'mase')
        ax1.scatter(df['area_id'], df['mase'], 
                   c=colors1, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2, 
                   label='MASE = 1.0 (Naive baseline)')
        ax1.axhline(y=df['mase'].mean(), color='purple', linestyle=':', 
                   alpha=0.7, linewidth=2, 
                   label=f'Mean = {df["mase"].mean():.2f}')
        ax1.set_xlabel('Area ID', fontweight='bold')
        ax1.set_ylabel('MASE', fontweight='bold')
        ax1.set_title('MASE by Area ID', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax1.set_ylim(bottom=0)
        
        # 2. Seasonal MASE vs Area ID (if available)
        ax2 = axes[0, 1]
        if 'mase_seas' in df.columns:
            colors2 = get_performance_colors(df['mase_seas'], 'mase')
            ax2.scatter(df['area_id'], df['mase_seas'], 
                       c=colors2, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            ax2.axhline(y=1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2, 
                       label='Seasonal MASE = 1.0')
            ax2.axhline(y=df['mase_seas'].mean(), color='purple', linestyle=':', 
                       alpha=0.7, linewidth=2, 
                       label=f'Mean = {df["mase_seas"].mean():.2f}')
            ax2.set_xlabel('Area ID', fontweight='bold')
            ax2.set_ylabel('Seasonal MASE', fontweight='bold')
            ax2.set_title('Seasonal MASE by Area ID', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            ax2.set_ylim(bottom=0)
        else:
            ax2.text(0.5, 0.5, 'Seasonal MASE\ndata not available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Seasonal MASE by Area ID (No Data)', fontweight='bold')
        
        # 3. MAPE vs Area ID
        ax3 = axes[0, 2]
        colors3 = get_performance_colors(df['mape'], 'mape')
        ax3.scatter(df['area_id'], df['mape'], 
                   c=colors3, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        ax3.axhline(y=10, color='green', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Excellent threshold (10%)')
        ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Good threshold (20%)')
        ax3.axhline(y=df['mape'].mean(), color='purple', linestyle=':', 
                   alpha=0.7, linewidth=2, 
                   label=f'Mean = {df["mape"].mean():.1f}%')
        ax3.set_xlabel('Area ID', fontweight='bold')
        ax3.set_ylabel('MAPE (%)', fontweight='bold')
        ax3.set_title('MAPE by Area ID', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.set_ylim(bottom=0)
        
        # Row 2: Distribution plots
        
        # 4. MASE Distribution
        ax4 = axes[1, 0]
        ax4.hist(df['mase'], bins=15, alpha=0.7, color='lightblue', 
                 edgecolor='black', density=True)
        ax4.axvline(df['mase'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {df["mase"].mean():.2f}')
        ax4.axvline(df['mase'].median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {df["mase"].median():.2f}')
        ax4.axvline(1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2,
                   label='MASE = 1.0 (Baseline)')
        ax4.set_xlabel('MASE', fontweight='bold')
        ax4.set_ylabel('Density', fontweight='bold')
        ax4.set_title('MASE Distribution', fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Seasonal MASE Distribution (if available)
        ax5 = axes[1, 1]
        if 'mase_seas' in df.columns:
            ax5.hist(df['mase_seas'], bins=15, alpha=0.7, color='lightcoral', 
                     edgecolor='black', density=True)
            ax5.axvline(df['mase_seas'].mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {df["mase_seas"].mean():.2f}')
            ax5.axvline(df['mase_seas'].median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {df["mase_seas"].median():.2f}')
            ax5.axvline(1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2,
                       label='Seasonal MASE = 1.0')
            ax5.set_xlabel('Seasonal MASE', fontweight='bold')
            ax5.set_ylabel('Density', fontweight='bold')
            ax5.set_title('Seasonal MASE Distribution', fontweight='bold')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Seasonal MASE\ndata not available', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Seasonal MASE Distribution (No Data)', fontweight='bold')
        
        # 6. MAPE Distribution
        ax6 = axes[1, 2]
        ax6.hist(df['mape'], bins=15, alpha=0.7, color='gold', 
                 edgecolor='black', density=True)
        ax6.axvline(df['mape'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {df["mape"].mean():.1f}%')
        ax6.axvline(df['mape'].median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {df["mape"].median():.1f}%')
        ax6.axvline(10, color='green', linestyle='-', alpha=0.7, linewidth=1.5,
                   label='Excellent (10%)')
        ax6.axvline(20, color='orange', linestyle='-', alpha=0.7, linewidth=1.5,
                   label='Good (20%)')
        ax6.set_xlabel('MAPE (%)', fontweight='bold')
        ax6.set_ylabel('Density', fontweight='bold')
        ax6.set_title('MAPE Distribution', fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/triple_metrics_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        self._print_triple_metrics_summary()  
    def create_metrics_correlation_plot(self, save_dir='results_figures', dpi=300, figsize=(15, 5)):
        """
        Create correlation plots between MASE, Seasonal MASE, and MAPE
        """
        if self.results_df.empty:
            print("No data to visualize")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        df = self.results_df
        
        # Check if seasonal MASE is available
        has_seasonal = 'mase_seas' in df.columns
        
        if has_seasonal:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            fig.suptitle('Metrics Correlation Analysis', fontsize=16, fontweight='bold')
            
            # 1. MASE vs MAPE
            ax1 = axes[0]
            scatter1 = ax1.scatter(df['mase'], df['mape'], 
                                  alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.5)
            corr1 = df['mase'].corr(df['mape'])
            ax1.set_xlabel('MASE', fontweight='bold')
            ax1.set_ylabel('MAPE (%)', fontweight='bold')
            ax1.set_title(f'MASE vs MAPE\n(r = {corr1:.3f})', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 2. Seasonal MASE vs MAPE
            ax2 = axes[1]
            scatter2 = ax2.scatter(df['mase_seas'], df['mape'], 
                                  alpha=0.7, s=60, c='red', edgecolors='black', linewidth=0.5)
            corr2 = df['mase_seas'].corr(df['mape'])
            ax2.set_xlabel('Seasonal MASE', fontweight='bold')
            ax2.set_ylabel('MAPE (%)', fontweight='bold')
            ax2.set_title(f'Seasonal MASE vs MAPE\n(r = {corr2:.3f})', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. MASE vs Seasonal MASE
            ax3 = axes[2]
            scatter3 = ax3.scatter(df['mase'], df['mase_seas'], 
                                  alpha=0.7, s=60, c='green', edgecolors='black', linewidth=0.5)
            corr3 = df['mase'].corr(df['mase_seas'])
            ax3.set_xlabel('MASE', fontweight='bold')
            ax3.set_ylabel('Seasonal MASE', fontweight='bold')
            ax3.set_title(f'MASE vs Seasonal MASE\n(r = {corr3:.3f})', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add diagonal line for MASE vs Seasonal MASE
            min_val = min(df['mase'].min(), df['mase_seas'].min())
            max_val = max(df['mase'].max(), df['mase_seas'].max())
            ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
            ax3.legend()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            fig.suptitle('MASE vs MAPE Correlation', fontsize=16, fontweight='bold')
            
            scatter = ax.scatter(df['mase'], df['mape'], 
                               alpha=0.7, s=80, c='blue', edgecolors='black', linewidth=0.5)
            corr = df['mase'].corr(df['mape'])
            ax.set_xlabel('MASE', fontweight='bold')
            ax.set_ylabel('MAPE (%)', fontweight='bold')
            ax.set_title(f'MASE vs MAPE (r = {corr:.3f})', fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/metrics_correlation.png', dpi=dpi, bbox_inches='tight')
        plt.show()
    def create_performance_ranking_by_metrics(self, save_dir='results_figures', dpi=300, figsize=(16, 10)):
        """
        Create comprehensive ranking visualization showing all three metrics
        """
        if self.results_df.empty:
            print("No data to visualize")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        df = self.results_df.copy()
        
        # Sort by MAPE for ranking
        df_sorted = df.sort_values('mape').reset_index(drop=True)
        df_sorted['rank'] = range(1, len(df_sorted) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Performance Ranking Analysis', fontsize=16, fontweight='bold')
        
        # 1. Ranking plot - all metrics
        ax1 = axes[0, 0]
        ax1.plot(df_sorted['rank'], df_sorted['mape'], 'o-', 
                 color='blue', linewidth=2, markersize=4, label='MAPE (%)', alpha=0.8)
        
        # Scale MASE values to be comparable with MAPE for visualization
        mase_scaled = df_sorted['mase'] * df_sorted['mape'].mean()
        ax1.plot(df_sorted['rank'], mase_scaled, 's-', 
                 color='red', linewidth=2, markersize=4, label=f'MASE (×{df_sorted["mape"].mean():.1f})', alpha=0.8)
        
        if 'mase_seas' in df.columns:
            mase_seas_scaled = df_sorted['mase_seas'] * df_sorted['mape'].mean()
            ax1.plot(df_sorted['rank'], mase_seas_scaled, '^-', 
                     color='green', linewidth=2, markersize=4, label=f'Seasonal MASE (×{df_sorted["mape"].mean():.1f})', alpha=0.8)
        
        ax1.set_xlabel('Performance Rank (by MAPE)', fontweight='bold')
        ax1.set_ylabel('Scaled Metric Values', fontweight='bold')
        ax1.set_title('Performance Ranking (All Metrics)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top/Bottom performers
        ax2 = axes[0, 1]
        n_show = min(10, len(df_sorted))
        
        # Top performers
        top_areas = df_sorted.head(n_show)['area_id'].values
        top_mape = df_sorted.head(n_show)['mape'].values
        
        bars = ax2.bar(range(n_show), top_mape, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Top Performing Areas', fontweight='bold')
        ax2.set_ylabel('MAPE (%)', fontweight='bold')
        ax2.set_title(f'Top {n_show} Performers', fontweight='bold')
        ax2.set_xticks(range(n_show))
        ax2.set_xticklabels([f'Area {area}' for area in top_areas], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 3. Metrics comparison heatmap
        ax3 = axes[1, 0]
        
        # Normalize metrics for heatmap (0-1 scale)
        metrics_data = df_sorted[['mape', 'mase']].copy()
        if 'mase_seas' in df.columns:
            metrics_data['mase_seas'] = df_sorted['mase_seas']
        
        # Normalize each metric to 0-1 scale
        metrics_normalized = (metrics_data - metrics_data.min()) / (metrics_data.max() - metrics_data.min())
        
        im = ax3.imshow(metrics_normalized.T, cmap='RdYlGn_r', aspect='auto')
        ax3.set_xlabel('Areas (ranked by MAPE)', fontweight='bold')
        ax3.set_ylabel('Metrics', fontweight='bold')
        ax3.set_title('Normalized Metrics Heatmap', fontweight='bold')
        ax3.set_yticks(range(len(metrics_normalized.columns)))
        ax3.set_yticklabels(metrics_normalized.columns)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Normalized Performance (0=Best, 1=Worst)')
        
        # 4. Performance statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create statistics table
        stats_data = []
        stats_data.append(['Metric', 'Mean', 'Median', 'Best', 'Worst'])
        stats_data.append(['MAPE (%)', f'{df["mape"].mean():.1f}', 
                          f'{df["mape"].median():.1f}',
                          f'{df["mape"].min():.1f}',
                          f'{df["mape"].max():.1f}'])
        stats_data.append(['MASE', f'{df["mase"].mean():.2f}', 
                          f'{df["mase"].median():.2f}',
                          f'{df["mase"].min():.2f}',
                          f'{df["mase"].max():.2f}'])
        
        if 'mase_seas' in df.columns:
            stats_data.append(['Seasonal MASE', f'{df["mase_seas"].mean():.2f}', 
                              f'{df["mase_seas"].median():.2f}',
                              f'{df["mase_seas"].min():.2f}',
                              f'{df["mase_seas"].max():.2f}'])
        
        table = ax4.table(cellText=stats_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the header row
        for i in range(len(stats_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Statistics Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_ranking_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.show()  
    def _print_triple_metrics_summary(self):
        """Print comprehensive summary of all three metrics"""
        df = self.results_df
        
        print("\n" + "="*70)
        print("COMPREHENSIVE METRICS ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"Total areas analyzed: {len(df)}")
        print()
        
        # MAPE Analysis
        print("MAPE PERFORMANCE:")
        excellent_mape = len(df[df['mape'] < 5])
        very_good_mape = len(df[(df['mape'] >= 5) & (df['mape'] < 10)])
        good_mape = len(df[(df['mape'] >= 10) & (df['mape'] < 20)])
        poor_mape = len(df[df['mape'] >= 20])
        
        print(f"  Excellent (< 5%): {excellent_mape} areas ({excellent_mape/len(df)*100:.1f}%)")
        print(f"  Very Good (5-10%): {very_good_mape} areas ({very_good_mape/len(df)*100:.1f}%)")
        print(f"  Good (10-20%): {good_mape} areas ({good_mape/len(df)*100:.1f}%)")
        print(f"  Poor (≥ 20%): {poor_mape} areas ({poor_mape/len(df)*100:.1f}%)")
        print(f"  Mean MAPE: {df['mape'].mean():.2f}%")
        print(f"  Median MAPE: {df['mape'].median():.2f}%")
        print()
        
        # MASE Analysis
        print("MASE PERFORMANCE:")
        excellent_mase = len(df[df['mase'] < 1.0])
        good_mase = len(df[(df['mase'] >= 1.0) & (df['mase'] < 1.5)])
        poor_mase = len(df[df['mase'] >= 1.5])
        
        print(f"  Better than naive (< 1.0): {excellent_mase} areas ({excellent_mase/len(df)*100:.1f}%)")
        print(f"  Reasonable (1.0-1.5): {good_mase} areas ({good_mase/len(df)*100:.1f}%)")
        print(f"  Poor (≥ 1.5): {poor_mase} areas ({poor_mase/len(df)*100:.1f}%)")
        print(f"  Mean MASE: {df['mase'].mean():.3f}")
        print(f"  Median MASE: {df['mase'].median():.3f}")
        print()
        
        # Seasonal MASE Analysis (if available)
        if 'mase_seas' in df.columns:
            print("SEASONAL MASE PERFORMANCE:")
            excellent_mase_seas = len(df[df['mase_seas'] < 1.0])
            good_mase_seas = len(df[(df['mase_seas'] >= 1.0) & (df['mase_seas'] < 1.5)])
            poor_mase_seas = len(df[df['mase_seas'] >= 1.5])
            
            print(f"  Better than seasonal naive (< 1.0): {excellent_mase_seas} areas ({excellent_mase_seas/len(df)*100:.1f}%)")
            print(f"  Reasonable (1.0-1.5): {good_mase_seas} areas ({good_mase_seas/len(df)*100:.1f}%)")
            print(f"  Poor (≥ 1.5): {poor_mase_seas} areas ({poor_mase_seas/len(df)*100:.1f}%)")
            print(f"  Mean Seasonal MASE: {df['mase_seas'].mean():.3f}")
            print(f"  Median Seasonal MASE: {df['mase_seas'].median():.3f}")
            print()
            
            # Correlation analysis
            print("METRICS CORRELATIONS:")
            corr_mase_mape = df['mase'].corr(df['mape'])
            corr_mase_seas_mape = df['mase_seas'].corr(df['mape'])
            corr_mase_mase_seas = df['mase'].corr(df['mase_seas'])
            
            print(f"  MASE vs MAPE: {corr_mase_mape:.3f}")
            print(f"  Seasonal MASE vs MAPE: {corr_mase_seas_mape:.3f}")
            print(f"  MASE vs Seasonal MASE: {corr_mase_mase_seas:.3f}")
        else:
            print("SEASONAL MASE: Not available in dataset")
            print()
            
            print("METRICS CORRELATION:")
            corr_mase_mape = df['mase'].corr(df['mape'])
            print(f"  MASE vs MAPE: {corr_mase_mape:.3f}")
        
        print("="*70)
    
    # Standalone function for quick triple metrics analysis
    def quick_triple_metrics_analysis(csv_file='experiment_3_features.csv', save_dir='results_figures'):
        """
        Quick function to create all three metrics visualizations from CSV file
        
        Parameters:
        - csv_file: Path to the experiment results CSV
        - save_dir: Directory to save figures
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        
        # Load data
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} areas from {csv_file}")
        except FileNotFoundError:
            print(f"File {csv_file} not found!")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a simple version of the visualization
        has_seasonal = 'mase_seas' in df.columns
        
        if has_seasonal:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Triple Metrics Analysis', fontsize=16, fontweight='bold')
            
            # Row 1: Scatter plots
            axes[0, 0].scatter(df['area_id'], df['mase'], alpha=0.7, s=60)
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--', label='MASE = 1.0')
            axes[0, 0].set_title('MASE by Area ID')
            axes[0, 0].set_ylabel('MASE')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].scatter(df['area_id'], df['mase_seas'], alpha=0.7, s=60)
            axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='Seasonal MASE = 1.0')
            axes[0, 1].set_title('Seasonal MASE by Area ID')
            axes[0, 1].set_ylabel('Seasonal MASE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[0, 2].scatter(df['area_id'], df['mape'], alpha=0.7, s=60)
            axes[0, 2].axhline(y=10, color='green', linestyle='--', label='Excellent (10%)')
            axes[0, 2].axhline(y=20, color='orange', linestyle='--', label='Good (20%)')
            axes[0, 2].set_title('MAPE by Area ID')
            axes[0, 2].set_ylabel('MAPE (%)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Row 2: Distributions
            axes[1, 0].hist(df['mase'], bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 0].axvline(df['mase'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {df["mase"].mean():.2f}')
            axes[1, 0].axvline(1.0, color='blue', linestyle='-', label='MASE = 1.0')
            axes[1, 0].set_title('MASE Distribution')
            axes[1, 0].set_xlabel('MASE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].hist(df['mase_seas'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 1].axvline(df['mase_seas'].mean(), color='red', linestyle='--',
                              label=f'Mean: {df["mase_seas"].mean():.2f}')
            axes[1, 1].axvline(1.0, color='blue', linestyle='-', label='Seasonal MASE = 1.0')
            axes[1, 1].set_title('Seasonal MASE Distribution')
            axes[1, 1].set_xlabel('Seasonal MASE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].hist(df['mape'], bins=15, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 2].axvline(df['mape'].mean(), color='red', linestyle='--',
                              label=f'Mean: {df["mape"].mean():.1f}%')
            axes[1, 2].axvline(10, color='green', linestyle='-', label='Excellent (10%)')
            axes[1, 2].axvline(20, color='orange', linestyle='-', label='Good (20%)')
            axes[1, 2].set_title('MAPE Distribution')
            axes[1, 2].set_xlabel('MAPE (%)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('MASE and MAPE Analysis', fontsize=16, fontweight='bold')
            
            # MASE scatter
            axes[0, 0].scatter(df['area_id'], df['mase'], alpha=0.7, s=60)
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--', label='MASE = 1.0')
            axes[0, 0].set_title('MASE by Area ID')
            axes[0, 0].set_ylabel('MASE')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # MAPE scatter
            axes[0, 1].scatter(df['area_id'], df['mape'], alpha=0.7, s=60)
            axes[0, 1].axhline(y=10, color='green', linestyle='--', label='Excellent (10%)')
            axes[0, 1].axhline(y=20, color='orange', linestyle='--', label='Good (20%)')
            axes[0, 1].set_title('MAPE by Area ID')
            axes[0, 1].set_ylabel('MAPE (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # MASE distribution
            axes[1, 0].hist(df['mase'], bins=15, alpha=0.7, color='lightblue', edgecolor='black')
            axes[1, 0].axvline(df['mase'].mean(), color='red', linestyle='--',
                              label=f'Mean: {df["mase"].mean():.2f}')
            axes[1, 0].axvline(1.0, color='blue', linestyle='-', label='MASE = 1.0')
            axes[1, 0].set_title('MASE Distribution')
            axes[1, 0].set_xlabel('MASE')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # MAPE distribution
            axes[1, 1].hist(df['mape'], bins=15, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 1].axvline(df['mape'].mean(), color='red', linestyle='--',
                              label=f'Mean: {df["mape"].mean():.1f}%')
            axes[1, 1].axvline(10, color='green', linestyle='-', label='Excellent (10%)')
            axes[1, 1].axvline(20, color='orange', linestyle='-', label='Good (20%)')
            axes[1, 1].set_title('MAPE Distribution')
            axes[1, 1].set_xlabel('MAPE (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/quick_triple_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\nTriple Metrics Summary:")
        print(f"Mean MAPE: {df['mape'].mean():.2f}%")
        print(f"Mean MASE: {df['mase'].mean():.3f}")
        if has_seasonal:
            print(f"Mean Seasonal MASE: {df['mase_seas'].mean():.3f}")
        
        print(f"Areas with MAPE < 10%: {len(df[df['mape'] < 10])}/{len(df)}")
        print(f"Areas with MASE < 1.0: {len(df[df['mase'] < 1.0])}/{len(df)}")
        if has_seasonal:
            print(f"Areas with Seasonal MASE < 1.0: {len(df[df['mase_seas'] < 1.0])}/{len(df)}")

########### adding seasonal mase plot
    def create_mase_vs_id_visualization(self, save_dir='results_figures', dpi=300, figsize=(12, 8)):
        """
        Create MASE vs Area ID visualization with performance thresholds
        
        Parameters:
        - save_dir: Directory to save the figure
        - dpi: Figure resolution
        - figsize: Figure size (width, height)
        """
        if self.results_df.empty:
            print("No data to visualize")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        df = self.results_df
        
        # Color points based on MASE performance
        # MASE < 1.0 = beating naive baseline (green)
        # MASE 1.0-1.5 = reasonable performance (yellow)
        # MASE > 1.5 = poor performance (red)
        colors = []
        for mase in df['mase']:
            if mase < 1.0:
                colors.append('green')
            elif mase < 1.5:
                colors.append('orange')
            else:
                colors.append('red')
        
        # Create scatter plot
        scatter = ax.scatter(df['area_id'], df['mase'], 
                            c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # Add horizontal reference lines
        ax.axhline(y=1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2, 
                   label='MASE = 1.0 (Naive baseline)')
        ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='MASE = 1.5 (Performance threshold)')
        ax.axhline(y=df['mase'].mean(), color='purple', linestyle=':', 
                   alpha=0.7, linewidth=2, 
                   label=f'Mean MASE = {df["mase"].mean():.2f}')
        
        # Customize plot
        ax.set_xlabel('Area ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('MASE (Mean Absolute Scaled Error)', fontsize=12, fontweight='bold')
        ax.set_title('MASE Performance by Area ID', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add text annotations for extreme values
        max_mase_idx = df['mase'].idxmax()
        min_mase_idx = df['mase'].idxmin()
        
        # Annotate worst performer
        worst_area = df.loc[max_mase_idx, 'area_id']
        worst_mase = df.loc[max_mase_idx, 'mase']
        ax.annotate(f'Worst: Area {worst_area}\nMASE = {worst_mase:.2f}', 
                    xy=(worst_area, worst_mase), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Annotate best performer
        best_area = df.loc[min_mase_idx, 'area_id']
        best_mase = df.loc[min_mase_idx, 'mase']
        ax.annotate(f'Best: Area {best_area}\nMASE = {best_mase:.2f}', 
                    xy=(best_area, best_mase), 
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Add performance statistics text box
        stats_text = f"""Performance Summary:
    Areas with MASE < 1.0: {len(df[df['mase'] < 1.0])} ({len(df[df['mase'] < 1.0])/len(df)*100:.1f}%)
    Areas with MASE < 1.5: {len(df[df['mase'] < 1.5])} ({len(df[df['mase'] < 1.5])/len(df)*100:.1f}%)
    Mean MASE: {df['mase'].mean():.3f}
    Median MASE: {df['mase'].median():.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set y-axis to start from 0 for better visualization
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/mase_vs_area_id.png', dpi=dpi, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n=== MASE vs Area ID Analysis ===")
        print(f"Total areas analyzed: {len(df)}")
        print(f"Areas beating naive baseline (MASE < 1.0): {len(df[df['mase'] < 1.0])}")
        print(f"Areas with good performance (MASE < 1.5): {len(df[df['mase'] < 1.5])}")
        print(f"Mean MASE: {df['mase'].mean():.3f}")
        print(f"Median MASE: {df['mase'].median():.3f}")
        print(f"Best performing area: {best_area} (MASE = {best_mase:.3f})")
        print(f"Worst performing area: {worst_area} (MASE = {worst_mase:.3f})")
    
    def create_dual_mase_visualization(self, save_dir='results_figures', dpi=300, figsize=(15, 6)):
        """
        Create side-by-side MASE visualizations: regular MASE and seasonal MASE
        
        Parameters:
        - save_dir: Directory to save the figure
        - dpi: Figure resolution
        - figsize: Figure size (width, height)
        """
        if self.results_df.empty:
            print("No data to visualize")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('MASE Performance Analysis by Area ID', fontsize=16, fontweight='bold')
        
        df = self.results_df
        
        # Function to get colors based on MASE values
        def get_mase_colors(mase_values):
            colors = []
            for mase in mase_values:
                if mase < 1.0:
                    colors.append('green')
                elif mase < 1.5:
                    colors.append('orange')
                else:
                    colors.append('red')
            return colors
        
        # Plot 1: Regular MASE
        colors1 = get_mase_colors(df['mase'])
        ax1.scatter(df['area_id'], df['mase'], 
                   c=colors1, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        ax1.axhline(y=1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2, 
                   label='MASE = 1.0 (Naive baseline)')
        ax1.axhline(y=df['mase'].mean(), color='purple', linestyle=':', 
                   alpha=0.7, linewidth=2, 
                   label=f'Mean = {df["mase"].mean():.2f}')
        
        ax1.set_xlabel('Area ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MASE', fontsize=12, fontweight='bold')
        ax1.set_title('Regular MASE vs Area ID', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(bottom=0)
        
        # Plot 2: Seasonal MASE (if available)
        if 'mase_seas' in df.columns:
            colors2 = get_mase_colors(df['mase_seas'])
            ax2.scatter(df['area_id'], df['mase_seas'], 
                       c=colors2, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            ax2.axhline(y=1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2, 
                       label='MASE = 1.0 (Seasonal naive baseline)')
            ax2.axhline(y=df['mase_seas'].mean(), color='purple', linestyle=':', 
                       alpha=0.7, linewidth=2, 
                       label=f'Mean = {df["mase_seas"].mean():.2f}')
            
            ax2.set_xlabel('Area ID', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Seasonal MASE', fontsize=12, fontweight='bold')
            ax2.set_title('Seasonal MASE vs Area ID', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(bottom=0)
        else:
            ax2.text(0.5, 0.5, 'Seasonal MASE data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Seasonal MASE vs Area ID (No Data)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/dual_mase_vs_area_id.png', dpi=dpi, bbox_inches='tight')
        plt.show()
    
    # Standalone function for quick MASE analysis
    def quick_mase_analysis(csv_file='experiment_3_features.csv', save_dir='results_figures'):
        """
        Quick function to create MASE vs Area ID plot from CSV file
        
        Parameters:
        - csv_file: Path to the experiment results CSV
        - save_dir: Directory to save the figure
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        
        # Load data
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} areas from {csv_file}")
        except FileNotFoundError:
            print(f"File {csv_file} not found!")
            return
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color points based on MASE performance
        colors = []
        for mase in df['mase']:
            if mase < 1.0:
                colors.append('green')
            elif mase < 1.5:
                colors.append('orange')
            else:
                colors.append('red')
        
        # Create scatter plot
        ax.scatter(df['area_id'], df['mase'], 
                  c=colors, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # Add reference lines
        ax.axhline(y=1.0, color='blue', linestyle='-', alpha=0.7, linewidth=2, 
                   label='MASE = 1.0 (Naive baseline)')
        ax.axhline(y=df['mase'].mean(), color='purple', linestyle=':', 
                   alpha=0.7, linewidth=2, 
                   label=f'Mean MASE = {df["mase"].mean():.2f}')
        
        # Customize plot
        ax.set_xlabel('Area ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('MASE (Mean Absolute Scaled Error)', fontsize=12, fontweight='bold')
        ax.set_title('MASE Performance by Area ID', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/mase_vs_area_id_quick.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\nMASE Analysis Summary:")
        print(f"Mean MASE: {df['mase'].mean():.3f}")
        print(f"Areas beating naive baseline (MASE < 1.0): {len(df[df['mase'] < 1.0])}/{len(df)}")
        print(f"Best area: {df.loc[df['mase'].idxmin(), 'area_id']} (MASE = {df['mase'].min():.3f})")
        print(f"Worst area: {df.loc[df['mase'].idxmax(), 'area_id']} (MASE = {df['mase'].max():.3f})")
    # Usage examples:
    # 1. Add to your ResultsOrganizer class by copying the first function
    # 2. Call it like: organizer.create_mase_vs_id_visualization()
    # 3. Or use the quick function: quick_mase_analysis('experiment_3_features.csv')

# Quick usage functions
def analyze_all_results(log_filepath='experiment_log.csv', output_dir='final_results'):
    """
    One-click comprehensive analysis of all results
    
    Parameters:
    - log_filepath: Path to experiment log
    - output_dir: Directory for output files
    
    Returns:
    - ResultsOrganizer instance with all analysis completed
    """
    print("Starting comprehensive results analysis...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize organizer
    organizer = ResultsOrganizer(log_filepath)
    
    if organizer.results_df.empty:
        print("No results data found. Please run experiments first.")
        return None
    
    # Create all analyses
    print("\n1. Creating summary statistics...")
    summary = organizer.create_summary_statistics()
    
    print("\n2. Creating performance ranking table...")
    ranking_table = organizer.create_performance_ranking_table()
    
    print("\n3. Generating comprehensive visualizations...")
    organizer.create_comprehensive_visualizations(f'{output_dir}/figures')
    
    print("\n4. Creating executive summary report...")
    organizer.create_executive_summary_report(f'{output_dir}/executive_summary.txt')
    
    # Save ranking table
    if ranking_table is not None:
        ranking_table.to_csv(f'{output_dir}/performance_ranking.csv', index=False)
        print(f"\n5. Performance ranking saved to {output_dir}/performance_ranking.csv")
        
        # Show top 10 performers
        print("\nTOP 10 PERFORMING AREAS:")
        print(ranking_table.head(10)[['Area', 'MAPE (%)', 'MASE', 'Performance', 'Data Quality (%)']].to_string(index=False))
    
    print(f"\nAll results organized in: {output_dir}/")
    
    return organizer
def quick_results_summary(log_filepath='experiment_log.csv'):
    """Quick summary of results without generating all files"""
    organizer = ResultsOrganizer(log_filepath)
    
    if organizer.results_df.empty:
        print("No results found.")
        return
    
    summary = organizer.create_summary_statistics()
    ranking = organizer.create_performance_ranking_table()
    
    print(f"\n{'='*60}")
    print("QUICK RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Areas analyzed: {summary['total_areas']}")
    print(f"Mean MAPE: {summary['mean_mape']:.1f}%")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"\nTop 5 performers:")
    print(ranking.head(5)[['Area', 'MAPE (%)', 'Performance']].to_string(index=False))
    
    return organizer
def compare_two_runs_statistically():
    # Compare the two runs statistically
    import pandas as pd
    from scipy import stats
    
    base = 0 
    second = 7
    run3_data = pd.read_csv(f'experiment_{base}_features.csv')
    run_sensitivity_data = pd.read_csv(f'experiment_{second}_features.csv')
    
    # Paired t-test on MAPE
    baseline_mape = run3_data['mape']
    sensitivity_mape = run_sensitivity_data['mape']
    
    t_stat, p_value = stats.ttest_rel(baseline_mape, sensitivity_mape)
    improvement = (sensitivity_mape.mean() - baseline_mape.mean()) / baseline_mape.mean() * 100
    
    print(f"{base} vs {second} features: {improvement:+.1f}% change in MAPE (p={p_value:.4f})")
def calculate_robust_metrics(results_df, metric='mape'):
    """Calculate robust metrics from results DataFrame"""    
    # Remove any NaN values
    values = results_df[metric].dropna()
    
    metrics = {
        'median': np.median(values),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
        'p10': np.percentile(values, 10),  # Best 10%
        'p90': np.percentile(values, 90),  # Worst 10%
        'mad': np.median(np.abs(values - np.median(values))),  # Median Absolute Deviation
        'mean': np.mean(values),  # For comparison
        'std': np.std(values)     # For comparison
    }
    return metrics
def calculate_success_metrics(results_df, metric='mape', thresholds=[5, 10, 15, 20]):
    """Calculate success rate metrics"""
    
    values = results_df[metric].dropna()
    n_areas = len(values)
    
    metrics = {}
    for threshold in thresholds:
        success_rate = (values < threshold).sum() / n_areas * 100
        metrics[f'success_rate_{threshold}'] = success_rate
    
    # Add summary success metrics
    metrics['excellent_rate'] = (values < 10).sum() / n_areas * 100
    metrics['acceptable_rate'] = (values < 15).sum() / n_areas * 100
    metrics['poor_rate'] = (values > 20).sum() / n_areas * 100
    
    return metrics
def summarize_experiment(results_df, run_name=None):
    """Generate comprehensive summary statistics for experiment"""
        
    print("="*70)
    print(f"EXPERIMENT SUMMARY{f' - {run_name}' if run_name else ''}")
    print("="*70)
    print(f"Total areas analyzed: {len(results_df)}")
    
    # Walk-forward metrics (always available)
    wf_mape = results_df['mape'].dropna()
    wf_mase = results_df['mase'].dropna()
    
    print("\n WALK-FORWARD METRICS")
    print("-"*40)
    print(f"MAPE - Median: {np.median(wf_mape):.2f}%")
    print(f"      IQR: {np.percentile(wf_mape, 75) - np.percentile(wf_mape, 25):.2f}")
    print(f"      Range: [{wf_mape.min():.2f}, {wf_mape.max():.2f}]")
    print(f"MASE - Median: {np.median(wf_mase):.2f}")
    
    # Performance distribution
    print("\n PERFORMANCE DISTRIBUTION (MAPE)")
    print("-"*40)
    print(f"Excellent (P10): {np.percentile(wf_mape, 10):.2f}%")
    print(f"Good (P25): {np.percentile(wf_mape, 25):.2f}%")
    print(f"Median (P50): {np.median(wf_mape):.2f}%")
    print(f"Poor (P75): {np.percentile(wf_mape, 75):.2f}%")
    print(f"Worst (P90): {np.percentile(wf_mape, 90):.2f}%")
    
    # Success rates
    print("\n SUCCESS RATES")
    print("-"*40)
    for threshold in [10, 15, 20]:
        rate = (wf_mape < threshold).sum() / len(wf_mape) * 100
        print(f"MAPE < {threshold}%: {rate:.1f}% of areas ({(wf_mape < threshold).sum()}/{len(wf_mape)})")
    
    # Best and worst areas
    sorted_results = results_df.sort_values('mape')
    
    print("\n TOP 5 AREAS")
    print("-"*40)
    for _, row in sorted_results.head(5).iterrows():
        print(f"Area {row['area_id']:>3}: {row['mape']:>6.2f}% (outliers: {row['outliers_found']:>2})")
    
    print("\n  BOTTOM 5 AREAS")
    print("-"*40)
    for _, row in sorted_results.tail(5).iterrows():
        print(f"Area {row['area_id']:>3}: {row['mape']:>6.2f}% (outliers: {row['outliers_found']:>2})")
    
    # Parameter analysis
    print("\n PARAMETER INSIGHTS")
    print("-"*40)
    # Group by seasonality mode
    mode_stats = results_df.groupby('seasonality_mode')['mape'].agg(['mean', 'median', 'count'])
    print("Seasonality Mode Performance:")
    for mode, stats in mode_stats.iterrows():
        print(f"  {mode}: median={stats['median']:.2f}%, n={stats['count']}")
    
    # Outlier impact
    print("\n OUTLIER ANALYSIS")
    print("-"*40)
    print(f"Areas with outliers: {(results_df['outliers_found'] > 0).sum()}")
    print(f"Avg outliers when present: {results_df[results_df['outliers_found'] > 0]['outliers_found'].mean():.1f}")
    
    # Composite score
    composite_score = calculate_composite_score(results_df)
    print(f"\n COMPOSITE SCORE: {composite_score:.2f} (lower is better)")
    
    return {
        'median_mape': np.median(wf_mape),
        'iqr': np.percentile(wf_mape, 75) - np.percentile(wf_mape, 25),
        'success_rate_15': (wf_mape < 15).sum() / len(wf_mape) * 100,
        'composite_score': composite_score,
        'n_areas': len(results_df)
    }
def calculate_composite_score(results_df):
    """Calculate a composite score combining multiple factors"""
    
    wf_mape = results_df['mape'].dropna()
    
    # Components of composite score
    median_component = np.median(wf_mape) * 0.4
    consistency_component = (np.percentile(wf_mape, 75) - np.percentile(wf_mape, 25)) * 0.3
    failure_component = ((wf_mape > 20).sum() / len(wf_mape) * 100) * 0.3
    
    return median_component + consistency_component + failure_component
def compare_runs(df1, df2, run1_name='Run 1', run2_name='Run 2'):
    """Compare two experiment runs"""
    
    print("="*70)
    print(f"EXPERIMENT COMPARISON: {run1_name} vs {run2_name}")
    print("="*70)
    
    metrics_to_compare = [
        ('mape', 'MAPE', 'lower'),
        ('mase', 'MASE', 'lower'),
        ('rmse', 'RMSE', 'lower')
    ]
    
    print(f"\n{'Metric':<20} {'Run 1':>12} {'Run 2':>12} {'Better':>10} {'Diff':>10}")
    print("-"*70)
    
    for metric, label, better_direction in metrics_to_compare:
        val1 = df1[metric].median()
        val2 = df2[metric].median()
        
        if better_direction == 'lower':
            better = run1_name if val1 < val2 else run2_name
            diff = ((val2 - val1) / val1 * 100)
        else:
            better = run1_name if val1 > val2 else run2_name
            diff = ((val1 - val2) / val2 * 100)
            
        print(f"{label+' (median)':<20} {val1:>12.2f} {val2:>12.2f} {better:>10} {diff:>9.1f}%")
    
    # Success rates comparison
    print(f"\n{'Success Rates':<20} {'Run 1':>12} {'Run 2':>12} {'Better':>10}")
    print("-"*70)
    
    for threshold in [10, 15, 20]:
        rate1 = (df1['mape'] < threshold).sum() / len(df1) * 100
        rate2 = (df2['mape'] < threshold).sum() / len(df2) * 100
        better = run1_name if rate1 > rate2 else run2_name
        print(f"MAPE < {threshold}%{'':<12} {rate1:>11.1f}% {rate2:>11.1f}% {better:>10}")
    
    # Consistency comparison
    print(f"\n{'Consistency':<20} {'Run 1':>12} {'Run 2':>12} {'Better':>10}")
    print("-"*70)
    
    iqr1 = np.percentile(df1['mape'], 75) - np.percentile(df1['mape'], 25)
    iqr2 = np.percentile(df2['mape'], 75) - np.percentile(df2['mape'], 25)
    better = run1_name if iqr1 < iqr2 else run2_name
    print(f"IQR (MAPE){'':<10} {iqr1:>12.2f} {iqr2:>12.2f} {better:>10}")
    
    # Overall winner
    score1 = calculate_composite_score(df1)
    score2 = calculate_composite_score(df2)
    
    print(f"\n{'OVERALL':<20} {'Run 1':>12} {'Run 2':>12} {'Winner':>10}")
    print("-"*70)
    print(f"Composite Score{'':<5} {score1:>12.2f} {score2:>12.2f} {run1_name if score1 < score2 else run2_name:>10}")
def analyze_parameter_impact(results_df):
    """Analyze how different parameters affect performance"""
    
    print("="*70)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*70)
    
    # Seasonality mode impact
    print("\n Seasonality Mode Impact")
    print("-"*40)
    mode_analysis = results_df.groupby('seasonality_mode').agg({
        'mape': ['median', 'mean', 'std', 'count'],
        'mase': 'median'
    }).round(2)
    print(mode_analysis)
    
    # Fourier order distribution
    print("\n Fourier Order Distribution")
    print("-"*40)
    fourier_analysis = results_df.groupby('fourier_order')['mape'].agg(['median', 'count']).round(2)
    print(fourier_analysis.sort_values('median'))
    
    # Outlier detection method impact
    print("\n Outlier Detection Method Impact")
    print("-"*40)
    outlier_analysis = results_df.groupby('outlier_detection').agg({
        'mape': ['median', 'count'],
        'outliers_found': 'mean'
    }).round(2)
    print(outlier_analysis)
    
    # Correlation analysis
    print("\n Parameter Correlations with MAPE")
    print("-"*40)
    numeric_params = ['changepoint_prior_scale', 'seasonality_prior_scale', 
                      'fourier_order', 'outliers_found']
    
    for param in numeric_params:
        if param in results_df.columns:
            corr = results_df[param].corr(results_df['mape'])
            print(f"{param}: {corr:.3f}")
# Quick usage function
def quick_summary(csv_path):
    """Load results and provide quick summary"""
    df = pd.read_csv(csv_path)
    return summarize_experiment(df, run_name=csv_path.split('/')[-1].replace('.csv', ''))

# Example usage:
# df = pd.read_csv('../results/run_1.csv')
# summarize_experiment(df, 'Baseline Run')
# 
# # Compare two runs
# df1 = pd.read_csv('../results/run_1.csv')
# df2 = pd.read_csv('../results/run_2.csv')
# compare_runs(df1, df2, 'Baseline', 'With Lags')

#filename17 = '../results/exp17_auto_regs_300_50.csv'
#df1 = pd.read_csv(filename17)

#filename17_2 = '../results/exp18_base_no_regs_300_50.csv'
#df2 = pd.read_csv(filename17_2)

#summary = quick_results_summary(filename17)
#print (summary)
#summary = quick_results_summary(filename17_2)
#print (summary)

filename8 = 'D:/WaterDemandForecastingTemp/results/exp8_auto_reg_2_10.csv'
df1 = pd.read_csv(filename8)
x = calculate_robust_metrics(df1)

#summary = summarize_experiment(df1, 'Baseline Run')
#print (summary)

compare_runs(df1, df2)
x = calculate_robust_metrics(df1)
print(x)
x = calculate_robust_metrics(df2)
print(x)

'''
x = calculate_robust_metrics(df)
y = calculate_success_metrics(df)
z = summarize_experiment(df)
w = analyze_parameter_impact(df)
quick_summary(filename)
'''
#compare_two_runs_statistically()       
#quick_results_summary('run_6_minimal_features.csv')

#analyze_all_results(filename8)
#analyze_all_results(filename2)
#compare_runs(exp6, exp8, run1_name='exp6', run2_name='exp8')

#analyze_all_results('experiment_3_features.csv')
#compare_two_runs_statistically()
# Usage examples:

# Method 1: Add to your ResultsOrganizer class
#organizer = ResultsOrganizer('experiment_7_features.csv')
#organizer.create_triple_metrics_visualization()
#organizer.create_metrics_correlation_plot()
#organizer.create_performance_ranking_by_metrics()

# Method 2: Quick standalone analysis
#quick_triple_metrics_analysis('experiment_3_features.csv')

# Method 3: Integrate into comprehensive analysis
# Add this line to your analyze_all_results function:
# organizer.create_triple_metrics_visualization(f'{output_dir}/figures')



