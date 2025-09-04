import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from pathlib import Path

def generate_performance_plot(df_path, run_name, output_dir='../results/', save_plot_mode='low'):
    """
    Generate  performance analysis plot
    
    Parameters:
    -----------
    df_path : str
        Path to the CSV file containing results
    run_name : str
        Name of the run (used for saving the figure)
    output_dir : str
        Directory to save the figure
    """
    # Load data
    df = pd.read_csv(df_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure with GridSpec for complex layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define colors and styles
    primary_color = '#2E7D32'  # Dark green
    secondary_color = '#1565C0'  # Dark blue
    excellent_color = '#1B5E20'  # Darker green (was #4CAF50)
    verygood_color = '#A5D6A7'  # Lighter green (was #81C784)
    good_color = '#FFC107'  # Amber
    poor_color = '#F44336'  # Red
    
    # Calculate statistics
    mean_mase = df['mase'].mean()
    mean_mase_seas = df['mase_seas'].mean()
    mean_mape = df['mape'].mean()
    median_mape = df['mape'].median()
    
    # 1. MASE by Area ID (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Sort by MASE for better visualization
    df_sorted = df.sort_values('mase')
    
    # Create scatter plot
    scatter1 = ax1.scatter(df_sorted['area_id'], df_sorted['mase'], 
                          c=primary_color, s=60, alpha=0.7, edgecolors='darkgreen')
    
    # Add horizontal lines for reference
    ax1.axhline(y=mean_mase, color='blue', linestyle='--', alpha=0.5, 
                label=f'Mean = {mean_mase:.2f}')
    ax1.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, 
                label='MASE = 1.0 (Naive baseline)')
    
    ax1.set_xlabel('Area ID', fontsize=11)
    ax1.set_ylabel('MASE', fontsize=11)
    ax1.set_title('MASE by Area ID', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(0, max(df['mase'].max() * 1.1, 2.0))
    
    # 2. Seasonal MASE by Area ID (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    
    scatter2 = ax2.scatter(df_sorted['area_id'], df_sorted['mase_seas'], 
                          c=primary_color, s=60, alpha=0.7, edgecolors='darkgreen')
    
    # Highlight outliers
    outlier_threshold = df['mase_seas'].mean() + 2 * df['mase_seas'].std()
    outliers = df[df['mase_seas'] > outlier_threshold]
    if len(outliers) > 0:
        ax2.scatter(outliers['area_id'], outliers['mase_seas'], 
                   c=poor_color, s=100, alpha=0.8, edgecolors='darkred', 
                   marker='o', label='Outliers')
    
    ax2.axhline(y=mean_mase_seas, color='blue', linestyle='--', alpha=0.5,
                label=f'Mean = {mean_mase_seas:.2f}')
    ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.5,
                label='Seasonal MASE = 1.0')
    
    ax2.set_xlabel('Area ID', fontsize=11)
    ax2.set_ylabel('Seasonal MASE', fontsize=11)
    ax2.set_title('Seasonal MASE by Area ID', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    
    # 3. MAPE by Area ID (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Color code by performance
    colors = []
    for mape in df_sorted['mape']:
        if mape < 5:
            colors.append(excellent_color)
        elif mape < 10:
            colors.append(verygood_color)
        elif mape < 20:
            colors.append(good_color)
        else:
            colors.append(poor_color)
    
    scatter3 = ax3.scatter(df_sorted['area_id'], df_sorted['mape'], 
                          c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add threshold lines
    ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5, 
                label='Good threshold (10%)')
    ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, 
                label='Acceptable threshold (20%)')
    ax3.axhline(y=mean_mape, color='blue', linestyle=':', alpha=0.5,
                label=f'Mean = {mean_mape:.1f}%')
    
    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=excellent_color, label='Excellent (<5%)'),
        Patch(facecolor=verygood_color, label='Very Good (5-10%)'),
        Patch(facecolor=good_color, label='Good (10-20%)'),
        Patch(facecolor=poor_color, label='Poor (>20%)')
    ]
    ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax3.set_xlabel('Area ID', fontsize=11)
    ax3.set_ylabel('MAPE (%)', fontsize=11)
    ax3.set_title('MAPE by Area ID', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. MASE Distribution (Bottom Left)
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Create histogram
    n, bins, patches = ax4.hist(df['mase'], bins=12, color='skyblue', 
                                alpha=0.7, edgecolor='navy')
    
    # Add vertical lines for statistics
    ax4.axvline(x=mean_mase, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_mase:.2f}')
    ax4.axvline(x=df['mase'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df["mase"].median():.2f}')
    ax4.axvline(x=1.0, color='black', linestyle=':', linewidth=2,
                label='MASE = 1.0 (Baseline)')
    
    ax4.set_xlabel('MASE', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('MASE Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Seasonal MASE Distribution (Bottom Middle)
    ax5 = fig.add_subplot(gs[1, 1])
    
    n, bins, patches = ax5.hist(df['mase_seas'], bins=12, color='lightcoral', 
                                alpha=0.7, edgecolor='darkred')
    
    ax5.axvline(x=mean_mase_seas, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_mase_seas:.2f}')
    ax5.axvline(x=df['mase_seas'].median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {df["mase_seas"].median():.2f}')
    ax5.axvline(x=1.0, color='black', linestyle=':', linewidth=2,
                label='Seasonal MASE = 1.0')
    
    ax5.set_xlabel('Seasonal MASE', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Seasonal MASE Distribution', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. MAPE Distribution (Bottom Right)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Create histogram with custom bins
    bins = np.arange(0, df['mape'].max() + 5, 2.5)
    n, bins, patches = ax6.hist(df['mape'], bins=bins, alpha=0.7, edgecolor='black')
    
    # Color bars based on performance categories
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 5:
            patch.set_facecolor(excellent_color)
        elif bin_center < 10:
            patch.set_facecolor(verygood_color)
        elif bin_center < 20:
            patch.set_facecolor(good_color)
        else:
            patch.set_facecolor(poor_color)
    
    # Add vertical lines
    ax6.axvline(x=mean_mape, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_mape:.1f}%')
    ax6.axvline(x=median_mape, color='green', linestyle=':', linewidth=2,
                label=f'Median: {median_mape:.1f}%')
    
    # Add text box with key metrics
    textstr = f'Excellent (<5%): {len(df[df["mape"] < 5])}\n' \
              f'Very Good (5-10%): {len(df[(df["mape"] >= 5) & (df["mape"] < 10)])}\n' \
              f'Good (10-20%): {len(df[(df["mape"] >= 10) & (df["mape"] < 20)])}\n' \
              f'Poor (>20%): {len(df[df["mape"] >= 20])}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax6.text(0.56, 0.82, textstr, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax6.set_xlabel('MAPE (%)', fontsize=11)
    ax6.set_ylabel('Density', fontsize=11)
    ax6.set_title('MAPE Distribution', fontsize=12, fontweight='bold')
    ax6.legend(loc = 'upper right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    
    if save_plot_mode in ['low', 'high']:
        output_dir = Path(f"../results/{run_name}_figures")
        output_dir.mkdir(parents=True, exist_ok=True)  # 
        
        if save_plot_mode == 'high':
            output_path = output_dir / "comprehensive_performance.png"  # Construct full path
            plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {output_path}")
        
            output_path = output_dir / "comprehensive_performance.tiff"  
            plt.savefig(output_path, format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
            print(f"TIFF saved to: {output_path}")
      
            output_path = output_dir / "comprehensive_performance.pdf"  
            plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
            print(f"PDF saved to: {output_path}")
        else:
            output_path = output_dir / "comprehensive_performance.png"  # Construct full path
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {output_path}")
        
        plt.close()
    
    # Generate summary statistics
    print("\nPERFORMANCE SUMMARY:")
    print("="*50)
    print(f"Total Areas: {len(df)}")
    print(f"Mean MAPE: {mean_mape:.2f}% (Median: {median_mape:.2f}%)")
    print(f"Mean MASE: {mean_mase:.3f}")
    print(f"Mean Seasonal MASE: {mean_mase_seas:.3f}")
    print(f"Excellent (<5% MAPE): {len(df[df['mape'] < 5])} areas")
    print(f"Very Good (5-10% MAPE): {len(df[(df['mape'] >= 5) & (df['mape'] < 10)])} areas")
    print(f"Good (10-20% MAPE): {len(df[(df['mape'] >= 10) & (df['mape'] < 20)])} areas")
    print(f"Poor (>20% MAPE): {len(df[df['mape'] >= 20])} areas")
    
    return fig

if __name__ == "__main__":
    generate_performance_plot(
        df_path='../results/exp2_adaptive_reg.csv',
        run_name='exp2_adaptive_reg',
        output_dir='../results/'
    )

