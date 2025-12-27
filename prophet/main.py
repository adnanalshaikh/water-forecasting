"""
Reproduce all results from: Automated Water Demand Forecasting for 
National-Scale Deployment: A Prophet-Based Framework for Palestinian 
Municipal Water Management

Step-by-step reproduction of figures, tables, and analyses.

QUICK START (using pre-run results):
    python main.py --skip-forecasting

FULL REPRODUCTION (regenerate all forecasts):
    python main.py
    
Estimated runtime:
    - Analysis only (skip_forecasting=True): ~5-10 minutes
    - Full reproduction (skip_forecasting=False): ~X hours
"""

from config import config_baseline
from prophet_forecasting import run_forecasting
from performance_comparison import generate_performance_comparison
from exploratory_analysis import generate_eda_figures
from adaptive_results_summary import generate_performance_summary
from top_regressors_analysis import generate_regressor_selection_analysis
from difficulty_analysis import generate_difficulty_analysis
from pathlib import Path
import shutil


def setup_prerun_results():
    """
    Copy pre-run results to expected locations for analysis.
    
    Maps:
    - prerun_results/exp_0.csv → results/run_0.csv
    - prerun_results/exp_1.csv → results/run_1.csv
    - ...
    - prerun_results/exp_adaptive.csv → results/run_adaptive.csv
    """
    prerun_dir = Path('../prerun_results')
    results_dir = Path('../results')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Mapping from pre-run names to expected names
    file_mapping = {
        'exp_0.csv': 'run_0.csv',
        'exp_1.csv': 'run_1.csv',
        'exp_2.csv': 'run_2.csv',
        'exp_3.csv': 'run_3.csv',
        'exp_4.csv': 'run_4.csv',
        'exp_5.csv': 'run_5.csv',
        'exp_6.csv': 'run_6.csv',
        'exp_7.csv': 'run_7.csv',
        'exp_adaptive.csv': 'run_adaptive.csv',
    }
    
    print("\n" + "="*70)
    print("USING PRE-RUN RESULTS (skipping forecasting)")
    print("="*70)
    print(f"  Source: {prerun_dir}")
    print(f"  Destination: {results_dir}")
    
    missing_files = []
    copied_files = []
    
    for prerun_name, run_name in file_mapping.items():
        src = prerun_dir / prerun_name
        dst = results_dir / run_name
        
        if src.exists():
            shutil.copy2(src, dst)
            copied_files.append(prerun_name)
            print(f"  ✓ {prerun_name} → {run_name}")
        else:
            missing_files.append(prerun_name)
            print(f"  ✗ {prerun_name} (missing)")
    
    if missing_files:
        print(f"\n  Warning: {len(missing_files)} pre-run files missing:")
        for f in missing_files:
            print(f"    - {f}")
        print(f"  Please ensure all files are in {prerun_dir}/")
        raise FileNotFoundError(f"Missing pre-run result files: {missing_files}")
    
    print(f"\n Copied {len(copied_files)} pre-run result files")
    return copied_files


def main(skip_forecasting=False, skip_eda=False):
    """
    Reproduce all results from the paper.
    
    Args:
        skip_forecasting: If True, use pre-run results from prerun_results/ folder
                         If False, regenerate all forecasts (takes hours)
        skip_eda: If True, skip exploratory data analysis
    
    Execute steps:
    0. Exploratory data analysis → Figures X, Y, Z (data characteristics)
    1. Adaptive forecasts → Figures 9, 10, Supplementary
    2. Fixed-k forecasts → Performance comparison data
    3. Performance comparison → Figure 6 (MAPE/MASE boxplots)
    4. Adaptive results summary → Figure 8 (29 areas performance)
    5. Regressor selection analysis → Figure 7 (k-distribution & regressor frequency)
    6. Difficulty analysis → Figure 5 (Difficulty Score vs MAPE)
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*70)
    print("REPRODUCING ALL RESULTS")
    if skip_forecasting:
        print("MODE: Analysis only (using pre-run results)")
    else:
        print("MODE: Full reproduction (regenerating all forecasts)")
    print("="*70)
    
    # ========================================================================
    # FORECASTING STEPS (Steps 1-2) - Can be skipped
    # ========================================================================
    
    # Step 0: Exploratory Data Analysis (optional)
    if not skip_eda:
        generate_eda_figures()

            
    if skip_forecasting:
        # Use pre-run results instead of regenerating
        setup_prerun_results()
    else:
        # Generate all forecasts from scratch
        # Step 1: Generate main forecast figures
        generate_adaptive_forecasts()
        
        # Step 2: Generate comparison data
        generate_fixed_k_forecasts()
    
    # ========================================================================
    # ANALYSIS STEPS (Steps 3-6) - Always run
    # ========================================================================
    
    # Step 3: Generate performance comparison figure
    print("\n" + "="*70)
    print("STEP 3: Generating performance comparison")
    print("        (Figure 6: MAPE/MASE boxplots)")
    print("="*70)
    generate_performance_comparison()
    print("Step 3 complete: Figure 6 generated")
    
    # Step 4: Generate adaptive results summary (Figure 8)
    print("\n" + "="*70)
    print("STEP 4: Generating adaptive results summary")
    print("        (Figure 8: Performance across 29 areas)")
    print("="*70)
    generate_performance_summary(run_name='run_adaptive', 
                                 results_dir='../results/', 
                                 output_dir='../figures/')
    print("Step 4 complete: Figure 8 generated")
    
    # Step 5: Analyze regressor selection patterns (Figure 7)
    print("\n" + "="*70)
    print("STEP 5: Analyzing regressor selection patterns")
    print("        (Figure 7: k-distribution and regressor frequency)")
    print("="*70)
    generate_regressor_selection_analysis(run_name='run_adaptive',
                                          results_dir='../results/',
                                          output_dir='../figures/',
                                          plot_res='low')
    print("Step 5 complete: Figure 7 generated")
    
    # Step 6: Difficulty analysis (Figure 5)
    print("\n" + "="*70)
    print("STEP 6: Analyzing forecasting difficulty")
    print("        (Figure 5: Difficulty Score vs MAPE correlation)")
    print("="*70)
    difficulty_results = generate_difficulty_analysis(run_name='run_adaptive',
                                                      results_dir='../results/',
                                                      output_dir='../figures/')
    print("Step 6 complete: Figure 5 generated")
    print(f"   Correlation: r = {difficulty_results['validation_results'].get('correlation', 'N/A')}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    runtime = time.time() - start_time
    print("\n" + "="*70)
    print(f"ALL STEPS COMPLETE")
    print(f"  Total runtime: {runtime/60:.1f} minutes")
    print(f"  Results saved in: ./results/")
    print(f"  Figures saved in: ./figures/")
    print("="*70)
    print("\nGenerated figures:")
    
    if not skip_eda:
        print("  - Figure 2: Variance and Additive decomposition")
        print("  - Figure 3: Distribution of seasonal amplitude, strength, and coefficient of variation")
        print("  - Figure 4: Stationarity and autocorrelation")
    else:
        print("  - run without --skip-eda to generate figures 2, 3, and 4" )
    print("  - Figure 5: Forecasting difficulty vs MAPE")
    print("  - Figure 6: Performance comparison across configurations")
    print("  - Figure 7: Regressor selection patterns")
    print("  - Figure 8: Adaptive performance across 29 areas")
    if not skip_forecasting:
        print("  - Figure 9, 10: Example forecasts")
        print("  - Supplementary: Additional forecast examples")
    else:
        print("  - Figure 9, 10: (use forecast figures from prerun_results/)")
    print("  - Table 2: printed in ../results/runs_statistics.txt")
    print("  - Table 3 and 4: printed on the screen")

def generate_adaptive_forecasts():
    """
    Generate forecasts using adaptive regressor selection.
    
    Produces:
    - Figure 9: Forecast example for area [X] with adaptive regressors
    - Figure 10: Forecast example for area [Y] with adaptive regressors
    - Supplementary Figures S1-S27: Forecasts for remaining 27 areas
    
    All 29 areas use adaptive selection (k=1 to 7 regressors optimized per area).
    Output directory: results/run_adaptive_figures/
    Output file: results/run_adaptive.csv
    """
    print("\n" + "="*70)
    print("STEP 1: Generating adaptive regressor forecasts")
    print("        (Figures 9, 10, and Supplementary Figures S1-S27)")
    print("="*70)
    
    config = config_baseline.copy()
    config.update({
        'area_id': (1, 29),
        'run_dir': 'run_adaptive',
        'add_regressors': True,
        'optimize_regressor_selection': True,
        'min_regressors': 1,
        'max_regressors': 7,
    })
    
    run_forecasting(config)
    print("\n✓ Step 1 complete: Adaptive forecasts generated")


def generate_fixed_k_forecasts():
    """
    Generate forecasts for fixed regressor counts (k=0 to k=6).
    
    Produces:
    - Baseline comparison: k=0 shows poor performance (demonstrates need for regressors)
    - Fixed configurations: k=1,2,...,6 for performance analysis
    
    Output directories: results/run_0/ through results/run_6/
    Output files: results/run_0.csv through results/run_6.csv
    """
    print("\n" + "="*70)
    print("STEP 2: Generating fixed-k forecasts for comparison")
    print("        (k=0 baseline vs k=1-6 with regressors)")
    print("="*70)
    
    for k in range(7):
        config = config_baseline.copy()
        config.update({
            'area_id': (1, 29),
            'run_dir': f'run_{k}',
            'add_regressors': (k > 0),
            'optimize_regressor_selection': (k > 0),
            'min_regressors': k if k > 0 else None,
            'max_regressors': k if k > 0 else None,
        })
        
        baseline_note = " (baseline - expected poor performance)" if k == 0 else ""
        print(f"\n  [{k+1}/7] Running k={k}{baseline_note}...")
        run_forecasting(config)
    
    print("\n✓ Step 2 complete: Fixed-k forecasts generated")


if __name__ == '__main__':    
    import argparse

    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--skip-forecasting', action='store_true',
                   help='Use pre-run results instead of regenerating forecasts (fast)')
    parser.add_argument('--skip-eda', action='store_true',
                   help='Skip exploratory data analysis')

    args = parser.parse_args()

    main(skip_forecasting=args.skip_forecasting, skip_eda=args.skip_eda)
    #main(skip_forecasting=True)

