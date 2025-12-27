# Automated Water Demand Forecasting for National-Scale Deployment: A Prophet-Based Framework for Palestinian Municipal Water Management  

> Scalable, automated water demand forecasting for national-scale deployment using Prophet and Optuna.  

This repository contains the code accompanying the paper:  
*"Automated Water Demand Forecasting for National-Scale Deployment: A Prophet-Based Framework for Palestinian Municipal Water Management"*  
by **Adnan Salman** and **Yamama Shaka’a**.  

👉 [GitHub Repository](https://github.com/adnanalshaikh/water-forecasting)  

---

## Overview  
This repository implements the methodology described in the paper to demonstrate the feasibility and results of the proposed automated forecasting system based on the Prophet algorithm. The system doesn’t require manual parameter tuning or feature engineering. Instead, parameters are tuned for each service area based on its specific characteristics, and the most influential external factors are automatically selected from a comprehensive set of potential factors for each area.  

---

## Methodology  
We developed an automated forecasting framework to predict monthly water consumption across 29 heterogeneous urban areas in Nablus (2019–2023) using the Prophet algorithm with automatic hyperparameter optimization via Optuna. The pipeline begins with data preprocessing (imputation of missing values, outlier detection, validation) and exploratory analysis (time-series decomposition, seasonality, predictability, and difficulty classification). Area-specific external regressors, including weather variables, holiday indicators, seasonal Fourier terms, and structural time components, are ranked and selected using ensemble importance scoring. For each area, Prophet models are tuned with Bayesian optimization to capture trend, seasonality, and uncertainty while ensuring scalability and adaptability to diverse demand patterns. Models are trained on 48 months of data and tested on the final 12 months, with walk-forward validation used during tuning. Performance is evaluated using MAE, RMSE, MAPE, and MASE against naive and seasonal baselines, enabling robust, area-specific forecasting for real-world water management.  

---

## Architecture  
![Automated Water Demand Forecasting Workflow](https://github.com/adnanalshaikh/water-forecasting/blob/main/doc/archit-prophet1.png)  
**Figure:** Automated water demand forecasting workflow. Raw consumption, weather, and holiday data are preprocessed, analyzed, and transformed into area-specific Prophet models with external regressors and hyperparameter optimization. The trained models are evaluated against multiple error metrics to ensure robust, scalable forecasting across heterogeneous urban areas.  

---

## Key Constraints  
A key challenge in this study is the short and heterogeneous time series (60 months per area, with highly diverse patterns across residential, industrial, refugee camps, and commercial zones). This constraint requires area-specific regressors and adaptive hyperparameter tuning to achieve reliable forecasts, since a one-size-fits-all model would fail to capture these variations.  

---

## Significance  
This framework delivers **scalable water demand forecasting** for Palestine’s 550+ service areas, validated in Nablus with **93%+ accuracy (MAPE < 10%)**. It adapts to diverse patterns across residential, industrial, and refugee zones, helping utilities move from **reactive crisis response to proactive planning**. Core innovations include **automated feature selection** and a **forecasting difficulty score** that optimize resources for national-scale deployment.  

---

## Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/adnanalshaikh/water-forecasting.git
cd water-forecasting
```
### **2. Install the required dependency**
```bash
pip install -r requirements.txt
```
---

## Quick Start

### Option 1: Fast Analysis (few minutes, using pre-run results)
```bash
# Generate all figures using pre-computed forecasts
python main.py --skip-forecasting
```

### Option 2: Full Reproduction (About an hours, regenerate everything)
```bash
# Regenerate all forecasts and analyses from scratch
python main.py
```
---

## Pre-run Results

The `prerun_results/` folder contains forecasting results used in the published paper:
- `exp_0.csv` - Baseline Prophet (no regressors)
- `exp_1.csv` through `exp_7.csv` - Fixed k regressors
- `exp_adaptive.csv` - Adaptive regressor selection (main results)

These allow you to reproduce all figures and analyses without re-running the time-consuming forecasting steps.

## Repository Structure
```bash
water-forecasting/
├── prophet/                    
│   ├── main.py                     # Main script
|   |── data_loader.py              # Load and impute area time-series 
│   ├── exploratory_analysis.py     # Figure 2, 3, 4: Generates EDA figures (variance, seasonality, autocorrelation) 
│   ├── config.py                   # Configuration dictionary for model and execution parameters 
│   ├── prophet_forecasting.py      # Figure 9, 10: Main forecasting module (outputs results .csv + figures) 
│   ├── performance_comparison.py   # Figure 6: Produces comparison figure + statistics table
│   ├── top_regressors_analysis.py  # Figure 7: Generates regressor ranking plots 
│   ├── difficulty_analysis.py      # Figure 5: Produces difficulty-weight fitting figure  
│   ├── regressors.py               # Regressor selection code 
│   ├── timing.py                   # Produces execution time statistics 
│   ├── adaptive_results_summary.py # Figure 8: performance results  
│   ├── outliers_selection.py       # Use for experimentation - not used now  
├── data/
│   ├── combined_water_data.csv     # 6 years of monthly water consumption (2018–2023)
├── results/
│   ├── {run_name}.csv              # Results file (MAPE, MASE for each area)
│   ├── eda/                        # Exploratory data analysis outputs
│   ├── config_perf_comp.png        # Comparison results  Figure 6
│   ├── {run_name}_figures/         # Forecasting and run-related figures
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation
```

## Reproducibility Notes (For Reviewers & Editors)  

The following scripts reproduce the main figures and tables in the paper:  

- **exploratory_analysis.py** → Figures 2–4 (variance decomposition, seasonality distribution, autocorrelation/stationarity)  
- **difficulty_analysis.py** → Figure 5 (difficulty score fitting)  
- **performance_comparison.py** → Figure 6 + Table 2 (config performance comparison)  
- **top_regressors_analysis.py** → Figure 7 (regressor importance ranking)  
- **adaptive_results_summary.py** → Figure 8 (forecasting performance)  
- **prophet_forecasting.py** → Figures 9–10 (forecasting outputs)  

## Citation  
If you use this code or methodology, please cite:  

Salman, A., & Shaka’a, Y. (2025). *Automated Water Demand Forecasting for National-Scale Deployment: A Prophet-Based Framework for Palestinian Municipal Water Management*.  


## Resources  
- [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)  
- [Optuna Documentation](https://optuna.org/)  
- [Time Series Forecasting Basics](https://otexts.com/fpp3/)  
