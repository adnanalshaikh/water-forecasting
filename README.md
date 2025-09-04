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
![Automated Water Demand Forecasting Workflow](https://github.com/adnanalshaikh/water-forecasting/blob/main/doc/Figures/archit-prophet1.png)  
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


