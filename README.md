# Automated Water Demand Forecasting for National-Scale Deployment: A Prophet-Based Framework for Palestinian Municipal Water Management

This repository contains the code accompanying the paper: *"Automated Water Demand Forecasting for National-Scale Deployment: A Prophet-Based Framework for Palestinian Municipal Water Management"* by **Adnan Salman** and **Yamama Shaka’a**.


## Overview
This repository implements the methodology described in the paper to demonstrate the feasibility and results of the 
proposed automated forecasting system based on the Prophet algorithm. The proposed system doesn’t require manual parameter tuning or feature engineering.
Parameters are tuned for each service area based on its specific characteristics, and the most influential external factors are automatically selected from a comprehensive set
of potential factors for each area.

### **Methodology**
We developed an automated forecasting framework to predict monthly water consumption across 29 heterogeneous urban areas in Nablus (2019–2023) using the Prophet algorithm with automatic hyperparameter optimization via Optuna. The pipeline begins with data preprocessing (imputation of missing values, outlier detection, validation) and exploratory analysis (time-series decomposition, seasonality, predictability, and difficulty classification). Area-specific external regressors, including weather variables, holiday indicators, seasonal Fourier terms, and structural time components, are ranked and selected using ensemble importance scoring. For each area, Prophet models are tuned with Bayesian optimization to capture trend, seasonality, and uncertainty while ensuring scalability and adaptability to diverse demand patterns. Models are trained on 48 months of data and tested on the final 12 months, with walk-forward validation used during tuning. Performance is evaluated using MAE, RMSE, MAPE, and MASE against naive and seasonal baselines, enabling robust, area-specific forecasting for real-world water management.

### **Architecture**
![Automated Water Demand Forecasting Workflow](https://github.com/adnanalshaikh/water-forecasting/blob/main/doc/Figures/archit-prophet1.png) 
**Figure:** Automated water demand forecasting workflow. Raw consumption, weather, and holiday data are preprocessed, analyzed, and transformed into area-specific Prophet models with external regressors and hyperparameter optimization. The trained models are evaluated against multiple error metrics to ensure robust, scalable forecasting across heterogeneous urban areas.

### **Key Constraints**
- **Viral Orders**: 
- **Data Filtering**:

### **Significance**




---

## **Installation**

### **1. Clone the Repository**
```bash
https://github.com/adnanalshaikh/water-forecasting.git
cd water-forecasting
```
### **2. Install the required dependency**
```bash
pip install -r requirements.txt
```

## **Directory Structure**
```bash
water-forecasting/
├── prophet/                            # Main source code
│   ├── dna_taxonomy.py             # Entry point for running the pipeline
│   ├── cnn_model.py                # CNN-LSTM model implementation
│   ├── measures.py                 # Functions for performance evaluation
│   ├── preprocess.py               # Preprocessing functions
│   ├── dna_data_loader.py          # Utilities for data retrieval from NCBI
├── data/                           # Input data
│   ├── Tymovirales_refseq.csv      # Extracted data using dna_data_loader.py
│   ├── Martellivirales_refseq.csv  # Extracted data using dna_data_loader.py
│   ├── Alsuviricetes_refseq.csv    # Extracted data using dna_data_loader.py
│   ├── VMR_MSL39_v1.xlsx           # Virus metadata file downloaded from ICTV (https://ictv.global/vmr)
├── results/                    
│   ├── Tymovirales/Tymovirales.csv         # Tymovirales classifier performance results
│   ├── Martellivirales/Martellivirales.csv # Martellivirales classifier performance results
│   ├── Alsuviricetes/
│   │   ├── Alsuviricetes.csv       # Alsuviricetes classifier performance results 
│   │   ├── Alsuviricetes_h.csv     # Heirarichal Alsuviricetes  classifier performance results
├── requirements.txt                # Dependencies for the project
└── README.md                       # Documentation
```

## **Citation**
<!--
If you use this code or methodology in your work, please cite the following paper:
Salman, A., & Khuffash, N. (2024). Hierarchical Classification of Viral DNA Sequences Using a CNN-LSTM Framework.
-->

## **Resources**

- **Virus Metadata File (VMR_MSL39_v1.xlsx)**:
  - The metadata file used in this project is downloaded from the [ICTV Virus Metadata Resource](https://ictv.global/vmr).
  - Visit the [ICTV VMR Page](https://ictv.global/vmr) for more details and updates.

