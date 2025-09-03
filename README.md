# Hierarchical Classification of Viral DNA Sequences Using a CNN-LSTM Framework

This repository contains the code accompanying the paper: *"Automated Water Demand Forecasting for National-Scale Deployment: A Prophet-Based Framework for Palestinian Municipal Water Management"* by **Adnan Salman** and **Yamama Shaka’a**.

## Overview
This repository implements the methodology described in the paper to demonstrate the feasibility and results of the 
proposed automated forecasting system based on the Prophet algorithm. The proposed system doesn’t require manual parameter tuning or feature engineering.
Parameters are tuned for each service area based on its specific characteristics, and the most influential external factors are automatically selected from a comprehensive set
of potential factors for each area.

### **Methodology**
We present an automated forecasting framework using the Prophet algorithm to predict monthly water demand 
across 29 diverse service areas in Nablus. The pipeline cleans and analyzes consumption data, measures 
forecasting difficulty, and selects the most relevant external factors (calendar, weather, seasonal, structural) for each area.
 An ensemble feature-ranking approach ensures only the most influential regressors are used, enabling scalable, accurate, 
 and reliable forecasts for real-world deployment.

### **Key Constraints**
- **Viral Orders**: 
- **Data Filtering**:

### **Significance**


### **Architecture**
![Architecture Diagram](https://github.com/adnanalshaikh/water-forecasting/blob/main/doc/Figures/archit-prophet1.png)

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
