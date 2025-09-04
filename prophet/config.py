# configuration file for tuning, optimization, and execution parameters

config_baseline = {
    'comment' : "",
    'area_id': (1, 29),  # tuple range or list of ids or id
    
    # splits (months)
    'test_size': 12, 
    'valid_size': 10,
    
    # missing/outliers
    'impute_thresh': None,          # if None, always impute using method
    'impute_method': 'linear',      
    'outlier_detection': 'iqr', 
    'outlier_treatment': 'interpolate', 
    'auto_outlier_methods': False, # select best outlier detection method automatically 
    
    # # Prophet/Optuna search space
    'optimize': True,       
    'n_trials': 300,
    'n_changepoints' : [5, 20],   # [min,max] (interpreted by code)
    'changepoint_prior_scale': [0.001, 0.6], 
    'seasonality_prior_scale': [0.01, 5], 
    'seasonality_mode': ['additive', 'multiplicative'], 
    'fourier_order': [1, 8],   # Higher for complex seasonality
    'early_stopping_patience' : 50,
    
    # cross validation (unused)
    'cv_folds': 0,             # not used 
    'cv_repeats': 0,           # Number of repetitions
    
    # execution
    'n_jobs' : -1,              # use single process, -1 for all cores
    'data_filepath': "../data/combined_water_data.csv", 
    
    # logging results 
    'run_dir' : 'test_run',
    'log_experiment_results': True,        
    "plot": {
        "show": False,        # whether to display on screen
        "save": True,         # whether to save to file
        "mode": "low",       # {"high","low"} resolution
        "formats": ["png"],   # list of formats, e.g. ["png","pdf","svg"]
        "dpi": 150            # dpi for saved figures when mode="high"
        },
    
    # regressor analysis 
    'add_regressors': True, 
    'top_regressors': 3,         

    # or tune for k regressors 
    'optimize_regressor_selection': True,  # Enable k optimization
    'min_regressors': 1,  # Minimum k value
    'max_regressors': 8,  # Maximum k value
    
    'regressors': {
        'ramadan_month_regressor': True, 
        'eid_alfitr_month_regressor': True, 
        'eid_aladha_month_regressor': True, 
        'fridays_per_month_regressor': True, 
        'temperature_regressor': True, 
        'temp_anomaly_regressor': True, 
        'humidity_regressor': True, 
        'rainfall_regressor': True, 
        'month_sin_regressor': True, 
        'month_cos_regressor': True, 
        'month_sin2_regressor': False, 
        'month_cos2_regressor': False, 
        'winter_regressor': True, 
        'spring_regressor': True, 
        'summer_regressor': True, 
        'fall_regressor': True, 
        'years_since_start_regressor': True, 
        'years_since_start_squared_regressor': True, 
        'years_since_start_power1_5_regressor': False, 
        'years_since_start_log_regressor': False, 
        'years_since_halfway_regressor': False, 
        'multi_year_cycle_regressor': False, 
        'cumulative_rainfall_regressor': False, 
        'temp_trend_interaction_regressor': False, 
        'y_ma3_regressor': False,  #
        'y_ma6_regressor': False, 
        'y_ma12_regressor': False, 
        'y_ewma3_regressor': False, #
        'y_ewma6_regressor': False, 
        'y_std3_regressor': False, 
        'y_std6_regressor': False, 
        'y_std12_regressor': False
        }
    }

# 1, 20, 16, 22, 26
config = config_baseline.copy()
config.update({
    'area_id': [1, 15, 22, 26], #3, 7, 14, 20] ,  #[16, 22, 26], 
    'run_dir' : 'run2',

    'add_regressors': True, 
    'optimize_regressor_selection': True,  # Enable k optimization
    'min_regressors': 1,  # Minimum k value
    'max_regressors': 6,  # Maximum k value    
})




