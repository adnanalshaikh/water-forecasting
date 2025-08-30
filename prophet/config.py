# run several experiments 
config_baseline = {
    'comment' : "",
    'area_id': (1, 29),  # id or [id1, id2, ...], (from_id, to_id)
    
    'test_size': 12, 
    'valid_size': 10,
    
    'impute_thresh': None, 
    'impute_method': 'linear', 
    'outlier_detection': 'iqr', 
    'outlier_treatment': 'interpolate', 
    'auto_outlier_methods': False,    # select best outlier detection method automatically 
    
    'n_trials': 80, 
    'optimize': True, 
    'changepoint_prior_scale': [0.001, 0.6], 
    'seasonality_prior_scale': [0.01, 10], 
    'seasonality_mode': ['additive', 'multiplicative'], 
    'n_changepoints' : [15, 15],
    'fourier_order': [1, 10], 
    'add_regressors': True, 
    'top_regressors': 3, 
    'cv_folds': 0,
    'early_stopping_patience' : 20,
    'n_jobs' : 0, 
    'cv_repeats': 2,           # Number of repetitions
    'n_bootstrap': 10,         # For bootstrap alternative
    'random_state': 42,        # For reproducibility  
    
    'log_experiment_results': True, 
    'log_filepath': 'junk.csv',
    'plot_forecast': False, 
    #'save_plot': True, 
    'save_plot_mode' : 'low', #'high', 'low', 'off'
    
    'data_filepath': '../data/combined_water_data.csv', 
    
    'run_dir' : 'junk',
    'optimize_regressor_selection': True,  # Enable k optimization
    'min_regressors': 2,  # Minimum k value
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
        'y_ma3_regressor': True, 
        'y_ma6_regressor': False, 
        'y_ma12_regressor': False, 
        'y_ewma3_regressor': True, 
        'y_ewma6_regressor': False, 
        'y_std3_regressor': False, 
        'y_std6_regressor': False, 
        'y_std12_regressor': False
        }
    }

# 1, 20, 16, 22, 26

config_baseline.update({
    'area_id': (1, 29), #3, 7, 14, 20] ,  #[16, 22, 26], 
    'run_dir' : 'run1',
    'n_trials': 300,
    'top_regressors':3,
    'n_jobs' : -1, 
    'n_changepoints' : [5, 20],
    'changepoint_prior_scale': [0.001, 0.6],
    'seasonality_prior_scale': [0.01, 5],
    'fourier_order': [1, 10],  # Higher for complex seasonality
    'early_stopping_patience' : 60,

    'optimize_regressor_selection': True,  # Enable k optimization
    'min_regressors': 1,  # Minimum k value
    'max_regressors': 6,  # Maximum k value    
    'add_regressors': True, 
    'save_plot': True,

})

config = config_baseline.copy()

if __name__ == '__main__':
    #from prophet_forecasting_reg import generate_multiple_areas
    #generate_multiple_areas()
    print()
