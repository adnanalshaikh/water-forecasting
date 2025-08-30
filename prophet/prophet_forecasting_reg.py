import pandas as pd
import numpy as np
import logging
import os
import warnings
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
    
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import parallel_backend
from outliers_selection import preprocess_with_outliers
from regressors import add_regressors
from sklearn.model_selection import TimeSeriesSplit
from outliers_selection import select_optimal_outlier_method_statistical
from regressors import regressors_importants_weighted
from dataloader import load_time_series
import time

results_dir = Path(__file__).parent.parent / 'results'
results_dir.mkdir(exist_ok=True, parents=True)

def configure_environment():
    """Configure logging, warnings, and display settings."""
    
    # Logging configuration
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("optuna").setLevel(logging.ERROR)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Environment variables
    os.environ["CMDSTANPY_VERBOSE"] = "False"
    
    # Matplotlib configuration
    plt.rcParams.update({
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.titlesize': 18
    })
    
    # Pandas display options
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 10)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option('display.float_format', '{:.2f}'.format)
   
class RepeatedTimeSeriesSplit:
    """
    Custom implementation of repeated time series cross-validation
    for more stable hyperparameter optimization
    """
    
    def __init__(self, n_splits=3, n_repeats=3, test_size=None, gap=0, random_state=None):
        """
        Parameters:
        - n_splits: Number of splits per repetition
        - n_repeats: Number of repetitions with different starting points
        - test_size: Size of test set for each split
        - gap: Gap between train and test sets
        - random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.test_size = test_size
        self.gap = gap
        self.random_state = random_state
        
    def split(self, X, y=None, groups=None):
        """Generate repeated time series splits"""
        n_samples = len(X)
        
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        for repeat in range(self.n_repeats):
            # Create different starting points for each repetition
            # This adds variability while respecting temporal order
            start_offset = repeat * (n_samples // (self.n_repeats * 2))
            available_samples = n_samples - start_offset
            
            if available_samples < 24:  # Need at least 2 years of data
                continue
                
            # Create TimeSeriesSplit for this repetition
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                gap=self.gap
            )
            
            # Generate splits with offset
            for train_idx, test_idx in tscv.split(X[start_offset:]):
                # Adjust indices to account for offset
                adjusted_train_idx = train_idx + start_offset
                adjusted_test_idx = test_idx + start_offset
                
                # Ensure we don't exceed array bounds
                if adjusted_test_idx[-1] < n_samples:
                    yield adjusted_train_idx, adjusted_test_idx

def get_early_stopping_callback(patience=10):
    """
    Returns an EarlyStopping callback, with fallback to manual implementation.
    """
    try:
        from optuna.callbacks import EarlyStopping
        return EarlyStopping(patience=patience)
    except ImportError:
        # Fall back to manual implementation if import fails
        print("Could not import EarlyStopping, using manual implementation")
        
        # Define a simple EarlyStopping class
        class EarlyStopping:
            def __init__(self, patience=10):
                self.patience = patience
                self.best_value = float('inf')
                self.no_improvement_count = 0
                
            def __call__(self, study, trial):
                if trial.value < self.best_value:
                    self.best_value = trial.value
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                    
                if self.no_improvement_count >= self.patience:
                    study.stop()
                    print(f"Early stopping after {self.no_improvement_count} trials without improvement")
        return EarlyStopping(patience=patience)

def train_test_split(df, test_size=12):
    """
    Split the data into training and test sets.
    
    Parameters:
    - df: DataFrame with columns 'ds' and 'y'
    - train_size: Number of periods to use for training
    
    Returns:
    - train: Training set
    - test: Test set
    """
    train_size = len(df) - test_size
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    print(f'Train size: {len(train)}')
    print(f'Test size: {len(test)}')
    
    return train, test

def tune_hyperparameters_with_repeated_cv(df, config):
    """
    Repeated CV that exactly matches your original regressor logic
    
    Parameters:
    - df: DataFrame with 'ds', 'y', and selected regressor columns (already filtered)
    - config: Dictionary with configuration parameters
    
    Returns:
    - Dictionary of best hyperparameters
    """
    # Setup repeated cross-validation
    n_splits = config.get('cv_folds', 3)
    n_repeats = config.get('cv_repeats', 3)
    test_size = config.get('valid_size', 12) #// n_splits
    
    repeated_cv = RepeatedTimeSeriesSplit(
        n_splits=n_splits,
        n_repeats=n_repeats,
        test_size=test_size,
        random_state=config.get('random_state', 42)
    )
    
    early_stopping = get_early_stopping_callback(patience=config.get('early_stopping_patience', 10))
    
    def objective(trial):
        # Define hyperparameters to tune
        cps = config['changepoint_prior_scale']
        sps = config['seasonality_prior_scale']
        changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', cps[0], cps[1])
        seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', sps[0], sps[1])
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        fourier_order = trial.suggest_int('fourier_order', config['fourier_order'][0], 
                                         config['fourier_order'][1])
        
        cv_errors = []
        fold_count = 0

        try:
            # Perform repeated time series cross-validation
            for train_idx, val_idx in repeated_cv.split(df):
                fold_count += 1
                print(f"Running fold {fold_count} (Trial {trial.number})")
                
                train_df = df.iloc[train_idx].copy()
                val_df = df.iloc[val_idx].copy()
                
                if len(val_df) < 2:
                    continue
                
                train_with_regressors = add_regressors(train_df, mode='forecast')
                
                model = Prophet(
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    seasonality_mode=seasonality_mode,
                )
                model.add_seasonality(name='yearly', period=12, fourier_order=fourier_order)
                
                regressors = [col for col in train_with_regressors.columns if col.endswith('_regressor')]
                for reg in regressors:
                    model.add_regressor(reg)
                        
                model.fit(train_with_regressors)

                future = model.make_future_dataframe(periods=len(val_df), freq='MS')
                
                future_with_regressors = add_regressors(future, mode='forecast')
                
                forecast = model.predict(future_with_regressors)
                
                y_true = val_df['y']
                y_pred = forecast['yhat'].iloc[-len(val_df):].values
                
                mae = mean_absolute_error(y_true, y_pred)
                cv_errors.append(mae)
                
                # Optional: Add pruning for efficiency
                if len(cv_errors) >= 3:
                    trial.report(np.mean(cv_errors), len(cv_errors))
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            return float('inf')
        
        return np.mean(cv_errors) if cv_errors else float('inf')
    
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    n_jobs = config.get('n_jobs', -1)
        
    if n_jobs != 0:
        with parallel_backend('loky', n_jobs=n_jobs):
            study.optimize(
                objective, 
                n_trials=config['n_trials'],
                n_jobs=n_jobs,
                callbacks=[early_stopping],
                show_progress_bar=True
            )
    else:
        print("Running serial with repeated CV...")
        study.optimize(
            objective, 
            n_trials=config['n_trials'], 
            callbacks=[early_stopping]
        )
        
    print(f'Best parameters: {study.best_params}')
    return study.best_params
    
def tune_hyperparameters_with_cv(df, config):
    """
    Optimize hyperparameters using Optuna with TimeSeriesSplit cross-validation.
    
    Parameters:
    - df: DataFrame with 'ds' and 'y' columns
    - config: Dictionary with configuration parameters
    
    Returns:
    - Dictionary of best hyperparameters
    """
    # Setup cross-validation
    n_folds = config.get('cv_folds', 3)
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=config.get('valid_size', 12) ) #// n_folds)
    early_stopping = get_early_stopping_callback(patience=config.get('early_stopping_patience', 10))
    
    def objective(trial):
        # Define hyperparameters to tune
        cps = config['changepoint_prior_scale']
        sps = config['seasonality_prior_scale']
        changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', cps[0], cps[1])
        seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', sps[0], sps[1])
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        fourier_order = trial.suggest_int('fourier_order', config['fourier_order'][0], 
                                         config['fourier_order'][1])
        
        # Store errors from each fold
        cv_errors = []

        # Perform time series cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
            # Create train/validation split
            print("running fold \n")
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            
            # Add regressors to train data
            train_with_regressors = add_regressors(train_df, mode='forecast')
            
            # Create and initialize model
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
            )
            model.add_seasonality(name='yearly', period=12, fourier_order=fourier_order)
            
            # Add regressors
            regressors = [col for col in train_with_regressors.columns if col.endswith('_regressor')]
            for reg in regressors:
                model.add_regressor(reg)
                    
            # Fit the model
            model.fit(train_with_regressors)
            
            # Create future dataframe for the validation period
            future = model.make_future_dataframe(periods=len(val_df), freq='MS')
            
            # Add regressors to future dataframe
            future_with_regressors = add_regressors(future, mode='forecast')
            
            # Make forecast
            forecast = model.predict(future_with_regressors)
            
            # Extract predictions for validation period
            y_true = val_df['y']
            y_pred = forecast['yhat'].iloc[-len(val_df):].values
            
            # Calculate error for this fold
            mae = mean_absolute_error(y_true, y_pred)
            cv_errors.append(mae)
                
        # Return the average error across all folds
        return np.mean(cv_errors) if cv_errors else float('inf')
    
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)

    
    n_jobs = config.get('n_jobs', -1)  # -1 uses all available cores
        
    # Use a context manager for parallel execution
    if n_jobs != 0:
        with parallel_backend('loky', n_jobs=n_jobs):
            study.optimize(
                objective, 
                n_trials=config['n_trials'],
                n_jobs=n_jobs,
                callbacks=[early_stopping],
                show_progress_bar=True
            )
    else:
        print ("running serial without early stopping ")
        study.optimize(objective, n_trials=config['n_trials'], callbacks=[early_stopping]
                       )
        
    print(f'Best parameters: {study.best_params}')
    return study.best_params

def tune_hyperparameters(df, config):
    valid_size = config['valid_size']
    train_size = len(df) - valid_size
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    early_stopping = get_early_stopping_callback(patience=config.get('early_stopping_patience', 10))
    n_jobs = config.get('n_jobs', -1)
    if n_jobs == 0:
        n_jobs = 1
    
    add_regrs = config.get('add_regressors', True)
    # Get min and max number of regressors to use
    if add_regrs:
        min_regressors = config.get('min_regressors', 2)
        max_regressors = config.get('max_regressors', 6)
        
        # Get the top regressors up to max_regressors
        _, top_regressors_ordered = regressors_importants_weighted(df, top_regressors=max_regressors)
        
        # Create subsets for each k value
        regressor_sets = {}
        for k in range(min_regressors, min(max_regressors + 1, len(top_regressors_ordered) + 1)):
            regressor_sets[k] = top_regressors_ordered[:k]
    
    start_time = time.time()
    
    def objective(trial):
        # Prophet hyperparameters
        cps = config['changepoint_prior_scale']
        changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', cps[0], cps[1])
        
        sps = config['seasonality_prior_scale']
        seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', sps[0], sps[1])
        
        cpn = config['n_changepoints']
        n_changepoints = trial.suggest_int('n_changepoints', cpn[0], cpn[1])
        seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
        fourier_order = trial.suggest_int('fourier_order', config['fourier_order'][0], 
                                         config['fourier_order'][1])
        
        # Select the number of regressors to use
        
        if add_regrs:
            n_regressors = trial.suggest_int('n_regressors', min_regressors, max_regressors)
            selected_regressors = regressor_sets[n_regressors]
        else:
            selected_regressors = []
        
        # Create and fit model
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            n_changepoints=n_changepoints,
            interval_width=0.95
        )
        
        model.add_seasonality(name='yearly', period=12, fourier_order=fourier_order)
        
        # Add selected regressors
        if add_regrs:
            for reg in selected_regressors:
                model.add_regressor(reg)
        
        # Fit model with only the necessary columns
        train_subset = train[['ds', 'y'] + selected_regressors].copy()
        model.fit(train_subset)
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(test), freq='MS')
        for col in selected_regressors:
            future[col] = df[col].values
        
        forecast = model.predict(future)
        
        # Calculate error
        y_true = test['y']
        y_pred = forecast['yhat'].iloc[-len(test):]
        mae = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        
        # Store selected regressors for later retrieval
        trial.set_user_attr('selected_regressors', selected_regressors)
        
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=config['n_trials'], callbacks=[early_stopping], n_jobs=n_jobs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total optimization time: {elapsed_time:.2f} seconds")
    print(f"Number of trials: {len(study.trials)}")
    
    # Get best parameters and selected regressors
    best_params = study.best_params.copy()
    best_regressors = study.best_trial.user_attrs.get('selected_regressors', [])
    
    # Remove n_regressors from params and add selected regressors
    best_params.pop('n_regressors', None)
    best_params['selected_regressors'] = best_regressors
    
    print(f'Best parameters: {best_params}')
    print(f'Selected regressors ({len(best_regressors)}): {best_regressors}')
    
    return best_params, len(study.trials), elapsed_time

def log_experiment_results(results, filepath='experiment_log.csv', log_option=True, config = None):
    """
    Log experiment results to a CSV file.
    
    Parameters:
    - results: Dictionary containing analysis results
    - filepath: Path to save the log file
    - log_option: 
        True: Append to existing file
        False: Don't log
        None: Create new file (overwrite existing)
    
    Returns:
    - The DataFrame row that was logged (or None if logging was skipped)
    """
    from datetime import datetime
    
    # Don't log if option is False
    if log_option is False:
        print("Logging skipped (log_option=False)")
        return None
    
    # Extract the data we want to log
    metrics_wf = results['metrics']
    params = results['parameters']
    outlier_info = results.get('outlier_info', {})
    
    # Create a dictionary with all information to log
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'area_id': results['area_id'],
        
        # Model parameters
        'seasonality_mode': params.get('seasonality_mode', 'None'),
        'changepoint_prior_scale': params.get('changepoint_prior_scale', 'None'),
        'seasonality_prior_scale': params.get('seasonality_prior_scale', 'None'),
        'fourier_order': params.get('fourier_order', 'None'),
        'n_trials': results.get('n_trials', 0),
        
        # Outlier handling information
        'outlier_detection': outlier_info.get('detection_method', 'None'),
        'outlier_treatment': outlier_info.get('treatment_method', 'None'),
        'outliers_found': outlier_info.get('count', 0),
        'outliers_percentage': outlier_info.get('percentage', 0),
        'missing_found' : results.get('missing_count', None),
             
        # Walk-forward metrics
        'mae': metrics_wf.get('mae', None),
        'rmse': metrics_wf.get('rmse', None),
        'mape': metrics_wf.get('mape', None),
        'mase': metrics_wf.get('mase', None),
        'mase_seas': metrics_wf.get('mase_seas', None),
        'metrics' : metrics_wf,
                
        # Additional configuration
        'test_size': results['config'].get('test_size', None),
        'regressors': results.get('regressors', None),
        'config' : results['config'],
        'exec_time' : results['exec_time']
    }
    
    # Convert to DataFrame
    log_df = pd.DataFrame([log_data])
    
    # Determine file handling based on log_option
    filepath  = results_dir / f"{config.get('run_dir', 'run_xxx')}.csv"
    file_exists = os.path.isfile(filepath)
    
    if log_option is None or (not file_exists):
        # Create new file (overwrite if exists)
        log_df.to_csv(filepath, index=False)
        print(f"Created new log file: {filepath}")
    else:  # log_option is True
        # Append to existing file
        if file_exists:
            log_df.to_csv(filepath, mode='a', header=False, index=False)
            print(f"Appended to existing log file: {filepath}")
        else:
            log_df.to_csv(filepath, index=False)
            print(f"Created new log file (append mode): {filepath}")
    
    return log_df

def plot_walkforward_forecast(train, test, walk_forward_forecast, intermediate_forecast, 
                              area_id, metrics_walk=None, config=None):
    """
    Create a visualization of walk-forward forecasting results with connected lines.
    """
    # Set plot style
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })

    # adnan 

    # Create the figure
    fig = plt.figure(figsize=(15, 8))

    # Plot training data
    plt.plot(train['ds'], train['y'], 'b-', linewidth=3, label='Training Data')
    
    # Plot test data
    plt.plot(test['ds'], test['y'], 'g-', linewidth=3, label='Test Data (Actual)')
    
    # Add connecting line between train and test
    plt.plot([train['ds'].iloc[-1], test['ds'].iloc[0]],
             [train['y'].iloc[-1], test['y'].iloc[0]],
             'b-', linewidth=3, alpha=1)
    
    # Plot model fit (intermediate forecast)
    plt.plot(intermediate_forecast['ds'], intermediate_forecast['yhat'], 'r--', alpha=1,
             label='Model Fit (Training Period)')
    
    # Plot walk-forward forecast confidence interval
    plt.fill_between(walk_forward_forecast['ds'],
                     walk_forward_forecast['y_lower'],
                     walk_forward_forecast['y_upper'],
                     color='m', alpha=0.1,
                     label='95% Confidence Interval')
    
    # Plot walk-forward forecast
    plt.plot(walk_forward_forecast['ds'], walk_forward_forecast['y_pred'], 'mo-',
             linewidth=3, markersize=6, label='Walk-Forward Forecast')
    
    # Add vertical line separating train/test
    plt.axvline(x=test['ds'].iloc[0], color='gray', linestyle='--', alpha=1)
    plt.text(test['ds'].iloc[0], plt.ylim()[1]*0.95, 'Train | Test',
             horizontalalignment='center', backgroundcolor='white', fontsize=18)
    
    # Add metrics if available
    if metrics_walk:
        metrics_text = (
            f"Metrics:\n"
            f"MAPE: {metrics_walk['mape']:.2f}%\n"
            f"MASE: {metrics_walk['mase']:.2f}\n"
            f"MASE_SEAS: {metrics_walk['mase_seas']:.2f}"
        )
        # area 3 : 0.84, 0.175
        # area 20: 0.02 0.175
        plt.text(0.84, 0.175, metrics_text, transform=plt.gca().transAxes,
                 fontsize=18, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=1))

    # Labels and formatting
    #plt.title(f'Area {area_id}: Monthly Forecasting', fontsize=20)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Water Consumption', fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()

    # Save figure in multiple formats
    save_plot_mode = config.get('save_plot_mode', 'off')
    if  save_plot_mode in ['high', 'low']:
        figures_dir = results_dir / f"{config.get('run_dir', 'run_xxx')}_figures"
        figures_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        
        base_name = figures_dir / f"area_{area_id}_forecast"

        if save_plot_mode ==  'high':
            for ext in ['png', 'tif', 'pdf']:
                plt.savefig(f"{base_name}.{ext}",
                            dpi=600,
                            bbox_inches='tight',
                            format=ext,
                            transparent=False,
                            facecolor='white')
        else:
                plt.savefig(f"{base_name}.png",
                            dpi=300,
                            bbox_inches='tight',
                            format='png',
                            transparent=False,
                            facecolor='white')
    
    plt.close(fig)  # Close for memory efficiency
    return fig

def evaluate_forecast(y_true, y_pred, y_train, label = ''):
    """
    Evaluate the forecast using multiple metrics.
    
    Parameters:
    - y_true: Actual values (test set)
    - y_pred: Predicted values
    - y_train: Training set actual values
    
    Returns:
    - Dictionary of evaluation metrics
    """
    # Calculate MAE and RMSE

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate MASE (Mean Absolute Scaled Error)
    naive_forecast = np.mean(np.abs(np.diff(y_train)))
    
    seasonal_period = 12
    naive_forecast_seas  = np.mean(np.abs(y_train[seasonal_period:].values - y_train[:-seasonal_period].values))
    
    mase = mae / naive_forecast
    mase_seas = mae / (naive_forecast_seas + 1e-9)
    
    
    print(label)
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAPE: {mape:.2f}')
    print(f'MASE: {mase:.2f}%\n')
    print(f'MASE_SEAS: {mase_seas:.2f}%\n')
    #seasonal_period = 12
    #naive_forecast_seas = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    #mase_s = mae / (naive_forecast_seas+0.000001)
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'mase': mase, 'mase_seas': mase_seas } #'MASE_S': mase_s,}

def forecast_walk_forward(df, test_size, params=None, interval_width=0.95):
    """
    Perform walk-forward forecasting with confidence intervals and intermediate forecasts.
    
    Parameters:
    - df: DataFrame with 'ds' and 'y' columns
    - test_size: Number of periods to forecast
    - params: Prophet model parameters
    - interval_width: Width of the prediction interval (0-1)
    
    Returns:
    - walk_forward_forecast: DataFrame with one-step-ahead forecasts and intervals
    - intermediate_forecast: DataFrame with Prophet's fit on the training data
    """
    train_size = len(df) - test_size
    forecasts = []
    
    # Store the final intermediate forecast (based on the last training iteration)
    intermediate_forecast = None
    
    for i in range(test_size):
        # Create training set up to current point
        train = df.iloc[:train_size + i]
        val = df.iloc[train_size + i:train_size + i + 1]
        
        # Configure Prophet model
        if params:
            model = Prophet(
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
                seasonality_mode=params.get('seasonality_mode', 'multiplicative'),
                interval_width=interval_width,
                n_changepoints=params.get('n_changepoints', 15),
            )
            if 'fourier_order' in params:
                model.add_seasonality(name='yearly', period=12, fourier_order=params['fourier_order'])
        else:
            model = Prophet(interval_width=interval_width)
            model.add_seasonality(name='yearly', period=12, fourier_order=5)
        
        # Add regressors
        regressors = [col for col in train.columns if col.endswith('_regressor')]
        for reg in regressors:
            model.add_regressor(reg)
        
        # Fit model
        model.fit(train)
        
        # Create future dataframe for the validation point
        future = pd.DataFrame({'ds': [val['ds'].iloc[0]]})
        for col in regressors:
            future[col] = val[col].values
        
        # Generate forecast with intervals
        forecast = model.predict(future)
        
        # Store results including intervals
        forecasts.append({
            'ds': val['ds'].iloc[0],
            'y': val['y'].iloc[0],
            'y_pred': forecast['yhat'].iloc[0],
            'y_lower': forecast['yhat_lower'].iloc[0],
            'y_upper': forecast['yhat_upper'].iloc[0]
        })
        
        # Store the intermediate forecast from the last iteration
        if i == test_size - 1:
            # Create future dataframe for the entire historical period
            future_full = model.make_future_dataframe(periods=0, freq='MS')
            
            # Add regressor values
            for col in regressors:
                future_full[col] = df.iloc[:train_size + i][col].values
            
            # Generate full forecast
            intermediate_forecast = model.predict(future_full)
    
    # Combine results
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    walk_forward_forecast = pd.DataFrame(forecasts)
    
    return walk_forward_forecast, intermediate_forecast, train, test

def run_forecasting_pipeline(df, area_id, test_size, params, config = None):
    """
    Runs the full forecasting workflow:
    - Makes single-shot forecast
    - Runs walk-forward validation
    - Evaluates both methods
    - Plots everything
    
    Parameters:
    - df: DataFrame with 'ds' and 'y' columns
    - area_id: Identifier for the plot title
    - test_size: Number of periods for the test set
    - params: Prophet model parameters
    - walk_forward_fn: Function that performs walk-forward validation
    
    Returns:
    - Dictionary with forecast, walk-forward results, and evaluation metrics
    """
    
    forecasts, intermediate_forecast, train, test = forecast_walk_forward(df, test_size, params)

    y_train = train['y']
    y_true = pd.Series(test['y'].values, index=test['ds'], name='y_true')
    y_pred_wf = pd.Series(forecasts['y_pred'].values, index=test['ds'], name='y_pred_wf')
    
    # Evaluation  
    metrics = evaluate_forecast(y_true, y_pred_wf, y_train, label='Walk-Forward Forecast')


    if config:  
        plot_forecast = config.get('plot_forecast', False)
        save_plot = config.get('save_plot', False)
        
    if plot_forecast or save_plot:
        plot_walkforward_forecast(train, test, forecasts, intermediate_forecast, 
                                  area_id, metrics, config = config)
                    
    return {'forecasts': forecasts, 'metrics': metrics }

def analyze_area(filepath, config):
    
    start_time = time.time()
    area_id = config['area_id'] if isinstance(config['area_id'], int) else config['area_id'][0]
    test_size = config['test_size']
    optimize = config['optimize']
    
    print(f"\n=== Analyzing Area {area_id} ===\n")
    
    df, missing_count = load_time_series(filepath, area_id)
    if df is None:
        return None, missing_count
    
    auto_outlier_methods = config.get('auto_outlier_methods', False)
    if auto_outlier_methods:
        best_combination, _ = select_optimal_outlier_method_statistical(df, area_id)
        print("best_combination: ", best_combination)
        config.update({'outlier_detection': best_combination[0], 'outlier_treatment': best_combination[1]})
    
    df, outlier_info = preprocess_with_outliers(df, config)
    
    df_filtered = df.copy()
    if config.get('add_regressors', False):
        which_regressors = config.get('regressors', None)
        df = add_regressors(df, which_regressors, mode='forecast')
        
        # If optimizing with regressor selection, use the full df for now
        if optimize and config.get('optimize_regressor_selection', True):
            df_filtered = df.copy()
            top_regressor_names = None  # Will be determined during optimization
        else:
            # Use the standard approach with fixed top_regressors
            df_filtered, top_regressor_names = regressors_importants_weighted(
                df, top_regressors=config.get('top_regressors', 3)
            )
    
    train, test = train_test_split(df_filtered, test_size)
    n_trials = 0
    if optimize:
        params, n_trials, exec_time = tune_hyperparameters(df_filtered, config)
        
        # Extract selected regressors from params
        selected_regressors = params.get('selected_regressors', [])
        if selected_regressors:
            # Filter df to only include selected regressors
            cols_to_keep = ['ds', 'y'] + selected_regressors
            df_filtered = df_filtered[cols_to_keep].copy()
            top_regressor_names = selected_regressors
    else:
        params = {}
        # If not optimizing but regressors were added, get the names
        if config.get('add_regressors', False) and top_regressor_names is None:
            _, top_regressor_names = regressors_importants_weighted(
                df, top_regressors=config.get('top_regressors', 3)
            )
    
    print()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"area_id: {area_id}, Time: {elapsed_time} ")
    #print("top regressors: ", selected_regressors)
    #print()
    
    forecast_results = run_forecasting_pipeline(df_filtered, area_id, test_size, params, config=config)
    
    # Build the final results dictionary
    final_results = {
        'area_id': area_id,
        'forecasts': forecast_results['forecasts'],
        'metrics': forecast_results['metrics'],
        'parameters': params,
        'outlier_info': outlier_info,
        'missing_count': missing_count,
        'regressors': [reg for reg in df_filtered.columns if reg.endswith('_regressor')],
        'n_trials': n_trials, #config.get('n_trials', 0),
        'exec_time': elapsed_time,
        'config': config,
    }
    
    if config.get('log_experiment_results', False) is not False:
        log_experiment_results(
            final_results,
            filepath=config.get('log_filepath', 'experiment_log.csv'),
            log_option=config.get('log_experiment_results', True),
            config=config
        )
    
    return final_results

def generate_multiple_areas():

    filepath = '../data/combined_water_data.csv'
    from config import config
    
    figures_dir = results_dir / f"{config.get('run_dir', 'run_xxx')}_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(config['area_id'], list):
        area_ids = config['area_id'] 
    elif isinstance(config['area_id'], tuple):
        area_ids = list(range(config['area_id'][0], config['area_id'][1]+1))
    else:
        area_ids = [config['area_id']]
        
    for id in area_ids:
        area_config = config.copy()
        area_config['area_id'] = id
        results = analyze_area(filepath, area_config)
        if results is None : 
            print (f'ERROR: bad time series or too many missing values: area id: {id}\n')
            continue 
        time.sleep(0.05)

if __name__ == '__main__':
    configure_environment()
    start_time = time.time()
    generate_multiple_areas()
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"exec_time: {exec_time}")
