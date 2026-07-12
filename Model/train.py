import os
import json
import time
import pickle
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FINAL_TUNING = True
USE_GRID_SEARCH = True

RUN_MODE = 'final' if FINAL_TUNING else 'debug'
params_filename = f'Model/hyperparameter/last_{RUN_MODE}_params.json'
total_fit_runs_counter = 0

def setup_environment():
    os.makedirs('Model/hyperparameter', exist_ok=True)
    report_base = f'Model/reports/{RUN_MODE}'
    os.makedirs(report_base, exist_ok=True)

    xai_dir = os.path.join(report_base, 'xai')
    if os.path.exists(xai_dir):
        clear_directory(xai_dir)
    else:
        os.makedirs(xai_dir, exist_ok=True)
    for sub in ['quick', 'detailed', 'analysis']:
        os.makedirs(os.path.join(xai_dir, sub), exist_ok=True)

    for f_name in ['training_metrics.json', 'grid_search_metrics.json', 'training_report.md']:
        f_path = os.path.join(report_base, f_name)
        if os.path.exists(f_path):
            os.unlink(f_path)
    clear_directory(os.path.join(report_base, 'horizon'))

def get_model_dir(mode, minutes=None):
    base = f'Model/models/{RUN_MODE}'
    return os.path.join(base, 'quick') if mode.lower() == 'quick' else os.path.join(base, 'detailed', f'{minutes}m')

def clear_directory(path):
    path = os.path.normpath(path)
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                shutil.rmtree(file_path) if os.path.isdir(file_path) else os.unlink(file_path)
            except Exception as e:
                print(f"[WARN] {file_path} nicht gelöscht: {e}")
    else:
        os.makedirs(path, exist_ok=True)

def load_and_preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError("ETL/pipeline.py zuerst ausführen")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1J Sliding Window (Anti Concept Drift)
    cutoff_date = df['timestamp'].max() - pd.Timedelta(days=365)
    return df[df['timestamp'] >= cutoff_date].reset_index(drop=True)

def load_saved_parameters():
    if not USE_GRID_SEARCH and os.path.exists(params_filename):
        try:
            with open(params_filename, 'r', encoding='utf-8') as f:
                print(f"Lade Hyperparameter aus: {params_filename}")
                return json.load(f)
        except Exception as e:
            print(f"Ladefehler {params_filename}: {e}. Benutze Grid Search...")
    return None

def run_pipeline(mode, train_df_h, test_df_h, features, continuous, saved_params_dict=None, minutes=None):
    global total_fit_runs_counter
    mode_str = f"{mode.upper()} ({minutes}m)" if minutes else mode.upper()
    print(f"\nTraining für {mode_str} gestartet...")
    
    cur_model_dir = get_model_dir(mode, minutes)
    clear_directory(cur_model_dir)
    
    X_tr, y_tr = train_df_h[features], train_df_h['target']
    X_te, y_te = test_df_h[features], test_df_h['target']
    
    # Skalierung (Anti Data Leakage)
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), continuous)], remainder='passthrough')
    
    model_classes = {
        'RandomForest': RandomForestRegressor(random_state=142),
        'GradientBoosting': GradientBoostingRegressor(random_state=142),
        'GBQuantile90': GradientBoostingRegressor(loss='quantile', alpha=0.90, random_state=142),
        'XGBoost': XGBRegressor(random_state=142, eval_metric='rmse'),
        'Ridge': Ridge(),
        'Lasso': Lasso(max_iter=10000, random_state=142),
        'ElasticNet': ElasticNet(max_iter=10000, random_state=142)
    }
    
    param_key = f"{mode}_{minutes}m" if minutes else mode
    current_saved_config = None
    if saved_params_dict and param_key in saved_params_dict:
        current_saved_config = saved_params_dict[param_key]['models']
    elif saved_params_dict and mode == 'Detailed' and 'Detailed' in saved_params_dict:
        current_saved_config = saved_params_dict['Detailed']['models']
            
    if current_saved_config:
        models_config = {
            name: (model_classes[name], {k: [v] for k, v in params.items()})
            for name, params in current_saved_config.items() if name in model_classes
        }
    else:
        grids = {
            'RandomForest': {'n_estimators': [100, 200, 500], 'max_depth': [10, 15, 20], 'min_samples_split': [2, 5]} if FINAL_TUNING else {'n_estimators': [50], 'max_depth': [10]},
            'GradientBoosting': {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5], 'subsample': [0.8, 1.0]} if FINAL_TUNING else {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [3]},
            'GBQuantile90': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]} if FINAL_TUNING else {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [3]},
            'XGBoost': {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5], 'subsample': [0.8, 1.0]} if FINAL_TUNING else {'n_estimators': [50], 'learning_rate': [0.1], 'max_depth': [3]},
            'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]} if FINAL_TUNING else {'alpha': [1.0]},
            'Lasso': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]} if FINAL_TUNING else {'alpha': [0.1]},
            'ElasticNet': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]} if FINAL_TUNING else {'alpha': [0.1], 'l1_ratio': [0.5]}
        }
        models_config = {name: (model_classes[name], grids[name]) for name in model_classes}
        
    results = {}
    
    for name, (model, grid) in models_config.items():
        model_start = time.time()
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipe_grid = {f"regressor__{k}": v for k, v in grid.items()}
        
        # TimeSeriesSplit (Kausalitätswahrung)
        cv_splitter = TimeSeriesSplit(n_splits=5)
        # cpu usage limit
        num_cpus = os.cpu_count()
        n_jobs = max(1, int(num_cpus * 0.9)) if num_cpus else 1
        
        gs = GridSearchCV(
            pipe, 
            pipe_grid, 
            cv=cv_splitter, 
            scoring={
                'neg_mse': 'neg_mean_squared_error',
                'r2': 'r2'
            },
            refit='neg_mse',
            n_jobs=n_jobs
        )
        gs.fit(X_tr, y_tr)
        best_pipeline = gs.best_estimator_
        best_model = best_pipeline.named_steps['regressor']
        
        y_pred = best_pipeline.predict(X_te)
        mse = mean_squared_error(y_te, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        clean_best_params = {k.replace('regressor__', ''): v for k, v in gs.best_params_.items()}
        
        # XAI-Gewichte extrahieren
        transformed_features = continuous + [f for f in features if f not in continuous]
        feature_importance_vals = best_model.feature_importances_.tolist() if hasattr(best_model, 'feature_importances_') else None
        coefficient_vals = best_model.coef_.tolist() if hasattr(best_model, 'coef_') else None

        # Grid Search Historie extrahieren
        cv_res = gs.cv_results_
        cv_runs = []
        for i in range(len(cv_res['params'])):
            cv_runs.append({
                'params': {k.replace('regressor__', ''): v for k, v in cv_res['params'][i].items()},
                'mean_fit_time': float(cv_res['mean_fit_time'][i]),
                'mean_test_score': float(cv_res['mean_test_neg_mse'][i]),
                'std_test_score': float(cv_res['std_test_neg_mse'][i])
            })
        total_fit_runs_counter += len(cv_res['params']) * 5 + 1

        # Fold-by-Fold Metriken des besten Modells extrahieren
        best_idx = gs.best_index_
        fold_metrics = []
        for f in range(5):
            neg_mse_val = float(cv_res[f'split{f}_test_neg_mse'][best_idx])
            r2_val = float(cv_res[f'split{f}_test_r2'][best_idx])
            fold_metrics.append({
                'fold': f + 1,
                'MSE': -neg_mse_val,
                'RMSE': np.sqrt(-neg_mse_val),
                'R2': r2_val
            })

        results[name] = {
            'R2': r2, 'RMSE': rmse, 'MSE': mse, 'MAE': mae,
            'best_params': clean_best_params, 'model_obj': best_model,
            'pipeline': best_pipeline,
            'feature_importances': feature_importance_vals,
            'coefficients': coefficient_vals,
            'transformed_features': transformed_features,
            'cv_runs': cv_runs,
            'fold_metrics': fold_metrics
        }
        
        suffix = f"_{minutes}m" if minutes else ""
        with open(os.path.join(cur_model_dir, f'model_{mode.lower()}{suffix}_{name.lower()}.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
            
        print(f"  {name:<16} | R2: {r2:.4f} | RMSE: {rmse:.2f} (Time: {time.time()-model_start:.1f}s)")
        
    best_name = max(results, key=lambda k: results[k]['R2'])
    suffix = f"_{minutes}m" if minutes else ""
    with open(os.path.join(cur_model_dir, f'best_model_{mode.lower()}{suffix}_{best_name.lower()}.pkl'), 'wb') as f:
        pickle.dump(results[best_name]['model_obj'], f)
        
    fitted_scaler = results[best_name]['pipeline'].named_steps['preprocessor'].named_transformers_['num']
    with open(os.path.join(cur_model_dir, f'scaler_{mode.lower()}{suffix}.pkl'), 'wb') as f:
        pickle.dump(fitted_scaler, f)
        
    print(f"  Bestes Modell für {mode_str}: {best_name} (R² = {results[best_name]['R2']:.4f})")
    return results

def train_all_models(df, saved_params):
    weekdays = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']
    base_features = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain', 'is_public_holiday', 'is_school_holiday', 'is_day'] + weekdays
    quick_continuous = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain']
    features_quick = quick_continuous + ['is_public_holiday', 'is_school_holiday', 'is_day'] + weekdays
    detailed_horizons = {5: 1, 15: 3, 30: 6, 60: 12, 90: 18, 120: 24}

    # Quick Modell
    df_quick = df.copy()
    df_quick['target'] = df_quick['utilization_percent']
    split_idx_q = int(len(df_quick) * 0.8)
    res_quick = run_pipeline(
        mode='Quick',
        train_df_h=df_quick.iloc[:split_idx_q],
        test_df_h=df_quick.iloc[split_idx_q:],
        features=features_quick,
        continuous=quick_continuous,
        saved_params_dict=saved_params
    )

    # Detailed Multi-Horizon Modelle
    res_detailed_dict = {}
    for mins, h in detailed_horizons.items():
        df_h = df.copy()
        df_h['target'] = df_h['utilization_percent']
        
        full_range = pd.date_range(start=df_h['timestamp'].min(), end=df_h['timestamp'].max(), freq='5min')
        df_full = df_h.set_index('timestamp').reindex(full_range)
        df_full = df_full.interpolate(method='linear')
        df_full['lag_inference'] = df_full['lag_1'].shift(h - 1)
        df_full['ema_1h_inference'] = df_full['ema_1h'].shift(h - 1)
        df_full['trend_15m_inference'] = df_full['trend_15m'].shift(h - 1)
        df_full['trend_2h_inference'] = df_full['trend_2h'].shift(h - 1)
        df_full['volatility_2h_inference'] = df_full['volatility_2h'].shift(h - 1)
        
        df_full = df_full.reset_index().rename(columns={'index': 'timestamp'})
        df_aligned = pd.merge(
            df_h[['timestamp', 'target', 'lag_1d', 'lag_7d_robust']],
            df_full[['timestamp', 'lag_inference', 'ema_1h_inference', 'trend_15m_inference', 'trend_2h_inference', 'volatility_2h_inference'] + base_features],
            on='timestamp', how='left'
        )
        
        features_h = base_features + ['lag_inference', 'ema_1h_inference', 'trend_15m_inference', 'trend_2h_inference', 'volatility_2h_inference', 'lag_1d', 'lag_7d_robust']
        continuous_h = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain', 'lag_inference', 'ema_1h_inference', 'trend_15m_inference', 'trend_2h_inference', 'volatility_2h_inference', 'lag_1d', 'lag_7d_robust']
            
        df_aligned = df_aligned.dropna(subset=features_h + ['target']).reset_index(drop=True)
        split_idx_h = int(len(df_aligned) * 0.8)
        
        res_detailed_dict[mins] = run_pipeline(
            mode='Detailed',
            train_df_h=df_aligned.iloc[:split_idx_h],
            test_df_h=df_aligned.iloc[split_idx_h:],
            features=features_h,
            continuous=continuous_h,
            saved_params_dict=saved_params,
            minutes=mins
        )
    return res_quick, res_detailed_dict

def export_results(res_quick, res_detailed_dict, global_start):
    # Hyperparameter exportieren
    if USE_GRID_SEARCH:
        best_params_json = {'Quick': {'best_model': max(res_quick, key=lambda k: res_quick[k]['R2']), 'models': {n: res_quick[n]['best_params'] for n in res_quick}}}
        for mins, res in res_detailed_dict.items():
            best_params_json[f"Detailed_{mins}m"] = {'best_model': max(res, key=lambda k: res[k]['R2']), 'models': {n: res[n]['best_params'] for n in res}}
        with open(params_filename, 'w', encoding='utf-8') as json_file:
            json.dump(best_params_json, json_file, indent=4)

    # Metriken exportieren
    metrics_to_save = {
        'meta': {
            'total_training_time_min': round((time.time() - global_start) / 60, 2),
            'total_fit_runs': total_fit_runs_counter,
            'cv_splits': 5,
            'final_tuning_mode': FINAL_TUNING
        },
        'Quick': {
            name: {
                'R2': metrics['R2'], 'RMSE': metrics['RMSE'], 'MSE': metrics['MSE'], 'MAE': metrics['MAE'],
                'best_params': metrics['best_params'], 'feature_importances': metrics['feature_importances'],
                'coefficients': metrics['coefficients'], 'transformed_features': metrics['transformed_features'],
                'fold_metrics': metrics['fold_metrics']
            } for name, metrics in res_quick.items()
        },
        'Detailed': {
            str(mins): {
                name: {
                    'R2': metrics['R2'], 'RMSE': metrics['RMSE'], 'MSE': metrics['MSE'], 'MAE': metrics['MAE'],
                    'best_params': metrics['best_params'], 'feature_importances': metrics['feature_importances'],
                    'coefficients': metrics['coefficients'], 'transformed_features': metrics['transformed_features'],
                    'fold_metrics': metrics['fold_metrics']
                } for name, metrics in res.items()
            } for mins, res in res_detailed_dict.items()
        }
    }

    metrics_path = f'Model/reports/{RUN_MODE}/training_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=4)

    # Grid Search Metriken exportieren
    gs_metrics_to_save = {
        'Quick': {
            name: metrics['cv_runs'] for name, metrics in res_quick.items()
        },
        'Detailed': {
            str(mins): {
                name: metrics['cv_runs'] for name, metrics in res.items()
            } for mins, res in res_detailed_dict.items()
        }
    }
    gs_metrics_path = f'Model/reports/{RUN_MODE}/grid_search_metrics.json'
    with open(gs_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(gs_metrics_to_save, f, indent=4)

def run_reporting():
    import sys
    import subprocess
    try:
        subprocess.run([sys.executable, os.path.join('Model', 'generate_report.py'), RUN_MODE], check=True)
    except Exception as e:
        print(f"[WARN] Report-Generator fehlgeschlagen: {e}")

def main():
    setup_environment()
    df = load_and_preprocess_data('data/gym_workload_mlready.csv')
    saved_params = load_saved_parameters()
    
    global_start = time.time()
    res_quick, res_detailed = train_all_models(df, saved_params)
    print(f"\nGesamtzeit: {(time.time() - global_start)/60:.2f}min")
    
    export_results(res_quick, res_detailed, global_start)
    run_reporting()

if __name__ == '__main__':
    main()
