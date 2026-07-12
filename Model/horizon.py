import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError("ETL/pipeline.py zuerst ausführen")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

def run_experiment(df, horizons, base_features):
    results = []
    print("==========================================\nStarte Horizont-Experiment (5min - 8h)\n==========================================")
    
    for label, h in horizons.items():
        df_h = df.copy()
        df_h['target'] = df_h['utilization_percent']
        df_h['lag_inference'] = df_h['utilization_percent'].shift(h)
        df_h['ema_1h_inference'] = df_h['ema_1h'].shift(h)
        df_h['trend_2h_inference'] = df_h['trend_2h'].shift(h)
        df_h['volatility_2h_inference'] = df_h['volatility_2h'].shift(h)
        features = base_features + ['lag_inference', 'ema_1h_inference', 'trend_2h_inference', 'volatility_2h_inference', 'lag_1d']
        continuous = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain', 'lag_inference', 'ema_1h_inference', 'trend_2h_inference', 'volatility_2h_inference', 'lag_1d']
        
        df_h = df_h.dropna(subset=features + ['target'])
        split_idx = int(len(df_h) * 0.8)
        train_df, test_df = df_h.iloc[:split_idx], df_h.iloc[split_idx:]
        X_tr, y_tr = train_df[features], train_df['target']
        X_te, y_te = test_df[features], test_df['target']
        
        # Skalierung nach Split (Anti Data Leakage)
        scaler = StandardScaler()
        X_tr_sc, X_te_sc = X_tr.copy(), X_te.copy()
        X_tr_sc[continuous] = scaler.fit_transform(X_tr[continuous])
        X_te_sc[continuous] = scaler.transform(X_te[continuous])
        
        models = {
            'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000),
            'GBR': GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(learning_rate=0.05, max_depth=3, n_estimators=100, random_state=42, eval_metric='rmse')
        }
        
        metrics = {'Horizont': label, 'Stunden': (h * 5) / 60}
        for name, model in models.items():
            model.fit(X_tr_sc, y_tr)
            preds = model.predict(X_te_sc)
            metrics[f'{name}_R2'] = r2_score(y_te, preds)
            metrics[f'{name}_RMSE'] = np.sqrt(mean_squared_error(y_te, preds))
            
        results.append(metrics)
        print(f"  {label:<8} | Lasso R²: {metrics['Lasso_R2']:.4f} | XGB R²: {metrics['XGBoost_R2']:.4f}")
    return results

def plot_results(res_df, horizons):
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'Lasso': 'royalblue', 'ElasticNet': 'orange', 'GBR': 'forestgreen', 'XGBoost': 'purple'}
    markers = {'Lasso': 'o', 'ElasticNet': 'd', 'GBR': 's', 'XGBoost': 'x'}
    models_order = ['ElasticNet', 'Lasso', 'GBR', 'XGBoost']
    linewidths = {'Lasso': 2.5, 'ElasticNet': 5.0, 'GBR': 2.5, 'XGBoost': 2.5}
    zorders = {'ElasticNet': 1, 'Lasso': 2, 'GBR': 3, 'XGBoost': 3}
    x_positions = np.arange(len(res_df))
    
    for name in models_order:
        sns.lineplot(x=x_positions, y=res_df[f'{name}_R2'], marker=markers[name], label=name, color=colors[name], linewidth=linewidths[name], zorder=zorders[name], ax=ax1)
        sns.lineplot(x=x_positions, y=res_df[f'{name}_RMSE'], marker=markers[name], label=name, color=colors[name], linewidth=linewidths[name], zorder=zorders[name], ax=ax2)
    
    ax1.set_title('R²-Score vs. Vorhersagehorizont', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Vorhersagehorizont')
    ax1.set_ylabel('R²-Score')
    ax1.set_ylim(-0.1, 1.05)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(list(horizons.keys()))
    
    ax2.set_title('RMSE vs. Vorhersagehorizont', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Vorhersagehorizont')
    ax2.set_ylabel('RMSE [%]')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(list(horizons.keys()))
    
    plt.tight_layout()
    os.makedirs('Model/reports/horizon', exist_ok=True)
    plt.savefig('Model/reports/horizon/horizon_analysis.png', dpi=300, transparent=True)
    plt.close()

def generate_report(results):
    with open('Model/reports/horizon/horizon_report.md', 'w', encoding='utf-8') as f:
        f.write("# Experimenteller Bericht: Vorhersagehorizont vs. Präzision (inkl. XGBoost)\n\n")
        f.write("## 1. Ergebnisse des Experiments\n")
        f.write("| Horizont | Stunden | Lasso R² | Lasso RMSE | ElasticNet R² | ElasticNet RMSE | GBR R² | GBR RMSE | XGBoost R² | XGBoost RMSE |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(f"| {r['Horizont']} | {r['Stunden']:.2f}h | {r['Lasso_R2']:.4f} | {r['Lasso_RMSE']:.2f}% | {r['ElasticNet_R2']:.4f} | {r['ElasticNet_RMSE']:.2f}% | {r['GBR_R2']:.4f} | {r['GBR_RMSE']:.2f}% | {r['XGBoost_R2']:.4f} | {r['XGBoost_RMSE']:.2f}% |\n")
            
        f.write("\n## 2. Interpretation & App-Integration (Lecture Match KI_03 & KI_04)\n")
        f.write("- **Lineare Baselines:** Dominanz bei sehr kurzen Horizonten (<= 2h) durch lineares Signal des unmittelbaren Lags.\n")
        f.write("- **Boosting-Modelle:** Ab 4h bricht die Kraft der Lags ein. GBR und XGBoost weisen beste Robustheit auf (R2 ~0.50 - 0.53) durch nicht-lineares Lernen.\n")
        f.write("- **Empfehlung:** Bei Horizonten <= 2h das Detailed-Modell nutzen; ab > 2h das Quick-Modell (XGBoost/GBR) verwenden.\n")
    print("\nErgebnisse gespeichert in: Model/reports/horizon/")

def main():
    csv_path = 'data/gym_workload_mlready.csv'
    horizons = {'5 Min': 1, '15 Min': 3, '30 Min': 6, '1 Std': 12, '2 Std': 24, '4 Std': 48, '6 Std': 72, '8 Std': 96}
    weekdays = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']
    base_features = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain', 'is_public_holiday', 'is_school_holiday', 'is_day'] + weekdays
    
    df = load_data(csv_path)
    results = run_experiment(df, horizons, base_features)
    res_df = pd.DataFrame(results)
    
    plot_results(res_df, horizons)
    generate_report(results)

if __name__ == '__main__':
    main()
