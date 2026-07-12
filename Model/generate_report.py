import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
RUN_MODE = sys.argv[1].lower() if len(sys.argv) > 1 and sys.argv[1].lower() in ['final', 'debug'] else 'final'
REPORT_BASE = f'Model/reports/{RUN_MODE}'
HORIZONS = [5, 15, 30, 60, 90, 120]
METRICS_FILE = f'{REPORT_BASE}/training_metrics.json'
REPORT_FILE = f'{REPORT_BASE}/training_report.md'

feature_labels = {
    'lag_inference': 'Auslastung vor 5 Minuten', 'lag_1d': 'Historischer Lag (24h alt)',
    'lag_7d_robust': 'Robustes Wochen-Lag (7d alt)',
    'temp': 'Temperatur', 'temp_diff': 'Temperaturdifferenz (24h)', 'humidity': 'Luftfeuchtigkeit', 'rain': 'Regenmenge',
    'sin_time': 'Tageszeit (Sinus)', 'cos_time': 'Tageszeit (Cosinus)',
    'is_public_holiday': 'Gesetzlicher Feiertag', 'is_school_holiday': 'Schulferien (kein Feiertag)', 'is_day': 'Tageslicht (Tag/Nacht)',
    'ema_1h_inference': 'Live-EMA (1h-Trend)', 'trend_2h_inference': 'Live-Trend (2h)', 'volatility_2h_inference': 'Live-Volatilität (2h)',
    'is_Monday': 'Montag', 'is_Tuesday': 'Dienstag', 'is_Wednesday': 'Mittwoch', 'is_Thursday': 'Donnerstag',
    'is_Friday': 'Freitag', 'is_Saturday': 'Samstag', 'is_Sunday': 'Sonntag'
}

# Methodische Begründung für XAI-Gruppierung (Erklärbarkeit/XAI):
# Um die kognitive Belastung bei der Interpretation der Feature-Wichtigkeit (MDI) und Regressionskoeffizienten zu reduzieren,
# werden die Features in semantische Blöcke (Lags, Wetter, Zeit & Kalender, Wochentage) unterteilt.
# Die Wochentage verbleiben statisch am unteren Ende (chronologisch sortiert), um ein einheitliches Bezugssystem für den Leser zu wahren.
# Die übrigen Gruppen werden dynamisch nach ihrer aggregierten Gesamt-Wichtigkeit über alle Horizonte hinweg sortiert.
# Dies verbessert die visuelle Hierarchie (wichtigste Konzepte oben) und erleichtert den wissenschaftlichen Modellvergleich.
def get_hybrid_order(features, values_for_sorting, is_abs=False):
    CATEGORIES = {
        'lag_inference': 'Lags',
        'lag_1d': 'Lags',
        'lag_7d_robust': 'Lags',
        'ema_1h_inference': 'Lags',
        'trend_2h_inference': 'Lags',
        'volatility_2h_inference': 'Lags',
        
        'temp': 'Wetter',
        'temp_diff': 'Wetter',
        'humidity': 'Wetter',
        'rain': 'Wetter',
        
        'sin_time': 'Zeit & Kalender',
        'cos_time': 'Zeit & Kalender',
        'is_day': 'Zeit & Kalender',
        'is_public_holiday': 'Zeit & Kalender',
        'is_school_holiday': 'Zeit & Kalender',
        
        'is_Monday': 'Wochentage',
        'is_Tuesday': 'Wochentage',
        'is_Wednesday': 'Wochentage',
        'is_Thursday': 'Wochentage',
        'is_Friday': 'Wochentage',
        'is_Saturday': 'Wochentage',
        'is_Sunday': 'Wochentage'
    }
    
    val_map = dict(zip(features, values_for_sorting))
    
    # Wochentage am Ende chronologisch beibehalten
    weekdays_list = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']
    wd_features = [f for f in weekdays_list if f in features]
    
    # Sortierung der übrigen Gruppen nach aggregierter Wichtigkeit
    categories_to_sort = ['Lags', 'Wetter', 'Zeit & Kalender']
    group_sums = {}
    for cat in categories_to_sort:
        cat_feats = [f for f in features if CATEGORIES.get(f) == cat]
        if is_abs:
            group_sum = sum(abs(val_map.get(f, 0.0)) for f in cat_feats)
        else:
            group_sum = sum(val_map.get(f, 0.0) for f in cat_feats)
        group_sums[cat] = group_sum
        
    sorted_categories = sorted(categories_to_sort, key=lambda c: group_sums[c], reverse=True)
    
    sorted_non_wd = []
    for cat in sorted_categories:
        cat_feats = [f for f in features if CATEGORIES.get(f) == cat]
        if is_abs:
            cat_feats_sorted = sorted(cat_feats, key=lambda f: abs(val_map.get(f, 0.0)), reverse=True)
        else:
            cat_feats_sorted = sorted(cat_feats, key=lambda f: val_map.get(f, 0.0), reverse=True)
        sorted_non_wd.extend(cat_feats_sorted)
        
    # Unkategorisierte Features (Fallback)
    all_known = set(CATEGORIES.keys())
    others = [f for f in features if f not in all_known]
    if others:
        if is_abs:
            others_sorted = sorted(others, key=lambda f: abs(val_map.get(f, 0.0)), reverse=True)
        else:
            others_sorted = sorted(others, key=lambda f: val_map.get(f, 0.0), reverse=True)
        sorted_non_wd = others_sorted + sorted_non_wd
        
    return sorted_non_wd + wd_features

def plot_fixed_xai(title, filename, vals, features, fixed_order, is_coef, mins=None):
    sns.set_theme(style="whitegrid")
    val_dict = dict(zip(features, vals))
    ordered_vals = [val_dict.get(f, 0.0) for f in fixed_order]
    
    lag_name = f'lag_{mins // 5}' if mins is not None else 'lag_1'
    ordered_labels = [f.replace('lag_inference', lag_name).replace('_inference', '') for f in fixed_order]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=ordered_vals, y=ordered_labels, hue=ordered_labels, palette="coolwarm" if is_coef else "viridis", legend=False)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Koeffizient' if is_coef else 'Wichtigkeit (Gini/MDI)', fontsize=12)
    if is_coef:
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close()

def generate_all_xai_plots(data):
    print("[INFO] Generiere XAI-Plots mit fixierter Y-Achse...")
    weekdays = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']
    detailed_features = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain', 'lag_inference', 'ema_1h_inference', 'trend_2h_inference', 'volatility_2h_inference', 'lag_1d', 'lag_7d_robust'] + ['is_public_holiday', 'is_school_holiday', 'is_day'] + weekdays
    quick_features = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain'] + ['is_public_holiday', 'is_school_holiday', 'is_day'] + weekdays

    # Quick Plots
    q_data = data.get('Quick', {})
    if q_data:
        l_name = 'ElasticNet'
        t_name = 'XGBoost'
        
        coefs = q_data[l_name].get('coefficients')
        imps = q_data[t_name].get('feature_importances')
        
        if imps:
            # Einheitliche Y-Achsen-Sortierung basierend auf der Feature Importance (XGBoost)
            ordered_feats = get_hybrid_order(quick_features, imps, is_abs=False)
            
            # Feature Importance Plots
            plot_fixed_xai(f'Feature Importance ({t_name}) - Quick', f'{REPORT_BASE}/xai/quick/feature_importance_quick.png', imps, quick_features, ordered_feats, False)
            plot_fixed_xai(f'Feature Importance ({t_name}) - Quick', f'{REPORT_BASE}/xai/analysis/feature_importance_quick.png', imps, quick_features, ordered_feats, False)
            
            # Koeffizienten Plots mit exakt derselben Y-Achsen-Sortierung
            if coefs:
                plot_fixed_xai(f'Koeffizienten ({l_name}) - Quick', f'{REPORT_BASE}/xai/quick/coefficients_quick.png', coefs, quick_features, ordered_feats, True)
                plot_fixed_xai(f'Koeffizienten ({l_name}) - Quick', f'{REPORT_BASE}/xai/analysis/coefficients_quick.png', coefs, quick_features, ordered_feats, True)

    # Detailed Plots
    det_data = data.get('Detailed', {})
    if det_data:
        all_imps, all_coefs, cache = [], [], {}
        for mins in HORIZONS:
            h_data = det_data.get(str(mins), {})
            l_name = 'ElasticNet'
            t_name = 'XGBoost'
            
            coefs_raw = h_data[l_name].get('coefficients')
            feats_lin = h_data[l_name].get('transformed_features', [])
            if coefs_raw and feats_lin:
                feat_to_coef = dict(zip(feats_lin, coefs_raw))
                coefs = [feat_to_coef.get(f, 0.0) for f in detailed_features]
            else:
                coefs = None
                
            imps_raw = h_data[t_name].get('feature_importances')
            feats_tree = h_data[t_name].get('transformed_features', [])
            if imps_raw and feats_tree:
                feat_to_imp = dict(zip(feats_tree, imps_raw))
                imps = [feat_to_imp.get(f, 0.0) for f in detailed_features]
            else:
                imps = None
                
            if coefs: all_coefs.append(np.abs(coefs))
            if imps: all_imps.append(imps)
            
            cache[mins] = {'coefs': coefs, 'imps': imps, 'l_name': l_name, 't_name': t_name}
            
        mean_imps = np.mean(all_imps, axis=0) if all_imps else []
        fixed_tree = get_hybrid_order(detailed_features, mean_imps, is_abs=False) if all_imps else []
        
        # Einheitliche Y-Achsen-Sortierung basierend auf der Feature Importance (XGBoost)
        fixed_lin = fixed_tree
        
        # 1. Generate individual plots for folder backup
        for mins in HORIZONS:
            c = cache[mins]
            if c['imps']:
                plot_fixed_xai(f'Feature Importance ({c["t_name"]}) - {mins}m', f'{REPORT_BASE}/xai/detailed/feature_importance_detailed_{mins}m.png', c['imps'], detailed_features, fixed_tree, False, mins=mins)
            if c['coefs']:
                plot_fixed_xai(f'Koeffizienten ({c["l_name"]}) - {mins}m', f'{REPORT_BASE}/xai/detailed/coefficients_detailed_{mins}m.png', c['coefs'], detailed_features, fixed_lin, True, mins=mins)

        # 2. Generate 2x3 Grid for Feature Importance (Detailed)
        fig_tree, axes_tree = plt.subplots(2, 3, figsize=(20, 12))
        axes_tree = axes_tree.flatten()
        for h_idx, mins in enumerate(HORIZONS):
            ax = axes_tree[h_idx]
            c = cache[mins]
            if c['imps']:
                val_dict = dict(zip(detailed_features, c['imps']))
                ordered_vals = [val_dict.get(f, 0.0) for f in fixed_tree]
                ordered_labels = [f.replace('lag_inference', f'lag_{mins // 5}').replace('_inference', '') for f in fixed_tree]
                
                sns.barplot(x=ordered_vals, y=ordered_labels, hue=ordered_labels, palette="viridis", legend=False, ax=ax)
                ax.set_title(f'Horizont: {mins} Minuten', fontsize=11, fontweight='bold')
                ax.set_xlabel('Wichtigkeit (Gini/MDI)', fontsize=9)
                ax.tick_params(axis='y', labelsize=8.5)
                ax.tick_params(axis='x', labelsize=8.5)
                
        fig_tree.suptitle('Feature Importance (Detailed - XGBoost)', fontsize=15, fontweight='bold', y=0.98)
        fig_tree.tight_layout()
        fig_tree.subplots_adjust(top=0.91)
        os.makedirs(f'{REPORT_BASE}/xai/detailed', exist_ok=True)
        plt.savefig(f'{REPORT_BASE}/xai/detailed/feature_importance_detailed_grid.png', dpi=300, transparent=True)
        plt.close()

        # 3. Generate 2x3 Grid for Coefficients (Detailed)
        fig_lin, axes_lin = plt.subplots(2, 3, figsize=(20, 12))
        axes_lin = axes_lin.flatten()
        for h_idx, mins in enumerate(HORIZONS):
            ax = axes_lin[h_idx]
            c = cache[mins]
            if c['coefs']:
                val_dict = dict(zip(detailed_features, c['coefs']))
                ordered_vals = [val_dict.get(f, 0.0) for f in fixed_lin]
                ordered_labels = [f.replace('lag_inference', f'lag_{mins // 5}').replace('_inference', '') for f in fixed_lin]
                
                sns.barplot(x=ordered_vals, y=ordered_labels, hue=ordered_labels, palette="coolwarm", legend=False, ax=ax)
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.set_title(f'Horizont: {mins} Minuten', fontsize=11, fontweight='bold')
                ax.set_xlabel('Regressions-Koeffizient', fontsize=9)
                ax.tick_params(axis='y', labelsize=8.5)
                ax.tick_params(axis='x', labelsize=8.5)
                
        fig_lin.suptitle('Koeffizientenverlauf (Detailed - ElasticNet)', fontsize=15, fontweight='bold', y=0.98)
        fig_lin.tight_layout()
        fig_lin.subplots_adjust(top=0.91)
        plt.savefig(f'{REPORT_BASE}/xai/detailed/coefficients_detailed_grid.png', dpi=300, transparent=True)
        plt.close()

def generate_additional_plots(data):
    print("[INFO] Generiere übergeordnete Vergleichsplots...")
    det_data = data.get('Detailed', {})
    if not det_data: return

    models = {'RandomForest': 'Random Forest', 'GradientBoosting': 'Gradient Boosting (GBR)', 'XGBoost': 'XGBoost', 'Ridge': 'Ridge (L2)', 'Lasso': 'Lasso (L1)', 'ElasticNet': 'ElasticNet (L1+L2)'}
    r2_list, rmse_list, mae_list, coef_data = [], [], [], []

    for mins in HORIZONS:
        h_data = det_data.get(str(mins), {})
        r2_row, rmse_row, mae_row = {'Horizont': mins}, {'Horizont': mins}, {'Horizont': mins}
        for name, label in models.items():
            metrics = h_data.get(name, {})
            if metrics:
                r2_row[label] = metrics.get('R2', 0.0)
                rmse_row[label] = metrics.get('RMSE', 0.0)
                mae_row[label] = metrics.get('MAE', 0.0)
                if name == 'ElasticNet' and metrics.get('coefficients'):
                    coef_row = {'Horizont': mins}
                    for f, v in zip(metrics.get('transformed_features', []), metrics.get('coefficients', [])):
                        coef_row[f] = v
                    coef_data.append(coef_row)
        r2_list.append(r2_row)
        rmse_list.append(rmse_row)
        mae_list.append(mae_row)

    colors = {'Random Forest': 'crimson', 'Gradient Boosting (GBR)': 'forestgreen', 'XGBoost': 'purple', 'Ridge (L2)': 'darkorange', 'Lasso (L1)': 'royalblue', 'ElasticNet (L1+L2)': 'teal'}
    markers = {'Random Forest': 'o', 'Gradient Boosting (GBR)': 'v', 'XGBoost': 'x', 'Ridge (L2)': '^', 'Lasso (L1)': '<', 'ElasticNet (L1+L2)': '>'}
    sns.set_theme(style="whitegrid")

    # Plot 1: R2
    if r2_list:
        plt.figure(figsize=(10, 6))
        df_r2 = pd.DataFrame(r2_list)
        for label in colors:
            if label in df_r2.columns:
                sns.lineplot(data=df_r2, x='Horizont', y=label, marker=markers[label], label=label, color=colors[label], linewidth=2.0)
        plt.title('Modell-Vergleich (R², Detailed)', fontsize=14, fontweight='bold')
        plt.xlabel('Vorhersagehorizont (Minuten)')
        plt.ylabel('Test R²-Score')
        plt.ylim(-0.1, 1.05)
        plt.xticks(HORIZONS)
        plt.tight_layout()
        os.makedirs(f'{REPORT_BASE}/horizon', exist_ok=True)
        plt.savefig(f'{REPORT_BASE}/horizon/model_comparison_detailed.png', dpi=300, transparent=True)
        plt.close()

    # Plot 1b: RMSE & MAE
    if rmse_list:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        df_rmse, df_mae = pd.DataFrame(rmse_list), pd.DataFrame(mae_list)
        for label in colors:
            if label in df_rmse.columns:
                sns.lineplot(data=df_rmse, x='Horizont', y=label, marker=markers[label], label=label, color=colors[label], ax=axes[0])
            if label in df_mae.columns:
                sns.lineplot(data=df_mae, x='Horizont', y=label, marker=markers[label], label=label, color=colors[label], ax=axes[1])
        axes[0].set_title('RMSE', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('RMSE [%] (niedriger ist besser)', fontsize=10)
        axes[0].set_xticks(HORIZONS)
        axes[1].set_title('MAE', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAE [%] (niedriger ist besser)', fontsize=10)
        axes[1].set_xticks(HORIZONS)
        plt.suptitle('Modell-Fehlervergleich (Detailed)', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig(f'{REPORT_BASE}/horizon/model_error_comparison_detailed.png', dpi=300, transparent=True)
        plt.close()

    # Plot 2: ElasticNet Gewichte
    if coef_data:
        plt.figure(figsize=(10, 6))
        df_coef = pd.DataFrame(coef_data)
        feats = ['lag_inference', 'lag_1d', 'lag_7d_robust', 'temp', 'cos_time', 'is_public_holiday', 'is_school_holiday']
        labels = {
            'lag_inference': 'lag_1',
            'lag_1d': 'lag_1d',
            'lag_7d_robust': 'lag_7d_robust',
            'temp': 'temp',
            'cos_time': 'cos_time',
            'is_public_holiday': 'is_public_holiday',
            'is_school_holiday': 'is_school_holiday'
        }
        f_colors = {
            'lag_inference': 'crimson',          # Live-Lags (Kurzzeit) - rot
            'lag_1d': 'royalblue',              # Historische Lags (24h / 7d) - blau
            'lag_7d_robust': 'mediumblue',       # Historische Lags (24h / 7d) - dunkelblau
            'temp': 'darkorange',               # Wetter - orange
            'cos_time': 'forestgreen',          # Zeit & Kalender - gruen
            'is_public_holiday': 'limegreen',   # Zeit & Kalender - hellgruen
            'is_school_holiday': 'yellowgreen'  # Zeit & Kalender - gelbgruen
        }
        f_markers = {'lag_inference': 'o', 'lag_1d': 's', 'lag_7d_robust': 'D', 'temp': '^', 'cos_time': 'd', 'is_public_holiday': 'x', 'is_school_holiday': 'v'}
        
        for f in feats:
            if f in df_coef.columns:
                sns.lineplot(data=df_coef, x='Horizont', y=f, marker=f_markers[f], label=labels[f], color=f_colors[f], linewidth=2.5)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Koeffizientenverlauf über Zeithorizonte (ElasticNet)', fontsize=14, fontweight='bold')
        plt.xlabel('Vorhersagehorizont (Minuten)')
        plt.ylabel('Gewicht')
        plt.xticks(HORIZONS)
        plt.tight_layout()
        os.makedirs(f'{REPORT_BASE}/xai/analysis', exist_ok=True)
        plt.savefig(f'{REPORT_BASE}/xai/analysis/coefficients_detailed.png', dpi=300, transparent=True)
        plt.close()

def generate_fold_performance_plots(data):
    print("[INFO] Generiere Fold-Performance-Plots...")
    q_data = data.get('Quick', {})
    det_data = data.get('Detailed', {})
    
    models = {
        'RandomForest': 'Random Forest', 
        'GradientBoosting': 'Gradient Boosting', 
        'XGBoost': 'XGBoost', 
        'Ridge': 'Ridge', 
        'Lasso': 'Lasso', 
        'ElasticNet': 'ElasticNet'
    }
    
    colors = {
        'Random Forest': 'crimson', 
        'Gradient Boosting': 'forestgreen', 
        'XGBoost': 'purple', 
        'Ridge': 'darkorange', 
        'Lasso': 'royalblue', 
        'ElasticNet': 'teal'
    }
    markers = {
        'Random Forest': 'o', 
        'Gradient Boosting': 'v', 
        'XGBoost': 'x', 
        'Ridge': '^', 
        'Lasso': '<', 
        'ElasticNet': '>'
    }
    
    sns.set_theme(style="whitegrid")
    
    # ---------------- 1. QUICK MODEL PLOTS ----------------
    if q_data:
        plot_data = []
        for model_name, label in models.items():
            m_info = q_data.get(model_name, {})
            fold_metrics = m_info.get('fold_metrics', [])
            for f_metric in fold_metrics:
                plot_data.append({
                    'Modell': label,
                    'Fold': f'Fold {f_metric["fold"]}',
                    'R2': f_metric['R2'],
                    'RMSE': f_metric['RMSE']
                })
        
        if plot_data:
            df_q = pd.DataFrame(plot_data)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot R2
            for label in colors:
                if label in df_q['Modell'].values:
                    df_m = df_q[df_q['Modell'] == label]
                    sns.lineplot(data=df_m, x='Fold', y='R2', marker=markers[label], label=label, color=colors[label], linewidth=2.0, ax=ax1)
            ax1.set_title('Bestimmtheitsmaß ($R^2$) pro Fold', fontsize=12, fontweight='bold')
            ax1.set_ylabel('$R^2$-Score')
            ax1.set_ylim(-0.1, 1.05)
            ax1.legend(loc='upper right')
            
            # Plot RMSE
            for label in colors:
                if label in df_q['Modell'].values:
                    df_m = df_q[df_q['Modell'] == label]
                    sns.lineplot(data=df_m, x='Fold', y='RMSE', marker=markers[label], label=label, color=colors[label], linewidth=2.0, ax=ax2)
            ax2.set_title('Vorhersagefehler (RMSE) pro Fold', fontsize=12, fontweight='bold')
            ax2.set_ylabel('RMSE [%] (niedriger ist besser)')
            ax2.legend(loc='upper right')
            
            plt.suptitle('Stabilitätsanalyse (Quick)', fontsize=15, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            os.makedirs(f'{REPORT_BASE}/horizon', exist_ok=True)
            plt.savefig(f'{REPORT_BASE}/horizon/fold_performance_comparison.png', dpi=300, transparent=True)
            plt.close()
            print(f"[OK] Quick Fold-Performance-Plot gespeichert.")

    # ---------------- 2. DETAILED MULTI-HORIZON PLOTS ----------------
    if det_data:
        plot_data_det = []
        for mins_str, res in det_data.items():
            mins = int(mins_str)
            for model_name, label in models.items():
                m_info = res.get(model_name, {})
                fold_metrics = m_info.get('fold_metrics', [])
                for f_metric in fold_metrics:
                    plot_data_det.append({
                        'Horizont': mins,
                        'Modell': label,
                        'Fold': f'Fold {f_metric["fold"]}',
                        'R2': f_metric['R2'],
                        'RMSE': f_metric['RMSE']
                    })
        
        if plot_data_det:
            df_det = pd.DataFrame(plot_data_det)
            horizons_sorted = sorted(df_det['Horizont'].unique())
            
            # --- R2 Grid (2x3) ---
            fig_r2, axes_r2 = plt.subplots(2, 3, figsize=(18, 10))
            axes_r2 = axes_r2.flatten()
            
            for h_idx, mins in enumerate(horizons_sorted):
                ax = axes_r2[h_idx]
                df_h = df_det[df_det['Horizont'] == mins]
                
                for label in colors:
                    if label in df_h['Modell'].values:
                        df_m = df_h[df_h['Modell'] == label]
                        sns.lineplot(data=df_m, x='Fold', y='R2', marker=markers[label], label=label, color=colors[label], linewidth=1.8, ax=ax)
                
                ax.set_title(f'Horizont: {mins} Minuten', fontsize=12, fontweight='bold')
                ax.set_ylabel('$R^2$-Score')
                ax.set_ylim(-0.15, 1.05)
                ax.legend().set_visible(False)
            
            # Gemeinsame Legende oben (ohne Überlappung dank subplots_adjust)
            handles, labels = axes_r2[0].get_legend_handles_labels()
            fig_r2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6, frameon=True, fontsize=11)
            
            fig_r2.suptitle('Stabilitätsanalyse ($R^2$, Detailed)', fontsize=16, fontweight='bold', y=0.98)
            fig_r2.tight_layout()
            fig_r2.subplots_adjust(top=0.865)
            plt.savefig(f'{REPORT_BASE}/horizon/fold_performance_detailed_r2.png', dpi=300, transparent=True)
            plt.close()
            print(f"[OK] Detailed Fold-Performance R²-Grid (2x3) gespeichert.")
            
            # --- RMSE Grid (2x3) ---
            fig_rmse, axes_rmse = plt.subplots(2, 3, figsize=(18, 10))
            axes_rmse = axes_rmse.flatten()
            
            for h_idx, mins in enumerate(horizons_sorted):
                ax = axes_rmse[h_idx]
                df_h = df_det[df_det['Horizont'] == mins]
                
                for label in colors:
                    if label in df_h['Modell'].values:
                        df_m = df_h[df_h['Modell'] == label]
                        sns.lineplot(data=df_m, x='Fold', y='RMSE', marker=markers[label], label=label, color=colors[label], linewidth=1.8, ax=ax)
                
                ax.set_title(f'Horizont: {mins} Minuten', fontsize=12, fontweight='bold')
                ax.set_ylabel('RMSE [%] (niedriger ist besser)')
                ax.legend().set_visible(False)
            
            # Gemeinsame Legende oben (ohne Überlappung dank subplots_adjust)
            handles, labels = axes_rmse[0].get_legend_handles_labels()
            fig_rmse.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6, frameon=True, fontsize=11)
            
            fig_rmse.suptitle('Stabilitätsanalyse (RMSE, Detailed)', fontsize=16, fontweight='bold', y=0.98)
            fig_rmse.tight_layout()
            fig_rmse.subplots_adjust(top=0.865)
            plt.savefig(f'{REPORT_BASE}/horizon/fold_performance_detailed_rmse.png', dpi=300, transparent=True)
            plt.close()
            print(f"[OK] Detailed Fold-Performance RMSE-Grid (2x3) gespeichert.")

def generate_xai_decay(data):
    print("[INFO] Generiere XAI-Decay-Plot...")
    det_data = data.get('Detailed', {})
    if not det_data: return

    weekdays = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']
    groups = {
        'Live-Lags (Kurzzeit)': ['lag_inference', 'ema_1h_inference', 'trend_2h_inference', 'volatility_2h_inference'],
        'Historische Lags (24h / 7d)': ['lag_1d', 'lag_7d_robust'],
        'Zeit & Kalender': ['sin_time', 'cos_time', 'is_public_holiday', 'is_school_holiday', 'is_day'] + weekdays,
        'Wetter': ['temp', 'temp_diff', 'humidity', 'rain']
    }

    decay_list = []
    for mins in HORIZONS:
        xgb = det_data.get(str(mins), {}).get('XGBoost', {})
        imps, feats = xgb.get('feature_importances'), xgb.get('transformed_features', [])
        if imps and feats:
            g_vals = {g: 0.0 for g in groups}
            for f, imp in zip(feats, imps):
                for g_name, g_feats in groups.items():
                    if f in g_feats:
                        g_vals[g_name] += imp
                        break
            row = {'Horizont (Minuten)': mins}
            row.update(g_vals)
            decay_list.append(row)

    if decay_list:
        df_dec = pd.DataFrame(decay_list)
        plt.figure(figsize=(10, 6))
        colors = {'Live-Lags (Kurzzeit)': 'crimson', 'Historische Lags (24h / 7d)': 'royalblue', 'Zeit & Kalender': 'forestgreen', 'Wetter': 'darkorange'}
        markers = {'Live-Lags (Kurzzeit)': 'o', 'Historische Lags (24h / 7d)': 's', 'Zeit & Kalender': 'd', 'Wetter': '^'}
        for g in groups:
            sns.lineplot(data=df_dec, x='Horizont (Minuten)', y=g, marker=markers[g], label=g, color=colors[g], linewidth=2.5)
        plt.title('Bedeutungswandel der Feature-Gruppen (XGBoost)', fontsize=14, fontweight='bold')
        plt.xlabel('Vorhersagehorizont (Minuten)')
        plt.ylabel('Kumulierte Feature-Wichtigkeit')
        plt.ylim(-0.05, 1.05)
        plt.xticks(HORIZONS)
        plt.tight_layout()
        plt.savefig(f'{REPORT_BASE}/xai/analysis/feature_importance_detailed.png', dpi=300, transparent=True)
        plt.close()

def generate_grid_search_plots(gs_data):
    print("[INFO] Generiere Grid Search Visualisierungen...")
    quick_gs = gs_data.get('Quick', {})
    detailed_gs = gs_data.get('Detailed', {})
    
    colors = {
        'RandomForest': 'crimson',
        'GradientBoosting': 'forestgreen',
        'XGBoost': 'purple',
        'Ridge': 'darkorange',
        'Lasso': 'royalblue',
        'ElasticNet': 'teal'
    }
    
    # Mapping for shorter labels on boxplot x-axis to prevent overlap in the 2x3 grid
    short_labels = {
        'RandomForest': 'RF',
        'GradientBoosting': 'GB',
        'XGBoost': 'XGB',
        'Ridge': 'Ridge',
        'Lasso': 'Lasso',
        'ElasticNet': 'EN'
    }
    
    sns.set_theme(style="whitegrid")
    os.makedirs(f'{REPORT_BASE}/horizon', exist_ok=True)

    # 1. QUICK MODUS PLOTS
    if quick_gs:
        # 1a. Plot: Effizienz (RMSE vs. Trainingszeit) - Quick
        plt.figure(figsize=(10, 6))
        plotted_models = 0
        for model_name, runs in quick_gs.items():
            if not runs:
                continue
            times = [r['mean_fit_time'] for r in runs]
            rmses = [np.sqrt(-r['mean_test_score']) for r in runs]
            
            sns.scatterplot(
                x=times, y=rmses, 
                label=model_name, 
                color=colors.get(model_name, 'black'),
                s=80, alpha=0.7, edgecolor='w', linewidth=1.5
            )
            plotted_models += 1
            
        if plotted_models > 0:
            plt.title('Tuning-Effizienz (Quick)', fontsize=14, fontweight='bold')
            plt.xlabel('Mittlere Trainingszeit pro Split (Sekunden)', fontsize=12)
            plt.ylabel('Validierungs-RMSE [%] (niedriger ist besser)', fontsize=12)
            plt.xscale('log')
            plt.legend(title="Modelltyp", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{REPORT_BASE}/horizon/grid_search_efficiency.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close()

        # 1b. Plot: Tuning Impact (Boxplot der RMSE-Verteilung) - Quick
        plt.figure(figsize=(10, 6))
        boxplot_data = []
        for model_name, runs in quick_gs.items():
            if not runs:
                continue
            for r in runs:
                boxplot_data.append({
                    'Modell': model_name,
                    'RMSE': np.sqrt(-r['mean_test_score'])
                })
                
        if boxplot_data:
            df_box = pd.DataFrame(boxplot_data)
            order = df_box.groupby('Modell')['RMSE'].min().sort_values().index
            
            sns.boxplot(
                data=df_box, x='Modell', y='RMSE', 
                order=order, palette="Set2", hue='Modell', legend=False
            )
            sns.stripplot(
                data=df_box, x='Modell', y='RMSE', 
                order=order, color='black', alpha=0.5, size=4, jitter=0.1
            )
            
            plt.title('Tuning-Sensitivität (Quick)', fontsize=14, fontweight='bold')
            plt.xlabel('Modelltyp (sortiert nach bester Konfiguration)', fontsize=12)
            plt.ylabel('Validierungs-RMSE [%] (niedriger ist besser)', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{REPORT_BASE}/horizon/grid_search_tuning_impact.png', dpi=300, transparent=True)
            plt.close()

    # 2. DETAILED MODUS PLOTS (2x3 Gitter für alle 6 Horizonte)
    if detailed_gs:
        horizons_sorted = sorted([int(k) for k in detailed_gs.keys()])
        
        # 2a. Plot: 2x3 Grid für Effizienz (RMSE vs. Trainingszeit)
        fig_eff, axes_eff = plt.subplots(2, 3, figsize=(18, 10))
        axes_eff = axes_eff.flatten()
        
        for h_idx, mins in enumerate(horizons_sorted):
            ax = axes_eff[h_idx]
            models_dict = detailed_gs.get(str(mins), {})
            
            for model_name, runs in models_dict.items():
                if model_name not in colors or not runs:
                    continue
                times = [r['mean_fit_time'] for r in runs]
                rmses = [np.sqrt(-r['mean_test_score']) for r in runs]
                
                ax.scatter(
                    times, rmses, 
                    label=model_name, 
                    color=colors[model_name],
                    s=50, alpha=0.6, edgecolor='w', linewidth=0.8
                )
                
            ax.set_title(f'Horizont: {mins} Minuten', fontsize=12, fontweight='bold')
            ax.set_xlabel('Trainingszeit (s)', fontsize=10)
            ax.set_ylabel('RMSE [%] (niedriger ist besser)', fontsize=10)
            ax.set_xscale('log')
            
        # Legend at the top (centered, no overlap, top=0.865)
        handles, labels = axes_eff[0].get_legend_handles_labels()
        if handles:
            fig_eff.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6, frameon=True, fontsize=11)
            
        fig_eff.suptitle('Tuning-Effizienz (Detailed)', fontsize=16, fontweight='bold', y=0.98)
        fig_eff.tight_layout()
        fig_eff.subplots_adjust(top=0.865)
        plt.savefig(f'{REPORT_BASE}/horizon/grid_search_efficiency_detailed.png', dpi=300, transparent=True)
        plt.close()

        # 2b. Plot: 2x3 Grid für Tuning Impact (Boxplots)
        fig_box, axes_box = plt.subplots(2, 3, figsize=(18, 10))
        axes_box = axes_box.flatten()
        
        for h_idx, mins in enumerate(horizons_sorted):
            ax = axes_box[h_idx]
            models_dict = detailed_gs.get(str(mins), {})
            
            boxplot_data_h = []
            for model_name, runs in models_dict.items():
                if model_name not in colors or not runs:
                    continue
                for r in runs:
                    boxplot_data_h.append({
                        'Modell': short_labels[model_name],
                        'RMSE': np.sqrt(-r['mean_test_score'])
                    })
            
            if boxplot_data_h:
                df_box_h = pd.DataFrame(boxplot_data_h)
                order_h = df_box_h.groupby('Modell')['RMSE'].min().sort_values().index
                
                sns.boxplot(
                    data=df_box_h, x='Modell', y='RMSE', 
                    order=order_h, palette="Set2", hue='Modell', legend=False, ax=ax
                )
                sns.stripplot(
                    data=df_box_h, x='Modell', y='RMSE', 
                    order=order_h, color='black', alpha=0.4, size=3, jitter=0.12, ax=ax
                )
                
            ax.set_title(f'Horizont: {mins} Minuten', fontsize=12, fontweight='bold')
            ax.set_xlabel('Modelltyp', fontsize=10)
            ax.set_ylabel('RMSE [%] (niedriger ist besser)', fontsize=10)
            
        fig_box.suptitle('Tuning-Sensitivität (Detailed)', fontsize=16, fontweight='bold', y=0.98)
        fig_box.tight_layout()
        fig_box.subplots_adjust(top=0.89)
        plt.savefig(f'{REPORT_BASE}/horizon/grid_search_tuning_impact_detailed.png', dpi=300, transparent=True)
        plt.close()

def generate_markdown_report(data):
    meta = data.get('meta', {})
    res_q, res_det = data.get('Quick', {}), data.get('Detailed', {})
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Wissenschaftlicher Modell- und Evaluierungsbericht\n\n")
        
        if meta:
            f.write("### 1.1 Trainings-Statistiken & Laufzeiten\n")
            tuning_str = "Final Tuning (Produktion)" if meta.get('final_tuning_mode') else "Debug Tuning (Schnelllauf)"
            f.write(f"- **Tuning-Modus:** {tuning_str}\n")
            f.write(f"- **Kreuzvalidierung:** {meta.get('cv_splits')}-fold TimeSeriesSplit\n")
            f.write(f"- **Gesamte Trainingsläufe:** {meta.get('total_fit_runs')} Modell-Fits (inkl. Kreuzvalidierungs-Splits & finalem Refit)\n")
            f.write(f"- **Gesamtlaufzeit des Trainings:** {meta.get('total_training_time_min')} Minuten\n\n")
            
        f.write("## 1. Methodischer Versuchsaufbau & Vorlesungsbezug\n")
        f.write("- **Data Leakage Prevention (KI_03):** Die Standardisierung mittels `StandardScaler` wurde strikt nach dem Train-Test-Split durchgeführt. Fit nur auf Train-Daten. Binäre Spalten blieben unskaliert.\n")
        f.write("- **TimeSeriesSplit (KI_03):** Sequentielles Validierungsverfahren zur Wahrung der Zeitkausalität.\n")
        f.write("- **Multi-Modell-Architektur (KI_03 & KI_04):** Da sich die Aussagekraft von historischen Lags über längere Vorhersagehorizonte verschiebt (Degradation), wurde für jeden Zeithorizont ein eigenständiges Modell trainiert.\n")
        f.write("- **Modellspeicherung:** Gespeichert in getrennten Unterordnern unter `Model/models/final/` (Quick und Detailed 5m bis 120m) für maximale Übersicht.\n")
        f.write("- **1-Jahres-Fenster (Sliding Window):** Um Concept Drift zu minimieren und Skalierbarkeit zu sichern, wurde das Training auf die Daten der letzten 365 Tage beschränkt.\n\n")
        
        # Quick
        f.write("## 2. Ergebnisse für Modus: Quick (Tagesplaner)\n")
        f.write("| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |\n| :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for n, m in res_q.items():
            f.write(f"| {n} | {m['R2']:.4f} | {m['RMSE']:.4f} | {m['MSE']:.4f} | {m['MAE']:.4f} | `{m['best_params']}` |\n")
        best_q = max(res_q, key=lambda k: res_q[k]['R2'])
        f.write(f"\n**Bestes Modell für Quick:** {best_q} ($R^2 = {res_q[best_q]['R2']:.4f}$).\n\n")
        f.write("### Stabilitätsanalyse über die Kreuzvalidierungs-Folds\n")
        f.write("Die folgende Grafik zeigt die Leistungsschwankungen der Ostele-Modelle über die 5 chronologischen Folds des TimeSeriesSplits. Dies dient der Validierung der zeitlichen Stabilität der Vorhersagen:\n\n")
        f.write("![Fold-Performance-Vergleich](horizon/fold_performance_comparison.png)\n\n")
        f.write("### XAI-Analysen (Quick)\n")
        f.write("| Feature Importance (Baumbasiert) | Koeffizienten (Linear) |\n| :---: | :---: |\n| ![Feature Importance (Baumbasiert)](xai/quick/feature_importance_quick.png) | ![Koeffizienten (Linear)](xai/quick/coefficients_quick.png) |\n\n")
        
        # Detailed
        f.write("## 3. Ergebnisse für Modus: Detailed (Live-Tracker)\nHier wurden die Modelle auf unterschiedliche Vorhersagehorizonte (5m bis 120m) trainiert. Man sieht die Degradation.\n\n")
        f.write("### Stabilitätsanalyse der Detailed-Modelle über die Folds\n")
        f.write("Die folgenden Gitter-Grafiken vergleichen die Leistungsschwankungen aller 6 detailed Modelle über die 5 chronologischen Folds für alle sechs Zeithorizonte:\n\n")
        f.write("| Stabilität R²-Score | Stabilität Vorhersagefehler (RMSE) |\n| :---: | :---: |\n| ![Fold-Stabilität R² Detailed](horizon/fold_performance_detailed_r2.png) | ![Fold-Stabilität RMSE Detailed](horizon/fold_performance_detailed_rmse.png) |\n\n")
        
        f.write("### Übergreifende XAI-Analysen (Detailed)\n")
        f.write("Die folgenden Gitter-Grafiken vergleichen die Feature-Importances (Baumbasiert) und Regressions-Gewichte (Linear) über alle sechs Horizonte hinweg:\n\n")
        f.write("| Feature Importance (Baumbasiert) | Koeffizienten (Linear) |\n")
        f.write("| :---: | :---: |\n")
        f.write("| ![Feature Importance Gitter](xai/detailed/feature_importance_detailed_grid.png) | ![Koeffizienten Gitter](xai/detailed/coefficients_detailed_grid.png) |\n\n")

        for mins in sorted([int(k) for k in res_det.keys()]):
            res = res_det[str(mins)]
            f.write(f"### 3.{mins // 5} Zeithorizont: {mins} Minuten\n")
            f.write("| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |\n| :--- | :---: | :---: | :---: | :---: | :--- |\n")
            for n, m in res.items():
                f.write(f"| {n} | {m['R2']:.4f} | {m['RMSE']:.4f} | {m['MSE']:.4f} | {m['MAE']:.4f} | `{m['best_params']}` |\n")
            best_h = max(res, key=lambda k: res[k]['R2'])
            f.write(f"\n**Bestes Modell für {mins}m:** {best_h} ($R^2 = {res[best_h]['R2']:.4f}$).\n\n")

        # Uebergreifend
        f.write("## 4. Übergreifende Analysen & Erkenntnisse (Vorlesungsbezug KI_03)\n\n### 4.1 Modell-Wettstreit über alle Horizonte (R²-Score & absolute Fehler)\n")
        f.write("| Modell-Vergleich (R²) | Modell-Fehlervergleich (RMSE/MAE) |\n| :---: | :---: |\n| ![Modell-Vergleich (R²)](horizon/model_comparison_detailed.png) | ![Modell-Fehlervergleich (RMSE/MAE)](horizon/model_error_comparison_detailed.png) |\n\n")
        f.write("### 4.2 Wandel der Feature-Wichtigkeit (XAI Decay) & Regressionsgewichte\n")
        f.write("| Wandel der Feature-Wichtigkeit (XAI Decay) | Koeffizienten-Verlauf (ElasticNet) |\n| :---: | :---: |\n| ![Feature-Importance-Decay](xai/analysis/feature_importance_detailed.png) | ![Koeffizienten-Verlauf](xai/analysis/coefficients_detailed.png) |\n\n")
        
        # Grid Search
        f.write("## 5. Hyperparameter-Tuning & Effizienzanalyse (Vorlesungsbezug KI_03)\n\n")
        f.write("Um die optimale Konfiguration der Modelle zu finden, wurde eine systematische Grid-Search über einen definiertem Suchraum durchgeführt. Die folgenden Grafiken analysieren die Robustheit und Effizienz der Modelle in beiden Modi:\n\n")
        f.write("| Modus | Tuning-Effizienz (Genauigkeit vs. Laufzeit) | Tuning-Sensitivität (RMSE-Spannweite) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write("| **Quick** | ![Tuning-Effizienz Quick](horizon/grid_search_efficiency.png) | ![Tuning-Sensitivität Quick](horizon/grid_search_tuning_impact.png) |\n")
        f.write("| **Detailed (2x3 Gitter)** | ![Tuning-Effizienz Detailed](horizon/grid_search_efficiency_detailed.png) | ![Tuning-Sensitivität Detailed](horizon/grid_search_tuning_impact_detailed.png) |\n\n")
        f.write("- **Tuning-Effizienz (Genauigkeit vs. Laufzeit):** Baumbasierte Ensemble-Methoden (Random Forest, XGBoost) erzielen signifikant niedrigere Fehler (RMSE [%], niedriger ist besser) als lineare Modelle (Ridge, Lasso), benötigen jedoch um Größenordnungen mehr Rechenzeit. Dies veranschaulicht den *Trade-off* zwischen Ressourcenverbrauch und Vorhersagegüte.\n")
        f.write("- **Tuning-Sensitivität (RMSE-Spannweite):** Zeigt den Einfluss der gewählten Hyperparameter auf die Modellgüte (RMSE [%], niedriger ist besser). Ein großer Boxplot (z. B. bei XGBoost oder Random Forest im produktiven Lauf) verdeutlicht, dass das Modell extrem sensitiv auf Hyperparameter reagiert und eine Grid-Search notwendig ist, um Overfitting zu vermeiden. **Hinweis zur Grafik:** Im Debug-Modus (`FINAL_TUNING = False`) wird pro Modell nur eine Konfiguration getestet, weshalb die Boxen zu einer einzelnen Linie kollabieren. Im produktiven Modus (`FINAL_TUNING = True`) wird die Spannweite sichtbar.\n\n")
    print(f"[OK] Markdown-Report generiert unter: {REPORT_FILE}")

def main():
    print("\nReport- & Visualisierungs-Generierung (Fast JSON Mode)")
    if not os.path.exists(METRICS_FILE):
        print(f"[ERROR] Metriken-Datei {METRICS_FILE} fehlt.")
        return
    with open(METRICS_FILE, 'r', encoding='utf-8') as f:
        m_data = json.load(f)
    generate_all_xai_plots(m_data)
    generate_additional_plots(m_data)
    generate_fold_performance_plots(m_data)
    generate_xai_decay(m_data)
    
    # Grid-Search Auswertungen laden (KI_03)
    gs_file = f'{REPORT_BASE}/grid_search_metrics.json'
    if os.path.exists(gs_file):
        with open(gs_file, 'r', encoding='utf-8') as f:
            gs_data = json.load(f)
        generate_grid_search_plots(gs_data)
    else:
        print("[WARN] Keine Grid-Search-Historie gefunden.")
        
    generate_markdown_report(m_data)

if __name__ == '__main__':
    main()
