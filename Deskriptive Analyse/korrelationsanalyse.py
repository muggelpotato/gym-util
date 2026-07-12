import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100
os.makedirs('Deskriptive Analyse/reports', exist_ok=True)

def plot_daily_profile(df):
    angles = np.arctan2(df['sin_time'], df['cos_time'])
    df_plot = df.copy()
    df_plot['hour_float'] = (angles / (2 * np.pi) * 24) % 24

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_plot, x='hour_float', y='utilization_percent', color='teal', linewidth=2)
    plt.title('Durchschnittliches Tagesprofil: Auslastung vs. Uhrzeit (Kombiniert)', fontsize=16)
    plt.xlabel('Uhrzeit (Stunden)', fontsize=12)
    plt.ylabel('Auslastung (%)', fontsize=12)
    plt.xticks(range(0, 25, 2))
    plt.xlim(0, 24)
    plt.tight_layout()
    plt.savefig('Deskriptive Analyse/reports/daily_profile.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

def plot_heatmap(df):
    plt.figure(figsize=(16, 12))
    korrelationen = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(korrelationen, dtype=bool))
    
    sns.heatmap(korrelationen, mask=mask, annot=True, cmap='coolwarm', 
                fmt=".2f", vmin=-1, vmax=1, square=True, linewidths=.5,
                cbar_kws={"shrink": .8}, annot_kws={"size": 9})
    
    plt.title('Korrelationsmatrix: Logisch sortierte Einflussfaktoren', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('Deskriptive Analyse/reports/correlation_heatmap.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

def plot_scatterplots(df, punkt_farbe='blue', transparenz=0.15):
    relevante_features = ['lag_1', 'lag_1d', 'temp', 'gap_duration']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    label_mapping = {
        'lag_1': 'Auslastung vor 5 Min. (lag_1) [%]',
        'lag_1d': 'Auslastung vor 24 Std. (lag_1d) [%]',
        'temp': 'Außentemperatur (temp) [°C]',
        'gap_duration': 'Imputations-Dauer (gap_duration) [5-Min-Schritte]'
    }

    for i, feature in enumerate(relevante_features):
        if feature in df.columns:
            sns.scatterplot(
                data=df, x=feature, y='utilization_percent', 
                alpha=transparenz, color=punkt_farbe, s=15, ax=axes[i]
            )
            axes[i].set_xlabel(label_mapping.get(feature, feature), fontsize=11)
            axes[i].set_ylabel('Aktuelle Auslastung [%]', fontsize=11)
            axes[i].set_title(f'{feature} vs. Auslastung', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Deskriptive Analyse/reports/scatter_analysis.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

def plot_vif(df):
    features = ['sin_time', 'cos_time', 'lag_1', 'lag_1d', 'trend_2h', 'volatility_2h', 'ema_1h', 'temp', 'temp_diff', 'humidity', 'rain']
    features = [f for f in features if f in df.columns]
    
    X = df[features].copy()
    X['const'] = 1.0
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data[vif_data['Feature'] != 'const'].sort_values(by='VIF', ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71' if v <= 5 else '#f39c12' if v <= 10 else '#e74c3c' for v in vif_data['VIF']]
    bars = plt.barh(vif_data['Feature'], vif_data['VIF'], color=colors, edgecolor='none', height=0.6)
    
    plt.axvline(x=5, color='#f39c12', linestyle='--', alpha=0.7, label='VIF = 5 (Moderate Schwelle)')
    plt.axvline(x=10, color='#e74c3c', linestyle='--', alpha=0.7, label='VIF = 10 (Kritische Schwelle)')
   
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                 va='center', ha='left', fontsize=10, fontweight='bold')
                 
    plt.title('Variance Inflation Factor (VIF): Multikollinearitäts-Analyse', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('VIF-Wert (Schrumpfungs-Faktor)', fontsize=12)
    plt.ylabel('Merkmale (Features)', fontsize=12)
    plt.xlim(0, max(vif_data['VIF'].max() + 3, 12))
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Deskriptive Analyse/reports/vif_analysis.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == "__main__":
    df_daten = pd.read_csv('data/gym_workload_mlready.csv')
    plot_daily_profile(df_daten)
    plot_heatmap(df_daten)
    plot_scatterplots(df_daten)
    plot_vif(df_daten)
