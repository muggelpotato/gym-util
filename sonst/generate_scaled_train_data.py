# -*- coding: utf-8 -*-
"""
Skript zur Generierung der final skalierten und reduzierten Trainings-CSV-Dateien
für das Quick-Modell und das Detailed-Modell (5-Minuten-Horizont).
Gibt den Zustand der Daten unmittelbar vor dem Modell-Fitting (fit) aus.
Autor: Antigravity AI
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Pfade relativ zum Skript-Speicherort bestimmen (für maximale Portabilität)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
MLREADY_CSV = os.path.join(WORKSPACE_DIR, "data", "gym_workload_mlready.csv")
OUTPUT_QUICK_CSV = os.path.join(WORKSPACE_DIR, "data", "gym_workload_scaled_quick.csv")
OUTPUT_DETAILED_CSV = os.path.join(WORKSPACE_DIR, "data", "gym_workload_scaled_detailed_5m.csv")

def main():
    print(f"[Start] Lade Datensatz aus {MLREADY_CSV}...")
    if not os.path.exists(MLREADY_CSV):
        raise FileNotFoundError(f"mlready-Daten unter {MLREADY_CSV} nicht gefunden! Bitte führen Sie zuerst pipeline.py aus.")
        
    df = pd.read_csv(MLREADY_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Wochentage extrahieren
    weekdays = [c for c in df.columns if c.startswith('is_') and c.replace('is_', '') in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
    base_binary = ['is_public_holiday', 'is_school_holiday', 'is_day']
    
    # =========================================================================
    # 1. Generierung des skalierten QUICK-Datensatzes
    # =========================================================================
    print("[Processing] Bereite Quick-Modell-Daten vor...")
    # Features definieren
    quick_continuous = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain']
    quick_binary = base_binary + weekdays
    quick_features = quick_continuous + quick_binary
    
    df_quick = df.copy()
    df_quick['target'] = df_quick['utilization_percent']
    
    # Zeilen mit NaNs löschen (dropna vor dem Fit)
    df_quick = df_quick.dropna(subset=quick_features + ['target']).reset_index(drop=True)
    
    # Skalierung der kontinuierlichen Merkmale
    scaler_q = StandardScaler()
    scaled_cont_q = pd.DataFrame(
        scaler_q.fit_transform(df_quick[quick_continuous]), 
        columns=quick_continuous
    )
    
    # Zusammenführen mit binären Merkmalen und Target
    df_quick_final = pd.concat([
        df_quick[['timestamp', 'target']], 
        scaled_cont_q, 
        df_quick[quick_binary]
    ], axis=1)
    
    df_quick_final.to_csv(OUTPUT_QUICK_CSV, index=False)
    print(f"[OK] Quick-Datensatz exportiert nach: {OUTPUT_QUICK_CSV} (Shape: {df_quick_final.shape})")
    
    # =========================================================================
    # 2. Generierung des skalierten DETAILED-Datensatzes (5 Min Horizont, h = 1)
    # =========================================================================
    print("[Processing] Bereite Detailed-Modell-Daten (5m Horizont) vor...")
    # h = 1 (5 Minuten Horizont entspricht 1 Zeitschritt)
    # lag_inference ist lag_1.shift(h - 1) -> lag_1.shift(0) = lag_1
    df_det = df.copy()
    df_det['target'] = df_det['utilization_percent']
    
    # Zeitreihen-Features für h=1 zuweisen
    df_det['lag_inference'] = df_det['lag_1']
    df_det['ema_1h_inference'] = df_det['ema_1h']
    df_det['trend_15m_inference'] = df_det['trend_15m']
    df_det['trend_2h_inference'] = df_det['trend_2h']
    df_det['volatility_2h_inference'] = df_det['volatility_2h']
    
    det_continuous = [
        'sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain', 
        'lag_inference', 'ema_1h_inference', 'trend_15m_inference', 
        'trend_2h_inference', 'volatility_2h_inference', 'lag_1d', 'lag_7d_robust'
    ]
    det_binary = base_binary + weekdays
    det_features = det_continuous + det_binary
    
    # NaNs dropen
    df_det = df_det.dropna(subset=det_features + ['target']).reset_index(drop=True)
    
    # Skalierung der kontinuierlichen Merkmale
    scaler_d = StandardScaler()
    scaled_cont_d = pd.DataFrame(
        scaler_d.fit_transform(df_det[det_continuous]), 
        columns=det_continuous
    )
    
    # Zusammenführen
    df_det_final = pd.concat([
        df_det[['timestamp', 'target']], 
        scaled_cont_d, 
        df_det[det_binary]
    ], axis=1)
    
    df_det_final.to_csv(OUTPUT_DETAILED_CSV, index=False)
    print(f"[OK] Detailed-Datensatz (5m) exportiert nach: {OUTPUT_DETAILED_CSV} (Shape: {df_det_final.shape})")
    print("[Fertig] Alle skalierten Trainings-CSVs wurden erfolgreich generiert.")

if __name__ == "__main__":
    main()
