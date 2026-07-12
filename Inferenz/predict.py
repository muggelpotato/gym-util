# -*- coding: utf-8 -*-
"""
Inferenz-Engine für die Gym-Auslastungsvorhersage.
Kompakte, modularisierte Single-Process-Variante zur Vermeidung von Next.js-Jank.
Autor: Antigravity AI
Vorlesungsbezug: KI_01 bis KI_03 (ETL, Feature Engineering, Alignment)
"""

import os
import sys
import json
import pickle
import argparse
from datetime import timedelta
import pandas as pd
import numpy as np
import holidays
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(WORKSPACE_DIR, "data", "new_data.csv")
VACATION_CACHE_PATH = os.path.join(WORKSPACE_DIR, "ETL", "config", "vacation_cache.json")

# Methodische Entscheidung zur Systemstabilität:
# Direkte relative Pfadauflösung zur Vermeidung von Git- und Windows-Symlink-Problemen.
# Bevorzugt den direkten Pfad im Workspace, andernfalls lokaler Fallback.
MODELS_DIR = os.path.join(WORKSPACE_DIR, "Model", "models", "final")
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = os.path.join(BASE_DIR, "models")

def clean_nans(val):
    """Sichert valides JSON durch rekursives Ersetzen von NaN durch None."""
    if isinstance(val, dict):
        return {k: clean_nans(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [clean_nans(v) for v in val]
    elif isinstance(val, (float, np.floating)) and np.isnan(val):
        return None
    return val

class InferenceEngine:
    WEEKDAYS = [f'is_{w}' for w in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]

    def __init__(self):
        self.df = None
        self.vacation_set = set()
        self.bw_feiertage = holidays.Germany(subdiv='BW')
        self.model_cache = {}  # Cache für geladene Modelle/Scaler zur Latenzreduktion
        self._load_vacation_cache()
        self.preprocess_data()
        
    def _load_vacation_cache(self):
        """Lädt Schulferien aus lokalem Cache zur API-Latenz-Vermeidung."""
        if os.path.exists(VACATION_CACHE_PATH):
            try:
                with open(VACATION_CACHE_PATH, 'r', encoding='utf-8') as f:
                    self.vacation_set = {pd.to_datetime(d).date() for d in json.load(f)}
                print(f"[Engine] {len(self.vacation_set)} Ferientage geladen.", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[WARN] Ferien-Cache Ladefehler: {e}", file=sys.stderr, flush=True)

    def preprocess_data(self):
        """Bereinigt & transformiert Rohdaten analog zur ETL-Pipeline."""
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Rohdaten unter {DATA_PATH} nicht gefunden!")
            
        df = pd.read_csv(DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('5min')
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        if 'rain' in df.columns:
            df['rain'] = df['rain'].clip(lower=0)

        # Zeit-Raster heilen (erzeugt Lücken bei Netzwerkausfällen)
        full_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='5min')
        df = df.set_index('timestamp').reindex(full_range).reset_index().rename(columns={'index': 'timestamp'})
        
        # Datenqualitäts-Kennzeichnung (basierend auf dem Zustand direkt nach dem Reindexing, vor der Interpolation)
        df['is_imputed'] = (df['utilization_percent'].isna() | df['temp'].isna()).astype(int)
        df['gap_duration'] = df['is_imputed'].groupby((df['is_imputed'] == 0).cumsum()).cumcount()
        
        if 'utilization_percent' in df.columns:
            df['utilization_percent'] = df['utilization_percent'].clip(0, 100)
            
        # Alle kontinuierlichen Daten (inklusive utilization_percent) nach dem Reindexing interpolieren
        kontinuierlich_zu_interpolieren = ['temp', 'apparent_temp', 'humidity', 'rain', 'utilization_percent']
        df[kontinuierlich_zu_interpolieren] = df[kontinuierlich_zu_interpolieren].interpolate(method='linear')
        
        df['temp_diff'] = df['apparent_temp'] - df['temp']
        df['is_day'] = df['is_day'].ffill().fillna(0)
        
        # Zyklische Zeit (Vermeidung von Mitternachts-Diskontinuität)
        minute_of_day = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
        df['sin_time'] = np.sin(2 * np.pi * minute_of_day / 1440)
        df['cos_time'] = np.cos(2 * np.pi * minute_of_day / 1440)
        
        # Kalender & Ferien
        df['date_only'] = df['timestamp'].dt.date
        df['is_public_holiday'] = df['date_only'].apply(lambda d: 1.0 if d in self.bw_feiertage else 0.0)
        df['is_school_holiday'] = df['date_only'].apply(lambda d: 1.0 if (d in self.vacation_set and d not in self.bw_feiertage) else 0.0)
        
        # Zeitreihen-Features
        df['lag_1'] = df['utilization_percent'].shift(1)
        df['lag_1d'] = df['utilization_percent'].shift(288)
        
        # Robustes Wochen-Lag (Feiertage überspringend)
        util, is_pub_hol = df['utilization_percent'].values, df['is_public_holiday'].values
        lag_7d_robust = np.zeros_like(util)
        for i in range(len(df)):
            lookback = 2016
            while i - lookback >= 0:
                if is_pub_hol[i - lookback] == 0.0:
                    lag_7d_robust[i] = util[i - lookback]
                    break
                lookback += 2016
            else:
                lag_7d_robust[i] = util[i - 2016] if i - 2016 >= 0 else np.nan
        df['lag_7d_robust'] = lag_7d_robust
        
        df['trend_15m'] = df['lag_1'] - df['utilization_percent'].shift(4)
        df['trend_2h'] = df['lag_1'] - df['utilization_percent'].shift(24)
        df['volatility_2h'] = df['utilization_percent'].shift(1).rolling(window=24).std()
        df['ema_1h'] = df['utilization_percent'].shift(1).ewm(span=12, adjust=False).mean()
        
        # Wochentage One-Hot
        df['day_of_week'] = pd.Categorical(df['timestamp'].dt.day_name(), categories=[
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        df = pd.get_dummies(df, columns=['day_of_week'], prefix='is', prefix_sep='_', dtype=float)
        for col in df.columns:
            if col.startswith('is_') or col in ['is_public_holiday', 'is_school_holiday', 'is_day']:
                df[col] = df[col].astype(float)
                
        # Methodische Entscheidung (Tagesprofil für XAI):
        # Berechnen des typischen Tagesverlaufs (Mittelwert und 95%-Konfidenzintervall des Mittelwerts)
        # zur Bereitstellung einer Baseline in der Benutzeroberfläche.
        # Dies dient der Erklärbarkeit, da es Abweichungen vom gewöhnlichen Verhalten visualisiert.
        # Wir berechnen dies wochentagsspezifisch und schließen Feiertage sowie Schulferien aus,
        # um eine saubere Repräsentation eines gewöhnlichen Tages zu erhalten.
        df['time_of_day'] = df['timestamp'].dt.time
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Ausschluss von Feiertagen und Schulferien für das wochentagsspezifische Profil
        normal_days = df[(df['is_public_holiday'] == 0.0) & (df['is_school_holiday'] == 0.0)]
        if len(normal_days) == 0:
            normal_days = df
            
        profile_stats = normal_days.groupby(['day_of_week', 'time_of_day'])['utilization_percent'].agg(['mean', 'std', 'count'])
        profile_stats['sem'] = profile_stats['std'] / np.sqrt(profile_stats['count'])
        profile_stats['ci_lower'] = np.clip(profile_stats['mean'] - 1.96 * profile_stats['sem'], 0, 100)
        profile_stats['ci_upper'] = np.clip(profile_stats['mean'] + 1.96 * profile_stats['sem'], 0, 100)
        
        self.typical_profile = {
            (day, t.strftime('%H:%M:%S')): {
                'mean': float(row['mean']),
                'ci_lower': float(row['ci_lower']) if not pd.isna(row['ci_lower']) else float(row['mean']),
                'ci_upper': float(row['ci_upper']) if not pd.isna(row['ci_upper']) else float(row['mean'])
            } for (day, t), row in profile_stats.iterrows()
        }
        
        # Fallback-Tagesprofil ohne Wochentagsdifferenzierung
        fallback_stats = df.groupby('time_of_day')['utilization_percent'].agg(['mean', 'std', 'count'])
        fallback_stats['sem'] = fallback_stats['std'] / np.sqrt(fallback_stats['count'])
        fallback_stats['ci_lower'] = np.clip(fallback_stats['mean'] - 1.96 * fallback_stats['sem'], 0, 100)
        fallback_stats['ci_upper'] = np.clip(fallback_stats['mean'] + 1.96 * fallback_stats['sem'], 0, 100)
        
        self.fallback_profile = {
            t.strftime('%H:%M:%S'): {
                'mean': float(row['mean']),
                'ci_lower': float(row['ci_lower']) if not pd.isna(row['ci_lower']) else float(row['mean']),
                'ci_upper': float(row['ci_upper']) if not pd.isna(row['ci_upper']) else float(row['mean'])
            } for t, row in fallback_stats.iterrows()
        }

        self.df = df.drop(columns=[c for c in ['date', 'time', 'hour', 'apparent_temp', 'date_only', 'time_of_day', 'day_of_week'] if c in df.columns]) \
                    .sort_values('timestamp').reset_index(drop=True)
        print(f"[Engine] Vorverarbeitung abgeschlossen. Shape: {self.df.shape}", file=sys.stderr, flush=True)

    def get_typical_value(self, timestamp, metric):
        """Hilfsfunktion zur Abfrage des wochentagsspezifischen Profils mit Fallback."""
        day_name = timestamp.strftime('%A')
        time_str = timestamp.strftime('%H:%M:%S')
        key = (day_name, time_str)
        if key in self.typical_profile:
            return self.typical_profile[key][metric]
        return self.fallback_profile.get(time_str, {}).get(metric, 0.0)

    def get_valid_timestamps(self):
        """Extrahiert Demo-Zeitstempel mit genügend historischem Verlauf."""
        start_time = self.df['timestamp'].min() + timedelta(days=1)
        valid = self.df[(self.df['timestamp'] >= start_time) & (self.df['timestamp'].dt.minute % 5 == 0)]
        return [t.isoformat() for t in valid['timestamp']]

    def _load_model_and_scaler(self, model_dir, prefix, model_name=None):
        """Methodischer Helper zum Cachen und Laden von Modellen/Scaler (Vermeidung von redundantem I/O)."""
        cache_key = (model_dir, prefix, model_name.lower() if model_name else "best")
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
            
        with open(os.path.join(model_dir, f"scaler_{prefix}.pkl"), 'rb') as f:
            scaler = pickle.load(f)
            
        if model_name is None:
            best_files = [f for f in os.listdir(model_dir) if f.startswith(f"best_model_{prefix}")]
            if not best_files:
                raise FileNotFoundError(f"Bestes Modell {prefix} in {model_dir} nicht gefunden!")
            model_file = best_files[0]
            algo_name = model_file.split('_')[-1].replace('.pkl', '')
        else:
            model_file = f"model_{prefix}_{model_name.lower()}.pkl"
            algo_name = model_name.lower()
            
        file_path = os.path.join(model_dir, model_file)
        if not os.path.exists(file_path):
            # Rekursiver Fallback auf das beste Modell
            return self._load_model_and_scaler(model_dir, prefix, None)
            
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            
        res = (model, scaler, algo_name)
        self.model_cache[cache_key] = res
        return res

    def _predict(self, model, scaler, continuous_data, binary_data, continuous_cols, binary_cols):
        """Einheitliche Feature-Aufbereitung und Vorhersage zur Vermeidung von Redundanz und Data Leakage."""
        if isinstance(continuous_data, dict):
            df_cont = pd.DataFrame([continuous_data])
            df_bin = pd.DataFrame([binary_data])
        else:
            df_cont = pd.DataFrame(continuous_data)
            df_bin = pd.DataFrame(binary_data)
            
        df_cont = df_cont[continuous_cols]
        df_bin = df_bin[binary_cols]
        
        # Imputation fehlender Wetterwerte
        for col in continuous_cols:
            if df_cont[col].isna().any():
                fallback = self.df[col].median() if col in self.df.columns and not self.df[col].isna().all() else 0.0
                df_cont[col] = df_cont[col].fillna(fallback)
                
        df_cont_scaled = pd.DataFrame(scaler.transform(df_cont), columns=continuous_cols, index=df_cont.index)
        
        # Methodische Korrektur (Feature Alignment):
        # Zusammenfügen von kontinuierlichen (skalierten) und binären (pass-through) Features.
        X_final = pd.concat([df_cont_scaled, df_bin], axis=1)[continuous_cols + binary_cols]
        return np.clip(model.predict(X_final.values), 0, 100)

    def predict_quick(self, target_df, model_name=None):
        """Führt Prognose mit dem Quick-Modell aus (wetterabhängige Echtzeit-Inferenz)."""
        model, scaler, algo = self._load_model_and_scaler(os.path.join(MODELS_DIR, "quick"), "quick", model_name)
        continuous = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain']
        binary = ['is_public_holiday', 'is_school_holiday', 'is_day'] + self.WEEKDAYS
        preds = self._predict(model, scaler, target_df[continuous], target_df[binary], continuous, binary)
        return preds.tolist(), algo

    def predict_detailed_horizon(self, current_time, mins, model_name=None):
        """Prognostiziert den Detailed-Zustand bei current_time + mins (autoregressive Inferenz)."""
        model, scaler, algo = self._load_model_and_scaler(os.path.join(MODELS_DIR, "detailed", f"{mins}m"), f"detailed_{mins}m", model_name)
        
        idx_current = self.df[self.df['timestamp'] == current_time].index[0]
        idx_target = idx_current + (mins // 5)
        current_row = self.df.iloc[idx_current]
        lag_inference = current_row['utilization_percent']
        
        # Trendberechnung des aktuellen Zustands
        if idx_current + 1 < len(self.df):
            ema_1h_inference = self.df.iloc[idx_current + 1]['ema_1h']
            trend_15m_inference = self.df.iloc[idx_current + 1]['trend_15m']
            trend_2h_inference = self.df.iloc[idx_current + 1]['trend_2h']
            volatility_2h_inference = self.df.iloc[idx_current + 1]['volatility_2h']
        else:
            alpha = 2.0 / 13.0
            last_ema = self.df.iloc[idx_current]['ema_1h']
            ema_1h_inference = (1.0 - alpha) * (last_ema if not pd.isna(last_ema) else lag_inference) + alpha * lag_inference
            trend_15m_inference = lag_inference - self.df.iloc[max(0, idx_current - 3)]['utilization_percent']
            trend_2h_inference = lag_inference - self.df.iloc[max(0, idx_current - 23)]['utilization_percent']
            volatility_2h_inference = self.df.iloc[max(0, idx_current - 23) : idx_current + 1]['utilization_percent'].std()
            if pd.isna(volatility_2h_inference): volatility_2h_inference = 0.0

        if idx_target >= len(self.df):
            # Synthetische Zukunftsberechnung
            target_time = current_time + timedelta(minutes=mins)
            last_row = self.df.iloc[-1]
            target_date = target_time.date()
            is_pub_hol = 1.0 if target_date in self.bw_feiertage else 0.0
            is_sch_hol = 1.0 if (target_date in self.vacation_set and target_date not in self.bw_feiertage) else 0.0
            minute_of_day = target_time.hour * 60 + target_time.minute
            sin_time, cos_time = np.sin(2 * np.pi * minute_of_day / 1440), np.cos(2 * np.pi * minute_of_day / 1440)
            weekdays_dict = {f'is_{w}': 1.0 if w == target_time.strftime('%A') else 0.0 for w in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
            
            def find_closest_lag(dt_ago, is_robust_7d=False):
                diffs = (self.df['timestamp'] - dt_ago).abs()
                idx = diffs.idxmin()
                if diffs.iloc[idx] < timedelta(minutes=10):
                    row = self.df.iloc[idx]
                    return row['lag_7d_robust'] if (is_robust_7d and row['is_public_holiday'] == 1.0) else row['utilization_percent']
                return last_row['lag_7d_robust'] if is_robust_7d else last_row['utilization_percent']
            
            lag_1d = find_closest_lag(target_time - timedelta(days=1))
            lag_7d_robust = find_closest_lag(target_time - timedelta(days=7), is_robust_7d=True)
            target_temp, target_temp_diff, target_humidity, target_rain = last_row['temp'], last_row['temp_diff'], last_row['humidity'], last_row['rain']
            target_is_day = 1.0 if 6 <= target_time.hour < 22 else 0.0
        else:
            target_row = self.df.iloc[idx_target]
            sin_time, cos_time = target_row['sin_time'], target_row['cos_time']
            target_temp, target_temp_diff, target_humidity, target_rain = target_row['temp'], target_row['temp_diff'], target_row['humidity'], target_row['rain']
            is_pub_hol, is_sch_hol, target_is_day = target_row['is_public_holiday'], target_row['is_school_holiday'], target_row['is_day']
            weekdays_dict = {f'is_{w}': target_row[f'is_{w}'] for w in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
            lag_1d, lag_7d_robust = target_row['lag_1d'], target_row['lag_7d_robust']

        continuous_data = {
            'sin_time': sin_time, 'cos_time': cos_time, 'temp': target_temp, 'temp_diff': target_temp_diff,
            'humidity': target_humidity, 'rain': target_rain, 'lag_inference': lag_inference,
            'ema_1h_inference': ema_1h_inference, 'trend_15m_inference': trend_15m_inference, 'trend_2h_inference': trend_2h_inference,
            'volatility_2h_inference': volatility_2h_inference, 'lag_1d': lag_1d, 'lag_7d_robust': lag_7d_robust
        }
        binary_data = {'is_public_holiday': is_pub_hol, 'is_school_holiday': is_sch_hol, 'is_day': target_is_day, **weekdays_dict}
        
        continuous_cols = ['sin_time', 'cos_time', 'temp', 'temp_diff', 'humidity', 'rain', 'lag_inference', 'ema_1h_inference', 'trend_15m_inference', 'trend_2h_inference', 'volatility_2h_inference', 'lag_1d', 'lag_7d_robust']
        binary_cols = ['is_public_holiday', 'is_school_holiday', 'is_day'] + self.WEEKDAYS
        
        pred = self._predict(model, scaler, continuous_data, binary_data, continuous_cols, binary_cols)[0]
        return float(pred), algo

    def run_prediction_pipeline(self, timestamp_str):
        """Führt Vorhersagepipeline (Vergangenheit, Zukunft, Quick & Detailed) aus."""
        dt = pd.to_datetime(timestamp_str)
        idx_current = (self.df['timestamp'] - dt).abs().idxmin()
        current_time = self.df.iloc[idx_current]['timestamp']
        
        if idx_current < 12:
            raise ValueError("Zeitstempel hat unzureichende Historie!")
            
        past_df = self.df.iloc[idx_current - 12 : idx_current + 1]
        
        # Horizon berechnen (mind. 25 Schritte)
        end_of_day = current_time.replace(hour=23, minute=55, second=0, microsecond=0)
        steps_to_end = int((end_of_day - current_time).total_seconds() / 300)
        num_future_steps = max(25, steps_to_end + 1)
        
        future_rows = []
        last_row = self.df.iloc[-1]
        for step in range(num_future_steps):
            idx_target = idx_current + step
            if idx_target < len(self.df):
                future_rows.append(self.df.iloc[idx_target].copy())
            else:
                target_time = current_time + timedelta(minutes=step * 5)
                target_date = target_time.date()
                is_pub_hol = 1.0 if target_date in self.bw_feiertage else 0.0
                is_sch_hol = 1.0 if (target_date in self.vacation_set and target_date not in self.bw_feiertage) else 0.0
                minute_of_day = target_time.hour * 60 + target_time.minute
                sin_time, cos_time = np.sin(2 * np.pi * minute_of_day / 1440), np.cos(2 * np.pi * minute_of_day / 1440)
                weekdays_dict = {f'is_{w}': 1.0 if w == target_time.strftime('%A') else 0.0 for w in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
                
                future_rows.append(pd.Series({
                    'timestamp': target_time, 'utilization_percent': np.nan, 'temp': float(last_row['temp']),
                    'temp_diff': float(last_row['temp_diff']), 'humidity': float(last_row['humidity']),
                    'rain': float(last_row['rain']), 'is_day': 1.0 if 6 <= target_time.hour < 22 else 0.0,
                    'is_public_holiday': is_pub_hol, 'is_school_holiday': is_sch_hol, 'is_imputed': 0, 'gap_duration': 0,
                    'sin_time': sin_time, 'cos_time': cos_time,
                    **weekdays_dict
                }))
                
        future_df = pd.DataFrame(future_rows).reset_index(drop=True)
        quick_preds, quick_algo = self.predict_quick(future_df)
        
        model_names = ["randomforest", "gradientboosting", "gbquantile90", "xgboost", "ridge", "lasso", "elasticnet"]
        quick_model_predictions = {}
        quick_model_predictions['best'] = [
            {"timestamp": r['timestamp'].isoformat(), "value": float(val)}
            for (_, r), val in zip(future_df.iterrows(), quick_preds)
        ]
        for name in model_names:
            try:
                preds, _ = self.predict_quick(future_df, model_name=name)
                quick_model_predictions[name] = [
                    {"timestamp": r['timestamp'].isoformat(), "value": float(val)}
                    for (_, r), val in zip(future_df.iterrows(), preds)
                ]
            except Exception as e:
                print(f"[WARN] Quick-Vorhersage für Modell {name} fehlgeschlagen: {e}", file=sys.stderr)
        
        detailed_preds = {}
        detailed_algos = {}
        for h in [5, 15, 30, 60, 90, 120]:
            pred, algo = self.predict_detailed_horizon(current_time, h)
            detailed_preds[h] = pred
            detailed_algos[h] = algo
            
        detailed_model_predictions = {}
        detailed_model_predictions['best'] = [
            {"horizon_mins": h, "timestamp": (current_time + timedelta(minutes=h)).isoformat(), "value": detailed_preds[h]}
            for h in [5, 15, 30, 60, 90, 120]
        ]
        for name in model_names:
            detailed_model_predictions[name] = []
            for h in [5, 15, 30, 60, 90, 120]:
                try:
                    pred, _ = self.predict_detailed_horizon(current_time, h, model_name=name)
                    detailed_model_predictions[name].append({
                        "horizon_mins": h,
                        "timestamp": (current_time + timedelta(minutes=h)).isoformat(),
                        "value": pred
                    })
                except Exception as e:
                    print(f"[WARN] Detailed-Vorhersage für Modell {name} bei Horizon {h} fehlgeschlagen: {e}", file=sys.stderr)
            
        return clean_nans({
            "requested_timestamp": timestamp_str,
            "actual_timestamp": current_time.isoformat(),
            "current_utilization": float(self.df.iloc[idx_current]['utilization_percent']),
            "is_imputed": int(self.df.iloc[idx_current]['is_imputed']),
            "gap_duration": int(self.df.iloc[idx_current]['gap_duration']),
            "current_weather": {
                "temp": float(self.df.iloc[idx_current]['temp']), "humidity": float(self.df.iloc[idx_current]['humidity']),
                "rain": float(self.df.iloc[idx_current]['rain']), "is_day": int(self.df.iloc[idx_current]['is_day'])
            },
            "past_data": [
                {
                    "timestamp": r['timestamp'].isoformat(),
                    "utilization": float(r['utilization_percent']),
                    "is_imputed": int(r['is_imputed']),
                    "typical_mean": float(self.get_typical_value(r['timestamp'], 'mean')),
                    "typical_ci_lower": float(self.get_typical_value(r['timestamp'], 'ci_lower')),
                    "typical_ci_upper": float(self.get_typical_value(r['timestamp'], 'ci_upper'))
                } for _, r in past_df.iterrows()
            ],
            "future_data": [
                {
                    "timestamp": r['timestamp'].isoformat(),
                    "utilization": float(r['utilization_percent']),
                    "temp": float(r['temp']),
                    "rain": float(r['rain']),
                    "humidity": float(r['humidity']),
                    "is_day": int(r['is_day']),
                    "is_holiday": int(r['is_public_holiday'] == 1.0 or r['is_school_holiday'] == 1.0),
                    "is_imputed": int(r['is_imputed']),
                    "typical_mean": float(self.get_typical_value(r['timestamp'], 'mean')),
                    "typical_ci_lower": float(self.get_typical_value(r['timestamp'], 'ci_lower')),
                    "typical_ci_upper": float(self.get_typical_value(r['timestamp'], 'ci_upper'))
                } for _, r in future_df.iterrows()
            ],
            "quick_predictions": {
                "algo": quick_algo,
                "values": [{"timestamp": r['timestamp'].isoformat(), "value": float(val)} for (_, r), val in zip(future_df.iterrows(), quick_preds)],
                "models": quick_model_predictions
            },
            "detailed_predictions": {
                "algos": detailed_algos,
                "values": [{"horizon_mins": h, "timestamp": (current_time + timedelta(minutes=h)).isoformat(), "value": val} for h, val in detailed_preds.items() if val is not None],
                "models": detailed_model_predictions
            }
        })

class APIRequestHandler(BaseHTTPRequestHandler):
    engine = None

    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-type', f'{content_type}; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if path in ['/', '/index.html']:
            try:
                with open(os.path.join(BASE_DIR, 'index.html'), 'r', encoding='utf-8') as f:
                    content = f.read()
                self._set_headers(200, 'text/html')
                self.wfile.write(content.encode('utf-8'))
            except Exception as e:
                self._set_headers(500, 'text/plain')
                self.wfile.write(f"Fehler: {e}".encode('utf-8'))
        elif path == '/api/predict':
            ts = query.get('timestamp', [None])[0]
            if not ts:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "timestamp Parameter fehlt"}).encode('utf-8'))
                return
            try:
                self._set_headers(200)
                self.wfile.write(json.dumps(self.engine.run_prediction_pipeline(ts), indent=2).encode('utf-8'))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        elif path == '/api/timestamps':
            try:
                self._set_headers(200)
                self.wfile.write(json.dumps({"timestamps": self.engine.get_valid_timestamps()}).encode('utf-8'))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Nicht gefunden"}).encode('utf-8'))

def start_server(port):
    APIRequestHandler.engine = InferenceEngine()
    server = HTTPServer(('localhost', port), APIRequestHandler)
    print(f"[Engine] API Server läuft auf http://localhost:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Engine] Server wird heruntergefahren...", file=sys.stderr, flush=True)
        server.server_close()

def main():
    parser = argparse.ArgumentParser(description="Inferenz-Engine CLI & Server")
    parser.add_argument("--timestamp", type=str, help="Führt Vorhersage für den angegebenen ISO-Zeitstempel aus.")
    parser.add_argument("--timestamps", action="store_true", help="Gibt Liste aller validen Demo-Zeitstempel aus.")
    parser.add_argument("--server", action="store_true", help="Startet die Engine als HTTP API Server.")
    parser.add_argument("--port", type=int, default=5000, help="Port für den Server (Standard: 5000).")
    args = parser.parse_args()
    
    if args.server:
        start_server(args.port)
    elif args.timestamps or args.timestamp:
        engine = InferenceEngine()
        if args.timestamps:
            print(json.dumps({"timestamps": engine.get_valid_timestamps()}))
        else:
            try:
                print(json.dumps(engine.run_prediction_pipeline(args.timestamp), indent=2))
            except Exception as e:
                print(json.dumps({"error": str(e)}))
                sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
