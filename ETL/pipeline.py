import sys
import pandas as pd
import numpy as np
import holidays
import urllib.request
import json
import os
sys.dont_write_bytecode = True

df = pd.read_csv('data/gym_workload.csv')

# ==========================================
# Datenbereinigung & Qualitätssicherung
# ==========================================
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].dt.round('5min')
df = df.drop_duplicates(subset=['timestamp'], keep='last')

full_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='5min')
df = df.set_index('timestamp').reindex(full_range).reset_index().rename(columns={'index': 'timestamp'})

df['is_imputed'] = (df['utilization_percent'].isna() | df['temp'].isna()).astype(int)
df['gap_duration'] = df['is_imputed'].groupby((df['is_imputed'] == 0).cumsum()).cumcount()

if 'utilization_percent' in df.columns:
    df['utilization_percent'] = df['utilization_percent'].clip(lower=0, upper=100)
if 'rain' in df.columns:
    df['rain'] = df['rain'].clip(lower=0)

kontinuierlich_zu_interpolieren = ['temp', 'apparent_temp', 'humidity', 'rain', 'utilization_percent']
df[kontinuierlich_zu_interpolieren] = df[kontinuierlich_zu_interpolieren].interpolate(method='linear')

df['temp_diff'] = df['apparent_temp'] - df['temp']
df['is_day'] = df['is_day'].ffill().fillna(0)

# ==========================================
# Advanced Feature Engineering
# ==========================================
minute_of_day = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
df['sin_time'] = np.sin(2 * np.pi * minute_of_day / 1440)
df['cos_time'] = np.cos(2 * np.pi * minute_of_day / 1440)

df['date_only'] = df['timestamp'].dt.date
bw_feiertage = holidays.Germany(subdiv='BW')

def get_bw_vacations(start_date, end_date):
    vacation_dates = set()
    cache_path = 'ETL/config/vacation_cache.json'
    url = f"https://openholidaysapi.org/SchoolHolidays?countryIsoCode=DE&subdivisionCode=DE-BW&validFrom={start_date.strftime('%Y-%m-%d')}&validTo={end_date.strftime('%Y-%m-%d')}&languageIsoCode=DE"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            if status == 200:
                data = json.loads(response.read().decode('utf-8'))
                for holiday in data:
                    start = pd.to_datetime(holiday['startDate']).date()
                    end = pd.to_datetime(holiday['endDate']).date()
                    curr = start
                    while curr <= end:
                        vacation_dates.add(curr)
                        curr += pd.Timedelta(days=1)
                try:
                    cache_data = sorted([d.strftime('%Y-%m-%d') for d in vacation_dates])
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=4)
                    print(f"Cache aktualisiert: HTTP {status} | {len(vacation_dates)} Ferientage.")
                except Exception as cache_err:
                    print(f"Cache-Write fehlgeschlagen: {cache_err}")
                return vacation_dates
            else:
                raise RuntimeError(f"API Status Code {status}")
    except Exception as e:
        print(f"Ferien-API Error: {e}. Lade lokalen Cache...")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                vacation_dates = {pd.to_datetime(d).date() for d in cache_data}
                print(f"Success: {len(vacation_dates)} Ferientage importiert.")
                return vacation_dates
            except Exception as cache_err:
                raise RuntimeError(f"Cache-Load fehlgeschlagen: {cache_err}. Abort")
        else:
            raise FileNotFoundError(f"Ferien-API failed ({e}) & kein lokaler Cache in '{cache_path}' -> Abort")
vacation_set = get_bw_vacations(df['timestamp'].min(), df['timestamp'].max())

df['is_public_holiday'] = df['date_only'].apply(lambda d: 1 if d in bw_feiertage else 0)
df['is_school_holiday'] = df['date_only'].apply(lambda d: 1 if (d in vacation_set and d not in bw_feiertage) else 0)

# ==========================================
# Zeitreihen-Features (Lags und Trends)
# ==========================================
df['lag_1'] = df['utilization_percent'].shift(1)
df['lag_1d'] = df['utilization_percent'].shift(288)

# 7dlag Feiertagscheck
util = df['utilization_percent'].values
is_pub_hol = df['is_public_holiday'].values
lag_7d_robust = np.zeros_like(util)
for i in range(len(df)):
    lookback = 2016  # 7d
    while i - lookback >= 0:
        if is_pub_hol[i - lookback] == 0:
            lag_7d_robust[i] = util[i - lookback]
            break
        lookback += 2016
    else:
        if i - 2016 >= 0:
            lag_7d_robust[i] = util[i - 2016]
        else:
            lag_7d_robust[i] = np.nan

df['lag_7d_robust'] = lag_7d_robust
df['trend_15m'] = df['lag_1'] - df['utilization_percent'].shift(4)
df['trend_2h'] = df['lag_1'] - df['utilization_percent'].shift(24)
df['volatility_2h'] = df['utilization_percent'].shift(1).rolling(window=24).std()
df['ema_1h'] = df['utilization_percent'].shift(1).ewm(span=12, adjust=False).mean()

# One-Hot Encoding
df['day_of_week'] = pd.Categorical(df['timestamp'].dt.day_name(), categories=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
df = pd.get_dummies(df, columns=['day_of_week'], prefix='is', prefix_sep='_')

# Filtern großer Lücken (Qualitätssicherung)
df_diag = df.copy()
is_invalid_gap = ((df['is_imputed'] == 1) & (df['gap_duration'] > 24)).astype(int)
is_polluted = is_invalid_gap.rolling(window=25, min_periods=1).max().fillna(0).astype(bool)

df = df[~is_polluted]
df = df.dropna()

# ==========================================
# Feature Selection & Sortierung
# ==========================================
spalten_zum_loeschen = ['date', 'time', 'hour', 'apparent_temp', 'date_only']
spalten_zum_loeschen = [col for col in spalten_zum_loeschen if col in df.columns]
df = df.drop(columns=spalten_zum_loeschen)

target = 'utilization_percent'
weekdays = ['is_Monday', 'is_Tuesday', 'is_Wednesday', 'is_Thursday', 'is_Friday', 'is_Saturday', 'is_Sunday']
weekdays = [w for w in weekdays if w in df.columns]
other_binary = ['is_public_holiday', 'is_school_holiday', 'is_imputed', 'is_day']
continuous_features = [
    'sin_time', 'cos_time', 'lag_1', 'lag_1d', 'lag_7d_robust', 'trend_15m', 'trend_2h', 'volatility_2h', 'ema_1h',
    'temp', 'temp_diff', 'humidity', 'rain', 'gap_duration'
]
wunsch_reihenfolge = ['timestamp', target] + continuous_features + other_binary + weekdays
df = df[wunsch_reihenfolge]

for col in df.columns:
    if col != 'timestamp':
        df[col] = df[col].astype(float)

df.to_csv('data/gym_workload_mlready.csv', index=False)
print(f"Zeilen/Spalten nach Filterung: {df.shape}")
print(f"Imputierte verbleibende Zeilen: {int(df['is_imputed'].sum())}")
print(f"Features: {df.columns.tolist()}")

# ==========================================
# Diagnostics & Reports (Ki-generiert)
# ==========================================
from diagnostics import generate_diagnostics_report
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

generate_diagnostics_report(df_diag, df)
