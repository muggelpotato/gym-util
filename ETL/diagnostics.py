"""
Modul: diagnostics.py
Autor: AI-Assistant (Pair-Programming)
Beschreibung:
    Dieses Modul kapselt die Diagnose- und Berichterstellungslogik des ETL-Prozesses.
    Es gibt den detaillierten Bericht farbig formatiert im Terminal aus und
    speichert zusätzlich den Markdown-Bericht sowie das Datenqualitäts-Donut-Diagramm.
    
Vorlesungsbezug (KI_01 - KI_03):
    - Datenbereinigung & Imputation: Visualisierung der Qualität durch Unterscheidung von
      realen, imputierten und aufgrund von Qualitätskriterien gelöschten Zeilen.
    - Analyse der Lücken und Ausfälle zur methodischen Bewertung der Systemstabilität.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless-Modus für Server-Kompatibilität
import matplotlib.pyplot as plt

# ANSI Farbcodes für lesbarere Konsolenausgaben (Terminal-Bericht)
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[92m"   # Positive/Normale Werte
C_YELLOW = "\033[93m"  # Warnungen, mittlere Werte
C_RED = "\033[91m"     # Ausfälle, Peaks, Extreme
C_BLUE = "\033[94m"    # Info, Gesamtwerte
C_CYAN = "\033[96m"    # Zeitstempel
C_MAGENTA = "\033[95m" # Header / Erfassungszeitraum

def generate_diagnostics_report(df_diag, df):
    """
    Erstellt das Datenqualitäts-Donut-Diagramm, speichert den Markdown-Bericht
    und gibt die KPIs farbig formatiert im Terminal aus.
    
    Parameter:
        df_diag (DataFrame): Der Datenbestand vor der Bereinigung der Großlücken.
        df (DataFrame): Der finale, bereinigte und ML-bereite Datenbestand.
    """
    os.makedirs('ETL/reports', exist_ok=True)
    
    # =========================================================================
    # 1. Donut-Diagramm zur Datenqualität (Visualisierung der Datenintegrität)
    # =========================================================================
    real_cnt = len(df_diag[df_diag['is_imputed'] == 0])
    imp_cnt = len(df_diag[(df_diag['is_imputed'] == 1) & (df_diag['gap_duration'] <= 24)])
    drop_cnt = len(df_diag) - len(df)
    total_cnt = len(df_diag)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Daten & Farben (Modernes, harmonisches Farbschema)
    recipe = [real_cnt, imp_cnt, drop_cnt]
    labels = [
        f'Reale Daten\n({real_cnt:,} Zeilen)', 
        f'Imputierte Daten\n({imp_cnt:,} Zeilen)', 
        f'Gelöschte Lücken (>2h)\n({drop_cnt:,} Zeilen)'
    ]
    colors = ['#10b981', '#f59e0b', '#ef4444']  # Smaragdgrün, Bernstein, Rosarot
    
    # Donut-Wedges zeichnen (Breite = 0.4 ergibt das Loch in der Mitte)
    wedges, texts, autotexts = ax.pie(
        recipe, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        pctdistance=0.75
    )
    
    # Text-Styling für bessere Lesbarkeit
    for text in texts:
        text.set_fontsize(9.5)
        text.set_fontweight('semibold')
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
        
    # Gesamtanzahl der Zeilen prominent in der Mitte platzieren
    ax.text(
        0, 0, 
        f'Gesamt\n{total_cnt:,}\nZeilen', 
        ha='center', va='center', 
        fontsize=12, fontweight='bold',
        color='#1f2937'
    )
    
    plt.title("Systemstabilität & Datenqualität des Gym-Datensatzes", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('ETL/reports/data_quality_pie.png', dpi=300, transparent=True)
    plt.close()
    
    # =========================================================================
    # 2. Berechnung & Ausgabe der Terminal-KPIs (Farbiger Bericht)
    # =========================================================================
    wochentage = {0: 'Montag', 1: 'Dienstag', 2: 'Mittwoch', 3: 'Donnerstag', 4: 'Freitag', 5: 'Samstag', 6: 'Sonntag'}
    start_date, end_date = df_diag['timestamp'].min(), df_diag['timestamp'].max()
    
    # Erfassungszeitraum berechnen
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    temp_date = start_date + pd.DateOffset(months=total_months)
    if temp_date > end_date:
        total_months -= 1
        temp_date = start_date + pd.DateOffset(months=total_months)
    days_diff = (end_date - temp_date).days
    minutes_diff = (end_date - temp_date).seconds // 60
    stamp_str = f"{total_months:02d}:{days_diff:02d}:{minutes_diff:04d}"
    
    # Terminal-Ausgabe Header
    print("\n" + "="*60)
    print(f"{C_BOLD}{C_MAGENTA} DIAGNOSTICS & DATASET KPIs{C_RESET}")
    print("="*60)
    
    print(f"Datenerfassungs-Dauer:")
    print(f"   - Format (MM:DD:MMMM):     {C_BOLD}{C_GREEN}{stamp_str}{C_RESET}")
    print(f"   - Ausgeschrieben:          {C_GREEN}{total_months} Monate, {days_diff} Tage, {minutes_diff} Minuten{C_RESET}")
    print(f"   - Von:                     {C_CYAN}{wochentage[start_date.dayofweek]}, {start_date.strftime('%d.%m.%Y %H:%M')}{C_RESET}")
    print(f"   - Bis:                     {C_CYAN}{wochentage[end_date.dayofweek]}, {end_date.strftime('%d.%m.%Y %H:%M')}{C_RESET}")
    print("-" * 60)
    
    # A. Auslastungs-KPIs
    util_col = 'utilization_percent'
    mean_all = df_diag[util_col].mean()
    mean_day = df_diag[df_diag['is_day'] == 1][util_col].mean()
    mean_night = df_diag[df_diag['is_day'] == 0][util_col].mean()
    
    print(f"1. Durchschnittliche Auslastung:")
    print(f"   - Gesamt:                  {C_BLUE}{mean_all:.2f}%{C_RESET}")
    print(f"   - Tag (is_day=1):          {C_GREEN}{mean_day:.2f}%{C_RESET}" if not pd.isna(mean_day) else "   - Tag: N/A")
    print(f"   - Nacht (is_day=0):        {C_YELLOW}{mean_night:.2f}%{C_RESET}" if not pd.isna(mean_night) else "   - Nacht: N/A")
    
    is_weekend = df_diag['timestamp'].dt.dayofweek >= 5
    mean_weekday = df_diag[~is_weekend][util_col].mean()
    mean_weekend = df_diag[is_weekend][util_col].mean()
    is_any_holiday = (df_diag['is_public_holiday'] == 1) | (df_diag['is_school_holiday'] == 1)
    mean_holiday = df_diag[is_any_holiday][util_col].mean()
    mean_regular = df_diag[~is_any_holiday][util_col].mean()
    
    print(f"2. Auslastung nach Wochentag & Feiertag:")
    print(f"   - Werktage (Mo-Fr):        {C_BLUE}{mean_weekday:.2f}%{C_RESET}" if not pd.isna(mean_weekday) else "   - Werktage: N/A")
    print(f"   - Wochenende (Sa-So):      {C_GREEN}{mean_weekend:.2f}%{C_RESET}" if not pd.isna(mean_weekend) else "   - Wochenende: N/A")
    print(f"   - Ferien/Feiertage:        {C_YELLOW}{mean_holiday:.2f}%{C_RESET}" if not pd.isna(mean_holiday) else "   - Ferien/Feiertage: N/A")
    print(f"   - Reguläre Tage:           {C_BLUE}{mean_regular:.2f}%{C_RESET}" if not pd.isna(mean_regular) else "   - Reguläre Tage: N/A")
    
    max_idx = df_diag[util_col].idxmax()
    if not pd.isna(max_idx):
        max_row = df_diag.loc[max_idx]
        max_val = max_row[util_col]
        max_time = max_row['timestamp']
        max_day = wochentage[max_time.dayofweek]
        print(f"3. Peak-Auslastung:")
        print(f"   - {C_RED}{C_BOLD}{max_val:.2f}%{C_RESET} am {C_CYAN}{max_day}, {max_time.strftime('%d.%m.%Y um %H:%M')}{C_RESET} Uhr")
    else:
        print("3. Peak-Auslastung: N/A")
        
    min_idx = df_diag[util_col].idxmin()
    if not pd.isna(min_idx):
        min_row = df_diag.loc[min_idx]
        min_val = min_row[util_col]
        min_time = min_row['timestamp']
        min_day = wochentage[min_time.dayofweek]
        print(f"4. Minimale Auslastung:")
        print(f"   - Absolute:                {C_BLUE}{min_val:.2f}%{C_RESET} am {C_CYAN}{min_day}, {min_time.strftime('%d.%m.%Y um %H:%M')}{C_RESET} Uhr")
        
        day_df = df_diag[df_diag['is_day'] == 1]
        if not day_df.empty:
            min_day_idx = day_df[util_col].idxmin()
            min_day_row = day_df.loc[min_day_idx]
            min_day_val = min_day_row[util_col]
            min_day_time = min_day_row['timestamp']
            min_day_name = wochentage[min_day_time.dayofweek]
            print(f"   - Tagsüber min:            {C_BLUE}{min_day_val:.2f}%{C_RESET} am {C_CYAN}{min_day_name}, {min_day_time.strftime('%d.%m.%Y um %H:%M')}{C_RESET} Uhr")
    else:
        print("4. Minimale Auslastung: N/A")
        
    hourly_avg = df_diag.groupby(df_diag['timestamp'].dt.hour)[util_col].mean()
    if not hourly_avg.empty:
        best_hour = hourly_avg.idxmax()
        worst_hour = hourly_avg.idxmin()
        print(f"5. Stündliche Auslastungstrends:")
        print(f"   - Rush Hour:               {C_RED}{best_hour:02d}:00 - {best_hour+1:02d}:00{C_RESET} Uhr (Ø {C_RED}{hourly_avg[best_hour]:.2f}%{C_RESET})")
        print(f"   - Ruhigste Stunde:         {C_GREEN}{worst_hour:02d}:00 - {worst_hour+1:02d}:00{C_RESET} Uhr (Ø {C_GREEN}{hourly_avg[worst_hour]:.2f}%{C_RESET})")
    else:
        print("5. Stündliche Auslastungstrends: N/A")
        
    print("-" * 60)
    
    # B. Wetter-KPIs
    if 'temp' in df_diag.columns:
        max_t_idx = df_diag['temp'].idxmax()
        min_t_idx = df_diag['temp'].idxmin()
        if not pd.isna(max_t_idx) and not pd.isna(min_t_idx):
            max_t_row = df_diag.loc[max_t_idx]
            min_t_row = df_diag.loc[min_t_idx]
            print(f"6. Extremtemperaturen:")
            print(f"   - Maximum:                 {C_RED}{max_t_row['temp']:.1f}°C{C_RESET} am {C_CYAN}{wochentage[max_t_row['timestamp'].dayofweek]}, {max_t_row['timestamp'].strftime('%d.%m.%Y um %H:%M')}{C_RESET} Uhr")
            print(f"   - Minimum:                 {C_BLUE}{min_t_row['temp']:.1f}°C{C_RESET} am {C_CYAN}{wochentage[min_t_row['timestamp'].dayofweek]}, {min_t_row['timestamp'].strftime('%d.%m.%Y um %H:%M')}{C_RESET} Uhr")
    else:
        print("6. Extremtemperaturen: N/A")
        
    if 'apparent_temp' in df_diag.columns and 'temp' in df_diag.columns:
        df_diag['abs_temp_diff'] = (df_diag['apparent_temp'] - df_diag['temp']).abs()
        max_diff_idx = df_diag['abs_temp_diff'].idxmax()
        if not pd.isna(max_diff_idx):
            max_diff_row = df_diag.loc[max_diff_idx]
            print(f"7. Maximale Temperaturdifferenz (Gefühlt vs. Real):")
            print(f"   - Differenz:               {C_RED}{max_diff_row['abs_temp_diff']:.1f}°C{C_RESET} am {C_CYAN}{wochentage[max_diff_row['timestamp'].dayofweek]}, {max_diff_row['timestamp'].strftime('%d.%m.%Y um %H:%M')}{C_RESET} Uhr")
            print(f"     (Real: {max_diff_row['temp']:.1f}°C | Gefühlt: {max_diff_row['apparent_temp']:.1f}°C)")
    else:
        print("7. Maximale Temperaturdifferenz: N/A")
        
    if 'rain' in df_diag.columns:
        max_rain_idx = df_diag['rain'].idxmax()
        if not pd.isna(max_rain_idx):
            max_rain_row = df_diag.loc[max_rain_idx]
            rain_ratio = (df_diag['rain'] > 0).mean() * 100
            print(f"8. Niederschlags-Statistik:")
            print(f"   - Stärkster Regen:         {C_RED}{max_rain_row['rain']:.2f} mm/h{C_RESET} am {C_CYAN}{wochentage[max_rain_row['timestamp'].dayofweek]}, {max_rain_row['timestamp'].strftime('%d.%m.%Y um %H:%M')}{C_RESET} Uhr")
            print(f"   - Regen-Wahrscheinlichkeit: {C_BLUE}{rain_ratio:.1f}%{C_RESET} aller Intervalle mit Niederschlag")
    else:
        print("8. Niederschlags-Statistik: N/A")
        
    print("-" * 60)
    
    # C. Datenqualität & Systemstabilität
    if 'is_imputed' in df_diag.columns and 'gap_duration' in df_diag.columns:
        max_gap = df_diag['gap_duration'].max()
        imputation_rate = df_diag['is_imputed'].mean() * 100
        is_imputed_series = df_diag['is_imputed']
        num_outages = (is_imputed_series.diff() == 1).sum() + (1 if len(is_imputed_series) > 0 and is_imputed_series.iloc[0] == 1 else 0)
        
        print(f"9. Längster API-Ausfall (Datenlücke):")
        if max_gap > 0:
            max_gap_idx = df_diag['gap_duration'].idxmax()
            end_time = df_diag.loc[max_gap_idx, 'timestamp']
            start_time = end_time - pd.Timedelta(minutes=int(max_gap) * 5)
            gap_hours = (int(max_gap) * 5) / 60
            print(f"   - Dauer:                   {C_RED}{gap_hours:.1f} Stunden{C_RESET} ({int(max_gap)} Intervalle)")
            print(f"   - Zeitraum:                von {C_CYAN}{wochentage[start_time.dayofweek]}, {start_time.strftime('%d.%m.%Y %H:%M')}{C_RESET} bis {C_CYAN}{wochentage[end_time.dayofweek]}, {end_time.strftime('%d.%m.%Y %H:%M')}{C_RESET}")
        else:
            print("   - Keine nennenswerten API-Ausfälle / Lücken im Datensatz.")
            
        print(f"10. Gesamt-Ausfallrate (Imputations-Quote):")
        print(f"    - Imputiert:              {C_RED}{imputation_rate:.2f}%{C_RESET} aller Intervalle im Datensatz")
        
        print(f"11. Anzahl der Systemausfälle:")
        print(f"    - Segmente gesamt:        {C_RED}{num_outages}{C_RESET} separate Ausfall-Blöcke")
        if num_outages > 0:
            avg_outage_min = (is_imputed_series.sum() * 5) / num_outages
            print(f"    - Mittlere Ausfalldauer:  {C_YELLOW}{avg_outage_min:.1f} Minuten{C_RESET}")
    else:
        print("9-11. Datenqualitäts-KPIs: N/A")
        
    print("="*60 + "\n")
    
    # =========================================================================
    # 3. Markdown-Bericht schreiben (ETL/reports/diagnostics_report.md)
    # =========================================================================
    # Dieser Bericht wird im Hintergrund gespeichert, um die Anforderungen an eine
    # schriftliche Projektabgabe (Lecture Match KI_01) zu erfüllen.
    with open('ETL/reports/diagnostics_report.md', 'w', encoding='utf-8') as f:
        f.write("# ETL Diagnose & Datensatz-KPIs\n\n")
        f.write(f"*Generiert am: {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M:%S')}*\n\n")
        
        f.write("Dieses Dokument enthält die statistischen Kennzahlen des Datensatzes nach Abschluss des ETL-Prozesses. ")
        f.write("Die Bereinigung und das Feature-Engineering erfolgten methodisch auf Basis der Vorlesungsinhalte ")
        f.write("zur Sicherung der Datenqualität und Vermeidung von Data Leakage.\n\n")
        
        f.write("## 1. Datensatz-Dimensionen & Erfassungszeitraum\n")
        f.write(f"- **Erfassungsdauer (MM:DD:Min):** `{stamp_str}`\n")
        f.write(f"- **Beginn der Aufzeichnung:** `{start_date.strftime('%d.%m.%Y %H:%M')}` Uhr\n")
        f.write(f"- **Ende der Aufzeichnung:** `{end_date.strftime('%d.%m.%Y %H:%M')}` Uhr\n")
        f.write(f"- **Rohdaten-Zeilenanzahl:** `{total_cnt:,}`\n")
        f.write(f"- **Bereinigte Zeilen (ML-ready):** `{len(df):,}`\n")
        f.write(f"- **Verworfene Zeilen aufgrund von Großlücken (> 2h):** `{drop_cnt:,}` (Vermeidung von Verzerrung durch zu lange Interpolationen)\n\n")
        
        f.write("## 2. Deskriptive Auslastungs-Statistik\n")
        f.write(f"- **Gesamte durchschnittliche Auslastung:** `{mean_all:.2f}%`\n")
        if not pd.isna(mean_day):
            f.write(f"- **Durchschnittliche Auslastung tagsüber (is_day=1):** `{mean_day:.2f}%`\n")
        if not pd.isna(mean_night):
            f.write(f"- **Durchschnittliche Auslastung nachts (is_day=0):** `{mean_night:.2f}%`\n")
        if not pd.isna(max_idx):
            f.write(f"- **Peak-Auslastung:** `{max_val:.2f}%` am `{max_time.strftime('%d.%m.%Y um %H:%M')}` Uhr ({wochentage[max_time.dayofweek]})\n\n")
        
        f.write("## 3. Wetter & Externe Parameter\n")
        if 'temp' in df_diag.columns:
            f.write(f"- **Temperaturbereich:** `{df_diag['temp'].min():.1f}°C` bis `{df_diag['temp'].max():.1f}°C`\n")
        if 'apparent_temp' in df_diag.columns and 'temp' in df_diag.columns:
            f.write(f"- **Maximale gefühlte Temperaturabweichung (apparent - real):** `{df_diag['abs_temp_diff'].max():.1f}°C`\n")
        if 'rain' in df_diag.columns:
            f.write(f"- **Maximaler Niederschlag:** `{df_diag['rain'].max():.2f} mm/h`\n")
            f.write(f"- **Niederschlagshäufigkeit (Regen-Intervalle):** `{rain_ratio:.1f}%` aller erfassten Zeitpunkte\n\n")
        
        f.write("## 4. Datenqualität & Systemstabilität (API-Ausfälle)\n")
        if 'is_imputed' in df_diag.columns and 'gap_duration' in df_diag.columns:
            f.write(f"- **Gesamt-Imputationsquote (Ausfallquote):** `{imputation_rate:.2f}%`\n")
            f.write(f"- **Anzahl detektierter Systemausfälle:** `{num_outages}` separate Ausfall-Blöcke\n")
            f.write(f"- **Mittlere Dauer eines Systemausfalls:** `{avg_outage_min:.1f} Minuten`\n")
            f.write(f"- **Maximale zusammenhängende Datenlücke:** `{(max_gap * 5) / 60:.1f} Stunden` (`{max_gap}` Intervalle)\n\n")
        
        f.write("---\n")
        f.write("*Hinweis: Die Visualisierung dieser Datenqualitäts-KPIs befindet sich unter [data_quality_pie.png](data_quality_pie.png).*")
