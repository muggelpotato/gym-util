import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

os.makedirs('ETL/reports', exist_ok=True)

df = pd.read_csv('data/gym_workload_mlready.csv')
auslastung = df['utilization_percent'].dropna()
acf_werte = acf(auslastung, nlags=48)

plt.figure(figsize=(12, 6))
plot_acf(auslastung, lags=48, alpha=0.05, ax=plt.gca(), color='teal')
plt.ylim(0, 1.05)
plt.title('Autokorrelationsfunktion (ACF) der Gym-Auslastung', fontsize=14)
plt.xlabel('Verzögerung in Lags (1 Lag = 5 Minuten)', fontsize=12)
plt.ylabel('Autokorrelationskoeffizient', fontsize=12)

messpunkte = [(1, '5 Minuten', 'green'), (6, '30 Minuten', 'olive'), (12, '1 Stunde', 'red'), (24, '2 Stunden', 'orange'), (48, '4 Stunden', 'purple')]
for lag, name, farbe in messpunkte:
    exakter_wert = acf_werte[lag]
    plt.axvline(x=lag, color=farbe, linestyle='--', alpha=0.7, 
                label=f'{name} (Lag {lag}): {exakter_wert:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('ETL/reports/autokorrelation_gym.png', dpi=300, transparent=True)
plt.close()
