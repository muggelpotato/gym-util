# Wissenschaftlicher Modell- und Evaluierungsbericht

### 1.1 Trainings-Statistiken & Laufzeiten
- **Tuning-Modus:** Debug Tuning (Schnelllauf)
- **Kreuzvalidierung:** 5-fold TimeSeriesSplit
- **Gesamte Trainingsläufe:** 294 Modell-Fits (inkl. Kreuzvalidierungs-Splits & finalem Refit)
- **Gesamtlaufzeit des Trainings:** 0.64 Minuten

## 1. Methodischer Versuchsaufbau & Vorlesungsbezug
- **Data Leakage Prevention (KI_03):** Die Standardisierung mittels `StandardScaler` wurde strikt nach dem Train-Test-Split durchgeführt. Fit nur auf Train-Daten. Binäre Spalten blieben unskaliert.
- **TimeSeriesSplit (KI_03):** Sequentielles Validierungsverfahren zur Wahrung der Zeitkausalität.
- **Multi-Modell-Architektur (KI_03 & KI_04):** Da sich die Aussagekraft von historischen Lags über längere Vorhersagehorizonte verschiebt (Degradation), wurde für jeden Zeithorizont ein eigenständiges Modell trainiert.
- **Modellspeicherung:** Gespeichert in getrennten Unterordnern unter `Model/models/final/` (Quick und Detailed 5m bis 120m) für maximale Übersicht.
- **1-Jahres-Fenster (Sliding Window):** Um Concept Drift zu minimieren und Skalierbarkeit zu sichern, wurde das Training auf die Daten der letzten 365 Tage beschränkt.

## 2. Ergebnisse für Modus: Quick (Tagesplaner)
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6559 | 4.2751 | 18.2767 | 3.1886 | `{'max_depth': 10, 'n_estimators': 50}` |
| GradientBoosting | 0.6363 | 4.3954 | 19.3195 | 3.2061 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| GBQuantile90 | 0.0327 | 7.1680 | 51.3807 | 5.5552 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| XGBoost | 0.6377 | 4.3865 | 19.2413 | 3.2054 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| Ridge | 0.4184 | 5.5579 | 30.8900 | 4.0967 | `{'alpha': 1.0}` |
| Lasso | 0.4718 | 5.2969 | 28.0567 | 3.8528 | `{'alpha': 0.1}` |
| ElasticNet | 0.4851 | 5.2299 | 27.3517 | 3.7991 | `{'alpha': 0.1, 'l1_ratio': 0.5}` |

**Bestes Modell für Quick:** RandomForest ($R^2 = 0.6559$).

### Stabilitätsanalyse über die Kreuzvalidierungs-Folds
Die folgende Grafik zeigt die Leistungsschwankungen der Ostele-Modelle über die 5 chronologischen Folds des TimeSeriesSplits. Dies dient der Validierung der zeitlichen Stabilität der Vorhersagen:

![Fold-Performance-Vergleich](horizon/fold_performance_comparison.png)

### XAI-Analysen (Quick)
| Feature Importance (Baumbasiert) | Koeffizienten (Linear) |
| :---: | :---: |
| ![Feature Importance (Baumbasiert)](xai/quick/feature_importance_quick.png) | ![Koeffizienten (Linear)](xai/quick/coefficients_quick.png) |

## 3. Ergebnisse für Modus: Detailed (Live-Tracker)
Hier wurden die Modelle auf unterschiedliche Vorhersagehorizonte (5m bis 120m) trainiert. Man sieht die Degradation.

### Stabilitätsanalyse der Detailed-Modelle über die Folds
Die folgenden Gitter-Grafiken vergleichen die Leistungsschwankungen aller 6 detailed Modelle über die 5 chronologischen Folds für alle sechs Zeithorizonte:

| Stabilität R²-Score | Stabilität Vorhersagefehler (RMSE) |
| :---: | :---: |
| ![Fold-Stabilität R² Detailed](horizon/fold_performance_detailed_r2.png) | ![Fold-Stabilität RMSE Detailed](horizon/fold_performance_detailed_rmse.png) |

### Übergreifende XAI-Analysen (Detailed)
Die folgenden Gitter-Grafiken vergleichen die Feature-Importances (Baumbasiert) und Regressions-Gewichte (Linear) über alle sechs Horizonte hinweg:

| Feature Importance (Baumbasiert) | Koeffizienten (Linear) |
| :---: | :---: |
| ![Feature Importance Gitter](xai/detailed/feature_importance_detailed_grid.png) | ![Koeffizienten Gitter](xai/detailed/coefficients_detailed_grid.png) |

### 3.1 Zeithorizont: 5 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.9601 | 1.4560 | 2.1200 | 0.9340 | `{'max_depth': 10, 'n_estimators': 50}` |
| GradientBoosting | 0.9615 | 1.4304 | 2.0460 | 0.8972 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| GBQuantile90 | 0.8918 | 2.3969 | 5.7452 | 2.0109 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| XGBoost | 0.9615 | 1.4301 | 2.0453 | 0.8992 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| Ridge | 0.9608 | 1.4433 | 2.0832 | 0.9109 | `{'alpha': 1.0}` |
| Lasso | 0.9609 | 1.4419 | 2.0791 | 0.8848 | `{'alpha': 0.1}` |
| ElasticNet | 0.9555 | 1.5377 | 2.3645 | 1.0320 | `{'alpha': 0.1, 'l1_ratio': 0.5}` |

**Bestes Modell für 5m:** XGBoost ($R^2 = 0.9615$).

### 3.3 Zeithorizont: 15 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.8889 | 2.4290 | 5.9003 | 1.6970 | `{'max_depth': 10, 'n_estimators': 50}` |
| GradientBoosting | 0.8922 | 2.3932 | 5.7272 | 1.6632 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| GBQuantile90 | 0.7269 | 3.8088 | 14.5072 | 3.0158 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| XGBoost | 0.8925 | 2.3897 | 5.7107 | 1.6648 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| Ridge | 0.8852 | 2.4693 | 6.0976 | 1.7503 | `{'alpha': 1.0}` |
| Lasso | 0.8875 | 2.4443 | 5.9747 | 1.6927 | `{'alpha': 0.1}` |
| ElasticNet | 0.8839 | 2.4835 | 6.1679 | 1.7554 | `{'alpha': 0.1, 'l1_ratio': 0.5}` |

**Bestes Modell für 15m:** XGBoost ($R^2 = 0.8925$).

### 3.6 Zeithorizont: 30 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.7735 | 3.4678 | 12.0259 | 2.4518 | `{'max_depth': 10, 'n_estimators': 50}` |
| GradientBoosting | 0.8020 | 3.2425 | 10.5140 | 2.2853 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| GBQuantile90 | 0.5403 | 4.9409 | 24.4127 | 3.8816 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| XGBoost | 0.8035 | 3.2301 | 10.4337 | 2.2827 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| Ridge | 0.7817 | 3.4047 | 11.5920 | 2.4836 | `{'alpha': 1.0}` |
| Lasso | 0.7900 | 3.3392 | 11.1503 | 2.3898 | `{'alpha': 0.1}` |
| ElasticNet | 0.7907 | 3.3339 | 11.1150 | 2.4032 | `{'alpha': 0.1, 'l1_ratio': 0.5}` |

**Bestes Modell für 30m:** XGBoost ($R^2 = 0.8035$).

### 3.12 Zeithorizont: 60 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6704 | 4.1832 | 17.4990 | 3.0472 | `{'max_depth': 10, 'n_estimators': 50}` |
| GradientBoosting | 0.7036 | 3.9664 | 15.7321 | 2.8185 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| GBQuantile90 | 0.2651 | 6.2458 | 39.0106 | 5.0007 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| XGBoost | 0.6995 | 3.9942 | 15.9533 | 2.8439 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| Ridge | 0.6536 | 4.2884 | 18.3900 | 3.1468 | `{'alpha': 1.0}` |
| Lasso | 0.6808 | 4.1163 | 16.9443 | 2.9867 | `{'alpha': 0.1}` |
| ElasticNet | 0.6843 | 4.0938 | 16.7594 | 2.9755 | `{'alpha': 0.1, 'l1_ratio': 0.5}` |

**Bestes Modell für 60m:** GradientBoosting ($R^2 = 0.7036$).

### 3.18 Zeithorizont: 90 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6616 | 4.2381 | 17.9613 | 3.0975 | `{'max_depth': 10, 'n_estimators': 50}` |
| GradientBoosting | 0.6799 | 4.1217 | 16.9886 | 2.9593 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| GBQuantile90 | 0.2141 | 6.4580 | 41.7056 | 5.1825 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| XGBoost | 0.6800 | 4.1211 | 16.9838 | 2.9601 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| Ridge | 0.5983 | 4.6170 | 21.3163 | 3.4460 | `{'alpha': 1.0}` |
| Lasso | 0.6507 | 4.3054 | 18.5361 | 3.1722 | `{'alpha': 0.1}` |
| ElasticNet | 0.6535 | 4.2885 | 18.3911 | 3.1520 | `{'alpha': 0.1, 'l1_ratio': 0.5}` |

**Bestes Modell für 90m:** XGBoost ($R^2 = 0.6800$).

### 3.24 Zeithorizont: 120 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6379 | 4.3830 | 19.2103 | 3.2449 | `{'max_depth': 10, 'n_estimators': 50}` |
| GradientBoosting | 0.6714 | 4.1752 | 17.4322 | 3.0201 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| GBQuantile90 | 0.2226 | 6.4223 | 41.2465 | 5.2135 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| XGBoost | 0.6711 | 4.1770 | 17.4476 | 3.0106 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}` |
| Ridge | 0.5682 | 4.7864 | 22.9093 | 3.5854 | `{'alpha': 1.0}` |
| Lasso | 0.6352 | 4.3993 | 19.3541 | 3.2506 | `{'alpha': 0.1}` |
| ElasticNet | 0.6382 | 4.3810 | 19.1933 | 3.2259 | `{'alpha': 0.1, 'l1_ratio': 0.5}` |

**Bestes Modell für 120m:** GradientBoosting ($R^2 = 0.6714$).

## 4. Übergreifende Analysen & Erkenntnisse (Vorlesungsbezug KI_03)

### 4.1 Modell-Wettstreit über alle Horizonte (R²-Score & absolute Fehler)
| Modell-Vergleich (R²) | Modell-Fehlervergleich (RMSE/MAE) |
| :---: | :---: |
| ![Modell-Vergleich (R²)](horizon/model_comparison_detailed.png) | ![Modell-Fehlervergleich (RMSE/MAE)](horizon/model_error_comparison_detailed.png) |

### 4.2 Wandel der Feature-Wichtigkeit (XAI Decay) & Regressionsgewichte
| Wandel der Feature-Wichtigkeit (XAI Decay) | Koeffizienten-Verlauf (ElasticNet) |
| :---: | :---: |
| ![Feature-Importance-Decay](xai/analysis/feature_importance_detailed.png) | ![Koeffizienten-Verlauf](xai/analysis/coefficients_detailed.png) |

## 5. Hyperparameter-Tuning & Effizienzanalyse (Vorlesungsbezug KI_03)

Um die optimale Konfiguration der Modelle zu finden, wurde eine systematische Grid-Search über einen definiertem Suchraum durchgeführt. Die folgenden Grafiken analysieren die Robustheit und Effizienz der Modelle in beiden Modi:

| Modus | Tuning-Effizienz (Genauigkeit vs. Laufzeit) | Tuning-Sensitivität (RMSE-Spannweite) |
| :--- | :---: | :---: |
| **Quick** | ![Tuning-Effizienz Quick](horizon/grid_search_efficiency.png) | ![Tuning-Sensitivität Quick](horizon/grid_search_tuning_impact.png) |
| **Detailed (2x3 Gitter)** | ![Tuning-Effizienz Detailed](horizon/grid_search_efficiency_detailed.png) | ![Tuning-Sensitivität Detailed](horizon/grid_search_tuning_impact_detailed.png) |

- **Tuning-Effizienz (Genauigkeit vs. Laufzeit):** Baumbasierte Ensemble-Methoden (Random Forest, XGBoost) erzielen signifikant niedrigere Fehler (RMSE [%], niedriger ist besser) als lineare Modelle (Ridge, Lasso), benötigen jedoch um Größenordnungen mehr Rechenzeit. Dies veranschaulicht den *Trade-off* zwischen Ressourcenverbrauch und Vorhersagegüte.
- **Tuning-Sensitivität (RMSE-Spannweite):** Zeigt den Einfluss der gewählten Hyperparameter auf die Modellgüte (RMSE [%], niedriger ist besser). Ein großer Boxplot (z. B. bei XGBoost oder Random Forest im produktiven Lauf) verdeutlicht, dass das Modell extrem sensitiv auf Hyperparameter reagiert und eine Grid-Search notwendig ist, um Overfitting zu vermeiden. **Hinweis zur Grafik:** Im Debug-Modus (`FINAL_TUNING = False`) wird pro Modell nur eine Konfiguration getestet, weshalb die Boxen zu einer einzelnen Linie kollabieren. Im produktiven Modus (`FINAL_TUNING = True`) wird die Spannweite sichtbar.

