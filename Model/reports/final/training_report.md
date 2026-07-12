# Wissenschaftlicher Modell- und Evaluierungsbericht

### 1.1 Trainings-Statistiken & Laufzeiten
- **Tuning-Modus:** Final Tuning (Produktion)
- **Kreuzvalidierung:** 5-fold TimeSeriesSplit
- **Gesamte Trainingsläufe:** 5614 Modell-Fits (inkl. Kreuzvalidierungs-Splits & finalem Refit)
- **Gesamtlaufzeit des Trainings:** 7.86 Minuten

## 1. Methodischer Versuchsaufbau & Vorlesungsbezug
- **Data Leakage Prevention (KI_03):** Die Standardisierung mittels `StandardScaler` wurde strikt nach dem Train-Test-Split durchgeführt. Fit nur auf Train-Daten. Binäre Spalten blieben unskaliert.
- **TimeSeriesSplit (KI_03):** Sequentielles Validierungsverfahren zur Wahrung der Zeitkausalität.
- **Multi-Modell-Architektur (KI_03 & KI_04):** Da sich die Aussagekraft von historischen Lags über längere Vorhersagehorizonte verschiebt (Degradation), wurde für jeden Zeithorizont ein eigenständiges Modell trainiert.
- **Modellspeicherung:** Gespeichert in getrennten Unterordnern unter `Model/models/final/` (Quick und Detailed 5m bis 120m) für maximale Übersicht.
- **1-Jahres-Fenster (Sliding Window):** Um Concept Drift zu minimieren und Skalierbarkeit zu sichern, wurde das Training auf die Daten der letzten 365 Tage beschränkt.

## 2. Ergebnisse für Modus: Quick (Tagesplaner)
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6555 | 4.2774 | 18.2962 | 3.1856 | `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}` |
| GradientBoosting | 0.6244 | 4.4668 | 19.9526 | 3.1768 | `{'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 200, 'subsample': 1.0}` |
| GBQuantile90 | 0.2116 | 6.4713 | 41.8773 | 4.7630 | `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}` |
| XGBoost | 0.6237 | 4.4704 | 19.9848 | 3.2707 | `{'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.8}` |
| Ridge | 0.4456 | 5.4264 | 29.4460 | 3.9786 | `{'alpha': 100.0}` |
| Lasso | 0.4205 | 5.5482 | 30.7820 | 4.1417 | `{'alpha': 1.0}` |
| ElasticNet | 0.4778 | 5.2666 | 27.7375 | 3.8275 | `{'alpha': 0.1, 'l1_ratio': 0.8}` |

**Bestes Modell für Quick:** RandomForest ($R^2 = 0.6555$).

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
| RandomForest | 0.9602 | 1.4545 | 2.1154 | 0.9357 | `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}` |
| GradientBoosting | 0.9617 | 1.4260 | 2.0336 | 0.8950 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| GBQuantile90 | 0.9071 | 2.2213 | 4.9340 | 1.7107 | `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}` |
| XGBoost | 0.9614 | 1.4321 | 2.0510 | 0.9004 | `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}` |
| Ridge | 0.9608 | 1.4435 | 2.0837 | 0.9109 | `{'alpha': 0.1}` |
| Lasso | 0.9610 | 1.4391 | 2.0709 | 0.8893 | `{'alpha': 0.01}` |
| ElasticNet | 0.9610 | 1.4397 | 2.0726 | 0.9019 | `{'alpha': 0.001, 'l1_ratio': 0.8}` |

**Bestes Modell für 5m:** GradientBoosting ($R^2 = 0.9617$).

### 3.3 Zeithorizont: 15 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.8900 | 2.4171 | 5.8426 | 1.6834 | `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 500}` |
| GradientBoosting | 0.8915 | 2.4009 | 5.7643 | 1.6680 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| GBQuantile90 | 0.7406 | 3.7120 | 13.7791 | 2.8216 | `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200}` |
| XGBoost | 0.8920 | 2.3956 | 5.7388 | 1.6656 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| Ridge | 0.8852 | 2.4693 | 6.0976 | 1.7503 | `{'alpha': 1.0}` |
| Lasso | 0.8875 | 2.4443 | 5.9747 | 1.6927 | `{'alpha': 0.1}` |
| ElasticNet | 0.8879 | 2.4399 | 5.9530 | 1.7114 | `{'alpha': 0.01, 'l1_ratio': 0.8}` |

**Bestes Modell für 15m:** XGBoost ($R^2 = 0.8920$).

### 3.6 Zeithorizont: 30 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.7785 | 3.4296 | 11.7621 | 2.4181 | `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 500}` |
| GradientBoosting | 0.8025 | 3.2387 | 10.4895 | 2.2778 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 1.0}` |
| GBQuantile90 | 0.5786 | 4.7302 | 22.3746 | 3.6054 | `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}` |
| XGBoost | 0.8043 | 3.2237 | 10.3925 | 2.2782 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| Ridge | 0.7860 | 3.3707 | 11.3619 | 2.4578 | `{'alpha': 100.0}` |
| Lasso | 0.7900 | 3.3392 | 11.1503 | 2.3898 | `{'alpha': 0.1}` |
| ElasticNet | 0.7910 | 3.3312 | 11.0969 | 2.3919 | `{'alpha': 0.1, 'l1_ratio': 0.8}` |

**Bestes Modell für 30m:** XGBoost ($R^2 = 0.8043$).

### 3.12 Zeithorizont: 60 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6785 | 4.1310 | 17.0649 | 2.9905 | `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 500}` |
| GradientBoosting | 0.7031 | 3.9699 | 15.7599 | 2.8266 | `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0}` |
| GBQuantile90 | 0.3596 | 5.8304 | 33.9937 | 4.4900 | `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}` |
| XGBoost | 0.7082 | 3.9356 | 15.4893 | 2.8088 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| Ridge | 0.6806 | 4.1176 | 16.9545 | 2.9923 | `{'alpha': 1000.0}` |
| Lasso | 0.6808 | 4.1163 | 16.9443 | 2.9867 | `{'alpha': 0.1}` |
| ElasticNet | 0.6830 | 4.1022 | 16.8284 | 2.9791 | `{'alpha': 0.1, 'l1_ratio': 0.8}` |

**Bestes Modell für 60m:** XGBoost ($R^2 = 0.7082$).

### 3.18 Zeithorizont: 90 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6707 | 4.1802 | 17.4737 | 3.0500 | `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}` |
| GradientBoosting | 0.6869 | 4.0763 | 16.6159 | 2.9248 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| GBQuantile90 | 0.3749 | 5.7597 | 33.1746 | 4.2597 | `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}` |
| XGBoost | 0.6895 | 4.0595 | 16.4794 | 2.9066 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 1.0}` |
| Ridge | 0.6474 | 4.3258 | 18.7125 | 3.1758 | `{'alpha': 1000.0}` |
| Lasso | 0.6294 | 4.4351 | 19.6699 | 3.3137 | `{'alpha': 1.0}` |
| ElasticNet | 0.6284 | 4.4407 | 19.7202 | 3.3171 | `{'alpha': 1.0, 'l1_ratio': 0.8}` |

**Bestes Modell für 90m:** XGBoost ($R^2 = 0.6895$).

### 3.24 Zeithorizont: 120 Minuten
| Modell | R²-Score | RMSE | MSE | MAE | Beste Hyperparameter |
| :--- | :---: | :---: | :---: | :---: | :--- |
| RandomForest | 0.6551 | 4.2780 | 18.3009 | 3.1543 | `{'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 500}` |
| GradientBoosting | 0.6830 | 4.1010 | 16.8185 | 2.9597 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| GBQuantile90 | 0.2904 | 6.1357 | 37.6472 | 4.7427 | `{'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 200}` |
| XGBoost | 0.6772 | 4.1382 | 17.1250 | 2.9771 | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}` |
| Ridge | 0.6317 | 4.4203 | 19.5391 | 3.2520 | `{'alpha': 1000.0}` |
| Lasso | 0.6116 | 4.5395 | 20.6067 | 3.3970 | `{'alpha': 1.0}` |
| ElasticNet | 0.6114 | 4.5405 | 20.6159 | 3.3938 | `{'alpha': 1.0, 'l1_ratio': 0.8}` |

**Bestes Modell für 120m:** GradientBoosting ($R^2 = 0.6830$).

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

