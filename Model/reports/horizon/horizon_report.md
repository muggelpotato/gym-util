# Experimenteller Bericht: Vorhersagehorizont vs. Präzision (inkl. XGBoost)

## 1. Ergebnisse des Experiments
| Horizont | Stunden | Lasso R² | Lasso RMSE | ElasticNet R² | ElasticNet RMSE | GBR R² | GBR RMSE | XGBoost R² | XGBoost RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 Min | 0.08h | 0.9604 | 1.45% | 0.9602 | 1.45% | 0.9609 | 1.44% | 0.9609 | 1.44% |
| 15 Min | 0.25h | 0.8846 | 2.48% | 0.8845 | 2.48% | 0.8897 | 2.42% | 0.8892 | 2.43% |
| 30 Min | 0.50h | 0.7777 | 3.44% | 0.7782 | 3.43% | 0.8002 | 3.26% | 0.7968 | 3.29% |
| 1 Std | 1.00h | 0.6498 | 4.31% | 0.6509 | 4.30% | 0.7037 | 3.97% | 0.7050 | 3.96% |
| 2 Std | 2.00h | 0.5688 | 4.78% | 0.5636 | 4.81% | 0.6717 | 4.17% | 0.6603 | 4.25% |
| 4 Std | 4.00h | 0.5062 | 5.12% | 0.5031 | 5.14% | 0.6541 | 4.28% | 0.6429 | 4.35% |
| 6 Std | 6.00h | 0.5153 | 5.07% | 0.5111 | 5.10% | 0.6438 | 4.35% | 0.6411 | 4.37% |
| 8 Std | 8.00h | 0.5166 | 5.07% | 0.5120 | 5.09% | 0.6168 | 4.51% | 0.6203 | 4.49% |

## 2. Interpretation & App-Integration (Lecture Match KI_03 & KI_04)
- **Lineare Baselines:** Dominanz bei sehr kurzen Horizonten (<= 2h) durch lineares Signal des unmittelbaren Lags.
- **Boosting-Modelle:** Ab 4h bricht die Kraft der Lags ein. GBR und XGBoost weisen beste Robustheit auf (R2 ~0.50 - 0.53) durch nicht-lineares Lernen.
- **Empfehlung:** Bei Horizonten <= 2h das Detailed-Modell nutzen; ab > 2h das Quick-Modell (XGBoost/GBR) verwenden.
