# ETL Diagnose & Datensatz-KPIs

*Generiert am: 09.06.2026 21:33:14*

Dieses Dokument enthält die statistischen Kennzahlen des Datensatzes nach Abschluss des ETL-Prozesses. Die Bereinigung und das Feature-Engineering erfolgten methodisch auf Basis der Vorlesungsinhalte zur Sicherung der Datenqualität und Vermeidung von Data Leakage.

## 1. Datensatz-Dimensionen & Erfassungszeitraum
- **Erfassungsdauer (MM:DD:Min):** `01:18:0710`
- **Beginn der Aufzeichnung:** `16.04.2026 23:15` Uhr
- **Ende der Aufzeichnung:** `04.06.2026 11:05` Uhr
- **Rohdaten-Zeilenanzahl:** `13,967`
- **Bereinigte Zeilen (ML-ready):** `11,723`
- **Verworfene Zeilen aufgrund von Großlücken (> 2h):** `2,244` (Vermeidung von Verzerrung durch zu lange Interpolationen)

## 2. Deskriptive Auslastungs-Statistik
- **Gesamte durchschnittliche Auslastung:** `10.64%`
- **Durchschnittliche Auslastung tagsüber (is_day=1):** `14.84%`
- **Durchschnittliche Auslastung nachts (is_day=0):** `3.83%`
- **Peak-Auslastung:** `48.75%` am `20.05.2026 um 16:20` Uhr (Mittwoch)

## 3. Wetter & Externe Parameter
- **Temperaturbereich:** `3.5°C` bis `32.7°C`
- **Maximale gefühlte Temperaturabweichung (apparent - real):** `5.2°C`
- **Maximaler Niederschlag:** `5.20 mm/h`
- **Niederschlagshäufigkeit (Regen-Intervalle):** `5.4%` aller erfassten Zeitpunkte

## 4. Datenqualität & Systemstabilität (API-Ausfälle)
- **Gesamt-Imputationsquote (Ausfallquote):** `17.61%`
- **Anzahl detektierter Systemausfälle:** `2285` separate Ausfall-Blöcke
- **Mittlere Dauer eines Systemausfalls:** `5.4 Minuten`
- **Maximale zusammenhängende Datenlücke:** `2.8 Stunden` (`34` Intervalle)

---
*Hinweis: Die Visualisierung dieser Datenqualitäts-KPIs befindet sich unter [data_quality_pie.png](data_quality_pie.png).*