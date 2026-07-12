import os
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn Theme setzen für konsistenten Stil mit anderen Projekt-Visualisierungen
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

def generate_validation_plot():
    # Erstellung der Figur
    fig, ax = plt.subplots(figsize=(10, 5))

    # Daten für die Folds definieren
    folds = [
        {"name": "Fold 1", "train_end": 5, "val_end": 7},
        {"name": "Fold 2", "train_end": 7, "val_end": 9},
        {"name": "Fold 3", "train_end": 9, "val_end": 11}
    ]

    # Farbpalette passend zu den anderen Plots (Teal für Daily Profile, Royalblue für Scatter Plots)
    color_train = "teal"
    color_val = "royalblue"

    bar_height = 0.45
    y_positions = [3, 2, 1]

    for idx, fold in enumerate(folds):
        y_pos = y_positions[idx]
        train_len = fold["train_end"]
        val_len = fold["val_end"] - fold["train_end"]
        
        # Train bar
        ax.barh(y_pos, train_len, left=0, height=bar_height, color=color_train, 
                edgecolor="none", label="Training: fit_transform() (µ & σ berechnen & anwenden)" if idx == 0 else "")
        
        # Textbeschriftung in Train-Bar
        ax.text(train_len / 2, y_pos, f"Train (0 bis t={train_len})", 
                va='center', ha='center', color='white', fontweight='bold', fontsize=9.5)
        
        # Val bar
        ax.barh(y_pos, val_len, left=train_len, height=bar_height, color=color_val, 
                edgecolor="none", label="Validierung: transform() (µ & σ nur anwenden)" if idx == 0 else "")
        
        # Textbeschriftung in Val-Bar
        ax.text(train_len + val_len / 2, y_pos, "Val", 
                va='center', ha='center', color='white', fontweight='bold', fontsize=9.5)
        
        # Fold Label auf der linken Seite
        ax.text(-0.3, y_pos, fold["name"], va='center', ha='right', fontweight='bold', fontsize=11)

    # Achsen- und Layout-Formattierung
    ax.set_ylim(0.3, 3.7)
    ax.set_xlim(0, 11.2)
    ax.set_yticks([])
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"t={i}" for i in range(12)], fontsize=10)
    ax.set_xlabel("Zeit (Chronologisch)", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_title("Walk-Forward-Validierung (TimeSeriesSplit) & Skalierungs-Logik", fontsize=14, fontweight='bold', pad=15)

    # Legende unterhalb des Plots platzieren
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    
    # Pfad absichern
    os.makedirs("Deskriptive Analyse/reports", exist_ok=True)
    output_path = "Deskriptive Analyse/reports/validation_scaling_concept.png"
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Visualisierung erfolgreich gespeichert unter: {output_path}")

if __name__ == "__main__":
    generate_validation_plot()
