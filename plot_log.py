"""
plot_log.py
Generates training_curve.png from training_log.csv.
Matplotlib is only imported inside this script to keep the rest of the project zero-dependency.
"""

import os
import sys
import csv


def main():
    csv_file = "training_log.csv"
    output_png = "training_curve.png"

    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Run train.py first to generate training logs.")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required to plot the training curve.")
        print("Install it with: pip install matplotlib")
        sys.exit(1)

    generations = []
    best_scores = []
    avg_scores = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row["generation"]))
            best_scores.append(float(row.get("best_fitness", row.get("best_score", 0))))
            avg_scores.append(float(row.get("avg_fitness", row.get("avg_score", 0))))

    if not generations:
        print(f"Error: '{csv_file}' is empty.")
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_scores, label="Best Fitness / Pipes", color="#00d26a", linewidth=2)
    plt.plot(generations, avg_scores, label="Average Fitness / Pipes", color="#3b82f6", linewidth=1.5, linestyle="--")
    plt.title("Neuroevolution Training Curve: Flappy Bird AI", fontsize=14, fontweight="bold")
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Pipes Cleared", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"✅ Training curve saved to {output_png} ({len(generations)} generations plotted).")


if __name__ == "__main__":
    main()
