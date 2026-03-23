from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT VISUAL GENERATOR
# Creates charts from model output CSVs in the same folder
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def make_model_comparison_chart():
    file_path = BASE_DIR / "model_comparison_summary_v4.csv"
    if not file_path.exists():
        print(f"Missing file: {file_path.name}")
        return

    df = pd.read_csv(file_path)

    metrics = [
        "accuracy_at_0_50",
        "precision_at_0_50",
        "recall_at_0_50",
        "f1_at_0_50",
        "roc_auc"
    ]

    plot_df = df.set_index("model")[metrics].T

    plt.figure(figsize=(10, 6))
    for model in plot_df.columns:
        plt.plot(plot_df.index, plot_df[model], marker="o", label=model)

    plt.title("Model Performance Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "chart_model_comparison.png", dpi=300)
    plt.close()
    print("Saved: chart_model_comparison.png")


def make_threshold_chart():
    file_path = BASE_DIR / "model_threshold_results_v4.csv"
    if not file_path.exists():
        print(f"Missing file: {file_path.name}")
        return

    df = pd.read_csv(file_path)

    plt.figure(figsize=(10, 6))

    for model in df["model"].unique():
        subset = df[df["model"] == model].sort_values("threshold")
        plt.plot(
            subset["threshold"],
            subset["recall"],
            marker="o",
            label=f"{model} Recall"
        )
        plt.plot(
            subset["threshold"],
            subset["precision"],
            marker="x",
            linestyle="--",
            label=f"{model} Precision"
        )

    plt.title("Precision and Recall vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "chart_threshold_precision_recall.png", dpi=300)
    plt.close()
    print("Saved: chart_threshold_precision_recall.png")


def make_rf_importance_chart():
    file_path = BASE_DIR / "rf_feature_importances_v4.csv"
    if not file_path.exists():
        print(f"Missing file: {file_path.name}")
        return

    df = pd.read_csv(file_path).sort_values("importance", ascending=False).head(15)
    df = df.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(df["feature"], df["importance"])
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "chart_rf_feature_importance.png", dpi=300)
    plt.close()
    print("Saved: chart_rf_feature_importance.png")


def make_lr_coeff_chart():
    file_path = BASE_DIR / "lr_coefficients_v4.csv"
    if not file_path.exists():
        print(f"Missing file: {file_path.name}")
        return

    df = pd.read_csv(file_path).sort_values("abs_coefficient", ascending=False).head(15)
    df = df.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(df["feature"], df["coefficient"])
    plt.title("Top 15 Logistic Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "chart_lr_coefficients.png", dpi=300)
    plt.close()
    print("Saved: chart_lr_coefficients.png")


def make_class_distribution_chart():
    # Use one of the merged sample CSVs to show ARR_DEL15 balance
    sample_files = sorted(BASE_DIR.glob("MERGED_*_100k_sample.csv"))
    if not sample_files:
        print("No MERGED_*_100k_sample.csv file found.")
        return

    file_path = sample_files[0]
    df = pd.read_csv(file_path, low_memory=False)

    if "ARR_DEL15" not in df.columns:
        print("ARR_DEL15 column not found.")
        return

    counts = df["ARR_DEL15"].value_counts(dropna=False)

    labels = []
    values = []

    for key, value in counts.items():
        if pd.isna(key):
            labels.append("Missing")
        elif key == 0:
            labels.append("On Time")
        elif key == 1:
            labels.append("Delayed")
        else:
            labels.append(str(key))
        values.append(value)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Target Class Distribution (ARR_DEL15)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "chart_class_distribution.png", dpi=300)
    plt.close()
    print("Saved: chart_class_distribution.png")

def make_threshold_summary_chart():
    file_path = BASE_DIR / "model_threshold_results_v4.csv"
    if not file_path.exists():
        print(f"Missing file: {file_path.name}")
        return

    df = pd.read_csv(file_path)

    metrics = ["accuracy", "precision", "recall", "f1"]
    models = list(df["model"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model in models:
            subset = df[df["model"] == model].sort_values("threshold")
            ax.plot(subset["threshold"], subset[metric], marker="o", label=model)
            ax.set_xticks(sorted(df["threshold"].unique()))

        ax.set_title(metric.upper())
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Threshold Summary Across Models", fontsize=16, y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(BASE_DIR / "chart_threshold_summary.png", dpi=300)
    plt.close()
    print("Saved: chart_threshold_summary.png")

def main():
    print(f"Using folder: {BASE_DIR}")

    make_model_comparison_chart()
    make_threshold_chart()
    make_rf_importance_chart()
    make_lr_coeff_chart()
    make_class_distribution_chart()
    make_threshold_summary_chart()

    print("\nDone. All visuals created in the same folder.")


if __name__ == "__main__":
    main()