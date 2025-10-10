from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_COLUMNS = [
    "Test_ACC",
    "Test_Precision",
    "Test_F1",
    "Test_AUC",
    "Test_Recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create bar charts for evaluation metrics grouped by model and feature selection."
        )
    )
    parser.add_argument(
        "results",
        type=Path,
        help="Path to the Excel file containing the results_df table.",
    )
    parser.add_argument(
        "--sheet",
        default="Sheet2",
        help="Sheet name that holds the results table (default: Sheet2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("img/results"),
        help="Directory where the charts will be stored (default: img/results).",
    )
    parser.add_argument(
        "--style",
        default="whitegrid",
        help="Seaborn style to apply before plotting (default: whitegrid).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for the saved figures (default: 300).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.5,
        help="Figure height in inches (default: 4.5).",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=7.5,
        help="Figure width in inches (default: 7.5).",
    )
    return parser.parse_args()


def load_results(results_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(results_path, sheet_name=sheet_name)
    if "Feature Selection" in df.columns:
        df["Feature Selection"] = df["Feature Selection"].fillna("None")
    return df


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, *, width: float, height: float) -> None:
    plt.figure(figsize=(width, height))
    sns.barplot(
        data=df,
        x="Model",
        y=metric,
        hue="Feature Selection",
        errorbar=None,
    )
    plt.title(metric.replace("_", " "))
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.legend(title="Feature Selection", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=output_path.suffix == ".svg" and None or None)
    plt.close()


def main() -> None:
    args = parse_args()

    sns.set_style(args.style)

    df = load_results(args.results, args.sheet)

    missing_columns = [col for col in METRIC_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required metric columns in results file: "
            + ", ".join(missing_columns)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRIC_COLUMNS:
        output_path = args.output_dir / f"{metric.lower()}.png"
        plot_metric(df, metric, output_path, width=args.width, height=args.height)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
