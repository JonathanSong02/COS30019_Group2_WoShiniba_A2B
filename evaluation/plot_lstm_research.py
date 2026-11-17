"""Generate aggregate graphs from the LSTM research experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SITES = {
    2825: "Low Traffic",
    3812: "Medium Traffic",
    4043: "High Traffic",
}
SEQUENCE_LENGTHS = [4, 8]
HORIZONS = [15, 30, 60]


def load_results(csv_path: str | Path) -> list[dict]:
    """Load and convert the LSTM experiment results."""

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    converted = []

    for row in rows:
        converted.append(
            {
                "scats_site": int(row["scats_site"]),
                "sequence_length": int(row["sequence_length"]),
                "horizon_minutes": int(row["horizon_minutes"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "nrmse": float(row["nrmse"]),
                "training_time_seconds": float(
                    row["training_time_seconds"]
                ),
            }
        )

    expected_count = (
        len(SITES) * len(SEQUENCE_LENGTHS) * len(HORIZONS)
    )

    if len(converted) != expected_count:
        raise ValueError(
            f"Expected {expected_count} experiments, "
            f"but found {len(converted)}"
        )

    return converted


def select_result(
    rows: list[dict],
    site: int,
    sequence_length: int,
    horizon: int,
) -> dict:
    """Return one unique experiment result."""

    matches = [
        row
        for row in rows
        if row["scats_site"] == site
        and row["sequence_length"] == sequence_length
        and row["horizon_minutes"] == horizon
    ]

    if len(matches) != 1:
        raise ValueError(
            "Each site, sequence and horizon combination "
            "must appear exactly once"
        )

    return matches[0]


def generate_research_graphs(
    summary_csv: str | Path = (
        "results/lstm/lstm_experiment_summary.csv"
    ),
    output_dir: str | Path = "results/lstm/research_graphs",
) -> dict:
    """Generate aggregate LSTM research graphs and statistics."""

    rows = load_results(summary_csv)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    site_graph_path = destination / "rmse_by_site.png"
    horizon_graph_path = (
        destination / "average_rmse_by_horizon.png"
    )
    nrmse_graph_path = (
        destination / "average_nrmse_by_site.png"
    )
    summary_path = destination / "research_summary.json"

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        sharey=True,
    )

    for axis, (site, traffic_label) in zip(axes, SITES.items()):
        for sequence_length in SEQUENCE_LENGTHS:
            rmse_values = [
                select_result(
                    rows,
                    site,
                    sequence_length,
                    horizon,
                )["rmse"]
                for horizon in HORIZONS
            ]

            axis.plot(
                HORIZONS,
                rmse_values,
                marker="o",
                label=f"Sequence {sequence_length}",
            )

        axis.set_title(f"Site {site} — {traffic_label}")
        axis.set_xlabel("Prediction Horizon (minutes)")
        axis.grid(alpha=0.3)
        axis.legend()

    axes[0].set_ylabel("RMSE (vehicles per 15 minutes)")
    figure.suptitle(
        "LSTM Prediction Error by Site, Horizon and Sequence Length"
    )
    figure.tight_layout()
    figure.savefig(site_graph_path, dpi=200)
    plt.close(figure)

    average_rmse_by_sequence_and_horizon = {}

    for sequence_length in SEQUENCE_LENGTHS:
        average_rmse_by_sequence_and_horizon[sequence_length] = [
            mean(
                row["rmse"]
                for row in rows
                if row["sequence_length"] == sequence_length
                and row["horizon_minutes"] == horizon
            )
            for horizon in HORIZONS
        ]

    figure, axis = plt.subplots(figsize=(8, 5))

    for sequence_length in SEQUENCE_LENGTHS:
        axis.plot(
            HORIZONS,
            average_rmse_by_sequence_and_horizon[
                sequence_length
            ],
            marker="o",
            label=f"Sequence {sequence_length}",
        )

    axis.set_title("Average LSTM RMSE Across Three SCATS Sites")
    axis.set_xlabel("Prediction Horizon (minutes)")
    axis.set_ylabel("Average RMSE")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(horizon_graph_path, dpi=200)
    plt.close(figure)

    site_names = [
        f"{site}\n{label}"
        for site, label in SITES.items()
    ]

    average_nrmse_by_site = [
        mean(
            row["nrmse"]
            for row in rows
            if row["scats_site"] == site
        )
        * 100.0
        for site in SITES
    ]

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(site_names, average_nrmse_by_site)

    axis.set_title("Average LSTM NRMSE by Traffic Site")
    axis.set_xlabel("SCATS Site")
    axis.set_ylabel("Average NRMSE (%)")
    axis.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, average_nrmse_by_site):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(nrmse_graph_path, dpi=200)
    plt.close(figure)

    statistics = {
        "experiment_count": len(rows),
        "average_rmse_by_sequence_length": {
            str(sequence_length): mean(
                row["rmse"]
                for row in rows
                if row["sequence_length"] == sequence_length
            )
            for sequence_length in SEQUENCE_LENGTHS
        },
        "average_rmse_by_horizon": {
            str(horizon): mean(
                row["rmse"]
                for row in rows
                if row["horizon_minutes"] == horizon
            )
            for horizon in HORIZONS
        },
        "average_rmse_by_site": {
            str(site): mean(
                row["rmse"]
                for row in rows
                if row["scats_site"] == site
            )
            for site in SITES
        },
        "best_experiment": min(
            rows,
            key=lambda row: row["rmse"],
        ),
    }

    summary_path.write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8",
    )

    return {
        "site_graph": str(site_graph_path),
        "horizon_graph": str(horizon_graph_path),
        "nrmse_graph": str(nrmse_graph_path),
        "research_summary": str(summary_path),
    }


if __name__ == "__main__":
    generated = generate_research_graphs()
    print(json.dumps(generated, indent=2))