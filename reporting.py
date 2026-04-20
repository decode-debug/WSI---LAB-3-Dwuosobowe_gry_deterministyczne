from __future__ import annotations

import csv
from pathlib import Path

from game_logic import ExperimentResult


def format_results_table(results: list[ExperimentResult]) -> str:
    """Builds a markdown table with experiment metrics."""
    header = (
        "| Wariant    | Glebokosc d | Partie | Wygrane [%] "
        "| Sredni czas ruchu [ms] | Srednia liczba wezlow |"
    )
    separator = (
        "|:-----------|------------:|-------:|------------:"
        "|-----------------------:|----------------------:|"
    )
    rows = [header, separator]
    for row in results:
        rows.append(
            f"| {row.variant:<10} | "
            f"{row.depth:>11} | "
            f"{row.games:>6} | "
            f"{row.win_rate_pct:>11.1f} | "
            f"{row.avg_time_ms:>22.3f} | "
            f"{row.avg_nodes:>21.2f} |"
        )
    return "\n".join(rows)


def save_results(results: list[ExperimentResult]) -> None:
    csv_path = Path("results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "variant",
                "depth",
                "games",
                "win_rate_pct",
                "avg_time_ms",
                "avg_nodes",
            ],
        )
        writer.writeheader()
        writer.writerows([row.__dict__ for row in results])

    report_path = Path("results.md")
    report_path.write_text(
        format_results_table(results) + "\n",
        encoding="utf-8",
    )
