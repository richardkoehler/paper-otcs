"""Load and plot cross-validation performance."""

from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from statannotations.stats.StatTest import StatTest

import plotting
import plotting_settings


def statistic(x, y, axis):
    return np.mean(a=x, axis=axis) - np.mean(a=y, axis=axis)


def _stat_test(
    x: pd.Series | np.ndarray, y: pd.Series | np.ndarray
) -> tuple[float, float]:
    res = scipy.stats.permutation_test(
        (x, y),
        statistic,
        vectorized=True,
        n_resamples=int(1e6),
        permutation_type="independent",
    )
    return res.statistic, res.pvalue


def main() -> None:
    """Main function of this script."""
    plotting_settings.activate()
    data_qupath = (
        Path(__file__).resolve().parents[1] / "data" / "nuclei_stardist_2022-01-23"
    )
    plot_root = data_qupath / "plots"
    plot_root.mkdir(exist_ok=True)
    researcher_1 = r"xgb_EP"
    file_1 = data_qupath / researcher_1 / "scores.csv"
    researcher_2 = r"xgb_RK"
    file_2 = data_qupath / researcher_2 / "scores.csv"
    for file in (file_1, file_2):
        if not file.exists():
            raise ValueError(f"File not found:\n{file}.")

    scores_1: pd.DataFrame = pd.read_csv(file_1)  # type: ignore
    scores_1["Annotation by"] = "A"
    scores_2: pd.DataFrame = pd.read_csv(file_2)  # type: ignore
    scores_2["Annotation by"] = "B"

    x = "Annotation by\nresearcher"
    y = "Balanced Accuracy"

    data = pd.concat([scores_1, scores_2]).rename(
        columns={
            "Annotation by": x,
            "Actual Balanced Accuracy": y,
        }
    )

    plotting.violinplot_results(
        data=data,
        outpath=plot_root / "performance_comparison_researchers.svg",
        x=x,
        y=y,
        hue=None,
        stat_test=StatTest(
            func=_stat_test,
            alpha=0.05,
            test_long_name="Permutation Test",
            test_short_name="Perm. Test",
        ),
        alpha=0.05,
        title=None,
        figsize=(1.8, 3),
    )

    fname_stats = plot_root / ("performance_comparison_researchers_stats.csv")
    fname_stats.unlink(missing_ok=True)

    with open(fname_stats, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["description", "mean", "std", "statistic", "P"])

        def statistic(x: np.ndarray, y: np.ndarray, axis=0):
            return np.mean(a=x, axis=axis) - np.mean(a=y, axis=axis)

        res_a = "A"
        res_b = "B"
        description = f"{res_a} vs {res_b}"
        print(f"{description = }")
        data_a = data.loc[data[x] == res_a, y].to_numpy()
        data_b = data.loc[data[x] == res_b, y].to_numpy()
        test = scipy.stats.permutation_test(
            (data_a, data_b),
            statistic,
            vectorized=True,
            n_resamples=int(1e6),
            permutation_type="independent",
        )
        print(f"statistic = {test.statistic}, P = {test.pvalue}")
        writer.writerow(
            [
                description,
                statistic(data_a, data_b),
                "n/a",
                test.statistic,
                test.pvalue,
            ]
        )

        statistic = np.mean
        for res_pick in (res_a, res_b):
            description = res_pick
            print(f"{description = }")
            data_cond = data.loc[data[x] == res_pick, y].to_numpy()
            test = scipy.stats.permutation_test(
                (data_cond - (1 / 3),),
                statistic,
                vectorized=True,
                n_resamples=int(1e6),
                permutation_type="samples",
            )
            print(f"statistic = {test.statistic}, P = {test.pvalue}")
            writer.writerow(
                [
                    description,
                    np.mean(data_cond),
                    np.std(data_cond),
                    test.statistic,
                    test.pvalue,
                ]
            )


if __name__ == "__main__":
    main()
