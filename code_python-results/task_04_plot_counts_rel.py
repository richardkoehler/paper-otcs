"""Plot classification results."""

from pathlib import Path

import pandas as pd

import cell_counts
import plotting_settings
import plot_quantification


def plot_cell_counts_rel() -> None:
    """Main function of this script."""
    data_qupath = Path(__file__).parents[1] / "data" / "nuclei_stardist_2022-01-23"
    plot_dir = Path(__file__).parents[1] / "plots"
    plot_dir.mkdir(exist_ok=True)

    plotting_settings.activate()
    cell_types = ["SiHa", "Keratinocytes"]
    y_lims_all = ((-0.06, 1.06), (-0.2, 3.8))
    y_label_old = "Counts: Treatment / Control (AU)"
    y_label = "Cell counts\nTreatment / Control"
    for compound in ["cisplatin", "efudix", "veregen"]:
        if compound == "cisplatin":
            unit = "µM"
        else:
            unit = "mg"
        x_label = f"Dose ({unit})"
        counts: pd.DataFrame = pd.read_csv(
            data_qupath / f"Results_{compound}_EP" / "counts.csv",
            index_col=0,
            dtype={"Dose": str},
        )  # type: ignore
        counts = cell_counts.average(counts=counts)
        counts = cell_counts.baseline_correct(counts=counts)
        counts = counts.replace(
            {val: val.removesuffix(unit) for val in counts["Dose"].unique()}
            | {"2 weeks + 2 weeks Regen.": "2 weeks + 2 weeks rec."}
        ).rename(columns={"Dose": x_label})
        counts = counts[counts[x_label] != "Control"]

        # Plot all cell types
        data = counts.query(f"`Classification Type` in {cell_types}").rename(
            columns={y_label_old: y_label}
        )
        outpath = plot_dir / (f"ratio_{compound}_all_cell_types.svg")
        plot_quantification.plot_quantification_all(
            counts=data,
            cell_types=cell_types,
            x=x_label,
            y=y_label,
            hue="Treatment Duration",
            y_lims=y_lims_all,
            y_logscale=False,
            outpath=outpath,
            show=True,
        )


if __name__ == "__main__":
    plot_cell_counts_rel()
