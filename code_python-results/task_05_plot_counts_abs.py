"""Plot classification results."""

from pathlib import Path

import pandas as pd

import cell_counts
import plotting_settings
import plot_quantification


def main() -> None:
    """Main function of this script."""
    data_qupath = Path(__file__).parents[1] / "data" / "nuclei_stardist_2022-01-23"
    plot_dir = Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True)

    plotting_settings.activate()
    for compound in ["cisplatin", "efudix", "veregen"]:
        if compound == "cisplatin":
            unit = "µM"
            legend = True
        else:
            unit = "mg"
            legend = False
        x_label = f"Dose ({unit})"
        counts: pd.DataFrame = pd.read_csv(
            data_qupath / f"Results_{compound}_EP" / "counts.csv",
            index_col=0,
            dtype={"Dose": str},
        )  # type: ignore
        counts = cell_counts.average(counts=counts)
        counts = counts.replace(
            {val: val.removesuffix(unit) for val in counts["Dose"].unique()}
            | {"2 weeks + 2 weeks Regen.": "2 weeks + 2 weeks rec."}
        ).rename(columns={"Dose": x_label})

        for cell_type in ["SiHa / Keratinocytes"]:
            data = counts[counts["Classification Type"] == cell_type]
            if cell_type == "Apoptosis/Necrosis":
                cell_type = "Degraded cells"
            if cell_type == "SiHa / Keratinocytes":
                y_label = "Ratio"
            else:
                y_label = "Counts"
            data = data.rename(columns={"Counts / mm²": y_label})
            cell_type_str = cell_type.replace(" / ", "vs").replace(" ", "_").lower()
            for y_logscale in (True,):  #  False):
                log_str = "_log" if y_logscale else ""
                outpath = plot_dir / (
                    f"ratio_{compound}_{cell_type_str}_abs{log_str}.svg"
                )
                y_lims = (0.01, 100) if y_logscale else (-5, 60)
                plot_quantification.plot_quantification(
                    counts=data,
                    x=x_label,
                    y=y_label,
                    hue="Treatment Duration",
                    figsize=(len(data[x_label].unique()) * 1.75 - 0.2, 2),
                    y_lims=y_lims,
                    y_logscale=y_logscale,
                    legend=legend,
                    outpath=outpath,
                    show=False,
                )


if __name__ == "__main__":
    main()
