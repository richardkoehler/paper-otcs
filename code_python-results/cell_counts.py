"""Module for loading results."""

import pandas as pd


def average(counts: pd.DataFrame) -> pd.DataFrame:
    """Average counts."""
    id_vars = ["Dose", "Specimen", "Treatment Duration"]
    orig_vars = [
        "Apoptosis/Necrosis",
        "SiHa",
        "Keratinocytes",
    ]
    new_var = "SiHa / Keratinocytes"
    final_vars = orig_vars + [new_var]

    results = []
    for specimen in counts["Specimen"].unique():
        counts_specimen = counts[counts.Specimen == specimen]
        result = counts_specimen[orig_vars].mean()
        for item in id_vars:
            result[item] = counts_specimen.iloc[0].at[item]  # type: ignore
        results.append(result)

    results_df = pd.concat(results, axis=1, ignore_index=True).transpose()
    results_df[new_var] = results_df["SiHa"] / results_df["Keratinocytes"]
    counts_average = pd.melt(
        frame=results_df,
        id_vars=id_vars,
        value_vars=final_vars,
        var_name="Classification Type",
        value_name="Counts / mm²",
        col_level=None,
    )
    return counts_average


def baseline_correct(counts: pd.DataFrame) -> pd.DataFrame:
    """Baseline correct counts with respect to control group."""
    data_corrected = []

    for class_type in counts["Classification Type"].unique():
        counts_type = counts[counts["Classification Type"] == class_type]
        for duration in counts_type["Treatment Duration"].unique():  # type:ignore
            counts_duration = counts_type[
                counts_type["Treatment Duration"] == duration
            ].copy()
            df_base = counts_duration[counts_duration["Dose"] == "Control"]
            mean = df_base["Counts / mm²"].mean()
            values = counts_duration["Counts / mm²"].divide(mean).to_numpy()
            counts_duration.loc[:, "Counts / mm²"] = values  # type: ignore
            counts_duration.rename(
                columns={"Counts / mm²": "Counts: Treatment / Control (AU)"},
                inplace=True,
            )  # type: ignore
            data_corrected.append(counts_duration)
    return pd.concat(data_corrected)  # type: ignore
