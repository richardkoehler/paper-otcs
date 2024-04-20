"""Module for plotting functions."""

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import matplotlib.lines
import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from plotting import save_fig

SINGLE_Y_LIMS = tuple[Union[float, None], Union[float, None]]


def plot_quantification_all(
    counts: pd.DataFrame,
    cell_types: Sequence[str],
    x: str,
    y: str = "Counts / mm²",
    hue: str = "Treatment Duration",
    figsize: Union[Literal["auto"], tuple[float, float]] = "auto",
    y_lims: Optional[Sequence[SINGLE_Y_LIMS]] = None,
    y_logscale: bool = False,
    plot_legend: bool = True,
    outpath: Optional[Union[Path, str]] = None,
    show: bool = True,
) -> None:
    """Plot quantification results"""
    if figsize == "auto":
        figsize = (len(counts[x].unique()) * 1.4 + 0.3, 2 * len(cell_types))
    fig, axs = plt.subplots(len(cell_types), 1, figsize=figsize)
    for i, (ax, cell_type) in enumerate(zip(axs, cell_types)):
        data = counts[counts["Classification Type"] == cell_type]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(y=1.0, color="black", linestyle="--")
        sns.boxplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            dodge=True,
            gap=0.2,
            width=0.8,
            showmeans=False,
            meanline=False,
            showcaps=False,
            showbox=False,
            showfliers=False,
            notch=False,
            whis=0.1,
            medianprops={
                "linestyle": "-",
                "linewidth": 3,
            },
            ax=ax,
        )
        sns.swarmplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            dodge=True,
            s=8,
            linewidth=1,
            alpha=0.9,
            ax=ax,
        )
        if plot_legend:
            handles_orig, labels = ax.get_legend_handles_labels()
            handles = []
            for handle in handles_orig:
                if not isinstance(handle, matplotlib.lines.Line2D):
                    continue
                handle.set_markeredgecolor("black")
                handles.append(handle)
            ax.legend(
                handles,
                labels[len(labels) // 2 :],
                loc="upper right",
                title="Treatment Duration",
            )
            plot_legend = False
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        if y_logscale:
            ax.set_yscale("log")
        if y_lims is not None:
            y_lim = y_lims[i]
            if y_lim:
                ax.set_ylim(*y_lim)
    fig.align_ylabels()
    fig.tight_layout()
    if outpath is not None:
        save_fig(fig, outpath)
    if show:
        plt.show(block=True)
    else:
        plt.close()


def plot_quantification(
    counts: pd.DataFrame,
    x: str,
    y: str = "Counts / mm²",
    hue: str = "Treatment Duration",
    figsize: Union[Literal["auto"], tuple[float, float]] = "auto",
    y_lims: Optional[tuple[Union[float, None], Union[float, None]]] = None,
    y_logscale: bool = False,
    legend: bool = True,
    outpath: Optional[Union[Path, str]] = None,
    show: bool = True,
) -> None:
    """Plot quantification results"""
    if figsize == "auto":
        figsize = (len(counts[x].unique()) * 1.5, 2)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=1.0, color="black", linestyle="--")

    sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        data=counts,
        # palette="Greys",
        dodge=True,
        gap=0.1,
        showmeans=False,
        meanline=False,
        showcaps=False,
        showbox=False,
        showfliers=False,
        notch=False,
        whis=0.1,
        medianprops={
            "linestyle": "-",
            "linewidth": 3,
        },
        ax=ax,
    )
    sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        data=counts,
        # palette="Greys",
        dodge=True,
        s=8,
        edgecolor="black",
        linewidth=1,
        alpha=0.9,
        ax=ax,
    )
    if legend:
        handles_orig, labels = ax.get_legend_handles_labels()
        handles = []
        for handle in handles_orig:
            if not isinstance(handle, matplotlib.lines.Line2D):
                continue
            handle.set_markeredgecolor("black")
            handles.append(handle)
        ax.legend(
            handles,
            labels[len(labels) // 2 :],
            loc="upper right",
            title="Treatment Duration",
        )
    else:
        legend_ = ax.get_legend()
        if legend_ is not None:
            legend_.remove()
    if y_lims:
        ax.set_ylim(*y_lims)
    if y_logscale:
        ax.set_yscale("log")
    fig.tight_layout()

    if outpath is not None:
        save_fig(fig, outpath)
    if show:
        plt.show(block=True)
    else:
        plt.close()
