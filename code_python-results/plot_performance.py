"""Module for plotting functions."""

from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

from numba import njit
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, collections
from matplotlib import pyplot as plt
from statannotations import Annotator
from statannotations.stats import StatTest

from plotting import save_fig


SINGLE_Y_LIMS = tuple[Union[float, None], Union[float, None]]


def violinplot_results(
    data: pd.DataFrame,
    outpath: Union[str, Path],
    x: str,
    y: str,
    hue: Optional[str] = None,
    order: Optional[Union[Sequence, np.ndarray]] = None,
    hue_order: Optional[Union[Sequence, np.ndarray]] = None,
    stat_test: Optional[Union[Callable, str]] = "Permutation",
    alpha: float = 0.05,
    add_lines: Optional[str] = None,
    title: Optional[str] = "Classification Performance",
    figsize: Union[tuple, str] = "auto",
) -> None:
    """Plot performance as violinplot."""
    if order is None:
        order = data[x].unique()

    if hue and hue_order is None:
        hue_order = data[hue].unique()

    if figsize == "auto":
        if not hue:
            hue_factor = 1
        else:
            hue_factor = len(hue_order)  # type: ignore
        figsize = (1.1 * len(order) * hue_factor, 4)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = sns.violinplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        cut=0,
        # palette="Greys",
        inner="box",
        width=0.9,
        alpha=1.0,
        saturation=1,
        ax=ax,
    )

    ax = sns.swarmplot(
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        data=data,
        color="white",
        edgecolor="black",
        linewidth=1,
        alpha=0.6,
        dodge=True,
        # s=6,
        ax=ax,
    )

    if stat_test is not None:
        _add_stats(
            ax=ax,
            data=data,
            x=x,
            y=y,
            order=order,
            hue=hue,
            hue_order=hue_order,
            stat_test=stat_test,
            alpha=alpha,
            location="outside",
        )

    if hue:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label.replace(" ", "\n") for label in labels[: len(labels) // 2]]
        _ = plt.legend(
            handles[: len(handles) // 2],
            new_labels,
            bbox_to_anchor=(1.02, 1),
            loc=2,
            borderaxespad=0.0,
            title=hue,
            labelspacing=0.7,
        )

    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    new_xlabels = [xtick.replace(" ", "\n") for xtick in xlabels]
    ax.set_xticklabels(new_xlabels)
    ax.set_ylim(bottom=None, top=1.0)
    ax.set_title(title, y=1.02)

    if add_lines:
        _add_lines(ax=ax, data=data, x=x, y=y, order=order, add_lines=add_lines)

    fig.tight_layout()
    save_fig(fig, outpath)
    plt.show(block=True)


def _add_lines(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Sequence[str],
    add_lines: str,
) -> None:
    """Add lines connecting single dots"""
    data = data.sort_values(  # type: ignore
        by=x, key=lambda k: k.map({item: i for i, item in enumerate(order)})
    )
    lines = [
        [[i, n] for i, n in enumerate(group)]
        for _, group in data.groupby([add_lines], sort=False)[y]
    ]
    ax.add_collection(collections.LineCollection(lines, colors="grey", linewidths=1))


def _add_stats(
    ax: axes.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Iterable,
    hue: Optional[str],
    hue_order: Optional[Iterable],
    stat_test: Union[str, StatTest.StatTest],
    alpha: float,
    location: str = "inside",
) -> None:
    """Perform statistical test and annotate graph."""
    if not hue:
        pairs = list(combinations(order, 2))
    else:
        pairs = [
            list(combinations(list(product([item], hue_order)), 2)) for item in order
        ]
        pairs = [item for sublist in pairs for item in sublist]

    if stat_test == "Permutation":
        stat_test = StatTest.StatTest(
            func=_permutation_wrapper,
            n_perm=10000,
            alpha=alpha,
            test_long_name="Permutation Test",
            test_short_name="Perm.",
            stat_name="Effect Size",
        )
    annotator = Annotator.Annotator(
        ax=ax,
        pairs=pairs,
        data=data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        order=order,
    )
    annotator.configure(
        alpha=alpha,
        test=stat_test,
        text_format="simple",
        loc=location,
        color="black",
        line_width=1,
        pvalue_format={
            "pvalue_thresholds": [
                [1e-6, "0.000001"],
                [1e-5, "0.00001"],
                [1e-4, "0.0001"],
                [1e-3, "0.001"],
                [1e-2, "0.01"],
                [5e-2, "0.05"],
            ],
            # "fontsize": 13,
            "text_format": "simple",
            "pvalue_format_string": "{:.3f}",
            "show_test_name": False,
        },
    )
    annotator.apply_and_annotate()


@njit
def permutation_twosample(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_perm: int = 10000,
    two_tailed: bool = True,
) -> tuple[float, float]:
    """Perform permutation test.

    Parameters
    ----------
    x : array_like
        First distribution
    y : array_like
        Second distribution
    n_perm : int
        Number of permutations
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-sampled permutation
        test, else True
    two_tailed : bool, default: True
        Set to False if you would like to perform a one-tailed permutation
        test, else True

    Returns
    -------
    float
        Estimated difference of distribution means
    float
        P-value of permutation test
    """
    if two_tailed:
        zeroed = np.abs(np.mean(data_a) - np.mean(data_b))
        data = np.concatenate((data_a, data_b), axis=0)
        half = int(len(data) / 2)
        p = np.empty(n_perm)
        for i in np.arange(0, n_perm):
            np.random.shuffle(data)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.abs(np.mean(data[:half]) - np.mean(data[half:]))
    else:
        zeroed = np.mean(data_a) - np.mean(data_b)
        data = np.concatenate((data_a, data_b), axis=0)
        half = int(len(data) / 2)
        p = np.empty(n_perm)
        for i in np.arange(0, n_perm):
            np.random.shuffle(data)
            # Compute permuted absolute difference of the two sampled
            # distributions
            p[i] = np.mean(data[:half]) - np.mean(data[half:])

    # Compute effect size (Cohen's d)
    n_a = data_a.size
    n_b = data_b.size
    pooled_std = np.sqrt(
        (
            ((n_a - 1) * np.square(np.std(data_a)))
            + ((n_b - 1) * np.square(np.std(data_b)))
        )
        / (n_a + n_b - 2)
    )
    effect_size = np.abs(np.mean(data_a) - np.mean(data_b)) / pooled_std
    effect_size = np.round(effect_size, 3)
    return effect_size, (np.sum(p >= zeroed) + 1) / (n_perm + 1)


def _permutation_wrapper(x, y, n_perm) -> tuple:
    """Wrapper for statannotations to convert pandas series to numpy array."""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    return permutation_twosample(data_a=x, data_b=y, n_perm=n_perm, two_tailed=True)
