"""Initialize settings for plotting."""

from cycler import cycler
import matplotlib as mpl
from matplotlib import pyplot as plt


def activate() -> None:
    palette_wjn_2022 = (
        (234 / 255, 234 / 255, 234 / 255),
        (100 / 255, 100 / 255, 100 / 255),
        (64 / 255, 64 / 255, 64 / 255),
        (55 / 255, 110 / 255, 180 / 255),
        (113 / 255, 188 / 255, 173 / 255),
        (223 / 255, 74 / 255, 74 / 255),
        (215 / 255, 178 / 255, 66 / 255),
        (215 / 255, 213 / 255, 203 / 255),
    )
    # mpl.rcParams["figure.figsize"] = [6.4, 4.8]
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 10

    mpl.rcParams["pdf.fonttype"] = "TrueType"
    mpl.rcParams["svg.fonttype"] = "none"

    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"

    mpl.rcParams["legend.title_fontsize"] = mpl.rcParams["legend.fontsize"]
    mpl.rcParams["legend.edgecolor"] = "black"

    mpl.rcParams["scatter.edgecolors"] = "black"

    mpl.rcParams["axes.prop_cycle"] = cycler(color=palette_wjn_2022)


if __name__ == "__main__":
    import seaborn as sns

    activate()
    sns.palplot(mpl.rcParams["axes.prop_cycle"].by_key()["color"])
    plt.show(block=True)
