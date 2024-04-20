"""Module for plotting functions."""

import os
import re
from typing import Union

from matplotlib import figure

SINGLE_Y_LIMS = tuple[Union[float, None], Union[float, None]]


def save_fig(fig: figure.Figure, outpath: Union[str, os.PathLike]) -> None:
    outpath = str(outpath)
    fig.savefig(outpath, bbox_inches="tight")
    if outpath.endswith(".svg"):
        with open(outpath, "r", encoding="utf-8") as f:
            svg_text = f.read()
        patched_svg = patch_affinity_svg(svg_text)
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(patched_svg)


def patch_affinity_svg(svg_text: str) -> str:
    """Patch Matplotlib SVG so that it can be read by Affinity Designer."""
    matches = [
        x
        for x in re.finditer(
            'font:( [0-9][0-9]?[0-9]?[0-9]?)? ([0-9.]+)px ([^;"]+)[";]',
            svg_text,
        )
    ]
    if not matches:
        return svg_text
    svg_pieces = [svg_text[: matches[0].start()]]
    for i, match in enumerate(matches):
        # Change "font" style property to separate "font-size" and
        # "font-family" properties because Affinity ignores "font".
        group = match.groups()
        if len(group) == 2:
            font_weight, font_size_px, font_family = match.groups()
            new_font_style = (
                f"font-size: {float(font_size_px):.1f}px; "
                f"font-family: {font_family}"
            )
        else:
            font_weight, font_size_px, font_family = match.groups()
            if font_weight is not None:
                new_font_style = (
                    f"font-weight: {font_weight}; "
                    f"font-size: {float(font_size_px):.1f}px; "
                    f"font-family: {font_family}"
                )
            else:
                new_font_style = (
                    f"font-size: {float(font_size_px):.1f}px; "
                    f"font-family: {font_family}"
                )
        svg_pieces.append(new_font_style)
        if i < len(matches) - 1:
            svg_pieces.append(svg_text[match.end() - 1 : matches[i + 1].start()])
        else:
            svg_pieces.append(svg_text[match.end() - 1 :])
    return "".join(svg_pieces)
