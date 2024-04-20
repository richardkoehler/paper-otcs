"""Module for helper functions."""

import os
from typing import Optional, Sequence, Union


def find_files(
    path: Union[str, bytes, os.PathLike],
    suffix: Union[Sequence, str],
    prefix: Optional[Union[Sequence, str]] = None,
    verbose: bool = False,
) -> list:
    """Return files in path with given suffixes and prefixes.

    Parameters
    ----------
        path : string
            Path to be searched for files
        suffix : sequence | string
            File suffix to filter by e.g. ["vhdr", "edf"] or ".json"
        prefix : sequence | string
            keyword to filter by e.g. ["SelfpacedRota", "ButtonPress]
        verbose : boolean
            Verbosity level (default=False)

    Returns
    -------
        filepaths : list of strings
    """
    path = str(path)

    if isinstance(suffix, str):
        suffix = [suffix]
    if isinstance(prefix, str):
        prefix = [prefix]

    filepaths = []
    for root, _, files in os.walk(path):
        for file in files:
            for suff in suffix:
                if file.endswith(suff.lower()):
                    if not prefix:
                        filepaths.append(os.path.join(root, file))
                    else:
                        for pref in prefix:
                            if pref.lower() in file.lower():
                                filepaths.append(os.path.join(root, file))
    if verbose and not filepaths:
        print("No corresponding files found.")
    if verbose and filepaths:
        print("Corresponding files found:")
        for idx, file in enumerate(filepaths):
            print(idx, ":", os.path.basename(file))
    return filepaths
