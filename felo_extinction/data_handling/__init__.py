from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Continuum windows (Angstrom) recommended for extinction fitting.
DEFAULT_CONTINUUM_WINDOWS_AA = np.array(
    [
        [1150.0, 1170.0],
        [1275.0, 1290.0],
        [1350.0, 1360.0],
        [1445.0, 1465.0],
        [1690.0, 1705.0],
        [1770.0, 1810.0],
        [1970.0, 2400.0],
        [2480.0, 2675.0],
        [2925.0, 3400.0],
        [3500.0, 3600.0],
        [3600.0, 4260.0],
        [4435.0, 4640.0],
        [5100.0, 5535.0],
        [6005.0, 6035.0],
        [6110.0, 6250.0],
        [6800.0, 7000.0],
        [7180.0, 7250.0],
        [7600.0, 7700.0],
        [7950.0, 8050.0],
        [8600.0, 8800.0],
        [9350.0, 9400.0],
        [9650.0, 9800.0],
        [10200.0, 10600.0],
        [11400.0, 12400.0],
    ]
)


def get_template_path(name: str = "Selsing2015_interpolated.dat") -> Path:
    """Return the packaged path to a quasar template."""

    with importlib.resources.as_file(
        importlib.resources.files("felo_extinction.data.templates").joinpath(name)
    ) as path:
        return path


def load_quasar_template(name: str = "Selsing2015_interpolated.dat") -> pd.DataFrame:
    """Load the quasar template shipped with the package."""

    template_path = get_template_path(name)
    return pd.read_csv(
        template_path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        names=["wave", "flux", "err"],
    )


def continuum_mask(
    wavelengths_aa: np.ndarray,
    continuum_windows_aa: Sequence[Sequence[float]] = DEFAULT_CONTINUUM_WINDOWS_AA,
    user_masks_aa: Iterable[Sequence[float]] | None = None,
) -> np.ndarray:
    """
    Build a boolean mask selecting continuum windows and removing user-specified regions.

    Parameters
    ----------
    wavelengths_aa : array
        Wavelength grid (Angstrom).
    continuum_windows_aa : array-like
        Inclusive wavelength windows (Angstrom) to keep.
    user_masks_aa : iterable of (start, stop)
        Additional regions to exclude (Angstrom).
    """

    wave = np.asarray(wavelengths_aa)
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in continuum_windows_aa:
        mask |= (wave >= lo) & (wave <= hi)

    if user_masks_aa:
        for lo, hi in user_masks_aa:
            mask &= ~((wave >= lo) & (wave <= hi))

    return mask


__all__ = [
    "DEFAULT_CONTINUUM_WINDOWS_AA",
    "continuum_mask",
    "get_template_path",
    "load_quasar_template",
]
