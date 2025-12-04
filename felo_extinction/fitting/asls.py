"""
Asymmetric least squares (AsLS) helpers for extinction-curve fitting.

The helpers here provide:
- a single AsLS solver tailored to extinction curves (non-negative, optional error weights)
- a small grid-search convenience wrapper to scan smoothing strengths
- an interpolator builder to evaluate the fitted extinction curve on arbitrary x grids
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve


@dataclass
class AslsFitResult:
    """Container for an AsLS fit on an extinction curve."""

    x: np.ndarray
    y_original: np.ndarray
    y_smooth: np.ndarray
    y_err: np.ndarray | None
    residuals: np.ndarray
    weights: np.ndarray
    smoothness: float
    asymmetry: float
    rms_residual: float
    mad_residual: float
    chi2_reduced: float


def _prepare_data(
    x: np.ndarray, y: np.ndarray, y_err: np.ndarray | None, x_range: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Mask NaNs/Infs and restrict to a fitting range."""
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= x_range[0])
        & (x <= x_range[1])
    )
    if y_err is not None:
        mask &= np.isfinite(y_err) & (y_err > 0)
    x_fit = x[mask]
    y_fit = y[mask]
    y_err_fit = y_err[mask] if y_err is not None else None
    return x_fit, y_fit, y_err_fit


def asymmetric_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None = None,
    *,
    lam: float = 1e4,
    p: float = 0.01,
    n_iter: int = 15,
    weight_small_values: bool = True,
    bias_factor: float = 2.0,
    x_range: Sequence[float] = (1.0, 6.0),
) -> AslsFitResult:
    """
    Fit an extinction curve using asymmetric least squares smoothing.

    Parameters
    ----------
    x : array-like
        Inverse wavelength (micron^-1).
    y : array-like
        Extinction A(x) in magnitudes.
    y_err : array-like, optional
        One-sigma errors on A(x); used for weighting.
    lam : float
        Smoothness parameter; higher is smoother.
    p : float
        Asymmetry parameter (0 < p < 1). Smaller values bias toward lower envelopes.
    n_iter : int
        Number of IRLS iterations.
    weight_small_values : bool
        If True, bias weights toward smaller A(x) values.
    bias_factor : float
        Strength of the bias toward smaller values when ``weight_small_values`` is True.
    x_range : tuple
        Limits in x to keep for the fit.

    Returns
    -------
    AslsFitResult
        Fit result with smoothed curve, weights, and quality metrics.
    """

    x_fit, y_fit, y_err_fit = _prepare_data(
        np.asarray(x), np.asarray(y), None if y_err is None else np.asarray(y_err), x_range
    )
    if x_fit.size == 0:
        raise ValueError("No valid points after masking; check x_range and inputs.")

    order = np.argsort(x_fit)
    x_sorted = x_fit[order]
    y_sorted = y_fit[order]
    y_err_sorted = y_err_fit[order] if y_err_fit is not None else None

    n_points = len(y_sorted)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n_points - 2, n_points))

    base_weights = np.ones(n_points)
    if y_err_sorted is not None:
        base_weights *= 1.0 / (y_err_sorted**2 + 1e-12)
    if weight_small_values:
        scale = np.std(y_sorted) if np.std(y_sorted) > 0 else 1.0
        value_weights = np.exp(-y_sorted / scale * bias_factor)
        base_weights *= value_weights
    base_weights /= np.mean(base_weights)

    w = np.ones(n_points)
    for _ in range(n_iter):
        w_total = base_weights * w
        W = sparse.diags(w_total, 0, shape=(n_points, n_points))
        A = W + lam * D.T.dot(D)
        z = spsolve(A, w_total * y_sorted)
        z = np.maximum(z, 0.0)
        w = p * (y_sorted > z) + (1 - p) * (y_sorted <= z)

    y_smooth_sorted = np.maximum(z, 0.0)

    y_smooth = np.zeros_like(y_fit)
    weights = np.zeros_like(y_fit)
    y_smooth[order] = y_smooth_sorted
    weights[order] = w_total

    residuals = y_fit - y_smooth
    chi2_reduced = np.nan
    if y_err_fit is not None:
        chi2 = np.sum((residuals / y_err_fit) ** 2)
        dof = max(len(y_fit) - 2, 1)
        chi2_reduced = chi2 / dof

    rms_residual = float(np.sqrt(np.mean(residuals**2)))
    mad_residual = float(np.median(np.abs(residuals)))

    return AslsFitResult(
        x=x_fit,
        y_original=y_fit,
        y_smooth=y_smooth,
        y_err=y_err_fit,
        residuals=residuals,
        weights=weights,
        smoothness=lam,
        asymmetry=p,
        rms_residual=rms_residual,
        mad_residual=mad_residual,
        chi2_reduced=chi2_reduced,
    )


def fit_extinction_asls_grid(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None = None,
    *,
    smoothness_grid: Iterable[float] = (1e2, 1e3, 1e4, 1e5),
    asymmetry: float = 0.01,
    x_range: Sequence[float] = (1.0, 6.0),
    selection_metric: str = "rms",
) -> tuple[dict[float, AslsFitResult], AslsFitResult]:
    """
    Run AsLS over a grid of smoothing parameters and pick the best fit.

    Parameters
    ----------
    smoothness_grid : iterable of float
        λ values to scan.
    asymmetry : float
        AsLS asymmetry parameter p.
    x_range : tuple
        Limits in x to keep for the fit.
    selection_metric : {'rms', 'mad', 'chi2'}
        Metric used to select the best result.

    Returns
    -------
    all_results : dict
        Map of λ to AslsFitResult.
    best_result : AslsFitResult
        Result chosen by ``selection_metric``.
    """

    all_results: dict[float, AslsFitResult] = {}
    for lam in smoothness_grid:
        all_results[lam] = asymmetric_least_squares(
            x, y, y_err, lam=lam, p=asymmetry, x_range=x_range
        )

    if selection_metric == "chi2":
        valid = [r for r in all_results.values() if not np.isnan(r.chi2_reduced)]
        if len(valid) == 0:
            selection_metric = "rms"
        else:
            best_result = min(valid, key=lambda r: r.chi2_reduced)
            return all_results, best_result

    if selection_metric == "mad":
        best_result = min(all_results.values(), key=lambda r: r.mad_residual)
    else:
        best_result = min(all_results.values(), key=lambda r: r.rms_residual)

    return all_results, best_result


def build_extinction_interpolator(
    result: AslsFitResult, *, extrapolation: str = "linear"
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build an interpolator from an AsLS fit.

    Parameters
    ----------
    result : AslsFitResult
        Output from ``asymmetric_least_squares`` or ``fit_extinction_asls_grid``.
    extrapolation : {'linear', 'edge'}
        Extrapolation style outside the fitted x range.

    Returns
    -------
    callable
        Function mapping x (micron^-1) -> A(x) with non-negative values.
    """

    if extrapolation == "edge":
        interp = interp1d(
            result.x,
            result.y_smooth,
            kind="linear",
            bounds_error=False,
            fill_value=(result.y_smooth[0], result.y_smooth[-1]),
        )
    else:
        interp = interp1d(
            result.x,
            result.y_smooth,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

    def _call(x: np.ndarray) -> np.ndarray:
        values = interp(x)
        return np.maximum(values, 0.0)

    return _call
