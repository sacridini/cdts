"""
QC/QA-band decoders that turn a sensor's raw quality-assurance layer into
per-observation reliability weights in [0, 1], suitable for the `weights`
argument of `DataArray.cdts.run_phenology` (which threads them through the
Whittaker/HANTS smoothers and the iterative curve-fit reweighting).

This mirrors phenofit's `qcFUN.R` (qc_summary, qc_StateQA, qc_sentinel2),
ported to numpy so the same "good/marginal/snow-or-cloud" weighting scheme
used there is available in cdts instead of requiring every caller to hand-roll
their own NaN mask before smoothing.
"""

import numpy as np

__all__ = ["qc_modis_summary", "qc_modis_state", "qc_sentinel2_scl"]


def _get_bits(x: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract bits [start, end] (inclusive, 0-indexed from the LSB)."""
    n_bits = end - start + 1
    mask = (1 << n_bits) - 1
    return (np.asarray(x).astype(np.int64) >> start) & mask


def qc_modis_summary(qa: np.ndarray, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> np.ndarray:
    """
    Weights from the MOD13A1/A2/Q1 "SummaryQA" (pixel reliability) band.
    Port of phenofit's `qc_summary()` (R/qcFUN.R).

        0 good      -> wmax
        1 marginal  -> wmid
        2 snow/ice  -> wmin
        3 cloudy    -> wmin
        other/fill  -> 0.0 (excluded)
    """
    qa = np.asarray(qa)
    w = np.zeros(qa.shape, dtype=np.float64)
    w[qa == 0] = wmax
    w[qa == 1] = wmid
    w[(qa >= 2) & (qa <= 3)] = wmin
    return w


def qc_modis_state(qa: np.ndarray, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> np.ndarray:
    """
    Weights from the MOD09A1/MYD09A1 500m "State QA" 16-bit flag.
    Port of phenofit's `qc_StateQA()` (R/qcFUN.R): decodes cloud state
    (bits 0-1), cloud shadow (bit 2), aerosol quantity (bits 6-7) and snow/ice
    (bit 12).

        clear/climatology-aerosol, no shadow, no snow -> wmax (good)
        snow, or cloudy/mixed cloud, or high aerosol   -> wmin (bad)
        everything else                                -> wmid (marginal)
    """
    qa = np.asarray(qa)
    qc_cloud = _get_bits(qa, 0, 1)
    qc_aerosol = _get_bits(qa, 6, 7)
    qc_snow = _get_bits(qa, 12, 12).astype(bool)

    w = np.full(qa.shape, wmid, dtype=np.float64)

    is_good = np.isin(qc_cloud, [0, 3]) & np.isin(qc_aerosol, [0, 1, 2]) & ~qc_snow
    is_bad = qc_snow | np.isin(qc_cloud, [1, 2]) | (qc_aerosol == 3)

    w[is_good] = wmax
    w[is_bad] = wmin
    return w


def qc_sentinel2_scl(scl: np.ndarray, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> np.ndarray:
    """
    Weights from the Sentinel-2 L2A Scene Classification Layer (SCL).
    Port of phenofit's `qc_sentinel2()` (R/qcFUN.R).

        4 vegetation, 5 bare soil, 6 water, 7 unclassified, 10 thin cirrus -> wmax
        8 cloud medium probability                                        -> wmid
        1 saturated/defective, 2 dark area, 3 cloud shadow, 9 cloud high  -> wmin
        11 snow (kept usable for phenology, per phenofit)                 -> wmin
        other (e.g. 0 no-data)                                            -> wmin
    """
    scl = np.asarray(scl)
    qc_good = (4, 5, 6, 7, 10)
    qc_mid = (8,)

    w = np.full(scl.shape, wmin, dtype=np.float64)
    w[np.isin(scl, qc_good)] = wmax
    w[np.isin(scl, qc_mid)] = wmid
    return w
