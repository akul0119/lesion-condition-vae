"""
Normalization functions for microstructure and lesion data
"""

import numpy as np


def fit_normalization_stats(X_micro, X_lesion, feat_names):
    """
    Fit normalization statistics on data

    Args:
        X_micro: (n_samples, n_points, n_features) microstructure data
        X_lesion: (n_samples, n_points, n_lesion_feats) lesion data
        feat_names: List of feature names

    Returns:
        Dictionary with "median", "mean", and "std" keys
    """
    X = X_micro.reshape(-1, X_micro.shape[-1])
    finite = np.isfinite(X)

    meds = np.zeros(X.shape[1], dtype=np.float32)
    mus = np.zeros(X.shape[1], dtype=np.float32)
    stds = np.zeros(X.shape[1], dtype=np.float32)

    for j, fj in enumerate(feat_names):
        xj = X[:, j]
        mask = finite[:, j]
        if mask.sum() > 0:
            meds[j] = np.nanmedian(xj[mask])
            mus[j] = np.nanmean(xj[mask])
            s = np.nanstd(xj[mask])
            stds[j] = max(s, 1e-6)
        else:
            meds[j] = 0.0
            mus[j] = 0.0
            stds[j] = 1.0

    return {"median": meds, "mean": mus, "std": stds}


def apply_normalization(X_micro, X_lesion, stats):
    """
    Apply normalization using pre-computed statistics

    Args:
        X_micro: (n_samples, n_points, n_features) microstructure data
        X_lesion: (n_samples, n_points, n_lesion_feats) lesion data
        stats: Dictionary with "median", "mean", and "std" keys

    Returns:
        Xz: Normalized microstructure data
        X_lesion: Cleaned lesion data
    """
    Xc = X_micro.copy()
    med = stats["median"][None, None, :]
    bad = ~np.isfinite(Xc)
    if bad.any():
        Xc[bad] = np.broadcast_to(med, Xc.shape)[bad]

    mu = stats["mean"][None, None, :]
    std = stats["std"][None, None, :]
    Xz = (Xc - mu) / std
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    X_lesion = np.nan_to_num(X_lesion, nan=0.0).astype(np.float32)

    return Xz, X_lesion
