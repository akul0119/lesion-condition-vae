import numpy as np
import pandas as pd
import pyvista as pv
from typing import List, Dict, Tuple, Optional

# ---------------------------
# Utility: streamline loading
# ---------------------------
def read_streamlines_from_vtk(vtk_path: str, max_streamlines: Optional[int] = None) -> List[np.ndarray]:
    mesh = pv.read(vtk_path)
    points = mesh.points
    lines = mesh.lines
    streamlines = []
    i = 0
    n = len(lines)

    while i < n:
        num_pts = int(lines[i])
        idx = lines[i+1 : i+1+num_pts]
        sl = points[idx]
        if sl.shape[0] > 2 and sl.shape[1] == 3 and np.isfinite(sl).all():
            streamlines.append(sl)
            if max_streamlines is not None and len(streamlines) >= max_streamlines:
                break
        i += 1 + num_pts
    return streamlines

# --------------------------------
# Geometry helpers (discrete curve)
# --------------------------------
def arc_length(sl: np.ndarray) -> float:
    diffs = np.diff(sl, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())

def end_to_end(sl: np.ndarray) -> float:
    return float(np.linalg.norm(sl[-1] - sl[0]))

def tortuosity(sl: np.ndarray) -> float:
    L = arc_length(sl)
    d = end_to_end(sl)
    return float(L / max(d, 1e-8))

def straightness(sl: np.ndarray) -> float:
    L = arc_length(sl)
    d = end_to_end(sl)
    return float(d / max(L, 1e-8))

def discrete_derivatives(sl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.gradient(sl, axis=0)
    a = np.gradient(v, axis=0)
    return v, a

def mean_curvature(sl: np.ndarray) -> float:
    if sl.shape[0] < 3:
        return 0.0
    v, a = discrete_derivatives(sl)
    cross_mag = np.linalg.norm(np.cross(v, a), axis=1)
    v_mag = np.linalg.norm(v, axis=1) + 1e-12
    kappa = cross_mag / (v_mag**3)
    kappa = kappa[np.isfinite(kappa)]
    return float(kappa.mean()) if kappa.size else 0.0

def std_curvature(sl: np.ndarray) -> float:
    if sl.shape[0] < 3:
        return 0.0
    v, a = discrete_derivatives(sl)
    cross_mag = np.linalg.norm(np.cross(v, a), axis=1)
    v_mag = np.linalg.norm(v, axis=1) + 1e-12
    kappa = cross_mag / (v_mag**3)
    kappa = kappa[np.isfinite(kappa)]
    return float(kappa.std()) if kappa.size else 0.0

def curvature_energy(sl: np.ndarray) -> float:
    if sl.shape[0] < 3:
        return 0.0
    diffs = np.diff(sl, axis=0)
    ds = np.linalg.norm(diffs, axis=1) + 1e-12
    v, a = discrete_derivatives(sl)
    cross_mag = np.linalg.norm(np.cross(v, a), axis=1)
    v_mag = np.linalg.norm(v, axis=1) + 1e-12
    kappa = np.nan_to_num(cross_mag / (v_mag**3))
    m = min(len(kappa), len(ds))
    return float(np.sum((kappa[:m]**2) * ds[:m]))

def torsion_mean(sl: np.ndarray) -> float:
    if sl.shape[0] < 4:
        return 0.0
    v = np.gradient(sl, axis=0)
    a = np.gradient(v, axis=0)
    b = np.cross(v, a)
    db = np.gradient(b, axis=0)
    denom = (np.linalg.norm(b, axis=1)**2) + 1e-12
    num = np.einsum('ij,ij->i', b, db)
    tau = num / denom
    tau = tau[np.isfinite(tau)]
    return float(np.mean(tau)) if tau.size else 0.0

def bending_angle_mean(sl: np.ndarray) -> float:
    if sl.shape[0] < 3:
        return 0.0
    diffs = np.diff(sl, axis=0)
    seg = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12)
    cosines = np.einsum('ij,ij->i', seg[:-1], seg[1:])
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    return float(np.mean(np.abs(angles))) if angles.size else 0.0

# -------------------------------------------
# Shape / compactness and spatial embedding
# -------------------------------------------
def centroid(sl: np.ndarray) -> np.ndarray:
    return sl.mean(axis=0)

def bounding_box_volume(sl: np.ndarray) -> float:
    mins = sl.min(axis=0)
    maxs = sl.max(axis=0)
    return float(np.prod(maxs - mins))

def pca_eigs(sl: np.ndarray) -> np.ndarray:
    X = sl - centroid(sl)
    C = np.cov(X.T)
    vals, _ = np.linalg.eigh(C)
    vals = np.sort(vals)[::-1]
    return vals  # λ1 ≥ λ2 ≥ λ3

def elongation_ratio(sl: np.ndarray) -> float:
    vals = pca_eigs(sl)
    if vals[1] <= 1e-12:
        return float('inf')
    return float(vals[0] / vals[1])

def planarity_ratio(sl: np.ndarray) -> float:
    vals = pca_eigs(sl)
    if vals[2] <= 1e-12:
        return float('inf')
    return float(vals[1] / vals[2])

def anisotropy_ratio(sl: np.ndarray) -> float:
    vals = pca_eigs(sl)
    denom = vals.sum() + 1e-12
    return float(vals[0] / denom)

def angular_dispersion(sl: np.ndarray) -> float:
    d = np.diff(sl, axis=0)
    t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    mean_t = t.mean(axis=0)
    var = np.mean(np.linalg.norm(t - mean_t, axis=1)**2)
    return float(var)

# --------------------------------------------
# Main API: compute metrics for TBI vs PTE work
# --------------------------------------------
def compute_streamline_metrics(vtk_path: str, max_streamlines: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns: df_sl (per streamline) and df_bundle (bundle-level summary)."""
    sls = read_streamlines_from_vtk(vtk_path, max_streamlines=max_streamlines)

    rows = []
    for sl in sls:
        L = arc_length(sl)
        if L <= 1e-8:
            continue

        e2e = end_to_end(sl)
        rows.append({
            # I. Local geometry
            "length": L,
            "end_to_end": e2e,
            "tortuosity": tortuosity(sl),
            "straightness": straightness(sl),
            "curv_mean": mean_curvature(sl),
            "curv_std": std_curvature(sl),
            "curv_energy": curvature_energy(sl),
            "torsion_mean": torsion_mean(sl),
            "bend_angle_mean": bending_angle_mean(sl),

            # II. Shape / compactness
            "bbox_vol": bounding_box_volume(sl),
            "elongation_ratio": elongation_ratio(sl),
            "planarity_ratio": planarity_ratio(sl),
            "anisotropy_ratio": anisotropy_ratio(sl),

            # III. Spatial embedding
            "centroid_x": float(centroid(sl)[0]),
            "centroid_y": float(centroid(sl)[1]),
            "centroid_z": float(centroid(sl)[2]),
            "ang_dispersion": angular_dispersion(sl)
        })

    df_sl = pd.DataFrame(rows)

    # Bundle-level summaries
    def _safe_mean(x): return float(np.nanmean(x)) if len(x) else np.nan
    def _safe_std(x):  return float(np.nanstd(x)) if len(x) else np.nan

    bundle = {
        "n_streamlines": len(df_sl),
        "length_mean": _safe_mean(df_sl["length"]),
        "tortuosity_mean": _safe_mean(df_sl["tortuosity"]),
        "curv_mean_avg": _safe_mean(df_sl["curv_mean"]),
        "curv_energy_mean": _safe_mean(df_sl["curv_energy"]),
        "torsion_mean_avg": _safe_mean(df_sl["torsion_mean"]),
        "bend_angle_mean_avg": _safe_mean(df_sl["bend_angle_mean"]),
        "elongation_ratio_mean": _safe_mean(df_sl["elongation_ratio"]),
        "planarity_ratio_mean": _safe_mean(df_sl["planarity_ratio"]),
        "anisotropy_ratio_mean": _safe_mean(df_sl["anisotropy_ratio"]),
        "ang_dispersion_mean": _safe_mean(df_sl["ang_dispersion"]),
        "centroid_x_mean": _safe_mean(df_sl["centroid_x"]),
        "centroid_y_mean": _safe_mean(df_sl["centroid_y"]),
        "centroid_z_mean": _safe_mean(df_sl["centroid_z"]),
    }
    df_bundle = pd.DataFrame([bundle])
    return df_sl, df_bundle

