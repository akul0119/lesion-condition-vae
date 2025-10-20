"""
Data loading and preprocessing for lesion-conditioned VAE analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path


def csv_path(base_path, subject_id, tp):
    """
    Generate path to subject's comprehensive tract data CSV

    Args:
        base_path: Base directory path
        subject_id: Subject ID
        tp: Timepoint (e.g., "2d", "1mo")

    Returns:
        Path to CSV file
    """
    return Path(base_path) / "results" / str(subject_id) / \
           f"timepoint_analysis_{subject_id}_{tp}" / \
           f"comprehensive_tract_data_{subject_id}_{tp}.csv"


def build_tensor_with_lesion_context(base_path, tract, tp, subjects,
                                     micro_feats, lesion_feats, groups_dict):
    """
    Build tensors including both microstructure and lesion information

    Args:
        base_path: Base directory containing data
        tract: Tract name (e.g., "fimbria_left")
        tp: Timepoint
        subjects: List of subject IDs
        micro_feats: List of microstructure feature names
        lesion_feats: List of lesion feature names
        groups_dict: Dictionary mapping group names to subject IDs

    Returns:
        X_micro: (n_streamlines, 100, n_micro_feats) microstructure data
        X_lesion: (n_streamlines, 100, n_lesion_feats) lesion data
        subj_ids: (n_streamlines,) subject IDs
        group_labels: (n_streamlines,) group labels
        s: (100,) position along tract values
    """
    X_micro_list, X_lesion_list = [], []
    subj_stream_ids, group_stream_labels = [], []

    # Create subject to group mapping
    subj_to_group = {}
    for group_name, group_subjects in groups_dict.items():
        for sid in group_subjects:
            subj_to_group[sid] = group_name

    for sid in subjects:
        fp = csv_path(base_path, sid, tp)
        if not fp.exists():
            print(f"[WARN] Missing {fp}")
            continue

        df = pd.read_csv(fp)
        df = df[df["tract_id"] == tract].copy()
        if df.empty:
            continue

        # Check for missing microstructure features
        missing_micro = [c for c in micro_feats if c not in df.columns]
        if missing_micro:
            print(f"[WARN] Missing microstructure features in {fp}: {missing_micro}")
            continue

        subject_group = subj_to_group[sid]

        # Handle missing lesion features
        for lf in lesion_feats:
            if lf not in df.columns:
                if subject_group == 'Sham':
                    if lf in ["in_lesion", "in_cavity"]:
                        df[lf] = False
                    elif lf == "lesion_distance":
                        df[lf] = 15.0
                else:
                    if lf in ["in_lesion", "in_cavity"]:
                        df[lf] = False
                    elif lf == "lesion_distance":
                        df[lf] = 15.0

                    if tp in ['2d', '9d']:
                        print(f"[INFO] {sid} ({subject_group}) @ {tp}: " +
                              "lesion data may be minimal (acute phase)")

        df.sort_values(["streamline_id", "point_id"], inplace=True)
        stream_ids = sorted(df["streamline_id"].unique())
        nodes = sorted(df["point_id"].unique())

        if len(nodes) != 100:
            print(f"[WARN] {fp} has {len(nodes)} nodes, expected 100")
            continue

        # Pivot to wide format
        wide_micro = df.pivot(index="point_id", columns="streamline_id",
                             values=micro_feats)
        wide_lesion = df.pivot(index="point_id", columns="streamline_id",
                              values=lesion_feats)

        for s_id in stream_ids:
            mat_micro = wide_micro.xs(s_id, axis=1, level=1)
            mat_micro = mat_micro.reindex(nodes)

            mat_lesion = wide_lesion.xs(s_id, axis=1, level=1)
            mat_lesion = mat_lesion.reindex(nodes)

            # Process lesion values
            lesion_vals = mat_lesion.values.astype(np.float32)
            lesion_vals[:, 2] = np.clip(lesion_vals[:, 2], 0, 15) / 15.0

            X_micro_list.append(mat_micro.values.astype(np.float32))
            X_lesion_list.append(lesion_vals)
            subj_stream_ids.append(sid)
            group_stream_labels.append(subj_to_group[sid])

    if not X_micro_list:
        raise ValueError(f"No data for {tract} @ {tp}")

    X_micro = np.stack(X_micro_list, axis=0)
    X_lesion = np.stack(X_lesion_list, axis=0)
    subj_ids = np.array(subj_stream_ids)
    group_labels = np.array(group_stream_labels)

    # Get position along tract
    s_vals = df["position_along_tract"].unique()
    s = np.linspace(0, 1, 100) if len(s_vals) != 100 else np.sort(s_vals.astype(float))

    # Print summary statistics
    for group in ['Sham', 'TBI', 'PTE']:
        group_mask = group_labels == group
        if group_mask.any():
            group_lesion = X_lesion[group_mask]
            in_lesion_pct = (group_lesion[:, :, 0].mean() * 100)
            in_cavity_pct = (group_lesion[:, :, 1].mean() * 100)
            print(f"  {group}: {group_mask.sum()} streamlines, " +
                  f"lesion={in_lesion_pct:.2f}%, cavity={in_cavity_pct:.2f}%")

    print(f"[INFO] {tract}@{tp}: X_micro={X_micro.shape}, X_lesion={X_lesion.shape}")

    return X_micro, X_lesion, subj_ids, group_labels, s
