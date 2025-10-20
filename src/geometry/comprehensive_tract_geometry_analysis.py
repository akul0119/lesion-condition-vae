#!/usr/bin/env python3
"""
Comprehensive Tract Geometry Analysis for All Groups and Timepoints

This script processes all tracts for Sham, TBI, and PTE groups across all timepoints
using the geometric feature extraction from tract_geom_proc.py.

Author: Comprehensive tract geometry pipeline
Date: October 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import gzip
import warnings
warnings.filterwarnings('ignore')

# Import tract geometry processing functions
from tract_geom_proc import compute_streamline_metrics

# Tract list as specified
TRACT_LIST = [
    'chip_right', 'hipcom', 'thalsub_left',
    'cing_left', 'thalsub_right',
    'cing_right',
    'fimbria_left', 'ant_comm', 'fimbria_right',
    'atr_left', 'fornix_left', 'intcap_left',
    'atr_right', 'chip_left', 'fornix_right', 'intcap_right'
]

def load_config():
    """Load tract configuration with subject metadata."""
    config_path = Path(__file__).parent / "lesion_vae_analysis" / "configs" / "tract_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def get_all_subjects(config):
    """Extract all subjects from config (Sham, TBI, PTE)."""
    subjects_by_group = {}
    groups = config.get('groups', {})

    for group, subject_list in groups.items():
        if group in ['Sham', 'TBI', 'PTE']:
            # Convert to string IDs
            subjects_by_group[group] = [str(subj) for subj in subject_list]

    return subjects_by_group


def decompress_vtk_if_needed(vtk_path):
    """
    Decompress .vtk.gz file if needed.
    Returns path to decompressed file (or original if not .gz).
    """
    vtk_path = Path(vtk_path)

    if vtk_path.suffix == '.gz':
        # Create temporary decompressed file
        decompressed_path = vtk_path.parent / vtk_path.stem

        if not decompressed_path.exists():
            try:
                with gzip.open(vtk_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        f_out.write(f_in.read())
            except Exception as e:
                print(f"      [ERROR] Failed to decompress {vtk_path}: {e}")
                return None

        return str(decompressed_path)

    return str(vtk_path)


def process_single_tract(subject_id, timepoint, tract_name, data_dir, group, max_streamlines=None):
    """
    Process a single tract and compute geometric features.

    Returns dictionary with all geometric metrics or None if processing fails.
    """
    # Construct path to tract file
    tract_file = data_dir / subject_id / timepoint / "bundles" / f"{tract_name}_curves.vtk.gz"

    if not tract_file.exists():
        # Try without .gz
        tract_file = data_dir / subject_id / timepoint / "bundles" / f"{tract_name}_curves.vtk"
        if not tract_file.exists():
            print(f"      [WARN] Tract file not found: {tract_file}")
            return None

    try:
        # Decompress if needed
        vtk_path = decompress_vtk_if_needed(tract_file)
        if vtk_path is None:
            return None

        # Compute metrics using tract_geom_proc functions
        df_sl, df_bundle = compute_streamline_metrics(vtk_path, max_streamlines=max_streamlines)

        if len(df_bundle) == 0:
            print(f"      [WARN] No streamlines processed for {tract_name}")
            return None

        # Extract bundle-level metrics
        metrics = df_bundle.iloc[0].to_dict()

        # Add metadata
        metrics['subject_id'] = subject_id
        metrics['timepoint'] = timepoint
        metrics['tract'] = tract_name
        metrics['group'] = group

        print(f"      ✓ {tract_name}: {metrics['n_streamlines']} streamlines, "
              f"length={metrics['length_mean']:.1f}mm")

        # Clean up temporary decompressed file if we created one
        if vtk_path != str(tract_file) and Path(vtk_path).exists():
            try:
                Path(vtk_path).unlink()
            except:
                pass

        return metrics

    except Exception as e:
        print(f"      [ERROR] Failed to process {tract_name}: {e}")
        return None


def process_all_tracts(config, data_dir, output_dir, max_streamlines=None):
    """
    Process all tracts for all subjects across all timepoints.

    Returns DataFrame with comprehensive tract geometry metrics.
    """
    subjects_by_group = get_all_subjects(config)

    # Count total subjects
    total_subjects = sum(len(subjs) for subjs in subjects_by_group.values())

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TRACT GEOMETRY ANALYSIS")
    print(f"{'='*80}")
    print(f"Groups: {list(subjects_by_group.keys())}")
    print(f"Total subjects: {total_subjects}")
    print(f"  - Sham: {len(subjects_by_group.get('Sham', []))}")
    print(f"  - TBI: {len(subjects_by_group.get('TBI', []))}")
    print(f"  - PTE: {len(subjects_by_group.get('PTE', []))}")
    print(f"Timepoints: 2d, 9d, 1mo, 5mo")
    print(f"Tracts: {len(TRACT_LIST)}")
    print(f"Max streamlines per tract: {max_streamlines if max_streamlines else 'all'}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timepoints to process
    timepoints = ['2d', '9d', '1mo', '5mo']

    results = []
    total_processed = 0
    total_failed = 0

    # Process each group
    for group, subjects in subjects_by_group.items():
        print(f"\n{'='*80}")
        print(f"Processing {group} group ({len(subjects)} subjects)")
        print(f"{'='*80}")

        for subject_id in sorted(subjects):
            print(f"\n  [{subject_id}] Group: {group}")

            for timepoint in timepoints:
                print(f"    Processing {timepoint}...")

                tp_success = 0
                tp_failed = 0

                for tract in TRACT_LIST:
                    metrics = process_single_tract(
                        subject_id, timepoint, tract, data_dir, group,
                        max_streamlines=max_streamlines
                    )

                    if metrics is not None:
                        results.append(metrics)
                        tp_success += 1
                        total_processed += 1
                    else:
                        tp_failed += 1
                        total_failed += 1

                print(f"    {timepoint}: ✓ {tp_success} tracts, ✗ {tp_failed} failed")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n[ERROR] No tracts successfully processed!")
        return results_df

    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total records: {len(results_df)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed: {total_failed}")
    print(f"\nRecords by group:")
    print(results_df.groupby('group').size())
    print(f"\nRecords by timepoint:")
    print(results_df.groupby('timepoint').size())
    print(f"\nRecords by tract:")
    print(results_df.groupby('tract').size())
    print(f"{'='*80}\n")

    return results_df


def generate_summary_statistics(results_df, output_dir):
    """Generate summary statistics by group, timepoint, and tract."""
    print(f"\n{'='*80}")
    print(f"GENERATING SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    # Group-level summary for key metrics
    key_metrics = [
        'length_mean', 'tortuosity_mean', 'curv_mean_avg',
        'elongation_ratio_mean', 'planarity_ratio_mean'
    ]

    # Summary by group and timepoint
    print("Summary by Group and Timepoint:")
    print("-" * 80)

    summary_stats = []

    for group in sorted(results_df['group'].unique()):
        for tp in sorted(results_df['timepoint'].unique()):
            subset = results_df[(results_df['group'] == group) & (results_df['timepoint'] == tp)]

            if len(subset) > 0:
                stat_row = {
                    'group': group,
                    'timepoint': tp,
                    'n_records': len(subset),
                    'n_subjects': subset['subject_id'].nunique(),
                    'n_tracts': subset['tract'].nunique()
                }

                for metric in key_metrics:
                    if metric in subset.columns:
                        stat_row[f'{metric}_mean'] = subset[metric].mean()
                        stat_row[f'{metric}_std'] = subset[metric].std()

                summary_stats.append(stat_row)

    summary_df = pd.DataFrame(summary_stats)

    # Save summary statistics
    summary_path = output_dir / "summary_statistics_by_group_timepoint.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved: {summary_path}")

    # Per-tract summary
    tract_summary = []

    for tract in sorted(results_df['tract'].unique()):
        for group in sorted(results_df['group'].unique()):
            subset = results_df[(results_df['tract'] == tract) & (results_df['group'] == group)]

            if len(subset) > 0:
                tract_row = {
                    'tract': tract,
                    'group': group,
                    'n_records': len(subset),
                    'length_mean': subset['length_mean'].mean(),
                    'length_std': subset['length_mean'].std(),
                    'tortuosity_mean': subset['tortuosity_mean'].mean(),
                    'tortuosity_std': subset['tortuosity_mean'].std(),
                    'curv_mean': subset['curv_mean_avg'].mean(),
                    'curv_std': subset['curv_mean_avg'].std()
                }
                tract_summary.append(tract_row)

    tract_summary_df = pd.DataFrame(tract_summary)

    # Save tract summary
    tract_summary_path = output_dir / "summary_statistics_by_tract_group.csv"
    tract_summary_df.to_csv(tract_summary_path, index=False)
    print(f"✓ Saved: {tract_summary_path}")

    return summary_df, tract_summary_df


def main():
    """Main analysis pipeline."""
    # Paths
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "lesion_vae_analysis" / "results" / "comprehensive_tract_geometry"

    # Load config
    config = load_config()

    # Process all tracts (limit to 100 streamlines per tract for faster processing)
    # Set to None to process all streamlines
    results_df = process_all_tracts(config, data_dir, output_dir, max_streamlines=100)

    if len(results_df) == 0:
        print("[ERROR] No results to save!")
        return

    # Save main results
    output_csv = output_dir / "comprehensive_tract_geometry_metrics.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Main results saved: {output_csv}")
    print(f"  Shape: {results_df.shape}")
    print(f"  Columns: {len(results_df.columns)}")

    # Generate summary statistics
    summary_df, tract_summary_df = generate_summary_statistics(results_df, output_dir)

    print(f"\n{'='*80}")
    print(f"ALL ANALYSIS COMPLETE!")
    print(f"Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
