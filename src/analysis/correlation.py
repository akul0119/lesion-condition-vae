#!/usr/bin/env python3
"""
Correlation Analysis: Lesion SH Descriptors vs Tract Geometry Metrics

Analyzes correlations between lesion shape (SH descriptors) and tract geometry
for PTE and TBI groups at timepoints 9d, 1mo, and 5mo.

Generates scatter plots for significant correlations (p < 0.05 uncorrected).

Author: Lesion-Tract Correlation Pipeline
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("muted")


def load_data():
    """Load lesion SH and tract geometry datasets."""
    print(f"\n{'='*80}")
    print(f"LOADING DATA")
    print(f"{'='*80}")

    # Paths
    base_path = Path(__file__).parent / "lesion_vae_analysis" / "results"

    lesion_path = base_path / "lesion_sh_heme_comprehensive" / "lesion_sh_heme_comprehensive.csv"
    tract_path = base_path / "comprehensive_tract_geometry" / "comprehensive_tract_geometry_metrics.csv"

    # Load datasets
    df_lesion = pd.read_csv(lesion_path)
    df_tract = pd.read_csv(tract_path)

    print(f"\nLesion SH data: {df_lesion.shape}")
    print(f"Tract geometry data: {df_tract.shape}")

    # Filter to TBI and PTE only
    df_lesion = df_lesion[df_lesion['group'].isin(['TBI', 'PTE'])].copy()
    df_tract = df_tract[df_tract['group'].isin(['TBI', 'PTE'])].copy()

    print(f"\nAfter filtering to TBI/PTE:")
    print(f"Lesion SH: {df_lesion.shape}")
    print(f"Tract geometry: {df_tract.shape}")

    return df_lesion, df_tract


def aggregate_tract_metrics(df_tract, timepoint, subject_id):
    """Aggregate tract metrics across all tracts for a subject."""
    df_subj = df_tract[
        (df_tract['timepoint'] == timepoint) &
        (df_tract['subject_id'] == subject_id)
    ]

    if len(df_subj) == 0:
        return None

    # Geometric feature columns
    feature_cols = [
        'n_streamlines', 'length_mean', 'tortuosity_mean', 'curv_mean_avg',
        'curv_energy_mean', 'torsion_mean_avg', 'bend_angle_mean_avg',
        'elongation_ratio_mean', 'planarity_ratio_mean', 'anisotropy_ratio_mean',
        'ang_dispersion_mean'
    ]

    # Aggregate by mean
    agg_metrics = {}
    for col in feature_cols:
        if col in df_subj.columns:
            agg_metrics[col] = df_subj[col].mean()

    return agg_metrics


def merge_lesion_tract_data(df_lesion, df_tract):
    """Merge lesion SH descriptors with aggregated tract metrics."""
    print(f"\n{'='*80}")
    print(f"MERGING DATASETS")
    print(f"{'='*80}")

    timepoints = ['9d', '1mo', '5mo']

    merged_data = []

    for tp in timepoints:
        df_lesion_tp = df_lesion[df_lesion['timepoint'] == tp]

        for _, row in df_lesion_tp.iterrows():
            subject_id = row['subject_id']
            group = row['group']

            # Get aggregated tract metrics
            tract_metrics = aggregate_tract_metrics(df_tract, tp, subject_id)

            if tract_metrics is None:
                continue

            # Combine lesion and tract data
            merged_row = {
                'subject_id': subject_id,
                'timepoint': tp,
                'group': group,
                # Lesion SH descriptors
                'P0': row['P0'],
                'P1': row['P1'],
                'P2': row['P2'],
                'P3': row['P3'],
                'P4': row['P4'],
                'P5': row['P5'],
                'P6': row['P6'],
                'lesion_volume': row['original_volume_mm3'],
                'lesion_brain_ratio': row['lesion_brain_ratio'],
                'brain_volume': row['brain_volume_mm3'],
            }

            # Add tract metrics
            merged_row.update(tract_metrics)

            merged_data.append(merged_row)

    df_merged = pd.DataFrame(merged_data)

    print(f"\nMerged dataset: {df_merged.shape}")
    print(f"Subjects by group:")
    print(df_merged.groupby(['group', 'timepoint']).size())

    return df_merged


def compute_correlations(df, group, timepoint, sh_features, tract_features):
    """
    Compute correlations between SH descriptors and tract metrics.

    Returns list of significant correlations.
    """
    df_subset = df[(df['group'] == group) & (df['timepoint'] == timepoint)]

    if len(df_subset) < 3:
        return []

    significant_corrs = []

    for sh_feat in sh_features:
        for tract_feat in tract_features:
            # Get data
            x = df_subset[sh_feat].values
            y = df_subset[tract_feat].values

            # Remove NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < 3:
                continue

            # Compute correlation
            r, p = pearsonr(x_clean, y_clean)

            # Check if significant (p < 0.05 uncorrected)
            if p < 0.05:
                significant_corrs.append({
                    'group': group,
                    'timepoint': timepoint,
                    'sh_feature': sh_feat,
                    'tract_feature': tract_feat,
                    'r': r,
                    'p': p,
                    'n': len(x_clean)
                })

    return significant_corrs


def plot_correlation_scatter(df, sh_feat, tract_feat, timepoint, output_dir, corr_info):
    """
    Create seaborn scatter plot with line of best fit for each group.
    Uses style="whitegrid", palette="muted"
    """
    # Set seaborn style explicitly for this plot
    sns.set_style("whitegrid")

    # Prepare data for this timepoint
    df_plot = df[df['timepoint'] == timepoint].copy()

    # Remove rows with NaN in either feature
    df_plot = df_plot.dropna(subset=[sh_feat, tract_feat])

    if len(df_plot) < 3:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use seaborn scatterplot with palette="muted"
    sns.scatterplot(data=df_plot, x=sh_feat, y=tract_feat, hue='group',
                   style='group', s=100, alpha=0.7,
                   palette="muted", edgecolor='black', linewidth=1.5,
                   ax=ax, legend=True)

    # Add regression lines for each group
    groups = ['TBI', 'PTE']
    colors = sns.color_palette("muted")
    group_colors = {'TBI': colors[3], 'PTE': colors[0]}  # coral/red for TBI, blue for PTE

    stats_text_list = []

    for group in groups:
        df_group = df_plot[df_plot['group'] == group]

        if len(df_group) < 3:
            continue

        x = df_group[sh_feat].values
        y = df_group[tract_feat].values

        # Line of best fit
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), color=group_colors[group],
               linewidth=2.5, alpha=0.8, linestyle='-', zorder=2)

        # Compute correlation
        r, p_val = pearsonr(x, y)

        # Collect stats text
        stats_text_list.append(f"{group}: r={r:.3f}, p={p_val:.4f}, n={len(x)}")

    # Add combined stats text box
    stats_text = '\n'.join(stats_text_list)
    ax.text(0.05, 0.95, stats_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat',
                    alpha=0.5, edgecolor='black', linewidth=1.5))

    ax.set_xlabel(sh_feat, fontweight='bold', fontsize=13)
    ax.set_ylabel(tract_feat, fontweight='bold', fontsize=13)
    ax.set_title(f'{sh_feat} vs {tract_feat}\nTimepoint: {timepoint}',
                fontweight='bold', fontsize=14)

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=11, loc='upper right',
             frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    # Save
    filename = f'corr_{sh_feat}_vs_{tract_feat}_{timepoint}.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def analyze_correlations(df_merged, output_dir):
    """
    Main correlation analysis function.
    """
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*80}")

    # Define features
    sh_features = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
                   'lesion_volume', 'lesion_brain_ratio']

    tract_features = [
        'n_streamlines', 'length_mean', 'tortuosity_mean', 'curv_mean_avg',
        'curv_energy_mean', 'torsion_mean_avg', 'bend_angle_mean_avg',
        'elongation_ratio_mean', 'planarity_ratio_mean', 'anisotropy_ratio_mean',
        'ang_dispersion_mean'
    ]

    timepoints = ['9d', '1mo', '5mo']
    groups = ['TBI', 'PTE']

    all_significant = []

    # Compute correlations for each group and timepoint
    for tp in timepoints:
        print(f"\n{tp}:")

        for group in groups:
            sig_corrs = compute_correlations(
                df_merged, group, tp, sh_features, tract_features
            )

            all_significant.extend(sig_corrs)

            print(f"  {group}: {len(sig_corrs)} significant correlations (p<0.05)")

    # Create DataFrame of significant correlations
    df_sig = pd.DataFrame(all_significant)

    if len(df_sig) == 0:
        print("\n[WARN] No significant correlations found!")
        return df_sig

    print(f"\nTotal significant correlations: {len(df_sig)}")

    # Save significant correlations
    sig_path = output_dir / 'significant_correlations.csv'
    df_sig.to_csv(sig_path, index=False)
    print(f"\n✓ Saved: {sig_path}")

    # Generate plots for significant correlations
    print(f"\n{'='*80}")
    print(f"GENERATING SCATTER PLOTS")
    print(f"{'='*80}")

    plots_dir = output_dir / "correlation_plots"
    plots_dir.mkdir(exist_ok=True)

    # Get unique combinations of features and timepoints
    unique_combos = df_sig[['sh_feature', 'tract_feature', 'timepoint']].drop_duplicates()

    print(f"\nGenerating {len(unique_combos)} plots...")

    plot_count = 0
    for _, row in unique_combos.iterrows():
        sh_feat = row['sh_feature']
        tract_feat = row['tract_feature']
        tp = row['timepoint']

        # Get correlation info for both groups
        corr_info = df_sig[
            (df_sig['sh_feature'] == sh_feat) &
            (df_sig['tract_feature'] == tract_feat) &
            (df_sig['timepoint'] == tp)
        ]

        try:
            output_path = plot_correlation_scatter(
                df_merged, sh_feat, tract_feat, tp, plots_dir, corr_info
            )
            plot_count += 1

            if plot_count % 10 == 0:
                print(f"  Generated {plot_count}/{len(unique_combos)} plots...")

        except Exception as e:
            print(f"  [ERROR] Failed to plot {sh_feat} vs {tract_feat} at {tp}: {e}")

    print(f"\n✓ Generated {plot_count} scatter plots")
    print(f"✓ Saved to: {plots_dir}")

    return df_sig


def create_summary_heatmap(df_sig, output_dir):
    """
    Create heatmap summarizing significant correlations.
    """
    print(f"\n{'='*80}")
    print(f"CREATING SUMMARY VISUALIZATIONS")
    print(f"{'='*80}")

    if len(df_sig) == 0:
        print("[WARN] No significant correlations to visualize")
        return

    # Count significant correlations by feature pair
    timepoints = ['9d', '1mo', '5mo']

    for tp in timepoints:
        df_tp = df_sig[df_sig['timepoint'] == tp]

        if len(df_tp) == 0:
            continue

        # Create separate plots for each group
        for group in ['TBI', 'PTE']:
            df_group = df_tp[df_tp['group'] == group]

            if len(df_group) == 0:
                continue

            # Create pivot table of r values
            pivot = df_group.pivot_table(
                values='r',
                index='sh_feature',
                columns='tract_feature',
                aggfunc='first'
            )

            if pivot.empty:
                continue

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(14, 8))

            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, vmin=-1, vmax=1,
                       cbar_kws={'label': 'Pearson r'},
                       linewidths=0.5, linecolor='gray',
                       ax=ax)

            ax.set_xlabel('Tract Geometry Features', fontweight='bold', fontsize=12)
            ax.set_ylabel('Lesion SH Descriptors', fontweight='bold', fontsize=12)
            ax.set_title(f'Significant Correlations: {group} at {tp}\n(p < 0.05 uncorrected)',
                        fontweight='bold', fontsize=14)

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)

            plt.tight_layout()

            output_path = output_dir / f'correlation_heatmap_{group}_{tp}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved: {output_path.name}")


def create_summary_report(df_sig, output_dir):
    """Generate summary report of correlation analysis."""
    if len(df_sig) == 0:
        print("\n[WARN] No significant correlations for summary report")
        return

    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")

    # Overall summary
    print(f"\nTotal significant correlations: {len(df_sig)}")
    print(f"\nBy group:")
    print(df_sig.groupby('group').size())
    print(f"\nBy timepoint:")
    print(df_sig.groupby('timepoint').size())

    # Top correlations
    print(f"\n{'='*80}")
    print(f"TOP 10 STRONGEST CORRELATIONS")
    print(f"{'='*80}")

    df_top = df_sig.nlargest(10, 'r')

    for idx, row in df_top.iterrows():
        print(f"\n{row['group']} @ {row['timepoint']}:")
        print(f"  {row['sh_feature']} <-> {row['tract_feature']}")
        print(f"  r = {row['r']:.3f}, p = {row['p']:.4f}, n = {row['n']}")

    # Most frequently significant features
    print(f"\n{'='*80}")
    print(f"MOST FREQUENTLY SIGNIFICANT FEATURES")
    print(f"{'='*80}")

    print(f"\nLesion SH Descriptors:")
    sh_counts = df_sig['sh_feature'].value_counts()
    for feat, count in sh_counts.head(5).items():
        print(f"  {feat}: {count} significant correlations")

    print(f"\nTract Geometry Metrics:")
    tract_counts = df_sig['tract_feature'].value_counts()
    for feat, count in tract_counts.head(5).items():
        print(f"  {feat}: {count} significant correlations")


def main():
    """Main analysis pipeline."""
    print(f"\n{'='*80}")
    print(f"LESION SH vs TRACT GEOMETRY CORRELATION ANALYSIS")
    print(f"{'='*80}")

    # Output directory
    output_dir = Path(__file__).parent / "lesion_vae_analysis" / "results" / \
                 "lesion_tract_correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_lesion, df_tract = load_data()

    # Merge datasets
    df_merged = merge_lesion_tract_data(df_lesion, df_tract)

    # Analyze correlations
    df_sig = analyze_correlations(df_merged, output_dir)

    # Create summary visualizations
    if len(df_sig) > 0:
        create_summary_heatmap(df_sig, output_dir)
        create_summary_report(df_sig, output_dir)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults directory: {output_dir}")
    print(f"✓ Significant correlations: {len(df_sig)}")
    print(f"✓ Scatter plots: {output_dir / 'correlation_plots'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
