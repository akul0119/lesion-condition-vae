#!/usr/bin/env python3
"""
Comprehensive Spherical Harmonic Shape Descriptor Analysis for Lesion Masks

Computes rotation-invariant SH shape descriptors up to degree L=6 for lesions.
Includes:
- Largest connected component extraction
- Unit volume normalization
- Uniform sphere sampling
- Power spectrum computation
- 3D visualizations
- Group-level analysis by timepoint and outcome
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pandas as pd
import json
from scipy import special, ndimage
from scipy.spatial import distance_matrix
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Try to import skimage for marching cubes
try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARN] skimage not available - using alternative surface extraction")


def load_config():
    """Load tract configuration with subject metadata."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "tract_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def get_tbi_pte_subjects(config):
    """Extract TBI and PTE subjects from config (exclude shams)."""
    subjects_by_group = {}

    groups = config.get('groups', {})

    for group, subject_list in groups.items():
        if group in ['TBI', 'PTE']:
            # Convert to string IDs
            subjects_by_group[group] = [str(subj) for subj in subject_list]

    return subjects_by_group


def extract_largest_connected_component(mask_data, threshold=0.5):
    """
    Extract the largest connected component from a binary or probabilistic mask.

    Parameters
    ----------
    mask_data : ndarray
        3D lesion mask (binary or probabilistic)
    threshold : float
        Threshold for binarization

    Returns
    -------
    largest_cc : ndarray
        Binary mask containing only the largest connected component
    """
    # Binarize
    binary_mask = mask_data > threshold

    # Label connected components
    labeled, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return np.zeros_like(mask_data)

    # Find largest component
    component_sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1

    largest_cc = (labeled == largest_label).astype(float)

    return largest_cc


def compute_centroid(mask_data):
    """Compute the centroid of a binary mask."""
    coords = np.argwhere(mask_data > 0)
    if len(coords) == 0:
        return np.array([0, 0, 0])
    centroid = coords.mean(axis=0)
    return centroid


def normalize_to_unit_volume(mask_data, affine):
    """
    Normalize lesion to unit volume while preserving shape.

    Returns scaling factor and normalized coordinates.
    """
    voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
    lesion_volume = np.sum(mask_data > 0) * voxel_volume

    # Scale factor to achieve unit volume
    # V_new = scale^3 * V_old = 1
    scale_factor = lesion_volume ** (-1/3)

    return scale_factor, lesion_volume


def fibonacci_sphere(num_samples=2000):
    """
    Generate uniformly distributed points on a unit sphere using Fibonacci spiral.

    Parameters
    ----------
    num_samples : int
        Number of points to generate

    Returns
    -------
    directions : ndarray (num_samples, 3)
        Unit vectors uniformly distributed on sphere
    theta : ndarray (num_samples,)
        Polar angles
    phi : ndarray (num_samples,)
        Azimuthal angles
    """
    golden_ratio = (1 + 5**0.5) / 2
    i = np.arange(num_samples)

    # Fibonacci spiral
    theta = np.arccos(1 - 2 * (i + 0.5) / num_samples)
    phi = 2 * np.pi * i / golden_ratio

    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    directions = np.column_stack([x, y, z])

    return directions, theta, phi


def extract_surface_points(mask_data, affine, num_points=2000):
    """
    Extract surface points using marching cubes or erosion-based method.

    Returns physical coordinates of surface points.
    """
    if SKIMAGE_AVAILABLE and np.sum(mask_data > 0) > 100:
        try:
            # Use marching cubes for better surface extraction
            verts, faces, normals, values = marching_cubes(mask_data, level=0.5)

            # Convert to physical coordinates
            surface_coords = nib.affines.apply_affine(affine, verts)

            # Subsample to desired number of points
            if len(surface_coords) > num_points:
                indices = np.random.choice(len(surface_coords), num_points, replace=False)
                surface_coords = surface_coords[indices]

            return surface_coords
        except Exception as e:
            print(f"[WARN] Marching cubes failed: {e}, using erosion method")

    # Fallback: erosion-based surface extraction
    binary_mask = mask_data > 0.5
    eroded = ndimage.binary_erosion(binary_mask, iterations=1)
    surface = binary_mask & ~eroded

    surface_voxels = np.argwhere(surface)
    surface_coords = nib.affines.apply_affine(affine, surface_voxels)

    # Subsample if too many points
    if len(surface_coords) > num_points:
        indices = np.random.choice(len(surface_coords), num_points, replace=False)
        surface_coords = surface_coords[indices]

    return surface_coords


def sample_radius_function(surface_coords, centroid, directions):
    """
    Sample the radius function r(θ,φ) by finding the maximum radius
    along each direction from the centroid.

    Parameters
    ----------
    surface_coords : ndarray (N, 3)
        Surface point coordinates in physical space
    centroid : ndarray (3,)
        Lesion centroid in physical space
    directions : ndarray (M, 3)
        Unit direction vectors on sphere

    Returns
    -------
    radii : ndarray (M,)
        Radius values along each direction
    """
    # Center surface points
    centered_surface = surface_coords - centroid

    # For each direction, find the maximum radius
    radii = np.zeros(len(directions))

    for i, direction in enumerate(directions):
        # Project all surface points onto this direction
        projections = np.dot(centered_surface, direction)
        # Take maximum projection (furthest point along this direction)
        radii[i] = np.max(projections) if len(projections) > 0 else 0

    return radii


def compute_spherical_harmonic(l, m, theta, phi):
    """Compute real spherical harmonic Y_l^m(θ,φ)."""
    # scipy.special.sph_harm uses (m, l, phi, theta) order
    Y = special.sph_harm(m, l, phi, theta)

    # Convert to real spherical harmonics
    if m < 0:
        # Y_l^m = i/√2 * (Y_l^|m| - (-1)^m * Y_l^{-|m|})
        return np.imag(Y) * np.sqrt(2)
    elif m > 0:
        # Y_l^m = 1/√2 * (Y_l^{-m} + (-1)^m * Y_l^m)
        return np.real(Y) * np.sqrt(2)
    else:
        # m = 0: already real
        return np.real(Y)


def fit_spherical_harmonics(radii, theta, phi, max_l=6):
    """
    Fit spherical harmonics to radius function using least squares.

    Solves: r(θ,φ) = Σ c_l^m Y_l^m(θ,φ)

    Parameters
    ----------
    radii : ndarray (N,)
        Observed radius values
    theta, phi : ndarray (N,)
        Spherical coordinates
    max_l : int
        Maximum SH degree

    Returns
    -------
    coefficients : dict
        SH coefficients {(l, m): c_l^m}
    """
    num_samples = len(radii)

    # Build design matrix
    # Each column corresponds to one Y_l^m basis function
    basis_functions = []
    indices = []

    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm = compute_spherical_harmonic(l, m, theta, phi)
            basis_functions.append(Y_lm)
            indices.append((l, m))

    # Design matrix: (num_samples, num_coefficients)
    A = np.column_stack(basis_functions)

    # Solve least squares: A * c = radii
    result = lsq_linear(A, radii)
    coeffs = result.x

    # Package into dictionary
    coefficients = {idx: coeff for idx, coeff in zip(indices, coeffs)}

    return coefficients


def compute_power_spectrum(coefficients, max_l=6):
    """
    Compute rotation-invariant power spectrum P_l.

    P_l = Σ_{m=-l}^{l} |c_l^m|^2

    Parameters
    ----------
    coefficients : dict
        SH coefficients {(l, m): c_l^m}
    max_l : int
        Maximum degree

    Returns
    -------
    powers : dict
        Power at each degree {l: P_l}
    """
    powers = {}

    for l in range(max_l + 1):
        power = 0
        for m in range(-l, l + 1):
            if (l, m) in coefficients:
                power += coefficients[(l, m)]**2
        powers[l] = power

    return powers


def normalize_powers(powers):
    """Normalize power spectrum so that Σ P_l = 1."""
    total_power = sum(powers.values())
    if total_power > 0:
        return {l: p / total_power for l, p in powers.items()}
    else:
        return powers


def reconstruct_surface(coefficients, theta, phi, max_l=6):
    """Reconstruct radius function from SH coefficients."""
    reconstructed = np.zeros_like(theta)

    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            if (l, m) in coefficients:
                Y_lm = compute_spherical_harmonic(l, m, theta, phi)
                reconstructed += coefficients[(l, m)] * Y_lm

    return reconstructed


def plot_3d_surface_comparison(original_coords, reconstructed_coords, subject_id, timepoint, output_dir):
    """
    Generate 3D visualization comparing original and reconstructed surfaces.
    """
    fig = plt.figure(figsize=(16, 7))

    # Original surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_coords[:, 0], original_coords[:, 1], original_coords[:, 2],
                c='blue', alpha=0.3, s=1)
    ax1.set_title(f'Original Surface\n{subject_id} @ {timepoint}')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')

    # Reconstructed surface
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstructed_coords[:, 0], reconstructed_coords[:, 1], reconstructed_coords[:, 2],
                c='red', alpha=0.3, s=1)
    ax2.set_title(f'SH Reconstructed (L=6)\n{subject_id} @ {timepoint}')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')

    plt.tight_layout()
    output_path = output_dir / f'surface_3d_{subject_id}_{timepoint}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_power_spectrum(powers, subject_id, timepoint, output_dir):
    """Generate bar plot of normalized power spectrum."""
    fig, ax = plt.subplots(figsize=(10, 6))

    degrees = sorted(powers.keys())
    power_values = [powers[l] for l in degrees]

    ax.bar(degrees, power_values, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Spherical Harmonic Degree (l)', fontsize=12)
    ax.set_ylabel('Normalized Power (P_l)', fontsize=12)
    ax.set_title(f'SH Power Spectrum: {subject_id} @ {timepoint}', fontsize=14)
    ax.set_xticks(degrees)
    ax.grid(axis='y', alpha=0.3)

    # Add values on bars
    for l, p in zip(degrees, power_values):
        ax.text(l, p, f'{p:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / f'power_spectrum_{subject_id}_{timepoint}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def compute_brain_volume(subject_id, timepoint, data_dir):
    """
    Compute total brain volume from tissue mask.

    Returns brain volume in mm³ or None if not available.
    """
    # Try tissue mask first (most reliable)
    tissue_path = data_dir / subject_id / timepoint / "tissue.nii.gz"

    if tissue_path.exists():
        try:
            tissue_img = nib.load(tissue_path)
            tissue_data = tissue_img.get_fdata()
            affine = tissue_img.affine

            # Compute voxel volume
            voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))

            # Tissue mask should already be binary (brain tissue = 1, background = 0)
            brain_mask = tissue_data > 0

            # Compute volume
            brain_volume = np.sum(brain_mask) * voxel_volume

            return brain_volume

        except Exception as e:
            print(f"    [WARN] Could not load tissue mask: {e}")

    # Fallback: try to estimate from FA map
    fa_path = data_dir / subject_id / timepoint / "dti_FA.nii.gz"

    if fa_path.exists():
        try:
            fa_img = nib.load(fa_path)
            fa_data = fa_img.get_fdata()
            affine = fa_img.affine

            # Compute voxel volume
            voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))

            # Brain mask: FA > 0.1 (conservative threshold)
            brain_mask = fa_data > 0.1

            # Clean up mask
            from scipy.ndimage import binary_erosion, binary_dilation
            brain_mask = binary_erosion(brain_mask, iterations=1)
            brain_mask = binary_dilation(brain_mask, iterations=1)

            # Compute volume
            brain_volume = np.sum(brain_mask) * voxel_volume

            return brain_volume

        except Exception as e:
            print(f"    [WARN] Could not compute brain volume from FA: {e}")

    return None


def analyze_single_lesion(subject_id, timepoint, data_dir, output_dir, max_l=6, num_samples=2000):
    """
    Analyze a single lesion mask and compute SH descriptors.

    Returns
    -------
    result : dict
        Dictionary containing all computed features
    success : bool
        Whether analysis was successful
    """
    # Use cleaned lesion mask
    lesion_path = data_dir / subject_id / timepoint / "lesion_cleaned.nii.gz"

    if not lesion_path.exists():
        print(f"[WARN] Cleaned lesion not found: {lesion_path}")
        return None, False

    # Load lesion mask
    lesion_img = nib.load(lesion_path)
    lesion_data = lesion_img.get_fdata()
    affine = lesion_img.affine

    # Compute brain volume
    brain_volume = compute_brain_volume(subject_id, timepoint, data_dir)

    # Extract largest connected component
    largest_cc = extract_largest_connected_component(lesion_data)

    if np.sum(largest_cc) < 10:
        print(f"[WARN] Lesion too small: {subject_id} @ {timepoint}")
        return None, False

    # Compute centroid in voxel space
    centroid_voxel = compute_centroid(largest_cc)
    centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)

    # Normalize to unit volume
    scale_factor, original_volume = normalize_to_unit_volume(largest_cc, affine)

    # Extract surface points
    surface_coords = extract_surface_points(largest_cc, affine, num_points=num_samples)

    if len(surface_coords) < 100:
        print(f"[WARN] Insufficient surface points: {subject_id} @ {timepoint}")
        return None, False

    # Generate uniform sphere sampling directions
    directions, theta, phi = fibonacci_sphere(num_samples)

    # Sample radius function
    radii = sample_radius_function(surface_coords, centroid_mm, directions)

    # Normalize radii by scale factor
    radii_normalized = radii * scale_factor

    # Fit spherical harmonics
    coefficients = fit_spherical_harmonics(radii_normalized, theta, phi, max_l=max_l)

    # Compute power spectrum
    powers = compute_power_spectrum(coefficients, max_l=max_l)
    powers_normalized = normalize_powers(powers)

    # Reconstruct surface for visualization
    radii_reconstructed = reconstruct_surface(coefficients, theta, phi, max_l=max_l)

    # Convert back to physical coordinates for visualization
    reconstructed_coords = centroid_mm + directions * radii_reconstructed[:, np.newaxis] / scale_factor

    # Generate visualizations
    print(f"  Generating 3D comparison plot...")
    plot_3d_surface_comparison(surface_coords, reconstructed_coords, subject_id, timepoint, output_dir)

    print(f"  Generating power spectrum plot...")
    plot_power_spectrum(powers_normalized, subject_id, timepoint, output_dir)

    # Compute lesion-to-brain volume ratio
    lesion_brain_ratio = original_volume / brain_volume if brain_volume is not None else None

    # Package results
    result = {
        'subject_id': subject_id,
        'timepoint': timepoint,
        'original_volume_mm3': original_volume,
        'brain_volume_mm3': brain_volume,
        'lesion_brain_ratio': lesion_brain_ratio,
        'scale_factor': scale_factor,
        'centroid_x': centroid_mm[0],
        'centroid_y': centroid_mm[1],
        'centroid_z': centroid_mm[2],
        'num_surface_points': len(surface_coords),
    }

    # Add normalized powers
    for l in range(max_l + 1):
        result[f'P{l}'] = powers_normalized.get(l, 0)

    # Add raw powers (unnormalized)
    for l in range(max_l + 1):
        result[f'P{l}_raw'] = powers.get(l, 0)

    # Add selected coefficients (for reference)
    # Store c_l^0 (axisymmetric components) for each degree
    for l in range(max_l + 1):
        if (l, 0) in coefficients:
            result[f'c{l}_0'] = coefficients[(l, 0)]

    # Reconstruction quality (correlation between original and reconstructed)
    correlation = np.corrcoef(radii_normalized, radii_reconstructed)[0, 1]
    result['reconstruction_r'] = correlation

    return result, True


def analyze_all_lesions(config, data_dir, output_dir, max_l=6, num_samples=2000):
    """
    Analyze all TBI and PTE lesions across all timepoints.

    Returns
    -------
    results_df : DataFrame
        One row per subject × timepoint
    """
    subjects_by_group = get_tbi_pte_subjects(config)
    all_subjects = []
    group_mapping = {}

    for group, subjects in subjects_by_group.items():
        all_subjects.extend(subjects)
        for subj in subjects:
            group_mapping[subj] = group

    print(f"\n{'='*80}")
    print(f"SPHERICAL HARMONIC LESION SHAPE DESCRIPTOR ANALYSIS")
    print(f"{'='*80}")
    print(f"Subjects: {len(all_subjects)} ({subjects_by_group})")
    print(f"Max SH degree: L={max_l}")
    print(f"Sphere samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Timepoints to check
    timepoints = ['2d', '9d', '1mo', '5mo']

    results = []

    for subject_id in sorted(all_subjects):
        group = group_mapping[subject_id]
        print(f"\n[{subject_id}] Group: {group}")

        for timepoint in timepoints:
            print(f"  Analyzing {timepoint}...")

            result, success = analyze_single_lesion(
                subject_id, timepoint, data_dir, plots_dir,
                max_l=max_l, num_samples=num_samples
            )

            if success and result is not None:
                result['group'] = group
                results.append(result)
                print(f"  ✓ Success - Volume: {result['original_volume_mm3']:.2f} mm³, "
                      f"Reconstruction R: {result['reconstruction_r']:.3f}")
            else:
                print(f"  ✗ Failed or no lesion")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n[ERROR] No lesions successfully analyzed!")
        return results_df

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"Total lesions analyzed: {len(results_df)}")
    print(f"By group: {results_df.groupby('group').size().to_dict()}")
    print(f"By timepoint: {results_df.groupby('timepoint').size().to_dict()}")
    print(f"{'='*80}\n")

    return results_df


def compute_group_statistics(results_df, output_dir):
    """
    Compute group-level statistics (mean ± SD) by timepoint and outcome group.
    """
    print(f"\n{'='*80}")
    print(f"GROUP-LEVEL STATISTICS")
    print(f"{'='*80}\n")

    # Power columns
    power_cols = [col for col in results_df.columns if col.startswith('P') and not col.endswith('_raw')]

    # Group by outcome and timepoint
    grouped = results_df.groupby(['group', 'timepoint'])

    stats_rows = []

    for (group, timepoint), group_data in grouped:
        n = len(group_data)
        print(f"\n{group} @ {timepoint} (n={n})")
        print(f"{'-'*40}")

        stats_row = {
            'group': group,
            'timepoint': timepoint,
            'n': n
        }

        for col in power_cols:
            mean = group_data[col].mean()
            std = group_data[col].std()
            stats_row[f'{col}_mean'] = mean
            stats_row[f'{col}_std'] = std

            if col in ['P0', 'P2', 'P4', 'P6']:  # Key degrees
                print(f"  {col}: {mean:.4f} ± {std:.4f}")

        # Reconstruction quality
        recon_mean = group_data['reconstruction_r'].mean()
        recon_std = group_data['reconstruction_r'].std()
        stats_row['reconstruction_r_mean'] = recon_mean
        stats_row['reconstruction_r_std'] = recon_std
        print(f"  Reconstruction R: {recon_mean:.3f} ± {recon_std:.3f}")

        stats_rows.append(stats_row)

    stats_df = pd.DataFrame(stats_rows)

    # Save statistics
    stats_path = output_dir / "group_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✓ Group statistics saved: {stats_path}")

    return stats_df


def visualize_group_spectra(results_df, stats_df, output_dir):
    """
    Visualize SH power spectra by group and timepoint.
    """
    print(f"\nGenerating group-level visualizations...")

    power_cols = [f'P{l}' for l in range(7)]
    timepoints = sorted(results_df['timepoint'].unique())
    groups = sorted(results_df['group'].unique())

    # 1. Spectra by group (averaged across timepoints)
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 5))
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        group_data = results_df[results_df['group'] == group]

        for timepoint in timepoints:
            tp_data = group_data[group_data['timepoint'] == timepoint]
            if len(tp_data) > 0:
                means = [tp_data[col].mean() for col in power_cols]
                stds = [tp_data[col].std() for col in power_cols]
                degrees = np.arange(len(power_cols))

                ax.errorbar(degrees, means, yerr=stds, marker='o', label=timepoint,
                           capsize=5, alpha=0.7)

        ax.set_xlabel('Spherical Harmonic Degree (l)')
        ax.set_ylabel('Normalized Power (P_l)')
        ax.set_title(f'{group} (n={len(group_data)})')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(range(7))

    plt.suptitle('SH Power Spectra by Group and Timepoint', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "group_spectra_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

    # 2. Heatmap of P2, P4, P6 by group and timepoint
    key_powers = ['P2', 'P4', 'P6']

    fig, axes = plt.subplots(1, len(key_powers), figsize=(16, 5))

    for ax, power in zip(axes, key_powers):
        pivot_data = results_df.pivot_table(
            values=power,
            index='timepoint',
            columns='group',
            aggfunc='mean'
        )

        im = ax.imshow(pivot_data.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        ax.set_title(f'{power} Power')
        ax.set_xlabel('Group')
        ax.set_ylabel('Timepoint')

        # Add values
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                value = pivot_data.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='white' if value > 0.15 else 'black', fontsize=10)

        plt.colorbar(im, ax=ax, label='Mean Power')

    plt.suptitle('Key SH Powers by Group and Timepoint', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "heatmap_key_powers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

    # 3. Temporal evolution for each group
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # P2 evolution
    ax = axes[0]
    for group in groups:
        group_means = []
        group_stds = []
        group_timepoints = []

        for timepoint in timepoints:
            data = results_df[(results_df['group'] == group) & (results_df['timepoint'] == timepoint)]
            if len(data) > 0:
                group_means.append(data['P2'].mean())
                group_stds.append(data['P2'].std())
                group_timepoints.append(timepoint)

        if len(group_means) > 0:
            ax.errorbar(range(len(group_timepoints)), group_means, yerr=group_stds,
                       marker='o', label=group, capsize=5, linewidth=2, markersize=8)

    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints)
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('P₂ Power (Ellipticity)', fontsize=12)
    ax.set_title('Temporal Evolution of P₂ (Ellipsoidal Deformation)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    # P4 evolution
    ax = axes[1]
    for group in groups:
        group_means = []
        group_stds = []
        group_timepoints = []

        for timepoint in timepoints:
            data = results_df[(results_df['group'] == group) & (results_df['timepoint'] == timepoint)]
            if len(data) > 0:
                group_means.append(data['P4'].mean())
                group_stds.append(data['P4'].std())
                group_timepoints.append(timepoint)

        if len(group_means) > 0:
            ax.errorbar(range(len(group_timepoints)), group_means, yerr=group_stds,
                       marker='o', label=group, capsize=5, linewidth=2, markersize=8)

    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints)
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('P₄ Power (Higher-Order Shape)', fontsize=12)
    ax.set_title('Temporal Evolution of P₄ (Complex Shape Features)', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle('Temporal Evolution of SH Shape Descriptors', fontsize=14, y=1.00)
    plt.tight_layout()

    output_path = output_dir / "temporal_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def visualize_brain_volume_analysis(results_df, output_dir):
    """
    Visualize lesion metrics normalized by brain volume.
    """
    print(f"\nGenerating brain-volume-normalized visualizations...")

    # Filter data with brain volume
    data_with_brain = results_df[results_df['brain_volume_mm3'].notna()].copy()

    if len(data_with_brain) == 0:
        print("  [WARN] No brain volume data available")
        return

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Brain-Volume-Normalized Lesion Analysis (Cleaned Lesions)',
                fontsize=18, fontweight='bold', y=0.995)

    timepoints = sorted(data_with_brain['timepoint'].unique())
    groups = sorted(data_with_brain['group'].unique())

    # 1. Lesion-to-brain ratio by group and timepoint
    ax1 = fig.add_subplot(gs[0, 0])

    for group in groups:
        means = []
        stds = []
        for tp in timepoints:
            subset = data_with_brain[(data_with_brain['group'] == group) &
                                    (data_with_brain['timepoint'] == tp)]
            if len(subset) > 0:
                means.append(subset['lesion_brain_ratio'].mean() * 100)  # as percentage
                stds.append(subset['lesion_brain_ratio'].std() * 100)
            else:
                means.append(np.nan)
                stds.append(np.nan)

        x = np.arange(len(timepoints))
        ax1.errorbar(x, means, yerr=stds, marker='o', label=group,
                    capsize=5, linewidth=2, markersize=8)

    ax1.set_xticks(range(len(timepoints)))
    ax1.set_xticklabels(timepoints)
    ax1.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Lesion Volume / Brain Volume (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Lesion-to-Brain Volume Ratio', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # 2. Absolute lesion volume by group and timepoint
    ax2 = fig.add_subplot(gs[0, 1])

    for group in groups:
        means = []
        stds = []
        for tp in timepoints:
            subset = data_with_brain[(data_with_brain['group'] == group) &
                                    (data_with_brain['timepoint'] == tp)]
            if len(subset) > 0:
                means.append(subset['original_volume_mm3'].mean())
                stds.append(subset['original_volume_mm3'].std())
            else:
                means.append(np.nan)
                stds.append(np.nan)

        x = np.arange(len(timepoints))
        ax2.errorbar(x, means, yerr=stds, marker='o', label=group,
                    capsize=5, linewidth=2, markersize=8)

    ax2.set_xticks(range(len(timepoints)))
    ax2.set_xticklabels(timepoints)
    ax2.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Lesion Volume (mm³)', fontweight='bold', fontsize=12)
    ax2.set_title('Absolute Lesion Volume', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    # 3. Brain volume by group and timepoint
    ax3 = fig.add_subplot(gs[0, 2])

    for group in groups:
        means = []
        stds = []
        for tp in timepoints:
            subset = data_with_brain[(data_with_brain['group'] == group) &
                                    (data_with_brain['timepoint'] == tp)]
            if len(subset) > 0:
                means.append(subset['brain_volume_mm3'].mean())
                stds.append(subset['brain_volume_mm3'].std())
            else:
                means.append(np.nan)
                stds.append(np.nan)

        x = np.arange(len(timepoints))
        ax3.errorbar(x, means, yerr=stds, marker='o', label=group,
                    capsize=5, linewidth=2, markersize=8)

    ax3.set_xticks(range(len(timepoints)))
    ax3.set_xticklabels(timepoints)
    ax3.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Brain Volume (mm³)', fontweight='bold', fontsize=12)
    ax3.set_title('Total Brain Volume', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)

    # 4. Scatter: lesion volume vs brain volume
    ax4 = fig.add_subplot(gs[1, 0])

    colors_map = {'TBI': 'coral', 'PTE': 'steelblue'}
    for group in groups:
        group_data = data_with_brain[data_with_brain['group'] == group]
        ax4.scatter(group_data['brain_volume_mm3'],
                   group_data['original_volume_mm3'],
                   alpha=0.6, s=80, label=group,
                   c=colors_map.get(group, 'gray'),
                   edgecolors='black', linewidth=1)

    ax4.set_xlabel('Brain Volume (mm³)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Lesion Volume (mm³)', fontweight='bold', fontsize=12)
    ax4.set_title('Lesion vs Brain Volume', fontweight='bold', fontsize=13)
    ax4.legend(fontsize=11)
    ax4.grid(alpha=0.3)

    # 5. Distribution of lesion-brain ratio
    ax5 = fig.add_subplot(gs[1, 1])

    for group in groups:
        group_data = data_with_brain[data_with_brain['group'] == group]
        ratios = group_data['lesion_brain_ratio'].dropna() * 100
        if len(ratios) > 0:
            ax5.hist(ratios, bins=15, alpha=0.6, label=group,
                    color=colors_map.get(group, 'gray'),
                    edgecolor='black', linewidth=1)

    ax5.set_xlabel('Lesion / Brain Volume (%)', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Frequency', fontweight='bold', fontsize=12)
    ax5.set_title('Distribution of Lesion-Brain Ratio', fontweight='bold', fontsize=13)
    ax5.legend(fontsize=11)
    ax5.grid(alpha=0.3, axis='y')

    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_text = "BRAIN VOLUME SUMMARY\n\n"

    for group in groups:
        group_data = data_with_brain[data_with_brain['group'] == group]
        summary_text += f"{group} (n={len(group_data)}):\n"
        summary_text += f"  Brain Volume:\n"
        summary_text += f"    Mean: {group_data['brain_volume_mm3'].mean():.0f} mm³\n"
        summary_text += f"    SD: {group_data['brain_volume_mm3'].std():.0f} mm³\n"
        summary_text += f"  Lesion Volume:\n"
        summary_text += f"    Mean: {group_data['original_volume_mm3'].mean():.1f} mm³\n"
        summary_text += f"    SD: {group_data['original_volume_mm3'].std():.1f} mm³\n"
        summary_text += f"  Lesion/Brain Ratio:\n"
        summary_text += f"    Mean: {group_data['lesion_brain_ratio'].mean()*100:.2f}%\n"
        summary_text += f"    SD: {group_data['lesion_brain_ratio'].std()*100:.2f}%\n"
        summary_text += "\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(output_dir / "brain_volume_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: brain_volume_analysis.png")


def main():
    """Main analysis pipeline."""
    # Paths
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    output_dir = Path(__file__).parent.parent.parent / "results" / "lesion_sh_descriptors_cleaned"

    # Load config
    config = load_config()

    # Run analysis
    results_df = analyze_all_lesions(config, data_dir, output_dir, max_l=6, num_samples=2000)

    if len(results_df) == 0:
        print("[ERROR] No results to save!")
        return

    # Save results
    output_csv = output_dir / "lesion_sh_descriptors.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")
    print(f"  Shape: {results_df.shape}")

    # Compute group statistics
    stats_df = compute_group_statistics(results_df, output_dir)

    # Visualize group-level spectra
    visualize_group_spectra(results_df, stats_df, output_dir)

    # Visualize brain volume analysis
    visualize_brain_volume_analysis(results_df, output_dir)

    print(f"\n{'='*80}")
    print(f"ALL ANALYSIS COMPLETE!")
    print(f"Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
