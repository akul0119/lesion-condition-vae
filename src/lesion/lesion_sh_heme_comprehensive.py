#!/usr/bin/env python3
"""
Comprehensive Lesion SH Descriptor and Heme Content Analysis

This script extends the lesion SH analysis to include:
1. All timepoints (2d, 9d, 1mo, 5mo)
2. For 2d with no lesion: SH descriptors set to 0, but brain_volume_mm3 included
3. Heme content analysis for all timepoints
4. Combined dataset with lesion SH and heme data for TBI and PTE rodents

Author: Extended analysis pipeline
Date: 2025
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
    """Extract the largest connected component from a binary or probabilistic mask."""
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
    """Normalize lesion to unit volume while preserving shape."""
    voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
    lesion_volume = np.sum(mask_data > 0) * voxel_volume

    # Scale factor to achieve unit volume
    scale_factor = lesion_volume ** (-1/3)

    return scale_factor, lesion_volume


def fibonacci_sphere(num_samples=2000):
    """Generate uniformly distributed points on a unit sphere using Fibonacci spiral."""
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
    """Extract surface points using marching cubes or erosion-based method."""
    if SKIMAGE_AVAILABLE and np.sum(mask_data > 0) > 100:
        try:
            verts, faces, normals, values = marching_cubes(mask_data, level=0.5)
            surface_coords = nib.affines.apply_affine(affine, verts)

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

    if len(surface_coords) > num_points:
        indices = np.random.choice(len(surface_coords), num_points, replace=False)
        surface_coords = surface_coords[indices]

    return surface_coords


def sample_radius_function(surface_coords, centroid, directions):
    """Sample the radius function r(θ,φ)."""
    centered_surface = surface_coords - centroid
    radii = np.zeros(len(directions))

    for i, direction in enumerate(directions):
        projections = np.dot(centered_surface, direction)
        radii[i] = np.max(projections) if len(projections) > 0 else 0

    return radii


def compute_spherical_harmonic(l, m, theta, phi):
    """Compute real spherical harmonic Y_l^m(θ,φ)."""
    Y = special.sph_harm(m, l, phi, theta)

    if m < 0:
        return np.imag(Y) * np.sqrt(2)
    elif m > 0:
        return np.real(Y) * np.sqrt(2)
    else:
        return np.real(Y)


def fit_spherical_harmonics(radii, theta, phi, max_l=6):
    """Fit spherical harmonics to radius function using least squares."""
    basis_functions = []
    indices = []

    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            Y_lm = compute_spherical_harmonic(l, m, theta, phi)
            basis_functions.append(Y_lm)
            indices.append((l, m))

    A = np.column_stack(basis_functions)
    result = lsq_linear(A, radii)
    coeffs = result.x

    coefficients = {idx: coeff for idx, coeff in zip(indices, coeffs)}
    return coefficients


def compute_power_spectrum(coefficients, max_l=6):
    """Compute rotation-invariant power spectrum P_l."""
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


def compute_brain_volume(subject_id, timepoint, data_dir):
    """Compute total brain volume from tissue mask."""
    tissue_path = data_dir / subject_id / timepoint / "tissue.nii.gz"

    if tissue_path.exists():
        try:
            tissue_img = nib.load(tissue_path)
            tissue_data = tissue_img.get_fdata()
            affine = tissue_img.affine
            voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
            brain_mask = tissue_data > 0
            brain_volume = np.sum(brain_mask) * voxel_volume
            return brain_volume
        except Exception as e:
            print(f"    [WARN] Could not load tissue mask: {e}")

    # Fallback: try FA map
    fa_path = data_dir / subject_id / timepoint / "dti_FA.nii.gz"

    if fa_path.exists():
        try:
            fa_img = nib.load(fa_path)
            fa_data = fa_img.get_fdata()
            affine = fa_img.affine
            voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))
            brain_mask = fa_data > 0.1
            brain_mask = ndimage.binary_erosion(brain_mask, iterations=1)
            brain_mask = ndimage.binary_dilation(brain_mask, iterations=1)
            brain_volume = np.sum(brain_mask) * voxel_volume
            return brain_volume
        except Exception as e:
            print(f"    [WARN] Could not compute brain volume from FA: {e}")

    return None


def compute_heme_content(subject_id, timepoint, data_dir):
    """
    Compute heme content metrics from heme.nii.gz file.

    Returns dictionary with:
    - heme_mean: Mean heme signal across brain
    - heme_std: Standard deviation of heme signal
    - heme_total: Total heme signal (sum)
    - heme_max: Maximum heme signal
    - heme_volume_mm3: Volume of voxels with significant heme (> threshold)
    """
    heme_path = data_dir / subject_id / timepoint / "heme.nii.gz"

    if not heme_path.exists():
        print(f"    [WARN] Heme file not found: {heme_path}")
        return None

    try:
        heme_img = nib.load(heme_path)
        heme_data = heme_img.get_fdata()
        affine = heme_img.affine
        voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))

        # Mask out background (heme > 0)
        brain_mask = heme_data > 0

        if np.sum(brain_mask) == 0:
            print(f"    [WARN] No heme signal detected")
            return {
                'heme_mean': 0.0,
                'heme_std': 0.0,
                'heme_total': 0.0,
                'heme_max': 0.0,
                'heme_volume_mm3': 0.0,
                'heme_95percentile': 0.0
            }

        heme_values = heme_data[brain_mask]

        # Compute metrics
        heme_metrics = {
            'heme_mean': float(np.mean(heme_values)),
            'heme_std': float(np.std(heme_values)),
            'heme_total': float(np.sum(heme_values)),
            'heme_max': float(np.max(heme_values)),
            'heme_95percentile': float(np.percentile(heme_values, 95)),
        }

        # Compute volume of high heme content (> 95th percentile)
        threshold = np.percentile(heme_values, 95)
        high_heme_mask = heme_data > threshold
        heme_metrics['heme_volume_mm3'] = float(np.sum(high_heme_mask) * voxel_volume)

        return heme_metrics

    except Exception as e:
        print(f"    [ERROR] Failed to compute heme content: {e}")
        return None


def analyze_single_lesion(subject_id, timepoint, data_dir, output_dir, max_l=6, num_samples=2000):
    """
    Analyze a single lesion mask and compute SH descriptors.
    If lesion is too small or doesn't exist, returns result with SH values = 0.
    """
    lesion_path = data_dir / subject_id / timepoint / "lesion_cleaned.nii.gz"

    # Compute brain volume (always needed)
    brain_volume = compute_brain_volume(subject_id, timepoint, data_dir)

    # Initialize result with zeros
    result = {
        'subject_id': subject_id,
        'timepoint': timepoint,
        'original_volume_mm3': 0.0,
        'brain_volume_mm3': brain_volume,
        'lesion_brain_ratio': 0.0,
        'scale_factor': 0.0,
        'centroid_x': 0.0,
        'centroid_y': 0.0,
        'centroid_z': 0.0,
        'num_surface_points': 0,
    }

    # Add zero SH descriptors
    for l in range(max_l + 1):
        result[f'P{l}'] = 0.0
        result[f'P{l}_raw'] = 0.0
        result[f'c{l}_0'] = 0.0

    result['reconstruction_r'] = 0.0

    # Check if lesion file exists
    if not lesion_path.exists():
        print(f"    [INFO] No lesion file found - using zeros for SH descriptors")
        return result, True

    # Load lesion mask
    try:
        lesion_img = nib.load(lesion_path)
        lesion_data = lesion_img.get_fdata()
        affine = lesion_img.affine

        # Extract largest connected component
        largest_cc = extract_largest_connected_component(lesion_data)

        if np.sum(largest_cc) < 10:
            print(f"    [INFO] Lesion too small - using zeros for SH descriptors")
            return result, True

        # Compute centroid in voxel space
        centroid_voxel = compute_centroid(largest_cc)
        centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)

        # Normalize to unit volume
        scale_factor, original_volume = normalize_to_unit_volume(largest_cc, affine)

        # Extract surface points
        surface_coords = extract_surface_points(largest_cc, affine, num_points=num_samples)

        if len(surface_coords) < 100:
            print(f"    [INFO] Insufficient surface points - using zeros for SH descriptors")
            return result, True

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

        # Reconstruct surface
        radii_reconstructed = reconstruct_surface(coefficients, theta, phi, max_l=max_l)

        # Compute lesion-to-brain volume ratio
        lesion_brain_ratio = original_volume / brain_volume if brain_volume is not None else 0.0

        # Update result with actual values
        result.update({
            'original_volume_mm3': original_volume,
            'lesion_brain_ratio': lesion_brain_ratio,
            'scale_factor': scale_factor,
            'centroid_x': centroid_mm[0],
            'centroid_y': centroid_mm[1],
            'centroid_z': centroid_mm[2],
            'num_surface_points': len(surface_coords),
        })

        # Add normalized powers
        for l in range(max_l + 1):
            result[f'P{l}'] = powers_normalized.get(l, 0)

        # Add raw powers
        for l in range(max_l + 1):
            result[f'P{l}_raw'] = powers.get(l, 0)

        # Add selected coefficients
        for l in range(max_l + 1):
            if (l, 0) in coefficients:
                result[f'c{l}_0'] = coefficients[(l, 0)]

        # Reconstruction quality
        correlation = np.corrcoef(radii_normalized, radii_reconstructed)[0, 1]
        result['reconstruction_r'] = correlation

        print(f"    ✓ Lesion analyzed - Volume: {original_volume:.2f} mm³, Reconstruction R: {correlation:.3f}")

    except Exception as e:
        print(f"    [WARN] Error analyzing lesion: {e} - using zeros")

    return result, True


def analyze_all_lesions_and_heme(config, data_dir, output_dir, max_l=6, num_samples=2000):
    """
    Analyze all TBI and PTE subjects across all timepoints.
    Includes both lesion SH descriptors and heme content.
    """
    subjects_by_group = get_tbi_pte_subjects(config)
    all_subjects = []
    group_mapping = {}

    for group, subjects in subjects_by_group.items():
        all_subjects.extend(subjects)
        for subj in subjects:
            group_mapping[subj] = group

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE LESION SH + HEME ANALYSIS")
    print(f"{'='*80}")
    print(f"Subjects: {len(all_subjects)} ({subjects_by_group})")
    print(f"Timepoints: 2d, 9d, 1mo, 5mo")
    print(f"Max SH degree: L={max_l}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timepoints to analyze
    timepoints = ['2d', '9d', '1mo', '5mo']

    results = []

    for subject_id in sorted(all_subjects):
        group = group_mapping[subject_id]
        print(f"\n[{subject_id}] Group: {group}")

        for timepoint in timepoints:
            print(f"  Processing {timepoint}...")

            # Analyze lesion (returns zeros if no lesion)
            lesion_result, success = analyze_single_lesion(
                subject_id, timepoint, data_dir, output_dir,
                max_l=max_l, num_samples=num_samples
            )

            if success and lesion_result is not None:
                lesion_result['group'] = group

                # Add heme content analysis
                print(f"    Computing heme content...")
                heme_metrics = compute_heme_content(subject_id, timepoint, data_dir)

                if heme_metrics is not None:
                    lesion_result.update(heme_metrics)
                    print(f"    ✓ Heme - Mean: {heme_metrics['heme_mean']:.2f}, "
                          f"Volume: {heme_metrics['heme_volume_mm3']:.2f} mm³")
                else:
                    # Add NaN for missing heme data
                    lesion_result.update({
                        'heme_mean': np.nan,
                        'heme_std': np.nan,
                        'heme_total': np.nan,
                        'heme_max': np.nan,
                        'heme_volume_mm3': np.nan,
                        'heme_95percentile': np.nan
                    })
                    print(f"    ✗ Heme data not available")

                results.append(lesion_result)
            else:
                print(f"  ✗ Failed to process")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("\n[ERROR] No results generated!")
        return results_df

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"Total records: {len(results_df)}")
    print(f"By group: {results_df.groupby('group').size().to_dict()}")
    print(f"By timepoint: {results_df.groupby('timepoint').size().to_dict()}")
    print(f"{'='*80}\n")

    return results_df


def main():
    """Main analysis pipeline."""
    # Paths
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    output_dir = Path(__file__).parent.parent.parent / "results" / "lesion_sh_heme_comprehensive"

    # Load config
    config = load_config()

    # Run comprehensive analysis
    results_df = analyze_all_lesions_and_heme(config, data_dir, output_dir, max_l=6, num_samples=2000)

    if len(results_df) == 0:
        print("[ERROR] No results to save!")
        return

    # Save combined results
    output_csv = output_dir / "lesion_sh_heme_comprehensive.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved: {output_csv}")
    print(f"  Shape: {results_df.shape}")
    print(f"  Columns: {len(results_df.columns)}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    print("Lesion Volume by Group and Timepoint:")
    print(results_df.pivot_table(values='original_volume_mm3',
                                  index='timepoint',
                                  columns='group',
                                  aggfunc='mean'))

    print("\n\nHeme Content by Group and Timepoint:")
    print(results_df.pivot_table(values='heme_mean',
                                  index='timepoint',
                                  columns='group',
                                  aggfunc='mean'))

    print(f"\n{'='*80}")
    print(f"ALL ANALYSIS COMPLETE!")
    print(f"Results: {output_csv}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
