#!/usr/bin/env python3
"""
Comprehensive TBI vs PTE Classification and Visualization Analysis

This script performs:
1. RF, SVM, ENR classification with 10-fold CV (class-weighted for imbalanced data)
2. Performance visualization (AUC, ROC, sensitivity, specificity)
3. Top predictor identification and visualization
4. Temporal trend analysis (line plots over time)
5. Within-subject centroid displacement analysis

Author: TBI/PTE Classification Pipeline
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_and_prepare_data(data_path):
    """Load tract geometry data and prepare for analysis."""
    df = pd.read_csv(data_path)

    print(f"\n{'='*80}")
    print(f"DATA LOADING AND PREPARATION")
    print(f"{'='*80}")
    print(f"Total records: {len(df)}")
    print(f"Groups: {df['group'].value_counts().to_dict()}")
    print(f"Timepoints: {df['timepoint'].value_counts().to_dict()}")

    # Filter to TBI and PTE only
    df_tbi_pte = df[df['group'].isin(['TBI', 'PTE'])].copy()

    print(f"\nAfter filtering to TBI and PTE:")
    print(f"Total records: {len(df_tbi_pte)}")
    print(f"TBI: {(df_tbi_pte['group'] == 'TBI').sum()}")
    print(f"PTE: {(df_tbi_pte['group'] == 'PTE').sum()}")

    # Convert group to binary (PTE=1, TBI=0)
    df_tbi_pte['group_binary'] = (df_tbi_pte['group'] == 'PTE').astype(int)

    return df_tbi_pte


def get_feature_columns(df):
    """Get feature columns (exclude centroid and metadata)."""
    exclude_cols = ['centroid_x_mean', 'centroid_y_mean', 'centroid_z_mean',
                    'subject_id', 'timepoint', 'tract', 'group', 'group_binary']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")

    return feature_cols


def aggregate_features_per_subject(df, timepoint, feature_cols):
    """
    Aggregate features across tracts for each subject.
    Uses mean aggregation to get subject-level features.
    """
    df_tp = df[df['timepoint'] == timepoint].copy()

    # Group by subject and compute mean across all tracts
    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['group_binary'] = 'first'  # Group is same for all tracts of a subject

    df_subj = df_tp.groupby('subject_id').agg(agg_dict).reset_index()

    return df_subj


def train_models_with_cv(X, y, random_state=42):
    """
    Train RF, SVM, and ENR with 10-fold CV and class weights.

    Returns results dictionary with predictions and performance.
    """
    # Compute class weights (give more weight to minority class PTE)
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    print(f"\n  Class weights: TBI={class_weight_dict[0]:.2f}, PTE={class_weight_dict[1]:.2f}")

    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight=class_weight_dict,
            random_state=random_state,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            class_weight=class_weight_dict,
            probability=True,
            random_state=random_state
        ),
        'Elastic Net': ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=random_state,
            max_iter=10000
        )
    }

    # 10-fold stratified CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    results = {}

    # Handle missing values by imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    for model_name, model in models.items():
        print(f"  Training {model_name}...")

        # For Elastic Net, we need to threshold predictions since it's a regressor
        if model_name == 'Elastic Net':
            # Cross-validation predictions
            y_pred_cont = cross_val_predict(model, X_scaled, y, cv=cv, n_jobs=-1)
            y_pred = (y_pred_cont > 0.5).astype(int)
            y_pred_proba = np.column_stack([1 - y_pred_cont, y_pred_cont])
        else:
            # Get predictions and probabilities
            y_pred = cross_val_predict(model, X_scaled, y, cv=cv, n_jobs=-1)
            y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv,
                                            method='predict_proba', n_jobs=-1)

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC
        try:
            if model_name == 'Elastic Net':
                auc = roc_auc_score(y, y_pred_cont)
            else:
                auc = roc_auc_score(y, y_pred_proba[:, 1])
        except:
            auc = 0.5

        # ROC curve
        if model_name == 'Elastic Net':
            fpr, tpr, _ = roc_curve(y, y_pred_cont)
        else:
            fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])

        # Feature importance (for RF)
        feature_importance = None
        if model_name == 'Random Forest':
            # Fit on full data to get feature importance
            model.fit(X_scaled, y)
            feature_importance = model.feature_importances_

        results[model_name] = {
            'y_true': y,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba[:, 1] if model_name != 'Elastic Net' else y_pred_cont,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }

        print(f"    Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, "
              f"Sens: {sensitivity:.3f}, Spec: {specificity:.3f}")

    return results, scaler


def plot_classification_results(results, timepoint, output_dir):
    """
    Create comprehensive classification performance plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'TBI vs PTE Classification Results - {timepoint}',
                 fontsize=16, fontweight='bold')

    models = list(results.keys())

    # 1. ROC Curves
    ax = axes[0, 0]
    for model_name in models:
        res = results[model_name]
        ax.plot(res['fpr'], res['tpr'], label=f"{model_name} (AUC={res['auc']:.3f})",
               linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. AUC Comparison
    ax = axes[0, 1]
    aucs = [results[m]['auc'] for m in models]
    bars = ax.bar(models, aucs, color=['steelblue', 'coral', 'mediumseagreen'],
                  edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('AUC', fontweight='bold')
    ax.set_title('AUC Comparison', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='red', linestyle='--', label='Chance', linewidth=1)
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Sensitivity and Specificity
    ax = axes[0, 2]
    x = np.arange(len(models))
    width = 0.35
    sens = [results[m]['sensitivity'] for m in models]
    spec = [results[m]['specificity'] for m in models]
    ax.bar(x - width/2, sens, width, label='Sensitivity', color='steelblue',
          edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.bar(x + width/2, spec, width, label='Specificity', color='coral',
          edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Sensitivity and Specificity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4-6. Confusion Matrices
    for idx, model_name in enumerate(models):
        ax = axes[1, idx]
        cm = results[model_name]['confusion_matrix']

        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=['TBI', 'PTE'], yticklabels=['TBI', 'PTE'],
                   ax=ax, cbar_kws={'label': 'Proportion'})
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f'classification_results_{timepoint}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_top_predictors(feature_importance, feature_names, timepoint, output_dir, top_n=10):
    """Plot top predictors from Random Forest."""
    # Get top features
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = feature_importance[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(top_features)), top_importances,
                   color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance', fontweight='bold', fontsize=12)
    ax.set_title(f'Top {top_n} Predictors - {timepoint}',
                fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    # Add values on bars
    for bar, imp in zip(bars, top_importances):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{imp:.4f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f'top_predictors_{timepoint}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

    return top_features


def plot_top_predictor_boxplots(df, timepoint, top_features, output_dir):
    """Create boxplots with jitters for top predictors."""
    df_tp = df[df['timepoint'] == timepoint].copy()

    # Aggregate by subject
    agg_dict = {feat: 'mean' for feat in top_features}
    agg_dict['group'] = 'first'
    df_subj = df_tp.groupby('subject_id').agg(agg_dict).reset_index()

    n_features = len(top_features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        # Boxplot
        bp = sns.boxplot(data=df_subj, x='group', y=feature, ax=ax,
                        palette={'TBI': 'coral', 'PTE': 'steelblue'},
                        width=0.5)

        # Add jitter
        sns.stripplot(data=df_subj, x='group', y=feature, ax=ax,
                     color='black', alpha=0.4, size=4, jitter=True)

        ax.set_xlabel('Group', fontweight='bold', fontsize=11)
        ax.set_ylabel(feature, fontweight='bold', fontsize=11)
        ax.set_title(f'{feature} - {timepoint}', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3, axis='y')

        # Add sample sizes
        tbi_n = (df_subj['group'] == 'TBI').sum()
        pte_n = (df_subj['group'] == 'PTE').sum()
        ax.text(0.02, 0.98, f'TBI n={tbi_n}\nPTE n={pte_n}',
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
               fontsize=9)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Top Predictors: TBI vs PTE - {timepoint}',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = output_dir / f'top_predictors_boxplots_{timepoint}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def plot_temporal_trends(df, feature_cols, output_dir):
    """Create line plots of geometric metrics over time."""
    timepoints = ['2d', '9d', '1mo', '5mo']

    # Aggregate by subject and timepoint
    temporal_data = []

    for tp in timepoints:
        df_tp = df[df['timepoint'] == tp]

        agg_dict = {feat: 'mean' for feat in feature_cols}
        agg_dict['group'] = 'first'

        df_subj = df_tp.groupby('subject_id').agg(agg_dict).reset_index()
        df_subj['timepoint'] = tp

        temporal_data.append(df_subj)

    df_temporal = pd.concat(temporal_data, ignore_index=True)

    # Select key metrics to plot
    key_metrics = ['length_mean', 'tortuosity_mean', 'curv_mean_avg',
                   'elongation_ratio_mean', 'planarity_ratio_mean']

    key_metrics = [m for m in key_metrics if m in feature_cols]

    n_cols = 3
    n_rows = int(np.ceil(len(key_metrics) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if len(key_metrics) > 1 else [axes]

    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]

        # Group by timepoint and group, compute mean and SEM
        grouped = df_temporal.groupby(['timepoint', 'group'])[metric].agg(['mean', 'sem']).reset_index()

        for group in ['TBI', 'PTE']:
            group_data = grouped[grouped['group'] == group]

            # Map timepoints to numeric for plotting
            tp_map = {'2d': 0, '9d': 1, '1mo': 2, '5mo': 3}
            x = [tp_map[tp] for tp in group_data['timepoint']]

            color = 'coral' if group == 'TBI' else 'steelblue'

            ax.plot(x, group_data['mean'], marker='o', linewidth=2.5,
                   markersize=8, label=group, color=color)
            ax.fill_between(x,
                           group_data['mean'] - group_data['sem'],
                           group_data['mean'] + group_data['sem'],
                           alpha=0.2, color=color)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(timepoints)
        ax.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
        ax.set_ylabel(metric, fontweight='bold', fontsize=11)
        ax.set_title(f'{metric} Over Time', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(len(key_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Temporal Trends: TBI vs PTE',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = output_dir / 'temporal_trends_tbi_vs_pte.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def analyze_centroid_displacement(df, output_dir):
    """
    Within-subject analysis of centroid displacement over time.
    """
    print(f"\n{'='*80}")
    print(f"CENTROID DISPLACEMENT ANALYSIS")
    print(f"{'='*80}")

    centroid_cols = ['centroid_x_mean', 'centroid_y_mean', 'centroid_z_mean']
    timepoints = ['2d', '9d', '1mo', '5mo']

    # Aggregate centroids by subject and timepoint
    temporal_centroids = []

    for tp in timepoints:
        df_tp = df[df['timepoint'] == tp]

        agg_dict = {col: 'mean' for col in centroid_cols}
        agg_dict['group'] = 'first'

        df_subj_tract = df_tp.groupby(['subject_id', 'tract'], as_index=False).agg(agg_dict)
        df_subj_tract['timepoint'] = tp

        temporal_centroids.append(df_subj_tract)

    df_centroids = pd.concat(temporal_centroids, ignore_index=True)

    # Calculate displacement from 2d baseline for each subject-tract combination
    displacement_data = []

    for subject in df_centroids['subject_id'].unique():
        for tract in df_centroids['tract'].unique():
            subj_tract_data = df_centroids[
                (df_centroids['subject_id'] == subject) &
                (df_centroids['tract'] == tract)
            ].sort_values('timepoint')

            if len(subj_tract_data) < 2:
                continue

            # Get baseline (2d) centroid
            baseline = subj_tract_data[subj_tract_data['timepoint'] == '2d']

            if len(baseline) == 0:
                continue

            baseline_x = baseline['centroid_x_mean'].values[0]
            baseline_y = baseline['centroid_y_mean'].values[0]
            baseline_z = baseline['centroid_z_mean'].values[0]

            group = baseline['group'].values[0]

            # Calculate displacement at each timepoint
            for _, row in subj_tract_data.iterrows():
                dx = row['centroid_x_mean'] - baseline_x
                dy = row['centroid_y_mean'] - baseline_y
                dz = row['centroid_z_mean'] - baseline_z

                # Euclidean displacement
                displacement = np.sqrt(dx**2 + dy**2 + dz**2)

                displacement_data.append({
                    'subject_id': subject,
                    'tract': tract,
                    'timepoint': row['timepoint'],
                    'group': group,
                    'displacement_mm': displacement,
                    'dx': dx,
                    'dy': dy,
                    'dz': dz
                })

    df_displacement = pd.DataFrame(displacement_data)

    # Save displacement data
    disp_path = output_dir / 'centroid_displacement_data.csv'
    df_displacement.to_csv(disp_path, index=False)
    print(f"\n✓ Saved displacement data: {disp_path}")

    # Plot 1: Overall displacement over time
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    grouped = df_displacement.groupby(['timepoint', 'group'])['displacement_mm'].agg(['mean', 'sem']).reset_index()

    tp_map = {'2d': 0, '9d': 1, '1mo': 2, '5mo': 3}

    for group in ['TBI', 'PTE']:
        group_data = grouped[grouped['group'] == group]
        x = [tp_map[tp] for tp in group_data['timepoint']]
        color = 'coral' if group == 'TBI' else 'steelblue'

        ax.plot(x, group_data['mean'], marker='o', linewidth=2.5,
               markersize=8, label=group, color=color)
        ax.fill_between(x,
                       group_data['mean'] - group_data['sem'],
                       group_data['mean'] + group_data['sem'],
                       alpha=0.2, color=color)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(timepoints)
    ax.set_xlabel('Timepoint', fontweight='bold', fontsize=12)
    ax.set_ylabel('Displacement from 2d Baseline (mm)', fontweight='bold', fontsize=12)
    ax.set_title('Mean Centroid Displacement Over Time', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Directional displacement (x, y, z)
    ax = axes[1]

    # Focus on final timepoint (5mo)
    df_5mo = df_displacement[df_displacement['timepoint'] == '5mo']

    directions = ['dx', 'dy', 'dz']
    dir_labels = ['X (L-R)', 'Y (P-A)', 'Z (I-S)']

    x_pos = np.arange(len(directions))
    width = 0.35

    for idx, group in enumerate(['TBI', 'PTE']):
        group_data = df_5mo[df_5mo['group'] == group]
        means = [group_data[d].mean() for d in directions]
        sems = [group_data[d].sem() for d in directions]

        color = 'coral' if group == 'TBI' else 'steelblue'
        offset = width/2 if group == 'PTE' else -width/2

        ax.bar(x_pos + offset, means, width, yerr=sems,
              label=group, color=color, edgecolor='black',
              linewidth=1.5, alpha=0.7, capsize=5)

    ax.set_ylabel('Displacement from Baseline (mm)', fontweight='bold', fontsize=12)
    ax.set_title('Directional Displacement at 5mo', fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dir_labels)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('Within-Subject Centroid Displacement Analysis',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = output_dir / 'centroid_displacement_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path}")

    # Summary statistics
    print(f"\nDisplacement Summary (from 2d baseline):")
    print("-" * 80)

    for tp in ['9d', '1mo', '5mo']:
        print(f"\n{tp}:")
        tp_data = df_displacement[df_displacement['timepoint'] == tp]

        for group in ['TBI', 'PTE']:
            group_data = tp_data[tp_data['group'] == group]
            mean_disp = group_data['displacement_mm'].mean()
            std_disp = group_data['displacement_mm'].std()
            print(f"  {group}: {mean_disp:.2f} ± {std_disp:.2f} mm")


def main():
    """Main analysis pipeline."""
    # Paths
    data_path = Path(__file__).parent / "lesion_vae_analysis" / "results" / \
                "comprehensive_tract_geometry" / "comprehensive_tract_geometry_metrics.csv"
    output_dir = Path(__file__).parent / "lesion_vae_analysis" / "results" / \
                 "tbi_pte_classification"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TBI vs PTE CLASSIFICATION AND VISUALIZATION ANALYSIS")
    print(f"{'='*80}")

    # Load data
    df = load_and_prepare_data(data_path)

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Timepoints to analyze
    timepoints = ['2d', '9d', '1mo', '5mo']

    all_results = {}
    all_top_features = {}

    # Process each timepoint
    for timepoint in timepoints:
        print(f"\n{'='*80}")
        print(f"TIMEPOINT: {timepoint}")
        print(f"{'='*80}")

        # Aggregate features per subject
        df_subj = aggregate_features_per_subject(df, timepoint, feature_cols)

        print(f"\nSubjects: {len(df_subj)} (TBI: {(df_subj['group_binary']==0).sum()}, "
              f"PTE: {(df_subj['group_binary']==1).sum()})")

        # Prepare X and y
        X = df_subj[feature_cols].values
        y = df_subj['group_binary'].values

        # Train models with CV
        results, scaler = train_models_with_cv(X, y)

        all_results[timepoint] = results

        # Plot classification results
        plot_classification_results(results, timepoint, output_dir)

        # Plot top predictors (from Random Forest)
        if results['Random Forest']['feature_importance'] is not None:
            top_features = plot_top_predictors(
                results['Random Forest']['feature_importance'],
                feature_cols, timepoint, output_dir
            )
            all_top_features[timepoint] = top_features

            # Plot boxplots for top predictors
            plot_top_predictor_boxplots(df, timepoint, top_features, output_dir)

    # Temporal trend analysis
    print(f"\n{'='*80}")
    print(f"TEMPORAL TREND ANALYSIS")
    print(f"{'='*80}")
    plot_temporal_trends(df, feature_cols, output_dir)

    # Centroid displacement analysis
    analyze_centroid_displacement(df, output_dir)

    # Save summary results
    summary_rows = []
    for tp in timepoints:
        for model_name in ['Random Forest', 'SVM', 'Elastic Net']:
            res = all_results[tp][model_name]
            summary_rows.append({
                'timepoint': tp,
                'model': model_name,
                'accuracy': res['accuracy'],
                'auc': res['auc'],
                'sensitivity': res['sensitivity'],
                'specificity': res['specificity']
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'classification_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults directory: {output_dir}")
    print(f"✓ Classification summary: {summary_path}")
    print(f"✓ Generated {len(timepoints) * 3} classification figures")
    print(f"✓ Generated temporal trends and displacement analyses")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
