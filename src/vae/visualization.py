"""
Visualization functions for lesion-conditioned VAE analysis

Note: This module contains core visualization functions.
Additional specialized visualizations can be added as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

GROUP_COLORS = {"Sham": "#2E8B57", "TBI": "#4169E1", "PTE": "#DC143C"}


def plot_lesion_aware_deviation_profiles(bundle_profiles, lesion_profiles,
                                        group_mappings, tp, save_path):
    """Plot z-score deviation profiles with lesion context"""

    tp_profiles = bundle_profiles[tp]
    tp_lesions = lesion_profiles[tp]
    tp_groups = group_mappings[tp]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[2.5, 1.5, 1])

    n_segments = len(next(iter(tp_profiles.values())))
    segment_positions = np.linspace(0, 1, n_segments)
    segment_width = 1.0 / n_segments

    # ===== PANEL 1: Z-score profiles with lesion overlay =====
    ax1 = axes[0]

    # Calculate mean lesion burden per segment for each group
    tbi_lesion_profile = []
    pte_lesion_profile = []

    for seg_idx in range(n_segments):
        tbi_lesions_seg = [tp_lesions[sid][seg_idx] for sid, g in tp_groups.items() if g == 'TBI']
        pte_lesions_seg = [tp_lesions[sid][seg_idx] for sid, g in tp_groups.items() if g == 'PTE']

        tbi_lesion_profile.append(np.mean(tbi_lesions_seg) if tbi_lesions_seg else 0)
        pte_lesion_profile.append(np.mean(pte_lesions_seg) if pte_lesions_seg else 0)

    tbi_lesion_profile = np.array(tbi_lesion_profile)
    pte_lesion_profile = np.array(pte_lesion_profile)

    # Highlight lesion regions
    lesion_threshold = 0.01
    for seg_idx in range(n_segments):
        seg_center = segment_positions[seg_idx]

        if tbi_lesion_profile[seg_idx] > lesion_threshold:
            alpha_tbi = min(0.3, tbi_lesion_profile[seg_idx] * 10)
            ax1.axvspan(seg_center - segment_width/2, seg_center + segment_width/2,
                       alpha=alpha_tbi, color='#4169E1', zorder=0)

        if pte_lesion_profile[seg_idx] > lesion_threshold:
            alpha_pte = min(0.3, pte_lesion_profile[seg_idx] * 10)
            ax1.axvspan(seg_center - segment_width/2, seg_center + segment_width/2,
                       alpha=alpha_pte, color='#DC143C', zorder=0)

    # Plot z-score profiles
    for group, color in GROUP_COLORS.items():
        group_profiles = [tp_profiles[sid] for sid, g in tp_groups.items() if g == group]

        if group_profiles:
            group_array = np.array(group_profiles)
            mean_profile = group_array.mean(axis=0)
            sem_profile = group_array.std(axis=0) / np.sqrt(len(group_array))

            ax1.fill_between(segment_positions, mean_profile - sem_profile,
                           mean_profile + sem_profile, alpha=0.25, color=color, zorder=2)
            linewidth = 3 if group in ['TBI', 'PTE'] else 2
            linestyle = '--' if group == 'Sham' else '-'
            ax1.plot(segment_positions, mean_profile, color=color, linewidth=linewidth,
                   linestyle=linestyle, label=f'{group} (n={len(group_profiles)})', zorder=3)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5,
               label='Normative (Sham)', zorder=1)

    ax1.set_ylabel('Mean |Z-Score| Deviation', fontsize=12)
    ax1.set_title(f'Lesion-Aware Z-Score Deviation Profiles @ {tp}\n' +
                 'Shaded regions indicate lesion location (blue=TBI, red=PTE)',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3, zorder=1)
    ax1.set_xlim(0, 1)

    lesion_legend_elements = [
        Patch(facecolor='#4169E1', alpha=0.2, label='TBI lesion location'),
        Patch(facecolor='#DC143C', alpha=0.2, label='PTE lesion location')
    ]
    ax1_legend2 = ax1.legend(handles=lesion_legend_elements, loc='upper right',
                             fontsize=10, framealpha=0.9)
    ax1.add_artist(ax1.get_legend())

    # ===== PANEL 2: TBI vs PTE difference =====
    ax2 = axes[1]

    tbi_profiles = [tp_profiles[sid] for sid, g in tp_groups.items() if g == 'TBI']
    pte_profiles = [tp_profiles[sid] for sid, g in tp_groups.items() if g == 'PTE']

    if tbi_profiles and pte_profiles:
        tbi_mean = np.array(tbi_profiles).mean(axis=0)
        pte_mean = np.array(pte_profiles).mean(axis=0)
        difference = pte_mean - tbi_mean

        positive_mask = difference > 0
        negative_mask = difference < 0

        ax2.fill_between(segment_positions, 0, difference, where=positive_mask,
                        color='red', alpha=0.6, label='PTE > TBI', zorder=2)
        ax2.fill_between(segment_positions, 0, difference, where=negative_mask,
                        color='blue', alpha=0.6, label='TBI > PTE', zorder=2)
        ax2.plot(segment_positions, difference, color='black', linewidth=2.5, zorder=3)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=1)

    ax2.set_ylabel('Z-Score Difference\n(PTE - TBI)', fontsize=11)
    ax2.set_title('Seizure-Specific Deviations (Z-Scores)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, zorder=1)
    ax2.set_xlim(0, 1)

    # ===== PANEL 3: Effect sizes =====
    ax3 = axes[2]

    if tbi_profiles and pte_profiles:
        from scipy import stats
        effect_sizes = []
        p_values = []

        for seg_idx in range(n_segments):
            tbi_vals = [p[seg_idx] for p in tbi_profiles]
            pte_vals = [p[seg_idx] for p in pte_profiles]

            pooled_std = np.sqrt(((len(tbi_vals)-1)*np.var(tbi_vals, ddof=1) +
                                 (len(pte_vals)-1)*np.var(pte_vals, ddof=1)) /
                                (len(tbi_vals) + len(pte_vals) - 2))
            if pooled_std > 0:
                cohens_d = (np.mean(pte_vals) - np.mean(tbi_vals)) / pooled_std
            else:
                cohens_d = 0
            effect_sizes.append(cohens_d)

            if len(tbi_vals) > 1 and len(pte_vals) > 1:
                _, p_val = stats.ttest_ind(tbi_vals, pte_vals, equal_var=False)
                p_values.append(p_val)
            else:
                p_values.append(1.0)

        effect_sizes = np.array(effect_sizes)
        p_values = np.array(p_values)

        ax3.plot(segment_positions, effect_sizes, color='purple', linewidth=2.5, zorder=2)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=1)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='medium', zorder=1)
        ax3.axhline(y=0.8, color='darkred', linestyle='--', alpha=0.5, linewidth=1.5, label='large', zorder=1)
        ax3.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
        ax3.axhline(y=-0.8, color='darkred', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)

        sig_mask = p_values < 0.05
        if sig_mask.any():
            ax3.scatter(segment_positions[sig_mask], effect_sizes[sig_mask],
                       s=100, color='gold', marker='*', edgecolors='black',
                       linewidth=1, zorder=4, label='p<0.05')

    ax3.set_ylabel("Cohen's d", fontsize=11)
    ax3.set_xlabel('Position Along Tract', fontsize=11)
    ax3.set_title("Effect Size with Significance Markers", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best', ncol=2)
    ax3.grid(True, alpha=0.3, zorder=1)
    ax3.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved z-score deviation profile: {save_path}")


def plot_lesion_impact_analysis(bundle_profiles, lesion_profiles, group_mappings,
                                tp, save_path):
    """Analyze how lesion burden correlates with z-score deviations"""

    tp_profiles = bundle_profiles[tp]
    tp_lesions = lesion_profiles[tp]
    tp_groups = group_mappings[tp]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ===== PANEL 1: Lesion burden vs mean z-score deviation =====
    ax1 = axes[0, 0]

    for group, color in GROUP_COLORS.items():
        if group == 'Sham':
            continue

        group_lesions = []
        group_deviations = []

        for sid, g in tp_groups.items():
            if g == group:
                mean_lesion = np.mean(tp_lesions[sid])
                mean_deviation = np.mean(tp_profiles[sid])
                group_lesions.append(mean_lesion * 100)
                group_deviations.append(mean_deviation)

        if group_lesions:
            ax1.scatter(group_lesions, group_deviations, color=color, s=100,
                       alpha=0.7, label=group, edgecolors='black', linewidth=1.5)

    ax1.set_xlabel('Mean Lesion Burden (%)', fontsize=11)
    ax1.set_ylabel('Mean |Z-Score| Deviation', fontsize=11)
    ax1.set_title('Lesion Burden vs Z-Score Deviation', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== PANEL 2: Distribution by lesion presence =====
    ax2 = axes[0, 1]

    for group, color in [('TBI', GROUP_COLORS['TBI']), ('PTE', GROUP_COLORS['PTE'])]:
        lesioned_devs = []
        nonlesioned_devs = []

        for sid, g in tp_groups.items():
            if g == group:
                profile = tp_profiles[sid]
                lesion = tp_lesions[sid]

                lesion_mask = lesion > 0.01
                if lesion_mask.any():
                    lesioned_devs.extend(profile[lesion_mask])

                nonlesion_mask = lesion <= 0.01
                if nonlesion_mask.any():
                    nonlesioned_devs.extend(profile[nonlesion_mask])

        if lesioned_devs and nonlesioned_devs:
            bp = ax2.boxplot([lesioned_devs, nonlesioned_devs],
                           positions=[0 if group=='TBI' else 2, 0.5 if group=='TBI' else 2.5],
                           widths=0.4, patch_artist=True, showmeans=True)

            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

    ax2.set_xticks([0.25, 2.25])
    ax2.set_xticklabels(['TBI', 'PTE'])
    ax2.set_ylabel('|Z-Score| Deviation', fontsize=11)
    ax2.set_title('Z-Score Deviations: Lesioned vs Non-Lesioned Segments',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== PANEL 3 & 4: Placeholder for additional analyses =====
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, 'Additional analysis panel', ha='center', va='center')
    ax3.set_title('Reserved for future analysis', fontsize=12)

    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'Additional analysis panel', ha='center', va='center')
    ax4.set_title('Reserved for future analysis', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved lesion impact analysis: {save_path}")


def plot_latent_space_with_lesion_context(latents, lesion_burden, groups, tp, save_path):
    """t-SNE of latent space colored by group and lesion burden"""
    from sklearn.manifold import TSNE

    Z = latents[tp]
    lesion = lesion_burden[tp]
    group_labels = groups[tp]

    unique_subjects = []
    subject_latents = []
    subject_lesions = []
    subject_groups = []

    for i, subj_id in enumerate(np.unique(group_labels)):
        mask = group_labels == subj_id
        if mask.sum() > 0:
            unique_subjects.append(subj_id)
            subject_latents.append(Z[mask].mean(axis=0))
            subject_lesions.append(lesion[mask].mean())
            subject_groups.append(group_labels[mask][0])

    subject_latents = np.array(subject_latents)
    subject_lesions = np.array(subject_lesions)

    if len(subject_latents) < 5:
        print(f"[WARN] Too few subjects for t-SNE at {tp}")
        return

    perplexity = min(5, len(subject_latents) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    Z_tsne = tsne.fit_transform(subject_latents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Colored by group
    for group, color in GROUP_COLORS.items():
        mask = np.array(subject_groups) == group
        if mask.any():
            ax1.scatter(Z_tsne[mask, 0], Z_tsne[mask, 1], c=color,
                       s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                       label=f'{group} (n={mask.sum()})')

    ax1.set_xlabel('t-SNE 1', fontsize=11)
    ax1.set_ylabel('t-SNE 2', fontsize=11)
    ax1.set_title(f'Latent Space @ {tp} (by Group)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Colored by lesion burden
    scatter = ax2.scatter(Z_tsne[:, 0], Z_tsne[:, 1],
                         c=subject_lesions * 100, cmap='YlOrRd',
                         s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

    plt.colorbar(scatter, ax=ax2, label='Lesion Burden (%)')
    ax2.set_xlabel('t-SNE 1', fontsize=11)
    ax2.set_ylabel('t-SNE 2', fontsize=11)
    ax2.set_title(f'Latent Space @ {tp} (by Lesion Burden)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved latent space plot: {save_path}")
