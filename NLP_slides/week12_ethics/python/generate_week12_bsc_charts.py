"""
Generate Week 12 BSc-level charts for Ethics & Fairness presentation
Following the Educational Presentation Framework with discovery-based pedagogy
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational color scheme
COLOR_HARM = '#FF6B6B'        # Red for harms/problems
COLOR_SOLUTION = '#95E77E'    # Green for solutions
COLOR_WARNING = '#FFE66D'     # Yellow for warnings
COLOR_INFO = '#4ECDC4'        # Teal for information
COLOR_NEUTRAL = '#E0E0E0'     # Gray for neutral

print("Generating Week 12 BSc-level charts (Educational Framework)...")
print("=" * 60)

# Chart 1: Hiring Bias Timeline (Visual Hook)
def plot_hiring_bias_timeline():
    fig, ax = plt.subplots(figsize=(14, 8))

    # Timeline events
    events = [
        (2014, "AI Training\nStarts", "10 years of\nresumes\n(mostly male)", COLOR_INFO),
        (2015, "Bias\nEmerges", "Penalizes\n'women's'\nkeyword", COLOR_WARNING),
        (2016, "Pattern\nDetected", "-5 stars for\nfemale colleges", COLOR_HARM),
        (2017, "Attempted\nFix", "Manual\nadjustments\nfail", COLOR_WARNING),
        (2018, "System\nScrapped", "Never\ndeployed\nin production", COLOR_SOLUTION)
    ]

    years = [e[0] for e in events]

    # Draw timeline
    ax.plot([2013.5, 2018.5], [0, 0], 'k-', linewidth=3, zorder=1)

    # Plot events
    for i, (year, title, desc, color) in enumerate(events):
        # Event marker
        ax.scatter(year, 0, s=500, c=color, edgecolors='black', linewidth=2, zorder=3)

        # Alternating up/down for readability
        y_offset = 0.3 if i % 2 == 0 else -0.3
        text_y = 0.5 if i % 2 == 0 else -0.5

        # Connect to timeline
        ax.plot([year, year], [0, y_offset], 'k--', linewidth=1, alpha=0.5, zorder=2)

        # Event box
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.text(year, text_y, f"{title}\n{desc}",
                fontsize=11, ha='center', va='center', bbox=bbox_props, weight='bold')

    # Add impact annotation
    ax.text(2016, -1.2, "Impact: Thousands of women's resumes potentially downranked",
            fontsize=13, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.7', facecolor=COLOR_HARM, alpha=0.3))

    # Add discovery insight
    ax.text(2016, 1.2, "Discovery: AI doesn't eliminate bias, it automates it at scale",
            fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor=COLOR_WARNING, alpha=0.3))

    ax.set_xlim(2013.5, 2018.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Year', fontsize=14, weight='bold')
    ax.set_title("Amazon's Hiring AI: A Case Study in Bias Amplification",
                 fontsize=16, weight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/hiring_bias_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("1/8: hiring_bias_timeline.pdf created")

# Chart 2: Bias Sources Flowchart
def plot_bias_sources_flowchart():
    fig, ax = plt.subplots(figsize=(14, 10))

    stages = [
        ("Data\nCollection", ["Sampling bias", "Historical bias", "Label bias"], 0.2),
        ("Model\nTraining", ["Architecture bias", "Optimization bias", "Proxy features"], 0.5),
        ("Deployment", ["Feedback loops", "User interaction", "Context shift"], 0.8)
    ]

    colors = [COLOR_HARM, COLOR_WARNING, COLOR_INFO]

    for i, (stage, biases, x_pos) in enumerate(stages):
        # Stage box
        bbox = FancyBboxPatch((x_pos - 0.1, 0.7), 0.2, 0.15,
                              boxstyle="round,pad=0.02",
                              facecolor=colors[i], edgecolor='black', linewidth=3)
        ax.add_patch(bbox)
        ax.text(x_pos, 0.775, stage, fontsize=13, ha='center', va='center', weight='bold')

        # Bias items
        for j, bias in enumerate(biases):
            y_pos = 0.5 - j * 0.15
            # Bias box
            bias_box = FancyBboxPatch((x_pos - 0.08, y_pos - 0.05), 0.16, 0.08,
                                      boxstyle="round,pad=0.01",
                                      facecolor=colors[i], alpha=0.3,
                                      edgecolor='black', linewidth=1.5)
            ax.add_patch(bias_box)
            ax.text(x_pos, y_pos, bias, fontsize=10, ha='center', va='center')

        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((x_pos + 0.1, 0.775), (stages[i+1][2] - 0.1, 0.775),
                                   arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
            ax.add_patch(arrow)
            ax.text((x_pos + stages[i+1][2]) / 2, 0.82, "propagates",
                   fontsize=11, ha='center', style='italic')

    # Amplification annotation
    ax.annotate('Bias Amplification:\nEach stage compounds previous biases',
                xy=(0.5, 0.1), fontsize=13, ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_HARM, alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Bias Sources: Where Unfairness Enters the ML Pipeline",
                 fontsize=16, weight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/bias_sources_flowchart.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("2/8: bias_sources_flowchart.pdf created")

# Chart 3: Harm Taxonomy Tree
def plot_harm_taxonomy_tree():
    fig, ax = plt.subplots(figsize=(14, 10))

    # Root node
    root_box = FancyBboxPatch((0.35, 0.85), 0.3, 0.1, boxstyle="round,pad=0.02",
                              facecolor=COLOR_HARM, edgecolor='black', linewidth=3)
    ax.add_patch(root_box)
    ax.text(0.5, 0.9, "AI Harms", fontsize=15, ha='center', va='center', weight='bold')

    # Four types of harms
    harms = [
        ("Allocative", ["Loan denied", "Resume rejected", "Treatment withheld"], 0.1),
        ("Representational", ["Stereotypes", "Erasure", "Denigration"], 0.35),
        ("Quality-of-Service", ["Lower accuracy", "Higher errors", "Worse UX"], 0.6),
        ("Social", ["Trust erosion", "Discrimination normalized", "Inequality"], 0.85)
    ]

    colors_harm = [COLOR_HARM, COLOR_WARNING, COLOR_INFO, '#9B59B6']

    for i, (harm_type, examples, x_pos) in enumerate(harms):
        # Connect to root
        ax.plot([0.5, x_pos + 0.075], [0.85, 0.7], 'k-', linewidth=2)

        # Harm type box
        type_box = FancyBboxPatch((x_pos, 0.65), 0.15, 0.08, boxstyle="round,pad=0.01",
                                  facecolor=colors_harm[i], edgecolor='black', linewidth=2)
        ax.add_patch(type_box)
        ax.text(x_pos + 0.075, 0.69, harm_type, fontsize=11, ha='center', va='center', weight='bold')

        # Examples
        for j, example in enumerate(examples):
            y_pos = 0.5 - j * 0.12
            # Connect to harm type
            if j == 0:
                ax.plot([x_pos + 0.075, x_pos + 0.075], [0.65, y_pos + 0.04], 'k--', linewidth=1, alpha=0.5)

            # Example box
            ex_box = FancyBboxPatch((x_pos, y_pos), 0.15, 0.06, boxstyle="round,pad=0.005",
                                    facecolor=colors_harm[i], alpha=0.3, edgecolor='black', linewidth=1)
            ax.add_patch(ex_box)
            ax.text(x_pos + 0.075, y_pos + 0.03, example, fontsize=9, ha='center', va='center')

    # Key insight
    ax.text(0.5, 0.05, "Key Insight: Same AI system can cause multiple harm types simultaneously",
            fontsize=13, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor=COLOR_WARNING, alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Taxonomy of AI Harms: Four Categories with Real Examples",
                 fontsize=16, weight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/harm_taxonomy_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("3/8: harm_taxonomy_tree.pdf created")

# Chart 4: Stakeholder Map
def plot_stakeholder_map():
    fig, ax = plt.subplots(figsize=(12, 12))

    # Central AI system
    center = Circle((0.5, 0.5), 0.08, facecolor=COLOR_INFO, edgecolor='black', linewidth=3)
    ax.add_patch(center)
    ax.text(0.5, 0.5, "AI\nSystem", fontsize=12, ha='center', va='center', weight='bold')

    # Stakeholders
    stakeholders = [
        ("Developers", ["Design choices", "Bias auditing", "Documentation"], 0.5, 0.85, COLOR_SOLUTION),
        ("Users", ["Input data", "Interpretation", "Feedback"], 0.15, 0.5, COLOR_INFO),
        ("Affected\nCommunities", ["Experience harm", "Provide context", "Demand accountability"], 0.5, 0.15, COLOR_HARM),
        ("Regulators", ["Set standards", "Audit compliance", "Enforce penalties"], 0.85, 0.5, COLOR_WARNING)
    ]

    for name, responsibilities, x, y, color in stakeholders:
        # Connect to center
        ax.plot([0.5, x], [0.5, y], 'k-', linewidth=2, alpha=0.5)

        # Stakeholder circle
        stakeholder_circle = Circle((x, y), 0.12, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(stakeholder_circle)
        ax.text(x, y, name, fontsize=11, ha='center', va='center', weight='bold')

        # Responsibilities (offset from circle)
        offset_x = x + (0.22 if x > 0.5 else -0.22 if x < 0.5 else 0)
        offset_y = y + (0.15 if y > 0.5 else -0.15 if y < 0.5 else 0)

        resp_text = '\n'.join(responsibilities)
        ax.text(offset_x, offset_y, resp_text, fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Stakeholder Responsibilities in AI Ethics", fontsize=16, weight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/stakeholder_map.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("4/8: stakeholder_map.pdf created")

# Chart 5: Fairness Metrics Comparison
def plot_fairness_metrics_comparison():
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = ['Statistical\nParity', 'Equalized\nOdds', 'Calibration']

    # Three groups for comparison
    groups = ['Group A', 'Group B']
    colors = [COLOR_INFO, COLOR_WARNING]

    # Sample data showing metric differences
    statistical_parity = [0.6, 0.4]  # Different acceptance rates
    equalized_odds_tpr = [0.8, 0.8]  # Same TPR
    equalized_odds_fpr = [0.2, 0.2]  # Same FPR
    calibration = [0.7, 0.7]  # Same calibration

    x = np.arange(len(metrics))
    width = 0.35

    # Plot bars for statistical parity
    ax.bar(x[0] - width/2, statistical_parity[0], width, label=groups[0], color=colors[0], edgecolor='black', linewidth=1.5)
    ax.bar(x[0] + width/2, statistical_parity[1], width, label=groups[1], color=colors[1], edgecolor='black', linewidth=1.5)

    # Equalized odds (TPR)
    ax.bar(x[1] - width/2, equalized_odds_tpr[0], width, color=colors[0], edgecolor='black', linewidth=1.5)
    ax.bar(x[1] + width/2, equalized_odds_tpr[1], width, color=colors[1], edgecolor='black', linewidth=1.5)

    # Calibration
    ax.bar(x[2] - width/2, calibration[0], width, color=colors[0], edgecolor='black', linewidth=1.5)
    ax.bar(x[2] + width/2, calibration[1], width, color=colors[1], edgecolor='black', linewidth=1.5)

    # Annotations
    ax.text(0, 1.05, "VIOLATED\n(0.6 ≠ 0.4)", ha='center', fontsize=10, weight='bold', color=COLOR_HARM)
    ax.text(1, 1.05, "SATISFIED\n(0.8 = 0.8)", ha='center', fontsize=10, weight='bold', color=COLOR_SOLUTION)
    ax.text(2, 1.05, "SATISFIED\n(0.7 = 0.7)", ha='center', fontsize=10, weight='bold', color=COLOR_SOLUTION)

    ax.set_ylabel('Rate', fontsize=13, weight='bold')
    ax.set_title('Fairness Metrics Can Conflict: Same Model, Different Fairness',
                 fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, weight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)

    # Add formulas
    formulas = [
        r'$P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)$',
        r'$P(\hat{Y}=1|Y=1,A=0) = P(\hat{Y}=1|Y=1,A=1)$',
        r'$P(Y=1|\hat{Y}=p,A=0) = P(Y=1|\hat{Y}=p,A=1)$'
    ]
    for i, formula in enumerate(formulas):
        ax.text(i, -0.15, formula, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('../figures/fairness_metrics_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("5/8: fairness_metrics_comparison.pdf created")

# Chart 6: Debiasing Process (Gender Subspace)
def plot_debiasing_process():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Before debiasing
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)

    # Gender axis
    ax1.arrow(-1.2, -1.2, 2.4, 2.4, head_width=0.1, head_length=0.1, fc=COLOR_HARM, ec=COLOR_HARM, linewidth=3)
    ax1.text(1.3, 1.5, "Gender\nAxis", fontsize=12, weight='bold', color=COLOR_HARM)

    # Gendered words
    ax1.scatter([0.8], [0.8], s=300, c=COLOR_HARM, marker='o', edgecolors='black', linewidth=2, label='Gendered')
    ax1.text(0.8, 1.0, "'he'", fontsize=11, ha='center', weight='bold')

    ax1.scatter([-0.8], [-0.8], s=300, c=COLOR_HARM, marker='o', edgecolors='black', linewidth=2)
    ax1.text(-0.8, -1.0, "'she'", fontsize=11, ha='center', weight='bold')

    # Biased neutral words
    ax1.scatter([0.6, 0.4], [0.5, 0.3], s=300, c=COLOR_WARNING, marker='s', edgecolors='black', linewidth=2, label='Biased Neutral')
    ax1.text(0.6, 0.7, "'doctor'", fontsize=11, ha='center', weight='bold')
    ax1.text(0.4, 0.5, "'engineer'", fontsize=11, ha='center', weight='bold')

    ax1.scatter([-0.5], [-0.4], s=300, c=COLOR_WARNING, marker='s', edgecolors='black', linewidth=2)
    ax1.text(-0.5, -0.6, "'nurse'", fontsize=11, ha='center', weight='bold')

    ax1.set_xlabel('Dimension 1', fontsize=12)
    ax1.set_ylabel('Dimension 2', fontsize=12)
    ax1.set_title('BEFORE Debiasing:\nNeutral words aligned with gender', fontsize=14, weight='bold', color=COLOR_HARM)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(alpha=0.3)

    # After debiasing
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)

    # Gender axis (preserved for gendered words)
    ax2.arrow(-1.2, -1.2, 2.4, 2.4, head_width=0.1, head_length=0.1, fc='gray', ec='gray', linewidth=2, alpha=0.3, linestyle='--')
    ax2.text(1.3, 1.5, "Gender\nAxis\n(removed)", fontsize=12, weight='bold', color='gray', alpha=0.5)

    # Gendered words (unchanged)
    ax2.scatter([0.8], [0.8], s=300, c=COLOR_HARM, marker='o', edgecolors='black', linewidth=2, label='Gendered (preserved)')
    ax2.text(0.8, 1.0, "'he'", fontsize=11, ha='center', weight='bold')

    ax2.scatter([-0.8], [-0.8], s=300, c=COLOR_HARM, marker='o', edgecolors='black', linewidth=2)
    ax2.text(-0.8, -1.0, "'she'", fontsize=11, ha='center', weight='bold')

    # Debiased neutral words (projected to perpendicular)
    ax2.scatter([0.0, 0.0], [0.8, 0.5], s=300, c=COLOR_SOLUTION, marker='s', edgecolors='black', linewidth=2, label='Debiased Neutral')
    ax2.text(0.0, 1.0, "'doctor'", fontsize=11, ha='center', weight='bold')
    ax2.text(0.0, 0.7, "'engineer'", fontsize=11, ha='center', weight='bold')

    ax2.scatter([0.0], [-0.6], s=300, c=COLOR_SOLUTION, marker='s', edgecolors='black', linewidth=2)
    ax2.text(0.0, -0.8, "'nurse'", fontsize=11, ha='center', weight='bold')

    # Arrows showing projection
    for orig_x, orig_y, new_x, new_y in [(0.6, 0.5, 0.0, 0.8), (0.4, 0.3, 0.0, 0.5), (-0.5, -0.4, 0.0, -0.6)]:
        ax2.arrow(orig_x, orig_y, new_x - orig_x, new_y - orig_y,
                 head_width=0.05, head_length=0.05, fc=COLOR_SOLUTION, ec=COLOR_SOLUTION,
                 linewidth=2, linestyle='--', alpha=0.6)

    ax2.set_xlabel('Dimension 1', fontsize=12)
    ax2.set_ylabel('Dimension 2', fontsize=12)
    ax2.set_title('AFTER Debiasing:\nNeutral words gender-neutral', fontsize=14, weight='bold', color=COLOR_SOLUTION)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(alpha=0.3)

    plt.suptitle('Word Embedding Debiasing: Removing Gender Bias via Projection',
                 fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/debiasing_process.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("6/8: debiasing_process.pdf created")

# Chart 7: Real World Harms 2024
def plot_real_world_harms_2024():
    fig, ax = plt.subplots(figsize=(14, 10))

    harms = [
        ("Healthcare:\nSkin Cancer\nDetection", "93% accuracy\n(white skin)\n\n68% accuracy\n(black skin)",
         COLOR_HARM, 0.2, 0.7),
        ("Criminal Justice:\nCOMPAS\nRecidivism", "45% false positive\n(Black defendants)\n\n23% false positive\n(White defendants)",
         COLOR_HARM, 0.5, 0.7),
        ("Language Models:\nGPT-3 Gender\nStereotypes", "'doctor' → 'he' (67%)\n'nurse' → 'she' (71%)\n\nImproved in GPT-4",
         COLOR_WARNING, 0.8, 0.7),
        ("Financial:\nCredit\nScoring", "Women pay higher\ninterest rates\n\nControlling for risk",
         COLOR_HARM, 0.35, 0.3),
        ("Employment:\nAmazon Hiring\nAI", "-5 stars for\n'women's' keyword\n\nSystem scrapped",
         COLOR_HARM, 0.65, 0.3)
    ]

    for title, description, color, x, y in harms:
        # Harm box
        harm_box = FancyBboxPatch((x - 0.12, y - 0.15), 0.24, 0.25,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor='black', linewidth=3, alpha=0.7)
        ax.add_patch(harm_box)

        # Title
        ax.text(x, y + 0.08, title, fontsize=11, ha='center', va='center', weight='bold')

        # Description
        ax.text(x, y - 0.05, description, fontsize=9, ha='center', va='center')

    # Impact summary
    impact_text = "Combined Impact: Millions affected by biased AI decisions in 2024\nAcross healthcare, justice, finance, employment, and language technology"
    ax.text(0.5, 0.05, impact_text, fontsize=13, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor=COLOR_HARM, alpha=0.3, edgecolor='black', linewidth=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Real-World AI Harms in 2024: Documented Cases with Quantified Impact",
                 fontsize=16, weight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/real_world_harms_2024.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("7/8: real_world_harms_2024.pdf created")

# Chart 8: Intervention Decision Tree
def plot_intervention_decision_tree():
    fig, ax = plt.subplots(figsize=(14, 11))

    # Root question
    root_box = FancyBboxPatch((0.35, 0.9), 0.3, 0.08, boxstyle="round,pad=0.02",
                              facecolor=COLOR_INFO, edgecolor='black', linewidth=3)
    ax.add_patch(root_box)
    ax.text(0.5, 0.94, "What type of bias?", fontsize=13, ha='center', va='center', weight='bold')

    # Three bias types
    bias_types = [
        ("Data Bias", 0.17, 0.75),
        ("Model Bias", 0.5, 0.75),
        ("Deployment Bias", 0.83, 0.75)
    ]

    for bias_type, x, y in bias_types:
        # Connect to root
        ax.plot([0.5, x], [0.9, y + 0.04], 'k-', linewidth=2)

        # Bias type box
        type_box = FancyBboxPatch((x - 0.08, y), 0.16, 0.06, boxstyle="round,pad=0.01",
                                  facecolor=COLOR_WARNING, edgecolor='black', linewidth=2)
        ax.add_patch(type_box)
        ax.text(x, y + 0.03, bias_type, fontsize=11, ha='center', va='center', weight='bold')

    # Interventions for each type
    interventions = [
        # Data Bias interventions
        [("Data\nAugmentation", "Add synthetic\nminority samples", 0.05, 0.55, COLOR_SOLUTION),
         ("Reweighting", "Weight samples\nby frequency", 0.17, 0.55, COLOR_SOLUTION),
         ("Better\nLabeling", "Audit and fix\nlabel bias", 0.29, 0.55, COLOR_SOLUTION)],

        # Model Bias interventions
        [("Adversarial\nDebiasing", "Adversary predicts\nprotected attr", 0.38, 0.55, COLOR_SOLUTION),
         ("Fairness\nConstraints", "Add fairness to\nloss function", 0.5, 0.55, COLOR_SOLUTION),
         ("Model\nSelection", "Choose less\nbiased arch", 0.62, 0.55, COLOR_SOLUTION)],

        # Deployment Bias interventions
        [("Post-process\nCalibration", "Adjust outputs\nper group", 0.71, 0.55, COLOR_SOLUTION),
         ("A/B Testing", "Monitor disparate\nimpact", 0.83, 0.55, COLOR_SOLUTION),
         ("Red Team\nAudits", "Stress test for\nbias", 0.95, 0.55, COLOR_SOLUTION)]
    ]

    for i, (bias_type, x_center, y_type) in enumerate(bias_types):
        for intervention_name, description, x, y, color in interventions[i]:
            # Connect to bias type
            ax.plot([x_center, x], [y_type, y + 0.05], 'k--', linewidth=1, alpha=0.5)

            # Intervention box
            int_box = FancyBboxPatch((x - 0.06, y), 0.12, 0.08, boxstyle="round,pad=0.008",
                                     facecolor=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.add_patch(int_box)
            ax.text(x, y + 0.04, intervention_name, fontsize=9, ha='center', va='center', weight='bold')

            # Description below
            ax.text(x, y - 0.06, description, fontsize=7, ha='center', va='center', style='italic')

    # Decision guidance
    guidance_text = "Decision Path: Identify bias type → Select appropriate intervention → Validate with fairness metrics"
    ax.text(0.5, 0.35, guidance_text, fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor=COLOR_INFO, alpha=0.3, edgecolor='black', linewidth=2))

    # Validation metrics
    metrics_text = "Validation Metrics:\nStatistical Parity | Equalized Odds | Calibration | Counterfactual Fairness"
    ax.text(0.5, 0.22, metrics_text, fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_SOLUTION, alpha=0.3))

    # Warning
    warning_text = "Warning: Some fairness metrics conflict - choose based on application context"
    ax.text(0.5, 0.08, warning_text, fontsize=10, ha='center', style='italic', color=COLOR_HARM, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_WARNING, alpha=0.3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Bias Intervention Decision Tree: From Detection to Mitigation",
                 fontsize=16, weight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/intervention_decision_tree.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("8/8: intervention_decision_tree.pdf created")

# Generate all charts
if __name__ == "__main__":
    plot_hiring_bias_timeline()
    plot_bias_sources_flowchart()
    plot_harm_taxonomy_tree()
    plot_stakeholder_map()
    plot_fairness_metrics_comparison()
    plot_debiasing_process()
    plot_real_world_harms_2024()
    plot_intervention_decision_tree()

    print("=" * 60)
    print("All 8 BSc charts generated successfully!")
    print("Location: NLP_slides/week12_ethics/figures/")
