import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('../figures', exist_ok=True)

# Clean color scheme (no boxes needed)
COLOR_PRIMARY = '#2E5090'
COLOR_SECONDARY = '#8B4513'
COLOR_TERTIARY = '#2F8F4F'
COLOR_GRAY = '#666666'

# Chart 1: Temperature Effect (Clean Table-Based)
def plot_temperature_clean():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data for temperature comparison
    temps = ['T=0.3\n(Factual)', 'T=0.7\n(Balanced)', 'T=1.0\n(Creative)']
    outputs = [
        'Federal Reserve chiefs have raised\ninterest rates to 5.00%-5.25%,\nhighest level in 16 years',
        'Federal Reserve President remained\ncalm in wake of interest rate flurry',
        'Federal Reserve chair said US rate\nhad been lowered (INCORRECT)'
    ]
    quality = [9, 6, 3]  # Quality scores (illustrative)

    # Bar chart for quality scores
    colors = ['#4CAF50', '#FFC107', '#F44336']
    bars = ax.barh(range(len(temps)), quality, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

    # Add text outputs
    for i, (temp, output, q) in enumerate(zip(temps, outputs, quality)):
        ax.text(-0.5, i, temp, ha='right', va='center', fontsize=11, weight='bold', color=COLOR_GRAY)
        ax.text(q + 0.3, i, output, ha='left', va='center', fontsize=9, color=COLOR_GRAY,
                family='monospace', linespacing=1.3)

    ax.set_xlabel('Relative Quality (Factual Accuracy)', fontsize=12, weight='bold')
    ax.set_xlim(-1, 12)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.title('Temperature Effect on Output Quality', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('../figures/temperature_effect_clean.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("1/4: Temperature effect (clean)")

# Chart 2: Nucleus Sampling (Already Clean - Keep)
def plot_nucleus_clean():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Probability distribution
    words = ["growth", "increase", "rise", "gain", "surge", "uptick", "boost", "jump"]
    probs = np.array([0.35, 0.25, 0.15, 0.10, 0.07, 0.04, 0.03, 0.01])
    cumsum = np.cumsum(probs)

    colors = ['#66BB66' if cumsum[i] <= 0.9 else '#CCCCCC' for i in range(len(words))]
    ax1.bar(range(len(words)), probs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Cumulative line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(words)), cumsum, 'r-', linewidth=2.5, marker='o', markersize=8)
    ax1_twin.axhline(y=0.9, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1_twin.set_ylabel('Cumulative Probability', fontsize=11, color='red', weight='bold')
    ax1_twin.set_ylim(0, 1.1)

    ax1.set_xlabel('Next Word Candidates', fontsize=11, weight='bold')
    ax1.set_ylabel('Probability', fontsize=11, weight='bold')
    ax1.set_xticks(range(len(words)))
    ax1.set_xticklabels(words, rotation=45, ha='right')
    ax1.set_title('Nucleus (Top-p=0.9) Sampling', fontsize=12, weight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: Explanation
    ax2.axis('off')
    explanation = """Top-p Algorithm:

1. Sort words by probability
2. Compute cumulative sum
3. Include words until sum ≥ p (0.9)
4. Sample from included set

Result: Dynamic vocabulary size
- Peaked distribution → few words
- Flat distribution → many words

Included (green): Top 90% probability
Excluded (gray): Bottom 10% (too unlikely)"""

    ax2.text(0.1, 0.5, explanation, fontsize=10, verticalalignment='center',
             family='monospace', color=COLOR_GRAY, linespacing=1.6)

    plt.tight_layout()
    plt.savefig('../figures/nucleus_sampling_clean.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("2/4: Nucleus sampling (clean)")

# Chart 3: Max Tokens Comparison (Text Table)
def plot_max_tokens_clean():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Table data
    configs = ['max_tokens=50\n(Too Short)', 'max_tokens=150\n(Just Right)', 'max_tokens=500\n(Too Verbose)']
    outputs = [
        'The study examined...\n[TRUNCATED]',
        'Study examined treatment efficacy in\n1000 patients. Results showed 25%\nimprovement with minimal side effects.\nRecommended for clinical use.',
        'The comprehensive longitudinal study\nmeticulously examined treatment efficacy\nacross multiple patient cohorts totaling\napproximately 1000 individuals.\nResults demonstrated statistically\nsignificant improvement of 25%...'
    ]
    ratings = ['X', 'OK', '?']
    colors_rating = {'X': '#F44336', 'OK': '#4CAF50', '?': '#FFC107'}

    y_positions = [0.7, 0.4, 0.1]

    for i, (config, output, rating, y) in enumerate(zip(configs, outputs, ratings, y_positions)):
        # Configuration label
        ax.text(0.05, y + 0.12, config, fontsize=11, weight='bold', color=COLOR_GRAY)

        # Output text
        ax.text(0.25, y + 0.08, output, fontsize=9, family='monospace',
                color=COLOR_GRAY, verticalalignment='top', linespacing=1.4)

        # Rating symbol
        ax.text(0.92, y + 0.08, rating, fontsize=24, weight='bold',
                color=colors_rating[rating], ha='center', va='center')

    ax.text(0.5, 0.95, 'Max Tokens: Length Control Comparison',
            fontsize=14, weight='bold', ha='center')

    ax.text(0.5, 0.02, 'Set max_tokens based on desired summary length (typically 100-200 for articles)',
            fontsize=10, ha='center', style='italic', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/max_tokens_clean.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("3/4: Max tokens comparison (clean)")

# Chart 4: Repetition Penalty (Text Comparison)
def plot_repetition_penalty_clean():
    fig, ax = plt.subplots(figsize=(12, 7))

    # Without penalty
    ax.text(0.5, 0.95, 'Repetition Penalty Effect', fontsize=14, weight='bold', ha='center')

    # Red box for problem
    ax.text(0.5, 0.85, 'WITHOUT Penalty (1.0)', fontsize=12, weight='bold', ha='center', color='#D32F2F')
    bad_text = '"The company reported strong results. The company announced\nstrong earnings. The company\'s financial performance was strong.\nThe company showed strong growth."'
    ax.text(0.5, 0.73, bad_text, fontsize=9, ha='center', family='monospace',
            color='#D32F2F', linespacing=1.5)
    ax.text(0.15, 0.68, '5x "company"\n5x "strong"', fontsize=9, weight='bold', color='#D32F2F')

    # Arrow
    ax.annotate('', xy=(0.5, 0.58), xytext=(0.5, 0.63),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_GRAY))
    ax.text(0.52, 0.605, 'Apply penalty=1.2', fontsize=10, style='italic', color=COLOR_GRAY)

    # Green for solution
    ax.text(0.5, 0.52, 'WITH Penalty (1.2)', fontsize=12, weight='bold', ha='center', color='#388E3C')
    good_text = '"The firm reported strong Q4 results, with revenue increasing 15%\nyear-over-year. This performance exceeded analyst expectations\nand demonstrated effective cost management."'
    ax.text(0.5, 0.40, good_text, fontsize=9, ha='center', family='monospace',
            color='#388E3C', linespacing=1.5)
    ax.text(0.15, 0.35, 'Varied vocabulary\nNatural flow', fontsize=9, weight='bold', color='#388E3C')

    # Bottom explanation
    ax.text(0.5, 0.22, 'Repetition Penalty: Reduces probability of recently used tokens',
            fontsize=11, ha='center', weight='bold', color=COLOR_GRAY)
    ax.text(0.5, 0.15, 'Values: 1.0 (none) | 1.1 (mild) | 1.2 (moderate) | 1.5+ (aggressive)',
            fontsize=10, ha='center', style='italic', color=COLOR_GRAY)
    ax.text(0.5, 0.08, 'For summarization: Use 1.1-1.2 to encourage diversity without awkwardness',
            fontsize=10, ha='center', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.savefig('../figures/repetition_penalty_clean.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("4/4: Repetition penalty (clean)")

# Generate all clean charts
if __name__ == "__main__":
    print("Generating 4 clean Python charts (no boxes)...")
    plot_temperature_clean()
    plot_nucleus_clean()
    plot_max_tokens_clean()
    plot_repetition_penalty_clean()
    print("\nAll 4 clean charts generated successfully!")
    print("Output: NLP_slides/summarization_module/figures/*_clean.pdf")
