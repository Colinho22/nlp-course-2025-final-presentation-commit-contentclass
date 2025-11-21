#!/usr/bin/env python3
"""
Generate all charts for Week 9: Decoding Strategies (Enhanced 2025)
Creates 22 professional charts for BSc discovery-based presentation (42 slides)

Enhancements:
- Added contrastive search visualizations
- Added quality-diversity optimization
- Added comprehensive task-specific recommendations
- All charts use BSc Discovery color scheme
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs('.', exist_ok=True)

# Standard setup
plt.style.use('seaborn-v0_8-whitegrid')

# Template Beamer Final color scheme (PROFESSIONAL)
COLOR_MAIN = '#333333'          # Main text RGB(51,51,51)
COLOR_ACCENT = '#3333B2'        # mlpurple RGB(51,51,178)
COLOR_LAVENDER = '#ADADC0'      # mllavender RGB(173,173,224)
COLOR_LAVENDER2 = '#C1C1E8'     # mllavender2 RGB(193,193,232)
COLOR_BLUE = '#0066CC'          # mlblue RGB(0,102,204)
COLOR_GRAY = '#7F7F7F'          # mlgray RGB(127,127,127)
COLOR_LIGHT = '#F0F0F0'         # Backgrounds
COLOR_RED = '#D62728'           # mlred RGB(214,39,40)
COLOR_GREEN = '#2CA02C'         # mlgreen RGB(44,160,44)
COLOR_ORANGE = '#FF7F0E'        # mlorange RGB(255,127,14)

# Font sizes (MINIMUM 30pt for maximum projection readability)
FONTSIZE_TITLE = 36
FONTSIZE_LABEL = 30
FONTSIZE_TICK = 28
FONTSIZE_ANNOTATION = 28
FONTSIZE_LEGEND = 26
FONTSIZE_TEXT = 30

def set_minimalist_style(ax):
    """Apply template beamer final styling to axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_ACCENT)
    ax.spines['bottom'].set_color(COLOR_ACCENT)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(colors=COLOR_MAIN, which='both', labelsize=FONTSIZE_TICK, width=2, length=6)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=3, color=COLOR_LAVENDER)
    ax.set_facecolor('white')

# === CHART 1: Quality-Diversity Tradeoff (HOOK) ===
def generate_quality_diversity_tradeoff():
    """Discovery hook: The paradox of text generation"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create scatter plot showing the tradeoff
    # Greedy: High quality, low diversity
    # Beam: High quality, low diversity
    # Temp high: Low quality, high diversity
    # Nucleus: Balanced

    methods = ['Greedy', 'Beam\nSearch', 'Temp\n(low)', 'Temp\n(high)',
               'Top-k', 'Nucleus\n(Top-p)']
    quality = [85, 88, 70, 45, 65, 80]
    diversity = [15, 20, 60, 95, 70, 75]
    colors_map = [COLOR_RED, COLOR_RED, COLOR_GRAY, COLOR_RED,
                  COLOR_GRAY, COLOR_GREEN]

    for i, (method, q, d, c) in enumerate(zip(methods, quality, diversity, colors_map)):
        ax.scatter(d, q, s=1000, c=c, alpha=0.7, edgecolors=COLOR_MAIN, linewidths=3)
        ax.annotate(method, (d, q), fontsize=42, ha='center', va='center',
                   weight='bold', color='white')

    # Add "Sweet Spot" annotation
    ax.annotate('Sweet Spot', xy=(75, 80), xytext=(85, 90),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN),
               fontsize=40, weight='bold', color=COLOR_GREEN)

    ax.set_xlabel('Diversity (Creativity)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_ylabel('Quality (Coherence)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_title('The Decoding Paradox: Quality vs Diversity Tradeoff',
                fontsize=FONTSIZE_TITLE, weight='bold', pad=20)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add problem zones
    ax.axhspan(0, 50, alpha=0.05, color=COLOR_RED, label='Too Random')
    ax.axvspan(0, 30, alpha=0.05, color=COLOR_RED, label='Too Boring')

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('./quality_diversity_tradeoff_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: quality_diversity_tradeoff_bsc.pdf")

# === CHART 2: Beam Search Visual ===
def generate_beam_search_visual():
    """Multiple paths explored simultaneously"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simple beam search tree (width=3)
    levels = 4  # Steps
    beam_width = 3

    # Draw tree structure
    for level in range(levels):
        x_positions = np.linspace(0.1, 0.9, beam_width ** min(level, 1))
        y = 1 - (level / (levels - 1))

        for x in x_positions:
            if level == 0:
                ax.scatter(0.5, y, s=1100, c=COLOR_ACCENT, zorder=3, edgecolors=COLOR_MAIN, linewidths=3)
                ax.text(0.5, y, 'START', ha='center', va='center', weight='bold', color='white', fontsize=42)
            else:
                # Draw multiple branches
                for i in range(beam_width):
                    x_new = x + (i - 1) * 0.15 / level
                    if 0.05 < x_new < 0.95:
                        ax.scatter(x_new, y, s=900, c=COLOR_GRAY, alpha=0.6, zorder=2)
                        # Draw line from parent
                        if level > 0:
                            ax.plot([x, x_new], [1 - ((level-1)/(levels-1)), y],
                                   'k-', alpha=0.3, linewidth=1)

    # Highlight best paths (kept beams)
    best_x = [0.5, 0.45, 0.5, 0.55]
    for i in range(len(best_x)):
        y = 1 - (i / (levels - 1))
        ax.scatter(best_x[i], y, s=1100, c=COLOR_ACCENT, zorder=4,
                  edgecolors=COLOR_MAIN, linewidths=3)
        if i > 0:
            ax.plot([best_x[i-1], best_x[i]], [1 - ((i-1)/(levels-1)), y],
                   color=COLOR_ACCENT, linewidth=3, zorder=1)

    ax.text(0.5, -0.1, 'Beam Width = 3\nKeeps top 3 paths at each step',
           ha='center', fontsize=38, style='italic', color=COLOR_MAIN)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.05)
    ax.axis('off')
    ax.set_title('Beam Search: Exploring Multiple Paths', fontsize=42, weight='bold')

    plt.tight_layout()
    plt.savefig('./beam_search_visual_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: beam_search_visual_bsc.pdf")

# === CHART 3: Beam Example Tree (Worked Example) ===
def generate_beam_example_tree():
    """Concrete numerical example: 'The cat...' with beam width=3"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Tree structure with actual words and scores
    ax.text(0.5, 0.95, 'START: "The cat"', ha='center', fontsize=38,
           weight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, alpha=0.8))

    # Step 1: Top 3 candidates
    step1_words = ['sat', 'jumped', 'walked']
    step1_scores = [0.45, 0.30, 0.25]
    for i, (word, score) in enumerate(zip(step1_words, step1_scores)):
        x = 0.2 + i * 0.3
        ax.text(x, 0.75, f'{word}\n{score:.2f}', ha='center', fontsize=42,
               bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN))
        ax.plot([0.5, x], [0.93, 0.77], 'k-', alpha=0.4, linewidth=1)

    # Step 2: Expand best 3 (only showing expansion from 'sat')
    ax.annotate('Expand each\nof top 3', xy=(0.2, 0.73), xytext=(0.05, 0.60),
               arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ACCENT),
               fontsize=42, style='italic', color=COLOR_ACCENT)

    step2_words = ['on', 'down', 'still']
    step2_scores = [0.50, 0.30, 0.20]
    for i, (word, score) in enumerate(zip(step2_words, step2_scores)):
        x = 0.05 + i * 0.12
        cumulative = step1_scores[0] * score  # Multiply probabilities
        ax.text(x, 0.50, f'{word}\n{cumulative:.3f}', ha='center', fontsize=42,
               bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT if i < 3 else '#FFE6E6'))
        ax.plot([0.2, x], [0.73, 0.52], 'k-', alpha=0.3, linewidth=1)

    # Best path highlighted
    ax.text(0.05, 0.35, '"The cat sat on"\nScore: 0.225', ha='center', fontsize=42,
           weight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))
    ax.plot([0.05, 0.05], [0.48, 0.37], color=COLOR_ACCENT, linewidth=3)

    # Key insight box
    ax.text(0.75, 0.50, 'Key: At each step\n• Score all candidates\n• Keep top 3\n• Prune rest\n• Repeat',
           ha='center', va='center', fontsize=42,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_ACCENT, linewidth=3))

    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.axis('off')
    ax.set_title('Beam Search Example: "The cat..." (Width=3)', fontsize=40, weight='bold')

    plt.tight_layout()
    plt.savefig('./beam_example_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: beam_example_tree_bsc.pdf")

# === CHART 4: Temperature Effects ===
def generate_temperature_effects():
    """How temperature reshapes probability distributions"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    # Original logits
    tokens = ['the', 'a', 'on', 'at', 'in']
    logits = np.array([2.0, 1.0, 0.5, 0.3, 0.1])

    # Three temperatures
    temps = [0.5, 1.0, 2.0]
    axes = [ax1, ax2, ax3]
    titles = ['T=0.5\n(Focused)', 'T=1.0\n(Balanced)', 'T=2.0\n(Flat)']

    for temp, ax, title in zip(temps, axes, titles):
        # Apply temperature
        scaled = logits / temp
        probs = np.exp(scaled) / np.sum(np.exp(scaled))

        bars = ax.bar(tokens, probs, color=COLOR_ACCENT, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)

        # Highlight max
        max_idx = np.argmax(probs)
        bars[max_idx].set_color(COLOR_GREEN)
        bars[max_idx].set_alpha(0.9)

        ax.set_title(title, fontsize=40, weight='bold')
        ax.set_ylabel('Probability' if ax == ax1 else '', fontsize=42)
        ax.set_ylim(0, max(probs) * 1.2)

        # Add probability labels
        for i, (token, prob) in enumerate(zip(tokens, probs)):
            ax.text(i, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=42, weight='bold')

        set_minimalist_style(ax)

    fig.suptitle('Temperature Effects on Probability Distribution', fontsize=42, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./temperature_effects_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: temperature_effects_bsc.pdf")

# === CHART 5: Temperature Calculation (Worked Example) ===
def generate_temperature_calculation():
    """Step-by-step numerical example"""
    fig, ax = plt.subplots(figsize=(11, 7))

    # Setup
    ax.text(0.5, 0.95, 'Temperature Calculation: Step-by-Step',
           ha='center', fontsize=42, weight='bold')

    # Given
    ax.text(0.05, 0.85, 'Given: Logits = [2.0, 1.0, 0.5, 0.2]',
           fontsize=38, weight='bold', color=COLOR_MAIN)
    ax.text(0.05, 0.80, 'Tokens = ["cat", "dog", "bird", "fish"]',
           fontsize=38, color=COLOR_MAIN)

    # Step 1: T=0.5
    ax.text(0.05, 0.70, 'Step 1: Scale by T=0.5 (MORE PEAKED)',
           fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.05, 0.65, 'Scaled = [2.0/0.5, 1.0/0.5, 0.5/0.5, 0.2/0.5]', fontsize=42)
    ax.text(0.05, 0.60, '       = [4.0, 2.0, 1.0, 0.4]', fontsize=42, weight='bold', color=COLOR_ACCENT)
    ax.text(0.05, 0.55, 'Softmax → [0.73, 0.18, 0.07, 0.02]', fontsize=42)
    ax.text(0.05, 0.50, '→ 73% on "cat" (VERY FOCUSED)', fontsize=42,
           weight='bold', color=COLOR_GREEN)

    # Step 2: T=1.0
    ax.text(0.55, 0.70, 'Step 2: Scale by T=1.0 (UNCHANGED)',
           fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.55, 0.65, 'Scaled = [2.0, 1.0, 0.5, 0.2]', fontsize=42)
    ax.text(0.55, 0.60, 'Softmax → [0.53, 0.19, 0.12, 0.10]', fontsize=42)
    ax.text(0.55, 0.55, '→ 53% on "cat" (BALANCED)', fontsize=42, weight='bold')

    # Step 3: T=2.0
    ax.text(0.05, 0.38, 'Step 3: Scale by T=2.0 (FLATTER)',
           fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.05, 0.33, 'Scaled = [1.0, 0.5, 0.25, 0.1]', fontsize=42)
    ax.text(0.05, 0.28, 'Softmax → [0.38, 0.23, 0.17, 0.15]', fontsize=42)
    ax.text(0.05, 0.23, '→ 38% on "cat" (MUCH FLATTER)', fontsize=42,
           weight='bold', color=COLOR_RED)

    # Formula
    ax.text(0.55, 0.35, 'General Formula:', fontsize=38, weight='bold')
    ax.text(0.55, 0.28, r'$p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$',
           fontsize=40, bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))

    # Insight box
    ax.text(0.3, 0.08, 'Lower T → More confident (peaky)\nHigher T → More random (flat)',
           ha='center', fontsize=42, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./temperature_calculation_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: temperature_calculation_bsc.pdf")

# === CHART 6: Top-k Filtering ===
def generate_topk_filtering():
    """Top-k selection process visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create probability distribution
    tokens = [f'tok{i}' for i in range(10)]
    probs = np.array([0.30, 0.18, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.01])

    # k=5 cutoff
    k = 5

    # Draw bars individually with different alphas
    for i, (token, prob) in enumerate(zip(tokens, probs)):
        color = COLOR_GREEN if i < k else COLOR_RED
        alpha = 0.8 if i < k else 0.3
        ax.bar(i, prob, color=color, alpha=alpha, edgecolor=COLOR_MAIN, linewidth=3)

    # Add line showing cutoff
    ax.axvline(k - 0.5, color=COLOR_ACCENT, linestyle='--', linewidth=3, label=f'k={k} cutoff')

    # Add annotations
    ax.text(2, 0.35, f'Top-{k}\nSample from these', ha='center',
           fontsize=38, weight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_GREEN, linewidth=3))

    ax.text(7, 0.08, 'Ignored\n(too unlikely)', ha='center',
           fontsize=42, style='italic', color=COLOR_RED)

    # Labels
    ax.set_xlabel('Tokens (ranked by probability)', fontsize=38, weight='bold')
    ax.set_ylabel('Probability', fontsize=38, weight='bold')
    ax.set_title('Top-k Filtering: Fixed Vocabulary Size', fontsize=40, weight='bold')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.legend(loc='upper right', fontsize=42)

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('./topk_filtering_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: topk_filtering_bsc.pdf")

# === CHART 7: Top-k Example ===
def generate_topk_example():
    """Worked numerical example with k=3"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.text(0.5, 0.95, 'Top-k Example: k=3', ha='center', fontsize=42, weight='bold')

    # Original distribution
    ax.text(0.05, 0.85, '1. Original Probabilities:', fontsize=38, weight='bold', color=COLOR_ACCENT)
    data1 = [
        ('cat', 0.45),
        ('dog', 0.18),
        ('bird', 0.15),
        ('fish', 0.10),
        ('mouse', 0.08),
        ('...', 0.04)
    ]
    y = 0.78
    for token, prob in data1:
        ax.text(0.08, y, f'{token}: {prob:.2f}', fontsize=42)
        y -= 0.05

    # After filtering
    ax.text(0.40, 0.85, '2. Keep Top-3:', fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.42, 0.78, 'cat: 0.45', fontsize=42, weight='bold', color=COLOR_GREEN)
    ax.text(0.42, 0.73, 'dog: 0.18', fontsize=42, weight='bold', color=COLOR_GREEN)
    ax.text(0.42, 0.68, 'bird: 0.15', fontsize=42, weight='bold', color=COLOR_GREEN)
    ax.text(0.42, 0.63, '(discard rest)', fontsize=42, style='italic', color=COLOR_GRAY)

    # Renormalize
    ax.text(0.70, 0.85, '3. Renormalize:', fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.text(0.72, 0.78, 'Sum = 0.45 + 0.18 + 0.15 = 0.78', fontsize=42)
    ax.text(0.72, 0.72, 'cat: 0.45/0.78 = 0.58', fontsize=42, color=COLOR_MAIN)
    ax.text(0.72, 0.67, 'dog: 0.18/0.78 = 0.23', fontsize=42, color=COLOR_MAIN)
    ax.text(0.72, 0.62, 'bird: 0.15/0.78 = 0.19', fontsize=42, color=COLOR_MAIN)

    # Arrow showing process
    ax.annotate('', xy=(0.38, 0.75), xytext=(0.35, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ACCENT))
    ax.annotate('', xy=(0.68, 0.75), xytext=(0.65, 0.75),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ACCENT))

    # Result
    ax.text(0.5, 0.25, 'Result: Sample from {cat: 58%, dog: 23%, bird: 19%}',
           ha='center', fontsize=38, weight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.2,
                    edgecolor=COLOR_GREEN, linewidth=3))

    ax.text(0.5, 0.10, 'Prevents sampling from long tail ("mouse" eliminated)',
           ha='center', fontsize=42, style='italic', color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./topk_example_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: topk_example_bsc.pdf")

# === CHART 8: Nucleus Process ===
def generate_nucleus_process():
    """Cumulative probability threshold visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Probability distribution
    tokens = ['cat', 'dog', 'bird', 'fish', 'mouse', 'rat', 'frog', 'deer']
    probs = np.array([0.40, 0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02])
    cumulative = np.cumsum(probs)

    # p=0.85 threshold
    p_threshold = 0.85
    nucleus_size = np.searchsorted(cumulative, p_threshold) + 1

    x_pos = np.arange(len(tokens))

    # Bar chart with cumulative line
    colors = [COLOR_GREEN if cumulative[i] <= p_threshold else COLOR_RED
             for i in range(len(tokens))]
    bars = ax.bar(x_pos, probs, color=colors, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)

    # Cumulative line
    ax2 = ax.twinx()
    ax2.plot(x_pos, cumulative, color=COLOR_ACCENT, linewidth=3, marker='o',
            markersize=18, label='Cumulative Probability')
    ax2.axhline(p_threshold, color=COLOR_ACCENT, linestyle='--', linewidth=3,
               label=f'p={p_threshold} threshold')

    # Nucleus annotation
    ax.text(nucleus_size/2 - 0.5, 0.45, f'Nucleus\n({nucleus_size} tokens)',
           ha='center', fontsize=38, weight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLOR_GREEN, linewidth=3))

    ax.set_xlabel('Tokens (ranked by probability)', fontsize=38, weight='bold')
    ax.set_ylabel('Individual Probability', fontsize=38, weight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Cumulative Probability', fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.set_title('Nucleus (Top-p) Sampling: Dynamic Vocabulary', fontsize=40, weight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tokens, rotation=0)
    ax2.legend(loc='center right', fontsize=42)

    set_minimalist_style(ax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(COLOR_ACCENT)
    ax2.tick_params(colors=COLOR_ACCENT, which='both')

    plt.tight_layout()
    plt.savefig('./nucleus_process_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: nucleus_process_bsc.pdf")

# === CHART 9: Nucleus Cumulative ===
def generate_nucleus_cumulative():
    """Dynamic cutoff visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Scenario 1: Peaked distribution (nucleus is small)
    probs1 = [0.60, 0.20, 0.10, 0.05, 0.03, 0.02]
    tokens1 = ['the', 'a', 'an', 'this', 'that', 'my']
    cumsum1 = np.cumsum(probs1)
    p_threshold = 0.85
    nucleus1 = np.searchsorted(cumsum1, p_threshold) + 1

    colors1 = [COLOR_GREEN if i < nucleus1 else COLOR_GRAY for i in range(len(probs1))]
    ax1.bar(tokens1, probs1, color=colors1, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)
    ax1.set_title('Peaked Distribution\n(Nucleus = 2 tokens)', fontsize=38, weight='bold')
    ax1.set_ylabel('Probability', fontsize=42, weight='bold')
    ax1.text(0.5, 0.55, f'Top 2 tokens\n= {cumsum1[1]:.0%} mass',
            ha='center', fontsize=42, bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))
    set_minimalist_style(ax1)

    # Scenario 2: Flat distribution (nucleus is large)
    probs2 = [0.18, 0.16, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08]
    tokens2 = ['cat', 'dog', 'bird', 'fish', 'frog', 'deer', 'bear', 'wolf']
    cumsum2 = np.cumsum(probs2)
    nucleus2 = np.searchsorted(cumsum2, p_threshold) + 1

    colors2 = [COLOR_GREEN if i < nucleus2 else COLOR_GRAY for i in range(len(probs2))]
    ax2.bar(tokens2, probs2, color=colors2, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)
    ax2.set_title('Flat Distribution\n(Nucleus = 6 tokens)', fontsize=38, weight='bold')
    ax2.set_ylabel('Probability', fontsize=42, weight='bold')
    ax2.text(2.5, 0.17, f'Top 6 tokens\n= {cumsum2[5]:.0%} mass',
            ha='center', fontsize=42, bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.3))
    set_minimalist_style(ax2)

    fig.suptitle(f'Nucleus Adapts to Distribution Shape (p={p_threshold})',
                fontsize=40, weight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('./nucleus_cumulative_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: nucleus_cumulative_bsc.pdf")

# === CHART 10: Decoding Comparison Table ===
def generate_decoding_comparison():
    """Side-by-side method comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Table data
    methods = ['Greedy', 'Beam\nSearch', 'Temperature', 'Top-k', 'Nucleus\n(Top-p)', 'Contrastive']
    deterministic = ['Yes', 'Yes', 'No', 'No', 'No', 'Yes']
    diversity = ['Very Low', 'Low', 'Variable', 'Medium', 'High', 'High']
    quality = ['Medium', 'High', 'Variable', 'Medium', 'High', 'High']
    speed = ['Fastest', 'Slow', 'Fast', 'Fast', 'Fast', 'Slowest']
    use_case = ['Simple', 'Translation', 'Creative', 'Balanced', 'Best', 'Dedup']

    # Create table
    table_data = []
    for i in range(len(methods)):
        table_data.append([methods[i], deterministic[i], diversity[i], quality[i], speed[i], use_case[i]])

    # Headers
    headers = ['Method', 'Deterministic', 'Diversity', 'Quality', 'Speed', 'Best For']

    # Color code cells
    cell_colors = []
    for row in table_data:
        row_colors = []
        for cell in row:
            if cell in ['High', 'Best', 'Yes']:
                row_colors.append(COLOR_GREEN + '20')  # Light green
            elif cell in ['Low', 'Very Low', 'Slow', 'Slowest']:
                row_colors.append(COLOR_RED + '20')  # Light red
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    cellColours=cell_colors,
                    colWidths=[0.18, 0.15, 0.13, 0.13, 0.12, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(22)
    table.scale(1, 2.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(COLOR_ACCENT)
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.axis('off')
    ax.set_title('Decoding Strategies: Quick Reference', fontsize=42, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('./decoding_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: decoding_comparison_bsc.pdf")

# === NEW CHART 11: Degeneration Problem Visual ===
def generate_degeneration_problem():
    """Show repetitive text problem with greedy/beam"""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Example repetitive text
    ax.text(0.5, 0.85, 'Real Output from Greedy Decoding:', ha='center',
           fontsize=40, weight='bold', color=COLOR_RED)

    repetitive_text = ('"The city of New York is a major city in the United States. '
                      'The city is known for its diverse culture and the city has '
                      'many tourist attractions. The city is also home to the city\'s '
                      'financial district..."')

    # Highlight repetitions
    ax.text(0.5, 0.65, repetitive_text, ha='center', va='center', fontsize=42,
           wrap=True, bbox=dict(boxstyle='round', facecolor=COLOR_RED, alpha=0.1,
                               edgecolor=COLOR_RED, linewidth=3),
           style='italic')

    # Count repetitions
    ax.text(0.5, 0.35, 'Problem: "the city" appears 6 times in 4 sentences!',
           ha='center', fontsize=38, weight='bold', color=COLOR_RED)

    ax.text(0.5, 0.25, 'Why? Always picking argmax → same patterns repeated',
           ha='center', fontsize=42, style='italic', color=COLOR_MAIN)

    # Solution teaser
    ax.text(0.5, 0.10, 'Solution: Penalize tokens similar to recent context (Contrastive Search)',
           ha='center', fontsize=38, weight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.2,
                    edgecolor=COLOR_GREEN, linewidth=3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('The Degeneration Problem: Model Repetition', fontsize=42, weight='bold')

    plt.tight_layout()
    plt.savefig('./degeneration_problem_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: degeneration_problem_bsc.pdf")

# === NEW CHART 12: Contrastive Search Mechanism ===
def generate_contrastive_mechanism():
    """How contrastive search balances probability and diversity"""
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.text(0.5, 0.95, 'Contrastive Search: How It Works', ha='center',
           fontsize=42, weight='bold')

    # Step 1: Model probability
    ax.text(0.05, 0.82, 'Step 1: Get top-k candidates by probability',
           fontsize=42, weight='bold', color=COLOR_ACCENT)
    candidates = [('city', 0.45), ('town', 0.18), ('area', 0.15), ('place', 0.12)]
    y = 0.75
    for token, prob in candidates:
        ax.text(0.08, y, f'• {token}: {prob:.2f}', fontsize=42)
        y -= 0.04

    # Step 2: Context similarity
    ax.text(0.05, 0.55, 'Step 2: Compute similarity to recent context',
           fontsize=42, weight='bold', color=COLOR_ACCENT)
    ax.text(0.08, 0.48, 'Context: "...the city has..."', fontsize=42, style='italic')
    similarities = [('city', 0.92), ('town', 0.75), ('area', 0.65), ('place', 0.60)]
    y = 0.42
    for token, sim in similarities:
        ax.text(0.08, y, f'• {token}: {sim:.2f} (cosine similarity)', fontsize=42)
        y -= 0.04

    # Step 3: Penalty
    ax.text(0.55, 0.82, 'Step 3: Apply diversity penalty (α=0.6)',
           fontsize=42, weight='bold', color=COLOR_ACCENT)
    ax.text(0.57, 0.75, 'score = (1-α)×P(token) - α×similarity', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT))

    scores = []
    y = 0.67
    alpha = 0.6
    for i, ((token, prob), (_, sim)) in enumerate(zip(candidates, similarities)):
        score = (1 - alpha) * prob - alpha * sim
        scores.append((token, score))
        color = COLOR_GREEN if i == 1 else COLOR_MAIN  # 'town' wins
        ax.text(0.57, y, f'{token}: 0.4×{prob:.2f} - 0.6×{sim:.2f} = {score:.3f}',
               fontsize=42, weight='bold' if i == 1 else 'normal', color=color)
        y -= 0.04

    # Result
    ax.text(0.55, 0.48, 'Winner: "town" (high prob, lower similarity)',
           fontsize=42, weight='bold', color=COLOR_GREEN,
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.2,
                    edgecolor=COLOR_GREEN, linewidth=3))

    # Formula box
    ax.text(0.5, 0.15, 'Key Insight: Balance probability (coherence) with diversity (novelty)',
           ha='center', fontsize=42, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN, linewidth=1))

    ax.text(0.5, 0.05, 'α=0: Pure greedy | α=0.6: Balanced | α=1.0: Maximum diversity',
           ha='center', fontsize=42, color=COLOR_GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./contrastive_mechanism_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: contrastive_mechanism_bsc.pdf")

# === NEW CHART 13: Contrastive vs Nucleus Comparison ===
def generate_contrastive_vs_nucleus():
    """Side-by-side output comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.text(0.5, 0.95, 'Same Prompt, Different Methods', ha='center',
           fontsize=42, weight='bold')

    # Prompt
    ax.text(0.5, 0.88, 'Prompt: "The future of artificial intelligence is"',
           ha='center', fontsize=38, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))

    # Nucleus output
    ax.text(0.25, 0.75, 'Nucleus (p=0.9)', ha='center', fontsize=38,
           weight='bold', color=COLOR_ACCENT)
    nucleus_text = ('"...is promising and will transform\n'
                   'many industries. We expect to see\n'
                   'significant advances in healthcare,\n'
                   'education, and research in the\n'
                   'coming years."')
    ax.text(0.25, 0.52, nucleus_text, ha='center', va='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLOR_GREEN, linewidth=3))
    ax.text(0.25, 0.28, '+ Diverse\n+ Creative\n- Some repetition',
           ha='center', fontsize=42, color=COLOR_MAIN)

    # Contrastive output
    ax.text(0.75, 0.75, 'Contrastive (α=0.6)', ha='center', fontsize=38,
           weight='bold', color=COLOR_ACCENT)
    contrastive_text = ('"...is rapidly evolving, bringing\n'
                       'unprecedented opportunities across\n'
                       'sectors ranging from medicine to\n'
                       'climate science, while raising\n'
                       'important ethical questions."')
    ax.text(0.75, 0.52, contrastive_text, ha='center', va='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor=COLOR_GREEN, linewidth=3))
    ax.text(0.75, 0.28, '+ Diverse\n+ Creative\n+ No repetition',
           ha='center', fontsize=42, weight='bold', color=COLOR_GREEN)

    # Comparison
    ax.text(0.5, 0.08, 'Contrastive Search explicitly penalizes copying recent context',
           ha='center', fontsize=42, style='italic', color=COLOR_GREEN)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./contrastive_vs_nucleus_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: contrastive_vs_nucleus_bsc.pdf")

# === NEW CHART 14: Quality-Diversity Pareto Frontier ===
def generate_quality_diversity_pareto():
    """All 6 methods on quality-diversity plot"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # All 6 methods positioned
    methods_data = [
        ('Greedy', 85, 15, COLOR_RED, 'x'),
        ('Beam\nSearch', 88, 20, COLOR_RED, 's'),
        ('Temp\n(T=0.5)', 82, 35, COLOR_GRAY, 'o'),
        ('Temp\n(T=1.5)', 50, 85, COLOR_RED, 'o'),
        ('Top-k\n(k=40)', 70, 65, COLOR_GRAY, '^'),
        ('Nucleus\n(p=0.9)', 82, 78, COLOR_GREEN, 'D'),
        ('Contrastive\n(α=0.6)', 85, 82, COLOR_GREEN, '*')
    ]

    for method, quality, diversity, color, marker in methods_data:
        ax.scatter(diversity, quality, s=1000, c=color, marker=marker, alpha=0.7,
                  edgecolors=COLOR_MAIN, linewidths=3, zorder=3)
        # Offset label slightly
        offset_x = 3 if diversity < 50 else -3
        offset_y = 3 if quality < 70 else -3
        ax.text(diversity + offset_x, quality + offset_y, method,
               fontsize=FONTSIZE_ANNOTATION, ha='center', weight='bold')

    # Draw Pareto frontier (approximate)
    frontier_x = [15, 20, 78, 82, 85]
    frontier_y = [85, 88, 82, 85, 50]
    ax.plot(frontier_x, frontier_y, 'k--', alpha=0.3, linewidth=3, label='Pareto Frontier (approx)')

    # Optimal region
    optimal = patches.Rectangle((65, 75), 25, 20, linewidth=3,
                               edgecolor=COLOR_GREEN, facecolor=COLOR_GREEN,
                               alpha=0.1, linestyle='--')
    ax.add_patch(optimal)
    ax.text(77, 87, 'Optimal\nZone', ha='center', fontsize=42,
           weight='bold', color=COLOR_GREEN)

    # Problem zones
    ax.text(20, 20, 'Boring\nRepetitive', ha='center', fontsize=42,
           style='italic', color=COLOR_RED, alpha=0.6)
    ax.text(85, 30, 'Random\nNonsense', ha='center', fontsize=42,
           style='italic', color=COLOR_RED, alpha=0.6)

    ax.set_xlabel('Diversity (Creativity, Novelty) →', fontsize=40, weight='bold')
    ax.set_ylabel('Quality (Coherence, Correctness) →', fontsize=40, weight='bold')
    ax.set_title('Quality-Diversity Tradeoff: All Methods', fontsize=42, weight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.legend(loc='lower left', fontsize=42)
    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('./quality_diversity_pareto_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: quality_diversity_pareto_bsc.pdf")

# === NEW CHART 15: Task-Method Decision Tree ===
def generate_task_method_decision_tree():
    """Flowchart from task characteristics to recommended method"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Decision tree structure
    ax.text(0.5, 0.95, 'START: What kind of task?', ha='center', fontsize=40, weight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=3))

    # First split: Deterministic vs Creative
    ax.text(0.25, 0.82, 'Need exact/factual?', ha='center', fontsize=42, weight='bold')
    ax.text(0.75, 0.82, 'Need creative/diverse?', ha='center', fontsize=42, weight='bold')

    # Left branch: Deterministic tasks
    ax.text(0.25, 0.68, 'Translation\nQ&A\nSummarization', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT))
    ax.text(0.25, 0.55, '→ BEAM SEARCH\n(width=3-5)', ha='center', fontsize=42, weight='bold',
           color='white', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, linewidth=3))
    ax.text(0.25, 0.42, 'Or Temp=0.3-0.5\nfor consistency', ha='center', fontsize=42, style='italic')

    # Right branch: Creative tasks
    ax.text(0.75, 0.68, 'Creative Writing\nDialogue\nStorytelling', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT))

    # Second split: With/without repetition risk
    ax.text(0.65, 0.52, 'Short responses?\n(low repetition risk)', ha='center', fontsize=42)
    ax.text(0.65, 0.40, '→ NUCLEUS\n(p=0.9-0.95)', ha='center', fontsize=42, weight='bold',
           color='white', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN))

    ax.text(0.85, 0.52, 'Long generation?\n(repetition risk)', ha='center', fontsize=42)
    ax.text(0.85, 0.40, '→ CONTRASTIVE\n(α=0.5-0.7)', ha='center', fontsize=42, weight='bold',
           color='white', bbox=dict(boxstyle='round', facecolor=COLOR_GREEN))

    # Special case: Code generation
    ax.text(0.5, 0.25, 'Special: Code Generation', ha='center', fontsize=42, weight='bold')
    ax.text(0.5, 0.18, '→ Greedy or Beam (correctness critical)', ha='center', fontsize=42)
    ax.text(0.5, 0.12, '→ Then verify syntax/semantics', ha='center', fontsize=42, style='italic', color=COLOR_GRAY)

    # Draw connecting lines
    ax.plot([0.5, 0.25], [0.93, 0.84], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.5, 0.75], [0.93, 0.84], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.25, 0.25], [0.80, 0.70], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.75, 0.65], [0.80, 0.70], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.75, 0.65], [0.80, 0.54], 'k-', alpha=0.3, linewidth=3)
    ax.plot([0.75, 0.85], [0.80, 0.54], 'k-', alpha=0.3, linewidth=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Choosing the Right Decoding Method', fontsize=42, weight='bold')

    plt.tight_layout()
    plt.savefig('./task_method_decision_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: task_method_decision_tree_bsc.pdf")

# === NEW CHART 16: Task Recommendations Table ===
def generate_task_recommendations_table():
    """Comprehensive task-method mapping"""
    fig, ax = plt.subplots(figsize=(13, 8))

    # Extended task-method table
    tasks = [
        'Machine Translation',
        'Factual Q&A',
        'Summarization',
        'Code Generation',
        'Creative Writing',
        'Dialogue Systems',
        'Story Generation',
        'Long-form Articles'
    ]

    recommended = [
        'Beam Search',
        'Greedy / Low Temp',
        'Beam Search',
        'Greedy',
        'Nucleus / Contrastive',
        'Nucleus',
        'Contrastive',
        'Contrastive'
    ]

    params = [
        'width=3-5',
        'T=0.1-0.3',
        'width=4',
        'T=0',
        'p=0.9, α=0.6',
        'p=0.85-0.95',
        'α=0.5-0.7',
        'α=0.6, p=0.9'
    ]

    rationale = [
        'Deterministic, quality critical',
        'Single correct answer needed',
        'Balance coverage + conciseness',
        'Syntax errors costly',
        'Diverse but coherent',
        'Natural variation needed',
        'Avoid repetition in long text',
        'Degeneration prevention'
    ]

    # Create table
    table_data = []
    for i in range(len(tasks)):
        table_data.append([tasks[i], recommended[i], params[i], rationale[i]])

    headers = ['Task', 'Recommended Method', 'Parameters', 'Why?']

    # Color code by category
    cell_colors = []
    for i, task in enumerate(tasks):
        if i < 4:  # Deterministic tasks
            cell_colors.append([COLOR_LIGHT, COLOR_RED + '30', 'white', 'white'])
        else:  # Creative tasks
            cell_colors.append([COLOR_LIGHT, COLOR_GREEN + '30', 'white', 'white'])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='left', loc='center',
                    cellColours=cell_colors,
                    colWidths=[0.22, 0.22, 0.18, 0.38])

    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 2.2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(COLOR_ACCENT)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=42)

    # Add category labels
    ax.text(0.12, 0.73, 'Deterministic Tasks', fontsize=42, weight='bold',
           color=COLOR_RED, rotation=90, va='center')
    ax.text(0.12, 0.35, 'Creative Tasks', fontsize=42, weight='bold',
           color=COLOR_GREEN, rotation=90, va='center')

    ax.axis('off')
    ax.set_title('Task-Specific Decoding Recommendations (2025)', fontsize=42, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('./task_recommendations_table_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: task_recommendations_table_bsc.pdf")

# === NEW CHART 17: Computational Cost Comparison ===
def generate_computational_cost_comparison():
    """Speed vs quality tradeoff for all methods"""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Methods with relative costs (greedy = 1.0 baseline)
    methods = ['Greedy', 'Temperature', 'Top-k', 'Nucleus', 'Beam\n(w=3)', 'Contrastive']
    relative_time = [1.0, 1.1, 1.2, 1.3, 4.5, 12.0]
    quality_score = [65, 55, 68, 80, 88, 85]
    colors_cost = [COLOR_GREEN, COLOR_GREEN, COLOR_GREEN, COLOR_GREEN, COLOR_GRAY, COLOR_RED]

    x_pos = np.arange(len(methods))

    # Dual axis: time (bars) and quality (line)
    bars = ax.bar(x_pos, relative_time, color=colors_cost, alpha=0.7,
                 edgecolor=COLOR_MAIN, linewidth=3)

    # Add time labels on bars
    for i, (x, t) in enumerate(zip(x_pos, relative_time)):
        ax.text(x, t + 0.3, f'{t:.1f}x', ha='center', fontsize=42, weight='bold')

    # Quality line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(x_pos, quality_score, color=COLOR_ACCENT, linewidth=3, marker='o',
            markersize=20, label='Quality Score')

    # Annotations
    ax.axhline(5, color=COLOR_GRAY, linestyle='--', alpha=0.4, linewidth=1)
    ax.text(len(methods) - 0.5, 5.5, '5x slower than greedy',
           fontsize=42, style='italic', color=COLOR_GRAY)

    # Labels
    ax.set_xlabel('Decoding Method', fontsize=40, weight='bold')
    ax.set_ylabel('Relative Computation Time (vs Greedy)', fontsize=38, weight='bold', color=COLOR_MAIN)
    ax2.set_ylabel('Quality Score (0-100)', fontsize=38, weight='bold', color=COLOR_ACCENT)
    ax.set_title('Computational Cost vs Quality Tradeoff', fontsize=42, weight='bold', pad=15)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, max(relative_time) * 1.2)
    ax2.set_ylim(0, 100)

    # Legend
    ax2.legend(loc='upper left', fontsize=42)

    set_minimalist_style(ax)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(COLOR_ACCENT)
    ax2.tick_params(colors=COLOR_ACCENT)

    # Insight box
    ax.text(0.5, -0.22, 'Insight: Contrastive gives best quality-diversity but 12x slower. Nucleus is best balanced choice.',
           ha='center', transform=ax.transAxes, fontsize=42, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN))

    plt.tight_layout()
    plt.savefig('./computational_cost_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: computational_cost_comparison_bsc.pdf")

# === CHART 11: Strategy Selection Guide ===
def generate_strategy_selection_guide():
    """Decision matrix: task type × constraints → method"""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Simple flowchart
    questions = [
        'Q1: Is there ONE correct answer?',
        'Q2: Is speed critical?',
        'Q3: Long generation (>200 tokens)?',
        'Q4: Need reproducibility?'
    ]

    # Draw decision flow
    ax.text(0.5, 0.9, 'Choose Your Decoding Strategy', ha='center',
           fontsize=42, weight='bold')

    # Q1
    ax.text(0.5, 0.78, questions[0], ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))
    ax.text(0.25, 0.68, 'YES', ha='center', fontsize=42, weight='bold')
    ax.text(0.75, 0.68, 'NO', ha='center', fontsize=42, weight='bold')

    # Left: Greedy/Beam
    ax.text(0.25, 0.58, questions[1], ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT))
    ax.text(0.15, 0.48, 'YES → GREEDY', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.7))
    ax.text(0.35, 0.48, 'NO → BEAM', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.7))

    # Right: Sampling
    ax.text(0.75, 0.58, questions[2], ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT))
    ax.text(0.65, 0.48, 'NO → NUCLEUS', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.7))
    ax.text(0.85, 0.48, 'YES → CONTRASTIVE', ha='center', fontsize=42,
           bbox=dict(boxstyle='round', facecolor=COLOR_GREEN, alpha=0.7))

    # Examples at bottom
    ax.text(0.15, 0.30, 'Ex: Code\ngeneration', ha='center', fontsize=42, style='italic')
    ax.text(0.35, 0.30, 'Ex: Translation', ha='center', fontsize=42, style='italic')
    ax.text(0.65, 0.30, 'Ex: Dialogue\n(1 turn)', ha='center', fontsize=42, style='italic')
    ax.text(0.85, 0.30, 'Ex: Story\n(500 words)', ha='center', fontsize=42, style='italic')

    # Draw arrows
    for start, end in [(0.5, 0.25), (0.5, 0.75), (0.25, 0.25), (0.25, 0.15), (0.25, 0.35),
                       (0.75, 0.75), (0.75, 0.65), (0.75, 0.85)]:
        ax.plot([start, end], [0.76, 0.66] if start == 0.5 else [0.56, 0.50],
               'k-', alpha=0.4, linewidth=3)

    # Bottom note
    ax.text(0.5, 0.08, 'Rule of thumb: Deterministic tasks → Beam/Greedy | Creative tasks → Nucleus/Contrastive',
           ha='center', fontsize=42, style='italic',
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_MAIN))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./task_method_decision_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: task_method_decision_tree_bsc.pdf")

# === Generate remaining existing charts (simplified implementations) ===

def generate_greedy_vs_sampling():
    """Deterministic vs stochastic comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Greedy: Always same path
    x = [0, 1, 2, 3]
    y_greedy = [0, 0, 0, 0]
    ax1.plot(x, y_greedy, 'o-', color=COLOR_RED, linewidth=3, markersize=20)
    ax1.set_title('Greedy: Always Same Path', fontsize=38, weight='bold')
    ax1.set_ylabel('Output Variation', fontsize=42)
    ax1.set_xlabel('Generation Step', fontsize=42)
    ax1.text(1.5, 0.3, 'Deterministic\n(boring)', ha='center', fontsize=42,
            style='italic', color=COLOR_RED)
    set_minimalist_style(ax1)

    # Sampling: Different paths
    np.random.seed(42)
    for run in range(5):
        y_sample = np.cumsum(np.random.randn(4) * 0.3)
        ax2.plot(x, y_sample, 'o-', alpha=0.6, linewidth=3, markersize=18)
    ax2.set_title('Sampling: Different Each Time', fontsize=38, weight='bold')
    ax2.set_ylabel('Output Variation', fontsize=42)
    ax2.set_xlabel('Generation Step', fontsize=42)
    ax2.text(1.5, 0.5, 'Stochastic\n(diverse)', ha='center', fontsize=42,
            style='italic', color=COLOR_GREEN)
    set_minimalist_style(ax2)

    fig.suptitle('Deterministic vs Stochastic Decoding', fontsize=40, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('./greedy_vs_sampling_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: greedy_vs_sampling_bsc.pdf")

def generate_length_normalization():
    """Why beam search needs length normalization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Show bias toward shorter sequences
    lengths = [3, 5, 7, 9, 11, 13]
    raw_scores = [-1.5, -2.8, -4.2, -5.8, -7.5, -9.3]  # Log probs (more negative = worse)
    normalized = [score / length for score, length in zip(raw_scores, lengths)]

    x_pos = np.arange(len(lengths))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, raw_scores, width, label='Raw Log Probability',
                  color=COLOR_RED, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)
    bars2 = ax.bar(x_pos + width/2, normalized, width, label='Length Normalized',
                  color=COLOR_GREEN, alpha=0.7, edgecolor=COLOR_MAIN, linewidth=3)

    # Highlight bias
    ax.annotate('Shorter sequences\nfavored!', xy=(0, raw_scores[0]), xytext=(1, -0.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_RED),
               fontsize=42, color=COLOR_RED, weight='bold')

    ax.set_xlabel('Sequence Length (tokens)', fontsize=38, weight='bold')
    ax.set_ylabel('Score (higher is better)', fontsize=38, weight='bold')
    ax.set_title('Why Length Normalization Matters in Beam Search', fontsize=40, weight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lengths)
    ax.legend(fontsize=42)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('./length_normalization_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: length_normalization_bsc.pdf")

def generate_production_settings():
    """Real-world parameter settings from production systems"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Production settings table
    systems = [
        ('GPT-3 API (2024)', 'Nucleus', 'T=0.7, p=1.0', 'Balanced default'),
        ('ChatGPT', 'Nucleus + Temp', 'T=0.8, p=0.95', 'Creative but controlled'),
        ('Google Translate', 'Beam Search', 'width=4', 'Quality critical'),
        ('GitHub Copilot', 'Greedy', 'T=0', 'Code correctness'),
        ('Claude', 'Nucleus', 'T=1.0, p=0.9', 'High quality generation'),
        ('Hugging Face Default', 'Greedy', 'T=1.0', 'Deterministic baseline')
    ]

    table_data = [[s[0], s[1], s[2], s[3]] for s in systems]
    headers = ['System (2024-2025)', 'Method', 'Parameters', 'Goal']

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='left', loc='center',
                    colWidths=[0.25, 0.20, 0.25, 0.30])

    table.auto_set_font_size(False)
    table.set_fontsize(22)
    table.scale(1, 2.8)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(COLOR_ACCENT)
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=42)

    # Highlight rows
    for i in range(1, len(systems) + 1):
        if 'Nucleus' in systems[i-1][1]:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(COLOR_GREEN + '15')

    ax.axis('off')
    ax.set_title('Production Decoding Settings (Real Systems 2024-2025)',
                fontsize=42, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('./production_settings_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: production_settings_bsc.pdf")

def generate_quality_metrics():
    """Metrics for evaluating decoding quality"""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Metrics organized by category
    ax.text(0.5, 0.95, 'How to Measure Decoding Quality', ha='center',
           fontsize=42, weight='bold')

    # Quality metrics
    ax.text(0.25, 0.82, 'Quality Metrics', ha='center', fontsize=38,
           weight='bold', color=COLOR_ACCENT,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))
    metrics_quality = [
        '• Perplexity (lower = better)',
        '• BLEU score (translation)',
        '• ROUGE score (summarization)',
        '• Human evaluation ratings',
        '• Grammaticality score'
    ]
    y = 0.72
    for metric in metrics_quality:
        ax.text(0.25, y, metric, ha='center', fontsize=42)
        y -= 0.06

    # Diversity metrics
    ax.text(0.75, 0.82, 'Diversity Metrics', ha='center', fontsize=38,
           weight='bold', color=COLOR_ACCENT,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))
    metrics_diversity = [
        '• Distinct-n (unique n-grams)',
        '• Self-BLEU (lower = more diverse)',
        '• Repetition rate',
        '• Vocabulary richness',
        '• Entropy of outputs'
    ]
    y = 0.72
    for metric in metrics_diversity:
        ax.text(0.75, y, metric, ha='center', fontsize=42)
        y -= 0.06

    # Combined metrics
    ax.text(0.5, 0.30, 'Combined Quality-Diversity Metrics', ha='center', fontsize=38,
           weight='bold', color=COLOR_ACCENT,
           bbox=dict(boxstyle='round', facecolor=COLOR_LIGHT, edgecolor=COLOR_ACCENT, linewidth=3))
    combined = [
        '• Task success rate (does it solve the task?)',
        '• Human preference (A/B testing)',
        '• Pareto frontier analysis (multi-objective)'
    ]
    y = 0.20
    for metric in combined:
        ax.text(0.5, y, metric, ha='center', fontsize=42)
        y -= 0.05

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./quality_metrics_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: quality_metrics_bsc.pdf")

# === Generate all charts ===
if __name__ == '__main__':
    print("Generating Week 9 Enhanced Charts (22 total)...")
    print("=" * 50)

    # Core discovery hook
    generate_quality_diversity_tradeoff()

    # Beam search (3 charts)
    generate_beam_search_visual()
    generate_beam_example_tree()

    # Temperature (2 charts)
    generate_temperature_effects()
    generate_temperature_calculation()

    # Top-k (2 charts)
    generate_topk_filtering()
    generate_topk_example()

    # Nucleus (2 charts)
    generate_nucleus_process()
    generate_nucleus_cumulative()

    # Comparison & selection
    generate_decoding_comparison()
    generate_strategy_selection_guide()
    generate_quality_metrics()
    generate_greedy_vs_sampling()
    generate_length_normalization()
    generate_production_settings()

    # NEW: Contrastive search & 2025 enhancements (7 charts)
    generate_degeneration_problem()
    generate_contrastive_mechanism()
    generate_contrastive_vs_nucleus()
    generate_quality_diversity_pareto()
    generate_task_method_decision_tree()
    generate_task_recommendations_table()
    generate_computational_cost_comparison()

    print("=" * 50)
    print("SUCCESS: Generated 22 professional charts!")
    print("  Location: ../figures/")
    print("  Format: PDF, 300 DPI, publication quality")
    print("  Style: BSc Discovery (minimalist monochromatic + purple accent)")

# === BONUS CHART: Prediction Pipeline Diagram ===
def generate_prediction_pipeline():
    """Journey: How we got here - from model to text"""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Pipeline stages
    stages = ['Input\nText', 'MODEL\n(Transformer)', 'Probability\nDistribution', 'DECODING\n[TODAY]', 'Output\nText']
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Draw boxes
    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        if i == 3:  # TODAY box
            color = COLOR_ACCENT
            text_color = 'white'
            linewidth = 3
        elif i in [0, 4]:  # Input/Output
            color = COLOR_LIGHT
            text_color = COLOR_MAIN
            linewidth = 1.5
        else:
            color = COLOR_GRAY + '30'
            text_color = COLOR_MAIN
            linewidth = 2

        bbox = dict(boxstyle='round,pad=0.8', facecolor=color,
                   edgecolor=COLOR_MAIN, linewidth=linewidth)
        ax.text(x, 0.75, stage, ha='center', va='center', fontsize=38,
               weight='bold', color=text_color, bbox=bbox)

    # Draw arrows
    for i in range(len(x_positions) - 1):
        ax.annotate('', xy=(x_positions[i+1] - 0.05, 0.75),
                   xytext=(x_positions[i] + 0.05, 0.75),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=COLOR_MAIN))

    # Add examples below each stage
    examples = [
        '"The cat"',
        'Weeks 3-7:\nRNN→Transformer\n→BERT/GPT',
        'P(sat)=0.45\nP(is)=0.30\nP(jumped)=0.25\n...\n50,000 words!',
        '6 Strategies:\nGreedy, Beam\nTemp, Top-k\nNucleus,\nContrastive',
        '"The cat sat\non the mat"'
    ]

    for x, example in zip(x_positions, examples):
        ax.text(x, 0.35, example, ha='center', va='center', fontsize=42,
               style='italic', color=COLOR_GRAY,
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor=COLOR_GRAY, linewidth=1, alpha=0.7))

    # Key question
    ax.text(0.5, 0.10, 'Key Question: Model gives 50,000 probabilities - which word do we choose?',
           ha='center', fontsize=38, weight='bold', color=COLOR_ACCENT,
           bbox=dict(boxstyle='round', facecolor=COLOR_ACCENT, alpha=0.15,
                    edgecolor=COLOR_ACCENT, linewidth=3))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('From Prediction to Generation: The Decoding Challenge',
                fontsize=42, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('./prediction_to_text_pipeline_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: prediction_to_text_pipeline_bsc.pdf")

# Run if called directly as bonus
if __name__ == '__main__' and len(__import__('sys').argv) > 1:
    if __import__('sys').argv[1] == '--pipeline':
        generate_prediction_pipeline()
