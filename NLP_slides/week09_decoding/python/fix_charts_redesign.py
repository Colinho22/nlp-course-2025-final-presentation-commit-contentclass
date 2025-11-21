#!/usr/bin/env python3
"""
Fix and generate charts for Week 9 complete redesign
1. Fix quality_diversity_tradeoff (consistent fonts, all 6 methods labeled)
2. Create vocabulary probability bar chart (slide 2)
3. Fix beam_example_tree (reduce size, add t=0)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Ensure output directory exists
os.makedirs('../figures', exist_ok=True)

# Template Beamer Final color scheme
COLOR_MAIN = '#333333'
COLOR_ACCENT = '#3333B2'
COLOR_LAVENDER = '#ADADC0'
COLOR_LAVENDER2 = '#C1C1E8'
COLOR_BLUE = '#0066CC'
COLOR_GRAY = '#7F7F7F'
COLOR_LIGHT = '#F0F0F0'
COLOR_RED = '#D62728'
COLOR_GREEN = '#2CA02C'
COLOR_ORANGE = '#FF7F0E'

# Consistent font sizes (FIXED - smaller for text labels)
FONTSIZE_TITLE = 28
FONTSIZE_LABEL = 22
FONTSIZE_TEXT = 18  # For method labels inside boxes
FONTSIZE_TICK = 18
FONTSIZE_ANNOTATION = 20

def set_minimalist_style(ax):
    """Apply template beamer final styling"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_ACCENT)
    ax.spines['bottom'].set_color(COLOR_ACCENT)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(colors=COLOR_MAIN, which='both', labelsize=FONTSIZE_TICK, width=2, length=6)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=2, color=COLOR_LAVENDER)
    ax.set_facecolor('white')

# === FIX 1: Quality-Diversity Tradeoff ===
def fix_quality_diversity_tradeoff():
    """FIXED: 6 methods, all properly labeled with consistent fonts"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # All 6 methods with proper positions
    methods = ['Greedy', 'Beam\nSearch', 'Temp\n(low)', 'Temp\n(high)', 'Top-k', 'Nucleus']
    quality = [85, 88, 70, 45, 65, 80]
    diversity = [15, 20, 60, 95, 70, 75]
    colors_map = [COLOR_RED, COLOR_RED, COLOR_GRAY, COLOR_RED, COLOR_GRAY, COLOR_GREEN]

    # Plot all 6 methods
    for i, (method, q, d, c) in enumerate(zip(methods, quality, diversity, colors_map)):
        # Draw box (scatter point)
        ax.scatter(d, q, s=2000, c=c, alpha=0.7, edgecolors=COLOR_MAIN, linewidths=2, zorder=2)
        # Add text label (FIXED - smaller font, ensured visibility)
        ax.text(d, q, method, ha='center', va='center',
                fontsize=FONTSIZE_TEXT, weight='bold', color='white', zorder=3)

    # Sweet spot annotation
    ax.annotate('Sweet\nSpot', xy=(75, 80), xytext=(85, 88),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN),
                fontsize=FONTSIZE_ANNOTATION, weight='bold', color=COLOR_GREEN)

    ax.set_xlabel('Diversity (Creativity)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_ylabel('Quality (Coherence)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_title('The Quality-Diversity Tradeoff', fontsize=FONTSIZE_TITLE, weight='bold', pad=20)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Problem zones
    ax.axhspan(0, 50, alpha=0.05, color=COLOR_RED)
    ax.axvspan(0, 30, alpha=0.05, color=COLOR_RED)

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/quality_diversity_tradeoff_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[FIXED] quality_diversity_tradeoff_bsc.pdf (6 methods, consistent fonts)")

# === CREATE: Vocabulary Probability Bar Chart (Slide 2) ===
def create_vocabulary_probability_chart():
    """NEW: Bar chart showing probability distribution over vocabulary"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Example: "The cat " probabilities
    words = ['sat', 'is', 'jumped', 'ran', 'slept', 'played', '...', '(49,994 more)']
    probs = [0.45, 0.30, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02]

    # Create horizontal bars
    y_positions = np.arange(len(words))
    bars = ax.barh(y_positions, probs, color=COLOR_ACCENT, edgecolor=COLOR_MAIN, linewidth=2, alpha=0.8)

    # Highlight top 3
    for i in range(3):
        bars[i].set_color(COLOR_GREEN)
        bars[i].set_alpha(0.9)

    # Fade last two (tail)
    bars[-2].set_alpha(0.3)
    bars[-1].set_alpha(0.1)

    # Add probability values
    for i, (prob, word) in enumerate(zip(probs, words)):
        if i < len(words) - 1:
            ax.text(prob + 0.02, i, f'{prob:.2f}', va='center',
                    fontsize=FONTSIZE_ANNOTATION, weight='bold', color=COLOR_MAIN)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(words, fontsize=FONTSIZE_LABEL)
    ax.set_xlabel('Probability P(word | "The cat ")', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_title('The Decoding Challenge: Choose From 50,000 Words',
                 fontsize=FONTSIZE_TITLE, weight='bold', pad=20)

    ax.set_xlim(0, 0.55)
    ax.invert_yaxis()  # Highest probability at top

    # Add annotation
    ax.text(0.52, 7, '← Tail (99.9% of vocab)', va='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_GRAY)

    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/vocabulary_probability_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] vocabulary_probability_bsc.pdf (slide 2 chart)")

# === FIX 3: Beam Search Example Tree (add t=0, reduce size) ===
def fix_beam_example_tree():
    """FIXED: Smaller size, added t=0 label at root"""
    fig, ax = plt.subplots(figsize=(10, 6))  # Reduced from potentially larger

    # Time steps
    times = ['t=0', 't=1', 't=2', 't=3']

    # Root node with t=0 label
    ax.scatter(0.5, 1.0, s=1200, c=COLOR_ACCENT, edgecolors=COLOR_MAIN, linewidths=2, zorder=3)
    ax.text(0.5, 1.0, 'START', ha='center', va='center', fontsize=18, weight='bold', color='white')
    ax.text(0.5, 1.08, 't=0', ha='center', va='bottom', fontsize=16, weight='bold', color=COLOR_ACCENT)

    # t=1: 3 beams
    t1_x = [0.25, 0.5, 0.75]
    t1_words = ['The', 'A', 'This']
    t1_probs = [0.45, 0.30, 0.25]

    for i, (x, word, prob) in enumerate(zip(t1_x, t1_words, t1_probs)):
        color = COLOR_GREEN if i < 3 else COLOR_GRAY
        ax.scatter(x, 0.65, s=900, c=color, alpha=0.8, edgecolors=COLOR_MAIN, linewidths=2, zorder=2)
        ax.plot([0.5, x], [1.0, 0.65], 'k-', alpha=0.3, linewidth=1.5, zorder=1)
        ax.text(x, 0.65, f'{word}\n{prob:.2f}', ha='center', va='center',
                fontsize=14, weight='bold', color='white')

    # t=2: Keep top 3, expand each
    t2_candidates = [
        (0.15, 'The cat', 0.20, True), (0.25, 'The dog', 0.15, False), (0.35, 'The bird', 0.10, False),
        (0.40, 'A dog', 0.13, True), (0.50, 'A cat', 0.12, False), (0.60, 'A bird', 0.05, False),
        (0.65, 'This is', 0.14, True), (0.75, 'This was', 0.08, False), (0.85, 'This will', 0.03, False)
    ]

    for x, text, prob, keep in t2_candidates:
        color = COLOR_GREEN if keep else COLOR_GRAY
        alpha = 0.8 if keep else 0.3
        size = 700 if keep else 500
        ax.scatter(x, 0.35, s=size, c=color, alpha=alpha, edgecolors=COLOR_MAIN, linewidths=2, zorder=2)
        # Draw connection to parent
        parent_x = t1_x[int(x * 1.2)]  # Simplified parent connection
        if x < 0.37:
            parent_x = t1_x[0]
        elif x < 0.62:
            parent_x = t1_x[1]
        else:
            parent_x = t1_x[2]
        ax.plot([parent_x, x], [0.65, 0.35], 'k-', alpha=0.2, linewidth=1, zorder=1)

        if keep:
            ax.text(x, 0.35, f'{text}\n{prob:.2f}', ha='center', va='center',
                    fontsize=11, weight='bold', color='white')

    # Annotations
    ax.text(0.5, 0.92, times[0], ha='center', fontsize=14, weight='bold', color=COLOR_ACCENT)
    ax.text(0.5, 0.57, times[1], ha='center', fontsize=14, weight='bold', color=COLOR_ACCENT)
    ax.text(0.5, 0.27, times[2], ha='center', fontsize=14, weight='bold', color=COLOR_ACCENT)

    # Legend
    ax.text(0.5, 0.05, 'Green = Kept (top 3) | Gray = Pruned', ha='center',
            fontsize=14, style='italic', color=COLOR_MAIN)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.axis('off')
    ax.set_title('Beam Search: Step-by-Step (width=3)', fontsize=FONTSIZE_TITLE, weight='bold')

    plt.tight_layout()
    plt.savefig('../figures/beam_example_tree_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[FIXED] beam_example_tree_bsc.pdf (added t=0, reduced size)")

# === MAIN EXECUTION ===
if __name__ == '__main__':
    print("Fixing and generating charts for Week 9 redesign...")
    print()

    fix_quality_diversity_tradeoff()
    create_vocabulary_probability_chart()
    fix_beam_example_tree()

    print()
    print("All 3 charts fixed/created successfully!")
    print("[FIXED] quality_diversity_tradeoff_bsc.pdf - fonts fixed, all 6 methods labeled")
    print("[CREATED] vocabulary_probability_bsc.pdf - NEW slide 2 chart")
    print("[FIXED] beam_example_tree_bsc.pdf - with t=0, smaller size")
