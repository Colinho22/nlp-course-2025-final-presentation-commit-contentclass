#!/usr/bin/env python3
"""
Fix and generate final charts for Week 9 presentation
1. Generate 6 problem output charts with integrated text
2. Complete redesign of beam search visualization
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

# Font sizes
FONTSIZE_TITLE = 24
FONTSIZE_LABEL = 20
FONTSIZE_TEXT = 16
FONTSIZE_SMALL = 14
FONTSIZE_ANNOTATION = 18

def set_minimalist_style(ax):
    """Apply template beamer final styling"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_ACCENT)
    ax.spines['bottom'].set_color(COLOR_ACCENT)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(colors=COLOR_MAIN, which='both', labelsize=FONTSIZE_SMALL, width=2, length=6)
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=1, color=COLOR_LAVENDER)
    ax.set_facecolor('white')

# === PROBLEM OUTPUT CHARTS WITH INTEGRATED TEXT ===

def generate_problem1_repetition():
    """Problem 1: Repetition loops with full output text"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Main output text box
    output_text = "The city is a major city in the city in the city..."

    # Create visual representation of repetition
    words = ["The", "city", "is", "a", "major", "city", "in", "the", "city", "in", "the", "city", "..."]
    x_positions = np.arange(len(words))

    # Highlight repeated phrase
    for i in range(len(words)):
        if words[i] == "city":
            ax.bar(i, 1, color=COLOR_RED, alpha=0.7, width=0.8)
        elif words[i] in ["in", "the"]:
            ax.bar(i, 1, color=COLOR_ORANGE, alpha=0.5, width=0.8)
        else:
            ax.bar(i, 1, color=COLOR_GRAY, alpha=0.3, width=0.8)

    # Add word labels
    for i, word in enumerate(words):
        ax.text(i, 0.5, word, ha='center', va='center',
                fontsize=FONTSIZE_TEXT, weight='bold', color='white')

    # Add loop indicator arrows
    ax.annotate('', xy=(10.5, 1.2), xytext=(5.5, 1.2),
                arrowprops=dict(arrowstyle='<->', lw=3, color=COLOR_RED))
    ax.text(8, 1.35, 'LOOP', ha='center', fontsize=FONTSIZE_LABEL,
            weight='bold', color=COLOR_RED)

    # Add full output text at bottom
    ax.text(6, -0.5, f'Output: "{output_text}"', ha='center', va='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_MAIN)

    # Add problem label
    ax.text(6, 1.8, 'Greedy Decoding Gets Stuck', ha='center',
            fontsize=FONTSIZE_TITLE, weight='bold', color=COLOR_MAIN)

    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(-0.7, 2)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/problem1_repetition_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem1_repetition_output_bsc.pdf")

def generate_problem2_diversity():
    """Problem 2: No diversity/nonsense words"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show multiple outputs with no diversity
    outputs = [
        "The weather is zxqp today",
        "I went to the flurb yesterday",
        "She likes to eat qwerty food",
        "The glorp is very blorptastic"
    ]

    y_pos = np.arange(len(outputs))

    # Create boxes for each output
    for i, output in enumerate(outputs):
        # Background box
        rect = patches.FancyBboxPatch((0.1, i - 0.35), 10, 0.7,
                                      boxstyle="round,pad=0.05",
                                      facecolor=COLOR_LIGHT,
                                      edgecolor=COLOR_RED, linewidth=2)
        ax.add_patch(rect)

        # Highlight nonsense words in red
        words = output.split()
        x_offset = 0.3
        for word in words:
            if word in ['zxqp', 'flurb', 'qwerty', 'glorp', 'blorptastic']:
                ax.text(x_offset, i, word, fontsize=FONTSIZE_TEXT,
                       color=COLOR_RED, weight='bold', va='center')
            else:
                ax.text(x_offset, i, word, fontsize=FONTSIZE_TEXT,
                       color=COLOR_MAIN, va='center')
            x_offset += 1.2

    # Add title
    ax.text(5, 4.5, 'High Temperature Creates Nonsense', ha='center',
            fontsize=FONTSIZE_TITLE, weight='bold', color=COLOR_MAIN)

    # Add problem indicator
    ax.text(5, -1, 'Generated words not in vocabulary!', ha='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_RED)

    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/problem2_diversity_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem2_diversity_output_bsc.pdf")

def generate_problem3_balance():
    """Problem 3: Boring/same outputs"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show 100 requests with same output
    num_requests = 10
    same_output = "The weather is nice today."

    # Visual grid of identical outputs
    for i in range(num_requests):
        y = i // 2
        x = (i % 2) * 5

        # Box for each output
        rect = patches.FancyBboxPatch((x + 0.2, y * 0.8 - 0.3), 4.5, 0.6,
                                      boxstyle="round,pad=0.02",
                                      facecolor=COLOR_LAVENDER2,
                                      edgecolor=COLOR_GRAY, linewidth=1)
        ax.add_patch(rect)

        # Request number
        ax.text(x + 0.5, y * 0.8, f'#{i+1}:', fontsize=FONTSIZE_SMALL,
                color=COLOR_GRAY, va='center', weight='bold')

        # Same output
        ax.text(x + 2.7, y * 0.8, same_output, fontsize=FONTSIZE_TEXT,
                color=COLOR_MAIN, va='center')

    # Add large "100x" indicator
    ax.text(10.5, 2, '100x', fontsize=36, weight='bold',
            color=COLOR_RED, rotation=15, alpha=0.7)

    # Title
    ax.text(5, 5, 'Zero Creativity: Always Same Output', ha='center',
            fontsize=FONTSIZE_TITLE, weight='bold', color=COLOR_MAIN)

    # Problem indicator
    ax.text(5, -1, f'Asked 100 times → Always: "{same_output}"', ha='center',
            fontsize=FONTSIZE_ANNOTATION, style='italic', color=COLOR_RED)

    ax.set_xlim(0, 11)
    ax.set_ylim(-1.5, 5.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/problem3_balance_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem3_balance_output_bsc.pdf")

def generate_problem4_paths():
    """Problem 4: Missed better paths"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Show path tree with missed opportunity
    # Root
    ax.scatter(6, 4, s=800, c=COLOR_ACCENT, edgecolors=COLOR_MAIN, linewidth=2, zorder=3)
    ax.text(6, 4, 'The', ha='center', va='center', fontsize=16, weight='bold', color='white')

    # Level 1 choices
    choices_l1 = [('cat', 0.45, 3, 3, COLOR_GREEN),
                  ('dog', 0.35, 9, 3, COLOR_GRAY)]

    for word, prob, x, y, color in choices_l1:
        ax.scatter(x, y, s=600, c=color, alpha=0.8, edgecolors=COLOR_MAIN, linewidth=2, zorder=2)
        ax.text(x, y, f'{word}\n{prob:.2f}', ha='center', va='center',
                fontsize=14, color='white', weight='bold')
        ax.plot([6, x], [4, y], 'k-', alpha=0.3, linewidth=2)

    # Level 2 - show missed path
    # From "cat" (chosen but leads to dead end)
    ax.scatter(2, 2, s=500, c=COLOR_RED, alpha=0.8, edgecolors=COLOR_MAIN, linewidth=2)
    ax.text(2, 2, 'sat\n0.30', ha='center', va='center', fontsize=12, color='white')
    ax.plot([3, 2], [3, 2], 'k-', alpha=0.3, linewidth=2)

    ax.scatter(4, 2, s=500, c=COLOR_RED, alpha=0.8, edgecolors=COLOR_MAIN, linewidth=2)
    ax.text(4, 2, 'is\n0.25', ha='center', va='center', fontsize=12, color='white')
    ax.plot([3, 4], [3, 2], 'k-', alpha=0.3, linewidth=2)

    # From "dog" (not explored but better)
    ax.scatter(8, 2, s=500, c=COLOR_GREEN, alpha=0.4, edgecolors=COLOR_MAIN, linewidth=2, linestyle='--')
    ax.text(8, 2, 'ran\n0.60!', ha='center', va='center', fontsize=12, color=COLOR_GREEN, weight='bold')
    ax.plot([9, 8], [3, 2], 'g--', alpha=0.5, linewidth=2)

    ax.scatter(10, 2, s=500, c=COLOR_GREEN, alpha=0.4, edgecolors=COLOR_MAIN, linewidth=2, linestyle='--')
    ax.text(10, 2, 'jumped\n0.55!', ha='center', va='center', fontsize=12, color=COLOR_GREEN, weight='bold')
    ax.plot([9, 10], [3, 2], 'g--', alpha=0.5, linewidth=2)

    # Add "MISSED!" annotation
    ax.annotate('BETTER PATH\nNOT EXPLORED!', xy=(9, 1.5), xytext=(6, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN),
                fontsize=FONTSIZE_LABEL, weight='bold', color=COLOR_GREEN, ha='center')

    # Add outputs
    ax.text(3, 0.8, 'Chosen: "The cat sat"', fontsize=FONTSIZE_TEXT, color=COLOR_RED, weight='bold')
    ax.text(9, 0.8, 'Better: "The dog ran"', fontsize=FONTSIZE_TEXT, color=COLOR_GREEN, weight='bold')

    # Title
    ax.text(6, 4.8, 'Greedy Misses Better Continuations', ha='center',
            fontsize=FONTSIZE_TITLE, weight='bold', color=COLOR_MAIN)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/problem4_paths_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem4_paths_output_bsc.pdf")

def generate_problem5_distribution():
    """Problem 5: Wrong probability distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Model's distribution (wrong)
    words1 = ['the', 'a', 'cat', 'dog', 'ran', 'jumped', 'quickly', '...']
    probs1 = [0.40, 0.30, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]

    bars1 = ax1.bar(range(len(words1)), probs1, color=COLOR_RED, alpha=0.7)
    bars1[0].set_color(COLOR_RED)
    bars1[0].set_alpha(0.9)

    ax1.set_xticks(range(len(words1)))
    ax1.set_xticklabels(words1, rotation=45, ha='right')
    ax1.set_ylabel('Probability', fontsize=FONTSIZE_LABEL)
    ax1.set_title('Model Distribution\n(Wrong for Context)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax1.set_ylim(0, 0.5)

    # Add problem indicator
    ax1.text(3.5, 0.35, 'Too much\nmass on\ncommon words!', ha='center',
            fontsize=FONTSIZE_SMALL, color=COLOR_RED, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLOR_RED))

    set_minimalist_style(ax1)

    # Right: True distribution (correct)
    words2 = ['quickly', 'gracefully', 'frantically', 'slowly', 'the', 'a', 'cat', '...']
    probs2 = [0.25, 0.20, 0.18, 0.15, 0.08, 0.06, 0.05, 0.03]

    bars2 = ax2.bar(range(len(words2)), probs2, color=COLOR_GREEN, alpha=0.7)
    for i in range(4):
        bars2[i].set_alpha(0.9)

    ax2.set_xticks(range(len(words2)))
    ax2.set_xticklabels(words2, rotation=45, ha='right')
    ax2.set_ylabel('Probability', fontsize=FONTSIZE_LABEL)
    ax2.set_title('True Distribution\n(Correct for Context)', fontsize=FONTSIZE_LABEL, weight='bold')
    ax2.set_ylim(0, 0.5)

    # Add correct indicator
    ax2.text(3.5, 0.35, 'Adverbs\nshould be\nmore likely!', ha='center',
            fontsize=FONTSIZE_SMALL, color=COLOR_GREEN, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLOR_GREEN))

    set_minimalist_style(ax2)

    # Overall title
    fig.suptitle('Context: "The cat ran ___"', fontsize=FONTSIZE_TITLE, weight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('../figures/problem5_distribution_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem5_distribution_output_bsc.pdf")

def generate_problem6_speed():
    """Problem 6: Too slow for real-time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Speed comparison bars
    methods = ['Greedy\n(Baseline)', 'Beam\n(k=5)', 'Beam\n(k=10)', 'Beam\n(k=50)']
    speeds = [10, 50, 100, 500]  # milliseconds
    colors = [COLOR_GREEN, COLOR_ORANGE, COLOR_RED, COLOR_RED]

    bars = ax.barh(range(len(methods)), speeds, color=colors, alpha=0.8)

    # Add speed values
    for i, (method, speed) in enumerate(zip(methods, speeds)):
        ax.text(speed + 20, i, f'{speed}ms', va='center',
                fontsize=FONTSIZE_TEXT, weight='bold')

        # Add latency indicator
        if speed > 100:
            ax.text(speed/2, i, 'TOO SLOW!', ha='center', va='center',
                    fontsize=12, color='white', weight='bold')

    # Add threshold line
    ax.axvline(x=100, color=COLOR_RED, linestyle='--', linewidth=2, alpha=0.7)
    ax.text(100, 3.5, 'Real-time threshold (100ms)',
            fontsize=FONTSIZE_SMALL, color=COLOR_RED, ha='center')

    # Add user experience indicators
    ax.text(600, 3, 'User gives up!', fontsize=FONTSIZE_LABEL,
            color=COLOR_RED, weight='bold', style='italic')
    ax.text(600, 2, '😕 Bad UX', fontsize=FONTSIZE_TEXT, color=COLOR_RED)

    ax.text(600, 0, '😊 Good UX', fontsize=FONTSIZE_TEXT, color=COLOR_GREEN)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=FONTSIZE_LABEL)
    ax.set_xlabel('Generation Time per Token', fontsize=FONTSIZE_LABEL, weight='bold')
    ax.set_title('Beam Search Makes Typing Assistants Unusable',
                 fontsize=FONTSIZE_TITLE, weight='bold')

    ax.set_xlim(0, 700)
    set_minimalist_style(ax)

    plt.tight_layout()
    plt.savefig('../figures/problem6_speed_output_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] problem6_speed_output_bsc.pdf")

# === COMPLETE REDESIGN: BEAM SEARCH VISUALIZATION ===

def generate_beam_search_redesign():
    """Complete redesign of beam search visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Time steps
    times = ['t=0', 't=1', 't=2', 't=3']
    time_x = [2, 6, 10, 14]

    # Draw time labels at top
    for t, x in zip(times, time_x):
        ax.text(x, 7.5, t, ha='center', fontsize=FONTSIZE_LABEL,
                weight='bold', color=COLOR_ACCENT)

    # t=0: Start token
    start_node = patches.Circle((2, 6), 0.4, facecolor=COLOR_ACCENT,
                                edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(start_node)
    ax.text(2, 6, 'START', ha='center', va='center',
            fontsize=14, weight='bold', color='white')

    # t=1: First expansion (beam width = 3)
    t1_words = ['The', 'A', 'This']
    t1_probs = [0.45, 0.30, 0.25]
    t1_y = [6.8, 6, 5.2]
    t1_nodes = []

    for word, prob, y in zip(t1_words, t1_probs, t1_y):
        # Node
        circle = patches.Circle((6, y), 0.35, facecolor=COLOR_GREEN,
                                alpha=0.8, edgecolor=COLOR_MAIN, linewidth=2)
        ax.add_patch(circle)
        ax.text(6, y, word, ha='center', va='center',
                fontsize=12, weight='bold', color='white')

        # Probability label
        ax.text(6, y - 0.6, f'p={prob:.2f}', ha='center',
                fontsize=10, color=COLOR_MAIN)

        # Edge from start
        ax.plot([2.4, 5.65], [6, y], 'k-', alpha=0.3, linewidth=1.5)

        t1_nodes.append((6, y, prob))

    # t=2: Expand each beam (3x3 = 9, keep top 3)
    t2_candidates = [
        ('The cat', 0.45*0.40, 7.2, True, 0),   # From "The"
        ('The dog', 0.45*0.30, 6.6, True, 0),    # From "The"
        ('The sun', 0.45*0.20, 6.0, False, 0),   # From "The" - pruned
        ('A dog', 0.30*0.50, 5.4, True, 1),      # From "A"
        ('A cat', 0.30*0.35, 4.8, False, 1),     # From "A" - pruned
        ('A bird', 0.30*0.15, 4.2, False, 1),    # From "A" - pruned
        ('This is', 0.25*0.60, 3.6, False, 2),   # From "This" - pruned
        ('This was', 0.25*0.25, 3.0, False, 2),  # From "This" - pruned
        ('This will', 0.25*0.15, 2.4, False, 2), # From "This" - pruned
    ]

    kept_nodes = []
    for text, score, y, keep, parent_idx in t2_candidates:
        parent_x, parent_y, parent_score = t1_nodes[parent_idx]

        if keep:
            # Kept node (larger, solid)
            circle = patches.Circle((10, y), 0.35, facecolor=COLOR_GREEN,
                                   alpha=0.9, edgecolor=COLOR_MAIN, linewidth=2)
            ax.add_patch(circle)
            ax.text(10, y, text.split()[1], ha='center', va='center',
                   fontsize=11, weight='bold', color='white')
            ax.text(10, y - 0.6, f'Σ={score:.3f}', ha='center',
                   fontsize=10, color=COLOR_MAIN, weight='bold')
            kept_nodes.append((10, y, score, text))
        else:
            # Pruned node (smaller, faded)
            circle = patches.Circle((10, y), 0.25, facecolor=COLOR_GRAY,
                                   alpha=0.3, edgecolor=COLOR_GRAY, linewidth=1)
            ax.add_patch(circle)
            ax.text(10, y, text.split()[1], ha='center', va='center',
                   fontsize=9, color=COLOR_GRAY, alpha=0.5)

        # Edge from parent
        ax.plot([parent_x + 0.35, 10 - 0.35], [parent_y, y],
               'k-' if keep else 'k--', alpha=0.3 if keep else 0.1,
               linewidth=1.5 if keep else 0.8)

    # Add pruning indicator
    ax.text(10, 1.8, 'Pruned\n(6 paths)', ha='center', fontsize=10,
            color=COLOR_GRAY, style='italic')
    ax.text(10, 8, 'Kept\n(top 3)', ha='center', fontsize=10,
            color=COLOR_GREEN, weight='bold')

    # t=3: Final expansion
    t3_finals = [
        ('The cat sat', 0.180*0.70, 7.0, 0),
        ('The dog ran', 0.135*0.80, 5.8, 1),
        ('A dog barked', 0.150*0.65, 4.6, 2),
    ]

    for i, (text, final_score, y, parent_idx) in enumerate(t3_finals):
        parent_x, parent_y, parent_score, parent_text = kept_nodes[parent_idx]

        # Final node
        if i == 0:  # Best path
            circle = patches.Circle((14, y), 0.4, facecolor=COLOR_BLUE,
                                   edgecolor=COLOR_MAIN, linewidth=3)
            ax.text(14, y - 0.7, 'BEST PATH', ha='center',
                   fontsize=10, color=COLOR_BLUE, weight='bold')
        else:
            circle = patches.Circle((14, y), 0.35, facecolor=COLOR_GREEN,
                                   alpha=0.7, edgecolor=COLOR_MAIN, linewidth=2)

        ax.add_patch(circle)
        ax.text(14, y, text.split()[2], ha='center', va='center',
               fontsize=11, weight='bold', color='white')
        ax.text(14, y + 0.5, f'Σ={final_score:.3f}', ha='center',
               fontsize=10, color=COLOR_MAIN, weight='bold')

        # Edge from parent
        ax.plot([parent_x + 0.35, 14 - 0.4], [parent_y, y],
               'b-' if i == 0 else 'k-', alpha=0.5 if i == 0 else 0.3,
               linewidth=2.5 if i == 0 else 1.5)

    # Add legend
    legend_y = 1.5
    ax.text(2, legend_y, 'Legend:', fontsize=12, weight='bold')

    # Kept node
    circle = patches.Circle((4, legend_y - 0.3), 0.2, facecolor=COLOR_GREEN,
                           alpha=0.9, edgecolor=COLOR_MAIN, linewidth=2)
    ax.add_patch(circle)
    ax.text(4.5, legend_y - 0.3, 'Kept (beam)', fontsize=10, va='center')

    # Pruned node
    circle = patches.Circle((7, legend_y - 0.3), 0.15, facecolor=COLOR_GRAY,
                           alpha=0.3, edgecolor=COLOR_GRAY, linewidth=1)
    ax.add_patch(circle)
    ax.text(7.5, legend_y - 0.3, 'Pruned', fontsize=10, va='center', color=COLOR_GRAY)

    # Best path
    circle = patches.Circle((10, legend_y - 0.3), 0.2, facecolor=COLOR_BLUE,
                           edgecolor=COLOR_MAIN, linewidth=3)
    ax.add_patch(circle)
    ax.text(10.5, legend_y - 0.3, 'Best path', fontsize=10, va='center', color=COLOR_BLUE, weight='bold')

    # Title
    ax.text(8, 8.5, 'Beam Search with k=3: Complete Example',
            ha='center', fontsize=FONTSIZE_TITLE, weight='bold')

    # Add statistics box
    stats_text = 'Paths explored: 9\nPaths kept: 3\nFinal winner: "The cat sat"'
    ax.text(1, 4, stats_text, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_LAVENDER2,
                     edgecolor=COLOR_ACCENT, linewidth=2))

    ax.set_xlim(0, 16)
    ax.set_ylim(1, 9)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('../figures/beam_search_example_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[CREATED] beam_search_example_bsc.pdf (complete redesign)")

# === MAIN EXECUTION ===
if __name__ == '__main__':
    print("Generating final Week 9 charts with integrated text...")
    print()

    # Generate all 6 problem charts
    generate_problem1_repetition()
    generate_problem2_diversity()
    generate_problem3_balance()
    generate_problem4_paths()
    generate_problem5_distribution()
    generate_problem6_speed()

    # Generate redesigned beam search
    generate_beam_search_redesign()

    print()
    print("All 7 charts generated successfully!")
    print("6 problem output charts with integrated text")
    print("1 completely redesigned beam search visualization")