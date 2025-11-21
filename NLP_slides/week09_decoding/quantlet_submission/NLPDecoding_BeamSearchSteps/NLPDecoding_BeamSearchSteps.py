"""
Generate step-by-step beam search visualization
Shows 4 steps with beam_size=3, including scoring and pruning
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_beam_step_by_step():
    """Create 4-step walkthrough of beam search algorithm."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Beam size
    beam_size = 3

    # Step 1: Initial expansions
    ax = axes[0]
    ax.set_title('Step 1: Generate First Word', fontsize=14, fontweight='bold')

    # Starting state
    ax.text(0.5, 0.9, 'START', ha='center', fontsize=12, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', edgecolor='black', linewidth=2))

    # Candidates
    candidates_1 = [
        ('The', 0.40),
        ('A', 0.30),
        ('I', 0.20),
        ('It', 0.10),
        ('One', 0.05),
    ]

    y_start = 0.65
    for i, (word, prob) in enumerate(candidates_1):
        x_pos = 0.15 + i * 0.17
        # Is it in beam?
        in_beam = i < beam_size
        color = 'lightgreen' if in_beam else 'lightcoral'
        edge_width = 2 if in_beam else 1
        alpha = 1.0 if in_beam else 0.5

        # Word box
        box = FancyBboxPatch((x_pos - 0.06, y_start - 0.05), 0.12, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black' if in_beam else 'gray',
                            linewidth=edge_width,
                            alpha=alpha,
                            transform=ax.transAxes)
        ax.add_patch(box)

        ax.text(x_pos, y_start, word, ha='center', va='center', fontsize=10,
                fontweight='bold' if in_beam else 'normal',
                transform=ax.transAxes)

        # Probability
        ax.text(x_pos, y_start - 0.08, f'P={prob:.2f}', ha='center', fontsize=8,
                color='green' if in_beam else 'red',
                fontweight='bold' if in_beam else 'normal',
                transform=ax.transAxes)

        # Log score
        log_score = np.log(prob)
        ax.text(x_pos, y_start - 0.13, f'log={log_score:.2f}', ha='center', fontsize=7,
                style='italic',
                transform=ax.transAxes)

        # Arrow from START
        arrow = FancyArrowPatch((0.5, 0.85), (x_pos, y_start + 0.05),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2 if in_beam else 1,
                               color='green' if in_beam else 'gray',
                               alpha=alpha,
                               transform=ax.transAxes)
        ax.add_patch(arrow)

    # Beam indicator
    ax.text(0.5, 0.3, f'Keep top {beam_size} → The, A, I',
            ha='center', fontsize=11, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

    ax.text(0.82, y_start, '✗ Pruned', ha='center', fontsize=9, color='red',
            fontweight='bold', transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Step 2: Expand beam
    ax = axes[1]
    ax.set_title('Step 2: Expand Top 3 Hypotheses', fontsize=14, fontweight='bold')

    # Previous beam
    beam_1 = [('The', 0.40, np.log(0.40)), ('A', 0.30, np.log(0.30)), ('I', 0.20, np.log(0.20))]

    y_beam = 0.85
    x_positions_beam = [0.2, 0.5, 0.8]

    for i, (word, prob, score) in enumerate(beam_1):
        x = x_positions_beam[i]
        box = FancyBboxPatch((x - 0.05, y_beam - 0.04), 0.1, 0.06,
                            boxstyle="round,pad=0.01",
                            facecolor='lightgreen',
                            edgecolor='black',
                            linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x, y_beam, word, ha='center', va='center', fontsize=9,
                fontweight='bold', transform=ax.transAxes)

    # Expansions
    expansions = [
        # From "The"
        (0.2, [('cat', 0.50), ('dog', 0.30), ('sun', 0.20)]),
        # From "A"
        (0.5, [('nice', 0.40), ('big', 0.35), ('small', 0.25)]),
        # From "I"
        (0.8, [('think', 0.60), ('know', 0.25), ('see', 0.15)]),
    ]

    y_expand = 0.55
    all_candidates = []

    for beam_idx, (x_beam, words_probs) in enumerate(expansions):
        parent_score = beam_1[beam_idx][2]

        for j, (word, prob) in enumerate(words_probs):
            x_pos = x_beam - 0.12 + j * 0.12
            new_score = parent_score + np.log(prob)

            # Store for ranking
            parent_word = beam_1[beam_idx][0]
            all_candidates.append((f'{parent_word} {word}', new_score, x_pos))

            # Draw
            box = FancyBboxPatch((x_pos - 0.045, y_expand - 0.04), 0.09, 0.06,
                                boxstyle="round,pad=0.01",
                                facecolor='lightyellow',
                                edgecolor='black',
                                linewidth=1,
                                transform=ax.transAxes)
            ax.add_patch(box)
            ax.text(x_pos, y_expand, word, ha='center', va='center', fontsize=8,
                    transform=ax.transAxes)
            ax.text(x_pos, y_expand - 0.07, f'{new_score:.2f}', ha='center', fontsize=7,
                    style='italic', transform=ax.transAxes)

            # Arrow
            arrow = FancyArrowPatch((x_beam, y_beam - 0.05), (x_pos, y_expand + 0.05),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1, color='gray',
                                   transform=ax.transAxes)
            ax.add_patch(arrow)

    # Show top 3
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    top_3 = all_candidates[:3]

    result_text = 'Top 3: ' + ', '.join([f'{seq} ({score:.2f})' for seq, score, _ in top_3])
    ax.text(0.5, 0.25, result_text,
            ha='center', fontsize=10, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    ax.text(0.5, 0.15, f'9 candidates → keep top {beam_size}',
            ha='center', fontsize=9, style='italic',
            transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Step 3: Continue beam
    ax = axes[2]
    ax.set_title('Step 3: Prune and Continue', fontsize=14, fontweight='bold')

    # Show pruning
    beam_2 = top_3[:3]

    y_pos = 0.8
    for i, (seq, score, _) in enumerate(beam_2):
        x = 0.2 + i * 0.3
        box = FancyBboxPatch((x - 0.08, y_pos - 0.05), 0.16, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor='lightgreen',
                            edgecolor='black',
                            linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x, y_pos, seq, ha='center', va='center', fontsize=9,
                fontweight='bold', transform=ax.transAxes)
        ax.text(x, y_pos - 0.08, f'Score: {score:.2f}', ha='center', fontsize=7,
                transform=ax.transAxes)

    # Show pruned
    pruned = all_candidates[3:6]
    y_pruned = 0.5
    for i, (seq, score, _) in enumerate(pruned):
        x = 0.2 + i * 0.3
        box = FancyBboxPatch((x - 0.08, y_pruned - 0.05), 0.16, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor='lightcoral',
                            edgecolor='gray',
                            linewidth=1,
                            alpha=0.5,
                            transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(x, y_pruned, seq, ha='center', va='center', fontsize=8,
                color='gray', transform=ax.transAxes)
        ax.text(x, y_pruned - 0.08, f'Score: {score:.2f}', ha='center', fontsize=7,
                color='gray', transform=ax.transAxes)

    ax.text(0.9, y_pruned, '✗ Pruned', ha='center', fontsize=10, color='red',
            fontweight='bold', transform=ax.transAxes)

    ax.text(0.5, 0.25, 'Only keep top 3 hypotheses at each step',
            ha='center', fontsize=11, fontweight='bold',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Step 4: Final selection
    ax = axes[3]
    ax.set_title('Step 4: Select Best Complete Sequence', fontsize=14, fontweight='bold')

    # Final beam (after a few more steps)
    final_sequences = [
        ('The cat sat on the mat', -3.2, True),
        ('A nice day for walking', -3.5, False),
        ('I think this is great', -3.8, False),
    ]

    y_final = 0.7
    for i, (seq, score, is_best) in enumerate(final_sequences):
        x = 0.5
        y = y_final - i * 0.2

        color = 'gold' if is_best else 'lightblue'
        edge_width = 3 if is_best else 1.5
        alpha = 1.0 if is_best else 0.7

        box = FancyBboxPatch((0.15, y - 0.06), 0.7, 0.1,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=edge_width,
                            alpha=alpha,
                            transform=ax.transAxes)
        ax.add_patch(box)

        ax.text(0.2, y, seq, ha='left', va='center', fontsize=10,
                fontweight='bold' if is_best else 'normal',
                transform=ax.transAxes)

        ax.text(0.82, y, f'Score: {score:.2f}', ha='right', va='center', fontsize=9,
                fontweight='bold' if is_best else 'normal',
                transform=ax.transAxes)

        if is_best:
            ax.text(0.9, y, '★', ha='center', va='center', fontsize=20,
                    color='gold', transform=ax.transAxes)

    ax.text(0.5, 0.15, '✓ Select highest-scoring complete sequence',
            ha='center', fontsize=12, fontweight='bold', color='green',
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Overall title
    fig.suptitle(f'Beam Search Step-by-Step (beam_size={beam_size})',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('./beam_search_step_by_step.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated beam_search_step_by_step.pdf")


if __name__ == "__main__":
    create_beam_step_by_step()
