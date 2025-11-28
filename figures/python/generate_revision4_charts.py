"""
Generate Revision 4 Charts for NLP Final Lecture.
Charts: ANN visualization, Chunking strategies, Agent loop equations, Intermediate computation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Color scheme (consistent with existing charts)
COLORS = {
    'mlblue': '#0066CC',
    'mlpurple': '#3333B2',
    'mlorange': '#FF7F0E',
    'mlgreen': '#2CA02C',
    'mlred': '#D62728',
    'mlgray': '#7F7F7F',
    'lightgray': '#F0F0F0',
    'lavender': '#E8E8F8',
    'lightblue': '#E3F2FD',
    'lightorange': '#FFF3E0',
    'lightgreen': '#E8F5E9',
    'lightred': '#FFEBEE',
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def create_ann_search_visualization():
    """Create ANN vs Exact search comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate random points for document embeddings
    np.random.seed(42)
    n_points = 50

    # Create clusters
    cluster1 = np.random.randn(15, 2) * 0.5 + np.array([2, 2])
    cluster2 = np.random.randn(15, 2) * 0.5 + np.array([-2, 1])
    cluster3 = np.random.randn(20, 2) * 0.6 + np.array([0, -2])

    all_points = np.vstack([cluster1, cluster2, cluster3])
    query = np.array([0.5, 0.5])

    # Calculate distances
    distances = np.sqrt(np.sum((all_points - query)**2, axis=1))
    k = 5
    exact_neighbors = np.argsort(distances)[:k]

    # Simulate ANN (might miss one exact neighbor)
    ann_neighbors = exact_neighbors.copy()
    ann_neighbors[4] = np.argsort(distances)[6]  # Miss one, get 6th closest instead

    # Left plot: Exact Search
    ax1 = axes[0]
    ax1.scatter(all_points[:, 0], all_points[:, 1], c=COLORS['mlgray'], s=60, alpha=0.6, label='Documents')
    ax1.scatter(all_points[exact_neighbors, 0], all_points[exact_neighbors, 1],
                c=COLORS['mlgreen'], s=120, marker='s', edgecolors='black', linewidth=2, label='Top-5 (exact)')
    ax1.scatter(query[0], query[1], c=COLORS['mlred'], s=200, marker='*', edgecolors='black', linewidth=2, label='Query')

    # Draw circle showing search radius
    circle = plt.Circle(query, distances[exact_neighbors[-1]], fill=False,
                        color=COLORS['mlgreen'], linestyle='--', linewidth=2)
    ax1.add_patch(circle)

    ax1.set_title('Exact Nearest Neighbor Search\n(O(n) - check every point)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Embedding Dimension 1')
    ax1.set_ylabel('Embedding Dimension 2')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Add annotation
    ax1.annotate('Guarantees optimal\nresults but slow', xy=(2.5, -3), fontsize=10,
                ha='center', style='italic', color=COLORS['mlgray'])

    # Right plot: ANN Search with HNSW structure hint
    ax2 = axes[1]
    ax2.scatter(all_points[:, 0], all_points[:, 1], c=COLORS['mlgray'], s=60, alpha=0.6, label='Documents')
    ax2.scatter(all_points[ann_neighbors, 0], all_points[ann_neighbors, 1],
                c=COLORS['mlblue'], s=120, marker='s', edgecolors='black', linewidth=2, label='Top-5 (approx)')
    ax2.scatter(query[0], query[1], c=COLORS['mlred'], s=200, marker='*', edgecolors='black', linewidth=2, label='Query')

    # Show the one missed point
    missed_idx = exact_neighbors[4]
    ax2.scatter(all_points[missed_idx, 0], all_points[missed_idx, 1],
                c=COLORS['mlorange'], s=100, marker='o', edgecolors='black', linewidth=2,
                label='Missed (acceptable)')

    # Draw simplified graph connections (HNSW-like)
    for i in range(len(all_points)):
        for j in range(i+1, len(all_points)):
            dist = np.sqrt(np.sum((all_points[i] - all_points[j])**2))
            if dist < 1.5:
                ax2.plot([all_points[i, 0], all_points[j, 0]],
                        [all_points[i, 1], all_points[j, 1]],
                        c=COLORS['lightgray'], linewidth=0.5, alpha=0.5, zorder=0)

    ax2.set_title('Approximate Nearest Neighbor (ANN)\n(O(log n) - use index structure)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Embedding Dimension 1')
    ax2.set_ylabel('Embedding Dimension 2')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.annotate('95%+ recall at\n10-100x speedup', xy=(2.5, -3), fontsize=10,
                ha='center', style='italic', color=COLORS['mlblue'])

    plt.tight_layout()

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'ann_search_visualization.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Created: {output_path}")


def create_chunking_strategies_visual():
    """Create visual comparison of chunking strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Sample text representation (boxes)
    def draw_document(ax, chunks, colors, title, subtitle):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Document outline
        doc_rect = FancyBboxPatch((0.5, 0.5), 9, 5, boxstyle="round,pad=0.02",
                                   facecolor='white', edgecolor=COLORS['mlgray'], linewidth=2)
        ax.add_patch(doc_rect)

        # Draw chunks
        y_pos = 4.5
        for i, (chunk_widths, color) in enumerate(zip(chunks, colors)):
            x_pos = 0.7
            for width in chunk_widths:
                rect = FancyBboxPatch((x_pos, y_pos - 0.4), width, 0.6, boxstyle="round,pad=0.01",
                                       facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
                ax.add_patch(rect)
                x_pos += width + 0.1
            y_pos -= 1.0

        ax.text(5, 0.1, subtitle, ha='center', fontsize=9, style='italic', color=COLORS['mlgray'])

    # 1. Fixed-size chunking
    fixed_chunks = [
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2]
    ]
    fixed_colors = [COLORS['mlblue']] * 4
    draw_document(axes[0, 0], fixed_chunks, fixed_colors,
                  'Fixed-Size Chunking', 'Simple: Split every N tokens (e.g., 512)')

    # 2. Semantic chunking (variable sizes)
    semantic_chunks = [
        [3, 5],
        [2, 4, 2],
        [6, 2],
        [4, 3]
    ]
    semantic_colors = [COLORS['mlgreen']] * 4
    draw_document(axes[0, 1], semantic_chunks, semantic_colors,
                  'Semantic Chunking', 'Split at paragraph/section boundaries')

    # 3. Hierarchical chunking
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Hierarchical Chunking', fontsize=12, fontweight='bold', pad=10)

    # Parent chunk (large)
    parent = FancyBboxPatch((0.5, 3), 9, 2.5, boxstyle="round,pad=0.02",
                            facecolor=COLORS['lightorange'], edgecolor=COLORS['mlorange'], linewidth=2)
    ax.add_patch(parent)
    ax.text(5, 5.2, 'Parent Chunk (broad context)', ha='center', fontsize=9, color=COLORS['mlorange'])

    # Child chunks (small, inside parent)
    child_colors = [COLORS['mlblue'], COLORS['mlblue'], COLORS['mlblue']]
    x_positions = [1, 4, 7]
    for i, x in enumerate(x_positions):
        child = FancyBboxPatch((x, 3.3), 2, 1.8, boxstyle="round,pad=0.01",
                               facecolor=COLORS['lightblue'], edgecolor=COLORS['mlblue'], linewidth=1.5)
        ax.add_patch(child)
    ax.text(5, 3.1, 'Child Chunks (specific details)', ha='center', fontsize=9, color=COLORS['mlblue'])

    ax.text(5, 0.5, 'Multi-level: Query routes to appropriate granularity',
            ha='center', fontsize=9, style='italic', color=COLORS['mlgray'])

    # 4. Sliding window
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Sliding Window', fontsize=12, fontweight='bold', pad=10)

    # Show overlapping chunks
    window_width = 3
    overlap = 0.8
    positions = [0.7, 0.7 + window_width - overlap, 0.7 + 2*(window_width - overlap),
                 0.7 + 3*(window_width - overlap)]
    alphas = [0.8, 0.6, 0.4, 0.3]

    for i, (x, alpha) in enumerate(zip(positions, alphas)):
        rect = FancyBboxPatch((x, 2.5), window_width, 2, boxstyle="round,pad=0.01",
                               facecolor=COLORS['mlpurple'], edgecolor='black',
                               linewidth=1, alpha=alpha)
        ax.add_patch(rect)

    # Overlap indicator
    ax.annotate('', xy=(3.7, 2.3), xytext=(0.7, 2.3),
                arrowprops=dict(arrowstyle='<->', color=COLORS['mlred'], lw=2))
    ax.text(2.2, 1.9, 'Overlap', ha='center', fontsize=9, color=COLORS['mlred'])

    ax.text(5, 0.5, 'Overlapping windows: No info lost at boundaries',
            ha='center', fontsize=9, style='italic', color=COLORS['mlgray'])

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'chunking_strategies_visual.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Created: {output_path}")


def create_agent_loop_equations():
    """Create visual flowchart showing agent loop with equations."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5.5, 6.7, 'The Agent Loop: Formal Definition', ha='center', fontsize=14, fontweight='bold')

    # Main boxes
    box_style = "round,pad=0.02"

    # Context box
    context_box = FancyBboxPatch((0.5, 5), 2.5, 1.2, boxstyle=box_style,
                                  facecolor=COLORS['lavender'], edgecolor=COLORS['mlpurple'], linewidth=2)
    ax.add_patch(context_box)
    ax.text(1.75, 5.7, 'Context', ha='center', fontsize=11, fontweight='bold')
    ax.text(1.75, 5.3, r'$c_t$', ha='center', fontsize=12, style='italic')

    # Thought box
    thought_box = FancyBboxPatch((4, 5), 2.5, 1.2, boxstyle=box_style,
                                  facecolor=COLORS['lightblue'], edgecolor=COLORS['mlblue'], linewidth=2)
    ax.add_patch(thought_box)
    ax.text(5.25, 5.7, 'Thought', ha='center', fontsize=11, fontweight='bold')
    ax.text(5.25, 5.3, r'$\tau_t$', ha='center', fontsize=12, style='italic')

    # Action box
    action_box = FancyBboxPatch((7.5, 5), 2.5, 1.2, boxstyle=box_style,
                                 facecolor=COLORS['lightorange'], edgecolor=COLORS['mlorange'], linewidth=2)
    ax.add_patch(action_box)
    ax.text(8.75, 5.7, 'Action', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 5.3, r'$a_t$', ha='center', fontsize=12, style='italic')

    # Environment box (below)
    env_box = FancyBboxPatch((4, 2.5), 2.5, 1.2, boxstyle=box_style,
                              facecolor=COLORS['lightgreen'], edgecolor=COLORS['mlgreen'], linewidth=2)
    ax.add_patch(env_box)
    ax.text(5.25, 3.2, 'Environment', ha='center', fontsize=11, fontweight='bold')
    ax.text(5.25, 2.8, r'$o_t = \mathrm{Env}(a_t)$', ha='center', fontsize=10, style='italic')

    # Arrows
    arrow_props = dict(arrowstyle='->', color=COLORS['mlgray'], lw=2,
                       connectionstyle='arc3,rad=0')

    # Context -> Thought
    ax.annotate('', xy=(4, 5.6), xytext=(3, 5.6), arrowprops=arrow_props)
    ax.text(3.5, 5.9, 'LLM', ha='center', fontsize=9)

    # Thought -> Action
    ax.annotate('', xy=(7.5, 5.6), xytext=(6.5, 5.6), arrowprops=arrow_props)
    ax.text(7, 5.9, 'LLM', ha='center', fontsize=9)

    # Action -> Environment
    ax.annotate('', xy=(6.5, 3.7), xytext=(8.75, 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlgray'], lw=2,
                               connectionstyle='arc3,rad=-0.3'))
    ax.text(8, 4.2, 'Execute', ha='center', fontsize=9)

    # Environment -> Context (feedback loop)
    ax.annotate('', xy=(1.75, 5), xytext=(4, 3.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlgreen'], lw=2,
                               connectionstyle='arc3,rad=0.4'))
    ax.text(2, 3.8, 'Observe', ha='center', fontsize=9, color=COLORS['mlgreen'])

    # Equations box
    eq_box = FancyBboxPatch((0.3, 0.3), 10.4, 1.8, boxstyle="round,pad=0.02",
                             facecolor=COLORS['lightgray'], edgecolor=COLORS['mlgray'], linewidth=1)
    ax.add_patch(eq_box)

    ax.text(5.5, 1.85, 'Agent Loop Equations', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 1.3, r'1. Generate Thought: $\tau_t = \mathrm{LLM}(c_t, h_{<t})$', fontsize=10, family='monospace')
    ax.text(2, 0.9, r'2. Select Action:      $a_t = \mathrm{LLM}(\tau_t, h_{<t})$', fontsize=10, family='monospace')
    ax.text(2, 0.5, r'3. Execute & Observe: $o_t = \mathrm{Env}(a_t)$', fontsize=10, family='monospace')

    ax.text(7.5, 1.1, r'History: $h_t = (c_0, \tau_0, a_0, o_0, ..., o_{t-1})$', fontsize=9, style='italic')
    ax.text(7.5, 0.6, 'Terminate: a_t = FINISH', fontsize=9, style='italic')

    output_path = os.path.join(OUTPUT_DIR, 'agent_loop_equations.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Created: {output_path}")


def create_intermediate_computation():
    """Create visualization of intermediate computation space (CoT scratchpad)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.7, 'Intermediate Computation Space: Why CoT Works', ha='center', fontsize=14, fontweight='bold')

    # Left side: Standard generation
    ax.text(2.5, 5.1, 'Standard Generation', ha='center', fontsize=11, fontweight='bold', color=COLORS['mlred'])

    # Question box
    q_box1 = FancyBboxPatch((0.3, 4), 4.4, 0.8, boxstyle="round,pad=0.01",
                             facecolor=COLORS['lightgray'], edgecolor=COLORS['mlgray'], linewidth=1.5)
    ax.add_patch(q_box1)
    ax.text(2.5, 4.4, 'Q: 17 + 28 = ?', ha='center', fontsize=10)

    # Direct arrow to answer
    ax.annotate('', xy=(2.5, 2.8), xytext=(2.5, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlred'], lw=2))
    ax.text(3.2, 3.4, 'Single\nforward pass', ha='left', fontsize=8, color=COLORS['mlred'])

    # Answer box (potentially wrong)
    a_box1 = FancyBboxPatch((0.3, 2), 4.4, 0.8, boxstyle="round,pad=0.01",
                             facecolor=COLORS['lightred'], edgecolor=COLORS['mlred'], linewidth=1.5)
    ax.add_patch(a_box1)
    ax.text(2.5, 2.4, 'A: 44 (maybe wrong!)', ha='center', fontsize=10)

    ax.text(2.5, 1.5, 'No space to "think"', ha='center', fontsize=9, style='italic', color=COLORS['mlgray'])

    # Divider
    ax.axvline(x=5.5, ymin=0.15, ymax=0.85, color=COLORS['mlgray'], linestyle='--', linewidth=1)

    # Right side: CoT generation
    ax.text(8.5, 5.1, 'Chain-of-Thought Generation', ha='center', fontsize=11, fontweight='bold', color=COLORS['mlgreen'])

    # Question box
    q_box2 = FancyBboxPatch((6, 4), 5, 0.8, boxstyle="round,pad=0.01",
                             facecolor=COLORS['lightgray'], edgecolor=COLORS['mlgray'], linewidth=1.5)
    ax.add_patch(q_box2)
    ax.text(8.5, 4.4, 'Q: 17 + 28 = ? Let\'s think...', ha='center', fontsize=10)

    # Reasoning tokens (scratchpad)
    scratchpad = FancyBboxPatch((6, 2.4), 5, 1.4, boxstyle="round,pad=0.01",
                                 facecolor=COLORS['lavender'], edgecolor=COLORS['mlpurple'], linewidth=2)
    ax.add_patch(scratchpad)
    ax.text(8.5, 3.5, 'Reasoning tokens (scratchpad):', ha='center', fontsize=9, fontweight='bold')
    ax.text(8.5, 3.1, '"17 + 28: First, 7+8=15, carry 1..."', ha='center', fontsize=9, style='italic')
    ax.text(8.5, 2.7, '"10+20+10+5 = 45"', ha='center', fontsize=9, style='italic')

    # Arrows
    ax.annotate('', xy=(8.5, 3.8), xytext=(8.5, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlpurple'], lw=2))

    ax.annotate('', xy=(8.5, 1.6), xytext=(8.5, 2.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['mlgreen'], lw=2))

    # Final answer
    a_box2 = FancyBboxPatch((6, 0.8), 5, 0.8, boxstyle="round,pad=0.01",
                             facecolor=COLORS['lightgreen'], edgecolor=COLORS['mlgreen'], linewidth=1.5)
    ax.add_patch(a_box2)
    ax.text(8.5, 1.2, 'A: 45 (verified!)', ha='center', fontsize=10)

    # Key insight box at bottom
    insight_box = FancyBboxPatch((2, 0), 8, 0.6, boxstyle="round,pad=0.01",
                                  facecolor=COLORS['lightorange'], edgecolor=COLORS['mlorange'], linewidth=1.5)
    ax.add_patch(insight_box)
    ax.text(6, 0.3, 'Key Insight: Reasoning tokens = intermediate computation that enables multi-step problem solving',
            ha='center', fontsize=9, fontweight='bold')

    output_path = os.path.join(OUTPUT_DIR, 'intermediate_computation.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Created: {output_path}")


if __name__ == "__main__":
    print("Generating Revision 4 Charts...")
    print("=" * 50)

    create_ann_search_visualization()
    create_chunking_strategies_visual()
    create_agent_loop_equations()
    create_intermediate_computation()

    print("=" * 50)
    print("All Revision 4 charts generated successfully!")
