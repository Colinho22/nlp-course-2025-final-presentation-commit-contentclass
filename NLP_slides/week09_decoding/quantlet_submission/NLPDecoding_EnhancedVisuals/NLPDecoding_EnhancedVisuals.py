"""
Generate Enhanced Charts for Week 9 Decoding Module
Created: November 16, 2025

This script generates 9 new charts for the enhanced Week 9 presentation:
1. Quality-Diversity scatter plot (replacing old tradeoff)
2. Four graphviz charts for Problem 4 (beam search limitations)
3. Four extreme case visualizations (greedy vs full search)

All charts use BSc Discovery color scheme for consistency.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches
import graphviz
import os

# Ensure figures directory exists
os.makedirs('.', exist_ok=True)

# BSc Discovery Color Scheme (Standard)
COLOR_MAIN = '#404040'      # Main elements (dark gray)
COLOR_ACCENT = '#3333B2'    # Key concepts (purple)
COLOR_GRAY = '#B4B4B4'      # Secondary elements
COLOR_LIGHT = '#F0F0F0'     # Backgrounds
COLOR_GREEN = '#2CA02C'     # Success/positive
COLOR_RED = '#D62728'       # Error/negative
COLOR_ORANGE = '#FF7F0E'    # Warning/medium
COLOR_BLUE = '#0066CC'      # Information

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette([COLOR_ACCENT, COLOR_ORANGE, COLOR_GREEN, COLOR_BLUE])


def generate_quality_diversity_scatter():
    """
    Generate scatter plot showing quality-diversity tradeoff for 4 core methods.
    Shows: Greedy, Beam, Top-k, Nucleus on coherence vs diversity axes.
    """
    print("Generating quality_diversity_scatter_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Method data: (diversity, coherence, label)
    # Based on typical behavior patterns
    methods = {
        'Greedy': (0.15, 0.85, COLOR_MAIN),
        'Beam (k=5)': (0.25, 0.80, COLOR_ACCENT),
        'Top-k (k=50)': (0.60, 0.65, COLOR_ORANGE),
        'Nucleus (p=0.9)': (0.70, 0.75, COLOR_GREEN)
    }

    # Plot methods
    for method, (div, coh, color) in methods.items():
        ax.scatter(div, coh, s=400, c=color, alpha=0.7,
                  edgecolors=COLOR_MAIN, linewidth=2, zorder=3)

        # Add labels with offset
        offset_x = 0.05 if div < 0.5 else -0.05
        offset_y = 0.03 if coh < 0.75 else -0.03
        ha = 'left' if div < 0.5 else 'right'

        ax.annotate(method, xy=(div, coh),
                   xytext=(div + offset_x, coh + offset_y),
                   fontsize=11, fontweight='bold', color=COLOR_MAIN,
                   ha=ha, va='center')

    # Draw Pareto frontier curve
    # Approximate frontier through the "good" methods
    frontier_x = np.array([0.15, 0.25, 0.60, 0.70, 0.75])
    frontier_y = np.array([0.85, 0.80, 0.65, 0.75, 0.70])

    # Fit smooth curve
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(frontier_x.min(), frontier_x.max(), 100)
    spl = make_interp_spline(frontier_x, frontier_y, k=2)
    y_smooth = spl(x_smooth)

    ax.plot(x_smooth, y_smooth, '--', color=COLOR_GRAY,
           linewidth=2, alpha=0.6, label='Pareto Frontier', zorder=1)

    # Add "Sweet Spot" region
    sweet_x = [0.55, 0.75]
    sweet_y = [0.65, 0.80]
    rect = Rectangle((sweet_x[0], sweet_y[0]),
                     sweet_x[1] - sweet_x[0],
                     sweet_y[1] - sweet_y[0],
                     facecolor=COLOR_GREEN, alpha=0.1,
                     edgecolor=COLOR_GREEN, linewidth=2,
                     linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text(0.65, 0.62, 'Sweet Spot', fontsize=10,
           color=COLOR_GREEN, fontweight='bold', ha='center')

    # Add region labels
    ax.text(0.15, 0.15, 'Too Deterministic\n(Repetitive)',
           fontsize=9, color=COLOR_RED, ha='center', alpha=0.7,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_RED, alpha=0.3))

    ax.text(0.85, 0.15, 'Too Random\n(Incoherent)',
           fontsize=9, color=COLOR_RED, ha='center', alpha=0.7,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_RED, alpha=0.3))

    # Formatting
    ax.set_xlabel('Diversity (Entropy)', fontsize=13, fontweight='bold', color=COLOR_MAIN)
    ax.set_ylabel('Coherence (Quality)', fontsize=13, fontweight='bold', color=COLOR_MAIN)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('./quality_diversity_scatter_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] quality_diversity_scatter_bsc.pdf created")


def generate_problem4_graphviz_charts():
    """
    Generate 4 graphviz charts showing different aspects of beam search limitations.
    1. Search tree pruning
    2. Path comparison (taken vs optimal)
    3. Probability evolution over time
    4. Recovery problem
    """

    # Chart 1: Search Tree Pruning
    print("Generating problem4_search_tree_pruning_bsc.pdf...")
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='transparent')
    dot.attr('node', shape='box', style='rounded,filled',
            fontname='Arial', fontsize='11')

    # Root
    dot.node('start', 'START', fillcolor='#E8E8F0', color=COLOR_ACCENT, penwidth='2')

    # Step 1: 3 options
    dot.node('A1', 'The (0.6)', fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')
    dot.node('A2', 'A (0.3)', fillcolor='#FFE6CC', color=COLOR_ORANGE)
    dot.node('A3', 'In (0.1)', fillcolor='#FFD6D6', color=COLOR_RED)

    dot.edge('start', 'A1', label='Keep', color=COLOR_GREEN, penwidth='2')
    dot.edge('start', 'A2', label='Keep', color=COLOR_ORANGE, penwidth='1.5')
    dot.edge('start', 'A3', label='Prune!', color=COLOR_RED,
            style='dashed', penwidth='1')

    # Step 2: Continue from top 2 (beam=2)
    dot.node('B1', 'cat (0.4)', fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')
    dot.node('B2', 'dog (0.2)', fillcolor='#FFE6CC', color=COLOR_ORANGE)
    dot.node('B3', 'quick (0.05)', fillcolor='#FFD6D6', color=COLOR_RED)
    dot.node('B4', 'red (0.25)', fillcolor='#FFD6D6', color=COLOR_RED)

    dot.edge('A1', 'B1', color=COLOR_GREEN, penwidth='2')
    dot.edge('A1', 'B2', color=COLOR_ORANGE, penwidth='1.5')
    dot.edge('A2', 'B3', label='Lost!', color=COLOR_RED, style='dashed')
    dot.edge('A2', 'B4', label='Lost!', color=COLOR_RED, style='dashed')

    # Note
    dot.node('note', '"In quick" had high probability\nbut pruned at step 1!',
            shape='note', fillcolor='#FFFFCC', color=COLOR_MAIN)

    dot.render('./problem4_search_tree_pruning_bsc', cleanup=True)
    print("[OK] problem4_search_tree_pruning_bsc.pdf created")


    # Chart 2: Path Comparison
    print("Generating problem4_path_comparison_bsc.pdf...")
    dot2 = graphviz.Digraph(format='pdf', engine='dot')
    dot2.attr(dpi='300', rankdir='LR', bgcolor='transparent')
    dot2.attr('node', shape='box', style='filled',
             fontname='Arial', fontsize='10')

    # Taken path (greedy/beam)
    dot2.node('taken', 'Path Taken by Beam\n(greedy-ish)',
             shape='box', fillcolor='#FFD6D6', color=COLOR_RED, penwidth='2')
    dot2.node('t1', 'The', fillcolor='#FFE6CC')
    dot2.node('t2', 'cat', fillcolor='#FFE6CC')
    dot2.node('t3', 'sat', fillcolor='#FFE6CC')
    dot2.node('t4', 'down', fillcolor='#FFE6CC')
    dot2.node('t_score', 'Score: 0.72\n(locally optimal)',
             shape='ellipse', fillcolor='#FFD6D6', color=COLOR_RED)

    dot2.edge('taken', 't1', color=COLOR_RED, penwidth='2')
    dot2.edge('t1', 't2', color=COLOR_RED, penwidth='2')
    dot2.edge('t2', 't3', color=COLOR_RED, penwidth='2')
    dot2.edge('t3', 't4', color=COLOR_RED, penwidth='2')
    dot2.edge('t4', 't_score', color=COLOR_RED, penwidth='2')

    # Optimal path (missed)
    dot2.node('optimal', 'Optimal Path\n(missed by beam)',
             shape='box', fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')
    dot2.node('o1', 'A', fillcolor='#E6F7E6')
    dot2.node('o2', 'small', fillcolor='#E6F7E6')
    dot2.node('o3', 'cat', fillcolor='#E6F7E6')
    dot2.node('o4', 'purred', fillcolor='#E6F7E6')
    dot2.node('o_score', 'Score: 0.89\n(globally optimal)',
             shape='ellipse', fillcolor='#D4E6D4', color=COLOR_GREEN)

    dot2.edge('optimal', 'o1', color=COLOR_GREEN, penwidth='2')
    dot2.edge('o1', 'o2', color=COLOR_GREEN, penwidth='2')
    dot2.edge('o2', 'o3', color=COLOR_GREEN, penwidth='2')
    dot2.edge('o3', 'o4', color=COLOR_GREEN, penwidth='2')
    dot2.edge('o4', 'o_score', color=COLOR_GREEN, penwidth='2')

    dot2.render('./problem4_path_comparison_bsc', cleanup=True)
    print("[OK] problem4_path_comparison_bsc.pdf created")


    # Chart 3: Probability Evolution
    print("Generating problem4_probability_evolution_bsc.pdf...")
    dot3 = graphviz.Digraph(format='pdf', engine='dot')
    dot3.attr(dpi='300', rankdir='LR', bgcolor='transparent')
    dot3.attr('node', shape='record', style='filled',
             fontname='Arial', fontsize='10')

    # Time steps showing probability accumulation
    dot3.node('t0', '{<f0> START | <f1> P=1.0}',
             fillcolor='#E8E8F0', color=COLOR_ACCENT)

    dot3.node('t1', '{<f0> Step 1 | <f1> "The": 0.6 | <f2> "A": 0.3 (pruned)}',
             fillcolor='#FFE6CC', color=COLOR_ORANGE)

    dot3.node('t2', '{<f0> Step 2 | <f1> "The cat": 0.24 | <f2> "A small": 0.12 (lost!)}',
             fillcolor='#FFD6D6', color=COLOR_RED)

    dot3.node('t3', '{<f0> Step 3 | <f1> "The cat sat": 0.096 | <f2> "A small cat": 0.048 (lost!)}',
             fillcolor='#FFD6D6', color=COLOR_RED)

    dot3.node('result', 'Early pruning causes\ncumulative probability loss',
             shape='note', fillcolor='#FFFFCC', color=COLOR_MAIN)

    dot3.edge('t0', 't1', color=COLOR_ACCENT, penwidth='2')
    dot3.edge('t1', 't2', color=COLOR_ORANGE, penwidth='2')
    dot3.edge('t2', 't3', color=COLOR_RED, penwidth='2')
    dot3.edge('t3', 'result', style='dashed', color=COLOR_GRAY)

    dot3.render('./problem4_probability_evolution_bsc', cleanup=True)
    print("[OK] problem4_probability_evolution_bsc.pdf created")


    # Chart 4: Recovery Problem
    print("Generating problem4_recovery_problem_bsc.pdf...")
    dot4 = graphviz.Digraph(format='pdf', engine='dot')
    dot4.attr(dpi='300', rankdir='TB', bgcolor='transparent')
    dot4.attr('node', shape='box', style='rounded,filled',
             fontname='Arial', fontsize='11')

    # Show the irreversibility
    dot4.node('step1', 'Step 1: Prune "A"\n(P=0.3)',
             fillcolor='#FFD6D6', color=COLOR_RED, penwidth='2')

    dot4.node('step2', 'Step 2: Continue with "The"\nAll descendants are "The ..."',
             fillcolor='#FFE6CC', color=COLOR_ORANGE)

    dot4.node('step3', 'Step 3-10: Still following\n"The cat sat ..."',
             fillcolor='#FFE6CC', color=COLOR_ORANGE)

    dot4.node('realize', 'Realize: "A small cat purred"\nwould have been better!',
             fillcolor='#FFFFCC', color=COLOR_MAIN, shape='ellipse')

    dot4.node('problem', 'PROBLEM:\nCannot recover!\nPath starting with "A" is gone forever.',
             fillcolor='#FFD6D6', color=COLOR_RED, penwidth='3', shape='box')

    dot4.edge('step1', 'step2', color=COLOR_RED, penwidth='2')
    dot4.edge('step2', 'step3', color=COLOR_ORANGE, penwidth='2')
    dot4.edge('step3', 'realize', color=COLOR_GRAY, style='dashed')
    dot4.edge('realize', 'problem', color=COLOR_RED, penwidth='2',
             label='No way back!', fontcolor=COLOR_RED, fontsize='10')

    # Add note about irreversibility
    dot4.node('note', 'Beam search is GREEDY:\nOnce pruned, paths are lost forever',
             shape='note', fillcolor='#E8E8F0', color=COLOR_ACCENT)

    dot4.render('./problem4_recovery_problem_bsc', cleanup=True)
    print("[OK] problem4_recovery_problem_bsc.pdf created")


def generate_extreme_case_1_greedy():
    """
    Extreme Case 1: Greedy decoding (single deterministic path)
    Completely new design with Input→Process→Output flow.
    Bold 2-tier font hierarchy (24pt headers, 18pt content) for maximum readability.
    """
    print("Generating extreme_greedy_single_path_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(16, 8))

    # LEFT: INPUT - Show initial text prompt
    input_x = -5
    input_y = 3.5

    # Draw input box
    input_rect = FancyBboxPatch((input_x - 1.5, input_y - 1.5), 3, 3,
                                boxstyle="round,pad=0.1",
                                facecolor=COLOR_LIGHT,
                                edgecolor=COLOR_ACCENT,
                                linewidth=2, zorder=2)
    ax.add_patch(input_rect)

    ax.text(input_x, input_y + 2, 'INPUT', fontsize=24, fontweight='bold',
           ha='center', color=COLOR_ACCENT)

    ax.text(input_x, input_y + 0.5, 'Prompt:', fontsize=18,
           ha='center', color=COLOR_MAIN, fontweight='bold')
    ax.text(input_x, input_y - 0.2, '"The cat"', fontsize=18,
           ha='center', color=COLOR_MAIN, style='italic')
    ax.text(input_x, input_y - 0.9, 'Task: Generate\nnext 3 words', fontsize=18,
           ha='center', color=COLOR_GRAY)

    # Arrow from INPUT to PROCESS
    ax.annotate('', xy=(-2.5, input_y), xytext=(input_x + 1.5, input_y),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_ACCENT))

    # CENTER: PROCESS - Show greedy selection at each step
    process_x = 0

    ax.text(process_x, 6, 'PROCESS: Greedy Selection', fontsize=24, fontweight='bold',
           ha='center', color=COLOR_ACCENT)

    # Show step-by-step greedy process
    steps = [
        ('Step 1', 'sat', 0.71, ['sat: 0.71', 'is: 0.15', 'ran: 0.08', '...']),
        ('Step 2', 'on', 0.69, ['on: 0.69', 'near: 0.12', 'by: 0.10', '...']),
        ('Step 3', 'the', 0.72, ['the: 0.72', 'a: 0.18', 'my: 0.05', '...'])
    ]

    for i, (step_label, word, prob, options) in enumerate(steps):
        y_pos = 4.5 - i * 1.8

        # Draw options box
        opts_rect = FancyBboxPatch((process_x - 2, y_pos - 0.7), 1.8, 1.4,
                                   boxstyle="round,pad=0.05",
                                   facecolor='white',
                                   edgecolor=COLOR_GRAY,
                                   linewidth=1, alpha=0.5)
        ax.add_patch(opts_rect)

        # Show top options
        ax.text(process_x - 1.1, y_pos + 0.3, step_label, fontsize=18,
               ha='center', color=COLOR_GRAY, fontweight='bold')
        for j, opt in enumerate(options[:3]):
            y_opt = y_pos - 0.1 - j * 0.2
            color = COLOR_GREEN if j == 0 else COLOR_GRAY
            weight = 'bold' if j == 0 else 'normal'
            ax.text(process_x - 1.1, y_opt, opt, fontsize=18,
                   ha='center', color=color, fontweight=weight)

        # Arrow showing selection
        ax.annotate('argmax', xy=(process_x + 0.3, y_pos), xytext=(process_x - 0.2, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_GREEN),
                   fontsize=18, ha='center', color=COLOR_GREEN)

        # Selected word box
        word_rect = FancyBboxPatch((process_x + 0.3, y_pos - 0.4), 1.5, 0.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=COLOR_GREEN,
                                   edgecolor=COLOR_ACCENT,
                                   linewidth=2, alpha=0.8)
        ax.add_patch(word_rect)

        ax.text(process_x + 1.05, y_pos, f'"{word}"', fontsize=18,
               ha='center', va='center', color='white', fontweight='bold')

        # Connect to next step
        if i < len(steps) - 1:
            ax.plot([process_x + 1.05, process_x - 1.1],
                   [y_pos - 0.5, y_pos - 1.3],
                   '--', color=COLOR_GRAY, linewidth=1.5, alpha=0.5)

    # Arrow from PROCESS to OUTPUT
    ax.annotate('', xy=(3.5, input_y), xytext=(process_x + 2, input_y),
               arrowprops=dict(arrowstyle='->', lw=3, color=COLOR_ACCENT))

    # RIGHT: OUTPUT - Show final result and statistics
    output_x = 5.5

    ax.text(output_x, input_y + 2, 'OUTPUT', fontsize=24, fontweight='bold',
           ha='center', color=COLOR_ACCENT)

    # Generated text box
    output_rect = FancyBboxPatch((output_x - 1.8, input_y + 0.3), 3.6, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor=COLOR_GREEN,
                                 edgecolor=COLOR_ACCENT,
                                 linewidth=2, alpha=0.1)
    ax.add_patch(output_rect)

    ax.text(output_x, input_y + 0.9, '"The cat sat on the"', fontsize=18,
           ha='center', color=COLOR_MAIN, fontweight='bold')

    # Statistics box
    stats_rect = FancyBboxPatch((output_x - 1.8, input_y - 2.5), 3.6, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=COLOR_LIGHT,
                                edgecolor=COLOR_RED,
                                linewidth=2)
    ax.add_patch(stats_rect)

    ax.text(output_x, input_y - 0.8, 'Statistics:', fontsize=18,
           ha='center', color=COLOR_RED, fontweight='bold')

    stats_lines = [
        'Paths explored: 1',
        'Paths possible: 100³',
        '= 1,000,000',
        'Coverage: 0.0001%'
    ]

    for i, line in enumerate(stats_lines):
        ax.text(output_x, input_y - 1.4 - i*0.3, line, fontsize=18,
               ha='center', color=COLOR_MAIN)

    # Add arrow pointing to stats
    ax.annotate('', xy=(4.5, 3.5), xytext=(2, 3),
               arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_RED, alpha=0.5))

    # Removed duplicate title and subtitle as requested

    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./extreme_greedy_single_path_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_greedy_single_path_bsc.pdf created (reordered left-to-right)")


def generate_extreme_case_2_full_beam():
    """
    Extreme Case 2: Full beam search (exponential explosion)
    Shows how vocabulary size 100 leads to exponential growth.
    """
    print("Generating extreme_full_beam_explosion_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw exponentially growing tree
    vocab_size = 100
    levels = 5

    for level in range(levels):
        y = 5 - level
        num_nodes = vocab_size ** level if level < 3 else 200  # Cap visual

        # Calculate actual count
        actual_count = vocab_size ** level

        if level < 3:
            # Draw all nodes
            x_span = min(16, 2 * np.sqrt(num_nodes))
            x_positions = np.linspace(-x_span/2, x_span/2, num_nodes)

            for x in x_positions:
                size = 0.3 if level == 0 else max(0.05, 0.3 - level * 0.05)
                circle = Circle((x, y), size,
                              facecolor=COLOR_ORANGE,
                              edgecolor=COLOR_RED,
                              linewidth=1, alpha=0.6)
                ax.add_patch(circle)
        else:
            # Just show representative cluster
            x_positions = np.linspace(-8, 8, 200)
            for x in x_positions:
                circle = Circle((x, y), 0.02,
                              facecolor=COLOR_RED,
                              edgecolor=COLOR_RED,
                              linewidth=0.5, alpha=0.4)
                ax.add_patch(circle)

        # Label with count
        label = f'Step {level}: {actual_count:,} nodes'
        if actual_count >= 1000000:
            label = f'Step {level}: {actual_count/1e6:.0f}M nodes'
        if actual_count >= 1000000000:
            label = f'Step {level}: {actual_count/1e9:.1f}B nodes'

        ax.text(-9, y, label, fontsize=10, ha='right', va='center',
               fontweight='bold', color=COLOR_MAIN,
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white', alpha=0.9))

    # Add title
    ax.text(0, 6.5, 'Extreme Case 2: Full Search Space',
           fontsize=16, fontweight='bold', ha='center', color=COLOR_MAIN)

    ax.text(0, 6.1, 'Vocabulary size = 100, explore ALL paths',
           fontsize=11, ha='center', color=COLOR_GRAY, style='italic')

    # Computation box
    comp_text = 'Total paths: 100^5 = 10 billion\n\nIf 1 μs per path:\n10 billion × 1 μs = 2.8 hours\n\nIf 1 ms per path:\n10 billion × 1 ms = 115 days!'
    ax.text(0, -0.5, comp_text, fontsize=9, ha='center',
           bbox=dict(boxstyle='round,pad=0.8',
                    facecolor=COLOR_LIGHT,
                    edgecolor=COLOR_RED, linewidth=2),
           color=COLOR_MAIN, fontweight='bold')

    ax.set_xlim(-11, 10)
    ax.set_ylim(-2, 7)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./extreme_full_beam_explosion_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_full_beam_explosion_bsc.pdf created")


def generate_extreme_case_3_computational_cost():
    """
    Extreme Case 3: Computational cost comparison
    Bar chart showing 1 vs 100 vs 10,000 vs 1,000,000 vs 10,000,000,000 paths.
    """
    print("Generating extreme_computational_cost_bsc.pdf...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Data for different search strategies
    strategies = ['Greedy\n(1 path)',
                 'Beam k=5\n(500 paths)',
                 'Beam k=50\n(50K paths)',
                 'Top-k=100\n(10M paths)',
                 'Full Search\n(10B paths)']

    paths = [1, 500, 50000, 10000000, 10000000000]

    # Time assuming 1 microsecond per path evaluation
    times_us = np.array(paths)  # in microseconds
    times_sec = times_us / 1e6  # convert to seconds

    # Use log scale
    log_times = np.log10(times_sec + 1e-10)  # Add small constant to avoid log(0)

    # Create bars with color gradient
    colors_list = [COLOR_GREEN, COLOR_BLUE, COLOR_ORANGE, COLOR_ORANGE, COLOR_RED]

    bars = ax.barh(strategies, log_times, color=colors_list,
                   edgecolor=COLOR_MAIN, linewidth=1.5, alpha=0.8)

    # Add actual time labels
    for i, (strategy, time_sec) in enumerate(zip(strategies, times_sec)):
        if time_sec < 0.001:
            time_label = f'{time_sec*1e6:.1f} μs'
        elif time_sec < 1:
            time_label = f'{time_sec*1e3:.1f} ms'
        elif time_sec < 60:
            time_label = f'{time_sec:.1f} sec'
        elif time_sec < 3600:
            time_label = f'{time_sec/60:.1f} min'
        elif time_sec < 86400:
            time_label = f'{time_sec/3600:.1f} hours'
        else:
            time_label = f'{time_sec/86400:.1f} days'

        ax.text(log_times[i] + 0.3, i, time_label,
               va='center', fontsize=11, fontweight='bold', color=COLOR_MAIN)

    # Formatting
    ax.set_xlabel('Computation Time (log scale)',
                  fontsize=13, fontweight='bold', color=COLOR_MAIN)
    ax.set_title('Computational Cost vs Search Breadth\n(Assuming 1 μs per path, sequence length=5)',
                fontsize=14, fontweight='bold', color=COLOR_MAIN, pad=20)

    # Add vertical lines for reference
    ax.axvline(x=np.log10(0.001), color=COLOR_GRAY, linestyle='--',
              alpha=0.5, linewidth=1)
    ax.text(np.log10(0.001), len(strategies), '1 ms',
           ha='center', va='bottom', fontsize=9, color=COLOR_GRAY)

    ax.axvline(x=np.log10(1), color=COLOR_GRAY, linestyle='--',
              alpha=0.5, linewidth=1)
    ax.text(np.log10(1), len(strategies), '1 sec',
           ha='center', va='bottom', fontsize=9, color=COLOR_GRAY)

    # Grid and spines
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom x-ticks
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_xticklabels(['1 μs', '1 ms', '1 sec', '16 min', '11 days'])

    plt.tight_layout()
    plt.savefig('./extreme_computational_cost_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_computational_cost_bsc.pdf created")


def generate_extreme_coverage_comparison():
    """
    The Extremes: Shows greedy vs full search coverage.
    Split from original search coverage - Part 1 of 2.
    """
    print("Generating extreme_coverage_comparison_bsc.pdf...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create 2D representation of search space
    # X-axis: First word choice (0-100)
    # Y-axis: Second word choice (0-100)

    methods = [
        ('Greedy Decoding\n(1 path)', 1, COLOR_RED, 'Reds'),
        ('Full Search\n(All 10,000 paths)', 'full', COLOR_GREEN, 'Greens')
    ]

    for idx, (method_name, beam_width, color, cmap) in enumerate(methods):
        ax = axes[idx]

        # Create heat map
        grid = np.zeros((100, 100))

        if beam_width == 1:
            # Greedy: single point
            grid[50, 50] = 1.0
        else:
            # Full search: everything explored
            grid = np.ones((100, 100))

        # Plot heat map
        im = ax.imshow(grid, cmap=cmap,
                      origin='lower', extent=[0, 100, 0, 100],
                      vmin=0, vmax=1, alpha=0.8)

        # Calculate coverage
        coverage = (grid > 0).sum() / (100 * 100) * 100

        # Title
        ax.set_title(f'{method_name}\nCoverage: {coverage:.2f}%',
                    fontsize=12, fontweight='bold', color=COLOR_MAIN, pad=10)

        ax.set_xlabel('First Word (100 options)', fontsize=10, color=COLOR_GRAY)
        ax.set_ylabel('Second Word (100 options)', fontsize=10, color=COLOR_GRAY)

        # Add grid
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Explored', fontsize=9, color=COLOR_GRAY)

    plt.suptitle('The Extremes: Coverage Comparison\n(Vocabulary=100, showing first 2 words only)',
                fontsize=14, fontweight='bold', color=COLOR_MAIN, y=1.02)

    plt.tight_layout()
    plt.savefig('./extreme_coverage_comparison_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] extreme_coverage_comparison_bsc.pdf created")


def generate_practical_methods_coverage():
    """
    Practical Solutions: Shows beam search, top-k, and nucleus coverage.
    Split from original search coverage - Part 2 of 2.
    """
    print("Generating practical_methods_coverage_bsc.pdf...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Seed for reproducibility
    np.random.seed(42)

    # Create 2D representation of search space
    methods = [
        ('Beam Search (k=5)\n25 paths', 5, COLOR_BLUE, 'Blues'),
        ('Top-k Sampling (k=20)\n~400 paths', 20, COLOR_ORANGE, 'Oranges'),
        ('Nucleus (p=0.9)\n~1000 paths', 35, COLOR_GREEN, 'Greens')
    ]

    for idx, (method_name, beam_width, color, cmap) in enumerate(methods):
        ax = axes[idx]

        # Create heat map
        grid = np.zeros((100, 100))

        if beam_width == 5:
            # Beam k=5: small cluster
            centers = [(50, 50), (48, 52), (52, 48), (51, 51), (49, 49)]
            for cx, cy in centers:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= cx+dx < 100 and 0 <= cy+dy < 100:
                            grid[cx+dx, cy+dy] = max(grid[cx+dx, cy+dy],
                                                    1.0 - (abs(dx) + abs(dy))/5)
        elif beam_width == 20:
            # Top-k: medium spread
            for _ in range(400):
                cx, cy = np.random.randint(20, 80), np.random.randint(20, 80)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if 0 <= cx+dx < 100 and 0 <= cy+dy < 100:
                            dist = np.sqrt(dx**2 + dy**2)
                            grid[cx+dx, cy+dy] = max(grid[cx+dx, cy+dy],
                                                    max(0, 1.0 - dist/5))
        else:
            # Nucleus: adaptive coverage
            for _ in range(1000):
                cx, cy = np.random.randint(15, 85), np.random.randint(15, 85)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= cx+dx < 100 and 0 <= cy+dy < 100:
                            dist = np.sqrt(dx**2 + dy**2)
                            grid[cx+dx, cy+dy] = max(grid[cx+dx, cy+dy],
                                                    max(0, 1.0 - dist/3))

        # Plot heat map
        im = ax.imshow(grid, cmap=cmap,
                      origin='lower', extent=[0, 100, 0, 100],
                      vmin=0, vmax=1, alpha=0.8)

        # Calculate coverage
        coverage = (grid > 0).sum() / (100 * 100) * 100

        # Title
        ax.set_title(f'{method_name}\nCoverage: {coverage:.2f}%',
                    fontsize=11, fontweight='bold', color=COLOR_MAIN, pad=10)

        ax.set_xlabel('First Word', fontsize=9, color=COLOR_GRAY)
        if idx == 0:
            ax.set_ylabel('Second Word', fontsize=9, color=COLOR_GRAY)

        # Add grid
        ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if idx == 2:
            cbar.set_label('Exploration\nIntensity', fontsize=8, color=COLOR_GRAY)

    # Add Sweet Spot indicator
    axes[2].text(85, 85, '← Sweet\n    Spot', fontsize=9,
               color=COLOR_GREEN, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLOR_GREEN, alpha=0.8))

    plt.suptitle('Practical Solutions: Balancing Coverage and Computation',
                fontsize=14, fontweight='bold', color=COLOR_MAIN, y=1.02)

    plt.tight_layout()
    plt.savefig('./practical_methods_coverage_bsc.pdf',
               dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] practical_methods_coverage_bsc.pdf created")


def main():
    """Generate all enhanced charts for Week 9."""
    print("\n" + "="*60)
    print("WEEK 9 ENHANCED CHART GENERATION")
    print("="*60 + "\n")

    print("Phase 1: Quality-Diversity Scatter Plot")
    print("-" * 60)
    generate_quality_diversity_scatter()

    print("\nPhase 2: Problem 4 Graphviz Charts (4 charts)")
    print("-" * 60)
    generate_problem4_graphviz_charts()

    print("\nPhase 3: Extreme Case Visualizations (5 charts)")
    print("-" * 60)
    generate_extreme_case_1_greedy()
    generate_extreme_case_2_full_beam()
    generate_extreme_case_3_computational_cost()
    # Split into two new functions
    generate_extreme_coverage_comparison()
    generate_practical_methods_coverage()

    print("\n" + "="*60)
    print("CHART GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated 10 new charts:")
    print("  1. quality_diversity_scatter_bsc.pdf")
    print("  2-5. problem4_*_bsc.pdf (4 graphviz charts)")
    print("  6. extreme_greedy_single_path_bsc.pdf (reordered)")
    print("  7. extreme_full_beam_explosion_bsc.pdf")
    print("  8. extreme_computational_cost_bsc.pdf")
    print("  9. extreme_coverage_comparison_bsc.pdf (NEW)")
    print("  10. practical_methods_coverage_bsc.pdf (NEW)")
    print("\nAll charts saved to: ../figures/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
