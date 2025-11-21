#!/usr/bin/env python3
"""
Generate 4 variations of "The cat __" probability setup chart
Font size: 20-22pt, clean layouts
User will choose which one to use
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import graphviz
import os

os.makedirs('.', exist_ok=True)

# Colors
PURPLE = '#3333B2'
GREEN = '#2CA02C'
BLUE = '#0066CC'
GRAY = '#7F7F7F'

print("Generating 4 'The cat __' setup chart variations...")
print("=" * 60)

# === SETUP A: Bar Chart (Top 10 Words) ===
def setup_chart_A_bars():
    """Vertical bar chart showing probability distribution"""
    fig, ax = plt.subplots(figsize=(12, 7))

    words = ['sat', 'is', 'jumped', 'walked', 'was', 'sleeping', 'ran', 'meowed', 'playing', 'hiding']
    probs = [0.45, 0.30, 0.25, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01]

    # Color top 3 differently
    colors = [GREEN if i < 3 else BLUE for i in range(len(words))]

    bars = ax.bar(range(len(words)), probs, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=2.5, width=0.7)

    # Probability labels on bars
    for i, (word, prob) in enumerate(zip(words, probs)):
        ax.text(i, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=20, weight='bold')

    ax.set_xlabel('Next Word Candidates', fontsize=24, weight='bold', labelpad=12)
    ax.set_ylabel('Probability P(word | "The cat")', fontsize=24, weight='bold', labelpad=12)
    ax.set_title('Model Output: Probability Distribution for "The cat __"',
                fontsize=26, weight='bold', pad=20, color=PURPLE)
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, fontsize=20, rotation=15)
    ax.set_ylim(0, 0.55)
    ax.tick_params(labelsize=20, width=2.5)
    ax.grid(axis='y', alpha=0.3, linewidth=1.5)

    # Annotation
    ax.text(7, 0.45, '50,000 words total\nTop 10 shown here',
            fontsize=18, style='italic', color=GRAY,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=GRAY, linewidth=2))

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color(PURPLE)

    plt.tight_layout()
    plt.savefig('./setup_A_bar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK Setup A: Bar chart (top 10 words)")

# === SETUP B: Horizontal List ===
def setup_chart_B_list():
    """Horizontal probability list with highlighting"""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.text(0.5, 0.85, 'Model Output for: "The cat __"',
            ha='center', fontsize=28, weight='bold', color=PURPLE)

    # Top probabilities as formatted list
    probs_text = [
        ('P(sat) = 0.45', GREEN, 'bold'),
        ('P(is) = 0.30', GREEN, 'bold'),
        ('P(jumped) = 0.25', GREEN, 'bold'),
        ('P(walked) = 0.12', BLUE, 'normal'),
        ('P(was) = 0.10', BLUE, 'normal'),
        ('P(sleeping) = 0.08', BLUE, 'normal'),
        ('P(ran) = 0.05', GRAY, 'normal'),
        ('P(meowed) = 0.03', GRAY, 'normal'),
        ('...', GRAY, 'italic'),
        ('P(word_49,990) = 0.000001', GRAY, 'italic'),
    ]

    y = 0.70
    for text, color, style in probs_text:
        weight = 'bold' if style in ['bold'] else 'normal'
        fontstyle = 'italic' if style == 'italic' else 'normal'
        ax.text(0.5, y, text, ha='center', fontsize=24, color=color,
                weight=weight, style=fontstyle)
        y -= 0.06

    # Highlight box around top 3
    highlight = mpatches.FancyBboxPatch((0.25, 0.52), 0.5, 0.20,
                                        boxstyle="round,pad=0.02", linewidth=3,
                                        edgecolor=GREEN, facecolor=GREEN, alpha=0.1)
    ax.add_patch(highlight)
    ax.text(0.5, 0.47, 'Top 3 candidates (85% of probability mass)',
            ha='center', fontsize=20, style='italic', color=GREEN, weight='bold')

    # Total vocabulary note
    ax.text(0.5, 0.12, 'Total vocabulary: 50,000 words',
            ha='center', fontsize=22, weight='bold', color=PURPLE,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                     edgecolor=PURPLE, linewidth=3))

    ax.text(0.5, 0.03, 'Challenge: Which word do we pick?',
            ha='center', fontsize=20, style='italic', color=GRAY)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('./setup_B_list.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK Setup B: Horizontal list with highlighting")

# === SETUP C: Distribution Curve ===
def setup_chart_C_curve():
    """Probability distribution showing steep drop-off"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Realistic distribution (power law)
    ranks = np.arange(1, 101)
    probs = 0.45 * (1 / ranks)**0.8  # Power law
    probs = probs / probs.sum() * 0.95  # Normalize to sum ~1

    ax.plot(ranks, probs, color=PURPLE, linewidth=4)
    ax.fill_between(ranks[:10], probs[:10], alpha=0.3, color=GREEN, label='Top 10 words')
    ax.fill_between(ranks[10:], probs[10:], alpha=0.2, color=GRAY, label='Remaining words')

    # Annotations
    top10_mass = probs[:10].sum()
    ax.text(5, 0.20, f'Top 10 words\n{top10_mass:.0%} of mass',
            fontsize=22, weight='bold', color=GREEN,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN, linewidth=2))

    ax.text(50, 0.08, f'Remaining 49,990 words\n{1-top10_mass:.0%} of mass',
            fontsize=20, style='italic', color=GRAY)

    # Steep drop annotation
    ax.annotate('Steep drop-off', xy=(3, probs[2]), xytext=(20, 0.30),
                arrowprops=dict(arrowstyle='->', lw=3, color=PURPLE),
                fontsize=22, weight='bold', color=PURPLE)

    ax.set_xlabel('Word Rank (1 = highest probability)', fontsize=24, weight='bold', labelpad=12)
    ax.set_ylabel('Probability', fontsize=24, weight='bold', labelpad=12)
    ax.set_title('Probability Distribution for "The cat __" (First 100 words shown)',
                fontsize=26, weight='bold', pad=20, color=PURPLE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(probs) * 1.1)
    ax.tick_params(labelsize=20, width=2.5)
    ax.legend(fontsize=20, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=1.5)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2.5)
        ax.spines[spine].set_color(PURPLE)

    plt.tight_layout()
    plt.savefig('./setup_C_curve.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("OK Setup C: Distribution curve (power law)")

# === SETUP D: Graphviz Node Diagram ===
def setup_chart_D_nodes():
    """Graphviz radial diagram showing choices"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='TB', fontsize='20', bgcolor='white', nodesep='1.0', ranksep='1.2')
    dot.attr('node', fontname='Arial', shape='box', style='rounded')

    # Central question
    dot.node('question', 'The cat __\\n(Which word?)', fillcolor=PURPLE, fontcolor='white',
             style='filled,rounded', fontsize='26', penwidth='4', width='3', height='1.2')

    # Top candidates with probabilities
    candidates = [
        ('sat', '0.45', GREEN),
        ('is', '0.30', GREEN),
        ('jumped', '0.25', GREEN),
        ('walked', '0.12', BLUE),
        ('was', '0.10', BLUE),
        ('sleeping', '0.08', BLUE),
        ('ran', '0.05', GRAY),
        ('meowed', '0.03', GRAY),
    ]

    for i, (word, prob, color) in enumerate(candidates):
        node_id = f'word{i}'
        label = f'{word}\\nP={prob}'
        penwidth = '3' if i < 3 else '2'
        fontsize = '22' if i < 3 else '18'

        dot.node(node_id, label, fillcolor=color, fontcolor='white',
                style='filled,rounded', fontsize=fontsize, penwidth=penwidth,
                width='1.8', height='0.8')

        edge_width = '4' if i < 3 else '2'
        edge_color = color
        dot.edge('question', node_id, penwidth=edge_width, color=edge_color)

    # Note about remaining vocab
    dot.node('remaining', '...49,992 more words\\n(combined P=0.02)',
             fillcolor='white', fontcolor=GRAY, style='rounded,dashed',
             fontsize='16', penwidth='2', width='4')
    dot.edge('question', 'remaining', style='dashed', color=GRAY, penwidth='1')

    dot.render('./setup_D_nodes', cleanup=True)
    print("OK Setup D: Node diagram (graphviz)")

# === RUN ALL 4 ===
if __name__ == '__main__':

    setup_chart_A_bars()
    setup_chart_B_list()
    setup_chart_C_curve()
    setup_chart_D_nodes()

    print("=" * 60)
    print("SUCCESS: 4 setup chart variations created!")
    print("  A: Bar chart (visual distribution)")
    print("  B: List format (clean text)")
    print("  C: Curve (power law decay)")
    print("  D: Node diagram (decision tree style)")
    print("=" * 60)
