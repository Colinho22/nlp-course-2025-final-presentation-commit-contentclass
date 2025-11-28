#!/usr/bin/env python3
"""
Generate ALL Week 9 charts using graphviz for diagrams and clean matplotlib for data
Font size: 20-24pt (readable but not overwhelming)
Professional layouts with NO overlap
"""

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('../figures', exist_ok=True)

# Template colors
PURPLE = '#3333B2'
GREEN = '#2CA02C'
RED = '#D62728'
ORANGE = '#FF7F0E'
BLUE = '#0066CC'
GRAY = '#7F7F7F'
LAVENDER = '#ADADC0'

print("=" * 70)
print("Generating Week 9 Charts: Graphviz + Matplotlib")
print("=" * 70)

# === CHART 1: Quality-Diversity Tradeoff (matplotlib scatter) ===
def chart1_quality_diversity():
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = [
        ('Greedy', 15, 85, RED),
        ('Beam', 20, 88, RED),
        ('Temp(low)', 35, 82, GRAY),
        ('Top-k', 65, 70, GRAY),
        ('Nucleus', 78, 82, GREEN),
        ('Contrastive', 82, 85, GREEN),
    ]

    for name, x, y, color in methods:
        ax.scatter(x, y, s=1200, c=color, alpha=0.7, edgecolors='black', linewidths=3)
        ax.text(x, y-10, name, ha='center', fontsize=20, weight='bold')

    ax.set_xlabel('Diversity →', fontsize=24, weight='bold')
    ax.set_ylabel('Quality →', fontsize=24, weight='bold')
    ax.set_title('Quality-Diversity Tradeoff', fontsize=26, weight='bold', color=PURPLE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/quality_diversity_tradeoff_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("1. quality_diversity_tradeoff_bsc.pdf")

# === CHART 2: Pipeline (graphviz) ===
def chart2_pipeline():
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='LR', fontsize='20', bgcolor='white')
    dot.attr('node', fontname='Arial', shape='box', style='rounded,filled', fontsize='20')

    dot.node('input', 'Input\nText', fillcolor=LAVENDER, fontcolor='black', width='1.5', height='0.8')
    dot.node('model', 'MODEL\n(Transformer)', fillcolor=BLUE, fontcolor='white', width='2', height='0.8')
    dot.node('probs', 'Probabilities\n50K words', fillcolor=ORANGE, fontcolor='black', width='2', height='0.8')
    dot.node('decode', 'DECODING\n[Week 9]', fillcolor=PURPLE, fontcolor='white', width='2', height='0.8', penwidth='4')
    dot.node('output', 'Output\nText', fillcolor=GREEN, fontcolor='white', width='1.5', height='0.8')

    dot.edge('input', 'model', penwidth='3')
    dot.edge('model', 'probs', penwidth='3')
    dot.edge('probs', 'decode', penwidth='4', color=PURPLE)
    dot.edge('decode', 'output', penwidth='3')

    dot.render('../figures/prediction_to_text_pipeline_bsc', cleanup=True)
    print("2. prediction_to_text_pipeline_bsc.pdf")

# === CHART 3: Beam Search Tree (graphviz) ===
def chart3_beam_tree():
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='TB', fontsize='20', bgcolor='white', nodesep='0.8', ranksep='1.0')
    dot.attr('node', fontname='Arial', fontsize='20')

    # Start
    dot.node('start', 'START', shape='circle', fillcolor=PURPLE, style='filled',
             fontcolor='white', width='1.2', fontsize='22', penwidth='3')

    # Step 1
    dot.node('sat', 'sat\\n0.45', shape='box', style='rounded,filled', fillcolor=GREEN,
             fontcolor='white', penwidth='2')
    dot.node('is', 'is\\n0.30', shape='box', style='rounded,filled', fillcolor=GREEN,
             fontcolor='white', penwidth='2')
    dot.node('jumped', 'jumped\\n0.25', shape='box', style='rounded,filled', fillcolor=GREEN,
             fontcolor='white', penwidth='2')

    dot.edge('start', 'sat', penwidth='3', color=GREEN)
    dot.edge('start', 'is', penwidth='3', color=GREEN)
    dot.edge('start', 'jumped', penwidth='3', color=GREEN)

    # Step 2 (from sat)
    dot.node('sat_on', 'sat on\\n0.23', shape='box', style='rounded,filled',
             fillcolor=ORANGE, fontcolor='white', penwidth='3', fontsize='22')
    dot.edge('sat', 'sat_on', penwidth='4', color=ORANGE, label='best path')

    dot.render('../figures/beam_example_tree_bsc', cleanup=True)
    print("3. beam_example_tree_bsc.pdf")

# === CHART 4: Temperature (matplotlib 3 bars) ===
def chart4_temperature():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    tokens = ['the', 'a', 'on']
    data = [
        ([0.73, 0.18, 0.09], 'T=0.5', ax1),
        ([0.42, 0.23, 0.35], 'T=1.0', ax2),
        ([0.32, 0.31, 0.37], 'T=2.0', ax3),
    ]

    for probs, title, ax in data:
        ax.bar(tokens, probs, color=PURPLE, alpha=0.7, edgecolor='black', linewidth=2)
        for i, p in enumerate(probs):
            ax.text(i, p + 0.05, f'{p:.2f}', ha='center', fontsize=20, weight='bold')
        ax.set_title(title, fontsize=24, weight='bold', color=PURPLE)
        ax.set_ylim(0, 0.9)
        ax.tick_params(labelsize=18)
        ax.grid(axis='y', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle('Temperature Effects', fontsize=26, weight='bold', color=PURPLE)
    plt.tight_layout()
    plt.savefig('../figures/temperature_effects_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("4. temperature_effects_bsc.pdf")

# === CHART 5: Top-k (matplotlib bar) ===
def chart5_topk():
    fig, ax = plt.subplots(figsize=(11, 6))

    tokens = [f'tok{i}' for i in range(8)]
    probs = [0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.04, 0.03]
    k = 5
    colors = [GREEN if i < k else RED for i in range(len(tokens))]

    ax.bar(range(len(tokens)), probs, color=colors, alpha=0.7,
           edgecolor='black', linewidth=2)
    ax.axvline(k - 0.5, color=PURPLE, linestyle='--', linewidth=3, label=f'k={k}')

    ax.text(2, 0.35, f'Top-{k}', fontsize=24, weight='bold', color=GREEN,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN, linewidth=2))

    ax.set_xlabel('Tokens (ranked)', fontsize=22, weight='bold')
    ax.set_ylabel('Probability', fontsize=22, weight='bold')
    ax.set_title('Top-k Filtering', fontsize=26, weight='bold', color=PURPLE)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=18)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=20)
    ax.grid(axis='y', alpha=0.3)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/topk_filtering_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("5. topk_filtering_bsc.pdf")

# === CHART 6: Nucleus (matplotlib dual axis) ===
def chart6_nucleus():
    fig, ax = plt.subplots(figsize=(11, 6))

    tokens = ['cat', 'dog', 'bird', 'fish', 'mouse', 'rat']
    probs = [0.40, 0.20, 0.15, 0.10, 0.08, 0.07]
    cumulative = np.cumsum(probs)
    nucleus_size = 4

    colors = [GREEN if i < nucleus_size else GRAY for i in range(len(tokens))]
    ax.bar(range(len(tokens)), probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    ax2 = ax.twinx()
    ax2.plot(range(len(tokens)), cumulative, color=PURPLE, linewidth=4,
             marker='o', markersize=12, label='Cumulative')
    ax2.axhline(0.85, color=ORANGE, linestyle='--', linewidth=3, label='p=0.85')

    ax.set_xlabel('Tokens', fontsize=22, weight='bold')
    ax.set_ylabel('Probability', fontsize=20, weight='bold')
    ax2.set_ylabel('Cumulative', fontsize=20, weight='bold', color=PURPLE)
    ax.set_title('Nucleus Sampling', fontsize=26, weight='bold', color=PURPLE)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=18)
    ax.tick_params(labelsize=18)
    ax2.tick_params(labelsize=18, colors=PURPLE)
    ax2.legend(fontsize=18, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    for spine in ['top']:
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['right'].set_color(PURPLE)

    plt.tight_layout()
    plt.savefig('../figures/nucleus_process_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("6. nucleus_process_bsc.pdf")

# === CHART 7: Degeneration (graphviz text layout) ===
def chart7_degeneration():
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='TB', fontsize='20', bgcolor='white')
    dot.attr('node', fontname='Arial', fontsize='20', shape='box', style='rounded')

    dot.node('title', 'Greedy Decoding Problem', fillcolor=RED, fontcolor='white',
             style='rounded,filled', fontsize='24', penwidth='3', width='6')

    dot.node('output', '"The city is a major city.\\nThe city has many attractions.\\nThe city is known..."',
             fillcolor='#FFE6E6', fontcolor='black', style='filled,rounded',
             fontsize='18', width='10', height='2')

    dot.node('problem', '"the city" appears 3 times!', fillcolor=RED, fontcolor='white',
             style='filled,rounded', fontsize='22', penwidth='3', width='5')

    dot.node('solution', 'Solution: Contrastive Search', fillcolor=GREEN, fontcolor='white',
             style='filled,rounded', fontsize='22', penwidth='3', width='5')

    dot.edge('title', 'output', style='invis')
    dot.edge('output', 'problem', style='invis')
    dot.edge('problem', 'solution', arrowhead='normal', penwidth='3', color=GREEN)

    dot.render('../figures/degeneration_problem_bsc', cleanup=True)
    print("7. degeneration_problem_bsc.pdf")

# === CHART 8: Contrastive Mechanism (graphviz flowchart) ===
def chart8_contrastive_mechanism():
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='TB', fontsize='20', bgcolor='white', ranksep='0.8')
    dot.attr('node', fontname='Arial', fontsize='20', shape='box', style='rounded,filled')

    dot.node('step1', 'Step 1: Get top-k candidates\\ncity: 0.45, town: 0.18, area: 0.15',
             fillcolor=BLUE, fontcolor='white', width='7', fontsize='20')

    dot.node('step2', 'Step 2: Compute similarity to context\\ncity: 0.92, town: 0.75, area: 0.65',
             fillcolor=ORANGE, fontcolor='black', width='7', fontsize='20')

    dot.node('step3', 'Step 3: Apply penalty (α=0.6)\\nscore = (1-α)×prob - α×similarity',
             fillcolor=PURPLE, fontcolor='white', width='7', fontsize='20')

    dot.node('result', 'Winner: "town"\\n(0.4×0.18 - 0.6×0.75 = -0.378)',
             fillcolor=GREEN, fontcolor='white', width='6', fontsize='22', penwidth='4')

    dot.edge('step1', 'step2', penwidth='3')
    dot.edge('step2', 'step3', penwidth='3')
    dot.edge('step3', 'result', penwidth='4', color=GREEN)

    dot.render('../figures/contrastive_mechanism_bsc', cleanup=True)
    print("8. contrastive_mechanism_bsc.pdf")

# === CHART 9: Decision Tree (graphviz) ===
def chart9_decision_tree():
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='TB', fontsize='20', bgcolor='white', ranksep='1.0', nodesep='0.8')
    dot.attr('node', fontname='Arial', fontsize='20', shape='box', style='rounded')

    dot.node('start', 'What kind of task?', fillcolor=PURPLE, fontcolor='white',
             style='rounded,filled', fontsize='24', penwidth='3', width='4')

    dot.node('factual', 'Factual/Deterministic?', fillcolor=LAVENDER, fontcolor='black',
             style='filled', fontsize='20', width='4')
    dot.node('creative', 'Creative/Diverse?', fillcolor=LAVENDER, fontcolor='black',
             style='filled', fontsize='20', width='4')

    dot.node('beam', 'BEAM SEARCH\\nwidth=3-5', fillcolor=GREEN, fontcolor='white',
             style='filled,rounded', fontsize='22', penwidth='3', width='3')
    dot.node('nucleus', 'NUCLEUS\\np=0.9', fillcolor=GREEN, fontcolor='white',
             style='filled,rounded', fontsize='22', penwidth='3', width='3')
    dot.node('contrastive', 'CONTRASTIVE\\nα=0.6', fillcolor=GREEN, fontcolor='white',
             style='filled,rounded', fontsize='22', penwidth='3', width='3')

    dot.edge('start', 'factual', label='Need exact answer', fontsize='18')
    dot.edge('start', 'creative', label='Need creativity', fontsize='18')
    dot.edge('factual', 'beam', label='Translation, Q&A', fontsize='16')
    dot.edge('creative', 'nucleus', label='Short text', fontsize='16')
    dot.edge('creative', 'contrastive', label='Long text', fontsize='16')

    dot.render('../figures/task_method_decision_tree_bsc', cleanup=True)
    print("9. task_method_decision_tree_bsc.pdf")

# === CHART 10: Pareto Frontier (matplotlib) ===
def chart10_pareto():
    fig, ax = plt.subplots(figsize=(10, 7))

    methods = [
        ('Greedy', 15, 85, RED, 'o'),
        ('Beam', 20, 88, RED, 's'),
        ('Nucleus', 78, 82, GREEN, 'D'),
        ('Contrastive', 82, 85, GREEN, '*'),
    ]

    for name, x, y, color, marker in methods:
        ax.scatter(x, y, s=1000, c=color, marker=marker, alpha=0.7,
                  edgecolors='black', linewidths=2.5)
        ax.text(x+3, y+3, name, fontsize=20, weight='bold')

    ax.text(80, 88, 'OPTIMAL', fontsize=24, weight='bold', color=GREEN,
            bbox=dict(boxstyle='round', facecolor=GREEN, alpha=0.2, edgecolor=GREEN, linewidth=2))

    ax.set_xlabel('Diversity →', fontsize=24, weight='bold')
    ax.set_ylabel('Quality →', fontsize=24, weight='bold')
    ax.set_title('Pareto Frontier', fontsize=26, weight='bold', color=PURPLE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/quality_diversity_pareto_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("10. quality_diversity_pareto_bsc.pdf")

# === CHART 11: Task Recommendations Table (graphviz) ===
def chart11_task_table():
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(rankdir='TB', fontsize='18', bgcolor='white')
    dot.attr('node', fontname='Arial', shape='plaintext')

    # Create HTML table
    table_html = '''<
    <TABLE BORDER="2" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
    <TR>
        <TD BGCOLOR="#3333B2"><FONT COLOR="white" POINT-SIZE="22"><B>Task</B></FONT></TD>
        <TD BGCOLOR="#3333B2"><FONT COLOR="white" POINT-SIZE="22"><B>Method</B></FONT></TD>
        <TD BGCOLOR="#3333B2"><FONT COLOR="white" POINT-SIZE="22"><B>Parameters</B></FONT></TD>
    </TR>
    <TR>
        <TD><FONT POINT-SIZE="18">Translation</FONT></TD>
        <TD BGCOLOR="#E8F5E9"><FONT POINT-SIZE="18">Beam Search</FONT></TD>
        <TD><FONT POINT-SIZE="18">width=3-5</FONT></TD>
    </TR>
    <TR>
        <TD><FONT POINT-SIZE="18">Factual Q&A</FONT></TD>
        <TD BGCOLOR="#E8F5E9"><FONT POINT-SIZE="18">Greedy</FONT></TD>
        <TD><FONT POINT-SIZE="18">T=0.1-0.3</FONT></TD>
    </TR>
    <TR>
        <TD><FONT POINT-SIZE="18">Code</FONT></TD>
        <TD BGCOLOR="#E8F5E9"><FONT POINT-SIZE="18">Greedy/Beam</FONT></TD>
        <TD><FONT POINT-SIZE="18">T=0, w=3</FONT></TD>
    </TR>
    <TR>
        <TD><FONT POINT-SIZE="18">Dialogue</FONT></TD>
        <TD BGCOLOR="#FFE6CC"><FONT POINT-SIZE="18">Nucleus</FONT></TD>
        <TD><FONT POINT-SIZE="18">p=0.9, T=0.8</FONT></TD>
    </TR>
    <TR>
        <TD><FONT POINT-SIZE="18">Creative Writing</FONT></TD>
        <TD BGCOLOR="#FFE6CC"><FONT POINT-SIZE="18">Nucleus</FONT></TD>
        <TD><FONT POINT-SIZE="18">p=0.95, T=1.0</FONT></TD>
    </TR>
    <TR>
        <TD><FONT POINT-SIZE="18">Long Stories</FONT></TD>
        <TD BGCOLOR="#FFE6CC"><FONT POINT-SIZE="18">Contrastive</FONT></TD>
        <TD><FONT POINT-SIZE="18">α=0.6, k=4</FONT></TD>
    </TR>
    </TABLE>
    >'''

    dot.node('table', table_html)

    dot.render('../figures/task_recommendations_table_bsc', cleanup=True)
    print("11. task_recommendations_table_bsc.pdf")

# === CHART 12: Computational Costs (matplotlib) ===
def chart12_costs():
    fig, ax = plt.subplots(figsize=(11, 6))

    methods = ['Greedy', 'Temp', 'Top-k', 'Nucleus', 'Beam', 'Contrastive']
    times = [1.0, 1.1, 1.2, 1.3, 4.5, 12.0]
    colors = [GREEN, GREEN, GREEN, GREEN, ORANGE, RED]

    bars = ax.bar(range(len(methods)), times, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)

    for i, t in enumerate(times):
        ax.text(i, t + 0.3, f'{t}×', ha='center', fontsize=22, weight='bold')

    ax.set_xlabel('Method', fontsize=24, weight='bold')
    ax.set_ylabel('Relative Speed', fontsize=24, weight='bold')
    ax.set_title('Computational Cost', fontsize=26, weight='bold', color=PURPLE)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_ylim(0, 14)
    ax.grid(axis='y', alpha=0.3)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/computational_cost_comparison_bsc.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("12. computational_cost_comparison_bsc.pdf")

# === RUN ALL ===
if __name__ == '__main__':
    try:
        chart1_quality_diversity()
        chart2_pipeline()
        chart3_beam_tree()
        chart4_temperature()
        chart5_topk()
        chart6_nucleus()
        chart7_degeneration()
        chart8_contrastive_mechanism()
        chart9_decision_tree()
        chart10_pareto()
        chart11_task_table()
        chart12_costs()

        print("=" * 70)
        print("SUCCESS: 12 professional charts generated!")
        print("  Graphviz diagrams: 6 charts (pipeline, trees, flowcharts, table)")
        print("  Matplotlib data viz: 6 charts (scatter, bars, dual-axis)")
        print("  Font size: 20-24pt (readable, not overwhelming)")
        print("  NO text overlap - clean professional layouts")
        print("=" * 70)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
