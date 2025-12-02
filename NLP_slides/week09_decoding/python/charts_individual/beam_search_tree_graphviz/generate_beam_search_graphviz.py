"""
Generate beam search tree visualization using graphviz.
Shows beam_size=3 with pruning at each step.
"""

import graphviz
import subprocess
import os

def create_beam_search_tree():
    """Create beam search tree with graphviz showing pruning."""

    # Create directed graph
    dot = graphviz.Digraph(comment='Beam Search Tree',
                          format='pdf',
                          engine='dot')

    # Graph attributes (30pt minimum fonts)
    dot.attr(rankdir='TB', size='14,10!', ratio='fill',
             fontname='Arial', fontsize='20',
             bgcolor='white', pad='0.5', nodesep='0.6', ranksep='0.8')

    # Node styles (30pt+ fonts)
    start_style = {'shape': 'circle', 'style': 'filled',
                   'fillcolor': '#3333B2', 'fontcolor': 'white',
                   'fontsize': '36', 'width': '1.5', 'penwidth': '5'}

    kept_style = {'shape': 'box', 'style': 'rounded,filled',
                  'fillcolor': '#2CA02C', 'fontcolor': 'white',
                  'fontsize': '30', 'height': '0.9', 'width': '2.2',
                  'penwidth': '4'}

    pruned_style = {'shape': 'box', 'style': 'rounded,filled,dashed',
                    'fillcolor': '#E0E0E0', 'fontcolor': '#808080',
                    'fontsize': '28', 'height': '0.8', 'width': '2.0',
                    'penwidth': '3'}

    final_style = {'shape': 'box', 'style': 'rounded,filled',
                   'fillcolor': '#FF7F0E', 'fontcolor': 'white',
                   'fontsize': '32', 'height': '1.0', 'width': '4.0',
                   'penwidth': '5'}

    # START node
    dot.node('START', 'START', **start_style)

    # Step 1: Generate first word (5 candidates, keep top 3)
    with dot.subgraph(name='cluster_step1') as c:
        c.attr(rank='same', label='Step 1: First Word (keep top 3)',
               fontsize='22', fontcolor='#3333B2', style='dashed',
               color='#3333B2', penwidth='3')

        # Top 3 (kept)
        c.node('The', 'The\\nP=0.40\\nlog=-0.92', **kept_style)
        c.node('A', 'A\\nP=0.30\\nlog=-1.20', **kept_style)
        c.node('I', 'I\\nP=0.20\\nlog=-1.61', **kept_style)

        # Pruned (lower probability)
        c.node('It', 'It\\nP=0.10', **pruned_style)
        c.node('One', 'One\\nP=0.05', **pruned_style)

    # Edges from START to Step 1
    dot.edge('START', 'The', label='', color='#27AE60', penwidth='2.5', arrowsize='1.0')
    dot.edge('START', 'A', label='', color='#27AE60', penwidth='2.5', arrowsize='1.0')
    dot.edge('START', 'I', label='', color='#27AE60', penwidth='2.5', arrowsize='1.0')
    dot.edge('START', 'It', label='', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')
    dot.edge('START', 'One', label='', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')

    # Step 2: Expand each of top 3 (9 candidates total, keep top 3)
    with dot.subgraph(name='cluster_step2') as c:
        c.attr(rank='same', label='Step 2: Second Word (9 candidates → keep top 3)',
               fontsize='22', fontcolor='#3333B2', style='dashed',
               color='#3333B2', penwidth='3')

        # From "The" (kept)
        c.node('The_cat', 'The cat\\nsum=-1.61', **kept_style)
        c.node('The_dog', 'The dog\\nsum=-2.12', **pruned_style)
        c.node('The_sun', 'The sun\\nsum=-2.53', **pruned_style)

        # From "A" (kept)
        c.node('A_nice', 'A nice\\nsum=-2.12', **kept_style)
        c.node('A_big', 'A big\\nsum=-2.25', **pruned_style)
        c.node('A_small', 'A small\\nsum=-2.59', **pruned_style)

        # From "I" (kept)
        c.node('I_think', 'I think\\nsum=-2.12', **kept_style)
        c.node('I_know', 'I know\\nsum=-3.00', **pruned_style)
        c.node('I_see', 'I see\\nsum=-3.51', **pruned_style)

    # Edges from Step 1 to Step 2
    # From "The"
    dot.edge('The', 'The_cat', color='#27AE60', penwidth='2.5', arrowsize='1.0')
    dot.edge('The', 'The_dog', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')
    dot.edge('The', 'The_sun', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')

    # From "A"
    dot.edge('A', 'A_nice', color='#27AE60', penwidth='2.5', arrowsize='1.0')
    dot.edge('A', 'A_big', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')
    dot.edge('A', 'A_small', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')

    # From "I"
    dot.edge('I', 'I_think', color='#27AE60', penwidth='2.5', arrowsize='1.0')
    dot.edge('I', 'I_know', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')
    dot.edge('I', 'I_see', color='#C0C0C0', penwidth='1', arrowsize='0.7', style='dashed')

    # Step 3: Final output (showing only top 3 complete sequences)
    with dot.subgraph(name='cluster_step3') as c:
        c.attr(rank='same', label='Step 3: Continue... → Final Selection',
               fontsize='22', fontcolor='#FF7F0E', style='dashed',
               color='#FF7F0E', penwidth='3')

        c.node('final', 'Best: "The cat sat on the mat"\\nFinal Score: -3.20', **final_style)

    # Show that top 3 continue (simplified view)
    dot.edge('The_cat', 'final', label='continues...', color='#F39C12',
             penwidth='3', arrowsize='1.2', style='bold')
    dot.edge('A_nice', 'final', label='', color='#E0E0E0',
             penwidth='1.5', arrowsize='0.8', style='dashed')
    dot.edge('I_think', 'final', label='', color='#E0E0E0',
             penwidth='1.5', arrowsize='0.8', style='dashed')

    # Add legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='', style='invisible')
        c.node('legend', '''Legend:
• Green (solid) = Kept in beam
• Gray (dashed) = Pruned
• Orange = Final winner
• At each step: keep top 3''',
               shape='box', style='rounded',
               fontsize='20', fontname='Arial',
               fillcolor='#FFFACD', color='black', penwidth='2')

    # Render to PDF
    output_path = '../figures/beam_search_tree_graphviz'
    dot.render(output_path, cleanup=True, view=False)

    print(f"Generated {output_path}.pdf using graphviz")

    # Also save the .dot source for reference
    with open(output_path + '.dot', 'w', encoding='utf-8') as f:
        f.write(dot.source)
    print(f"Saved graphviz source to {output_path}.dot")


if __name__ == "__main__":
    # Check if graphviz is available
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
        create_beam_search_tree()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Graphviz not found. Please install graphviz:")
        print("  Windows: choco install graphviz  OR  download from graphviz.org")
        print("  After install, add to PATH and restart terminal")
        exit(1)
