"""
Generate Full Exploration Explosion Tree using Graphviz
Shows exponential explosion of paths when exploring all options.
Created: November 18, 2025
"""

import graphviz
import subprocess
import os

def create_full_exploration_tree():
    """Create full exploration explosion tree showing exponential growth."""

    # Create directed graph
    dot = graphviz.Digraph(comment='Full Exploration Explosion',
                          format='pdf',
                          engine='dot')

    # Graph attributes (30pt minimum fonts)
    dot.attr(rankdir='TB', size='14,10!', ratio='fill',
             fontname='Arial', fontsize='20',
             bgcolor='white', pad='0.5', nodesep='0.5', ranksep='0.6')

    # Node styles (30pt+ fonts)
    start_style = {'shape': 'circle', 'style': 'filled',
                   'fillcolor': '#3333B2', 'fontcolor': 'white',
                   'fontsize': '36', 'width': '1.5', 'penwidth': '5'}

    explored_style = {'shape': 'box', 'style': 'rounded,filled',
                     'fillcolor': '#D62728', 'fontcolor': 'white',
                     'fontsize': '24', 'height': '0.7', 'width': '1.2',
                     'penwidth': '3'}

    explosion_style = {'shape': 'box', 'style': 'rounded,filled,dashed',
                      'fillcolor': '#FFE0E0', 'fontcolor': '#D62728',
                      'fontsize': '20', 'height': '0.6', 'width': '1.0',
                      'penwidth': '2'}

    annotation_style = {'shape': 'none', 'fontsize': '26',
                       'fontcolor': '#808080'}

    # START node
    dot.node('START', 'START', **start_style)

    # Step 1: 100 branches (show subset)
    with dot.subgraph(name='cluster_step1') as c:
        c.attr(rank='same', label='Step 1: First Word (100 options)',
               fontsize='22', fontcolor='#3333B2', style='dashed',
               color='#3333B2', penwidth='3')

        # Show first few explicitly
        for i in range(5):
            c.node(f'w1_{i}', f'Word{i+1}', **explored_style)
            dot.edge('START', f'w1_{i}', color='#D62728', penwidth='2', arrowsize='0.8')

        # Show dots
        c.node('dots1', '...', **annotation_style)

        # Show last few
        for i in range(95, 100):
            c.node(f'w1_{i}', f'Word{i+1}', **explored_style)
            dot.edge('START', f'w1_{i}', color='#D62728', penwidth='2', arrowsize='0.8')

    # Step 2: Each word branches to 100 more (10,000 total)
    with dot.subgraph(name='cluster_step2') as c:
        c.attr(rank='same', label='Step 2: Second Word (100 × 100 = 10,000 paths)',
               fontsize='22', fontcolor='#D62728', style='dashed',
               color='#D62728', penwidth='3')

        # From Word1 show expansion
        for j in range(3):
            c.node(f'w2_0_{j}', f'1-{j+1}', **explosion_style)
            dot.edge('w1_0', f'w2_0_{j}', color='#FFB0B0', penwidth='1.5',
                    arrowsize='0.6', style='dashed')

        c.node('exp1', '...97 more', shape='plaintext', fontsize='18',
               fontcolor='#D62728')

        # From Word2 show expansion
        for j in range(2):
            c.node(f'w2_1_{j}', f'2-{j+1}', **explosion_style)
            dot.edge('w1_1', f'w2_1_{j}', color='#FFB0B0', penwidth='1.5',
                    arrowsize='0.6', style='dashed')

        c.node('exp2', '...98 more', shape='plaintext', fontsize='18',
               fontcolor='#D62728')

        # Middle explosion indicator
        c.node('explosion', '× 100 for\neach word!', shape='ellipse',
               style='filled', fillcolor='#FFCCCC', fontcolor='#D62728',
               fontsize='24', penwidth='3')

        # From Word100 show expansion
        for j in range(2):
            c.node(f'w2_99_{j}', f'100-{j+1}', **explosion_style)
            dot.edge('w1_99', f'w2_99_{j}', color='#FFB0B0', penwidth='1.5',
                    arrowsize='0.6', style='dashed')

        c.node('exp100', '...98 more', shape='plaintext', fontsize='18',
               fontcolor='#D62728')

    # Step 3: Exponential explosion continues
    with dot.subgraph(name='cluster_step3') as c:
        c.attr(rank='same', label='Step 3: Third Word (100³ = 1,000,000 paths)',
               fontsize='22', fontcolor='#D62728', style='bold',
               color='#D62728', penwidth='4')

        # Show massive expansion with ellipsis
        c.node('massive1', '...', shape='plaintext', fontsize='24',
               fontcolor='#D62728', fontweight='bold')
        c.node('massive2', '1 Million\nPaths', shape='box',
               style='rounded,filled', fillcolor='#D62728',
               fontcolor='white', fontsize='18', penwidth='4',
               height='1.2', width='2.0')
        c.node('massive3', '...', shape='plaintext', fontsize='24',
               fontcolor='#D62728', fontweight='bold')

        # Connect some paths
        dot.edge('w2_0_0', 'massive1', color='#FFB0B0', penwidth='1',
                arrowsize='0.5', style='dotted')
        dot.edge('w2_1_0', 'massive2', color='#FFB0B0', penwidth='1',
                arrowsize='0.5', style='dotted')

    # Final explosion indicator
    with dot.subgraph(name='cluster_final') as c:
        c.attr(rank='same', label='Continue to Step 5...',
               fontsize='18', fontcolor='#808080', style='dotted',
               color='#808080', penwidth='2')

        c.node('final', '100⁵ = 10 BILLION PATHS!\\nComputationally Infeasible',
               shape='box', style='rounded,filled,bold',
               fillcolor='#D62728', fontcolor='white',
               fontsize='22', height='1.5', width='6.0',
               penwidth='6')

    dot.edge('massive2', 'final', label='continues...', color='#D62728',
             penwidth='3', arrowsize='1.0', style='bold',
             fontsize='24', fontcolor='#D62728')

    # Warning box removed for cleaner visualization

    # Render to PDF
    output_path = '../figures/full_exploration_tree_graphviz'
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
        create_full_exploration_tree()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Graphviz not found. Please install graphviz:")
        print("  Windows: choco install graphviz  OR  download from graphviz.org")
        print("  After install, add to PATH and restart terminal")
        exit(1)