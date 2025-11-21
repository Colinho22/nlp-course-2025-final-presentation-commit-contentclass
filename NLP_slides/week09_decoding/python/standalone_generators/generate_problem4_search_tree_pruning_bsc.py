"""
Problem 4: Search Tree Pruning

Generated chart: problem4_search_tree_pruning_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

import graphviz

COLOR_ACCENT = '#3333B2'
COLOR_GREEN = '#2CA02C'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'
COLOR_MAIN = '#333333'

def generate_problem4_search_tree_pruning_bsc():
    """Graphviz tree showing beam search pruning."""
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
    dot.edge('start', 'A3', label='Prune!', color=COLOR_RED, style='dashed', penwidth='1')

    # Step 2
    dot.node('B1', 'cat (0.4)', fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')
    dot.node('B2', 'dog (0.2)', fillcolor='#FFE6CC', color=COLOR_ORANGE)
    dot.node('B3', 'quick (0.05)', fillcolor='#FFD6D6', color=COLOR_RED)
    dot.node('B4', 'red (0.25)', fillcolor='#FFD6D6', color=COLOR_RED)

    dot.edge('A1', 'B1', color=COLOR_GREEN, penwidth='2')
    dot.edge('A1', 'B2', color=COLOR_ORANGE, penwidth='1.5')
    dot.edge('A2', 'B3', label='Lost!', color=COLOR_RED, style='dashed')
    dot.edge('A2', 'B4', label='Lost!', color=COLOR_RED, style='dashed')

    # Note
    dot.node('note', '"In quick" had high probability\\nbut pruned at step 1!',
            shape='note', fillcolor='#FFFFCC', color=COLOR_MAIN)

    dot.render('./problem4_search_tree_pruning_bsc', cleanup=True)
    print("Generated problem4_search_tree_pruning_bsc.pdf")

if __name__ == "__main__":
    generate_problem4_search_tree_pruning_bsc()
