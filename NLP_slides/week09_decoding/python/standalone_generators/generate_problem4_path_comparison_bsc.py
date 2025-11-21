"""
Problem 4: Path Comparison

Generated chart: problem4_path_comparison_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

import graphviz

COLOR_GREEN = '#2CA02C'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'

def generate_problem4_path_comparison_bsc():
    """Compare beam path vs optimal path."""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='LR', bgcolor='transparent')
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='10')

    # Taken path
    dot.node('taken', 'Path Taken by Beam\\n(greedy-ish)',
             shape='box', fillcolor='#FFD6D6', color=COLOR_RED, penwidth='2')
    dot.node('t1', 'The', fillcolor='#FFE6CC')
    dot.node('t2', 'cat', fillcolor='#FFE6CC')
    dot.node('t3', 'sat', fillcolor='#FFE6CC')
    dot.node('t4', 'down', fillcolor='#FFE6CC')
    dot.node('t_score', 'Score: 0.72\\n(locally optimal)',
             shape='ellipse', fillcolor='#FFD6D6', color=COLOR_RED)

    dot.edge('taken', 't1', color=COLOR_RED, penwidth='2')
    dot.edge('t1', 't2', color=COLOR_RED, penwidth='2')
    dot.edge('t2', 't3', color=COLOR_RED, penwidth='2')
    dot.edge('t3', 't4', color=COLOR_RED, penwidth='2')
    dot.edge('t4', 't_score', color=COLOR_RED, penwidth='2')

    # Optimal path
    dot.node('optimal', 'Optimal Path\\n(missed)',
             shape='box', fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')
    dot.node('o1', 'A', fillcolor='#E6F7E6')
    dot.node('o2', 'small', fillcolor='#E6F7E6')
    dot.node('o3', 'cat', fillcolor='#E6F7E6')
    dot.node('o4', 'purred', fillcolor='#E6F7E6')
    dot.node('o_score', 'Score: 0.89\\n(globally optimal)',
             shape='ellipse', fillcolor='#D4E6D4', color=COLOR_GREEN)

    dot.edge('optimal', 'o1', color=COLOR_GREEN, penwidth='2')
    dot.edge('o1', 'o2', color=COLOR_GREEN, penwidth='2')
    dot.edge('o2', 'o3', color=COLOR_GREEN, penwidth='2')
    dot.edge('o3', 'o4', color=COLOR_GREEN, penwidth='2')
    dot.edge('o4', 'o_score', color=COLOR_GREEN, penwidth='2')

    # Add QuantLet attribution
    dot.node('quantlet_attr', 'Code: quantlet.com/NLPDecoding_Problem4_Path_Comparison',
             shape='plaintext', fontsize='8', fontcolor='#888888')
    dot.render('./problem4_path_comparison_bsc', cleanup=True)
    print("Generated problem4_path_comparison_bsc.pdf")

if __name__ == "__main__":
    generate_problem4_path_comparison_bsc()
