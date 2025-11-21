"""
Problem 4: Probability Evolution

Generated chart: problem4_probability_evolution_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

import graphviz

COLOR_ACCENT = '#3333B2'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'
COLOR_GRAY = '#7F7F7F'
COLOR_MAIN = '#333333'

def generate_problem4_probability_evolution_bsc():
    """Show cumulative probability loss."""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='LR', bgcolor='transparent')
    dot.attr('node', shape='record', style='filled', fontname='Arial', fontsize='10')

    dot.node('t0', '{<f0> START | <f1> P=1.0}',
             fillcolor='#E8E8F0', color=COLOR_ACCENT)

    dot.node('t1', '{<f0> Step 1 | <f1> "The": 0.6 | <f2> "A": 0.3 (pruned)}',
             fillcolor='#FFE6CC', color=COLOR_ORANGE)

    dot.node('t2', '{<f0> Step 2 | <f1> "The cat": 0.24 | <f2> "A small": 0.12 (lost!)}',
             fillcolor='#FFD6D6', color=COLOR_RED)

    dot.node('t3', '{<f0> Step 3 | <f1> "The cat sat": 0.096 | <f2> "A small cat": 0.048 (lost!)}',
             fillcolor='#FFD6D6', color=COLOR_RED)

    dot.node('result', 'Early pruning causes\\ncumulative probability loss',
             shape='note', fillcolor='#FFFFCC', color=COLOR_MAIN)

    dot.edge('t0', 't1', color=COLOR_ACCENT, penwidth='2')
    dot.edge('t1', 't2', color=COLOR_ORANGE, penwidth='2')
    dot.edge('t2', 't3', color=COLOR_RED, penwidth='2')
    dot.edge('t3', 'result', style='dashed', color=COLOR_GRAY)

    # Add QuantLet attribution
    dot.node('quantlet_attr', 'Code: quantlet.com/NLPDecoding_Problem4_Probability_Evolution',
             shape='plaintext', fontsize='8', fontcolor='#888888')
    dot.render('./problem4_probability_evolution_bsc', cleanup=True)
    print("Generated problem4_probability_evolution_bsc.pdf")

if __name__ == "__main__":
    generate_problem4_probability_evolution_bsc()
