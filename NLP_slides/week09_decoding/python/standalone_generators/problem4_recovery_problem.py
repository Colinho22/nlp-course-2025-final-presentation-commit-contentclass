"""
Problem 4: Recovery Problem

Generated chart: problem4_recovery_problem_bsc.pdf
Part of Week 9: NLP Decoding Strategies
"""

# QuantLet Metadata
QUANTLET_URL = "https://quantlet.com/NLPDecoding_Problem4Recoveryproblem"


import graphviz

COLOR_ACCENT = '#3333B2'
COLOR_ORANGE = '#FF7F0E'
COLOR_RED = '#D62728'
COLOR_GRAY = '#7F7F7F'
COLOR_MAIN = '#333333'

def generate_problem4_recovery_problem_bsc():
    """Show irreversibility of beam search pruning."""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='transparent')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='11')

    dot.node('step1', 'Step 1: Prune "A"\\n(P=0.3)',
             fillcolor='#FFD6D6', color=COLOR_RED, penwidth='2')

    dot.node('step2', 'Step 2: Continue with "The"\\nAll descendants are "The ..."',
             fillcolor='#FFE6CC', color=COLOR_ORANGE)

    dot.node('step3', 'Step 3-10: Still following\\n"The cat sat ..."',
             fillcolor='#FFE6CC', color=COLOR_ORANGE)

    dot.node('realize', 'Realize: "A small cat purred"\\nwould have been better!',
             fillcolor='#FFFFCC', color=COLOR_MAIN, shape='ellipse')

    dot.node('problem', 'PROBLEM:\\nCannot recover!\\nPath starting with "A" is gone forever.',
             fillcolor='#FFD6D6', color=COLOR_RED, penwidth='3', shape='box')

    dot.edge('step1', 'step2', color=COLOR_RED, penwidth='2')
    dot.edge('step2', 'step3', color=COLOR_ORANGE, penwidth='2')
    dot.edge('step3', 'realize', color=COLOR_GRAY, style='dashed')
    dot.edge('realize', 'problem', color=COLOR_RED, penwidth='2',
             label='No way back!', fontcolor=COLOR_RED, fontsize='10')

    dot.node('note', 'Beam search is GREEDY:\\nOnce pruned, paths are lost forever',
             shape='note', fillcolor='#E8E8F0', color=COLOR_ACCENT)

    # Add QuantLet attribution
    dot.node('quantlet_attr', 'Code: quantlet.com/NLPDecoding_Problem4_Recovery_Problem',
             shape='plaintext', fontsize='8', fontcolor='#888888')
    dot.render('./problem4_recovery_problem_bsc', cleanup=True)
    print("Generated problem4_recovery_problem_bsc.pdf")

if __name__ == "__main__":
    generate_problem4_recovery_problem_bsc()
