"""
Generate Beam Search Graphviz Chart for Week 9 Decoding
Created: November 16, 2025

Simple 3-step beam search example showing:
- Beam width k=3
- Actual words with probabilities
- Compact, readable layout
- Color coding for kept vs pruned paths
"""

import graphviz
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)

# BSc Discovery Color Scheme
COLOR_MAIN = '#404040'      # Main elements
COLOR_ACCENT = '#3333B2'    # Key concepts (purple)
COLOR_GREEN = '#2CA02C'     # Kept paths
COLOR_RED = '#D62728'       # Pruned paths
COLOR_GRAY = '#B4B4B4'      # Secondary


def generate_beam_search_example():
    """
    Generate compact beam search example with k=3.
    Shows 3 time steps: "The" -> "cat" -> "sat"
    """
    print("Generating beam_search_example_graphviz_bsc.pdf...")

    # Create directed graph with compact layout
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', bgcolor='transparent', rankdir='LR')
    dot.attr('node', fontname='Arial', fontsize='11', margin='0.2,0.1')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # Configure layout for compactness
    dot.attr(nodesep='0.3', ranksep='0.8')

    # START node
    dot.node('start', 'START\nP=1.0',
            shape='box', style='rounded,filled',
            fillcolor='#E8E8F0', color=COLOR_ACCENT, penwidth='2')

    # ============================================
    # STEP 1: First word (keep top 3)
    # ============================================

    # Top 3 candidates (kept)
    dot.node('the1', 'The\nP=0.60',
            shape='box', style='rounded,filled',
            fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')

    dot.node('a1', 'A\nP=0.30',
            shape='box', style='rounded,filled',
            fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')

    dot.node('in1', 'In\nP=0.05',
            shape='box', style='rounded,filled',
            fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')

    # Pruned candidates (shown grayed)
    dot.node('pruned1', '[47 others\npruned]',
            shape='box', style='dashed,filled',
            fillcolor='#FFE6E6', color=COLOR_RED, penwidth='1')

    # Edges from START
    dot.edge('start', 'the1', label='  0.60', color=COLOR_GREEN, penwidth='2')
    dot.edge('start', 'a1', label='  0.30', color=COLOR_GREEN, penwidth='1.5')
    dot.edge('start', 'in1', label='  0.05', color=COLOR_GREEN, penwidth='1')
    dot.edge('start', 'pruned1', style='dashed', color=COLOR_RED)

    # ============================================
    # STEP 2: Second word (keep top 3 paths)
    # ============================================

    # From "The" (0.60)
    dot.node('the_cat', 'The cat\nP=0.36',
            shape='box', style='rounded,filled',
            fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')

    dot.node('the_dog', 'The dog\nP=0.24',
            shape='box', style='rounded,filled',
            fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')

    # From "A" (0.30)
    dot.node('a_cat', 'A cat\nP=0.18',
            shape='box', style='rounded,filled',
            fillcolor='#D4E6D4', color=COLOR_GREEN, penwidth='2')

    # Pruned paths
    dot.node('pruned2', '[other\npaths\npruned]',
            shape='box', style='dashed,filled',
            fillcolor='#FFE6E6', color=COLOR_RED, penwidth='1')

    # Edges to step 2
    dot.edge('the1', 'the_cat', label=' cat 0.6', color=COLOR_GREEN, penwidth='2')
    dot.edge('the1', 'the_dog', label=' dog 0.4', color=COLOR_GREEN, penwidth='1.5')
    dot.edge('a1', 'a_cat', label=' cat 0.6', color=COLOR_GREEN, penwidth='1.5')
    dot.edge('in1', 'pruned2', style='dashed', color=COLOR_RED)
    dot.edge('a1', 'pruned2', style='dashed', color=COLOR_RED)

    # ============================================
    # STEP 3: Third word (final selection)
    # ============================================

    # From "The cat" (0.36) - BEST PATH
    dot.node('the_cat_sat', 'The cat sat\nP=0.18',
            shape='box', style='rounded,filled',
            fillcolor='#E8D4F0', color=COLOR_ACCENT, penwidth='3')

    dot.node('the_cat_walked', 'The cat walked\nP=0.14',
            shape='box', style='rounded,filled',
            fillcolor='#FFE6CC', color='#FF7F0E', penwidth='1')

    # From "The dog" (0.24)
    dot.node('the_dog_barked', 'The dog barked\nP=0.12',
            shape='box', style='rounded,filled',
            fillcolor='#FFE6CC', color='#FF7F0E', penwidth='1')

    # Edges to step 3
    dot.edge('the_cat', 'the_cat_sat', label=' sat 0.5',
            color=COLOR_ACCENT, penwidth='3')
    dot.edge('the_cat', 'the_cat_walked', label=' walked 0.4',
            color='#FF7F0E', penwidth='1')
    dot.edge('the_dog', 'the_dog_barked', label=' barked 0.5',
            color='#FF7F0E', penwidth='1')

    # Add legend/annotations
    dot.node('legend',
            'BEAM SEARCH (k=3)\n\n'
            'Green: Top-3 kept\n'
            'Purple: Selected\n'
            'Red: Pruned\n\n'
            'At each step:\n'
            '1. Expand all beams\n'
            '2. Keep top-k by P',
            shape='note', style='filled',
            fillcolor='#FFFFCC', color=COLOR_MAIN,
            fontsize='10')

    # Render
    dot.render('../figures/beam_search_example_graphviz_bsc', cleanup=True)
    print("[OK] beam_search_example_graphviz_bsc.pdf created")


def main():
    """Generate beam search graphviz chart."""
    print("\n" + "="*60)
    print("BEAM SEARCH GRAPHVIZ CHART GENERATION")
    print("="*60 + "\n")

    generate_beam_search_example()

    print("\n" + "="*60)
    print("CHART GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated: beam_search_example_graphviz_bsc.pdf")
    print("Location: ../figures/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
