import graphviz
import os

# Create a new directed graph
dot = graphviz.Digraph('Attention_Example', format='pdf', engine='dot')

# Graph settings for better layout
dot.attr(rankdir='TB', size='10,8', dpi='300')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
dot.attr('edge', fontname='Arial', fontsize='9')

# Title
dot.attr(label='Computing Attention: Concrete Example\n"The cat sat" → Focus on "cat"',
         labelloc='t', fontsize='16', fontname='Arial Bold')

# Input: Example with 3 words for simplicity
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input: 3 words, d=4', style='rounded,filled', fillcolor='#f0f0f0',
           fontsize='12', fontname='Arial Bold')

    # Query for "cat"
    c.node('Q', 'Query: "cat"\n[0.5, 0.3, 0.8, 0.2]',
           fillcolor='#ffcccc', width='2.2', height='0.6')

    # Keys for all words
    c.node('K', 'Keys (transposed):\n"The": [0.1, 0.9, 0.2, 0.4]\n"cat": [0.6, 0.4, 0.7, 0.3]\n"sat": [0.3, 0.2, 0.5, 0.8]',
           fillcolor='#ccffcc', width='2.8', height='1.0')

    # Values for all words
    c.node('V', 'Values:\n"The": [0.2, 0.3]\n"cat": [0.8, 0.9]\n"sat": [0.5, 0.4]',
           fillcolor='#ccccff', width='2.2', height='0.8')

# Step 1: Dot Product
with dot.subgraph(name='cluster_dotprod') as c:
    c.attr(label='Step 1: Q · K^T (Dot Product)', style='rounded,filled', fillcolor='#ffe6e6',
           fontsize='12', fontname='Arial Bold')

    c.node('scores', 'Attention Scores:\n"The": 0.5×0.1 + 0.3×0.9 + 0.8×0.2 + 0.2×0.4 = 0.56\n"cat": 0.5×0.6 + 0.3×0.4 + 0.8×0.7 + 0.2×0.3 = 1.04\n"sat": 0.5×0.3 + 0.3×0.2 + 0.8×0.5 + 0.2×0.8 = 0.77',
           fillcolor='#ffffcc', width='5.0', height='1.0')

# Step 2: Scale and Softmax
with dot.subgraph(name='cluster_softmax') as c:
    c.attr(label='Step 2: Scale by √d=2, then Softmax', style='rounded,filled', fillcolor='#e6ffe6',
           fontsize='12', fontname='Arial Bold')

    c.node('scaled', 'Scaled Scores:\n"The": 0.56/2 = 0.28\n"cat": 1.04/2 = 0.52\n"sat": 0.77/2 = 0.385',
           fillcolor='#ffeecc', width='2.5', height='0.8')

    c.node('softmax', 'Attention Weights:\n"The": e^0.28 / Σ = 0.30 (30%)\n"cat": e^0.52 / Σ = 0.39 (39%)\n"sat": e^0.385 / Σ = 0.31 (31%)',
           fillcolor='#ffddaa', width='3.0', height='0.8', penwidth='2')

# Step 3: Apply Weights to Values
with dot.subgraph(name='cluster_weight') as c:
    c.attr(label='Step 3: Multiply Weights × Values', style='rounded,filled', fillcolor='#e6e6ff',
           fontsize='12', fontname='Arial Bold')

    c.node('weighted', 'Weighted Values:\n0.30 × [0.2, 0.3] = [0.06, 0.09]\n0.39 × [0.8, 0.9] = [0.31, 0.35]\n0.31 × [0.5, 0.4] = [0.16, 0.12]',
           fillcolor='#ddccff', width='3.5', height='0.8')

# Step 4: Sum
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Step 4: Sum All Weighted Values', style='rounded,filled', fillcolor='#e6ffe6',
           fontsize='12', fontname='Arial Bold')

    c.node('output', 'Output for "cat":\n[0.06, 0.09] +\n[0.31, 0.35] +\n[0.16, 0.12] =\n[0.53, 0.56]',
           fillcolor='#90EE90', width='2.0', height='1.2',
           shape='box', style='rounded,filled,bold', penwidth='3')

# Add edges
dot.edge('Q', 'scores', label='multiply', color='red')
dot.edge('K', 'scores', label='multiply', color='green')
dot.edge('scores', 'scaled', label='÷√4', fontweight='bold')
dot.edge('scaled', 'softmax', label='e^x/Σe^x', color='orange', fontweight='bold')
dot.edge('softmax', 'weighted', label='multiply', color='blue')
dot.edge('V', 'weighted', label='by weights', color='blue')
dot.edge('weighted', 'output', label='sum all', color='darkgreen', penwidth='2', fontweight='bold')

# Add key insight box
dot.node('insight', '"cat" gets 39%\nattention to itself!\n(highest weight)',
         shape='note', fillcolor='yellow', style='filled', fontsize='11', fontweight='bold')
dot.edge('softmax', 'insight', style='dotted', arrowhead='none', color='orange')

# Add dimension note
dot.node('dims', 'Dimensions:\nQ: [1×4]\nK: [3×4]\nV: [3×2]\nOutput: [1×2]',
         shape='plaintext', fontsize='9', fontname='Courier')

# Render the graph
output_path = '../figures/attention_example_graphviz'
dot.render(output_path, cleanup=True)
print(f"Generated {output_path}.pdf")