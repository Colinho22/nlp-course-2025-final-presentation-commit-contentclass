import graphviz
import os

# Create a new directed graph
dot = graphviz.Digraph('Attention_Steps', format='pdf', engine='dot')

# Graph settings
dot.attr(rankdir='LR', size='10,5', dpi='300')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='12')
dot.attr('edge', fontname='Arial', fontsize='11', fontweight='bold')

# Step 1: Q and K matrices
with dot.subgraph(name='cluster_step1') as c:
    c.attr(label='Step 1: Dot Product', style='rounded,filled', fillcolor='#f0f0f0',
           fontsize='14', fontname='Arial Bold')
    c.node('Q', 'Query Matrix\nQ', fillcolor='#ffcccc', width='1.5', height='0.8')
    c.node('K', 'Key Matrix\nK^T', fillcolor='#ccffcc', width='1.5', height='0.8')
    c.node('QK', 'Q·K^T\n(Scores)', fillcolor='#ffffcc', width='1.5', height='0.8')

# Step 2: Scaling and Softmax
with dot.subgraph(name='cluster_step2') as c:
    c.attr(label='Step 2: Normalize', style='rounded,filled', fillcolor='#f0f0f0',
           fontsize='14', fontname='Arial Bold')
    c.node('scaled', 'Scaled\nScores/√d_k', fillcolor='#ffeecc', width='1.5', height='0.8')
    c.node('softmax', 'Attention\nWeights', fillcolor='#ffddaa', width='1.5', height='0.8')

# Step 3: Apply to Values
with dot.subgraph(name='cluster_step3') as c:
    c.attr(label='Step 3: Weight Values', style='rounded,filled', fillcolor='#f0f0f0',
           fontsize='14', fontname='Arial Bold')
    c.node('V', 'Value Matrix\nV', fillcolor='#ccccff', width='1.5', height='0.8')
    c.node('weighted', 'Weighted\nValues', fillcolor='#ddccff', width='1.5', height='0.8')

# Step 4: Final Output
with dot.subgraph(name='cluster_step4') as c:
    c.attr(label='Step 4: Sum', style='rounded,filled', fillcolor='#f0f0f0',
           fontsize='14', fontname='Arial Bold')
    c.node('output', 'Final\nOutput', fillcolor='#90EE90', width='1.5', height='0.8',
           shape='box', style='rounded,filled,bold')

# Add edges for the flow
dot.edge('Q', 'QK', label='multiply')
dot.edge('K', 'QK', label='multiply')
dot.edge('QK', 'scaled', label='scale')
dot.edge('scaled', 'softmax', label='softmax', color='red', fontcolor='red')
dot.edge('softmax', 'weighted', label='multiply')
dot.edge('V', 'weighted', label='weight by')
dot.edge('weighted', 'output', label='sum', color='green', fontcolor='green', penwidth='2')

# Add mathematical notation boxes
dot.node('math1', 'QK^T', shape='plaintext', fontsize='10', fontname='Courier')
dot.node('math2', 'softmax(QK^T/√d_k)', shape='plaintext', fontsize='10', fontname='Courier')
dot.node('math3', 'Attention(Q,K,V)', shape='plaintext', fontsize='10', fontname='Courier')

# Position math annotations
dot.edge('QK', 'math1', style='dotted', arrowhead='none', color='gray')
dot.edge('softmax', 'math2', style='dotted', arrowhead='none', color='gray')
dot.edge('output', 'math3', style='dotted', arrowhead='none', color='gray')

# Add title
dot.attr(label='Step-by-Step: Computing Attention', labelloc='t',
         fontsize='18', fontname='Arial Bold')

# Add note about dimensions
dot.node('note', 'Each step preserves\nsequence length n\nand transforms features',
         shape='note', fillcolor='lightyellow', style='filled', fontsize='10')

# Render the graph
output_path = '../figures/attention_steps_graphviz'
dot.render(output_path, cleanup=True)
print(f"Generated {output_path}.pdf")