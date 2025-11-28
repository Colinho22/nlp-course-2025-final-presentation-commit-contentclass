import graphviz
import os

# Create a new directed graph
dot = graphviz.Digraph('QKV_Transformation', format='pdf', engine='dot')

# Graph settings for better layout
dot.attr(rankdir='TB', size='8,6', dpi='300')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='12')
dot.attr('edge', fontname='Arial', fontsize='10')

# Input words layer
with dot.subgraph(name='cluster_0') as c:
    c.attr(style='invisible')
    c.attr(rank='same')
    c.node('word1', 'The', fillcolor='lightblue', width='1.2')
    c.node('word2', 'cat', fillcolor='lightblue', width='1.2')
    c.node('word3', 'sat', fillcolor='lightblue', width='1.2')
    c.node('word4', 'on', fillcolor='lightblue', width='1.2')
    c.node('word5', 'mat', fillcolor='lightblue', width='1.2')
    c.attr(label='Input Words', labelloc='t', fontsize='14', fontname='Arial Bold')

# Transformation layer - Linear projections
with dot.subgraph(name='cluster_transform') as c:
    c.attr(style='invisible')
    c.node('linear', 'Linear Transformation\nW_Q, W_K, W_V',
           shape='rectangle', fillcolor='lightyellow', height='0.8', width='3')

# QKV layer
with dot.subgraph(name='cluster_1') as c:
    c.attr(style='invisible')
    c.attr(rank='same')

    # Query nodes
    c.node('Q1', 'Q: The', fillcolor='#ffcccc', width='1.2')
    c.node('Q2', 'Q: cat', fillcolor='#ffcccc', width='1.2')
    c.node('Q3', 'Q: sat', fillcolor='#ffcccc', width='1.2')
    c.node('Q4', 'Q: on', fillcolor='#ffcccc', width='1.2')
    c.node('Q5', 'Q: mat', fillcolor='#ffcccc', width='1.2')

with dot.subgraph(name='cluster_2') as c:
    c.attr(style='invisible')
    c.attr(rank='same')

    # Key nodes
    c.node('K1', 'K: The', fillcolor='#ccffcc', width='1.2')
    c.node('K2', 'K: cat', fillcolor='#ccffcc', width='1.2')
    c.node('K3', 'K: sat', fillcolor='#ccffcc', width='1.2')
    c.node('K4', 'K: on', fillcolor='#ccffcc', width='1.2')
    c.node('K5', 'K: mat', fillcolor='#ccffcc', width='1.2')

with dot.subgraph(name='cluster_3') as c:
    c.attr(style='invisible')
    c.attr(rank='same')

    # Value nodes
    c.node('V1', 'V: The', fillcolor='#ccccff', width='1.2')
    c.node('V2', 'V: cat', fillcolor='#ccccff', width='1.2')
    c.node('V3', 'V: sat', fillcolor='#ccccff', width='1.2')
    c.node('V4', 'V: on', fillcolor='#ccccff', width='1.2')
    c.node('V5', 'V: mat', fillcolor='#ccccff', width='1.2')

# Add edges from words to linear transformation
for i in range(1, 6):
    dot.edge(f'word{i}', 'linear', arrowhead='none')

# Add edges from linear to Q, K, V
for i in range(1, 6):
    dot.edge('linear', f'Q{i}', color='red', label='W_Q' if i == 3 else '')
    dot.edge('linear', f'K{i}', color='green', label='W_K' if i == 3 else '')
    dot.edge('linear', f'V{i}', color='blue', label='W_V' if i == 3 else '')

# Add labels for Q, K, V roles
dot.node('Q_label', 'Query:\nWhat am I\nlooking for?',
         shape='note', fillcolor='#ffcccc', style='filled', fontsize='10')
dot.node('K_label', 'Key:\nWhat do I\noffer?',
         shape='note', fillcolor='#ccffcc', style='filled', fontsize='10')
dot.node('V_label', 'Value:\nWhat info do\nI carry?',
         shape='note', fillcolor='#ccccff', style='filled', fontsize='10')

# Position labels
dot.edge('Q3', 'Q_label', style='dotted', arrowhead='none', color='gray')
dot.edge('K3', 'K_label', style='dotted', arrowhead='none', color='gray')
dot.edge('V3', 'V_label', style='dotted', arrowhead='none', color='gray')

# Add title
dot.attr(label='Query, Key, Value Transformation', labelloc='t',
         fontsize='16', fontname='Arial Bold')

# Render the graph
output_path = '../figures/qkv_transformation_graphviz'
dot.render(output_path, cleanup=True)
print(f"Generated {output_path}.pdf")