"""
Generate pipeline visualization using graphviz showing how Weeks 1-8 converge to
probability distributions, and Week 9 performs decoding.

Uses graphviz for cleaner, more professional flow diagram.
"""

import graphviz
import subprocess
import os

def create_graphviz_pipeline():
    """Create professional pipeline chart using graphviz."""

    # Create directed graph
    dot = graphviz.Digraph(comment='NLP Course Pipeline to Decoding',
                          format='pdf',
                          engine='dot')

    # Graph attributes for better layout
    dot.attr(rankdir='TB', size='16,10!', ratio='fill',
             fontname='Arial', fontsize='28',
             bgcolor='white', pad='0.5', nodesep='0.8', ranksep='1.2')

    # Node styles
    week_style = {'shape': 'box', 'style': 'rounded,filled',
                  'fillcolor': '#4A90E2', 'fontcolor': 'white',
                  'fontsize': '11', 'height': '0.6', 'width': '1.5',
                  'penwidth': '2'}

    convergence_style = {'shape': 'box', 'style': 'filled,bold',
                        'fillcolor': '#E74C3C', 'fontcolor': 'white',
                        'fontsize': '14', 'height': '0.8', 'width': '8.0',
                        'penwidth': '3'}

    prob_style = {'shape': 'box', 'style': 'filled',
                 'fillcolor': '#F39C12', 'fontcolor': 'black',
                 'fontsize': '10', 'height': '1.2', 'width': '7.0',
                 'penwidth': '2'}

    question_style = {'shape': 'box', 'style': 'filled,bold',
                     'fillcolor': '#FFEB3B', 'fontcolor': '#D32F2F',
                     'fontsize': '13', 'height': '0.7', 'width': '6.0',
                     'penwidth': '3'}

    decode_style = {'shape': 'box', 'style': 'rounded,filled',
                   'fillcolor': '#27AE60', 'fontcolor': 'white',
                   'fontsize': '10', 'height': '0.5', 'width': '1.3',
                   'penwidth': '2'}

    output_style = {'shape': 'box', 'style': 'filled',
                   'fillcolor': '#A5D6A7', 'fontcolor': 'black',
                   'fontsize': '12', 'height': '0.6', 'width': '4.0',
                   'penwidth': '2'}

    # Week nodes (top tier)
    weeks = [
        ('w1', 'Week 1\nN-grams'),
        ('w2', 'Week 2\nNeural LM'),
        ('w3', 'Week 3\nRNN/LSTM'),
        ('w4', 'Week 4\nSeq2seq'),
        ('w5', 'Week 5\nTransformers'),
        ('w6', 'Week 6\nBERT/GPT'),
        ('w7', 'Week 7\nAdvanced'),
        ('w8', 'Week 8\nTokenization')
    ]

    with dot.subgraph(name='cluster_weeks') as c:
        c.attr(rank='same', label='Weeks 1-8: Different Architectures',
               fontsize='14', fontcolor='#4A90E2', style='dashed',
               color='#4A90E2', penwidth='2')
        for node_id, label in weeks:
            c.node(node_id, label, **week_style)

    # Convergence node
    dot.node('conv', 'ALL Methods Output:\nP(word | context) for ENTIRE Vocabulary\n(50,000+ words)',
             **convergence_style)

    # Probability distribution node
    prob_text = ('Example: "The weather is __"\n\n'
                'nice: 0.60  |  beautiful: 0.20  |  perfect: 0.10\n'
                'gorgeous: 0.05  |  awful: 0.02  |  mild: 0.02\n'
                'unpredictable: 0.01  |  ... (49,993 more words)')
    dot.node('prob', prob_text, **prob_style)

    # Week 9 question
    dot.node('question', 'WEEK 9 QUESTION:\nWhich word should we actually pick?',
             **question_style)

    # Decoding strategies (bottom tier)
    strategies = [
        ('s1', 'Greedy'),
        ('s2', 'Beam\nSearch'),
        ('s3', 'Temperature'),
        ('s4', 'Top-k'),
        ('s5', 'Top-p'),
        ('s6', 'Task-\nSpecific')
    ]

    with dot.subgraph(name='cluster_strategies') as c:
        c.attr(rank='same', label='Week 9: Decoding Strategies',
               fontsize='14', fontcolor='#27AE60', style='dashed',
               color='#27AE60', penwidth='2')
        for node_id, label in strategies:
            c.node(node_id, label, **decode_style)

    # Final output
    dot.node('output', 'Final Output:\n"The weather is nice"',
             **output_style)

    # Edges from weeks to convergence
    for node_id, _ in weeks:
        dot.edge(node_id, 'conv', color='#E74C3C', penwidth='2', arrowsize='0.8')

    # Edge from convergence to probability
    dot.edge('conv', 'prob', color='#F39C12', penwidth='3', arrowsize='1.0')

    # Edge from probability to question
    dot.edge('prob', 'question', color='#D32F2F', penwidth='4', arrowsize='1.2',
             style='bold')

    # Edge from question to strategies
    for node_id, _ in strategies:
        dot.edge('question', node_id, color='#27AE60', penwidth='2', arrowsize='0.8')

    # Edges from strategies to output
    for node_id, _ in strategies:
        dot.edge(node_id, 'output', color='#66BB6A', penwidth='1.5', arrowsize='0.7')

    # Render to PDF
    output_path = '../figures/weeks_pipeline_graphviz'
    dot.render(output_path, cleanup=True, view=False)

    print(f"Generated {output_path}.pdf using graphviz")

    # Also save the .dot source for reference
    with open(output_path + '.dot', 'w') as f:
        f.write(dot.source)
    print(f"Saved graphviz source to {output_path}.dot")


if __name__ == "__main__":
    # Check if graphviz is available
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
        create_graphviz_pipeline()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: Graphviz not found. Please install graphviz:")
        print("  Windows: choco install graphviz  OR  download from graphviz.org")
        print("  After install, add to PATH and restart terminal")
        exit(1)
