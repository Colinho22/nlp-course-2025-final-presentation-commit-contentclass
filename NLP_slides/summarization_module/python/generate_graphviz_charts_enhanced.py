#!/usr/bin/env python3
"""
Generate GraphViz charts for LLM Summarization Module with LARGER FONTS
Creates 6 graphviz charts with enhanced font sizes (16-18pt)
BSc Discovery color scheme
"""

import graphviz
import os

# Create output directory
os.makedirs('../figures', exist_ok=True)

# BSc Discovery color scheme
COLOR_MAIN = '#404040'
COLOR_ACCENT = '#3333B2'
COLOR_GRAY = '#B4B4B4'
COLOR_LIGHT = '#F0F0F0'
COLOR_GREEN = '#2CA02C'
COLOR_RED = '#D62728'
COLOR_ORANGE = '#FF7F0E'
COLOR_BLUE = '#1F77B4'

# Enhanced font sizes for graphviz
FONT_NODE = '18'  # Node labels
FONT_EDGE = '16'  # Edge labels
FONT_GRAPH = '20' # Graph titles

def generate_human_paraphrasing_graphviz():
    """Chart 1: Human paraphrasing capability visualization"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='white')
    dot.attr('node', shape='box', style='rounded,filled', fontsize=FONT_NODE, height='0.8')
    dot.attr('edge', fontsize=FONT_EDGE)

    # Title node
    dot.node('title', 'Human-Like Paraphrasing',
             fillcolor=COLOR_ACCENT, fontcolor='white', shape='box', style='filled,bold')

    # Original text
    dot.node('original', 'Original Text:\n"The study demonstrates that\nthe treatment was effective\nin 75% of cases"',
             fillcolor=COLOR_LIGHT, fontcolor=COLOR_MAIN, width='4')

    # LLM Processing
    dot.node('llm', 'LLM Processing',
             fillcolor=COLOR_BLUE, fontcolor='white', shape='ellipse')

    # Paraphrased versions
    dot.node('para1', 'Version 1:\n"Research shows treatment\nsucceeded for three-quarters\nof patients"',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN)

    dot.node('para2', 'Version 2:\n"75% efficacy rate observed\nin clinical trials"',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN)

    dot.node('para3', 'Version 3:\n"Treatment proved beneficial\nfor majority of subjects"',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN)

    # Edges
    dot.edge('title', 'original', style='invis')
    dot.edge('original', 'llm', 'Understands\nMeaning', fontsize=FONT_EDGE, color=COLOR_ACCENT)
    dot.edge('llm', 'para1', 'Generate', color=COLOR_GREEN)
    dot.edge('llm', 'para2', 'Generate', color=COLOR_GREEN)
    dot.edge('llm', 'para3', 'Generate', color=COLOR_GREEN)

    dot.render('../figures/human_paraphrasing_graphviz', cleanup=True)
    print("[1/6] Generated: human_paraphrasing_graphviz.pdf")

def generate_llm_pipeline_graphviz():
    """Chart 2: LLM summarization pipeline"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='LR', bgcolor='white', nodesep='0.5')
    dot.attr('node', shape='box', style='rounded,filled', fontsize=FONT_NODE, height='0.7')
    dot.attr('edge', fontsize=FONT_EDGE)

    # Input
    dot.node('input', 'Long Document\n(10,000 tokens)',
             fillcolor=COLOR_LIGHT, fontcolor=COLOR_MAIN, width='2.5')

    # Processing stages
    dot.node('chunk', 'Chunking\n(512 token\nwindows)',
             fillcolor=COLOR_BLUE + '30', fontcolor=COLOR_MAIN)

    dot.node('encode', 'Encode\nto Vectors',
             fillcolor=COLOR_ACCENT + '30', fontcolor=COLOR_MAIN)

    dot.node('attend', 'Attention\nMechanism',
             fillcolor=COLOR_ORANGE + '30', fontcolor=COLOR_MAIN)

    dot.node('decode', 'Decode\nto Text',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN)

    # Output
    dot.node('output', 'Summary\n(200 tokens)',
             fillcolor=COLOR_GREEN, fontcolor='white', width='2')

    # Edges with labels
    dot.edge('input', 'chunk', 'Split', color=COLOR_MAIN)
    dot.edge('chunk', 'encode', 'Transform', color=COLOR_BLUE)
    dot.edge('encode', 'attend', 'Focus', color=COLOR_ACCENT)
    dot.edge('attend', 'decode', 'Generate', color=COLOR_ORANGE)
    dot.edge('decode', 'output', 'Output', color=COLOR_GREEN)

    dot.render('../figures/llm_pipeline_graphviz', cleanup=True)
    print("[2/6] Generated: llm_pipeline_graphviz.pdf")

def generate_chain_of_thought_graphviz():
    """Chart 3: Chain-of-thought for summarization"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='white')
    dot.attr('node', shape='box', style='rounded,filled', fontsize=FONT_NODE, height='0.6')
    dot.attr('edge', fontsize=FONT_EDGE)

    # Title
    dot.node('title', 'Chain-of-Thought Summarization',
             fillcolor=COLOR_ACCENT, fontcolor='white', style='filled,bold', width='4')

    # Steps with subgraph clustering
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Step-by-Step Reasoning', fontsize=FONT_GRAPH)
        c.attr(style='dotted', color=COLOR_GRAY)

        c.node('step1', '1. Identify main topics:\n• Treatment efficacy\n• Patient outcomes\n• Side effects',
               fillcolor=COLOR_LIGHT, fontcolor=COLOR_MAIN, width='3.5')

        c.node('step2', '2. Extract key facts:\n• 75% improvement rate\n• Minimal side effects\n• 3-month study',
               fillcolor=COLOR_LIGHT, fontcolor=COLOR_MAIN, width='3.5')

        c.node('step3', '3. Determine importance:\n• Primary: efficacy\n• Secondary: safety\n• Context: duration',
               fillcolor=COLOR_LIGHT, fontcolor=COLOR_MAIN, width='3.5')

        c.node('step4', '4. Synthesize summary:\nCombine key points\nin logical order',
               fillcolor=COLOR_LIGHT, fontcolor=COLOR_MAIN, width='3.5')

    # Final output
    dot.node('output', 'Final Summary:\n"The 3-month study showed\n75% improvement with\nminimal side effects"',
             fillcolor=COLOR_GREEN, fontcolor='white', width='4')

    # Edges
    dot.edge('title', 'step1', style='invis')
    dot.edge('step1', 'step2', 'Process', color=COLOR_ACCENT)
    dot.edge('step2', 'step3', 'Analyze', color=COLOR_ACCENT)
    dot.edge('step3', 'step4', 'Organize', color=COLOR_ACCENT)
    dot.edge('step4', 'output', 'Generate', color=COLOR_GREEN, style='bold')

    dot.render('../figures/chain_of_thought_graphviz', cleanup=True)
    print("[3/6] Generated: chain_of_thought_graphviz.pdf")

def generate_chunking_strategy_graphviz():
    """Chart 4: Document chunking strategy"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='white', ranksep='0.8')
    dot.attr('node', shape='box', style='filled', fontsize=FONT_NODE, height='0.6')
    dot.attr('edge', fontsize=FONT_EDGE)

    # Document
    dot.node('doc', 'Original Document\n(50 pages, 25K tokens)',
             fillcolor=COLOR_ACCENT, fontcolor='white', width='4', style='filled,bold')

    # Chunking methods with subgraphs
    with dot.subgraph(name='cluster_methods') as c:
        c.attr(label='Chunking Strategies', fontsize=FONT_GRAPH)
        c.attr(style='rounded', color=COLOR_GRAY, bgcolor=COLOR_LIGHT + '20')

        # Fixed size
        c.node('fixed', 'Fixed Size\n(512 tokens)',
               fillcolor='white', fontcolor=COLOR_MAIN, shape='box')

        # Sliding window
        c.node('sliding', 'Sliding Window\n(512 tokens,\n128 overlap)',
               fillcolor='white', fontcolor=COLOR_MAIN, shape='box')

        # Semantic
        c.node('semantic', 'Semantic\n(Paragraph/\nSection based)',
               fillcolor='white', fontcolor=COLOR_MAIN, shape='box')

    # Chunks
    dot.node('chunk1', 'Chunk 1\nIntroduction',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN, shape='note')
    dot.node('chunk2', 'Chunk 2\nMethods',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN, shape='note')
    dot.node('chunk3', 'Chunk 3\nResults',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN, shape='note')
    dot.node('chunk4', '...More chunks...',
             fillcolor=COLOR_GREEN + '30', fontcolor=COLOR_MAIN, shape='note')

    # Processing
    dot.node('process', 'Process Each\nChunk',
             fillcolor=COLOR_ORANGE, fontcolor='white', shape='ellipse')

    # Summary
    dot.node('summary', 'Combined\nSummary',
             fillcolor=COLOR_GREEN, fontcolor='white', width='3', style='filled,bold')

    # Edges
    dot.edge('doc', 'fixed', 'Option 1', color=COLOR_BLUE)
    dot.edge('doc', 'sliding', 'Option 2', color=COLOR_BLUE)
    dot.edge('doc', 'semantic', 'Option 3', color=COLOR_BLUE)

    dot.edge('semantic', 'chunk1', style='dotted')
    dot.edge('semantic', 'chunk2', style='dotted')
    dot.edge('semantic', 'chunk3', style='dotted')
    dot.edge('semantic', 'chunk4', style='dotted')

    dot.edge('chunk1', 'process')
    dot.edge('chunk2', 'process')
    dot.edge('chunk3', 'process')
    dot.edge('chunk4', 'process')

    dot.edge('process', 'summary', 'Merge', color=COLOR_GREEN, style='bold')

    dot.render('../figures/chunking_strategy_graphviz', cleanup=True)
    print("[4/6] Generated: chunking_strategy_graphviz.pdf")

def generate_map_reduce_graphviz():
    """Chart 5: Map-reduce summarization pattern"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='white')
    dot.attr('node', shape='box', style='rounded,filled', fontsize=FONT_NODE)
    dot.attr('edge', fontsize=FONT_EDGE)

    # Title
    dot.node('title', 'Map-Reduce Summarization',
             fillcolor=COLOR_ACCENT, fontcolor='white', width='4', height='0.8', style='filled,bold')

    # Document chunks (Map phase)
    with dot.subgraph(name='cluster_map') as c:
        c.attr(label='MAP: Summarize Each Chunk', fontsize=FONT_GRAPH)
        c.attr(style='rounded,dashed', color=COLOR_BLUE)
        c.attr(rank='same')

        c.node('chunk1', 'Chunk 1\n(Pages 1-10)', fillcolor=COLOR_LIGHT)
        c.node('sum1', 'Summary 1\n(50 tokens)', fillcolor=COLOR_BLUE + '30')

        c.node('chunk2', 'Chunk 2\n(Pages 11-20)', fillcolor=COLOR_LIGHT)
        c.node('sum2', 'Summary 2\n(50 tokens)', fillcolor=COLOR_BLUE + '30')

        c.node('chunk3', 'Chunk 3\n(Pages 21-30)', fillcolor=COLOR_LIGHT)
        c.node('sum3', 'Summary 3\n(50 tokens)', fillcolor=COLOR_BLUE + '30')

    # Reduce phase
    with dot.subgraph(name='cluster_reduce') as c:
        c.attr(label='REDUCE: Combine Summaries', fontsize=FONT_GRAPH)
        c.attr(style='rounded,dashed', color=COLOR_GREEN)

        c.node('combine', 'Combine All\nSummaries',
               fillcolor=COLOR_ORANGE, fontcolor='white', shape='ellipse', height='0.8')

        c.node('final', 'Final Summary\n(150 tokens)',
               fillcolor=COLOR_GREEN, fontcolor='white', width='3', height='0.8')

    # Edges
    dot.edge('title', 'chunk1', style='invis')

    dot.edge('chunk1', 'sum1', 'Map', color=COLOR_BLUE)
    dot.edge('chunk2', 'sum2', 'Map', color=COLOR_BLUE)
    dot.edge('chunk3', 'sum3', 'Map', color=COLOR_BLUE)

    dot.edge('sum1', 'combine', color=COLOR_ORANGE)
    dot.edge('sum2', 'combine', color=COLOR_ORANGE)
    dot.edge('sum3', 'combine', color=COLOR_ORANGE)

    dot.edge('combine', 'final', 'Reduce', color=COLOR_GREEN, style='bold')

    dot.render('../figures/map_reduce_graphviz', cleanup=True)
    print("[5/6] Generated: map_reduce_graphviz.pdf")

def generate_recursive_hierarchical_graphviz():
    """Chart 6: Recursive hierarchical summarization"""
    dot = graphviz.Digraph(format='pdf', engine='dot')
    dot.attr(dpi='300', rankdir='TB', bgcolor='white', nodesep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontsize=FONT_NODE)
    dot.attr('edge', fontsize=FONT_EDGE)

    # Title
    dot.node('title', 'Recursive Hierarchical Summarization',
             fillcolor=COLOR_ACCENT, fontcolor='white', width='5', height='0.8', style='filled,bold')

    # Level 1: Original chunks
    with dot.subgraph(name='cluster_l1') as c:
        c.attr(label='Level 1: Original Chunks (8)', fontsize=FONT_GRAPH)
        c.attr(rank='same', style='invis')

        for i in range(1, 9):
            c.node(f'c{i}', f'C{i}', fillcolor=COLOR_LIGHT, width='0.8', height='0.6')

    # Level 2: First summarization
    with dot.subgraph(name='cluster_l2') as c:
        c.attr(label='Level 2: First Summary (4)', fontsize=FONT_GRAPH)
        c.attr(rank='same', style='invis')

        for i in range(1, 5):
            c.node(f's2_{i}', f'S{i}', fillcolor=COLOR_BLUE + '40', width='1', height='0.6')

    # Level 3: Second summarization
    with dot.subgraph(name='cluster_l3') as c:
        c.attr(label='Level 3: Second Summary (2)', fontsize=FONT_GRAPH)
        c.attr(rank='same', style='invis')

        c.node('s3_1', 'Summary A', fillcolor=COLOR_ORANGE + '40', width='1.5', height='0.7')
        c.node('s3_2', 'Summary B', fillcolor=COLOR_ORANGE + '40', width='1.5', height='0.7')

    # Level 4: Final
    dot.node('final', 'Final Hierarchical\nSummary',
             fillcolor=COLOR_GREEN, fontcolor='white', width='3', height='0.8', style='filled,bold')

    # Edges
    dot.edge('title', 'c4', style='invis')

    # Level 1 to 2
    for i in range(1, 9, 2):
        dot.edge(f'c{i}', f's2_{(i+1)//2}', color=COLOR_BLUE + '80')
        dot.edge(f'c{i+1}', f's2_{(i+1)//2}', color=COLOR_BLUE + '80')

    # Level 2 to 3
    dot.edge('s2_1', 's3_1', color=COLOR_ORANGE + '80')
    dot.edge('s2_2', 's3_1', color=COLOR_ORANGE + '80')
    dot.edge('s2_3', 's3_2', color=COLOR_ORANGE + '80')
    dot.edge('s2_4', 's3_2', color=COLOR_ORANGE + '80')

    # Level 3 to final
    dot.edge('s3_1', 'final', color=COLOR_GREEN, style='bold')
    dot.edge('s3_2', 'final', color=COLOR_GREEN, style='bold')

    # Add note
    dot.node('note', 'Preserves document structure\nthrough hierarchical merging',
             shape='note', fillcolor='lightyellow', fontsize='14', style='filled')
    dot.edge('final', 'note', style='dotted', color=COLOR_GRAY)

    dot.render('../figures/recursive_hierarchical_graphviz', cleanup=True)
    print("[6/6] Generated: recursive_hierarchical_graphviz.pdf")

# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Generating Enhanced GraphViz Charts with LARGER FONTS...")
    print("Font sizes: Node=18pt, Edge=16pt, Graph=20pt")
    print("="*60)

    # Generate all 6 graphviz charts
    generate_human_paraphrasing_graphviz()
    generate_llm_pipeline_graphviz()
    generate_chain_of_thought_graphviz()
    generate_chunking_strategy_graphviz()
    generate_map_reduce_graphviz()
    generate_recursive_hierarchical_graphviz()

    print("\n" + "="*60)
    print("All 6 GraphViz charts generated with enhanced font sizes!")
    print("Files saved to ../figures/ directory")