import graphviz
import os

# Create output directory
os.makedirs('../figures', exist_ok=True)

# Professional color scheme for graphviz
COLOR_INPUT = '#E8F4F8'      # Light blue
COLOR_PROCESS = '#FFF4E6'    # Light orange
COLOR_OUTPUT = '#E8F5E9'     # Light green
COLOR_DECISION = '#F3E5F5'   # Light purple
COLOR_HIGHLIGHT = '#FFEBEE'  # Light red

# Chart 1: LLM Summarization Pipeline
def create_llm_pipeline():
    dot = graphviz.Digraph(comment='LLM Pipeline', format='pdf')
    dot.attr(rankdir='LR', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='11')

    # Nodes
    dot.node('doc', 'Long\nDocument\n(up to 100K tokens)', fillcolor=COLOR_INPUT)
    dot.node('prompt', 'Prompt\nEngineering\n(system + examples)', fillcolor=COLOR_PROCESS)
    dot.node('llm', 'LLM\nProcessing\n(GPT/Claude/LLaMA)', fillcolor=COLOR_HIGHLIGHT, penwidth='2.5')
    dot.node('decode', 'Decoding\nControl\n(T, top-p, penalty)', fillcolor=COLOR_PROCESS)
    dot.node('output', 'Summary\nOutput\n(concise result)', fillcolor=COLOR_OUTPUT)

    # Edges
    dot.edge('doc', 'prompt', label='Input text')
    dot.edge('prompt', 'llm', label='Instruction')
    dot.edge('llm', 'decode', label='Token probs')
    dot.edge('decode', 'output', label='Generated text')

    dot.render('../figures/llm_pipeline_graphviz', cleanup=True)
    print("1/6: LLM pipeline (graphviz)")

# Chart 2: Human Paraphrasing Comparison
def create_paraphrasing_comparison():
    dot = graphviz.Digraph(comment='Paraphrasing', format='pdf')
    dot.attr(rankdir='TB', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    # Original text
    dot.node('orig', 'Original Text:\n"Company reported 25% revenue increase,\nmargins expanded to 18%"',
             fillcolor=COLOR_INPUT, shape='box', width='4')

    # Three approaches
    with dot.subgraph(name='cluster_methods') as c:
        c.attr(label='Three Approaches', fontsize='12', style='dashed')
        c.node('extract', 'Extractive:\n"Company reported 25%\nrevenue increase"',
               fillcolor='#FFE6E6', width='2.5')
        c.node('old', 'Old Abstractive:\n"Revenue increased 25%.\nMargins expanded."',
               fillcolor='#FFF8E6', width='2.5')
        c.node('llm', 'LLM Abstractive:\n"Firm achieved impressive growth,\nsales up 25%, profitability\nimproved to 18% margins"',
               fillcolor='#E6FFE6', width='2.5', penwidth='2')

    # Edges
    dot.edge('orig', 'extract', label='Copy')
    dot.edge('orig', 'old', label='Simple rewrite')
    dot.edge('orig', 'llm', label='Rephrase', penwidth='2')

    # Annotations
    dot.node('note', 'Key: LLMs paraphrase like humans\n(different words, same meaning)',
             shape='note', fillcolor='#FFFACD', fontsize='11')

    dot.render('../figures/human_paraphrasing_graphviz', cleanup=True)
    print("2/6: Human paraphrasing comparison (graphviz)")

# Chart 3: Chain-of-Thought Flow
def create_chain_of_thought():
    dot = graphviz.Digraph(comment='CoT', format='pdf')
    dot.attr(rankdir='TB', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    # Problem
    dot.node('problem', 'Problem:\n50-page report\n(20K tokens)\nExceeds context limit',
             fillcolor=COLOR_INPUT, shape='box')

    # CoT steps
    with dot.subgraph(name='cluster_cot') as c:
        c.attr(label='Chain-of-Thought Strategy', fontsize='12', style='rounded', color='blue')
        c.node('step1', 'Step 1:\nExtract sections\n(Intro, Methods, Results)', fillcolor=COLOR_PROCESS)
        c.node('step2', 'Step 2:\nSummarize each\n(2-3 sentences per)', fillcolor=COLOR_PROCESS)
        c.node('step3', 'Step 3:\nIdentify key findings\n(List main results)', fillcolor=COLOR_PROCESS)
        c.node('step4', 'Step 4:\nSynthesize\n(Combine coherently)', fillcolor=COLOR_PROCESS)

    # Output
    dot.node('final', 'Final Summary:\n"Study analyzed X using Y,\nfinding Z with W implications"',
             fillcolor=COLOR_OUTPUT, penwidth='2')

    # Flow
    dot.edge('problem', 'step1')
    dot.edge('step1', 'step2')
    dot.edge('step2', 'step3')
    dot.edge('step3', 'step4')
    dot.edge('step4', 'final', penwidth='2')

    dot.render('../figures/chain_of_thought_graphviz', cleanup=True)
    print("3/6: Chain-of-thought flow (graphviz)")

# Chart 4: Chunking Strategy
def create_chunking_strategy():
    dot = graphviz.Digraph(comment='Chunking', format='pdf')
    dot.attr(rankdir='LR', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    # Input
    dot.node('longdoc', 'Long Document\n20,000 tokens\n(Exceeds limit)', fillcolor=COLOR_INPUT)

    # Split process
    dot.node('split', 'Split into chunks\nsize=5000, overlap=500', fillcolor=COLOR_DECISION, shape='diamond')

    # Chunks
    with dot.subgraph(name='cluster_chunks') as c:
        c.attr(rank='same')
        c.node('c1', 'Chunk 1\n(0-5000)', fillcolor=COLOR_PROCESS)
        c.node('c2', 'Chunk 2\n(4500-9500)', fillcolor=COLOR_PROCESS)
        c.node('c3', 'Chunk 3\n(9000-14000)', fillcolor=COLOR_PROCESS)
        c.node('c4', 'Chunk 4\n(13500-20000)', fillcolor=COLOR_PROCESS)

    # Summarize
    dot.node('summarize', 'Summarize\neach chunk', fillcolor=COLOR_DECISION, shape='diamond')

    # Chunk summaries
    with dot.subgraph(name='cluster_sums') as c:
        c.attr(rank='same')
        c.node('s1', 'Sum 1\n(50 tokens)', fillcolor='#E3F2FD', width='1.2')
        c.node('s2', 'Sum 2\n(50 tokens)', fillcolor='#E3F2FD', width='1.2')
        c.node('s3', 'Sum 3\n(50 tokens)', fillcolor='#E3F2FD', width='1.2')
        c.node('s4', 'Sum 4\n(50 tokens)', fillcolor='#E3F2FD', width='1.2')

    # Merge
    dot.node('merge', 'Merge\nsummaries', fillcolor=COLOR_DECISION, shape='diamond')
    dot.node('final', 'Final Summary\n(80 tokens)', fillcolor=COLOR_OUTPUT, penwidth='2')

    # Flow
    dot.edge('longdoc', 'split')
    dot.edge('split', 'c1')
    dot.edge('split', 'c2')
    dot.edge('split', 'c3')
    dot.edge('split', 'c4')

    dot.edge('c1', 's1')
    dot.edge('c2', 's2')
    dot.edge('c3', 's3')
    dot.edge('c4', 's4')

    dot.edge('s1', 'merge')
    dot.edge('s2', 'merge')
    dot.edge('s3', 'merge')
    dot.edge('s4', 'merge')

    dot.edge('merge', 'final', penwidth='2')

    dot.render('../figures/chunking_strategy_graphviz', cleanup=True)
    print("4/6: Chunking strategy (graphviz)")

# Chart 5: Map-Reduce Summarization
def create_map_reduce():
    dot = graphviz.Digraph(comment='MapReduce', format='pdf')
    dot.attr(rankdir='TB', dpi='300')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')

    # Multiple docs
    with dot.subgraph(name='cluster_docs') as c:
        c.attr(label='Multiple Documents', fontsize='11', rank='same')
        c.node('d1', 'Doc 1\n(5K)', fillcolor=COLOR_INPUT, width='1')
        c.node('d2', 'Doc 2\n(7K)', fillcolor=COLOR_INPUT, width='1')
        c.node('d3', 'Doc 3\n(6K)', fillcolor=COLOR_INPUT, width='1')
        c.node('d4', 'Doc 4\n(8K)', fillcolor=COLOR_INPUT, width='1')

    # MAP phase
    dot.node('map', 'MAP PHASE\n(Parallel)', fillcolor='#FFFFCC', shape='box', penwidth='2')

    # Individual summaries
    with dot.subgraph(name='cluster_sums') as c:
        c.attr(rank='same')
        c.node('s1', 'Sum 1', fillcolor=COLOR_PROCESS, width='0.8')
        c.node('s2', 'Sum 2', fillcolor=COLOR_PROCESS, width='0.8')
        c.node('s3', 'Sum 3', fillcolor=COLOR_PROCESS, width='0.8')
        c.node('s4', 'Sum 4', fillcolor=COLOR_PROCESS, width='0.8')

    # REDUCE phase
    dot.node('reduce', 'REDUCE PHASE\n(Combine)', fillcolor='#FFFFCC', shape='box', penwidth='2')

    # Final output
    dot.node('final', 'Final Unified\nSummary', fillcolor=COLOR_OUTPUT, penwidth='2.5')

    # Flow
    dot.edge('d1', 'map', style='invis')
    dot.edge('d2', 'map', style='invis')
    dot.edge('d3', 'map', style='invis')
    dot.edge('d4', 'map', style='invis')

    dot.edge('map', 's1')
    dot.edge('map', 's2')
    dot.edge('map', 's3')
    dot.edge('map', 's4')

    dot.edge('s1', 'reduce')
    dot.edge('s2', 'reduce')
    dot.edge('s3', 'reduce')
    dot.edge('s4', 'reduce')

    dot.edge('reduce', 'final', penwidth='2')

    dot.render('../figures/map_reduce_graphviz', cleanup=True)
    print("5/6: Map-reduce flow (graphviz)")

# Chart 6: Recursive Hierarchical Summarization
def create_hierarchical():
    dot = graphviz.Digraph(comment='Hierarchical', format='pdf')
    dot.attr(rankdir='BT', dpi='300')  # Bottom to top
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='9')

    # Level 0: Sections
    with dot.subgraph(name='cluster_level0') as c:
        c.attr(label='Level 0: Original Sections', fontsize='10')
        c.attr(rank='same')
        c.node('sec1', 'Intro\n(2K)', fillcolor=COLOR_INPUT, width='0.9')
        c.node('sec2', 'Methods\n(3K)', fillcolor=COLOR_INPUT, width='0.9')
        c.node('sec3', 'Results1\n(4K)', fillcolor=COLOR_INPUT, width='0.9')
        c.node('sec4', 'Results2\n(4K)', fillcolor=COLOR_INPUT, width='0.9')
        c.node('sec5', 'Results3\n(3K)', fillcolor=COLOR_INPUT, width='0.9')
        c.node('sec6', 'Discuss\n(3K)', fillcolor=COLOR_INPUT, width='0.9')

    # Level 1: Grouped summaries
    with dot.subgraph(name='cluster_level1') as c:
        c.attr(label='Level 1: Grouped Summaries', fontsize='10')
        c.attr(rank='same')
        c.node('g1', 'Intro+Methods', fillcolor=COLOR_PROCESS, width='1.3')
        c.node('g2', 'All Results', fillcolor=COLOR_PROCESS, width='1.3')
        c.node('g3', 'Discussion', fillcolor=COLOR_PROCESS, width='1.3')

    # Level 2: Final summary
    dot.node('final', 'Complete Study\nSummary', fillcolor=COLOR_OUTPUT, penwidth='2.5', width='1.8')

    # Connections Level 0 -> Level 1
    dot.edge('sec1', 'g1')
    dot.edge('sec2', 'g1')
    dot.edge('sec3', 'g2')
    dot.edge('sec4', 'g2')
    dot.edge('sec5', 'g2')
    dot.edge('sec6', 'g3')

    # Connections Level 1 -> Level 2
    dot.edge('g1', 'final', penwidth='1.5')
    dot.edge('g2', 'final', penwidth='1.5')
    dot.edge('g3', 'final', penwidth='1.5')

    dot.render('../figures/recursive_hierarchical_graphviz', cleanup=True)
    print("6/6: Recursive hierarchical (graphviz)")

# Generate all graphviz diagrams
if __name__ == "__main__":
    print("Generating 6 graphviz flow diagrams...")
    print("(Requires graphviz installed: pip install graphviz)")

    try:
        create_llm_pipeline()
        create_paraphrasing_comparison()
        create_chain_of_thought()
        create_chunking_strategy()
        create_map_reduce()
        create_hierarchical()
        print("\nAll 6 graphviz diagrams generated successfully!")
        print("Output: NLP_slides/summarization_module/figures/*_graphviz.pdf")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure graphviz is installed: pip install graphviz")
        print("And ensure Graphviz binaries are in PATH")
