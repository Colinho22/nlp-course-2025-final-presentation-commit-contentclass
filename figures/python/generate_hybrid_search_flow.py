"""
Generate Hybrid Search Flow chart using Graphviz.
Shows BM25 + Dense retrieval fusion pipeline.
"""

import subprocess
import os

OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dot_code = """
digraph HybridSearch {
    // Graph settings
    rankdir=TB;
    bgcolor="white";
    fontname="Helvetica";
    node [fontname="Helvetica", fontsize=11];
    edge [fontname="Helvetica", fontsize=9];

    // Define styles
    node [shape=box, style="rounded,filled"];

    // Input
    query [label="User Query", fillcolor="#C1C1E8", color="#3333B2", penwidth=2];

    // Split paths
    node [shape=ellipse, style=filled];
    split [label="Split", fillcolor="#E8E8E8", color="#7F7F7F", penwidth=1];

    // BM25 Path (orange)
    node [shape=box, style="rounded,filled"];
    bm25_encode [label="Tokenize\\n(Keywords)", fillcolor="#FFF3E0", color="#FF7F0E", penwidth=2];
    bm25_search [label="BM25 Search\\n(Inverted Index)", fillcolor="#FFF3E0", color="#FF7F0E", penwidth=2];
    bm25_results [label="BM25 Candidates\\n(with TF-IDF scores)", fillcolor="#FFF3E0", color="#FF7F0E", penwidth=2];

    // Dense Path (blue)
    dense_encode [label="Embed Query\\n(Encoder Model)", fillcolor="#E3F2FD", color="#0066CC", penwidth=2];
    dense_search [label="ANN Search\\n(Vector DB)", fillcolor="#E3F2FD", color="#0066CC", penwidth=2];
    dense_results [label="Dense Candidates\\n(with cosine scores)", fillcolor="#E3F2FD", color="#0066CC", penwidth=2];

    // Fusion (purple)
    fusion [label="Reciprocal Rank Fusion\\n(RRF)", fillcolor="#E8E8F8", color="#3333B2", penwidth=2, shape=box];

    // Reranking (green)
    rerank [label="Cross-Encoder\\nReranking", fillcolor="#E8F5E9", color="#2CA02C", penwidth=2];

    // Output
    output [label="Final Top-K\\nDocuments", fillcolor="#C8E6C9", color="#2CA02C", penwidth=2];

    // Main flow
    query -> split [penwidth=2, color="#3333B2"];

    // BM25 branch
    split -> bm25_encode [penwidth=1.5, color="#FF7F0E", label="  sparse"];
    bm25_encode -> bm25_search [penwidth=1.5, color="#FF7F0E"];
    bm25_search -> bm25_results [penwidth=1.5, color="#FF7F0E"];
    bm25_results -> fusion [penwidth=1.5, color="#FF7F0E"];

    // Dense branch
    split -> dense_encode [penwidth=1.5, color="#0066CC", label="dense  "];
    dense_encode -> dense_search [penwidth=1.5, color="#0066CC"];
    dense_search -> dense_results [penwidth=1.5, color="#0066CC"];
    dense_results -> fusion [penwidth=1.5, color="#0066CC"];

    // Fusion to output
    fusion -> rerank [penwidth=2, color="#3333B2", label="  merged candidates"];
    rerank -> output [penwidth=2, color="#2CA02C"];

    // Layout hints
    {rank=same; bm25_encode; dense_encode}
    {rank=same; bm25_search; dense_search}
    {rank=same; bm25_results; dense_results}

    // RRF formula annotation
    rrf_note [shape=note, label="RRF Score:\\nscore(d) = sum(1/(k + rank_i(d)))\\nk = 60 (typical)", fillcolor="#FFFDE7", fontsize=9];
    fusion -> rrf_note [style=dashed, color="#7F7F7F", arrowhead=none];

    // Title
    labelloc="t";
    label="\\nHybrid Search: Combining BM25 and Dense Retrieval\\n";
    fontsize=16;
    fontcolor="#3333B2";
}
"""

# Write DOT file
dot_path = os.path.join(OUTPUT_DIR, "hybrid_search_flow.dot")
pdf_path = os.path.join(OUTPUT_DIR, "hybrid_search_flow.pdf")

with open(dot_path, 'w') as f:
    f.write(dot_code)

print(f"DOT file written to: {dot_path}")

# Try to compile with Graphviz
try:
    result = subprocess.run(
        ['dot', '-Tpdf', dot_path, '-o', pdf_path],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"PDF generated: {pdf_path}")
    else:
        print(f"Graphviz error: {result.stderr}")
except FileNotFoundError:
    print("Graphviz 'dot' command not found. Please install Graphviz.")
    print("The DOT file has been created and can be compiled manually.")
except Exception as e:
    print(f"Error: {e}")
