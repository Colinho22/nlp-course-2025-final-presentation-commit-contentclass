"""
Convert Final Lecture Python scripts to Quantlet folder structure.

Creates numbered folders for each chart with:
- Python script
- Generated PDF chart
- metainfo.txt (Quantlet standard format)
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

# Define the chart scripts and their outputs
CHART_SCRIPTS = {
    'generate_ann_math_concept.py': {
        'output': 'ann_math_concept.pdf',
        'name': 'ANN Mathematical Concept',
        'description': 'Visualization of artificial neural network mathematical formulation including input, hidden layers, and output',
        'keywords': 'neural network, deep learning, mathematical formulation, ANN, visualization'
    },
    'generate_hnsw_explanation.py': {
        'output': 'hnsw_explanation.pdf',
        'name': 'HNSW Algorithm Explanation',
        'description': 'Hierarchical Navigable Small World graph visualization for approximate nearest neighbor search',
        'keywords': 'HNSW, approximate nearest neighbor, vector search, graph algorithm, RAG'
    },
    'generate_hnsw_cities_example.py': {
        'output': 'hnsw_cities_example.pdf',
        'name': 'HNSW Cities Example',
        'description': 'Concrete example of HNSW algorithm using city locations as embedding space',
        'keywords': 'HNSW, vector search, cities example, nearest neighbor, visualization'
    },
    'generate_hybrid_search_flow.py': {
        'output': 'hybrid_search_flow.pdf',
        'name': 'Hybrid Search Flow',
        'description': 'Flowchart showing hybrid search combining sparse (BM25) and dense vector retrieval',
        'keywords': 'hybrid search, BM25, dense retrieval, RAG, information retrieval'
    },
    'generate_rag_failures.py': {
        'output': 'rag_failures_flowchart.pdf',
        'name': 'RAG Failure Modes',
        'description': 'Flowchart illustrating common failure modes in Retrieval-Augmented Generation systems',
        'keywords': 'RAG, failure modes, retrieval errors, generation errors, debugging'
    },
    'generate_rag_conditional_probs.py': {
        'output': 'rag_conditional_probs.pdf',
        'name': 'RAG Conditional Probabilities',
        'description': 'Visualization of conditional probability decomposition in RAG: p(y|x) via marginalization over documents',
        'keywords': 'RAG, conditional probability, marginalization, Bayesian, retrieval'
    },
    'generate_rag_venn_diagrams.py': {
        'output': 'rag_venn_diagrams.pdf',
        'name': 'RAG Probability Venn Diagrams',
        'description': 'Venn diagram interpretation of RAG probabilities p(Z|X) and p(Y|X,Z)',
        'keywords': 'RAG, Venn diagram, conditional probability, set theory, visualization'
    },
    'generate_vector_db_architecture.py': {
        'output': 'vector_db_architecture.pdf',
        'name': 'Vector Database Architecture',
        'description': 'Architecture diagram showing components of a vector database for RAG systems',
        'keywords': 'vector database, architecture, embeddings, indexing, RAG'
    }
}

# Base info
AUTHOR = 'Digital-AI-Finance'
PUBLISHED_IN = 'Natural Language Processing - Final Lecture'
BASE_URL = 'https://github.com/Digital-AI-Finance/Natural-Language-Processing/tree/main/FinalLecture'


def create_metainfo(folder_name, script_name, pdf_name, chart_info):
    """Create Quantlet metainfo.txt content."""
    return f"""Name of Quantlet: '{folder_name}'

Published in: '{PUBLISHED_IN}'

Description: '{chart_info["description"]}'

Keywords: '{chart_info["keywords"]}'

Author: '{AUTHOR}'

Submitted: '{datetime.now().strftime("%Y-%m-%d")}'

Datafile: 'None'

Input: '{script_name}'

Output: '{pdf_name}'

Example: 'python {script_name}'
"""


def convert_to_quantlet():
    """Convert scripts to Quantlet folder structure."""
    print("Converting Final Lecture charts to Quantlet format...\n")

    # Create FinalLecture folder for Quantlet structure
    quantlet_dir = Path('FinalLecture')
    if not quantlet_dir.exists():
        quantlet_dir.mkdir()
        print(f"Created: {quantlet_dir}/\n")

    # Source directories
    python_dir = Path('figures/python')
    figures_dir = Path('figures')

    chart_num = 1

    for script_name, chart_info in CHART_SCRIPTS.items():
        script_path = python_dir / script_name
        pdf_name = chart_info['output']
        pdf_path = figures_dir / pdf_name

        # Create numbered folder
        folder_name = f"{chart_num:02d}_{script_name.replace('generate_', '').replace('.py', '')}"
        folder_path = quantlet_dir / folder_name

        if not folder_path.exists():
            folder_path.mkdir()

        print(f"Processing: {folder_name}")

        # Copy Python script
        if script_path.exists():
            shutil.copy(script_path, folder_path / script_name)
            print(f"    -> Copied: {script_name}")
        else:
            print(f"    WARNING: Script not found: {script_name}")

        # Copy PDF
        if pdf_path.exists():
            shutil.copy(pdf_path, folder_path / pdf_name)
            print(f"    -> Copied: {pdf_name}")
        else:
            print(f"    WARNING: PDF not found: {pdf_name}")

        # Create metainfo.txt
        metainfo_content = create_metainfo(folder_name, script_name, pdf_name, chart_info)
        metainfo_path = folder_path / 'metainfo.txt'
        with open(metainfo_path, 'w', encoding='utf-8') as f:
            f.write(metainfo_content)
        print(f"    -> Created: metainfo.txt")

        print()
        chart_num += 1

    # Create README for the FinalLecture folder
    readme_content = f"""# NLP Final Lecture - Quantlet Charts

This folder contains Quantlet-formatted chart scripts from the NLP Final Lecture covering:

- **RAG (Retrieval-Augmented Generation)**
- **AI Agents and Tool Use**
- **Advanced Reasoning (Chain-of-Thought, o1, DeepSeek-R1)**
- **RLHF and Alignment**

## Charts

| Folder | Description |
|--------|-------------|
"""
    for script_name, chart_info in CHART_SCRIPTS.items():
        folder_name = f"{list(CHART_SCRIPTS.keys()).index(script_name)+1:02d}_{script_name.replace('generate_', '').replace('.py', '')}"
        readme_content += f"| [{folder_name}]({folder_name}/) | {chart_info['name']} |\n"

    readme_content += f"""
## Usage

Each folder contains:
- `generate_*.py` - Python script to generate the chart
- `*.pdf` - Generated chart PDF
- `metainfo.txt` - Quantlet metadata

To regenerate a chart:
```bash
cd <folder>
python generate_*.py
```

## Author

{AUTHOR}

## Repository

{BASE_URL}
"""

    readme_path = quantlet_dir / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"Created: {readme_path}")

    print("\n" + "="*60)
    print(f"COMPLETE: Created {chart_num-1} Quantlet folders in FinalLecture/")
    print("="*60)


if __name__ == '__main__':
    convert_to_quantlet()
