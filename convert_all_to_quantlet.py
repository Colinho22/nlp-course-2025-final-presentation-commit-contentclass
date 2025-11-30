"""
Convert ALL Final Lecture chart PDFs to Quantlet folder structure.
Creates numbered folders for each chart with metainfo.txt and QR codes.
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

# Configuration
REPO_URL = "https://github.com/Digital-AI-Finance/Natural-Language-Processing"
AUTHOR = 'Digital-AI-Finance'
PUBLISHED_IN = 'Natural Language Processing - Final Lecture'

# All chart PDFs with descriptions
ALL_CHARTS = {
    'agent_loop.pdf': {
        'name': 'AI Agent Loop',
        'description': 'Visualization of the AI agent execution loop with perception, reasoning, and action phases',
        'keywords': 'AI agent, agent loop, perception, reasoning, action, LLM agents'
    },
    'agent_loop_equations.pdf': {
        'name': 'Agent Loop Equations',
        'description': 'Mathematical formulation of the AI agent loop with state transitions and policy functions',
        'keywords': 'AI agent, equations, policy function, state transition, MDP'
    },
    'ai_timeline.pdf': {
        'name': 'AI Timeline',
        'description': 'Historical timeline of AI and NLP developments from early models to modern transformers',
        'keywords': 'AI history, timeline, NLP milestones, transformers, GPT'
    },
    'alignment_timeline.pdf': {
        'name': 'AI Alignment Timeline',
        'description': 'Timeline of AI alignment research and safety developments',
        'keywords': 'AI alignment, safety, RLHF timeline, AI ethics, responsible AI'
    },
    'ann_math_concept.pdf': {
        'name': 'ANN Mathematical Concept',
        'description': 'Visualization of artificial neural network mathematical formulation including input, hidden layers, and output',
        'keywords': 'neural network, deep learning, mathematical formulation, ANN, visualization'
    },
    'ann_search_visualization.pdf': {
        'name': 'ANN Search Visualization',
        'description': 'Visualization of approximate nearest neighbor search in embedding space',
        'keywords': 'ANN search, nearest neighbor, embedding space, vector search'
    },
    'attention_heatmap.pdf': {
        'name': 'Attention Heatmap',
        'description': 'Heatmap visualization of transformer attention weights across tokens',
        'keywords': 'attention mechanism, transformer, heatmap, self-attention, visualization'
    },
    'bleu_comparison.pdf': {
        'name': 'BLEU Score Comparison',
        'description': 'Comparison of BLEU scores across different translation models',
        'keywords': 'BLEU score, machine translation, evaluation metrics, NLP metrics'
    },
    'bm25_vs_dense.pdf': {
        'name': 'BM25 vs Dense Retrieval',
        'description': 'Comparison of sparse BM25 and dense vector retrieval methods',
        'keywords': 'BM25, dense retrieval, sparse retrieval, information retrieval, RAG'
    },
    'chunking_strategies_visual.pdf': {
        'name': 'Chunking Strategies',
        'description': 'Visual comparison of different text chunking strategies for RAG systems',
        'keywords': 'chunking, text splitting, RAG, document processing, retrieval'
    },
    'chunk_size_tradeoff.pdf': {
        'name': 'Chunk Size Tradeoff',
        'description': 'Analysis of tradeoffs between chunk size and retrieval quality in RAG',
        'keywords': 'chunk size, RAG optimization, retrieval quality, context window'
    },
    'convergence_diagram.pdf': {
        'name': 'Training Convergence',
        'description': 'Diagram showing model training convergence patterns and loss curves',
        'keywords': 'convergence, training, loss curve, optimization, deep learning'
    },
    'cot_accuracy_gains.pdf': {
        'name': 'Chain-of-Thought Accuracy',
        'description': 'Accuracy improvements from Chain-of-Thought prompting across tasks',
        'keywords': 'chain-of-thought, CoT, prompting, reasoning, accuracy'
    },
    'course_journey.pdf': {
        'name': 'Course Journey',
        'description': 'Visual overview of the NLP course structure and learning path',
        'keywords': 'course structure, NLP curriculum, learning path, education'
    },
    'deepseek_r1_pipeline.pdf': {
        'name': 'DeepSeek-R1 Pipeline',
        'description': 'Architecture and training pipeline of DeepSeek-R1 reasoning model',
        'keywords': 'DeepSeek-R1, reasoning model, training pipeline, reinforcement learning'
    },
    'dpo_vs_rlhf_comparison.pdf': {
        'name': 'DPO vs RLHF Comparison',
        'description': 'Comparison of Direct Preference Optimization and RLHF approaches',
        'keywords': 'DPO, RLHF, preference learning, alignment, fine-tuning'
    },
    'embedding_space_2d.pdf': {
        'name': 'Embedding Space 2D',
        'description': '2D visualization of word or document embeddings in semantic space',
        'keywords': 'embeddings, vector space, dimensionality reduction, semantic similarity'
    },
    'encoder_decoder.pdf': {
        'name': 'Encoder-Decoder Architecture',
        'description': 'Visualization of encoder-decoder transformer architecture',
        'keywords': 'encoder-decoder, transformer, seq2seq, architecture, attention'
    },
    'hnsw_cities_example.pdf': {
        'name': 'HNSW Cities Example',
        'description': 'Concrete example of HNSW algorithm using city locations as embedding space',
        'keywords': 'HNSW, vector search, cities example, nearest neighbor, visualization'
    },
    'hnsw_explanation.pdf': {
        'name': 'HNSW Algorithm Explanation',
        'description': 'Hierarchical Navigable Small World graph visualization for approximate nearest neighbor search',
        'keywords': 'HNSW, approximate nearest neighbor, vector search, graph algorithm, RAG'
    },
    'hybrid_search_flow.pdf': {
        'name': 'Hybrid Search Flow',
        'description': 'Flowchart showing hybrid search combining sparse (BM25) and dense vector retrieval',
        'keywords': 'hybrid search, BM25, dense retrieval, RAG, information retrieval'
    },
    'inference_scaling_curve.pdf': {
        'name': 'Inference Scaling Curve',
        'description': 'Scaling curves showing inference compute vs performance tradeoffs',
        'keywords': 'inference scaling, compute scaling, performance, efficiency'
    },
    'intermediate_computation.pdf': {
        'name': 'Intermediate Computation',
        'description': 'Visualization of intermediate computation steps in neural networks',
        'keywords': 'intermediate computation, hidden states, neural network, forward pass'
    },
    'model_comparison.pdf': {
        'name': 'Model Comparison',
        'description': 'Comparison of different language model architectures and capabilities',
        'keywords': 'model comparison, LLM benchmark, architecture comparison, performance'
    },
    'rag_architecture.pdf': {
        'name': 'RAG Architecture',
        'description': 'Complete architecture diagram of Retrieval-Augmented Generation systems',
        'keywords': 'RAG, architecture, retrieval, generation, knowledge base'
    },
    'rag_conditional_probs.pdf': {
        'name': 'RAG Conditional Probabilities',
        'description': 'Visualization of conditional probability decomposition in RAG: p(y|x) via marginalization over documents',
        'keywords': 'RAG, conditional probability, marginalization, Bayesian, retrieval'
    },
    'rag_failures_flowchart.pdf': {
        'name': 'RAG Failure Modes',
        'description': 'Flowchart illustrating common failure modes in Retrieval-Augmented Generation systems',
        'keywords': 'RAG, failure modes, retrieval errors, generation errors, debugging'
    },
    'rag_venn_diagrams.pdf': {
        'name': 'RAG Probability Venn Diagrams',
        'description': 'Venn diagram interpretation of RAG probabilities p(Z|X) and p(Y|X,Z)',
        'keywords': 'RAG, Venn diagram, conditional probability, set theory, visualization'
    },
    'reward_hacking_examples.pdf': {
        'name': 'Reward Hacking Examples',
        'description': 'Examples of reward hacking and specification gaming in RL systems',
        'keywords': 'reward hacking, specification gaming, RLHF, alignment, safety'
    },
    'rlhf_detailed_pipeline.pdf': {
        'name': 'RLHF Detailed Pipeline',
        'description': 'Detailed pipeline of Reinforcement Learning from Human Feedback training',
        'keywords': 'RLHF, pipeline, reward model, PPO, human feedback'
    },
    'rlhf_vs_dpo.pdf': {
        'name': 'RLHF vs DPO Overview',
        'description': 'Overview comparison of RLHF and DPO training approaches',
        'keywords': 'RLHF, DPO, comparison, alignment methods, preference learning'
    },
    'self_consistency_voting.pdf': {
        'name': 'Self-Consistency Voting',
        'description': 'Visualization of self-consistency decoding with majority voting',
        'keywords': 'self-consistency, voting, decoding, reasoning, ensemble'
    },
    'sequence_analysis.pdf': {
        'name': 'Sequence Analysis',
        'description': 'Analysis and visualization of sequence patterns in NLP',
        'keywords': 'sequence analysis, pattern recognition, NLP, text analysis'
    },
    'test_time_scaling.pdf': {
        'name': 'Test-Time Scaling',
        'description': 'Analysis of test-time compute scaling for improved model performance',
        'keywords': 'test-time scaling, inference compute, o1, reasoning, scaling laws'
    },
    'training_dashboard.pdf': {
        'name': 'Training Dashboard',
        'description': 'Dashboard visualization of model training metrics and progress',
        'keywords': 'training dashboard, metrics, monitoring, deep learning, visualization'
    },
    'vector_db_architecture.pdf': {
        'name': 'Vector Database Architecture',
        'description': 'Architecture diagram showing components of a vector database for RAG systems',
        'keywords': 'vector database, architecture, embeddings, indexing, RAG'
    }
}


def create_metainfo(folder_name, pdf_name, chart_info):
    """Create Quantlet metainfo.txt content."""
    return f"""Name of Quantlet: '{folder_name}'

Published in: '{PUBLISHED_IN}'

Description: '{chart_info["description"]}'

Keywords: '{chart_info["keywords"]}'

Author: '{AUTHOR}'

Submitted: '{datetime.now().strftime("%Y-%m-%d")}'

Datafile: 'None'

Input: 'None'

Output: '{pdf_name}'

Example: 'View PDF chart'
"""


def convert_all_charts():
    """Convert all chart PDFs to Quantlet folder structure."""
    print("Converting ALL charts to Quantlet format...")
    print("=" * 60)

    final_lecture_dir = Path('FinalLecture')
    figures_dir = Path('figures')

    # Get existing folders to determine next number
    existing = [d.name for d in final_lecture_dir.iterdir() if d.is_dir()]
    existing_nums = [int(d[:2]) for d in existing if d[:2].isdigit()]
    next_num = max(existing_nums) + 1 if existing_nums else 1

    # Track which PDFs already have folders
    existing_pdfs = set()
    for d in final_lecture_dir.iterdir():
        if d.is_dir():
            for pdf in d.glob('*.pdf'):
                existing_pdfs.add(pdf.name)

    created_count = 0

    for pdf_name, chart_info in ALL_CHARTS.items():
        pdf_path = figures_dir / pdf_name

        if not pdf_path.exists():
            print(f"  [SKIP] {pdf_name}: not found in figures/")
            continue

        if pdf_name in existing_pdfs:
            print(f"  [EXISTS] {pdf_name}: already in Quantlet format")
            continue

        # Create folder name from PDF name
        base_name = pdf_name.replace('.pdf', '')
        folder_name = f"{next_num:02d}_{base_name}"
        folder_path = final_lecture_dir / folder_name

        folder_path.mkdir(exist_ok=True)

        # Copy PDF
        shutil.copy(pdf_path, folder_path / pdf_name)

        # Create metainfo.txt
        metainfo_content = create_metainfo(folder_name, pdf_name, chart_info)
        with open(folder_path / 'metainfo.txt', 'w', encoding='utf-8') as f:
            f.write(metainfo_content)

        # Generate QR code
        if HAS_QRCODE:
            chart_url = f"{REPO_URL}/tree/main/FinalLecture/{folder_name}"
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=2,
            )
            qr.add_data(chart_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(folder_path / 'qr_code.png')

        print(f"  [OK] {folder_name}/")
        next_num += 1
        created_count += 1

    print("=" * 60)
    print(f"Created {created_count} new Quantlet folders")
    print(f"Total folders: {next_num - 1}")

    return next_num - 1


if __name__ == '__main__':
    convert_all_charts()
