#!/usr/bin/env python3
"""
Comprehensive page improvement script for NLP Course Site
- Converts PDF charts to PNG
- Updates all HTML pages with improved content, learning objectives, and chart galleries
"""

import os
import re
import subprocess
from pathlib import Path
from bs4 import BeautifulSoup

def get_docs_dir():
    return Path(__file__).parent.parent

def get_nlp_slides_dir():
    return Path(__file__).parent.parent.parent / 'NLP_slides'

# Topic data with learning objectives and chart mappings
TOPIC_DATA = {
    'ngrams': {
        'title': 'N-gram Language Models',
        'subtitle': 'Statistical Foundation of Language Modeling',
        'overview': 'Learn how to predict the next word using statistical patterns from text. N-gram models form the mathematical foundation for understanding language probability and serve as the baseline for all modern language models.',
        'objectives': [
            'Calculate conditional probabilities P(word|context) from corpus statistics',
            'Implement smoothing techniques (Laplace, Add-k) to handle unseen n-grams',
            'Evaluate language models using perplexity metrics',
            'Understand the trade-offs between model complexity and data sparsity'
        ],
        'charts': ['bigram_calculation_steps_bsc', 'conditional_probability_tree_bsc', 'corpus_statistics_bsc', 'addk_smoothing_bsc'],
        'week_folder': 'week01_foundations'
    },
    'embeddings': {
        'title': 'Word Embeddings',
        'subtitle': 'From Words to Vectors',
        'overview': 'Transform discrete words into continuous vector spaces where semantic relationships emerge as geometric properties. Word2Vec, GloVe, and FastText revolutionized NLP by capturing meaning in dense representations.',
        'objectives': [
            'Understand Skip-gram and CBOW architectures for learning embeddings',
            'Implement negative sampling for efficient training',
            'Perform vector arithmetic (king - man + woman = queen)',
            'Evaluate embedding quality using analogy tasks'
        ],
        'charts': ['word_arithmetic_3d_bsc', 'skipgram_architecture_bsc', 'negative_sampling_process_bsc', 'training_evolution_bsc'],
        'week_folder': 'week02_neural_lm'
    },
    'rnn-lstm': {
        'title': 'RNN & LSTM Networks',
        'subtitle': 'Sequential Memory in Neural Networks',
        'overview': 'Process sequences of arbitrary length using recurrent connections. LSTMs solve the vanishing gradient problem with gated memory cells, enabling learning of long-range dependencies in text.',
        'objectives': [
            'Implement forward and backward passes through recurrent networks',
            'Understand why vanilla RNNs fail on long sequences (vanishing gradients)',
            'Design LSTM cells with input, forget, and output gates',
            'Apply bidirectional RNNs for context from both directions'
        ],
        'charts': ['lstm_architecture_bsc', 'vanishing_gradient_bsc', 'gradient_flow_comparison_bsc', 'rnn_unrolled_bsc'],
        'week_folder': 'week03_rnn'
    },
    'seq2seq': {
        'title': 'Sequence-to-Sequence',
        'subtitle': 'Encoder-Decoder Architectures',
        'overview': 'Map variable-length input sequences to variable-length outputs using encoder-decoder architectures. Attention mechanisms allow the decoder to focus on relevant parts of the input.',
        'objectives': [
            'Design encoder-decoder architectures for translation and summarization',
            'Implement attention mechanisms (Bahdanau, Luong)',
            'Apply beam search for improved decoding quality',
            'Evaluate sequence outputs using BLEU and ROUGE scores'
        ],
        'charts': ['encoder_decoder_architecture_bsc', 'attention_heatmap_bsc', 'beam_search_tree_bsc', 'bleu_comparison_bsc'],
        'week_folder': 'week04_seq2seq'
    },
    'transformers': {
        'title': 'Transformer Architecture',
        'subtitle': 'Attention Is All You Need',
        'overview': 'Replace recurrence with self-attention for parallel processing of sequences. Transformers process all positions simultaneously while learning rich contextual representations through multi-head attention.',
        'objectives': [
            'Implement scaled dot-product and multi-head attention',
            'Understand positional encoding for sequence order',
            'Design transformer blocks with residual connections and layer normalization',
            'Compare computational complexity: O(n) recurrent vs O(1) transformer depth'
        ],
        'charts': ['3d_transformer_architecture', 'attention_heatmap_bsc', 'multihead_attention_bsc', 'positional_encoding_bsc'],
        'week_folder': 'week05_transformers'
    },
    'pretrained': {
        'title': 'Pre-trained Models',
        'subtitle': 'BERT, GPT, and Transfer Learning',
        'overview': 'Leverage massive pre-training to learn universal language representations. BERT uses masked language modeling while GPT uses autoregressive prediction, each excelling at different downstream tasks.',
        'objectives': [
            'Compare BERT (bidirectional) vs GPT (autoregressive) pre-training objectives',
            'Fine-tune pre-trained models for classification, QA, and NER',
            'Understand the cost-performance trade-offs of large language models',
            'Apply prompt engineering techniques for few-shot learning'
        ],
        'charts': ['bert_architecture_bsc', 'bert_vs_gpt_architecture_bsc', 'pretraining_workflow_bsc', 'bert_results_glue_bsc'],
        'week_folder': 'week06_pretrained'
    },
    'scaling': {
        'title': 'Scaling & Advanced Topics',
        'subtitle': 'From GPT-2 to GPT-4',
        'overview': 'Explore scaling laws that govern model performance: more parameters, more data, more compute yield predictable improvements. Emergent abilities appear at scale that smaller models lack.',
        'objectives': [
            'Apply Chinchilla scaling laws to optimize training budgets',
            'Understand emergent abilities that appear only at scale',
            'Design mixture-of-experts architectures for efficient scaling',
            'Evaluate the compute-performance frontier for language models'
        ],
        'charts': ['scaling_laws_bsc', 'emergent_abilities_chart_bsc', 'model_scale_timeline_bsc', 'compute_scaling_bsc'],
        'week_folder': 'week07_advanced'
    },
    'tokenization': {
        'title': 'Tokenization',
        'subtitle': 'From Characters to Subwords',
        'overview': 'Convert raw text into model-digestible tokens. BPE, WordPiece, and SentencePiece balance vocabulary size against sequence length, handling rare words through subword decomposition.',
        'objectives': [
            'Implement Byte-Pair Encoding (BPE) from scratch',
            'Compare tokenization strategies (word, subword, character)',
            'Handle multilingual text with unified tokenizers',
            'Analyze the impact of vocabulary size on model performance'
        ],
        'charts': ['bpe_visualization_bsc', 'bpe_progression_visual_bsc', 'tokenization_comparison_bsc', 'tokenization_impact_bsc'],
        'week_folder': 'week08_tokenization'
    },
    'decoding': {
        'title': 'Decoding Strategies',
        'subtitle': 'From Logits to Text',
        'overview': 'Transform model probability distributions into coherent text. Different strategies trade off quality, diversity, and speed: greedy for determinism, sampling for creativity, beam search for quality.',
        'objectives': [
            'Implement greedy, beam search, and nucleus sampling strategies',
            'Understand temperature scaling and its effect on output diversity',
            'Diagnose and fix text degeneration (repetition, incoherence)',
            'Choose appropriate decoding strategies for different applications'
        ],
        'charts': ['beam_search_tree_graphviz', 'temperature_effects_bsc', 'degeneration_problem_bsc', 'contrastive_vs_nucleus_bsc'],
        'week_folder': 'week09_decoding'
    },
    'finetuning': {
        'title': 'Fine-tuning Methods',
        'subtitle': 'Efficient Adaptation',
        'overview': 'Adapt pre-trained models to specific tasks efficiently. LoRA and adapter methods reduce trainable parameters by 99% while maintaining performance, enabling deployment on resource-limited devices.',
        'objectives': [
            'Implement Low-Rank Adaptation (LoRA) for efficient fine-tuning',
            'Design adapter modules for task-specific layers',
            'Prevent catastrophic forgetting during fine-tuning',
            'Compare full fine-tuning vs parameter-efficient methods'
        ],
        'charts': ['lora_explanation_bsc', 'adapter_architecture_bsc', 'parameter_spectrum_bsc', 'catastrophic_forgetting_bsc'],
        'week_folder': 'week10_finetuning'
    },
    'efficiency': {
        'title': 'Efficiency & Optimization',
        'subtitle': 'Making Models Practical',
        'overview': 'Deploy models efficiently through quantization, pruning, and distillation. Reduce model size by 4-8x with minimal quality loss, enabling inference on edge devices and reducing serving costs.',
        'objectives': [
            'Apply quantization (INT8, INT4) to reduce model size and latency',
            'Implement knowledge distillation from large to small models',
            'Design pruning strategies for sparse, efficient networks',
            'Optimize inference pipelines for production deployment'
        ],
        'charts': ['model_compression_landscape_bsc', 'quantization_levels_bsc', 'distillation_architecture_bsc', 'deployment_pipeline_bsc'],
        'week_folder': 'week11_efficiency'
    },
    'ethics': {
        'title': 'Ethics & Bias',
        'subtitle': 'Responsible AI Development',
        'overview': 'Understand and mitigate biases in language models. Models learn societal biases from training data; responsible development requires bias detection, mitigation, and ongoing monitoring.',
        'objectives': [
            'Identify sources of bias in training data and model outputs',
            'Implement fairness metrics and bias detection methods',
            'Apply debiasing techniques during training and inference',
            'Design responsible AI governance frameworks'
        ],
        'charts': ['bias_sources_flowchart_bsc', 'fairness_metrics_comparison_bsc', 'harm_taxonomy_tree_bsc', 'real_world_harms_bsc'],
        'week_folder': 'week12_ethics'
    }
}

# Module data
MODULE_DATA = {
    'embeddings': {
        'title': 'Word Embeddings Deep Dive',
        'subtitle': 'Skip-gram Mathematics & 3D Visualizations',
        'overview': 'Comprehensive coverage of word embedding algorithms with interactive 3D visualizations. Explore the mathematical foundations of Word2Vec, GloVe, and FastText.',
        'objectives': [
            'Derive the Skip-gram objective function mathematically',
            'Visualize embedding spaces in 3D',
            'Understand the geometry of semantic relationships',
            'Implement embedding training from scratch'
        ],
        'charts': ['word_arithmetic_3d_bsc', 'skipgram_architecture_bsc', 'cbow_architecture_bsc', 'training_evolution_bsc']
    },
    'summarization': {
        'title': 'LLM Summarization',
        'subtitle': 'Extractive and Abstractive Methods',
        'overview': 'Techniques for automatic text summarization using large language models. Compare extractive methods that select key sentences with abstractive methods that generate new text.',
        'objectives': [
            'Implement extractive summarization with sentence scoring',
            'Apply transformer models for abstractive summarization',
            'Evaluate summaries using ROUGE metrics',
            'Design prompts for zero-shot summarization'
        ],
        'charts': ['abstractive_vs_extractive_bsc', 'attention_mechanism_visual_bsc', 'rouge_comparison_bsc', 'summary_pipeline_bsc']
    },
    'sentiment': {
        'title': 'Sentiment Analysis',
        'subtitle': 'BERT Fine-tuning for Classification',
        'overview': 'Fine-tune BERT for sentiment classification. Learn the complete pipeline from data preparation through model deployment for production sentiment analysis.',
        'objectives': [
            'Prepare text data for transformer classification',
            'Fine-tune BERT with a classification head',
            'Interpret attention patterns for explainability',
            'Deploy sentiment models to production'
        ],
        'charts': ['bert_attention_sentiment_bsc', 'fine_tuning_process_sentiment_bsc', 'confusion_matrix_bsc', 'attention_heatmap_bsc']
    },
    'lstm-primer': {
        'title': 'LSTM Primer',
        'subtitle': 'Understanding Recurrent Memory',
        'overview': 'Deep dive into LSTM architecture and gating mechanisms. Understand why LSTMs solve the vanishing gradient problem and how to implement them from scratch.',
        'objectives': [
            'Implement LSTM cells with gating mechanisms',
            'Trace gradient flow through LSTM networks',
            'Compare LSTM vs GRU architectures',
            'Debug common LSTM training issues'
        ],
        'charts': ['lstm_architecture_bsc', 'cell_state_sequence_bsc', 'gradient_flow_comparison_bsc', 'bptt_visualization_bsc']
    }
}


def find_chart_pdf(chart_name, week_folder):
    """Find a PDF chart in the NLP_slides folder"""
    nlp_dir = get_nlp_slides_dir()
    figures_dir = nlp_dir / week_folder / 'figures'

    if figures_dir.exists():
        # Try exact match first
        pdf_path = figures_dir / f'{chart_name}.pdf'
        if pdf_path.exists():
            return pdf_path

        # Try without _bsc suffix
        base_name = chart_name.replace('_bsc', '')
        pdf_path = figures_dir / f'{base_name}.pdf'
        if pdf_path.exists():
            return pdf_path

        # Search for partial match
        for pdf in figures_dir.glob('*.pdf'):
            if base_name in pdf.stem.lower():
                return pdf

    return None


def convert_pdf_to_png(pdf_path, output_path, dpi=150):
    """Convert PDF to PNG using pdftoppm or pdf2image"""
    try:
        # Try using pdf2image (requires poppler)
        from pdf2image import convert_from_path
        images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
        if images:
            images[0].save(str(output_path), 'PNG')
            return True
    except ImportError:
        pass
    except Exception as e:
        print(f"    pdf2image failed: {e}")

    # Fallback: try ImageMagick convert
    try:
        result = subprocess.run(
            ['magick', 'convert', '-density', str(dpi), f'{pdf_path}[0]', str(output_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    return False


def convert_charts_for_topic(topic_id, topic_data, output_dir):
    """Convert PDF charts to PNG for a topic"""
    converted = []
    week_folder = topic_data.get('week_folder', '')

    for chart_name in topic_data.get('charts', []):
        output_path = output_dir / f'{chart_name}.png'

        # Skip if already exists
        if output_path.exists():
            converted.append(chart_name)
            continue

        # Find and convert PDF
        pdf_path = find_chart_pdf(chart_name, week_folder)
        if pdf_path:
            print(f"  Converting {pdf_path.name} -> {chart_name}.png")
            if convert_pdf_to_png(pdf_path, output_path):
                converted.append(chart_name)
            else:
                print(f"    Failed to convert {pdf_path.name}")
        else:
            print(f"    Chart not found: {chart_name} in {week_folder}")

    return converted


def generate_topic_html(topic_id, topic_data, existing_charts):
    """Generate improved HTML content for a topic page"""
    charts_html = ""
    if existing_charts:
        chart_items = []
        for chart in existing_charts[:4]:  # Max 4 charts
            chart_title = chart.replace('_bsc', '').replace('_', ' ').title()
            chart_items.append(f'''        <div class="chart-item">
          <img src="../assets/images/{chart}.png" alt="{chart_title}" loading="lazy">
          <span>{chart_title[:20]}</span>
        </div>''')
        charts_html = f'''
      <section class="section topic-charts">
        <h3>Key Visualizations</h3>
        <div class="chart-grid">
{chr(10).join(chart_items)}
        </div>
      </section>'''

    objectives_html = "\n".join([f'          <li>{obj}</li>' for obj in topic_data.get('objectives', [])])

    return {
        'overview': topic_data.get('overview', ''),
        'objectives': objectives_html,
        'charts_section': charts_html,
        'subtitle': topic_data.get('subtitle', '')
    }


def update_topic_page(html_path, topic_id, topic_data, existing_charts):
    """Update a topic HTML page with improved content"""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Generate new content
    new_content = generate_topic_html(topic_id, topic_data, existing_charts)

    # Find and update overview
    overview_section = soup.find('section', class_='section')
    if overview_section:
        h2 = overview_section.find('h2')
        if h2 and 'Overview' in h2.text:
            p = overview_section.find('p')
            if p:
                p.string = new_content['overview']

    # Check if learning objectives section exists, if not add it
    sections = soup.find_all('section', class_='section')
    has_objectives = any('objectives' in str(s).lower() for s in sections)

    if not has_objectives and new_content['objectives']:
        # Find the overview section and add objectives after it
        for section in sections:
            h2 = section.find('h2')
            if h2 and 'Overview' in h2.text:
                objectives_section = soup.new_tag('section')
                objectives_section['class'] = 'section'
                objectives_section.append(BeautifulSoup(f'''
        <h2>Learning Objectives</h2>
        <ul class="objectives-list">
{new_content['objectives']}
        </ul>''', 'html.parser'))
                section.insert_after(objectives_section)
                break

    # Check if chart gallery exists, if not add it
    has_charts = soup.find('section', class_='topic-charts')
    if not has_charts and new_content['charts_section']:
        # Find key topics section and add charts before it
        for section in sections:
            h2 = section.find('h2')
            if h2 and 'Key Topics' in h2.text:
                charts_section = BeautifulSoup(new_content['charts_section'], 'html.parser')
                section.insert_before(charts_section)
                break

    # Write updated content
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))

    return True


def main():
    print("=" * 60)
    print("NLP Course Site - Page Improvement Script")
    print("=" * 60)

    docs_dir = get_docs_dir()
    images_dir = docs_dir / 'assets' / 'images'

    # Phase 1: Convert charts
    print("\n[1/3] Converting PDF charts to PNG...")
    all_converted = []

    for topic_id, topic_data in TOPIC_DATA.items():
        print(f"\n  Topic: {topic_id}")
        converted = convert_charts_for_topic(topic_id, topic_data, images_dir)
        all_converted.extend(converted)

    print(f"\n  Total charts: {len(all_converted)} available")

    # Phase 2: Update topic pages
    print("\n[2/3] Updating topic HTML pages...")
    topics_dir = docs_dir / 'topics'

    for topic_id, topic_data in TOPIC_DATA.items():
        html_path = topics_dir / f'{topic_id}.html'
        if html_path.exists():
            # Get existing charts for this topic
            existing_charts = [c for c in topic_data.get('charts', [])
                            if (images_dir / f'{c}.png').exists()]

            print(f"  Updating {topic_id}.html ({len(existing_charts)} charts)")
            update_topic_page(html_path, topic_id, topic_data, existing_charts)

    # Phase 3: Update module pages
    print("\n[3/3] Updating module HTML pages...")
    modules_dir = docs_dir / 'modules'

    for module_id, module_data in MODULE_DATA.items():
        html_path = modules_dir / f'{module_id}.html'
        if html_path.exists():
            existing_charts = [c for c in module_data.get('charts', [])
                            if (images_dir / f'{c}.png').exists()]

            print(f"  Updating {module_id}.html ({len(existing_charts)} charts)")
            # Use same update logic as topics
            update_topic_page(html_path, module_id, module_data, existing_charts)

    print("\n" + "=" * 60)
    print("PAGE IMPROVEMENT COMPLETE")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
