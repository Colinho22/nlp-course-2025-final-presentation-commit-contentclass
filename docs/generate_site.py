#!/usr/bin/env python3
"""
NLP Course Site Generator
Generates all pages with consistent layout, sidebar, top navigation, and content-based naming
"""

import os
from pathlib import Path

# GitHub repository base URL for raw files
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Digital-AI-Finance/Natural-Language-Processing/main"
GITHUB_REPO = "https://github.com/Digital-AI-Finance/Natural-Language-Processing"

# Content data - topic-based naming
WEEKS = [
    {
        'num': '01',
        'id': 'ngrams',
        'title': 'N-gram Language Models',
        'subtitle': 'Statistical Foundations of NLP',
        'part': 1,
        'description': 'Learn how to predict the next word using probability and counting. Foundation for all language models.',
        'topics': ['Probability basics', 'Bigram models', 'Smoothing techniques', 'Perplexity evaluation'],
        'slides': 42,
        'pdf_path': 'NLP_slides/week01_foundations/presentations',
        'lab_path': 'NLP_slides/week01_foundations/lab'
    },
    {
        'num': '02',
        'id': 'embeddings',
        'title': 'Word Embeddings',
        'subtitle': 'From Words to Vectors',
        'part': 1,
        'description': 'Transform words into dense vectors that capture meaning. Word2Vec, GloVe, and semantic relationships.',
        'topics': ['One-hot encoding', 'Skip-gram architecture', 'Negative sampling', 'Word analogies'],
        'slides': 35,
        'pdf_path': 'NLP_slides/week02_embeddings/presentations',
        'lab_path': 'NLP_slides/week02_embeddings/lab'
    },
    {
        'num': '03',
        'id': 'rnn-lstm',
        'title': 'RNN & LSTM Networks',
        'subtitle': 'Sequential Memory',
        'part': 1,
        'description': 'Process sequences with memory. Understand vanishing gradients and how LSTMs solve them.',
        'topics': ['Recurrent connections', 'Vanishing gradients', 'LSTM gates', 'Sequence modeling'],
        'slides': 21,
        'pdf_path': 'NLP_slides/week03_rnn_lstm/presentations',
        'lab_path': 'NLP_slides/week03_rnn_lstm/lab'
    },
    {
        'num': '04',
        'id': 'seq2seq',
        'title': 'Sequence-to-Sequence',
        'subtitle': 'Encoder-Decoder Architecture',
        'part': 2,
        'description': 'Map sequences to sequences. The foundation of machine translation and summarization.',
        'topics': ['Encoder-decoder', 'Attention mechanism', 'Teacher forcing', 'Beam search'],
        'slides': 38,
        'pdf_path': 'NLP_slides/week04_seq2seq/presentations',
        'lab_path': 'NLP_slides/week04_seq2seq/lab'
    },
    {
        'num': '05',
        'id': 'transformers',
        'title': 'Transformer Architecture',
        'subtitle': 'Attention Is All You Need',
        'part': 2,
        'description': 'The architecture that revolutionized NLP. Self-attention, multi-head attention, and positional encoding.',
        'topics': ['Self-attention', 'Multi-head attention', 'Positional encoding', 'Feed-forward layers'],
        'slides': 45,
        'pdf_path': 'NLP_slides/week05_transformers/presentations',
        'lab_path': 'NLP_slides/week05_transformers/lab'
    },
    {
        'num': '06',
        'id': 'pretrained',
        'title': 'Pre-trained Models',
        'subtitle': 'BERT, GPT & Transfer Learning',
        'part': 2,
        'description': 'Leverage massive pre-training. BERT, GPT, and the transfer learning revolution.',
        'topics': ['BERT architecture', 'GPT models', 'Fine-tuning', 'Transfer learning'],
        'slides': 52,
        'pdf_path': 'NLP_slides/week06_pretrained/presentations',
        'lab_path': 'NLP_slides/week06_pretrained/lab'
    },
    {
        'num': '07',
        'id': 'scaling',
        'title': 'Scaling & Advanced Topics',
        'subtitle': 'Large Language Models',
        'part': 3,
        'description': 'Scale to billions of parameters. Scaling laws, emergent abilities, and modern LLMs.',
        'topics': ['Scaling laws', 'Emergent abilities', 'In-context learning', 'Chain-of-thought'],
        'slides': 35,
        'pdf_path': 'NLP_slides/week07_advanced/presentations',
        'lab_path': 'NLP_slides/week07_advanced/lab'
    },
    {
        'num': '08',
        'id': 'tokenization',
        'title': 'Tokenization',
        'subtitle': 'BPE, WordPiece & SentencePiece',
        'part': 3,
        'description': 'Break text into tokens. Subword algorithms that power modern language models.',
        'topics': ['BPE algorithm', 'WordPiece', 'SentencePiece', 'Vocabulary optimization'],
        'slides': 35,
        'pdf_path': 'NLP_slides/week08_tokenization/presentations',
        'lab_path': 'NLP_slides/week08_tokenization/lab'
    },
    {
        'num': '09',
        'id': 'decoding',
        'title': 'Decoding Strategies',
        'subtitle': 'From Greedy to Nucleus Sampling',
        'part': 3,
        'description': 'Generate text from models. Greedy, beam search, temperature, top-k, and nucleus sampling.',
        'topics': ['Greedy decoding', 'Beam search', 'Temperature scaling', 'Top-k and nucleus'],
        'slides': 66,
        'pdf_path': 'NLP_slides/week09_decoding/presentations',
        'lab_path': 'NLP_slides/week09_decoding/lab'
    },
    {
        'num': '10',
        'id': 'finetuning',
        'title': 'Fine-tuning Methods',
        'subtitle': 'LoRA, Adapters & PEFT',
        'part': 4,
        'description': 'Adapt pre-trained models efficiently. LoRA, adapters, and parameter-efficient methods.',
        'topics': ['Full fine-tuning', 'LoRA', 'Adapters', 'Prompt tuning'],
        'slides': 38,
        'pdf_path': 'NLP_slides/week10_finetuning/presentations',
        'lab_path': 'NLP_slides/week10_finetuning/lab'
    },
    {
        'num': '11',
        'id': 'efficiency',
        'title': 'Efficiency & Optimization',
        'subtitle': 'Quantization & Distillation',
        'part': 4,
        'description': 'Make models faster and smaller. Quantization, pruning, and knowledge distillation.',
        'topics': ['Quantization', 'Pruning', 'Knowledge distillation', 'Inference optimization'],
        'slides': 41,
        'pdf_path': 'NLP_slides/week11_efficiency/presentations',
        'lab_path': 'NLP_slides/week11_efficiency/lab'
    },
    {
        'num': '12',
        'id': 'ethics',
        'title': 'Ethics & Bias',
        'subtitle': 'Responsible AI Development',
        'part': 4,
        'description': 'Build responsible AI systems. Bias detection, fairness, and ethical considerations.',
        'topics': ['Bias in models', 'Fairness metrics', 'Mitigation strategies', 'Responsible deployment'],
        'slides': 26,
        'pdf_path': 'NLP_slides/week12_ethics/presentations',
        'lab_path': 'NLP_slides/week12_ethics/lab'
    }
]

MODULES = [
    {
        'id': 'embeddings',
        'title': 'Word Embeddings Deep Dive',
        'description': 'Comprehensive coverage of word embedding algorithms with interactive 3D visualizations.',
        'topics': ['Skip-gram mathematics', 'GloVe co-occurrence', 'FastText subwords', 'Contextual embeddings'],
        'slides': 48,
        'pdf_path': 'embeddings',
        'lab_path': 'embeddings'
    },
    {
        'id': 'summarization',
        'title': 'LLM Summarization',
        'description': 'Text summarization using extractive and abstractive methods with modern LLMs.',
        'topics': ['Extractive methods', 'Abstractive generation', 'RAG enhancement', 'Evaluation metrics'],
        'slides': 40,
        'pdf_path': 'NLP_slides/summarization_module/presentations',
        'lab_path': 'NLP_slides/summarization_module/lab'
    },
    {
        'id': 'sentiment',
        'title': 'Sentiment Analysis',
        'description': 'BERT fine-tuning for sentiment classification with technical deep dives.',
        'topics': ['BERT classifier head', 'Pre-training objectives', 'Fine-tuning process', 'Deployment pipeline'],
        'slides': 26,
        'pdf_path': 'NLP_slides/sentiment_analysis_module/presentations',
        'lab_path': 'NLP_slides/sentiment_analysis_module/lab'
    },
    {
        'id': 'lstm-primer',
        'title': 'LSTM Primer',
        'description': 'Zero-prerequisite introduction to LSTM networks with clear visualizations.',
        'topics': ['Why RNNs fail', 'Gate mechanisms', 'Cell state flow', 'Practical applications'],
        'slides': 32,
        'pdf_path': 'NLP_slides/lstm_primer/presentations',
        'lab_path': 'NLP_slides/lstm_primer/lab'
    }
]

PART_NAMES = {
    1: 'Language Foundations',
    2: 'Core Architectures',
    3: 'Advanced Methods',
    4: 'Applications'
}

# Full CSS including sidebar styles
FULL_STYLES = '''
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.5;
    }
    .top-nav {
      background: #1e3a5f;
      padding: 10px 20px;
      display: flex;
      justify-content: center;
      gap: 25px;
      flex-wrap: wrap;
    }
    .top-nav a { color: white; text-decoration: none; font-size: 0.9rem; font-weight: 500; opacity: 0.9; }
    .top-nav a:hover { opacity: 1; text-decoration: underline; }
    .container { display: flex; max-width: 1600px; margin: 0 auto; }
    .sidebar {
      width: 280px;
      background: #ffffff;
      border-right: 1px solid #e1e4e8;
      height: calc(100vh - 42px);
      overflow-y: auto;
      position: sticky;
      top: 0;
      flex-shrink: 0;
    }
    .sidebar-header {
      padding: 15px;
      border-bottom: 1px solid #e1e4e8;
      background: linear-gradient(135deg, #1e3a5f 0%, #3333B2 100%);
    }
    .sidebar-header a { display: flex; align-items: center; text-decoration: none; color: white; }
    .sidebar-logo { width: 36px; height: 36px; border-radius: 6px; margin-right: 10px; background: white; padding: 3px; }
    .course-title { font-size: 15px; font-weight: 700; }
    .search-container { padding: 10px 15px; border-bottom: 1px solid #e1e4e8; }
    .search-container input {
      width: 100%; padding: 6px 10px; border: 1px solid #e1e4e8;
      border-radius: 4px; font-size: 12px; outline: none;
    }
    .search-container input:focus { border-color: #3333B2; box-shadow: 0 0 0 2px rgba(51, 51, 178, 0.1); }
    .sidebar-nav { padding: 8px 0; }
    .part-section { border-bottom: 1px solid #f0f0f0; }
    .part-section summary {
      padding: 10px 15px; font-weight: 600; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.5px; color: #24292e;
      cursor: pointer; list-style: none;
      display: flex; align-items: center; justify-content: space-between;
    }
    .part-section summary::-webkit-details-marker { display: none; }
    .part-section summary::after { content: "+"; font-size: 12px; color: #586069; }
    .part-section[open] summary::after { content: "-"; }
    .part-section summary:hover { background: #f6f8fa; }
    .part-section ul { list-style: none; padding: 0 0 6px 0; margin: 0; }
    .part-section li { margin: 0; }
    .part-section a {
      display: block; padding: 6px 15px 6px 25px; color: #586069;
      text-decoration: none; font-size: 12px; border-left: 2px solid transparent;
    }
    .part-section a:hover { background: #f6f8fa; color: #3333B2; }
    .part-section a.active { border-left-color: #3333B2; color: #3333B2; background: #f6f8fa; }
    .topic-count { font-size: 10px; color: #959da5; font-weight: normal; }
    .hidden { display: none !important; }
    .main-content { flex: 1; min-width: 0; }
    .hero {
      background: linear-gradient(135deg, #1e3a5f 0%, #3333B2 100%);
      color: white;
      padding: 30px 20px;
      text-align: center;
    }
    .hero h1 { font-size: 2rem; margin-bottom: 8px; }
    .hero .subtitle { font-size: 1rem; opacity: 0.9; margin-bottom: 12px; }
    .hero .badge {
      display: inline-block;
      background: rgba(255,255,255,0.2);
      padding: 5px 15px;
      border-radius: 20px;
      font-size: 0.85rem;
      margin-bottom: 10px;
    }
    .content-area { padding: 25px; }
    .section {
      background: white;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
      border: 1px solid #e2e8f0;
    }
    .section h2 {
      color: #1e3a5f;
      font-size: 1.2rem;
      margin-bottom: 12px;
      border-bottom: 2px solid #3333B2;
      padding-bottom: 6px;
    }
    .topics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }
    .topic-item {
      background: #f8fafc;
      padding: 10px 12px;
      border-radius: 6px;
      border-left: 3px solid #3333B2;
      font-size: 0.9rem;
    }
    .nav-cards {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-top: 15px;
    }
    .nav-card {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px;
      text-decoration: none;
      color: inherit;
      transition: box-shadow 0.2s;
    }
    .nav-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .nav-card .label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; }
    .nav-card .title { font-weight: 600; color: #1e3a5f; font-size: 0.9rem; }
    .resource-btn {
      display: inline-block;
      padding: 10px 18px;
      border-radius: 5px;
      text-decoration: none;
      font-weight: 500;
      margin: 4px;
      font-size: 0.9rem;
    }
    .resource-btn.primary { background: #3333B2; color: white; }
    .resource-btn.primary:hover { background: #2929a0; }
    .resource-btn.secondary { background: white; color: #1e3a5f; border: 1px solid #e2e8f0; }
    .resource-btn.secondary:hover { background: #f8fafc; }
    .footer {
      text-align: center;
      padding: 15px;
      color: #64748b;
      font-size: 0.8rem;
      border-top: 1px solid #e2e8f0;
      margin-top: 20px;
    }
    .footer a { color: #3333B2; }
    @media (max-width: 900px) {
      .container { flex-direction: column; }
      .sidebar { width: 100%; height: auto; position: relative; border-right: none; border-bottom: 1px solid #e1e4e8; max-height: 40vh; }
    }
    @media (max-width: 600px) {
      .topics-grid { grid-template-columns: 1fr; }
      .nav-cards { grid-template-columns: 1fr; }
    }
'''

def generate_sidebar_html(active_id=None, is_module=False, relative_path='../'):
    """Generate the sidebar HTML with navigation"""
    sidebar_parts = {1: [], 2: [], 3: [], 4: []}
    for week in WEEKS:
        active_class = ' class="active"' if week['id'] == active_id and not is_module else ''
        sidebar_parts[week['part']].append(
            f'<li><a href="{relative_path}weeks/{week["id"]}.html"{active_class}>{week["title"]}</a></li>'
        )

    sidebar_html = ''
    for part_num, part_name in PART_NAMES.items():
        links = '\n            '.join(sidebar_parts[part_num])
        # Open the section if active item is in this part
        active_week = next((w for w in WEEKS if w['id'] == active_id), None)
        open_attr = ' open' if (active_week and active_week['part'] == part_num) or part_num == 1 else ''
        sidebar_html += f'''
        <details class="part-section"{open_attr}>
          <summary>Part {part_num}: {part_name} <span class="topic-count">(3)</span></summary>
          <ul>
            {links}
          </ul>
        </details>'''

    # Generate module links
    module_links = []
    for m in MODULES:
        active_class = ' class="active"' if m['id'] == active_id and is_module else ''
        module_links.append(f'<li><a href="{relative_path}modules/{m["id"]}.html"{active_class}>{m["title"]}</a></li>')
    module_links_html = '\n            '.join(module_links)

    open_modules = ' open' if is_module else ''
    sidebar_html += f'''
        <details class="part-section"{open_modules}>
          <summary>Special Modules <span class="topic-count">({len(MODULES)})</span></summary>
          <ul>
            {module_links_html}
          </ul>
        </details>'''

    return f'''    <aside class="sidebar">
      <div class="sidebar-header">
        <a href="{relative_path}index.html">
          <img src="https://quantlet.com/images/Q.png" alt="QuantLet" class="sidebar-logo">
          <span class="course-title">NLP Course</span>
        </a>
      </div>
      <div class="search-container">
        <input type="text" id="topic-search" placeholder="Search topics...">
      </div>
      <nav class="sidebar-nav">{sidebar_html}
      </nav>
    </aside>'''

def generate_search_script():
    """Generate the search functionality script"""
    return '''
  <script>
    document.getElementById('topic-search').addEventListener('input', function(e) {
      const query = e.target.value.toLowerCase();
      const links = document.querySelectorAll('.sidebar-nav .part-section li');
      const sections = document.querySelectorAll('.sidebar-nav .part-section');
      links.forEach(li => {
        const text = li.textContent.toLowerCase();
        li.classList.toggle('hidden', query !== '' && !text.includes(query));
      });
      sections.forEach(section => {
        const visibleLinks = section.querySelectorAll('li:not(.hidden)');
        if (visibleLinks.length === 0 && query !== '') {
          section.classList.add('hidden');
        } else {
          section.classList.remove('hidden');
          if (query !== '') section.setAttribute('open', '');
        }
      });
    });
  </script>'''

def generate_top_nav(relative_path=''):
    """Generate consistent top navigation"""
    base = f'{relative_path}' if relative_path else ''
    return f'''  <nav class="top-nav">
    <a href="{base}index.html">Home</a>
    <a href="{base}index.html#topics">Topics</a>
    <a href="{base}index.html#gallery">Charts</a>
    <a href="{base}index.html#modules">Modules</a>
    <a href="{GITHUB_REPO}">GitHub</a>
  </nav>'''

def generate_week_page(week, prev_week=None, next_week=None):
    """Generate a single week page with sidebar layout"""
    topics_html = '\n          '.join([f'<div class="topic-item">{t}</div>' for t in week['topics']])

    # Navigation cards
    nav_cards = ''
    if prev_week:
        nav_cards += f'''
        <a href="{prev_week['id']}.html" class="nav-card">
          <div class="label">Previous</div>
          <div class="title">{prev_week['title']}</div>
        </a>'''
    if next_week:
        nav_cards += f'''
        <a href="{next_week['id']}.html" class="nav-card">
          <div class="label">Next</div>
          <div class="title">{next_week['title']}</div>
        </a>'''

    # Resource links - point to GitHub folder
    slides_url = f"{GITHUB_REPO}/tree/main/{week['pdf_path']}"
    lab_url = f"{GITHUB_REPO}/tree/main/{week['lab_path']}"

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{week['title']} | NLP Course</title>
  <style>
{FULL_STYLES}
  </style>
</head>
<body>
{generate_top_nav('../')}

  <div class="container">
{generate_sidebar_html(week['id'], is_module=False, relative_path='../')}

    <main class="main-content">
      <section class="hero">
        <div class="badge">{week['slides']} SLIDES</div>
        <h1>{week['title']}</h1>
        <p class="subtitle">{week['subtitle']}</p>
      </section>

      <div class="content-area">
        <section class="section">
          <h2>Overview</h2>
          <p>{week['description']}</p>
        </section>

        <section class="section">
          <h2>Key Topics</h2>
          <div class="topics-grid">
          {topics_html}
          </div>
        </section>

        <section class="section">
          <h2>Resources</h2>
          <a href="{slides_url}" class="resource-btn primary" target="_blank">View Slides</a>
          <a href="{lab_url}" class="resource-btn secondary" target="_blank">View Lab Notebook</a>
          <a href="../index.html#gallery" class="resource-btn secondary">Chart Gallery</a>
        </section>

        <div class="nav-cards">{nav_cards}
        </div>
      </div>

      <footer class="footer">
        <p>Part {week['part']}: {PART_NAMES[week['part']]} | <a href="https://github.com/Digital-AI-Finance">Digital-AI-Finance</a></p>
      </footer>
    </main>
  </div>
{generate_search_script()}
</body>
</html>
'''

def generate_module_page(module):
    """Generate a module page with sidebar layout"""
    topics_html = '\n          '.join([f'<div class="topic-item">{t}</div>' for t in module['topics']])

    # Resource links - point to GitHub folder
    slides_url = f"{GITHUB_REPO}/tree/main/{module['pdf_path']}"
    lab_url = f"{GITHUB_REPO}/tree/main/{module['lab_path']}"

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{module['title']} | NLP Course</title>
  <style>
{FULL_STYLES}
  </style>
</head>
<body>
{generate_top_nav('../')}

  <div class="container">
{generate_sidebar_html(module['id'], is_module=True, relative_path='../')}

    <main class="main-content">
      <section class="hero">
        <div class="badge">{module['slides']} SLIDES</div>
        <h1>{module['title']}</h1>
      </section>

      <div class="content-area">
        <section class="section">
          <h2>Overview</h2>
          <p>{module['description']}</p>
        </section>

        <section class="section">
          <h2>Key Topics</h2>
          <div class="topics-grid">
          {topics_html}
          </div>
        </section>

        <section class="section">
          <h2>Resources</h2>
          <a href="{slides_url}" class="resource-btn primary" target="_blank">View Slides</a>
          <a href="{lab_url}" class="resource-btn secondary" target="_blank">View Lab Notebook</a>
          <a href="../index.html#gallery" class="resource-btn secondary">Chart Gallery</a>
        </section>
      </div>

      <footer class="footer">
        <p>Special Module | <a href="https://github.com/Digital-AI-Finance">Digital-AI-Finance</a></p>
      </footer>
    </main>
  </div>
{generate_search_script()}
</body>
</html>
'''

def generate_index_page():
    """Generate the main index page with content-based naming"""

    # Generate sidebar links
    sidebar_parts = {1: [], 2: [], 3: [], 4: []}
    for week in WEEKS:
        sidebar_parts[week['part']].append(
            f'<li><a href="weeks/{week["id"]}.html">{week["title"]}</a></li>'
        )

    sidebar_html = ''
    for part_num, part_name in PART_NAMES.items():
        links = '\n            '.join(sidebar_parts[part_num])
        open_attr = ' open' if part_num == 1 else ''
        sidebar_html += f'''
        <details class="part-section"{open_attr}>
          <summary>Part {part_num}: {part_name} <span class="topic-count">(3)</span></summary>
          <ul>
            {links}
          </ul>
        </details>'''

    # Generate module links
    module_links = '\n            '.join([
        f'<li><a href="modules/{m["id"]}.html">{m["title"]}</a></li>' for m in MODULES
    ])
    sidebar_html += f'''
        <details class="part-section">
          <summary>Special Modules <span class="topic-count">({len(MODULES)})</span></summary>
          <ul>
            {module_links}
          </ul>
        </details>'''

    # Generate topic cards by part
    topic_sections = ''
    for part_num, part_name in PART_NAMES.items():
        part_weeks = [w for w in WEEKS if w['part'] == part_num]
        cards = ''
        for week in part_weeks:
            cards += f'''
            <a href="weeks/{week['id']}.html" class="topic-card">
              <img src="assets/images/week{week['num']}.png" alt="{week['title']}" class="topic-thumb">
              <div class="topic-info">
                <span class="topic-num">PART {part_num}</span>
                <span class="topic-title">{week['title']}</span>
              </div>
            </a>'''

        topic_sections += f'''
        <!-- Part {part_num} -->
        <div class="part-section-main" id="part{part_num}">
          <div class="part-header"><div class="part-number">{part_num}</div><h3 class="part-title">{part_name}</h3></div>
          <div class="topic-grid">{cards}
          </div>
        </div>
'''

    # Generate module cards
    module_cards = ''
    for module in MODULES:
        module_cards += f'''
          <a href="modules/{module['id']}.html" class="module-card">
            <h3>{module['title']}</h3>
            <p>{module['slides']} slides - {module['topics'][0]}</p>
          </a>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Natural Language Processing | From N-grams to Transformers</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.5;
    }}
    .top-nav {{
      background: #1e3a5f;
      padding: 10px 20px;
      display: flex;
      justify-content: center;
      gap: 25px;
      flex-wrap: wrap;
    }}
    .top-nav a {{ color: white; text-decoration: none; font-size: 0.9rem; font-weight: 500; opacity: 0.9; }}
    .top-nav a:hover {{ opacity: 1; text-decoration: underline; }}
    .container {{ display: flex; max-width: 1600px; margin: 0 auto; }}
    .sidebar {{
      width: 280px;
      background: #ffffff;
      border-right: 1px solid #e1e4e8;
      height: calc(100vh - 42px);
      overflow-y: auto;
      position: sticky;
      top: 0;
      flex-shrink: 0;
    }}
    .sidebar-header {{
      padding: 15px;
      border-bottom: 1px solid #e1e4e8;
      background: linear-gradient(135deg, #1e3a5f 0%, #3333B2 100%);
    }}
    .sidebar-header a {{ display: flex; align-items: center; text-decoration: none; color: white; }}
    .sidebar-logo {{ width: 36px; height: 36px; border-radius: 6px; margin-right: 10px; background: white; padding: 3px; }}
    .course-title {{ font-size: 15px; font-weight: 700; }}
    .search-container {{ padding: 10px 15px; border-bottom: 1px solid #e1e4e8; }}
    .search-container input {{
      width: 100%; padding: 6px 10px; border: 1px solid #e1e4e8;
      border-radius: 4px; font-size: 12px; outline: none;
    }}
    .search-container input:focus {{ border-color: #3333B2; box-shadow: 0 0 0 2px rgba(51, 51, 178, 0.1); }}
    .sidebar-nav {{ padding: 8px 0; }}
    .part-section {{ border-bottom: 1px solid #f0f0f0; }}
    .part-section summary {{
      padding: 10px 15px; font-weight: 600; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.5px; color: #24292e;
      cursor: pointer; list-style: none;
      display: flex; align-items: center; justify-content: space-between;
    }}
    .part-section summary::-webkit-details-marker {{ display: none; }}
    .part-section summary::after {{ content: "+"; font-size: 12px; color: #586069; }}
    .part-section[open] summary::after {{ content: "-"; }}
    .part-section summary:hover {{ background: #f6f8fa; }}
    .part-section ul {{ list-style: none; padding: 0 0 6px 0; margin: 0; }}
    .part-section li {{ margin: 0; }}
    .part-section a {{
      display: block; padding: 6px 15px 6px 25px; color: #586069;
      text-decoration: none; font-size: 12px; border-left: 2px solid transparent;
    }}
    .part-section a:hover {{ background: #f6f8fa; color: #3333B2; }}
    .topic-count {{ font-size: 10px; color: #959da5; font-weight: normal; }}
    .hidden {{ display: none !important; }}
    .main-content {{ flex: 1; min-width: 0; }}
    .hero {{
      background: linear-gradient(135deg, #1e3a5f 0%, #3333B2 100%);
      color: white; padding: 25px 15px; text-align: center;
    }}
    .hero-content {{ max-width: 800px; margin: 0 auto; }}
    .hero-logo {{ width: 50px; height: 50px; border-radius: 8px; background: white; padding: 4px; margin-bottom: 10px; }}
    .hero h1 {{ font-size: 2rem; font-weight: 800; margin-bottom: 6px; }}
    .tagline {{ font-size: 1rem; opacity: 0.9; margin-bottom: 12px; }}
    .stats {{ display: flex; justify-content: center; gap: 25px; margin-bottom: 15px; }}
    .stat-item {{ text-align: center; }}
    .stat-number {{ font-size: 1.5rem; font-weight: 700; display: block; }}
    .stat-label {{ font-size: 0.75rem; opacity: 0.8; }}
    .cta-button {{
      display: inline-block; background: white; color: #1e3a5f;
      padding: 10px 24px; border-radius: 5px; font-size: 0.95rem;
      font-weight: 600; text-decoration: none;
    }}
    .cta-button:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
    .features {{ padding: 18px 12px; background: white; }}
    .features-container {{
      max-width: 1000px; margin: 0 auto;
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    }}
    .feature-card {{
      background: #f8fafc; border-radius: 6px; padding: 14px;
      text-align: center; border: 1px solid #e2e8f0;
    }}
    .feature-icon {{ font-size: 1.6rem; margin-bottom: 6px; }}
    .feature-card h3 {{ font-size: 0.9rem; color: #1e3a5f; margin-bottom: 4px; }}
    .feature-card p {{ color: #64748b; font-size: 0.75rem; }}
    .journey {{ padding: 18px 12px; background: #f1f5f9; }}
    .journey h2 {{ text-align: center; font-size: 1.2rem; color: #1e3a5f; margin-bottom: 15px; }}
    .journey-tracker {{
      max-width: 900px; margin: 0 auto;
      display: flex; justify-content: space-between; align-items: center;
      position: relative;
    }}
    .journey-tracker::before {{
      content: ''; position: absolute; top: 18px; left: 50px; right: 50px;
      height: 2px; background: #cbd5e1; z-index: 0;
    }}
    .journey-step {{
      display: flex; flex-direction: column; align-items: center;
      z-index: 1; text-decoration: none; color: inherit;
    }}
    .step-circle {{
      width: 36px; height: 36px; border-radius: 50%; background: white;
      border: 2px solid #3333B2;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 0.9rem; color: #3333B2; margin-bottom: 5px;
    }}
    .step-label {{ font-size: 0.7rem; color: #64748b; text-align: center; max-width: 80px; }}
    .topics-section {{ padding: 18px 12px; }}
    .part-section-main {{ margin-bottom: 20px; }}
    .part-header {{ display: flex; align-items: center; margin-bottom: 10px; }}
    .part-number {{
      background: #3333B2; color: white; width: 28px; height: 28px;
      border-radius: 5px; display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 0.85rem; margin-right: 10px;
    }}
    .part-title {{ font-size: 1.1rem; color: #1e3a5f; }}
    .topic-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
    .topic-card {{
      background: white; border-radius: 6px; overflow: hidden;
      border: 1px solid #e2e8f0; text-decoration: none; color: inherit;
      transition: box-shadow 0.2s;
    }}
    .topic-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
    .topic-thumb {{ width: 100%; height: 80px; object-fit: cover; background: #f1f5f9; }}
    .topic-info {{ padding: 8px; }}
    .topic-num {{
      display: inline-block; background: #e0e7ff; color: #3333B2;
      font-size: 0.65rem; font-weight: 600; padding: 2px 6px;
      border-radius: 3px; margin-bottom: 3px;
    }}
    .topic-title {{ display: block; font-weight: 600; color: #1e293b; font-size: 0.8rem; }}
    .chart-gallery {{ padding: 18px 12px; background: white; }}
    .chart-gallery h2 {{ text-align: center; font-size: 1.2rem; color: #1e3a5f; margin-bottom: 15px; }}
    .gallery-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }}
    .gallery-item {{ border-radius: 4px; overflow: hidden; border: 1px solid #e2e8f0; }}
    .gallery-item img {{ width: 100%; height: 70px; object-fit: cover; display: block; background: #f1f5f9; }}
    .gallery-item span {{
      display: block; text-align: center; font-size: 0.6rem; padding: 4px;
      background: #f8fafc; color: #64748b; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis;
    }}
    .special-modules {{ padding: 18px 12px; background: #f1f5f9; }}
    .special-modules h2 {{ text-align: center; font-size: 1.2rem; color: #1e3a5f; margin-bottom: 15px; }}
    .module-grid {{
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
      max-width: 1000px; margin: 0 auto;
    }}
    .module-card {{
      background: white; border-radius: 6px; padding: 12px; text-align: center;
      border: 1px solid #e2e8f0; text-decoration: none; color: inherit;
    }}
    .module-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
    .module-card h3 {{ font-size: 0.85rem; color: #1e3a5f; margin-bottom: 4px; }}
    .module-card p {{ font-size: 0.7rem; color: #64748b; }}
    .resources {{ padding: 18px 12px; background: white; text-align: center; }}
    .resources h2 {{ font-size: 1.2rem; color: #1e3a5f; margin-bottom: 12px; }}
    .resource-buttons {{ display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }}
    .resource-btn {{
      display: inline-flex; align-items: center; gap: 5px;
      padding: 8px 14px; border-radius: 5px; text-decoration: none;
      font-weight: 500; font-size: 0.85rem;
    }}
    .resource-btn.primary {{ background: #3333B2; color: white; }}
    .resource-btn.secondary {{ background: white; color: #1e3a5f; border: 1px solid #e2e8f0; }}
    .footer {{ padding: 18px 12px; background: #1e3a5f; color: white; text-align: center; }}
    .footer a {{ color: #93c5fd; }}
    .footer p {{ opacity: 0.8; font-size: 0.8rem; }}
    @media (max-width: 1200px) {{
      .topic-grid {{ grid-template-columns: repeat(3, 1fr); }}
      .gallery-grid {{ grid-template-columns: repeat(5, 1fr); }}
      .features-container {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 900px) {{
      .container {{ flex-direction: column; }}
      .sidebar {{ width: 100%; height: auto; position: relative; border-right: none; border-bottom: 1px solid #e1e4e8; max-height: 40vh; }}
      .topic-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .gallery-grid {{ grid-template-columns: repeat(4, 1fr); }}
      .module-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 600px) {{
      .topic-grid {{ grid-template-columns: 1fr; }}
      .gallery-grid {{ grid-template-columns: repeat(3, 1fr); }}
      .module-grid {{ grid-template-columns: 1fr; }}
      .journey-tracker {{ flex-wrap: wrap; gap: 10px; justify-content: center; }}
      .journey-tracker::before {{ display: none; }}
      .features-container {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <nav class="top-nav">
    <a href="index.html">Home</a>
    <a href="#topics">Topics</a>
    <a href="#gallery">Charts</a>
    <a href="#modules">Modules</a>
    <a href="{GITHUB_REPO}">GitHub</a>
  </nav>

  <div class="container">
    <aside class="sidebar">
      <div class="sidebar-header">
        <a href="index.html">
          <img src="https://quantlet.com/images/Q.png" alt="QuantLet" class="sidebar-logo">
          <span class="course-title">NLP Course</span>
        </a>
      </div>
      <div class="search-container">
        <input type="text" id="topic-search" placeholder="Search topics...">
      </div>
      <nav class="sidebar-nav">{sidebar_html}
      </nav>
    </aside>

    <main class="main-content">
      <section class="hero">
        <div class="hero-content">
          <img src="https://quantlet.com/images/Q.png" alt="QuantLet" class="hero-logo">
          <h1>Natural Language Processing</h1>
          <p class="tagline">From N-grams to Transformers: A Complete Graduate Course</p>
          <div class="stats">
            <div class="stat-item"><span class="stat-number">12</span><span class="stat-label">Topics</span></div>
            <div class="stat-item"><span class="stat-number">500+</span><span class="stat-label">Slides</span></div>
            <div class="stat-item"><span class="stat-number">250+</span><span class="stat-label">Charts</span></div>
            <div class="stat-item"><span class="stat-number">26+</span><span class="stat-label">Notebooks</span></div>
          </div>
          <a href="weeks/ngrams.html" class="cta-button">Start Learning</a>
        </div>
      </section>

      <section class="features">
        <div class="features-container">
          <div class="feature-card">
            <div class="feature-icon">&#128218;</div>
            <h3>Discovery-Based</h3>
            <p>Problem-first pedagogy with worked examples</p>
          </div>
          <div class="feature-card">
            <div class="feature-icon">&#128202;</div>
            <h3>Rich Visualizations</h3>
            <p>250+ Python-generated charts</p>
          </div>
          <div class="feature-card">
            <div class="feature-icon">&#128187;</div>
            <h3>Hands-On Labs</h3>
            <p>Jupyter notebooks with exercises</p>
          </div>
          <div class="feature-card">
            <div class="feature-icon">&#128736;</div>
            <h3>Production Ready</h3>
            <p>From theory to deployment</p>
          </div>
        </div>
      </section>

      <section class="journey">
        <h2>Learning Journey</h2>
        <div class="journey-tracker">
          <a href="#part1" class="journey-step"><div class="step-circle">1</div><span class="step-label">Language Foundations</span></a>
          <a href="#part2" class="journey-step"><div class="step-circle">2</div><span class="step-label">Core Architectures</span></a>
          <a href="#part3" class="journey-step"><div class="step-circle">3</div><span class="step-label">Advanced Methods</span></a>
          <a href="#part4" class="journey-step"><div class="step-circle">4</div><span class="step-label">Applications</span></a>
        </div>
      </section>

      <section class="topics-section" id="topics">{topic_sections}
      </section>

      <section class="chart-gallery" id="gallery">
        <h2>Chart Gallery (250+ Visualizations)</h2>
        <div class="gallery-grid">
          <div class="gallery-item"><img src="assets/images/skipgram_architecture_bsc.png" alt="Skip-gram"><span>Skip-gram</span></div>
          <div class="gallery-item"><img src="assets/images/cbow_architecture_bsc.png" alt="CBOW"><span>CBOW</span></div>
          <div class="gallery-item"><img src="assets/images/3d_transformer_architecture.png" alt="Transformer"><span>Transformer</span></div>
          <div class="gallery-item"><img src="assets/images/beam_search_tree_graphviz.png" alt="Beam Search"><span>Beam Search</span></div>
          <div class="gallery-item"><img src="assets/images/temperature_effects_bsc.png" alt="Temperature"><span>Temperature</span></div>
          <div class="gallery-item"><img src="assets/images/word_arithmetic_3d_bsc.png" alt="Word Arithmetic"><span>Word Arithmetic</span></div>
          <div class="gallery-item"><img src="assets/images/training_evolution_bsc.png" alt="Training"><span>Training</span></div>
          <div class="gallery-item"><img src="assets/images/negative_sampling_process_bsc.png" alt="Neg. Sampling"><span>Neg. Sampling</span></div>
          <div class="gallery-item"><img src="assets/images/contrastive_vs_nucleus_bsc.png" alt="Contrastive"><span>Contrastive</span></div>
          <div class="gallery-item"><img src="assets/images/degeneration_problem_bsc.png" alt="Degeneration"><span>Degeneration</span></div>
          <div class="gallery-item"><img src="assets/images/topk_example_bsc.png" alt="Top-K"><span>Top-K</span></div>
          <div class="gallery-item"><img src="assets/images/vocabulary_probability_bsc.png" alt="Vocab Dist"><span>Vocab Dist</span></div>
        </div>
      </section>

      <section class="special-modules" id="modules">
        <h2>Special Modules</h2>
        <div class="module-grid">{module_cards}
        </div>
      </section>

      <section class="resources">
        <h2>Resources</h2>
        <div class="resource-buttons">
          <a href="{GITHUB_REPO}" class="resource-btn primary">GitHub Repo</a>
          <a href="modules/embeddings.html" class="resource-btn secondary">Embeddings Module</a>
          <a href="#gallery" class="resource-btn secondary">Full Chart Gallery</a>
        </div>
      </section>

      <footer class="footer">
        <p><a href="https://github.com/Digital-AI-Finance">Digital-AI-Finance</a> | Graduate NLP Course | MIT License</p>
      </footer>
    </main>
  </div>

  <script>
    document.getElementById('topic-search').addEventListener('input', function(e) {{
      const query = e.target.value.toLowerCase();
      const links = document.querySelectorAll('.sidebar-nav .part-section li');
      const sections = document.querySelectorAll('.sidebar-nav .part-section');
      links.forEach(li => {{
        const text = li.textContent.toLowerCase();
        li.classList.toggle('hidden', query !== '' && !text.includes(query));
      }});
      sections.forEach(section => {{
        const visibleLinks = section.querySelectorAll('li:not(.hidden)');
        if (visibleLinks.length === 0 && query !== '') {{
          section.classList.add('hidden');
        }} else {{
          section.classList.remove('hidden');
          if (query !== '') section.setAttribute('open', '');
        }}
      }});
    }});
  </script>
</body>
</html>
'''

def main():
    """Generate all site pages"""
    docs_dir = Path(__file__).parent
    weeks_dir = docs_dir / 'weeks'
    modules_dir = docs_dir / 'modules'

    # Create directories
    weeks_dir.mkdir(exist_ok=True)
    modules_dir.mkdir(exist_ok=True)

    # Generate index page
    print("Generating index.html...")
    with open(docs_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(generate_index_page())

    # Generate week pages with navigation
    print("Generating week pages with sidebar...")
    for i, week in enumerate(WEEKS):
        prev_week = WEEKS[i - 1] if i > 0 else None
        next_week = WEEKS[i + 1] if i < len(WEEKS) - 1 else None

        # Remove old week files
        old_file = weeks_dir / f'week{week["num"]}.html'
        if old_file.exists():
            old_file.unlink()

        # Write new content-based file
        filepath = weeks_dir / f'{week["id"]}.html'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(generate_week_page(week, prev_week, next_week))
        print(f"  - {filepath.name}")

    # Generate module pages
    print("Generating module pages with sidebar...")
    for module in MODULES:
        filepath = modules_dir / f'{module["id"]}.html'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(generate_module_page(module))
        print(f"  - {filepath.name}")

    print("\nSite generation complete!")
    print(f"  - 1 index page")
    print(f"  - {len(WEEKS)} topic pages (with sidebar)")
    print(f"  - {len(MODULES)} module pages (with sidebar)")

if __name__ == '__main__':
    main()
