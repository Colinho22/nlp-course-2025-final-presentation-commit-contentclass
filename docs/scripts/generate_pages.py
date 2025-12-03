"""Generate all week and module pages for NLP course site."""
import os
from pathlib import Path

# Week data
WEEKS = [
    {"num": "01", "title": "N-gram Foundations", "subtitle": "Statistical Language Models",
     "desc": "Introduction to language modeling with n-grams, probability estimation, and smoothing techniques.",
     "topics": ["Markov assumption", "N-gram probabilities", "Laplace smoothing", "Perplexity evaluation"]},
    {"num": "02", "title": "Word Embeddings", "subtitle": "From Words to Vectors",
     "desc": "Dense vector representations of words using Word2Vec, GloVe, and FastText.",
     "topics": ["One-hot encoding limitations", "Skip-gram & CBOW", "Negative sampling", "Word analogies"]},
    {"num": "03", "title": "RNN & LSTM", "subtitle": "Sequential Neural Networks",
     "desc": "Recurrent neural networks and Long Short-Term Memory for sequence modeling.",
     "topics": ["Vanishing gradients", "LSTM gates", "Bidirectional RNNs", "Sequence tagging"]},
    {"num": "04", "title": "Seq2Seq Models", "subtitle": "Encoder-Decoder Architecture",
     "desc": "Sequence-to-sequence models for machine translation and text generation.",
     "topics": ["Encoder-decoder", "Attention mechanism", "Teacher forcing", "Beam search decoding"]},
    {"num": "05", "title": "Transformers", "subtitle": "Attention Is All You Need",
     "desc": "The transformer architecture that revolutionized NLP.",
     "topics": ["Self-attention", "Multi-head attention", "Positional encoding", "Layer normalization"]},
    {"num": "06", "title": "Pre-trained Models", "subtitle": "BERT, GPT, and Beyond",
     "desc": "Pre-training and fine-tuning paradigm for NLP tasks.",
     "topics": ["BERT architecture", "Masked language modeling", "GPT autoregressive", "Transfer learning"]},
    {"num": "07", "title": "Advanced Topics", "subtitle": "Scaling Laws and Emergent Abilities",
     "desc": "Advanced concepts in large language models and their capabilities.",
     "topics": ["Scaling laws", "Emergent abilities", "In-context learning", "Chain-of-thought"]},
    {"num": "08", "title": "Tokenization", "subtitle": "BPE and Subword Methods",
     "desc": "Tokenization strategies for handling vocabulary and unknown words.",
     "topics": ["Byte-Pair Encoding", "WordPiece", "SentencePiece", "Vocabulary optimization"]},
    {"num": "09", "title": "Decoding Strategies", "subtitle": "From Greedy to Nucleus Sampling",
     "desc": "Methods for generating text from language model predictions.",
     "topics": ["Greedy decoding", "Beam search", "Temperature sampling", "Top-k and nucleus sampling"]},
    {"num": "10", "title": "Fine-tuning", "subtitle": "Adapting Pre-trained Models",
     "desc": "Efficient methods for adapting large models to specific tasks.",
     "topics": ["Full fine-tuning", "LoRA adapters", "Prompt tuning", "Instruction tuning"]},
    {"num": "11", "title": "Efficiency", "subtitle": "Quantization and Optimization",
     "desc": "Techniques for efficient inference and deployment of language models.",
     "topics": ["Quantization", "Pruning", "Knowledge distillation", "Flash attention"]},
    {"num": "12", "title": "Ethics & Bias", "subtitle": "Responsible AI Development",
     "desc": "Ethical considerations and bias mitigation in NLP systems.",
     "topics": ["Bias detection", "Fairness metrics", "Toxicity filtering", "Responsible deployment"]},
]

# Module data
MODULES = [
    {"id": "embeddings", "title": "Word Embeddings Deep Dive", "slides": 48,
     "desc": "Comprehensive coverage of word embedding algorithms with interactive 3D visualizations.",
     "topics": ["Skip-gram mathematics", "GloVe co-occurrence", "FastText subwords", "Contextual embeddings"]},
    {"id": "summarization", "title": "LLM Summarization", "slides": 40,
     "desc": "Text summarization using extractive and abstractive methods with modern LLMs.",
     "topics": ["Extractive methods", "Abstractive generation", "RAG enhancement", "Evaluation metrics"]},
    {"id": "sentiment", "title": "Sentiment Analysis", "slides": 26,
     "desc": "BERT fine-tuning for sentiment classification with technical deep dives.",
     "topics": ["BERT classifier head", "Pre-training objectives", "Fine-tuning process", "Deployment pipeline"]},
    {"id": "lstm-primer", "title": "LSTM Primer", "slides": 32,
     "desc": "Zero-prerequisite introduction to LSTM networks with clear visualizations.",
     "topics": ["Why RNNs fail", "Gate mechanisms", "Cell state flow", "Practical applications"]},
]

def generate_week_page(week, output_dir):
    """Generate a single week page."""
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Week {week["num"]}: {week["title"]} | NLP Course</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.6;
    }}
    .nav {{ background: #1e3a5f; padding: 12px 20px; display: flex; justify-content: center; gap: 25px; }}
    .nav a {{ color: white; text-decoration: none; font-size: 0.9rem; opacity: 0.9; }}
    .nav a:hover {{ opacity: 1; text-decoration: underline; }}
    .hero {{
      background: linear-gradient(135deg, #1e3a5f 0%, #3333B2 100%);
      color: white; padding: 40px 20px; text-align: center;
    }}
    .hero h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
    .hero .subtitle {{ font-size: 1.1rem; opacity: 0.9; margin-bottom: 15px; }}
    .hero .week-badge {{
      display: inline-block; background: rgba(255,255,255,0.2);
      padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; margin-bottom: 15px;
    }}
    .container {{ max-width: 1000px; margin: 0 auto; padding: 30px 20px; }}
    .section {{ background: white; border-radius: 8px; padding: 25px; margin-bottom: 20px; border: 1px solid #e2e8f0; }}
    .section h2 {{ color: #1e3a5f; font-size: 1.3rem; margin-bottom: 15px; border-bottom: 2px solid #3333B2; padding-bottom: 8px; }}
    .topics-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .topic-item {{
      background: #f8fafc; padding: 12px 15px; border-radius: 6px;
      border-left: 3px solid #3333B2; font-size: 0.95rem;
    }}
    .resource-btn {{
      display: inline-block; padding: 10px 20px; border-radius: 5px;
      text-decoration: none; font-weight: 500; margin: 5px;
    }}
    .resource-btn.primary {{ background: #3333B2; color: white; }}
    .resource-btn.secondary {{ background: white; color: #1e3a5f; border: 1px solid #e2e8f0; }}
    .nav-links {{ display: flex; justify-content: space-between; margin-top: 30px; }}
    .nav-links a {{
      padding: 10px 20px; background: white; border-radius: 5px;
      text-decoration: none; color: #3333B2; border: 1px solid #e2e8f0;
    }}
    .footer {{ text-align: center; padding: 20px; color: #64748b; font-size: 0.85rem; }}
    @media (max-width: 600px) {{
      .topics-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <nav class="nav">
    <a href="../">Home</a>
    <a href="../#topics">All Weeks</a>
    <a href="../#gallery">Charts</a>
    <a href="https://github.com/Digital-AI-Finance/Natural-Language-Processing">GitHub</a>
  </nav>

  <section class="hero">
    <div class="week-badge">WEEK {week["num"]}</div>
    <h1>{week["title"]}</h1>
    <p class="subtitle">{week["subtitle"]}</p>
  </section>

  <div class="container">
    <section class="section">
      <h2>Overview</h2>
      <p>{week["desc"]}</p>
    </section>

    <section class="section">
      <h2>Key Topics</h2>
      <div class="topics-grid">
        {"".join(f'<div class="topic-item">{topic}</div>' for topic in week["topics"])}
      </div>
    </section>

    <section class="section">
      <h2>Resources</h2>
      <a href="#" class="resource-btn primary">Download Slides (PDF)</a>
      <a href="#" class="resource-btn secondary">View Lab Notebook</a>
      <a href="#" class="resource-btn secondary">Chart Gallery</a>
    </section>

    <nav class="nav-links">
      {"<a href='week" + str(int(week["num"])-1).zfill(2) + ".html'>Previous Week</a>" if int(week["num"]) > 1 else "<span></span>"}
      {"<a href='week" + str(int(week["num"])+1).zfill(2) + ".html'>Next Week</a>" if int(week["num"]) < 12 else "<span></span>"}
    </nav>
  </div>

  <footer class="footer">
    <p>Digital-AI-Finance | Natural Language Processing Course</p>
  </footer>
</body>
</html>'''

    filepath = output_dir / f'week{week["num"]}.html'
    filepath.write_text(html, encoding='utf-8')
    print(f"Generated: {filepath}")

def generate_module_page(module, output_dir):
    """Generate a single module page."""
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{module["title"]} | NLP Course</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.6;
    }}
    .nav {{ background: #1e3a5f; padding: 12px 20px; display: flex; justify-content: center; gap: 25px; }}
    .nav a {{ color: white; text-decoration: none; font-size: 0.9rem; opacity: 0.9; }}
    .nav a:hover {{ opacity: 1; text-decoration: underline; }}
    .hero {{
      background: linear-gradient(135deg, #1e3a5f 0%, #3333B2 100%);
      color: white; padding: 40px 20px; text-align: center;
    }}
    .hero h1 {{ font-size: 2.2rem; margin-bottom: 8px; }}
    .hero .slides-badge {{
      display: inline-block; background: rgba(255,255,255,0.2);
      padding: 5px 15px; border-radius: 20px; font-size: 0.85rem; margin-bottom: 15px;
    }}
    .container {{ max-width: 1000px; margin: 0 auto; padding: 30px 20px; }}
    .section {{ background: white; border-radius: 8px; padding: 25px; margin-bottom: 20px; border: 1px solid #e2e8f0; }}
    .section h2 {{ color: #1e3a5f; font-size: 1.3rem; margin-bottom: 15px; border-bottom: 2px solid #3333B2; padding-bottom: 8px; }}
    .topics-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
    .topic-item {{
      background: #f8fafc; padding: 12px 15px; border-radius: 6px;
      border-left: 3px solid #3333B2; font-size: 0.95rem;
    }}
    .resource-btn {{
      display: inline-block; padding: 10px 20px; border-radius: 5px;
      text-decoration: none; font-weight: 500; margin: 5px;
    }}
    .resource-btn.primary {{ background: #3333B2; color: white; }}
    .resource-btn.secondary {{ background: white; color: #1e3a5f; border: 1px solid #e2e8f0; }}
    .footer {{ text-align: center; padding: 20px; color: #64748b; font-size: 0.85rem; }}
    @media (max-width: 600px) {{
      .topics-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <nav class="nav">
    <a href="../">Home</a>
    <a href="../#modules">All Modules</a>
    <a href="../#gallery">Charts</a>
    <a href="https://github.com/Digital-AI-Finance/Natural-Language-Processing">GitHub</a>
  </nav>

  <section class="hero">
    <div class="slides-badge">{module["slides"]} SLIDES</div>
    <h1>{module["title"]}</h1>
  </section>

  <div class="container">
    <section class="section">
      <h2>Overview</h2>
      <p>{module["desc"]}</p>
    </section>

    <section class="section">
      <h2>Key Topics</h2>
      <div class="topics-grid">
        {"".join(f'<div class="topic-item">{topic}</div>' for topic in module["topics"])}
      </div>
    </section>

    <section class="section">
      <h2>Resources</h2>
      <a href="#" class="resource-btn primary">Download Slides (PDF)</a>
      <a href="#" class="resource-btn secondary">View Lab Notebook</a>
      <a href="#" class="resource-btn secondary">Chart Gallery</a>
    </section>
  </div>

  <footer class="footer">
    <p>Digital-AI-Finance | Natural Language Processing Course</p>
  </footer>
</body>
</html>'''

    filepath = output_dir / f'{module["id"]}.html'
    filepath.write_text(html, encoding='utf-8')
    print(f"Generated: {filepath}")

def main():
    docs_dir = Path(__file__).parent
    weeks_dir = docs_dir / 'weeks'
    modules_dir = docs_dir / 'modules'

    weeks_dir.mkdir(exist_ok=True)
    modules_dir.mkdir(exist_ok=True)

    print("Generating week pages...")
    for week in WEEKS:
        generate_week_page(week, weeks_dir)

    print("\nGenerating module pages...")
    for module in MODULES:
        generate_module_page(module, modules_dir)

    print("\nDone! Generated all pages.")

if __name__ == "__main__":
    main()
