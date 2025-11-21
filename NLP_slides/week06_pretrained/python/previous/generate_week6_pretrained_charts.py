import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrow, ConnectionPatch
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# 1. Pre-training Workflow
def plot_pretraining_workflow():
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Stage 1: Data Collection
    data_box = FancyBboxPatch((0.05, 0.8), 0.18, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='lightblue',
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(data_box)
    ax.text(0.14, 0.875, 'Data Collection\n\n• Web crawl\n• Books, Wikipedia\n• 40GB+ text\n• 3.3B tokens', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow to preprocessing
    ax.arrow(0.23, 0.875, 0.05, 0, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=2)
    
    # Stage 2: Preprocessing
    prep_box = FancyBboxPatch((0.3, 0.8), 0.18, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='lightgreen',
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(prep_box)
    ax.text(0.39, 0.875, 'Preprocessing\n\n• Tokenization\n• WordPiece\n• Special tokens\n• Sentence pairs', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow to training objectives
    ax.arrow(0.48, 0.875, 0.05, 0, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=2)
    
    # Stage 3: Training Objectives
    obj_box = FancyBboxPatch((0.55, 0.8), 0.18, 0.15,
                            boxstyle="round,pad=0.02",
                            facecolor='lightyellow',
                            edgecolor='black',
                            linewidth=2)
    ax.add_patch(obj_box)
    ax.text(0.64, 0.875, 'Training Objectives\n\nMLM: 15% masking\nNSP: Sentence pairs\nCombined loss\nBidirectional', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow to model architecture
    ax.arrow(0.73, 0.875, 0.05, 0, head_width=0.02, head_length=0.02, 
             fc='black', ec='black', linewidth=2)
    
    # Stage 4: Model Architecture
    arch_box = FancyBboxPatch((0.8, 0.8), 0.18, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='lightcoral',
                             edgecolor='black',
                             linewidth=2)
    ax.add_patch(arch_box)
    ax.text(0.89, 0.875, 'Architecture\n\n12/24 layers\n768/1024 hidden\n110M/340M params\nTransformer', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Training Infrastructure (below)
    infra_box = FancyBboxPatch((0.1, 0.5), 0.8, 0.2,
                              boxstyle="round,pad=0.02",
                              facecolor='lavender',
                              edgecolor='black',
                              linewidth=3)
    ax.add_patch(infra_box)
    ax.text(0.5, 0.6, 'Training Infrastructure', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Infrastructure details in columns
    infra_details = [
        ('Hardware:\n• 64 TPU v3 cores\n• 4 days training\n• $7000+ cost', 0.2),
        ('Optimization:\n• Adam optimizer\n• Learning rate 1e-4\n• Warmup schedule', 0.4),
        ('Scale:\n• Batch size 256\n• 1M training steps\n• 40 epochs', 0.6),
        ('Checkpointing:\n• Save every 1000 steps\n• Early stopping\n• Model validation', 0.8)
    ]
    
    for detail, x in infra_details:
        ax.text(x, 0.55, detail, ha='center', va='center', fontsize=10)
    
    # Arrow down to results
    ax.arrow(0.5, 0.48, 0, -0.08, head_width=0.03, head_length=0.02, 
             fc='purple', ec='purple', linewidth=3)
    
    # Results section
    results_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.25,
                                boxstyle="round,pad=0.02",
                                facecolor='lightcyan',
                                edgecolor='black',
                                linewidth=3)
    ax.add_patch(results_box)
    
    ax.text(0.5, 0.3, 'Pre-trained Model Capabilities', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Capability boxes
    capabilities = [
        ('Language\nUnderstanding', 'Grammar, syntax,\nsemantics learned', 0.15, 'lightgreen'),
        ('World\nKnowledge', 'Facts, relations,\ncommon sense', 0.35, 'lightblue'), 
        ('Transfer\nReady', 'Fine-tune for\nany NLP task', 0.55, 'lightyellow'),
        ('Few-shot\nLearning', 'Generalize from\nfew examples', 0.75, 'lightcoral'),
        ('Contextual\nEmbeddings', 'Dynamic word\nrepresentations', 0.95, 'lavender')
    ]
    
    for title, desc, x, color in capabilities:
        cap_box = FancyBboxPatch((x-0.08, 0.15), 0.16, 0.1,
                                boxstyle="round,pad=0.01",
                                facecolor=color,
                                edgecolor='black',
                                linewidth=2)
        ax.add_patch(cap_box)
        ax.text(x, 0.22, title, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, 0.18, desc, ha='center', va='center', fontsize=8)
    
    # Timeline at bottom
    ax.text(0.5, 0.05, 'Timeline: Months of preparation → 4 days training → Decade of applications', 
            ha='center', va='center', fontsize=12, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='gold'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.text(0.5, 0.97, 'Pre-training Workflow: From Raw Text to Foundation Models', 
            ha='center', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/pretraining_workflow.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. BERT vs GPT Architecture Comparison (moved up from #5)

# 3. BERT Results on GLUE
def plot_bert_results_glue():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # GLUE tasks and scores
    tasks = ['CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI']
    
    # Previous SOTA and BERT scores
    previous_sota = [45.4, 94.9, 88.5, 85.9, 89.5, 84.6, 91.3, 71.1, 65.1]
    bert_base = [52.1, 93.5, 88.9, 87.1, 71.2, 84.6, 90.5, 66.4, 65.1]
    bert_large = [60.5, 94.9, 89.3, 87.6, 72.1, 86.7, 92.7, 70.1, 65.1]
    
    x = np.arange(len(tasks))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, previous_sota, width, label='Previous SOTA', color='gray')
    bars2 = ax.bar(x, bert_base, width, label='BERT-Base', color='#3498db')
    bars3 = ax.bar(x + width, bert_large, width, label='BERT-Large', color='#2ecc71')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Labels and title
    ax.set_xlabel('GLUE Tasks', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('BERT Dominates GLUE Benchmark\nGeneral Language Understanding Evaluation', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add average scores
    avg_prev = np.mean(previous_sota)
    avg_base = np.mean(bert_base)
    avg_large = np.mean(bert_large)
    
    ax.text(0.7, 0.85, f'Average Scores:\nPrevious SOTA: {avg_prev:.1f}\nBERT-Base: {avg_base:.1f}\nBERT-Large: {avg_large:.1f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('../figures/bert_results_glue.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bert_vs_gpt_architecture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # BERT (Bidirectional)
    ax1.set_title('BERT: Bidirectional Encoder', fontsize=14, fontweight='bold')
    
    # Input tokens
    tokens = ['[CLS]', 'The', 'cat', '[MASK]', 'on', 'mat', '[SEP]']
    positions = np.linspace(0.1, 0.9, len(tokens))
    
    for i, (token, pos) in enumerate(zip(tokens, positions)):
        color = 'gold' if token == '[MASK]' else 'lightblue'
        if token in ['[CLS]', '[SEP]']:
            color = 'lightgray'
        
        box = FancyBboxPatch((pos-0.05, 0.1), 0.1, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax1.add_patch(box)
        ax1.text(pos, 0.14, token, ha='center', va='center', fontsize=10)
    
    # Bidirectional arrows
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if i != j:
                alpha = 0.1 if abs(i-j) > 2 else 0.3
                ax1.arrow(positions[i], 0.2, positions[j]-positions[i], 0.3,
                         head_width=0.02, head_length=0.02, 
                         fc='blue', ec='blue', alpha=alpha, length_includes_head=True)
    
    # Transformer block
    trans_box = FancyBboxPatch((0.2, 0.5), 0.6, 0.15,
                              boxstyle="round,pad=0.02",
                              facecolor='lightcyan',
                              edgecolor='black',
                              linewidth=3)
    ax1.add_patch(trans_box)
    ax1.text(0.5, 0.575, 'Transformer Encoder\n(Sees all tokens)', 
             ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax1.text(0.5, 0.8, 'Can see both left AND right context', 
             ha='center', fontsize=10, color='green',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # GPT (Unidirectional)
    ax2.set_title('GPT: Unidirectional Decoder', fontsize=14, fontweight='bold')
    
    # Input tokens
    tokens_gpt = ['The', 'cat', 'sat', 'on', 'the', '?']
    positions_gpt = np.linspace(0.1, 0.9, len(tokens_gpt))
    
    for i, (token, pos) in enumerate(zip(tokens_gpt, positions_gpt)):
        color = 'lightcoral' if token == '?' else 'lightblue'
        
        box = FancyBboxPatch((pos-0.05, 0.1), 0.1, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax2.add_patch(box)
        ax2.text(pos, 0.14, token, ha='center', va='center', fontsize=10)
    
    # Causal mask (can only look left)
    for i in range(len(tokens_gpt)):
        for j in range(i):
            alpha = 0.1 if i-j > 2 else 0.3
            ax2.arrow(positions_gpt[i], 0.2, positions_gpt[j]-positions_gpt[i], 0.3,
                     head_width=0.02, head_length=0.02, 
                     fc='red', ec='red', alpha=alpha, length_includes_head=True)
    
    # Transformer block
    trans_box2 = FancyBboxPatch((0.2, 0.5), 0.6, 0.15,
                               boxstyle="round,pad=0.02",
                               facecolor='lightcyan',
                               edgecolor='black',
                               linewidth=3)
    ax2.add_patch(trans_box2)
    ax2.text(0.5, 0.575, 'Transformer Decoder\n(Causal masking)', 
             ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax2.text(0.5, 0.8, 'Can only see left context (for generation)', 
             ha='center', fontsize=10, color='red',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    # Clean up axes
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.9)
        ax.axis('off')
    
    plt.suptitle('BERT vs GPT: Different Objectives, Different Architectures', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bert_vs_gpt_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all Week 6 visualizations
if __name__ == "__main__":
    print("Generating Week 6 Pre-trained Model visualizations...")
    plot_pretraining_workflow()
    print("- Pre-training workflow visualization created")
    plot_bert_vs_gpt_architecture()
    print("- BERT vs GPT architecture comparison created")  
    plot_bert_results_glue()
    print("- BERT GLUE results visualization created")
    print("\nWeek 6 visualizations completed!")