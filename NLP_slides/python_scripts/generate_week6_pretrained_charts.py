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

# 1. BERT Training Process Visualization
def plot_bert_training_process():
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Original sentence
    original = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'in', 'the', 'sun']
    positions = np.linspace(0.1, 0.9, len(original))
    
    # Masked sentence (15% masking)
    masked_indices = [1, 5, 8]  # cat, mat, sun
    masked = original.copy()
    masked[1] = '[MASK]'
    masked[5] = '[MASK]'
    masked[8] = 'house'  # Random replacement
    
    # Draw original sentence
    for i, (word, pos) in enumerate(zip(original, positions)):
        color = 'lightcoral' if i in masked_indices else 'lightblue'
        box = FancyBboxPatch((pos-0.04, 0.7), 0.08, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(pos, 0.74, word, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 0.82, 'Original Sentence', ha='center', fontsize=12, fontweight='bold')
    
    # Arrow down
    ax.arrow(0.5, 0.65, 0, -0.08, head_width=0.03, head_length=0.02, fc='black', ec='black')
    ax.text(0.52, 0.58, 'Masking\nProcess', ha='left', fontsize=10)
    
    # Draw masked sentence
    for i, (word, pos) in enumerate(zip(masked, positions)):
        if word == '[MASK]':
            color = 'gold'
        elif i == 8:  # Random replacement
            color = 'lightgreen'
        else:
            color = 'lightblue'
        
        box = FancyBboxPatch((pos-0.04, 0.4), 0.08, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(pos, 0.44, word, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 0.52, 'Training Input (15% masked)', ha='center', fontsize=12, fontweight='bold')
    
    # BERT model box
    bert_box = FancyBboxPatch((0.3, 0.2), 0.4, 0.12,
                             boxstyle="round,pad=0.02",
                             facecolor='lightcyan',
                             edgecolor='black',
                             linewidth=3)
    ax.add_patch(bert_box)
    ax.text(0.5, 0.26, 'BERT\n(Bidirectional Transformer)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows from masked tokens to BERT
    for idx in [1, 5, 8]:
        pos = positions[idx]
        ax.arrow(pos, 0.36, 0, -0.02, head_width=0.02, head_length=0.01, 
                fc='red', ec='red', linewidth=2)
    
    # Predictions
    predictions = ['cat', 'mat', 'sun']
    pred_positions = [positions[i] for i in masked_indices]
    
    for pred, pos in zip(predictions, pred_positions):
        ax.arrow(pos, 0.18, 0, -0.02, head_width=0.02, head_length=0.01, 
                fc='green', ec='green', linewidth=2)
        ax.text(pos, 0.1, pred, ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    ax.text(0.5, 0.05, 'Predictions', ha='center', fontsize=12, fontweight='bold')
    
    # Legend
    legend_items = [
        (0.1, 0.9, 'gold', '[MASK] token'),
        (0.3, 0.9, 'lightgreen', 'Random replacement'),
        (0.5, 0.9, 'lightcoral', 'Original (10% kept)'),
        (0.7, 0.9, 'lightblue', 'Unchanged')
    ]
    
    for x, y, color, label in legend_items:
        box = FancyBboxPatch((x-0.02, y-0.02), 0.04, 0.04,
                            boxstyle="round,pad=0.01",
                            facecolor=color,
                            edgecolor='black')
        ax.add_patch(box)
        ax.text(x+0.03, y, label, ha='left', va='center', fontsize=9)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'BERT Training: Masked Language Modeling', 
            ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../figures/bert_training_process.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Fine-tuning Process
def plot_fine_tuning_process():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Pre-trained BERT
    bert_box = FancyBboxPatch((0.3, 0.7), 0.4, 0.15,
                             boxstyle="round,pad=0.02",
                             facecolor='lightblue',
                             edgecolor='black',
                             linewidth=3)
    ax.add_patch(bert_box)
    ax.text(0.5, 0.775, 'Pre-trained BERT\n(110M parameters)\nKnows: Grammar, syntax,\nword meanings', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Three fine-tuning paths
    tasks = [
        ('Sentiment\nAnalysis', 0.2, 'lightcoral'),
        ('Question\nAnswering', 0.5, 'lightgreen'),
        ('Named Entity\nRecognition', 0.8, 'lightyellow')
    ]
    
    for task, x, color in tasks:
        # Arrow from BERT
        ax.arrow(0.5, 0.68, x-0.5, -0.13, head_width=0.02, head_length=0.02, 
                fc='gray', ec='gray', linewidth=2)
        
        # Task-specific head
        head_box = FancyBboxPatch((x-0.08, 0.45), 0.16, 0.08,
                                 boxstyle="round,pad=0.01",
                                 facecolor=color,
                                 edgecolor='black',
                                 linewidth=2)
        ax.add_patch(head_box)
        ax.text(x, 0.49, f'{task}\nHead', ha='center', va='center', fontsize=9)
        
        # Training data
        ax.text(x, 0.38, 'Training Data:', ha='center', fontsize=9, fontweight='bold')
        ax.text(x, 0.34, '1K examples', ha='center', fontsize=8)
        
        # Arrow to result
        ax.arrow(x, 0.42, 0, -0.05, head_width=0.02, head_length=0.01, 
                fc='black', ec='black')
        
        # Results
        results = ['98% accuracy', '91.3 F1 score', '95% precision']
        ax.text(x, 0.28, 'Result:', ha='center', fontsize=9, fontweight='bold')
        ax.text(x, 0.24, results[tasks.index((task, x, color))], 
                ha='center', fontsize=8, color='green', fontweight='bold')
    
    # Training details
    ax.text(0.5, 0.15, 'Fine-tuning Details:', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.11, '• Learning rate: 2e-5 (very small)', ha='center', fontsize=9)
    ax.text(0.5, 0.08, '• Training time: 2-4 hours on single GPU', ha='center', fontsize=9)
    ax.text(0.5, 0.05, '• Only update top layers initially', ha='center', fontsize=9)
    
    # Title
    ax.text(0.5, 0.95, 'Fine-tuning Pre-trained Models', 
            ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.91, 'One model, many tasks with minimal data', 
            ha='center', fontsize=12, style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/fine_tuning_process.pdf', dpi=300, bbox_inches='tight')
    plt.close()

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

# 4. Transfer Learning Efficiency
def plot_transfer_learning_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Data efficiency
    num_examples = [10, 50, 100, 500, 1000, 5000, 10000]
    from_scratch = [52, 58, 65, 74, 81, 87, 90]
    with_bert = [72, 81, 85, 90, 92, 94, 95]
    
    ax1.plot(num_examples, from_scratch, 'o-', label='Training from scratch', 
             color='red', linewidth=2, markersize=8)
    ax1.plot(num_examples, with_bert, 's-', label='Fine-tuning BERT', 
             color='blue', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of Training Examples', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Data Efficiency: BERT vs From Scratch', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate('100 examples with BERT\n= 5000 from scratch!', 
                xy=(100, 85), xytext=(200, 70),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # Right: Training time comparison
    methods = ['From\nScratch', 'Fine-tune\nBERT', 'Feature\nExtraction']
    training_time = [168, 4, 0.5]  # hours
    colors = ['red', 'blue', 'green']
    
    bars = ax2.bar(methods, training_time, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, time in zip(bars, training_time):
        label = f'{time}h' if time >= 1 else f'{int(time*60)}min'
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                label, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Training Time (hours)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 180)
    
    # Add cost estimates
    ax2.text(0.5, 0.85, 'Estimated AWS Cost:\n'
                       'From Scratch: ~$500\n'
                       'Fine-tune: ~$10\n'
                       'Feature Extract: ~$1',
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.suptitle('Why Pre-trained Models Changed Everything', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/transfer_learning_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 5. BERT vs GPT Architecture Comparison
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
    plot_bert_training_process()
    print("- BERT training process visualization created")
    plot_fine_tuning_process()
    print("- Fine-tuning process visualization created")
    plot_bert_results_glue()
    print("- BERT GLUE results visualization created")
    plot_transfer_learning_efficiency()
    print("- Transfer learning efficiency visualization created")
    plot_bert_vs_gpt_architecture()
    print("- BERT vs GPT architecture comparison created")
    print("\nWeek 6 visualizations completed!")