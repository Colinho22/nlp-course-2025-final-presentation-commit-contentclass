import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow, FancyArrowPatch
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

def plot_bert_training_process():
    """Visualize BERT's training process with masking"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top subplot: Masked Language Model (MLM)
    sentence = "The cat sat on the mat"
    words = sentence.split()
    positions = np.arange(len(words))
    
    # BERT masking example
    masked_sentence = ["The", "[MASK]", "sat", "on", "[MASK]", "mat"]
    predictions = ["cat", "the"]
    mask_positions = [1, 4]
    
    # Draw input sequence
    for i, (word, masked_word) in enumerate(zip(words, masked_sentence)):
        if i in mask_positions:
            # Masked word
            box = FancyBboxPatch((i-0.4, 1.5), 0.8, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor='red', edgecolor='black', linewidth=2)
            ax1.add_patch(box)
            ax1.text(i, 1.8, masked_word, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')
        else:
            # Regular word
            box = FancyBboxPatch((i-0.4, 1.5), 0.8, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor='black')
            ax1.add_patch(box)
            ax1.text(i, 1.8, word, ha='center', va='center',
                    fontsize=12, fontweight='bold')
    
    # BERT model representation
    bert_box = FancyBboxPatch((1, 0.2), 4, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='orange', edgecolor='black', linewidth=3)
    ax1.add_patch(bert_box)
    ax1.text(3, 0.6, 'BERT\n(Bidirectional Transformer)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')
    
    # Prediction arrows and results
    pred_idx = 0
    for i, pos in enumerate(mask_positions):
        # Arrow from BERT to prediction
        ax1.arrow(pos, 1.0, 0, -0.2,
                 head_width=0.1, head_length=0.05, fc='red', ec='red', linewidth=2)
        
        # Prediction box
        pred_box = FancyBboxPatch((pos-0.3, -0.5), 0.6, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor='green', edgecolor='black')
        ax1.add_patch(pred_box)
        ax1.text(pos, -0.3, predictions[pred_idx], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        pred_idx += 1
    
    # Labels
    ax1.text(-0.8, 1.8, 'Input:', fontsize=12, fontweight='bold', ha='right')
    ax1.text(-0.8, 0.6, 'Model:', fontsize=12, fontweight='bold', ha='right')
    ax1.text(-0.8, -0.3, 'Prediction:', fontsize=12, fontweight='bold', ha='right')
    
    ax1.set_xlim(-1.2, 6.5)
    ax1.set_ylim(-0.8, 2.5)
    ax1.set_title('BERT Training: Masked Language Modeling (MLM)', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Add explanation
    ax1.text(3, -0.7, '15% of words masked randomly during training', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Bottom subplot: Next Sentence Prediction (NSP)
    ax2.text(0.5, 0.85, 'Next Sentence Prediction (NSP)', ha='center', va='center',
            fontsize=16, fontweight='bold', transform=ax2.transAxes)
    
    # Positive example
    ax2.text(0.05, 0.7, 'Sentence A:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.18, 0.7, '"The man went to the store."', fontsize=11, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    
    ax2.text(0.05, 0.6, 'Sentence B:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.18, 0.6, '"He bought milk and bread."', fontsize=11, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    
    ax2.text(0.05, 0.5, 'Label:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.13, 0.5, 'IsNext', fontsize=11, fontweight='bold', transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='green', color='white'))
    
    # Negative example
    ax2.text(0.55, 0.7, 'Sentence A:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.68, 0.7, '"The cat slept all day."', fontsize=11, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    
    ax2.text(0.55, 0.6, 'Sentence B:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.68, 0.6, '"Quantum physics is complex."', fontsize=11, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral'))
    
    ax2.text(0.55, 0.5, 'Label:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.63, 0.5, 'NotNext', fontsize=11, fontweight='bold', transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='red', color='white'))
    
    # Training data statistics
    ax2.text(0.5, 0.3, 'Training Data', ha='center', va='center',
            fontsize=14, fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.25, 0.2, '• BookCorpus: 800M words', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.25, 0.15, '• Wikipedia: 2.5B words', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.25, 0.1, '• Total: 3.3B words', fontsize=11, fontweight='bold', transform=ax2.transAxes)
    
    ax2.text(0.75, 0.2, '• Training time: 4 days', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.75, 0.15, '• Hardware: 4-16 TPUs', fontsize=11, transform=ax2.transAxes)
    ax2.text(0.75, 0.1, '• Parameters: 110M-340M', fontsize=11, fontweight='bold', transform=ax2.transAxes)
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle('BERT Pre-training: Two-Task Learning', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bert_training_process.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fine_tuning_process():
    """Visualize the fine-tuning process"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Pre-training phase
    ax.text(2, 8.5, 'Phase 1: Pre-training', fontsize=16, fontweight='bold', ha='center')
    
    # Large corpus
    corpus_box = FancyBboxPatch((0.5, 7), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor='lightgray', edgecolor='black')
    ax.add_patch(corpus_box)
    ax.text(2, 7.5, 'Large Unlabeled Corpus\n(BookCorpus + Wikipedia)\n3.3B words', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow down
    ax.arrow(2, 6.8, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # BERT pre-training
    bert_pretrain_box = FancyBboxPatch((0.5, 5.5), 3, 1,
                                      boxstyle="round,pad=0.1",
                                      facecolor='orange', edgecolor='black')
    ax.add_patch(bert_pretrain_box)
    ax.text(2, 6, 'BERT Pre-training\nMLM + NSP\n340M parameters', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrow down
    ax.arrow(2, 5.3, 0, -0.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Pre-trained BERT
    pretrained_box = FancyBboxPatch((0.5, 4), 3, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor='blue', edgecolor='black')
    ax.add_patch(pretrained_box)
    ax.text(2, 4.5, 'Pre-trained BERT\n(Generic Language\nRepresentations)', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Fine-tuning phase
    ax.text(8, 8.5, 'Phase 2: Fine-tuning', fontsize=16, fontweight='bold', ha='center')
    
    # Task-specific data
    task_data_box = FancyBboxPatch((6.5, 7), 3, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightgreen', edgecolor='black')
    ax.add_patch(task_data_box)
    ax.text(8, 7.5, 'Task-Specific Dataset\n(e.g., Sentiment Analysis)\n10K labeled examples', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow from pre-trained to fine-tuning
    ax.arrow(3.7, 4.5, 2.6, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=3)
    ax.text(5.2, 4.8, 'Transfer', ha='center', va='center', fontsize=11, fontweight='bold', color='blue')
    
    # Fine-tuning box
    finetune_box = FancyBboxPatch((6.5, 4), 3, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='green', edgecolor='black')
    ax.add_patch(finetune_box)
    ax.text(8, 4.5, 'Fine-tuning\n+ Task-specific head\nSmall learning rate', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrow down for fine-tuning
    ax.arrow(8, 6.8, 0, -1.5, head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=2)
    
    # Final model
    final_box = FancyBboxPatch((6.5, 2.5), 3, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='purple', edgecolor='black')
    ax.add_patch(final_box)
    ax.text(8, 3, 'Task-Specific Model\n(e.g., Sentiment Classifier)\nReady for deployment', 
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Comparison table
    ax.text(5, 1.5, 'Comparison: From Scratch vs Fine-tuning', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Table headers
    ax.text(2, 1, 'From Scratch', ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7))
    ax.text(8, 1, 'BERT Fine-tuning', ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.7))
    
    # Comparison data
    comparisons = [
        ('Training Data', '100K+ examples', '1K examples'),
        ('Training Time', '2-3 days', '1-2 hours'),
        ('GPU Requirements', '8+ GPUs', '1 GPU'),
        ('Accuracy', '85%', '95%')
    ]
    
    for i, (metric, scratch, finetune) in enumerate(comparisons):
        y_pos = 0.5 - i * 0.15
        ax.text(0.5, y_pos, metric + ':', ha='right', va='center', fontsize=11, fontweight='bold')
        ax.text(2, y_pos, scratch, ha='center', va='center', fontsize=11)
        ax.text(8, y_pos, finetune, ha='center', va='center', fontsize=11)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 9)
    ax.set_title('BERT Fine-tuning: Transfer Learning in Action', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/fine_tuning_process.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bert_results_glue():
    """Plot BERT results on GLUE benchmark"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # GLUE tasks and scores
    tasks = ['CoLA', 'SST-2', 'MRPC', 'STS-B', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI']
    task_names = ['Linguistic\nAcceptability', 'Sentiment\nAnalysis', 'Paraphrase\nDetection', 
                  'Semantic\nSimilarity', 'Question\nPairs', 'Textual\nEntailment', 
                  'Question\nAnswering', 'Textual\nEntailment\n(Small)', 'Winograd\nSchema']
    
    # Scores (simplified for visualization)
    previous_sota = [35.0, 93.2, 86.3, 81.0, 86.1, 80.6, 82.3, 61.7, 53.4]
    bert_base = [52.1, 93.5, 88.9, 85.8, 89.2, 84.6, 90.5, 66.4, 65.1]
    bert_large = [60.5, 94.9, 89.3, 87.1, 89.3, 86.7, 92.7, 70.1, 65.1]
    
    # Plot 1: Task performance comparison
    x = np.arange(len(tasks))
    width = 0.25
    
    bars1 = ax1.bar(x - width, previous_sota, width, label='Previous SOTA', 
                   color='red', alpha=0.7)
    bars2 = ax1.bar(x, bert_base, width, label='BERT-Base', 
                   color='blue', alpha=0.7)
    bars3 = ax1.bar(x + width, bert_large, width, label='BERT-Large', 
                   color='green', alpha=0.7)
    
    ax1.set_xlabel('GLUE Tasks', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('BERT Performance on GLUE Benchmark', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add improvement percentages for BERT-Large
    for i, (prev, bert) in enumerate(zip(previous_sota, bert_large)):
        improvement = ((bert - prev) / prev) * 100
        if improvement > 5:  # Only show significant improvements
            ax1.text(i + width, bert + 2, f'+{improvement:.0f}%', 
                    ha='center', va='bottom', fontweight='bold', color='green', fontsize=9)
    
    # Plot 2: Overall GLUE score progression
    models = ['ELMo', 'OpenAI GPT', 'BERT-Base', 'BERT-Large']
    overall_scores = [68.7, 72.8, 78.3, 82.1]
    years = [2018.1, 2018.5, 2018.8, 2018.8]
    
    bars = ax2.bar(models, overall_scores, color=['orange', 'purple', 'blue', 'green'], alpha=0.7)
    ax2.set_ylabel('Overall GLUE Score', fontsize=12, fontweight='bold')
    ax2.set_title('GLUE Score Evolution (2018)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(60, 90)
    
    # Add score labels on bars
    for bar, score in zip(bars, overall_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Add human performance line
    ax2.axhline(y=87.1, color='red', linestyle='--', linewidth=2, label='Human Performance')
    ax2.legend(fontsize=11)
    
    # Highlight BERT's jump
    ax2.annotate('BERT breakthrough', xy=(2.5, 80), xytext=(1.5, 85),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, fontweight='bold', color='green')
    
    plt.suptitle('BERT: Dominating Natural Language Understanding', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bert_results_glue.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_transfer_learning_efficiency():
    """Plot transfer learning efficiency"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance vs Training Data Size
    data_sizes = np.array([10, 50, 100, 500, 1000, 5000, 10000])
    
    # From scratch performance (slower scaling)
    from_scratch_acc = 50 + 35 * (1 - np.exp(-data_sizes / 3000))
    # Fine-tuned BERT performance (faster scaling, higher ceiling)
    bert_finetune_acc = 65 + 30 * (1 - np.exp(-data_sizes / 500))
    
    ax1.semilogx(data_sizes, from_scratch_acc, 'r-o', linewidth=3, markersize=8,
                label='From Scratch')
    ax1.semilogx(data_sizes, bert_finetune_acc, 'b-s', linewidth=3, markersize=8,
                label='BERT Fine-tuning')
    
    ax1.set_xlabel('Training Examples', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Task Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Sample Efficiency', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(45, 100)
    
    # Highlight the key point
    ax1.annotate('BERT with 100 examples ≈\nFrom scratch with 10K examples', 
                xy=(100, bert_finetune_acc[2]), xytext=(500, 60),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=11, fontweight='bold', color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    # 2. Training Time Comparison
    tasks = ['Sentiment\nAnalysis', 'Question\nAnswering', 'Text\nClassification', 'NER']
    from_scratch_hours = [48, 120, 72, 96]
    bert_finetune_hours = [2, 4, 3, 4]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, from_scratch_hours, width, label='From Scratch', 
                   color='red', alpha=0.7)
    bars2 = ax2.bar(x + width/2, bert_finetune_hours, width, label='BERT Fine-tuning', 
                   color='blue', alpha=0.7)
    
    ax2.set_xlabel('NLP Tasks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (Hours)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time Reduction', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add speedup labels
    for i, (scratch, bert) in enumerate(zip(from_scratch_hours, bert_finetune_hours)):
        speedup = scratch / bert
        ax2.text(i, max(scratch, bert) * 1.5, f'{speedup:.0f}x faster',
                ha='center', va='bottom', fontweight='bold', color='green')
    
    # 3. Cross-lingual transfer
    languages = ['English', 'German', 'Chinese', 'Arabic', 'Hindi']
    monolingual_scores = [88, 0, 0, 0, 0]  # Only English trained
    multilingual_bert = [88, 78, 75, 72, 69]  # mBERT scores
    
    x = np.arange(len(languages))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, monolingual_scores, width, 
                   label='English-only Model', color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, multilingual_bert, width, 
                   label='Multilingual BERT', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Languages', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Task Performance', fontsize=12, fontweight='bold')
    ax3.set_title('Cross-lingual Transfer', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(languages)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # 4. Domain adaptation
    domains = ['News', 'Medical', 'Legal', 'Scientific', 'Social Media']
    general_bert = [85, 72, 68, 74, 79]
    domain_adapted = [87, 89, 84, 88, 85]
    
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, general_bert, width, 
                   label='General BERT', color='orange', alpha=0.7)
    bars2 = ax4.bar(x + width/2, domain_adapted, width, 
                   label='Domain-Adapted', color='green', alpha=0.7)
    
    ax4.set_xlabel('Domains', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Task Performance', fontsize=12, fontweight='bold')
    ax4.set_title('Domain Adaptation', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(domains, rotation=15, ha='right')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(60, 95)
    
    # Add improvement labels
    for i, (gen, domain) in enumerate(zip(general_bert, domain_adapted)):
        improvement = domain - gen
        if improvement > 2:
            ax4.text(i + width/2, domain + 1, f'+{improvement:.0f}',
                    ha='center', va='bottom', fontweight='bold', color='green')
    
    plt.suptitle('Transfer Learning: The Power of Pre-trained Models', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/transfer_learning_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Generating missing Week 6 BERT/Pre-trained Model figures...")
    
    print("1. Creating BERT training process visualization...")
    plot_bert_training_process()
    
    print("2. Creating fine-tuning process visualization...")
    plot_fine_tuning_process()
    
    print("3. Creating BERT GLUE results...")
    plot_bert_results_glue()
    
    print("4. Creating transfer learning efficiency comparison...")
    plot_transfer_learning_efficiency()
    
    print("Week 6 figures generated successfully!")

if __name__ == "__main__":
    main()