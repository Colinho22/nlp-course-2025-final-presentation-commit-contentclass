import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import matplotlib.patches as patches

# Set style for professional visuals
plt.style.use('seaborn-v0_8-whitegrid')

def create_transfer_learning_landscape():
    """Create 3D landscape showing knowledge transfer from pre-training to fine-tuning"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    resolution = 100
    x = np.linspace(-5, 5, resolution)
    y = np.linspace(-5, 5, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create pre-training landscape (broad knowledge)
    Z_pretrain = np.zeros_like(X)
    
    # Large central peak (general language understanding)
    Z_pretrain += 3 * np.exp(-((X)**2 + (Y)**2) / 8)
    
    # Surrounding knowledge areas
    knowledge_areas = [
        (-2, -2, 1.5, 1.0),  # Grammar
        (2, -2, 1.5, 1.0),   # Syntax
        (-2, 2, 1.5, 1.0),   # Semantics
        (2, 2, 1.5, 1.0),    # World knowledge
        (0, -3, 1.2, 0.8),   # Named entities
        (0, 3, 1.2, 0.8),    # Common sense
        (-3, 0, 1.2, 0.8),   # Factual knowledge
        (3, 0, 1.2, 0.8),    # Relationships
    ]
    
    for x_pos, y_pos, height, width in knowledge_areas:
        Z_pretrain += height * np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / (width**2))
    
    # Add fine-tuning peak (specialized task)
    Z_finetune = Z_pretrain.copy()
    Z_finetune += 2.5 * np.exp(-((X - 1.5)**2 + (Y - 1.5)**2) / 0.3)  # Sharp peak for specific task
    
    # Apply smoothing
    Z_pretrain = gaussian_filter(Z_pretrain, sigma=1)
    Z_finetune = gaussian_filter(Z_finetune, sigma=1)
    
    # Normalize
    Z_pretrain = (Z_pretrain - Z_pretrain.min()) / (Z_pretrain.max() - Z_pretrain.min())
    Z_finetune = (Z_finetune - Z_finetune.min()) / (Z_finetune.max() - Z_finetune.min())
    
    # Create custom colormap
    colors = ['#2E4057', '#048ABF', '#04D9C4', '#F2B705', '#F26419']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('transfer', colors, N=n_bins)
    
    # Plot pre-training surface
    surf = ax.plot_surface(X, Y, Z_pretrain, cmap=cmap, alpha=0.6,
                           linewidth=0, antialiased=True, label='Pre-training')
    
    # Plot fine-tuning peak
    mask = ((X - 1.5)**2 + (Y - 1.5)**2) < 1.5
    Z_peak = np.where(mask, Z_finetune, np.nan)
    ax.plot_surface(X, Y, Z_peak, cmap='Reds', alpha=0.8,
                   linewidth=0, antialiased=True)
    
    # Add contour lines at base
    ax.contour(X, Y, Z_pretrain, levels=10, colors='gray', alpha=0.3,
              linewidths=0.5, offset=0, zdir='z')
    
    # Labels and styling
    ax.set_xlabel('Language Features', fontsize=12, labelpad=10)
    ax.set_ylabel('Task Dimensions', fontsize=12, labelpad=10)
    ax.set_zlabel('Knowledge Depth', fontsize=12, labelpad=10)
    ax.set_title('Transfer Learning: From General Knowledge to Task Expertise',
                fontsize=16, fontweight='bold', pad=20)
    
    # Add annotations
    ax.text(0, 0, 0.6, 'General\nLanguage\nUnderstanding', fontsize=10,
           ha='center', fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
    
    ax.text(1.5, 1.5, 0.9, 'Task-Specific\nExpertise', fontsize=10,
           ha='center', fontweight='bold', color='white',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('../figures/transfer_learning_landscape.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_bert_vs_gpt_architecture():
    """Create side-by-side 3D comparison of BERT and GPT architectures"""
    fig = plt.figure(figsize=(16, 8))
    
    # BERT Architecture (left)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Create BERT blocks (bidirectional)
    layers = 12
    hidden_dim = 768
    heads = 12
    
    # Transformer blocks
    for layer in range(layers):
        z = layer * 0.8
        # Bidirectional attention (full cube)
        x = [0, 3, 3, 0, 0, 3, 3, 0]
        y = [0, 0, 2, 2, 0, 0, 2, 2]
        z_coords = [z, z, z, z, z+0.6, z+0.6, z+0.6, z+0.6]
        
        # Draw cube faces
        for i in range(4):
            ax1.plot([x[i], x[i+4]], [y[i], y[i+4]], [z_coords[i], z_coords[i+4]], 
                    'b-', alpha=0.3)
        
        # Color based on layer depth
        color = plt.cm.Blues(0.3 + 0.6 * layer / layers)
        
        # Add faces
        verts = [list(zip(x[0:4], y[0:4], z_coords[0:4]))]
        ax1.add_collection3d(Poly3DCollection(verts, alpha=0.7, facecolor=color, edgecolor='darkblue'))
    
    # Add bidirectional arrows
    ax1.quiver(1.5, 1, layers/2, 1, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2)
    ax1.quiver(1.5, 1, layers/2, -1, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2)
    
    ax1.set_title('BERT: Bidirectional Encoder', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Depth')
    ax1.set_zlabel('Layers')
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-1, 3)
    ax1.set_zlim(0, layers)
    ax1.view_init(elev=20, azim=45)
    
    # GPT Architecture (right)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Create GPT blocks (unidirectional)
    for layer in range(layers):
        z = layer * 0.8
        # Unidirectional attention (triangular prism)
        x = [0, 3, 0, 0, 3, 0]
        y = [0, 0, 2, 0, 0, 2]
        z_coords = [z, z, z, z+0.6, z+0.6, z+0.6]
        
        # Draw triangular prism
        for i in range(3):
            ax2.plot([x[i], x[i+3]], [y[i], y[i+3]], [z_coords[i], z_coords[i+3]], 
                    'g-', alpha=0.3)
        
        # Color based on layer depth
        color = plt.cm.Greens(0.3 + 0.6 * layer / layers)
        
        # Add faces
        verts = [list(zip(x[0:3], y[0:3], z_coords[0:3]))]
        ax2.add_collection3d(Poly3DCollection(verts, alpha=0.7, facecolor=color, edgecolor='darkgreen'))
    
    # Add unidirectional arrow
    ax2.quiver(1, 1, layers/2, 1, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2)
    
    ax2.set_title('GPT: Unidirectional Decoder', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Depth')
    ax2.set_zlabel('Layers')
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 3)
    ax2.set_zlim(0, layers)
    ax2.view_init(elev=20, azim=45)
    
    plt.suptitle('BERT vs GPT: Architectural Differences', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bert_vs_gpt_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_pretraining_timeline():
    """Create timeline showing evolution of pre-trained models"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Timeline data
    models = [
        (2018.0, 'ELMo', 94, 'First deep contextualized'),
        (2018.5, 'GPT', 117, 'Unsupervised pre-training'),
        (2018.8, 'BERT', 340, 'Bidirectional pre-training'),
        (2019.2, 'GPT-2', 1500, 'Zero-shot task transfer'),
        (2019.6, 'RoBERTa', 355, 'Optimized BERT'),
        (2019.8, 'T5', 11000, 'Text-to-text unified'),
        (2020.5, 'GPT-3', 175000, 'Few-shot learning'),
        (2022.0, 'PaLM', 540000, 'Pathways language model'),
        (2022.3, 'ChatGPT', 175000, 'Instruction following'),
        (2023.0, 'GPT-4', 1760000, 'Multimodal understanding'),
        (2024.0, 'Claude-3', 2000000, 'Constitutional AI'),
    ]
    
    # Extract data
    years = [m[0] for m in models]
    names = [m[1] for m in models]
    params = [m[2] for m in models]
    descriptions = [m[3] for m in models]
    
    # Create log scale for parameters
    log_params = np.log10(np.array(params))
    
    # Create timeline
    ax.scatter(years, log_params, s=200, c=years, cmap='viridis', 
              edgecolors='black', linewidth=2, zorder=3)
    
    # Add trend line
    z = np.polyfit(years, log_params, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(2018, 2024, 100)
    ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2, 
           label='Exponential growth trend')
    
    # Add labels
    for i, (year, name, param, desc) in enumerate(models):
        # Alternate label positions
        if i % 2 == 0:
            xytext = (year, log_params[i] + 0.3)
            va = 'bottom'
        else:
            xytext = (year, log_params[i] - 0.3)
            va = 'top'
        
        ax.annotate(f'{name}\n{param/1000:.1f}B params',
                   xy=(year, log_params[i]), xytext=xytext,
                   fontsize=9, ha='center', va=va,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.2',
                                 color='gray', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Model Size (log scale)', fontsize=12)
    ax.set_title('Evolution of Pre-trained Language Models', fontsize=16, fontweight='bold')
    
    # Y-axis labels
    y_labels = ['100M', '1B', '10B', '100B', '1T', '10T']
    y_positions = [2, 3, 4, 5, 6, 7]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2017.5, 2024.5)
    ax.set_ylim(1.5, 7)
    
    # Add era annotations
    ax.axvspan(2018, 2019, alpha=0.1, color='blue', label='Contextualized Era')
    ax.axvspan(2019, 2020.5, alpha=0.1, color='green', label='Scale Era')
    ax.axvspan(2020.5, 2024, alpha=0.1, color='red', label='Foundation Model Era')
    
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../figures/pretraining_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_fine_tuning_process():
    """Create visualization of fine-tuning process"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Loss curves comparison
    epochs = np.linspace(0, 20, 100)
    
    # Training from scratch
    scratch_loss = 2.5 * np.exp(-epochs/5) + 0.3 + 0.1 * np.random.randn(100) * np.exp(-epochs/10)
    scratch_acc = 1 - (0.7 * np.exp(-epochs/8) + 0.1)
    
    # Fine-tuning
    finetune_loss = 0.8 * np.exp(-epochs/2) + 0.1 + 0.05 * np.random.randn(100) * np.exp(-epochs/10)
    finetune_acc = 1 - (0.2 * np.exp(-epochs/3) + 0.05)
    
    # Plot loss
    ax1.plot(epochs, scratch_loss, 'r-', linewidth=2, label='Training from Scratch')
    ax1.plot(epochs, finetune_loss, 'b-', linewidth=2, label='Fine-tuning Pre-trained')
    ax1.fill_between(epochs, scratch_loss, finetune_loss, where=(scratch_loss > finetune_loss),
                     alpha=0.3, color='green', label='Efficiency Gain')
    
    ax1.set_xlabel('Training Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Efficiency: Scratch vs Fine-tuning', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 3)
    
    # Add annotations
    ax1.annotate('Converges in 3 epochs!', xy=(3, finetune_loss[15]), xytext=(6, 1),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=11, color='blue', fontweight='bold')
    ax1.annotate('Still learning basics', xy=(10, scratch_loss[50]), xytext=(12, 2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')
    
    # Right: Accuracy comparison
    ax2.plot(epochs, scratch_acc * 100, 'r-', linewidth=2, label='Training from Scratch')
    ax2.plot(epochs, finetune_acc * 100, 'b-', linewidth=2, label='Fine-tuning Pre-trained')
    
    # Add performance milestones
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    ax2.text(15, 91, '90% Accuracy Target', fontsize=10, color='green')
    
    # Mark when each reaches 90%
    scratch_90 = np.where(scratch_acc * 100 >= 90)[0]
    finetune_90 = np.where(finetune_acc * 100 >= 90)[0]
    
    if len(scratch_90) > 0:
        ax2.plot(epochs[scratch_90[0]], 90, 'ro', markersize=10)
        ax2.text(epochs[scratch_90[0]], 85, f'{epochs[scratch_90[0]]:.1f} epochs',
                ha='center', fontsize=9, color='red')
    
    if len(finetune_90) > 0:
        ax2.plot(epochs[finetune_90[0]], 90, 'bo', markersize=10)
        ax2.text(epochs[finetune_90[0]], 85, f'{epochs[finetune_90[0]]:.1f} epochs',
                ha='center', fontsize=9, color='blue')
    
    ax2.set_xlabel('Training Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Performance: Scratch vs Fine-tuning', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(20, 100)
    
    plt.suptitle('The Power of Pre-training: 10x Faster Convergence', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/fine_tuning_process.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_size_comparison():
    """Create comparison of model sizes and capabilities"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model data
    models = {
        'BERT-Base': {'params': 0.11, 'layers': 12, 'performance': 82},
        'BERT-Large': {'params': 0.34, 'layers': 24, 'performance': 85},
        'GPT-2': {'params': 1.5, 'layers': 48, 'performance': 88},
        'T5-Base': {'params': 0.22, 'layers': 12, 'performance': 84},
        'T5-Large': {'params': 0.77, 'layers': 24, 'performance': 87},
        'GPT-3': {'params': 175, 'layers': 96, 'performance': 93},
        'PaLM': {'params': 540, 'layers': 118, 'performance': 95},
        'GPT-4': {'params': 1760, 'layers': 120, 'performance': 97},
    }
    
    # Extract data
    names = list(models.keys())
    params = [models[m]['params'] for m in names]
    layers = [models[m]['layers'] for m in names]
    performance = [models[m]['performance'] for m in names]
    
    # Left: Model size bubble chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    
    for i, (name, param, layer, perf) in enumerate(zip(names, params, layers, performance)):
        # Bubble size proportional to log of parameters
        size = np.log10(param + 1) * 500 + 100
        ax1.scatter(layer, perf, s=size, c=[colors[i]], alpha=0.6,
                   edgecolors='black', linewidth=2)
        
        # Add labels
        if param < 10:
            label = f'{name}\n{param:.2f}B'
        else:
            label = f'{name}\n{param:.0f}B'
        
        ax1.annotate(label, xy=(layer, perf), 
                    fontsize=9, ha='center', va='center')
    
    ax1.set_xlabel('Number of Layers', fontsize=12)
    ax1.set_ylabel('Performance Score (%)', fontsize=12)
    ax1.set_title('Model Architecture vs Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 130)
    ax1.set_ylim(80, 100)
    
    # Right: Parameter count bar chart (log scale)
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, params, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, param) in enumerate(zip(bars, params)):
        width = bar.get_width()
        if param < 10:
            label = f'{param:.2f}B'
        else:
            label = f'{param:.0f}B'
        ax2.text(width * 1.1, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontsize=10)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Parameters (Billions, log scale)', fontsize=12)
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0.05, 5000)
    
    # Add size categories
    ax2.axvspan(0.05, 1, alpha=0.1, color='green', label='Small (<1B)')
    ax2.axvspan(1, 100, alpha=0.1, color='yellow', label='Medium (1-100B)')
    ax2.axvspan(100, 5000, alpha=0.1, color='red', label='Large (>100B)')
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.suptitle('Pre-trained Models: Size, Architecture, and Performance', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/model_size_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_bert_masking_visualization():
    """Create visualization of BERT's masked language modeling"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original sentence
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    positions = list(range(len(sentence)))
    
    # Subplot 1: Original sentence
    ax = axes[0, 0]
    for i, word in enumerate(sentence):
        rect = Rectangle((i, 0), 0.9, 0.8, facecolor='lightblue', 
                        edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.4, word, ha='center', va='center', 
               fontsize=12, fontweight='bold')
    
    ax.set_xlim(-0.1, len(sentence))
    ax.set_ylim(-0.1, 1)
    ax.set_title('Step 1: Original Sentence', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Subplot 2: Masked sentence
    ax = axes[0, 1]
    masked_idx = 1  # Mask "cat"
    for i, word in enumerate(sentence):
        if i == masked_idx:
            color = 'red'
            text = '[MASK]'
        else:
            color = 'lightblue'
            text = word
        
        rect = Rectangle((i, 0), 0.9, 0.8, facecolor=color, 
                        edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.4, text, ha='center', va='center',
               fontsize=12, fontweight='bold',
               color='white' if i == masked_idx else 'black')
    
    ax.set_xlim(-0.1, len(sentence))
    ax.set_ylim(-0.1, 1)
    ax.set_title('Step 2: Apply Masking (15% randomly)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Subplot 3: Bidirectional attention
    ax = axes[1, 0]
    for i, word in enumerate(sentence):
        if i == masked_idx:
            color = 'red'
            text = '[MASK]'
        else:
            color = 'lightgreen'
            text = word
        
        rect = Rectangle((i, 0), 0.9, 0.8, facecolor=color,
                        edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.4, text, ha='center', va='center',
               fontsize=12, fontweight='bold',
               color='white' if i == masked_idx else 'black')
        
        # Draw attention arrows
        if i != masked_idx:
            ax.arrow(i + 0.45, 0.8, 
                    (masked_idx - i) * 0.8, 0.7,
                    head_width=0.1, head_length=0.1,
                    fc='blue', ec='blue', alpha=0.5)
    
    ax.set_xlim(-0.1, len(sentence))
    ax.set_ylim(-0.1, 2)
    ax.set_title('Step 3: Bidirectional Context', fontsize=14, fontweight='bold')
    ax.text(masked_idx + 0.45, 1.7, 'All words contribute\nto prediction',
           ha='center', va='center', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    ax.axis('off')
    
    # Subplot 4: Prediction
    ax = axes[1, 1]
    predictions = [("cat", 0.89), ("dog", 0.05), ("rat", 0.03), ("bat", 0.02), ("hat", 0.01)]
    
    for i, word in enumerate(sentence):
        if i == masked_idx:
            # Show prediction distribution
            rect = Rectangle((i, 0), 0.9, 0.8, facecolor='lightgreen',
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i + 0.45, 0.4, 'cat', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='darkgreen')
            
            # Show top predictions
            for j, (pred_word, prob) in enumerate(predictions):
                y_pos = 1.2 + j * 0.25
                color = 'green' if j == 0 else 'gray'
                ax.text(i + 0.45, y_pos, f'{pred_word}: {prob:.2%}',
                       ha='center', va='center', fontsize=10,
                       color=color, fontweight='bold' if j == 0 else 'normal')
        else:
            rect = Rectangle((i, 0), 0.9, 0.8, facecolor='lightblue',
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(i + 0.45, 0.4, word, ha='center', va='center',
                   fontsize=12, fontweight='bold')
    
    ax.set_xlim(-0.1, len(sentence))
    ax.set_ylim(-0.1, 3)
    ax.set_title('Step 4: Predict Masked Token', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('BERT Masked Language Modeling Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../figures/bert_masking_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Week 6 Pre-trained Models visualizations...")
    
    print("Creating transfer learning landscape...")
    create_transfer_learning_landscape()
    print("Created: transfer_learning_landscape.pdf")
    
    print("Creating BERT vs GPT architecture comparison...")
    create_bert_vs_gpt_architecture()
    print("Created: bert_vs_gpt_architecture.pdf")
    
    print("Creating pre-training timeline...")
    create_pretraining_timeline()
    print("Created: pretraining_timeline.pdf")
    
    print("Creating fine-tuning process visualization...")
    create_fine_tuning_process()
    print("Created: fine_tuning_process.pdf")
    
    print("Creating model size comparison...")
    create_model_size_comparison()
    print("Created: model_size_comparison.pdf")
    
    print("Creating BERT masking visualization...")
    create_bert_masking_visualization()
    print("Created: bert_masking_visualization.pdf")
    
    print("\nAll Week 6 visualizations generated successfully!")