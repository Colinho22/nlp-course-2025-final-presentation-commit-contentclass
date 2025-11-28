import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

# Set style for professional visuals
plt.style.use('seaborn-v0_8-whitegrid')

def create_paradigm_shift_visualization():
    """Create powerful 3D visualization showing the pre-training revolution impact"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create three subplots for before, breakthrough, after
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.2)
    
    # Colors for consistency
    color_traditional = '#E74C3C'  # Red for inefficiency
    color_pretrain = '#3498DB'     # Blue for foundation
    color_finetune = '#2ECC71'     # Green for efficiency
    color_neutral = '#95A5A6'      # Gray for neutral
    
    # ============= TOP ROW: 3D Visualizations =============
    
    # --- BEFORE: Traditional Approach ---
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Create scattered peaks for different tasks
    resolution = 50
    x = np.linspace(-3, 3, resolution)
    y = np.linspace(-3, 3, resolution)
    X, Y = np.meshgrid(x, y)
    Z_before = np.zeros_like(X)
    
    # Individual task peaks (isolated learning)
    tasks = [
        (-1.5, -1.5, 0.8, 0.3, 'Sentiment'),
        (1.5, -1.5, 0.7, 0.3, 'NER'),
        (-1.5, 1.5, 0.9, 0.3, 'QA'),
        (1.5, 1.5, 0.6, 0.3, 'Translation'),
        (0, 0, 0.75, 0.3, 'Classification'),
    ]
    
    for x_pos, y_pos, height, width, label in tasks:
        Z_before += height * np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / (width**2))
    
    # Plot surface
    surf1 = ax1.plot_surface(X, Y, Z_before, cmap='Reds', alpha=0.7,
                             linewidth=0, antialiased=True)
    
    # Add task labels
    for x_pos, y_pos, height, width, label in tasks:
        ax1.text(x_pos, y_pos, height + 0.1, label, fontsize=8,
                ha='center', color=color_traditional)
    
    ax1.set_title('BEFORE: Isolated Task Learning\n(Each task learns from scratch)', 
                  fontsize=14, fontweight='bold', color=color_traditional)
    ax1.set_xlabel('Feature Space', fontsize=10)
    ax1.set_ylabel('Task Space', fontsize=10)
    ax1.set_zlabel('Knowledge', fontsize=10)
    ax1.set_zlim(0, 2)
    ax1.view_init(elev=25, azim=45)
    
    # Add waste indicator
    ax1.text2D(0.5, 0.02, 'Cost: $500K per task\nTime: 2 weeks each\nDuplication: 90%', 
               transform=ax1.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))
    
    # --- CENTER: Pre-training Breakthrough ---
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    
    Z_pretrain = np.zeros_like(X)
    
    # Massive central mountain (general knowledge)
    Z_pretrain += 1.8 * np.exp(-((X)**2 + (Y)**2) / 2)
    
    # Surrounding knowledge areas
    knowledge_areas = [
        (-1, -1, 0.8, 0.5),  # Grammar
        (1, -1, 0.8, 0.5),   # Syntax
        (-1, 1, 0.8, 0.5),   # Semantics
        (1, 1, 0.8, 0.5),    # World knowledge
    ]
    
    for x_pos, y_pos, height, width in knowledge_areas:
        Z_pretrain += height * np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / (width**2))
    
    # Smooth the surface
    Z_pretrain = gaussian_filter(Z_pretrain, sigma=0.5)
    
    # Plot surface with gradient colormap
    colors_gradient = ['#2C3E50', '#3498DB', '#5DADE2', '#85C1E2', '#AED6F1']
    cmap_pretrain = LinearSegmentedColormap.from_list('pretrain', colors_gradient)
    surf2 = ax2.plot_surface(X, Y, Z_pretrain, cmap=cmap_pretrain, alpha=0.8,
                             linewidth=0, antialiased=True)
    
    # Add contour lines
    ax2.contour(X, Y, Z_pretrain, levels=8, colors='gray', alpha=0.3,
               linewidths=0.5, offset=0, zdir='z')
    
    ax2.set_title('BREAKTHROUGH: Pre-training Foundation\n(Learn once from all text)', 
                  fontsize=14, fontweight='bold', color=color_pretrain)
    ax2.set_xlabel('Feature Space', fontsize=10)
    ax2.set_ylabel('Task Space', fontsize=10)
    ax2.set_zlabel('Knowledge', fontsize=10)
    ax2.set_zlim(0, 2)
    ax2.view_init(elev=25, azim=45)
    
    # Add foundation label
    ax2.text(0, 0, 1.9, 'Universal\nLanguage\nUnderstanding', fontsize=10,
            ha='center', fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color_pretrain, alpha=0.8))
    
    # Add efficiency indicator
    ax2.text2D(0.5, 0.02, 'One-time cost: $1M\nReusable foundation\nShared knowledge: 100%', 
               transform=ax2.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.2))
    
    # --- AFTER: Fine-tuning Efficiency ---
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    
    # Start with pre-trained foundation
    Z_finetune = Z_pretrain.copy()
    
    # Add fine-tuning peaks on top
    finetune_tasks = [
        (-1.5, -1.5, 0.4, 0.15, 'Sentiment\n99% acc'),
        (1.5, -1.5, 0.35, 0.15, 'NER\n95% F1'),
        (-1.5, 1.5, 0.45, 0.15, 'QA\n92% EM'),
        (1.5, 1.5, 0.3, 0.15, 'Translation\n45 BLEU'),
        (0, 0, 0.4, 0.15, 'Classification\n98% acc'),
    ]
    
    for x_pos, y_pos, height, width, label in finetune_tasks:
        Z_finetune += height * np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / (width**2))
    
    # Plot foundation in blue
    surf3a = ax3.plot_surface(X, Y, Z_pretrain, cmap=cmap_pretrain, alpha=0.4,
                              linewidth=0, antialiased=True)
    
    # Plot fine-tuning peaks in green
    for x_pos, y_pos, height, width, label in finetune_tasks:
        mask = ((X - x_pos)**2 + (Y - y_pos)**2) < 0.5
        Z_peak = np.where(mask, Z_finetune, np.nan)
        ax3.plot_surface(X, Y, Z_peak, color=color_finetune, alpha=0.8,
                        linewidth=0, antialiased=True)
        # Add performance labels
        # Find closest indices in the grid
        x_idx = np.abs(x - x_pos).argmin()
        y_idx = np.abs(y - y_pos).argmin()
        z_pos = Z_pretrain[y_idx, x_idx] + height + 0.1
        ax3.text(x_pos, y_pos, z_pos, label, fontsize=7,
                ha='center', color=color_finetune, fontweight='bold')
    
    ax3.set_title('AFTER: Efficient Fine-tuning\n(Adapt pre-trained model to tasks)', 
                  fontsize=14, fontweight='bold', color=color_finetune)
    ax3.set_xlabel('Feature Space', fontsize=10)
    ax3.set_ylabel('Task Space', fontsize=10)
    ax3.set_zlabel('Knowledge', fontsize=10)
    ax3.set_zlim(0, 2)
    ax3.view_init(elev=25, azim=45)
    
    # Add efficiency indicator
    ax3.text2D(0.5, 0.02, 'Cost: $1K per task\nTime: 2 hours each\nEfficiency: 100x', 
               transform=ax3.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
    
    # ============= BOTTOM ROW: Timeline and Metrics =============
    
    # Timeline
    ax4 = fig.add_subplot(gs[1, :])
    
    # Timeline data
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    models = ['', 'BERT\nELMo\nULMFiT', 'GPT-2\nRoBERTa\nXLNet', 
              'GPT-3\nT5', 'DALL-E\nCodex', 'ChatGPT\nPaLM', 
              'GPT-4\nClaude', 'Gemini\nLlama 3']
    
    # Performance improvement (relative)
    performance = [1, 2.5, 4, 8, 15, 30, 50, 75]
    
    # Plot timeline
    ax4.plot(years, performance, 'o-', color=color_pretrain, linewidth=3, markersize=10)
    
    # Add model labels
    for i, (year, model, perf) in enumerate(zip(years, models, performance)):
        if model:
            ax4.annotate(model, (year, perf), textcoords="offset points", 
                        xytext=(0, 15), ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='lightblue' if i < 4 else 'lightgreen', 
                                 alpha=0.7))
    
    # Highlight pre-training era
    ax4.axvspan(2018, 2024, alpha=0.2, color='green', label='Pre-training Era')
    ax4.axvspan(2017, 2018, alpha=0.2, color='red', label='Traditional Era')
    
    # Labels and styling
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Relative Performance\n(vs Traditional Baseline)', fontsize=12, fontweight='bold')
    ax4.set_title('The Pre-training Revolution: Exponential Progress', 
                  fontsize=16, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.set_xlim(2016.5, 2024.5)
    ax4.set_ylim(0, 80)
    
    # Add impact metrics
    metrics_text = (
        'Key Metrics:\n'
        '• Training Cost: 100x reduction\n'
        '• Time to Deploy: 50x faster\n'
        '• Performance: 10x improvement\n'
        '• Accessibility: From labs to everyone'
    )
    ax4.text(0.02, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.2))
    
    # Add arrows showing progression
    arrow1 = ConnectionPatch((2017.5, 40), (2018.5, 40), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc=color_traditional, ec=color_traditional,
                            linewidth=2)
    ax4.add_artist(arrow1)
    ax4.text(2018, 42, 'Paradigm\nShift', ha='center', fontsize=10, 
            fontweight='bold', color=color_traditional)
    
    plt.suptitle('The Pre-training Revolution: How Foundation Models Changed Everything', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('../figures/paradigm_shift_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/paradigm_shift_impact.png', dpi=150, bbox_inches='tight')
    print("Generated: paradigm_shift_impact.pdf/png")

def create_knowledge_transfer_mountain():
    """Create detailed 3D mountain visualization of knowledge transfer"""
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create high-resolution meshgrid
    resolution = 150
    x = np.linspace(-6, 6, resolution)
    y = np.linspace(-6, 6, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create the main mountain (pre-trained knowledge)
    Z_mountain = np.zeros_like(X)
    
    # Central peak - general language understanding
    Z_mountain += 4 * np.exp(-((X)**2 + (Y)**2) / 10)
    
    # Knowledge ridges extending outward
    ridges = [
        # Direction, length, width, height
        (0, 1, 3, 0.5, 2),      # North: Syntax
        (1, 0, 3, 0.5, 2),      # East: Semantics
        (0, -1, 3, 0.5, 2),     # South: Grammar
        (-1, 0, 3, 0.5, 2),     # West: Pragmatics
        (0.7, 0.7, 2, 0.4, 1.5),   # NE: Entities
        (-0.7, 0.7, 2, 0.4, 1.5),  # NW: Relations
        (0.7, -0.7, 2, 0.4, 1.5),  # SE: Context
        (-0.7, -0.7, 2, 0.4, 1.5), # SW: Discourse
    ]
    
    for dx, dy, length, width, height in ridges:
        for i in np.linspace(0, length, 20):
            Z_mountain += height * np.exp(-((X - dx*i)**2 + (Y - dy*i)**2) / (width**2)) * np.exp(-i/length)
    
    # Add foothills (basic features)
    foothills = [
        (-3, -3, 0.8, 1.0), (-3, 0, 0.7, 1.0), (-3, 3, 0.8, 1.0),
        (0, -3, 0.7, 1.0), (0, 3, 0.7, 1.0),
        (3, -3, 0.8, 1.0), (3, 0, 0.7, 1.0), (3, 3, 0.8, 1.0),
    ]
    
    for x_pos, y_pos, height, width in foothills:
        Z_mountain += height * np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / (width**2))
    
    # Apply smoothing for realistic terrain
    Z_mountain = gaussian_filter(Z_mountain, sigma=1.5)
    
    # Normalize
    Z_mountain = (Z_mountain - Z_mountain.min()) / (Z_mountain.max() - Z_mountain.min()) * 3
    
    # Create custom colormap for terrain
    colors_terrain = ['#2C3E50', '#34495E', '#5D6D7E', '#7F8C8D', '#95A5A6', 
                      '#A9CCE3', '#85C1E2', '#5DADE2', '#3498DB', '#2E86AB']
    cmap_terrain = LinearSegmentedColormap.from_list('terrain', colors_terrain)
    
    # Plot the mountain
    surf = ax.plot_surface(X, Y, Z_mountain, cmap=cmap_terrain, alpha=0.9,
                          linewidth=0, antialiased=True, shade=True)
    
    # Add contour lines
    levels = np.linspace(0, 3, 15)
    ax.contour(X, Y, Z_mountain, levels=levels, colors='gray', alpha=0.2,
              linewidths=0.5, offset=0, zdir='z')
    
    # Add task-specific adaptations as colored peaks
    tasks = [
        (1.5, 1.5, 'Sentiment\nAnalysis', '#E74C3C'),
        (-1.5, 1.5, 'Question\nAnswering', '#F39C12'),
        (1.5, -1.5, 'Named Entity\nRecognition', '#27AE60'),
        (-1.5, -1.5, 'Machine\nTranslation', '#8E44AD'),
        (0, 2.5, 'Text\nSummarization', '#3498DB'),
    ]
    
    for x_pos, y_pos, label, color in tasks:
        # Get base height at this position
        idx_x = np.abs(x - x_pos).argmin()
        idx_y = np.abs(y - y_pos).argmin()
        base_height = Z_mountain[idx_y, idx_x]
        
        # Add task peak
        Z_task = base_height + 0.5 * np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / 0.1)
        mask = ((X - x_pos)**2 + (Y - y_pos)**2) < 0.3
        Z_task_masked = np.where(mask, Z_task, np.nan)
        
        ax.plot_surface(X, Y, Z_task_masked, color=color, alpha=0.8)
        
        # Add label
        ax.text(x_pos, y_pos, base_height + 0.7, label, fontsize=9,
               ha='center', color=color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add main labels
    ax.text(0, 0, 3.2, 'Pre-trained\nFoundation', fontsize=14,
           ha='center', fontweight='bold', color='#2C3E50',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    # Add knowledge area labels
    knowledge_labels = [
        (0, 3, 1.5, 'Syntax'),
        (3, 0, 1.5, 'Semantics'),
        (0, -3, 1.5, 'Grammar'),
        (-3, 0, 1.5, 'Pragmatics'),
    ]
    
    for x_pos, y_pos, z_pos, label in knowledge_labels:
        ax.text(x_pos, y_pos, z_pos, label, fontsize=10,
               ha='center', color='#34495E', fontweight='bold', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Language Features', fontsize=12, labelpad=10)
    ax.set_ylabel('Task Dimensions', fontsize=12, labelpad=10)
    ax.set_zlabel('Knowledge Depth', fontsize=12, labelpad=10)
    ax.set_title('The Knowledge Transfer Mountain:\nHow Pre-training Creates a Foundation for All Tasks',
                fontsize=16, fontweight='bold', pad=20)
    
    # Set viewing angle for best perspective
    ax.view_init(elev=30, azim=135)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(0, 3.5)
    
    # Add legend for task colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label.replace('\n', ' '))
                      for _, _, label, color in tasks]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             title='Fine-tuned Tasks', title_fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../figures/knowledge_transfer_mountain.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/knowledge_transfer_mountain.png', dpi=150, bbox_inches='tight')
    print("Generated: knowledge_transfer_mountain.pdf/png")

def create_mlm_interactive_animation():
    """Create MLM masking and prediction visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Masked Language Modeling: How BERT Learns', fontsize=18, fontweight='bold')
    
    # Example sentence
    sentence = "The quick brown fox jumps over the lazy dog"
    words = sentence.split()
    
    # Colors
    color_normal = '#3498DB'
    color_masked = '#E74C3C'
    color_predicted = '#27AE60'
    color_context = '#F39C12'
    
    # Step 1: Original sentence
    ax = axes[0, 0]
    ax.set_title('Step 1: Original Text', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    for i, word in enumerate(words):
        x = 0.1 + i * 0.1
        y = 0.5
        ax.add_patch(Rectangle((x - 0.04, y - 0.15), 0.08, 0.3,
                               facecolor=color_normal, alpha=0.3, edgecolor=color_normal))
        ax.text(x, y, word, ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Step 2: Random masking
    ax = axes[0, 1]
    ax.set_title('Step 2: Random Masking (15%)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    masked_indices = [1, 4, 7]  # "quick", "jumps", "lazy"
    for i, word in enumerate(words):
        x = 0.1 + i * 0.1
        y = 0.5
        if i in masked_indices:
            color = color_masked
            display_word = '[MASK]'
            alpha = 0.5
        else:
            color = color_normal
            display_word = word
            alpha = 0.3
        
        ax.add_patch(Rectangle((x - 0.04, y - 0.15), 0.08, 0.3,
                               facecolor=color, alpha=alpha, edgecolor=color))
        ax.text(x, y, display_word, ha='center', va='center', fontsize=12,
               fontweight='bold' if i in masked_indices else 'normal')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Step 3: Context encoding
    ax = axes[0, 2]
    ax.set_title('Step 3: Bidirectional Context', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    for i, word in enumerate(words):
        x = 0.1 + i * 0.1
        y = 0.5
        if i in masked_indices:
            color = color_masked
            display_word = '[MASK]'
        else:
            color = color_context
            display_word = word
        
        ax.add_patch(Rectangle((x - 0.04, y - 0.15), 0.08, 0.3,
                               facecolor=color, alpha=0.4, edgecolor=color))
        ax.text(x, y, display_word, ha='center', va='center', fontsize=12)
        
        # Draw bidirectional arrows
        if i in masked_indices:
            # Left arrow
            if i > 0:
                ax.arrow(x - 0.08, y, 0.03, 0, head_width=0.05, head_length=0.01,
                        fc=color_context, ec=color_context, alpha=0.5)
            # Right arrow
            if i < len(words) - 1:
                ax.arrow(x + 0.05, y, 0.03, 0, head_width=0.05, head_length=0.01,
                        fc=color_context, ec=color_context, alpha=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Step 4: Prediction
    ax = axes[1, 0]
    ax.set_title('Step 4: Prediction', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    predictions = {1: 'quick', 4: 'jumps', 7: 'lazy'}
    for i, word in enumerate(words):
        x = 0.1 + i * 0.1
        y = 0.5
        if i in masked_indices:
            color = color_predicted
            display_word = predictions[i]
            alpha = 0.5
        else:
            color = color_normal
            display_word = word
            alpha = 0.3
        
        ax.add_patch(Rectangle((x - 0.04, y - 0.15), 0.08, 0.3,
                               facecolor=color, alpha=alpha, edgecolor=color))
        ax.text(x, y, display_word, ha='center', va='center', fontsize=12,
               fontweight='bold' if i in masked_indices else 'normal')
        
        if i in masked_indices:
            ax.text(x, y - 0.25, 'Predicted!', ha='center', fontsize=8,
                   color=color_predicted, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Step 5: Learning signal
    ax = axes[1, 1]
    ax.set_title('Step 5: Learning from Errors', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Show loss calculation
    ax.text(0.5, 0.7, 'Loss Calculation:', fontsize=12, fontweight='bold', ha='center')
    
    losses = [
        ('quick → quick', '✓ Correct!', color_predicted),
        ('jumps → leaps', '✗ Wrong', color_masked),
        ('lazy → lazy', '✓ Correct!', color_predicted),
    ]
    
    for i, (pred, result, color) in enumerate(losses):
        y = 0.5 - i * 0.15
        ax.text(0.3, y, pred, fontsize=11, ha='right')
        ax.text(0.5, y, result, fontsize=11, ha='center', color=color, fontweight='bold')
    
    ax.text(0.5, 0.1, 'Update weights to improve predictions', 
           fontsize=10, ha='center', style='italic')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Step 6: What BERT learns
    ax = axes[1, 2]
    ax.set_title('What BERT Learns', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    learnings = [
        'Grammar patterns',
        'Word relationships',
        'Context understanding',
        'Semantic meanings',
        'World knowledge',
    ]
    
    for i, learning in enumerate(learnings):
        y = 0.8 - i * 0.15
        ax.add_patch(Circle((0.2, y), 0.03, facecolor=color_predicted, alpha=0.5))
        ax.text(0.3, y, learning, fontsize=11, va='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('../figures/mlm_interactive_animation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/mlm_interactive_animation.png', dpi=150, bbox_inches='tight')
    print("Generated: mlm_interactive_animation.pdf/png")

def create_scale_comparison_chart():
    """Create chart showing scale of models over time"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('The Scale Revolution: How Big Models Got', fontsize=18, fontweight='bold')
    
    # Data
    models = ['BERT-Base', 'BERT-Large', 'GPT-2', 'GPT-3', 'PaLM', 'GPT-4']
    years = [2018, 2018.5, 2019, 2020, 2022, 2023]
    params = [0.11, 0.34, 1.5, 175, 540, 1760]  # In billions
    data_size = [16, 16, 40, 570, 780, 13000]  # In GB of text
    compute_cost = [1, 3, 10, 1000, 5000, 100000]  # Relative cost
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))
    
    # 1. Parameters over time
    ax = axes[0, 0]
    bars1 = ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Parameters (Billions)', fontsize=12, fontweight='bold')
    ax.set_title('Model Size Growth', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, param in zip(bars1, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{param}B', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Training data size
    ax = axes[0, 1]
    bars2 = ax.bar(models, data_size, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Training Data (GB of text)', fontsize=12, fontweight='bold')
    ax.set_title('Training Data Scale', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, data in zip(bars2, data_size):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{data}GB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Compute cost
    ax = axes[1, 0]
    ax.plot(years, compute_cost, 'o-', color='#E74C3C', linewidth=3, markersize=10)
    ax.fill_between(years, compute_cost, alpha=0.3, color='#E74C3C')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Compute Cost', fontsize=12, fontweight='bold')
    ax.set_title('Computational Requirements', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add model annotations
    for year, cost, model in zip(years, compute_cost, models):
        ax.annotate(model, (year, cost), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 4. Performance vs Scale
    ax = axes[1, 1]
    
    # Create scatter plot with size representing parameters
    sizes = [p * 5 for p in params]  # Scale for visibility
    scatter = ax.scatter(params, [85, 88, 90, 93, 95, 97], s=sizes, c=colors,
                        alpha=0.6, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Parameters (Billions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (Benchmark Score)', fontsize=12, fontweight='bold')
    ax.set_title('Scaling Laws: Bigger is Better', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (params[i], [85, 88, 90, 93, 95, 97][i]),
                   fontsize=8, ha='center', va='bottom')
    
    # Add trend line
    from scipy import stats
    log_params = np.log10(params)
    scores = [85, 88, 90, 93, 95, 97]
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, scores)
    line_x = np.logspace(-1, 3.5, 100)
    line_y = slope * np.log10(line_x) + intercept
    ax.plot(line_x, line_y, 'r--', alpha=0.5, label=f'Scaling Law (R²={r_value**2:.3f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../figures/scale_comparison_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/scale_comparison_chart.png', dpi=150, bbox_inches='tight')
    print("Generated: scale_comparison_chart.pdf/png")

def create_finetuning_strategy_flowchart():
    """Create decision tree for fine-tuning strategies"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Fine-tuning Strategy Decision Tree', fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Define node positions
    nodes = {
        'start': (0.5, 0.9),
        'data_size': (0.5, 0.75),
        'small_data': (0.25, 0.6),
        'medium_data': (0.5, 0.6),
        'large_data': (0.75, 0.6),
        'few_shot': (0.15, 0.45),
        'lora': (0.35, 0.45),
        'partial': (0.5, 0.45),
        'full': (0.65, 0.45),
        'custom': (0.85, 0.45),
        'compute': (0.5, 0.3),
        'low_compute': (0.3, 0.15),
        'high_compute': (0.7, 0.15),
    }
    
    # Colors for different strategies
    colors = {
        'decision': '#3498DB',
        'strategy': '#27AE60',
        'consideration': '#F39C12',
        'final': '#E74C3C',
    }
    
    # Draw nodes
    def draw_node(pos, text, node_type='decision', width=0.12, height=0.06):
        x, y = pos
        if node_type == 'decision':
            # Diamond shape for decisions
            diamond = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                            boxstyle="round,pad=0.01",
                                            facecolor=colors[node_type], alpha=0.3,
                                            edgecolor=colors[node_type], linewidth=2)
            ax.add_patch(diamond)
        else:
            # Rectangle for strategies
            rect = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                        boxstyle="round,pad=0.01",
                                        facecolor=colors[node_type], alpha=0.3,
                                        edgecolor=colors[node_type], linewidth=2)
            ax.add_patch(rect)
        
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
               fontweight='bold', wrap=True)
    
    # Draw all nodes
    draw_node(nodes['start'], 'Start:\nChoose Strategy', 'decision')
    draw_node(nodes['data_size'], 'How much\ntask data?', 'decision')
    draw_node(nodes['small_data'], '<1K examples', 'consideration', 0.1, 0.05)
    draw_node(nodes['medium_data'], '1K-100K', 'consideration', 0.08, 0.05)
    draw_node(nodes['large_data'], '>100K', 'consideration', 0.08, 0.05)
    
    draw_node(nodes['few_shot'], 'Few-shot\nPrompting', 'strategy')
    draw_node(nodes['lora'], 'LoRA/\nPEFT', 'strategy')
    draw_node(nodes['partial'], 'Partial\nFine-tuning', 'strategy')
    draw_node(nodes['full'], 'Full\nFine-tuning', 'strategy')
    draw_node(nodes['custom'], 'Custom\nArchitecture', 'strategy')
    
    draw_node(nodes['compute'], 'Compute\nBudget?', 'decision')
    draw_node(nodes['low_compute'], 'Use LoRA\n(1% params)', 'final', 0.1, 0.05)
    draw_node(nodes['high_compute'], 'Full tuning\n(100% params)', 'final', 0.1, 0.05)
    
    # Draw connections
    def draw_arrow(start, end, label='', style='-'):
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        ax.annotate('', xy=(x2, y2 + 0.03), xytext=(x1, y1 - 0.03),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, label, fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Draw all arrows
    draw_arrow('start', 'data_size')
    draw_arrow('data_size', 'small_data')
    draw_arrow('data_size', 'medium_data')
    draw_arrow('data_size', 'large_data')
    draw_arrow('small_data', 'few_shot')
    draw_arrow('small_data', 'lora')
    draw_arrow('medium_data', 'partial')
    draw_arrow('medium_data', 'lora')
    draw_arrow('large_data', 'full')
    draw_arrow('large_data', 'custom')
    draw_arrow('partial', 'compute')
    draw_arrow('compute', 'low_compute', 'Limited')
    draw_arrow('compute', 'high_compute', 'Abundant')
    
    # Add strategy descriptions
    descriptions = [
        (0.05, 0.35, 'Few-shot:\n• No training\n• Quick setup\n• Limited performance'),
        (0.25, 0.35, 'LoRA:\n• 1% parameters\n• Memory efficient\n• Good results'),
        (0.45, 0.35, 'Partial:\n• Top layers only\n• Balanced approach\n• Moderate cost'),
        (0.65, 0.35, 'Full:\n• All parameters\n• Best performance\n• High cost'),
        (0.85, 0.35, 'Custom:\n• Task-specific\n• Maximum control\n• Expert required'),
    ]
    
    for x, y, text in descriptions:
        ax.text(x, y, text, fontsize=8, va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=colors['decision'], alpha=0.3, label='Decision Point'),
        patches.Patch(facecolor=colors['strategy'], alpha=0.3, label='Strategy Option'),
        patches.Patch(facecolor=colors['consideration'], alpha=0.3, label='Data Size'),
        patches.Patch(facecolor=colors['final'], alpha=0.3, label='Final Choice'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('../figures/finetuning_strategy_flowchart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/finetuning_strategy_flowchart.png', dpi=150, bbox_inches='tight')
    print("Generated: finetuning_strategy_flowchart.pdf/png")

def create_application_ecosystem_map():
    """Create visualization of all applications enabled by pre-training"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_title('The Pre-training Application Ecosystem', fontsize=20, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Central node
    center = (0.5, 0.5)
    ax.add_patch(Circle(center, 0.08, facecolor='#3498DB', alpha=0.8, edgecolor='black', linewidth=3))
    ax.text(center[0], center[1], 'Pre-trained\nFoundation\nModel', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Application categories with their tasks
    categories = {
        'Understanding': {
            'position': (0.2, 0.8),
            'color': '#E74C3C',
            'tasks': ['Sentiment Analysis', 'Intent Detection', 'Emotion Recognition', 'Sarcasm Detection']
        },
        'Generation': {
            'position': (0.5, 0.85),
            'color': '#27AE60',
            'tasks': ['Text Generation', 'Story Writing', 'Code Generation', 'Poetry Creation']
        },
        'Translation': {
            'position': (0.8, 0.8),
            'color': '#F39C12',
            'tasks': ['Language Translation', 'Style Transfer', 'Paraphrasing', 'Simplification']
        },
        'Information': {
            'position': (0.85, 0.5),
            'color': '#8E44AD',
            'tasks': ['Question Answering', 'Information Extraction', 'Fact Checking', 'Knowledge Base']
        },
        'Analysis': {
            'position': (0.8, 0.2),
            'color': '#16A085',
            'tasks': ['Named Entity Recognition', 'POS Tagging', 'Dependency Parsing', 'Coreference']
        },
        'Conversation': {
            'position': (0.5, 0.15),
            'color': '#2980B9',
            'tasks': ['Chatbots', 'Virtual Assistants', 'Customer Service', 'Therapy Bots']
        },
        'Summarization': {
            'position': (0.2, 0.2),
            'color': '#C0392B',
            'tasks': ['Document Summary', 'News Digest', 'Meeting Notes', 'Research Abstract']
        },
        'Classification': {
            'position': (0.15, 0.5),
            'color': '#D35400',
            'tasks': ['Topic Classification', 'Spam Detection', 'Content Moderation', 'Genre Detection']
        },
    }
    
    # Draw categories and connections
    for category, info in categories.items():
        cat_pos = info['position']
        cat_color = info['color']
        
        # Draw connection to center
        ax.plot([center[0], cat_pos[0]], [center[1], cat_pos[1]],
               'k-', alpha=0.2, linewidth=2)
        
        # Draw category node
        ax.add_patch(Circle(cat_pos, 0.06, facecolor=cat_color, alpha=0.7,
                           edgecolor=cat_color, linewidth=2))
        ax.text(cat_pos[0], cat_pos[1], category, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')
        
        # Draw tasks around category
        n_tasks = len(info['tasks'])
        angle_step = 2 * np.pi / n_tasks
        task_radius = 0.12
        
        for i, task in enumerate(info['tasks']):
            angle = i * angle_step - np.pi/2
            task_x = cat_pos[0] + task_radius * np.cos(angle)
            task_y = cat_pos[1] + task_radius * np.sin(angle)
            
            # Draw task node
            ax.add_patch(Circle((task_x, task_y), 0.025, facecolor=cat_color,
                              alpha=0.3, edgecolor=cat_color, linewidth=1))
            
            # Draw connection
            ax.plot([cat_pos[0], task_x], [cat_pos[1], task_y],
                   color=cat_color, alpha=0.3, linewidth=1)
            
            # Add task label
            # Adjust text position based on angle
            if angle < -np.pi/2 or angle > np.pi/2:
                ha = 'right'
                offset_x = -0.03
            else:
                ha = 'left'
                offset_x = 0.03
                
            ax.text(task_x + offset_x, task_y, task, fontsize=8,
                   ha=ha, va='center', color=cat_color, fontweight='bold')
    
    # Add statistics
    stats_text = (
        'Impact Statistics:\n'
        '• 100+ distinct applications\n'
        '• 50+ languages supported\n'
        '• 1000x faster deployment\n'
        '• Billions of users served daily'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # Add performance indicators
    perf_text = (
        'Performance Gains:\n'
        '• 95%+ accuracy on many tasks\n'
        '• Human-level on some benchmarks\n'
        '• 10x fewer training examples needed\n'
        '• Works out-of-the-box for new tasks'
    )
    ax.text(0.98, 0.98, perf_text, transform=ax.transAxes,
           fontsize=10, va='top', ha='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('../figures/application_ecosystem_map.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/application_ecosystem_map.png', dpi=150, bbox_inches='tight')
    print("Generated: application_ecosystem_map.pdf/png")

def create_performance_timeline():
    """Create timeline showing SOTA improvements"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    fig.suptitle('State-of-the-Art Performance Evolution', fontsize=20, fontweight='bold')
    
    # Timeline data
    years = np.array([2017, 2018, 2018.5, 2019, 2019.5, 2020, 2021, 2022, 2023, 2024])
    
    # GLUE benchmark scores (normalized to 100)
    glue_scores = np.array([65, 75, 82, 85, 88, 90, 92, 94, 96, 97])
    
    # SuperGLUE scores
    superglue_scores = np.array([np.nan, np.nan, 60, 69, 75, 84, 88, 91, 94, 95])
    
    # Model names for annotations
    models = ['', 'BERT', 'RoBERTa', 'XLNet', 'ALBERT', 'T5', 'DeBERTa', 'PaLM', 'GPT-4', 'Claude-3']
    
    # Colors for different eras
    era_colors = {
        'pre': '#E74C3C',
        'early': '#F39C12',
        'mature': '#27AE60',
        'current': '#3498DB'
    }
    
    # --- Top plot: Performance scores ---
    ax1.set_title('Benchmark Performance Over Time', fontsize=16, fontweight='bold')
    
    # Plot GLUE scores
    ax1.plot(years, glue_scores, 'o-', label='GLUE Score', color='#2980B9',
            linewidth=3, markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    # Plot SuperGLUE scores
    valid_sg = ~np.isnan(superglue_scores)
    ax1.plot(years[valid_sg], superglue_scores[valid_sg], 's-', label='SuperGLUE Score',
            color='#8E44AD', linewidth=3, markersize=10, markeredgecolor='black', markeredgewidth=1)
    
    # Add human performance line
    ax1.axhline(y=87, color='red', linestyle='--', alpha=0.5, label='Human Performance')
    ax1.fill_between([2017, 2024], [87, 87], [85, 85], alpha=0.2, color='red')
    
    # Annotate models
    for i, (year, glue, model) in enumerate(zip(years, glue_scores, models)):
        if model:
            ax1.annotate(model, (year, glue), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # Mark eras
    ax1.axvspan(2017, 2018, alpha=0.1, color=era_colors['pre'], label='Pre-BERT Era')
    ax1.axvspan(2018, 2020, alpha=0.1, color=era_colors['early'], label='Early Pre-training')
    ax1.axvspan(2020, 2022, alpha=0.1, color=era_colors['mature'], label='Mature Pre-training')
    ax1.axvspan(2022, 2024, alpha=0.1, color=era_colors['current'], label='Large-scale Era')
    
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Benchmark Score (%)', fontsize=12, fontweight='bold')
    ax1.set_ylim(60, 100)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom plot: Specific task improvements ---
    ax2.set_title('Task-Specific Performance Gains', fontsize=16, fontweight='bold')
    
    tasks = ['Sentiment\nAnalysis', 'Named Entity\nRecognition', 'Question\nAnswering',
             'Text\nSummarization', 'Machine\nTranslation']
    
    # Performance before and after pre-training
    before_scores = [82, 78, 65, 55, 72]
    after_scores = [97, 95, 92, 88, 94]
    
    x_pos = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, before_scores, width, label='Before Pre-training (2017)',
                   color=era_colors['pre'], alpha=0.7, edgecolor='black', linewidth=2)
    bars2 = ax2.bar(x_pos + width/2, after_scores, width, label='With Pre-training (2024)',
                   color=era_colors['current'], alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add improvement percentages
    for i, (before, after) in enumerate(zip(before_scores, after_scores)):
        improvement = ((after - before) / before) * 100
        ax2.text(i, after + 2, f'+{improvement:.0f}%', ha='center', fontsize=10,
                color='green', fontweight='bold')
    
    ax2.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(tasks)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../figures/performance_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../figures/performance_timeline.png', dpi=150, bbox_inches='tight')
    print("Generated: performance_timeline.pdf/png")

# Generate all visualizations
if __name__ == "__main__":
    print("Generating Week 6 Paradigm Shift Visualizations...")
    print("=" * 50)
    
    # Generate the main paradigm shift visualization
    print("Creating paradigm shift impact chart...")
    create_paradigm_shift_visualization()
    
    # Generate knowledge transfer mountain
    print("Creating knowledge transfer mountain...")
    create_knowledge_transfer_mountain()
    
    # Generate MLM animation
    print("Creating MLM interactive animation...")
    create_mlm_interactive_animation()
    
    # Generate scale comparison
    print("Creating scale comparison chart...")
    create_scale_comparison_chart()
    
    # Generate fine-tuning strategy flowchart
    print("Creating fine-tuning strategy flowchart...")
    create_finetuning_strategy_flowchart()
    
    # Generate application ecosystem
    print("Creating application ecosystem map...")
    create_application_ecosystem_map()
    
    # Generate performance timeline
    print("Creating performance timeline...")
    create_performance_timeline()
    
    print("=" * 50)
    print("All visualizations generated successfully!")
    print("Files saved to ../figures/")