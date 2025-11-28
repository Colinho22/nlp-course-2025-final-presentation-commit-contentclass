"""
Generate Week 1 overview visualizations
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.patches import Rectangle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Educational colors
COLOR_CURRENT = '#FF6B6B'  # Red
COLOR_CONTEXT = '#4ECDC4'  # Teal  
COLOR_PREDICT = '#95E77E'  # Green
COLOR_NEUTRAL = '#E0E0E0'  # Gray

def create_learning_objectives():
    """Create learning objectives visualization"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define objectives with difficulty levels
    objectives = [
        {'text': 'Understand\nProbability\nBasics', 'level': 1, 'color': COLOR_CONTEXT},
        {'text': 'Compute\nN-gram\nProbabilities', 'level': 2, 'color': COLOR_CONTEXT},
        {'text': 'Build\nLanguage\nModels', 'level': 3, 'color': COLOR_CURRENT},
        {'text': 'Handle\nSparse\nData', 'level': 2, 'color': COLOR_PREDICT},
        {'text': 'Evaluate\nPerplexity', 'level': 3, 'color': COLOR_PREDICT},
        {'text': 'Generate\nText', 'level': 4, 'color': COLOR_CURRENT}
    ]
    
    # Create learning path
    x_positions = np.linspace(0.1, 0.9, len(objectives))
    
    for i, (obj, x) in enumerate(zip(objectives, x_positions)):
        # Draw difficulty bars
        for j in range(obj['level']):
            rect = Rectangle((x - 0.04, 0.15 + j * 0.08), 0.08, 0.06,
                           facecolor=obj['color'], alpha=0.7,
                           edgecolor='white', linewidth=2)
            ax.add_patch(rect)
        
        # Draw objective circle
        y = 0.6
        circle = Circle((x, y), 0.08, facecolor=obj['color'], 
                       edgecolor='white', linewidth=3, alpha=0.9)
        ax.add_patch(circle)
        
        # Add text
        ax.text(x, y, obj['text'], fontsize=9, ha='center', 
               va='center', fontweight='bold', color='white')
        
        # Add arrow to next
        if i < len(objectives) - 1:
            ax.annotate('', xy=(x_positions[i+1] - 0.08, y),
                       xytext=(x + 0.08, y),
                       arrowprops=dict(arrowstyle='->', lw=2, 
                                     color=COLOR_NEUTRAL, alpha=0.5))
    
    # Add progression labels
    ax.text(0.5, 0.9, 'Week 1 Learning Journey', fontsize=16,
           fontweight='bold', ha='center', color='darkblue')
    
    ax.text(0.1, 0.05, 'Foundation', fontsize=11, ha='center',
           color=COLOR_CONTEXT, fontweight='bold')
    ax.text(0.5, 0.05, 'Core Skills', fontsize=11, ha='center',
           color=COLOR_CURRENT, fontweight='bold')
    ax.text(0.9, 0.05, 'Applications', fontsize=11, ha='center',
           color=COLOR_PREDICT, fontweight='bold')
    
    # Add difficulty legend
    ax.text(0.02, 0.45, 'Difficulty:', fontsize=10, fontweight='bold')
    for i in range(1, 5):
        for j in range(i):
            rect = Rectangle((0.02 + j * 0.015, 0.38 - i * 0.08), 
                           0.012, 0.05,
                           facecolor='gray', alpha=0.5)
            ax.add_patch(rect)
        level_text = ['Basic', 'Intermediate', 'Advanced', 'Expert'][i-1]
        ax.text(0.08, 0.40 - i * 0.08, level_text, fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/week1_learning_objectives.pdf', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_ngram_visualization():
    """Create n-gram concept visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: N-gram extraction
    text = "The quick brown fox jumps"
    words = text.split()
    
    # Unigrams
    y_uni = 0.8
    ax1.text(0.05, y_uni, 'Unigrams:', fontsize=11, fontweight='bold')
    for i, word in enumerate(words):
        x = 0.25 + i * 0.15
        rect = FancyBboxPatch((x - 0.06, y_uni - 0.05), 0.12, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_CONTEXT, alpha=0.7,
                              edgecolor='white', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y_uni, word, fontsize=9, ha='center', va='center',
                color='white', fontweight='bold')
    
    # Bigrams
    y_bi = 0.5
    ax1.text(0.05, y_bi, 'Bigrams:', fontsize=11, fontweight='bold')
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    for i, bigram in enumerate(bigrams):
        x = 0.25 + i * 0.2
        rect = FancyBboxPatch((x - 0.08, y_bi - 0.05), 0.16, 0.08,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_CURRENT, alpha=0.7,
                              edgecolor='white', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y_bi, bigram, fontsize=8, ha='center', va='center',
                color='white', fontweight='bold')
    
    # Trigrams
    y_tri = 0.2
    ax1.text(0.05, y_tri, 'Trigrams:', fontsize=11, fontweight='bold')
    trigrams = [f"{words[i]} {words[i+1]}\n{words[i+2]}" 
                for i in range(len(words)-2)]
    for i, trigram in enumerate(trigrams):
        x = 0.3 + i * 0.25
        rect = FancyBboxPatch((x - 0.1, y_tri - 0.07), 0.2, 0.12,
                              boxstyle="round,pad=0.01",
                              facecolor=COLOR_PREDICT, alpha=0.7,
                              edgecolor='white', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y_tri, trigram, fontsize=7, ha='center', va='center',
                color='white', fontweight='bold')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('N-gram Extraction', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Right: Probability calculation
    # Create probability tree
    ax2.text(0.5, 0.95, 'P(jumps | brown fox)', fontsize=12, 
            fontweight='bold', ha='center')
    
    # Count visualization
    contexts = [
        ('brown fox', 10, COLOR_CONTEXT),
        ('... jumps', 3, COLOR_PREDICT),
        ('... runs', 2, COLOR_NEUTRAL),
        ('... walks', 5, COLOR_NEUTRAL)
    ]
    
    y_start = 0.7
    for i, (text, count, color) in enumerate(contexts):
        y = y_start - i * 0.15
        # Draw bar
        width = count / 10 * 0.6
        rect = Rectangle((0.2, y - 0.03), width, 0.06,
                        facecolor=color, alpha=0.7,
                        edgecolor='white', linewidth=2)
        ax2.add_patch(rect)
        
        # Add text
        ax2.text(0.15, y, text, fontsize=10, ha='right', va='center')
        ax2.text(0.2 + width + 0.02, y, f'{count}', fontsize=10, 
                ha='left', va='center', fontweight='bold')
    
    # Add probability calculation
    ax2.text(0.5, 0.15, 'P(jumps | brown fox) = 3/10 = 0.3',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='yellow', alpha=0.7))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Probability Calculation', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('../figures/week1_ngram_concept.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

def create_applications_chart():
    """Create applications and tools visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Applications pie chart
    applications = ['Spell Check', 'Auto-complete', 'Machine Translation',
                   'Speech Recognition', 'Text Generation']
    sizes = [20, 25, 20, 15, 20]
    colors = [COLOR_CONTEXT, COLOR_CURRENT, COLOR_PREDICT, 
             '#FFD700', '#FF8C00']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=applications, 
                                        colors=colors, autopct='%1.0f%%',
                                        startangle=90)
    
    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax1.set_title('N-gram Applications', fontsize=12, fontweight='bold')
    
    # Right: Complexity comparison
    models = ['Unigram', 'Bigram', 'Trigram', '4-gram', '5-gram']
    memory = [1, 10, 100, 1000, 10000]  # Relative memory
    accuracy = [60, 75, 85, 88, 89]  # Accuracy percentage
    
    x = np.arange(len(models))
    width = 0.35
    
    # Create twin axis
    ax2_twin = ax2.twinx()
    
    # Plot bars
    bars1 = ax2.bar(x - width/2, np.log10(memory), width, 
                   label='Memory (log)', color=COLOR_CURRENT, alpha=0.7)
    bars2 = ax2_twin.bar(x + width/2, accuracy, width,
                        label='Accuracy %', color=COLOR_PREDICT, alpha=0.7)
    
    # Labels and formatting
    ax2.set_xlabel('N-gram Order', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Memory Usage (log scale)', fontsize=10, 
                  color=COLOR_CURRENT, fontweight='bold')
    ax2_twin.set_ylabel('Accuracy (%)', fontsize=10, 
                       color=COLOR_PREDICT, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_title('Memory vs Accuracy Trade-off', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars1, memory):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val}', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color=COLOR_CURRENT)
    
    for bar, val in zip(bars2, accuracy):
        height = bar.get_height()
        ax2_twin.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{val}%', ha='center', va='bottom', fontsize=8,
                     fontweight='bold', color=COLOR_PREDICT)
    
    # Legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/week1_applications.pdf', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == '__main__':
    create_learning_objectives()
    create_ngram_visualization() 
    create_applications_chart()
    print("Week 1 overview visualizations generated successfully!")