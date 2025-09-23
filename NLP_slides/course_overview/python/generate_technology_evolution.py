"""
Generate technology evolution visualization showing NLP breakthroughs over time
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
from scipy.interpolate import make_interp_spline

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_technology_evolution():
    fig = plt.figure(figsize=(14, 8))
    
    # Create 2D plot (main visualization)
    ax = plt.subplot(111)
    
    # Technology milestones with year, performance (relative), and model size
    technologies = [
        {'year': 1990, 'name': 'N-grams', 'performance': 1, 'params': 0.001, 'color': '#808080'},
        {'year': 2003, 'name': 'Neural LM', 'performance': 2, 'params': 0.01, 'color': '#9370DB'},
        {'year': 2013, 'name': 'Word2Vec', 'performance': 5, 'params': 0.1, 'color': '#FF6B6B'},
        {'year': 2014, 'name': 'LSTM', 'performance': 10, 'params': 1, 'color': '#4169E1'},
        {'year': 2017, 'name': 'Transformer', 'performance': 50, 'params': 10, 'color': '#FFD700'},
        {'year': 2018, 'name': 'BERT', 'performance': 100, 'params': 100, 'color': '#32CD32'},
        {'year': 2019, 'name': 'GPT-2', 'performance': 200, 'params': 1500, 'color': '#FF8C00'},
        {'year': 2020, 'name': 'GPT-3', 'performance': 1000, 'params': 175000, 'color': '#DC143C'},
        {'year': 2023, 'name': 'GPT-4', 'performance': 10000, 'params': 1700000, 'color': '#FF1493'},
        {'year': 2024, 'name': 'Claude-3', 'performance': 15000, 'params': 2000000, 'color': '#8B008B'}
    ]
    
    # Extract data
    years = [t['year'] for t in technologies]
    performances = [t['performance'] for t in technologies]
    params = [t['params'] for t in technologies]
    names = [t['name'] for t in technologies]
    colors = [t['color'] for t in technologies]
    
    # Create smooth performance curve
    years_smooth = np.linspace(min(years), max(years), 300)
    spl = make_interp_spline(years, np.log10(performances), k=3)
    performance_smooth = 10 ** spl(years_smooth)
    
    # Plot performance curve
    ax.semilogy(years_smooth, performance_smooth, 'k-', alpha=0.2, linewidth=2)
    ax.fill_between(years_smooth, 0.1, performance_smooth, alpha=0.1, color='blue')
    
    # Plot technology points
    for i, tech in enumerate(technologies):
        # Size based on model parameters (log scale)
        size = np.log10(tech['params'] + 1) * 200 + 100
        
        # Plot main point
        ax.scatter(tech['year'], tech['performance'], s=size, 
                  color=tech['color'], alpha=0.8, edgecolors='white', 
                  linewidth=2, zorder=5)
        
        # Add label with arrow
        if i % 2 == 0:  # Alternate label positions
            y_offset = tech['performance'] * 2
            va = 'bottom'
        else:
            y_offset = tech['performance'] * 0.5
            va = 'top'
        
        ax.annotate(tech['name'], 
                   xy=(tech['year'], tech['performance']),
                   xytext=(tech['year'], y_offset),
                   fontsize=10, fontweight='bold',
                   ha='center', va=va,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor=tech['color'], alpha=0.3),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.2',
                                 color=tech['color'], lw=1.5))
    
    # Add era backgrounds
    eras = [
        {'start': 1985, 'end': 2003, 'name': 'Statistical Era', 'color': '#808080'},
        {'start': 2003, 'end': 2013, 'name': 'Early Neural Era', 'color': '#9370DB'},
        {'start': 2013, 'end': 2017, 'name': 'Embedding Era', 'color': '#FF6B6B'},
        {'start': 2017, 'end': 2024, 'name': 'Transformer Era', 'color': '#FFD700'}
    ]
    
    for era in eras:
        ax.axvspan(era['start'], era['end'], alpha=0.1, color=era['color'])
        ax.text((era['start'] + era['end'])/2, 0.2, era['name'], 
               fontsize=11, ha='center', color=era['color'], 
               fontweight='bold', alpha=0.7)
    
    # Add breakthrough annotations
    breakthroughs = [
        (2013, 10, 'Semantic Understanding\nUnlocked'),
        (2017, 100, 'Attention Revolution\nBegins'),
        (2020, 2000, 'Few-shot Learning\nEmerges'),
        (2023, 20000, 'Human-level\nPerformance')
    ]
    
    for year, perf, text in breakthroughs:
        ax.plot([year, year], [0.1, perf], 'r--', alpha=0.3, linewidth=1)
        ax.text(year, perf * 1.5, text, fontsize=9, ha='center',
               color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # Add parameter size legend
    legend_sizes = [0.001, 1, 1000, 1000000]
    legend_labels = ['1K', '1M', '1B', '1T']
    legend_y = 30000
    
    ax.text(1987, legend_y * 2, 'Model Size:', fontsize=11, fontweight='bold')
    for i, (size, label) in enumerate(zip(legend_sizes, legend_labels)):
        x_pos = 1990 + i * 3
        marker_size = np.log10(size + 1) * 200 + 100
        ax.scatter(x_pos, legend_y, s=marker_size, color='gray', 
                  alpha=0.5, edgecolors='white', linewidth=2)
        ax.text(x_pos, legend_y * 0.5, label, fontsize=9, ha='center')
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Performance (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('The Exponential Evolution of NLP Technology', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(1985, 2025)
    ax.set_ylim(0.1, 50000)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_axisbelow(True)
    
    # Add performance multiplier text
    ax.text(2024, 25000, '10,000x\nimprovement\nin 30 years!', 
           fontsize=14, fontweight='bold', color='green',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
           ha='center')
    
    plt.tight_layout()
    plt.savefig('../figures/technology_evolution.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == '__main__':
    create_technology_evolution()
    print("Technology evolution visualization generated successfully!")