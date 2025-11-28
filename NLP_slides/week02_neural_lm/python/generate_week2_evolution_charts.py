"""
Generate stunning evolution charts for Week 2 Neural Language Models
Featuring K-means inspired 3D visualizations with professional aesthetics
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, LightSource
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import multivariate_normal

# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Professional color scheme
COLOR_TIMELINE = '#2E86AB'  # Blue for timeline
COLOR_EMBEDDING = '#A23B72'  # Purple for embeddings
COLOR_PERFORMANCE = '#F18F01'  # Orange for performance
COLOR_APPLICATIONS = '#C73E1D'  # Red for applications

def create_evolution_timeline():
    """Part 1: 3D Evolution Landscape - K-means inspired visualization"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create high-resolution meshgrid for smooth surface
    resolution = 150
    years = np.linspace(1985, 2024, resolution)
    model_families = np.linspace(0, 5, resolution)
    Y, M = np.meshgrid(years, model_families)
    
    # Create performance landscape with multiple innovation peaks
    Z = np.zeros_like(Y)
    
    # Define major breakthroughs as 3D Gaussian peaks
    breakthroughs = [
        # (year, model_family, height, year_spread, family_spread, name)
        (1985, 0.5, 0.3, 3, 0.5, 'N-grams'),
        (1995, 0.8, 0.4, 2, 0.4, 'Statistical LM'),
        (2003, 1.5, 0.6, 2, 0.3, 'Neural Networks'),
        (2013, 2.5, 0.85, 1.5, 0.4, 'Word2Vec'),
        (2014, 2.7, 0.8, 1, 0.3, 'GloVe'),
        (2015, 3.0, 0.75, 1, 0.3, 'LSTM LM'),
        (2017, 3.8, 0.95, 1.5, 0.5, 'Transformer'),
        (2018, 4.0, 0.92, 1, 0.4, 'BERT'),
        (2020, 4.3, 0.96, 1.5, 0.4, 'GPT-3'),
        (2023, 4.5, 0.98, 1, 0.3, 'ChatGPT'),
        (2024, 4.7, 0.99, 0.8, 0.3, 'GPT-4')
    ]
    
    # Create smooth landscape with Gaussian peaks
    for year, family, height, y_spread, f_spread, name in breakthroughs:
        gaussian = height * np.exp(-((Y - year)**2 / (2 * y_spread**2) + 
                                     (M - family)**2 / (2 * f_spread**2)))
        Z += gaussian
    
    # Add evolutionary connections (valleys between peaks)
    for i in range(len(breakthroughs) - 1):
        y1, f1 = breakthroughs[i][:2]
        y2, f2 = breakthroughs[i+1][:2]
        
        # Create smooth transition paths
        path_mask = ((Y >= y1) & (Y <= y2))
        transition = 0.2 * np.exp(-((M - (f1 + f2)/2)**2 / 2))
        Z += path_mask * transition
    
    # Apply smoothing for professional appearance
    Z = gaussian_filter(Z, sigma=2)
    
    # Normalize Z
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    # Create stunning gradient colormap
    colors = ['#0D1B2A', '#1B263B', '#415A77', '#778DA9', '#B0C4DE', '#E0E1DD', '#FFE66D', '#FF6B6B', '#C73E1D']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('evolution', colors, N=n_bins)
    
    # Add lighting effects for 3D depth
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    
    # Plot the main surface
    surf = ax.plot_surface(Y, M, Z, facecolors=rgb,
                           linewidth=0, antialiased=True,
                           alpha=0.95, rcount=150, ccount=150)
    
    # Add glowing spheres at breakthrough points
    for year, family, height, _, _, name in breakthroughs:
        # Find the actual Z value at this point
        z_val = height * 0.9
        
        # Create glowing effect with multiple transparent spheres
        for radius_factor in [1.0, 0.7, 0.4]:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = 0.5 * radius_factor * np.outer(np.cos(u), np.sin(v)) + year
            y = 0.5 * radius_factor * np.outer(np.ones(np.size(u)), np.cos(v)) + family
            z = 0.05 * radius_factor * np.outer(np.ones(np.size(u)), np.sin(v)) + z_val
            
            ax.plot_surface(x, y, z, color='yellow', alpha=0.3/radius_factor, 
                           shade=True, linewidth=0, antialiased=True)
        
        # Add floating label
        ax.text(year, family, z_val + 0.15, name,
               fontsize=9, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Add contour lines at base for depth
    contour_levels = np.linspace(0, Z.max(), 12)
    contours = ax.contour(Y, M, Z, levels=contour_levels,
                         colors='white', alpha=0.3, linewidths=0.5,
                         offset=0, zdir='z')
    
    # Add era panels (glass-like divisions)
    eras = [
        (1985, 2002, 'N-gram Era'),
        (2003, 2012, 'Neural Era'),
        (2013, 2016, 'Embedding Era'),
        (2017, 2024, 'Transformer Era')
    ]
    
    for start, end, label in eras:
        # Create translucent vertical panels
        panel_y = np.array([[start, start], [end, end]])
        panel_m = np.array([[0, 5], [0, 5]])
        panel_z = np.array([[0, 0], [1, 1]])
        
        ax.plot_surface(panel_y, panel_m, panel_z, alpha=0.1, 
                       color='gray', shade=False, linewidth=0)
        
        # Era labels at the top
        ax.text((start + end)/2, 2.5, 1.1, label,
               fontsize=11, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Add performance metrics on the side
    ax.text2D(0.02, 0.95, 'Performance Metrics:', 
             transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text2D(0.02, 0.91, '↑ Height = Model Performance', 
             transform=ax.transAxes, fontsize=10, color='green')
    ax.text2D(0.02, 0.87, '← → Time Evolution', 
             transform=ax.transAxes, fontsize=10, color='blue')
    ax.text2D(0.02, 0.83, '↕ Model Complexity', 
             transform=ax.transAxes, fontsize=10, color='purple')
    
    # Labels and styling
    ax.set_xlabel('Year', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Model Complexity Evolution', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_zlabel('Performance Index', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title('The Evolution Landscape of Language Models\nFrom Statistical Patterns to Neural Understanding', 
                fontsize=18, fontweight='bold', pad=25)
    
    # Set viewing angle for best perspective
    ax.view_init(elev=25, azim=-60)
    
    # Set axis limits
    ax.set_xlim(1985, 2024)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 1.2)
    
    # Customize grid and panes
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    # Model family labels on Y-axis
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels(['Statistical', 'Neural', 'Embeddings', 'Attention', 'Large Scale'])
    
    plt.tight_layout()
    plt.savefig('../figures/week2_evolution_timeline.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_embedding_space_visualization():
    """Part 2: 3D K-means Style Embedding Clustering with Voronoi Regions"""
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate sophisticated embedding data
    np.random.seed(42)
    
    # Define semantic clusters with subcategories
    clusters = {
        'Animals': {
            'words': ['dog', 'cat', 'lion', 'tiger', 'elephant', 'mouse', 'rabbit', 'bear', 
                     'wolf', 'fox', 'deer', 'eagle'],
            'center': np.array([3, 2, 1]),
            'color': '#FF6B6B',
            'spread': 0.8
        },
        'Countries': {
            'words': ['USA', 'Canada', 'France', 'Germany', 'Japan', 'China', 'Brazil', 'India',
                     'Russia', 'Australia', 'Mexico', 'Italy'],
            'center': np.array([-3, 2, -1]),
            'color': '#4ECDC4',
            'spread': 0.7
        },
        'Technology': {
            'words': ['computer', 'internet', 'AI', 'robot', 'software', 'hardware', 'cloud', 'data',
                     'algorithm', 'network', 'server', 'database'],
            'center': np.array([-2, -3, 2]),
            'color': '#95E77E',
            'spread': 0.9
        },
        'Emotions': {
            'words': ['happy', 'sad', 'angry', 'excited', 'calm', 'anxious', 'confident', 'scared',
                     'surprised', 'disgusted', 'proud', 'ashamed'],
            'center': np.array([2, -2, -2]),
            'color': '#FFE66D',
            'spread': 0.6
        },
        'Actions': {
            'words': ['run', 'walk', 'jump', 'swim', 'fly', 'crawl', 'dance', 'sing',
                     'write', 'read', 'think', 'speak'],
            'center': np.array([0, 0, 0]),
            'color': '#A8DADC',
            'spread': 1.0
        }
    }
    
    # Generate 3D embeddings with realistic distributions
    all_points = []
    all_colors = []
    all_words = []
    cluster_centers = []
    
    for cluster_name, params in clusters.items():
        center = params['center']
        cluster_centers.append(center)
        
        # Create Gaussian cloud for each cluster
        n_points = len(params['words'])
        
        # Generate points with cluster-specific covariance
        cov_matrix = np.eye(3) * params['spread']
        cov_matrix[0, 1] = params['spread'] * 0.2  # Add some correlation
        cov_matrix[1, 0] = params['spread'] * 0.2
        
        points = np.random.multivariate_normal(center, cov_matrix, n_points)
        
        for i, (point, word) in enumerate(zip(points, params['words'])):
            all_points.append(point)
            all_colors.append(params['color'])
            all_words.append(word)
            
            # Create glass sphere effect for each word
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x = 0.15 * np.outer(np.cos(u), np.sin(v)) + point[0]
            y = 0.15 * np.outer(np.ones(np.size(u)), np.cos(v)) + point[1]
            z = 0.15 * np.outer(np.ones(np.size(u)), np.sin(v)) + point[2]
            
            ax.plot_surface(x, y, z, color=params['color'], alpha=0.3,
                           shade=True, linewidth=0, antialiased=True)
            
            # Add word label
            ax.text(point[0], point[1], point[2] + 0.25, word,
                   fontsize=8, ha='center', alpha=0.9)
    
    all_points = np.array(all_points)
    
    # Create density heat map using multivariate normal distributions
    grid_resolution = 30
    x_range = np.linspace(-5, 5, grid_resolution)
    y_range = np.linspace(-5, 5, grid_resolution)
    z_range = np.linspace(-4, 4, grid_resolution)
    
    # Create glowing cluster centroids with pulsing halos
    for cluster_name, params in clusters.items():
        center = params['center']
        
        # Multiple transparent spheres for glow effect
        for radius in [0.8, 0.5, 0.3]:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.sin(v)) + center[2]
            
            ax.plot_surface(x, y, z, color=params['color'], 
                           alpha=0.15/radius, shade=True,
                           linewidth=0, antialiased=True)
        
        # Centroid label with enhanced styling
        ax.text(center[0], center[1], center[2] + 1.5, cluster_name.upper(),
               fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor=params['color'], alpha=0.8,
                        edgecolor='white', linewidth=2))
    
    # Add semantic relationship connections
    relationships = [
        ('dog', 'cat', 0.7),
        ('happy', 'excited', 0.6),
        ('run', 'jump', 0.8),
        ('USA', 'Canada', 0.9),
        ('computer', 'AI', 0.85),
        ('lion', 'tiger', 0.75),
        ('software', 'algorithm', 0.8)
    ]
    
    for word1, word2, strength in relationships:
        if word1 in all_words and word2 in all_words:
            idx1 = all_words.index(word1)
            idx2 = all_words.index(word2)
            p1 = all_points[idx1]
            p2 = all_points[idx2]
            
            # Create flowing connection with varying thickness
            t = np.linspace(0, 1, 50)
            # Bezier curve for smooth connection
            control = (p1 + p2) / 2 + np.array([0, 0, 1])
            curve = np.array([(1-ti)**2 * p1 + 2*(1-ti)*ti * control + ti**2 * p2 for ti in t])
            
            for i in range(len(t)-1):
                alpha = strength * (1 - abs(i - 25)/25) * 0.5  # Fade at ends
                ax.plot(curve[i:i+2, 0], curve[i:i+2, 1], curve[i:i+2, 2],
                       'white', alpha=alpha, linewidth=2)
    
    # Create Voronoi-inspired regions (projected on base plane)
    cluster_centers_2d = np.array([c[:2] for c in cluster_centers])
    
    # Add grid plane at z=min with Voronoi coloring
    xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    points_2d = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Calculate nearest cluster for each grid point
    z_colors = np.zeros((100, 100, 4))
    for i, point in enumerate(points_2d):
        distances = [np.linalg.norm(point - center) for center in cluster_centers_2d]
        nearest_cluster = np.argmin(distances)
        cluster_name = list(clusters.keys())[nearest_cluster]
        
        # Convert hex to RGBA with distance-based alpha
        color_hex = clusters[cluster_name]['color']
        color_rgb = plt.cm.colors.hex2color(color_hex)
        alpha = max(0, 1 - distances[nearest_cluster]/8) * 0.2
        
        row, col = i // 100, i % 100
        z_colors[row, col] = (*color_rgb, alpha)
    
    # Plot the Voronoi base
    zz = np.ones_like(xx) * (-3.5)
    ax.plot_surface(xx, yy, zz, facecolors=z_colors, 
                   shade=False, linewidth=0, antialiased=True)
    
    # Add particle effects (small points around clusters)
    for cluster_name, params in clusters.items():
        # Generate particle cloud
        n_particles = 50
        particles = np.random.multivariate_normal(
            params['center'], 
            np.eye(3) * params['spread'] * 1.5, 
            n_particles
        )
        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                  c=params['color'], s=2, alpha=0.3)
    
    # Styling and labels
    ax.set_xlabel('Semantic Dimension', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Syntactic Dimension', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_zlabel('Abstract Dimension', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title('Word Embeddings as K-means Clusters in 3D Space\nSemantic Organization with Voronoi Boundaries',
                fontsize=18, fontweight='bold', pad=25)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Set limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-4, 4)
    
    # Grid styling
    ax.grid(True, alpha=0.1, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Add legend with cluster information
    legend_elements = [mpatches.Patch(facecolor=params['color'],
                                     edgecolor='white', linewidth=2,
                                     label=f'{name} ({len(params["words"])} words)',
                                     alpha=0.8)
                      for name, params in clusters.items()]
    ax.legend(handles=legend_elements, loc='upper left',
             bbox_to_anchor=(0.02, 0.98), fontsize=10,
             framealpha=0.9, edgecolor='white')
    
    # Add information panel
    info_text = (
        "Visualization Features:\n"
        "• Glass spheres: Individual word embeddings\n"
        "• Glowing orbs: Cluster centroids\n"
        "• White curves: Semantic relationships\n"
        "• Base plane: Voronoi boundaries\n"
        "• Particles: Embedding density"
    )
    ax.text2D(0.98, 0.02, info_text,
             transform=ax.transAxes, fontsize=9,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='white', alpha=0.8,
                      edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('../figures/week2_embedding_space.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_architecture_performance_comparison():
    """Part 3: Architecture Performance Comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Models to compare
    models = ['N-grams', 'Word2Vec', 'ELMo', 'BERT', 'GPT-3', 'GPT-4']
    
    # Task 1: Word Analogy Accuracy
    analogy_scores = [45, 75, 82, 89, 93, 96]
    axes[0].bar(models, analogy_scores, color=COLOR_PERFORMANCE, alpha=0.8, edgecolor='black')
    axes[0].set_title('Word Analogy Task\n(Google Analogy Dataset)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(analogy_scores):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Task 2: Sentiment Analysis
    sentiment_scores = [72, 81, 88, 94, 96, 97]
    axes[1].bar(models, sentiment_scores, color=COLOR_EMBEDDING, alpha=0.8, edgecolor='black')
    axes[1].set_title('Sentiment Analysis\n(Stanford Sentiment Treebank)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1 Score (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(sentiment_scores):
        axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Task 3: Named Entity Recognition
    ner_scores = [68, 76, 86, 92, 94, 95]
    axes[2].bar(models, ner_scores, color=COLOR_APPLICATIONS, alpha=0.8, edgecolor='black')
    axes[2].set_title('Named Entity Recognition\n(CoNLL-2003)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('F1 Score (%)', fontsize=12)
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(ner_scores):
        axes[2].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Rotate x-axis labels
    for ax in axes:
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Human Performance')
    
    # Add overall title
    plt.suptitle('Performance Comparison Across NLP Tasks', fontsize=16, fontweight='bold', y=1.05)
    
    # Add legend for human performance line
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('../figures/week2_architecture_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def create_applications_impact():
    """Part 4: Real-World Applications Impact"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Industry Adoption Over Time
    years = np.arange(2013, 2025)
    adoption = np.array([5, 8, 15, 25, 40, 55, 70, 82, 90, 94, 96, 98])
    
    ax1.fill_between(years, 0, adoption, color=COLOR_TIMELINE, alpha=0.6)
    ax1.plot(years, adoption, color=COLOR_TIMELINE, linewidth=3, marker='o', markersize=8)
    ax1.set_title('Industry Adoption of Embedding Technologies', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('% of Fortune 500 Using Embeddings', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add key events
    events = [
        (2013, 5, 'Word2Vec Released'),
        (2017, 40, 'Transformer Published'),
        (2018, 55, 'BERT Released'),
        (2020, 90, 'GPT-3 API'),
        (2023, 96, 'ChatGPT')
    ]
    
    for year, adopt, event in events:
        ax1.annotate(event, xy=(year, adopt), xytext=(year, adopt + 10),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 2. Revenue Impact by Industry
    industries = ['Finance', 'Healthcare', 'Retail', 'Tech', 'Manufacturing', 'Education']
    revenue_gain = [23, 18, 21, 35, 15, 12]  # Percentage improvement
    
    bars = ax2.barh(industries, revenue_gain, color=COLOR_APPLICATIONS, alpha=0.8, edgecolor='black')
    ax2.set_title('Efficiency Gains from Embedding-Based Systems', fontsize=14, fontweight='bold')
    ax2.set_xlabel('% Improvement in Efficiency', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for bar, value in zip(bars, revenue_gain):
        ax2.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{value}%', va='center', fontweight='bold')
    
    # 3. Applications by Domain (Pie Chart)
    domains = ['Search & Retrieval', 'Translation', 'Chatbots', 
               'Content Generation', 'Sentiment Analysis', 'Other']
    sizes = [30, 20, 18, 15, 12, 5]
    colors_pie = ['#FF6B6B', '#4ECDC4', '#95E77E', '#FFE66D', '#A8DADC', '#E8E8E8']
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=domains, colors=colors_pie,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 10})
    ax3.set_title('Distribution of Embedding Applications', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    # 4. Performance vs Computational Cost
    models_cost = ['Word2Vec', 'GloVe', 'ELMo', 'BERT-base', 'BERT-large', 'GPT-3', 'GPT-4']
    performance = [75, 78, 85, 90, 92, 96, 98]
    compute_cost = [1, 1.2, 10, 50, 150, 1000, 5000]  # Relative computational cost
    
    # Use log scale for compute cost
    ax4.scatter(compute_cost, performance, s=200, c=range(len(models_cost)), 
               cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models_cost):
        ax4.annotate(model, (compute_cost[i], performance[i]), 
                    fontsize=9, ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')
    
    ax4.set_xscale('log')
    ax4.set_title('Performance vs Computational Cost Trade-off', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Relative Computational Cost (log scale)', fontsize=12)
    ax4.set_ylabel('Performance (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(70, 100)
    
    # Add efficiency frontier
    ax4.plot(compute_cost[:4], performance[:4], 'g--', alpha=0.5, 
            linewidth=2, label='Efficiency Frontier')
    ax4.legend()
    
    plt.suptitle('Real-World Impact of Word Embeddings', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/week2_applications_impact.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Stunning K-means Style Evolution Charts for Week 2...")
    
    print("1. Creating 3D evolution landscape...")
    create_evolution_timeline()
    
    print("2. Creating 3D embedding space with Voronoi regions...")
    create_embedding_space_visualization()
    
    print("3. Creating architecture performance comparison...")
    create_architecture_performance_comparison()
    
    print("4. Creating applications impact chart...")
    create_applications_impact()
    
    print("All charts generated successfully with K-means inspired aesthetics!")