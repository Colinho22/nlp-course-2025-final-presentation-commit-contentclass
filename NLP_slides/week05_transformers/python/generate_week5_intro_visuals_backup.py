import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from matplotlib.colors import LightSource
import matplotlib.patches as mpatches
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import multivariate_normal

# Set style for professional visuals
plt.style.use('seaborn-v0_8-whitegrid')

def create_attention_landscape():
    """Create a professional 3D surface plot showing attention as a landscape"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create high-resolution meshgrid
    resolution = 100
    x = np.linspace(0, 10, resolution)
    y = np.linspace(0, 10, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create attention landscape with multiple attention peaks
    # Simulating attention between different word pairs
    Z = np.zeros_like(X)
    
    # Major attention peaks (strong relationships)
    peaks = [
        (2, 3, 2.5, 0.8, 0.8),   # Subject-Verb
        (3, 7, 2.2, 0.6, 0.9),   # Verb-Object
        (5, 5, 1.8, 1.2, 1.2),   # Modifier-Noun
        (7, 2, 2.0, 0.7, 0.7),   # Pronoun-Reference
        (8, 8, 2.3, 0.5, 0.5),   # Adjective-Noun
        (1, 8, 1.5, 0.9, 0.6),   # Long-range dependency
    ]
    
    for px, py, height, sx, sy in peaks:
        gaussian = height * np.exp(-((X - px)**2 / (2 * sx**2) + (Y - py)**2 / (2 * sy**2)))
        Z += gaussian
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.02, Z.shape)
    Z += noise
    
    # Apply Gaussian smoothing for professional appearance
    Z = gaussian_filter(Z, sigma=1.5)
    
    # Normalize Z to [0, 1]
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    # Create custom colormap (similar to viridis but more vibrant)
    colors = ['#440154', '#31688e', '#35b779', '#fde725']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
    
    # Create light source for shading
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=cmap, vert_exag=0.1, blend_mode='soft')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, facecolors=rgb, 
                           linewidth=0, antialiased=True, 
                           alpha=0.95, rcount=100, ccount=100)
    
    # Add contour lines at the base
    contour_levels = np.linspace(0, Z.max(), 8)
    contours = ax.contour(X, Y, Z, levels=contour_levels, 
                          colors='black', alpha=0.2, linewidths=0.5,
                          offset=0, zdir='z')
    
    # Add contour projections on walls
    ax.contour(X, Y, Z, levels=5, colors='gray', alpha=0.3, 
               linewidths=0.5, offset=10, zdir='y')
    ax.contour(X, Y, Z, levels=5, colors='gray', alpha=0.3, 
               linewidths=0.5, offset=10, zdir='x')
    
    # Labels and styling
    ax.set_xlabel('Query Position in Sequence', fontsize=12, labelpad=10)
    ax.set_ylabel('Key Position in Sequence', fontsize=12, labelpad=10)
    ax.set_zlabel('Attention Weight', fontsize=12, labelpad=10)
    ax.set_title('The Attention Landscape: How Transformers See Relationships', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set viewing angle for best perspective
    ax.view_init(elev=25, azim=45)
    
    # Set limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 1)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Add colorbar
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(Z)
    cbar = plt.colorbar(m, ax=ax, pad=0.1, shrink=0.7, aspect=20)
    cbar.set_label('Attention Intensity', rotation=270, labelpad=20, fontsize=11)
    
    # Add text annotations for peaks
    peak_labels = ['Subject-Verb', 'Verb-Object', 'Modifier', 'Reference', 'Adjective', 'Long-range']
    for i, (px, py, height, _, _) in enumerate(peaks[:6]):
        if i < len(peak_labels):
            z_pos = Z[int(py * 10), int(px * 10)] + 0.05
            ax.text(px, py, z_pos, peak_labels[i], fontsize=9, 
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('../figures/attention_revolution_3d.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_embedding_space_clustering():
    """Create professional 3D clustering visualization for embeddings"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate synthetic embedding data with clear clusters
    np.random.seed(42)
    n_points_per_cluster = 50
    
    # Define semantic clusters with meaningful labels
    clusters = {
        'Animals': {'center': [2, 2, 2], 'cov': 0.3, 'color': '#FF6B6B'},
        'Technology': {'center': [-2, 2, -1], 'cov': 0.4, 'color': '#4ECDC4'},
        'Emotions': {'center': [2, -2, 0], 'cov': 0.35, 'color': '#95E77E'},
        'Actions': {'center': [-2, -2, 1], 'cov': 0.3, 'color': '#FFE66D'},
        'Objects': {'center': [0, 0, -2], 'cov': 0.45, 'color': '#C9B1FF'},
        'Places': {'center': [0, 3, 0], 'cov': 0.4, 'color': '#FF9999'}
    }
    
    all_points = []
    all_colors = []
    all_sizes = []
    
    for cluster_name, params in clusters.items():
        # Generate points for this cluster
        cov_matrix = np.eye(3) * params['cov']
        points = np.random.multivariate_normal(params['center'], cov_matrix, n_points_per_cluster)
        all_points.append(points)
        
        # Vary sizes based on distance from center
        center = np.array(params['center'])
        distances = np.linalg.norm(points - center, axis=1)
        sizes = 100 * np.exp(-distances / 2)  # Larger points near center
        all_sizes.extend(sizes)
        
        # Colors with gradient based on distance
        for i, point in enumerate(points):
            all_colors.append(params['color'])
    
    all_points = np.vstack(all_points)
    
    # Create background density field
    x_range = np.linspace(-4, 4, 40)
    y_range = np.linspace(-4, 4, 40)
    z_range = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate density at z=0 plane
    Z_density = np.zeros_like(X)
    for cluster_name, params in clusters.items():
        rv = multivariate_normal(params['center'][:2], [[params['cov'], 0], [0, params['cov']]])
        pos = np.dstack((X, Y))
        Z_density += rv.pdf(pos) * 0.3
    
    # Apply Gaussian filter for smoothness
    Z_density = gaussian_filter(Z_density, sigma=1)
    
    # Plot density surface at bottom
    surf = ax.plot_surface(X, Y, np.ones_like(X) * -3, facecolors=plt.cm.viridis(Z_density / Z_density.max()),
                           alpha=0.3, linewidth=0, antialiased=True)
    
    # Add contour lines on the bottom
    ax.contour(X, Y, np.ones_like(X) * -3 + 0.01, 
               Z_density, levels=5, colors='gray', alpha=0.3, linewidths=1)
    
    # Plot the points with varying sizes and alpha
    for i, point in enumerate(all_points):
        ax.scatter(point[0], point[1], point[2], 
                  c=all_colors[i], s=all_sizes[i], 
                  alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Add cluster centers with labels
    for cluster_name, params in clusters.items():
        center = params['center']
        # Large marker for cluster center
        ax.scatter(center[0], center[1], center[2], 
                  c=params['color'], s=500, alpha=1,
                  edgecolors='black', linewidth=2, marker='*')
        
        # Add label
        ax.text(center[0], center[1], center[2] + 0.5, cluster_name,
               fontsize=11, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', alpha=0.8, edgecolor=params['color']))
    
    # Add connecting lines between related clusters
    connections = [
        ('Animals', 'Actions', 0.3),
        ('Technology', 'Objects', 0.4),
        ('Emotions', 'Actions', 0.5),
        ('Places', 'Objects', 0.3)
    ]
    
    for cluster1, cluster2, alpha in connections:
        p1 = clusters[cluster1]['center']
        p2 = clusters[cluster2]['center']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
               'gray', alpha=alpha, linewidth=1, linestyle='--')
    
    # Styling
    ax.set_xlabel('Dimension 1 (Semantic)', fontsize=12, labelpad=10)
    ax.set_ylabel('Dimension 2 (Syntactic)', fontsize=12, labelpad=10)
    ax.set_zlabel('Dimension 3 (Abstract)', fontsize=12, labelpad=10)
    ax.set_title('Token Embeddings in 3D Space: Semantic Clustering', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Set limits
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-3, 3)
    
    # Grid and pane styling
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=params['color'], 
                                     edgecolor='black', 
                                     label=name, alpha=0.7)
                      for name, params in clusters.items()]
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    # Add annotation about parallel processing
    ax.text2D(0.98, 0.02, 'All tokens processed simultaneously\nNo sequential bottleneck', 
             transform=ax.transAxes, fontsize=11,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('../figures/paradigm_shift_3d.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Generating professional 3D visualizations for Week 5...")
    
    # Generate the visualizations
    print("Creating attention landscape visualization...")
    create_attention_landscape()
    print("Created: attention_revolution_3d.pdf")
    
    print("Creating embedding space clustering visualization...")
    create_embedding_space_clustering()
    print("Created: paradigm_shift_3d.pdf")
    
    print("All professional visualizations generated successfully!")