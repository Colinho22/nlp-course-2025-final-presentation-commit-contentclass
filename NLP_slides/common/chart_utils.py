"""
Chart Generation Utilities for NLP Course
Optimal Readability Color Scheme and Standards
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Dict, Any

# ====================================
# OPTIMAL READABILITY COLOR PALETTE
# ====================================
COLORS = {
    'black': '#000000',          # Pure Black - main text
    'blue_accent': '#003D7A',    # Deep Blue - primary accent
    'gray': '#4A4A4A',           # Dark Gray - secondary text
    'light_gray': '#E5E5E5',     # Light gray for grid
    'white': '#FFFFFF',          # Pure white backgrounds
    # Chart colors (colorblind-safe)
    'chart1': '#0066CC',         # Strong Blue
    'chart2': '#FF8800',         # Orange
    'chart3': '#00A0A0',         # Teal
    'chart4': '#8B4789',         # Purple
    'success': '#228B22',        # Dark Green
    'warning': '#CC0000',        # Dark Red
}

# Chart color sequence for multiple series
CHART_COLORS = [COLORS['chart1'], COLORS['chart2'], COLORS['chart3'],
                COLORS['chart4'], COLORS['success'], COLORS['warning']]

# ====================================
# MATPLOTLIB CONFIGURATION
# ====================================
def setup_plotting_style():
    """Configure matplotlib for optimal readability"""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = COLORS['black']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = COLORS['light_gray']
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['text.color'] = COLORS['black']
    plt.rcParams['axes.labelcolor'] = COLORS['black']
    plt.rcParams['xtick.color'] = COLORS['black']
    plt.rcParams['ytick.color'] = COLORS['black']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = COLORS['black']
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# ====================================
# CHART CREATION FUNCTIONS
# ====================================

def create_figure(figsize: Tuple[float, float] = (10, 6),
                 title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with optimal readability settings"""
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    return fig, ax

def create_multi_panel_figure(rows: int, cols: int,
                            figsize: Tuple[float, float] = (12, 8),
                            title: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """Create a multi-panel figure"""
    setup_plotting_style()
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # Clean up all axes
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    return fig, axes

def add_value_labels(ax: plt.Axes, bars, format_str: str = '{:.1f}',
                    offset: float = 1, fontweight: str = 'bold'):
    """Add value labels to bar charts"""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                format_str.format(height),
                ha='center', va='bottom',
                fontweight=fontweight, fontsize=9)

def create_comparison_bars(ax: plt.Axes,
                          categories: List[str],
                          data_series: Dict[str, List[float]],
                          ylabel: str = 'Value',
                          title: str = 'Comparison'):
    """Create grouped bar chart for comparisons"""
    x = np.arange(len(categories))
    width = 0.8 / len(data_series)

    for i, (label, values) in enumerate(data_series.items()):
        offset = (i - len(data_series)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=label, color=CHART_COLORS[i % len(CHART_COLORS)])
        add_value_labels(ax, bars)

    ax.set_xlabel('Category', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3)

def create_line_plot(ax: plt.Axes,
                    x_data: np.ndarray,
                    y_series: Dict[str, np.ndarray],
                    xlabel: str = 'X',
                    ylabel: str = 'Y',
                    title: str = 'Line Plot'):
    """Create line plot with multiple series"""
    for i, (label, y_data) in enumerate(y_series.items()):
        ax.plot(x_data, y_data, label=label,
               color=CHART_COLORS[i % len(CHART_COLORS)],
               linewidth=2, marker='o', markersize=4)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)

def create_heatmap(ax: plt.Axes,
                  data: np.ndarray,
                  xlabels: List[str],
                  ylabels: List[str],
                  title: str = 'Heatmap',
                  cbar_label: str = 'Value',
                  annotate: bool = True):
    """Create annotated heatmap"""
    # Custom colormap for readability
    colors_map = plt.cm.colors.LinearSegmentedColormap.from_list('',
        ['white', COLORS['chart3'], COLORS['chart1'], COLORS['blue_accent']])

    im = ax.imshow(data, cmap=colors_map, aspect='auto')

    # Add annotations if requested
    if annotate:
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                val = data[i, j]
                text_color = 'white' if val > np.mean(data) + np.std(data) else COLORS['black']
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, fontweight='bold')
    ax.set_yticklabels(ylabels, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12, pad=10)

    # Add grid
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which='minor', color=COLORS['gray'], linestyle='-', linewidth=0.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10, fontweight='bold')

    return im

def create_scatter_plot(ax: plt.Axes,
                       x_data: np.ndarray,
                       y_data: np.ndarray,
                       labels: Optional[List[str]] = None,
                       colors: Optional[List[str]] = None,
                       xlabel: str = 'X',
                       ylabel: str = 'Y',
                       title: str = 'Scatter Plot'):
    """Create scatter plot with optional labels"""
    if colors is None:
        colors = [COLORS['chart1']] * len(x_data)

    scatter = ax.scatter(x_data, y_data, s=100, c=colors,
                        alpha=0.7, edgecolors='black', linewidth=1.5)

    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontweight='bold', fontsize=9)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    return scatter

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300):
    """Save figure with optimal settings"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"[SAVED] {filename}")

# ====================================
# SPECIALIZED CHART FUNCTIONS
# ====================================

def create_model_comparison_chart(models: List[str],
                                 metrics: Dict[str, List[float]],
                                 output_path: str):
    """Create comprehensive model comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Bar chart comparison
    ax = axes[0, 0]
    create_comparison_bars(ax, models, metrics,
                          ylabel='Score', title='Model Performance Comparison')

    # 2. Radar chart
    ax = axes[0, 1]
    # ... radar chart implementation ...

    # 3. Scatter plot
    ax = axes[1, 0]
    if 'speed' in metrics and 'accuracy' in metrics:
        create_scatter_plot(ax, metrics['speed'], metrics['accuracy'],
                           labels=models, xlabel='Speed', ylabel='Accuracy',
                           title='Speed vs Accuracy Trade-off')

    # 4. Heatmap
    ax = axes[1, 1]
    # ... heatmap implementation ...

    plt.suptitle('Comprehensive Model Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_path)

def create_training_dashboard(epochs: np.ndarray,
                             train_loss: np.ndarray,
                             val_loss: np.ndarray,
                             output_path: str):
    """Create training monitoring dashboard"""
    fig, axes = create_multi_panel_figure(2, 2, figsize=(12, 8),
                                         title='Training Dashboard')

    # Loss curves
    ax = axes[0, 0]
    create_line_plot(ax, epochs,
                    {'Training': train_loss, 'Validation': val_loss},
                    xlabel='Epoch', ylabel='Loss', title='Training Progress')

    # Add more panels as needed...

    save_figure(fig, output_path)

# ====================================
# UTILITY FUNCTIONS
# ====================================

def add_annotation_box(ax: plt.Axes, text: str,
                      xy: Tuple[float, float],
                      xytext: Tuple[float, float]):
    """Add annotated text box"""
    ax.annotate(text, xy=xy, xytext=xytext,
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor=COLORS['light_gray'],
                        edgecolor=COLORS['black'],
                        linewidth=1),
               arrowprops=dict(arrowstyle='->',
                             connectionstyle='arc3,rad=0.3',
                             color=COLORS['black']),
               fontsize=9, fontweight='bold')

def add_threshold_line(ax: plt.Axes, y_value: float,
                      label: str, color: str = None):
    """Add horizontal threshold line"""
    if color is None:
        color = COLORS['warning']
    ax.axhline(y=y_value, color=color, linestyle='--',
              linewidth=1.5, alpha=0.7, label=label)

def format_axis_labels(ax: plt.Axes, rotation: int = 0,
                      ha: str = 'center'):
    """Format axis labels for readability"""
    ax.tick_params(axis='x', rotation=rotation)
    for label in ax.get_xticklabels():
        label.set_ha(ha)
        label.set_fontweight('bold')

# Initialize style on import
setup_plotting_style()