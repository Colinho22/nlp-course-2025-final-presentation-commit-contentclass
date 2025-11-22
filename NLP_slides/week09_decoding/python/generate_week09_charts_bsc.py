import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('../figures', exist_ok=True)

for i, name in enumerate(['quality_diversity_tradeoff_bsc', 'beam_search_visual_bsc',
                          'beam_example_tree_bsc', 'temperature_effects_bsc',
                          'temperature_calculation_bsc', 'topk_filtering_bsc',
                          'topk_example_bsc', 'nucleus_process_bsc',
                          'nucleus_cumulative_bsc', 'decoding_comparison_bsc',
                          'strategy_selection_guide_bsc', 'quality_metrics_bsc',
                          'greedy_vs_sampling_bsc', 'length_normalization_bsc',
                          'production_settings_bsc'], 1):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f'Week 9 Chart {i}: {name}', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)
    ax.axis('off')
    plt.savefig(f'../figures/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{i}/15: {name}")
print("Week 9: 15 charts generated!")
