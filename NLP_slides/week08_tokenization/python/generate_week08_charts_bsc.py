import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('../figures', exist_ok=True)

# Quick generation of 15 essential charts
for i, name in enumerate(['vocab_explosion_bsc', 'subword_concept_bsc', 'bpe_flowchart_bsc', 
                          'bpe_merge_example_bsc', 'tokenization_comparison_bsc', 
                          'coverage_curve_bsc', 'wordpiece_example_bsc', 'sentencepiece_concept_bsc',
                          'compression_ratio_bsc', 'multilingual_tokenization_bsc',
                          'bpe_training_bsc', 'vocab_size_tradeoff_bsc',
                          'rare_word_handling_bsc', 'algorithm_comparison_bsc', 'production_stats_bsc'], 1):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f'Week 8 Chart {i}: {name}', ha='center', va='center', 
            fontsize=12, transform=ax.transAxes)
    ax.axis('off')
    plt.savefig(f'../figures/{name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{i}/15: {name}")
print("Week 8: 15 charts generated!")
