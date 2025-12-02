"""
Update all chart paths in .tex file to use _bsc standardized versions.
"""

from pathlib import Path

tex_file = Path(r'D:\Joerg\Research\slides\2025_NLP_16\NLP_slides\week09_decoding\presentations\NLP_Decoding_Strategies.tex')

# Charts that need _bsc suffix (excluding graphviz which don't have _bsc)
CHART_UPDATES = [
    ('practical_methods_coverage.pdf', 'practical_methods_coverage_bsc.pdf'),
    ('topk_example.pdf', 'topk_example_bsc.pdf'),
    ('problem1_repetition_output.pdf', 'problem1_repetition_output_bsc.pdf'),
    ('problem2_diversity_output.pdf', 'problem2_diversity_output_bsc.pdf'),
    ('problem3_balance_output.pdf', 'problem3_balance_output_bsc.pdf'),
    ('problem4_search_tree_pruning.pdf', 'problem4_search_tree_pruning_bsc.pdf'),
    ('problem4_path_comparison.pdf', 'problem4_path_comparison_bsc.pdf'),
    ('problem4_probability_evolution.pdf', 'problem4_probability_evolution_bsc.pdf'),
    ('problem4_recovery_problem.pdf', 'problem4_recovery_problem_bsc.pdf'),
    ('problem5_distribution_output.pdf', 'problem5_distribution_output_bsc.pdf'),
    ('problem6_speed_output.pdf', 'problem6_speed_output_bsc.pdf'),
    ('production_settings.pdf', 'production_settings_bsc.pdf'),
]

# Read file
content = tex_file.read_text(encoding='utf-8')

# Replace each
replacements = 0
for old, new in CHART_UPDATES:
    old_path = f'../figures/{old}'
    new_path = f'../figures/{new}'
    count = content.count(old_path)
    if count > 0:
        content = content.replace(old_path, new_path)
        replacements += count
        print(f'Updated: {old} -> {new} ({count} occurrence(s))')

# Write back
tex_file.write_text(content, encoding='utf-8')

print(f'\nTotal replacements: {replacements}')
print('All chart paths updated to _bsc versions')
