"""
Create README.md in each chart folder
======================================
Documents which script/function created each figure.
"""

from pathlib import Path

# Map figure to (script, function_name)
FIGURE_INFO = {
    'beam_search_tree_graphviz.pdf': ('generate_beam_search_graphviz.py', 'create_beam_search_tree', 'Standalone'),
    'full_exploration_tree_graphviz.pdf': ('generate_full_exploration_graphviz.py', 'create_full_exploration_tree', 'Standalone'),
    'greedy_suboptimal_comparison_bsc.pdf': ('generate_greedy_suboptimal_comparison.py', 'generate_greedy_suboptimal_comparison', 'Standalone'),
    'vocabulary_probability_bsc.pdf': ('fix_charts_redesign.py', 'create_vocabulary_probability_chart', 'Function in multi-chart script'),
    'prediction_to_text_pipeline_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_prediction_pipeline', 'Function in multi-chart script'),
    'temperature_effects_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_temperature_effects', 'Function in multi-chart script'),
    'temperature_calculation_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_temperature_calculation', 'Function in multi-chart script'),
    'topk_filtering_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_topk_filtering', 'Function in multi-chart script'),
    'topk_example_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_topk_example', 'Function in multi-chart script'),
    'nucleus_process_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_nucleus_process', 'Function in multi-chart script'),
    'nucleus_cumulative_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_nucleus_cumulative', 'Function in multi-chart script'),
    'degeneration_problem_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_degeneration_problem', 'Function in multi-chart script'),
    'contrastive_mechanism_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_contrastive_mechanism', 'Function in multi-chart script'),
    'contrastive_vs_nucleus_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_contrastive_vs_nucleus', 'Function in multi-chart script'),
    'quality_diversity_pareto_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_quality_diversity_pareto', 'Function in multi-chart script'),
    'task_method_decision_tree_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_task_method_decision_tree', 'Function in multi-chart script'),
    'task_recommendations_table_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_task_recommendations_table', 'Function in multi-chart script'),
    'production_settings_bsc.pdf': ('generate_week09_charts_bsc_enhanced.py', 'generate_production_settings', 'Function in multi-chart script'),
    'quality_diversity_scatter_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_quality_diversity_scatter', 'Function in multi-chart script'),
    'extreme_greedy_single_path_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_case_1_greedy', 'Function in multi-chart script'),
    'extreme_full_beam_explosion_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_case_2_full_beam', 'Function in multi-chart script'),
    'extreme_computational_cost_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_case_3_computational_cost', 'Function in multi-chart script'),
    'extreme_coverage_comparison_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_extreme_coverage_comparison', 'Function in multi-chart script'),
    'practical_methods_coverage_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_practical_methods_coverage', 'Function in multi-chart script'),
    'problem4_search_tree_pruning_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts', 'Part of 4-chart function'),
    'problem4_path_comparison_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts', 'Part of 4-chart function'),
    'problem4_probability_evolution_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts', 'Part of 4-chart function'),
    'problem4_recovery_problem_bsc.pdf': ('generate_week09_enhanced_charts.py', 'generate_problem4_graphviz_charts', 'Part of 4-chart function'),
    'problem1_repetition_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem1_repetition', 'Function in multi-chart script'),
    'problem2_diversity_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem2_diversity', 'Function in multi-chart script'),
    'problem3_balance_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem3_balance', 'Function in multi-chart script'),
    'problem5_distribution_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem5_distribution', 'Function in multi-chart script'),
    'problem6_speed_output_bsc.pdf': ('fix_week09_final_charts.py', 'generate_problem6_speed', 'Function in multi-chart script'),
}

charts_dir = Path("python/charts_individual")

created = 0
for fig, (script, func, note) in FIGURE_INFO.items():
    folder_name = fig.replace('.pdf', '')
    folder = charts_dir / folder_name

    if not folder.exists():
        continue

    readme = folder / "README.md"
    content = f"""# {folder_name.replace('_', ' ').title()}

**Figure**: `{fig}`
**Used in**: NLP_Decoding_Strategies.tex (Week 9 presentation)

## Source

**Script**: `python/{script}`
**Function**: `{func}()`
**Type**: {note}

## How to Regenerate

```bash
cd NLP_slides/week09_decoding/python/charts_individual/{folder_name}
python {script if note == 'Standalone' else 'generate_' + fig}
```

## Notes

{f"This is a standalone script - run it directly." if note == 'Standalone' else f"This figure is generated by the `{func}()` function in `{script}`."}
{f"The original script generates multiple charts - see python/{script}" if 'multi-chart' in note.lower() else ""}
"""

    with open(readme, 'w', encoding='utf-8') as f:
        f.write(content)

    created += 1

print(f"Created {created} README.md files in individual folders")
