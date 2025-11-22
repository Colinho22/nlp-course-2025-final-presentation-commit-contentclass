"""
Cleanup script for NLP course presentations
Moves older versions to previous/ or deprecated/ folders
Keeps only the most recent canonical version in main presentations/ folder
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')

# Define canonical versions to KEEP in main folder (based on newest timestamps)
CANONICAL_FILES = {
    'week01_foundations': [
        '20250120_0140_week01_optimal_template.pdf',
        '20250120_0140_week01_optimal_template.tex'
    ],
    'week02_neural_lm': [
        '20250929_1545_week02_neural_lm_template.pdf',
        '20250929_1545_week02_neural_lm_template.tex'
    ],
    'week03_rnn': [
        '20250929_1027_week03_rnn_template.pdf',
        '20250929_1027_week03_rnn_template.tex'
    ],
    'week04_seq2seq': [
        '20250929_2300_week04_seq2seq_pedagogical.pdf',
        '20250929_2300_week04_seq2seq_pedagogical.tex'
    ],
    'week05_transformers': [
        '20250929_2400_week05_transformers_pedagogical.pdf',
        '20250929_2400_week05_transformers_pedagogical.tex'
    ],
    'week06_pretrained': [
        '20250930_0100_week06_pretrained_pedagogical.pdf',
        '20250930_0100_week06_pretrained_pedagogical.tex'
    ],
    'week07_advanced': [
        '20250921_1729_week07_optimal_template.pdf',
        '20250921_1729_week07_optimal_template.tex'
    ],
    'week08_tokenization': [
        '20250923_2110_week08_tokenization_optimal.pdf',
        '20250923_2110_week08_tokenization_optimal.tex'
    ],
    'week09_decoding': [
        'week09_decoding.pdf',  # No timestamped version yet
        'week09_decoding.tex'
    ],
    'week10_finetuning': [
        '20250923_2110_week10_finetuning_optimal.pdf',
        '20250923_2110_week10_finetuning_optimal.tex'
    ],
    'week11_efficiency': [
        '20250923_2110_week11_efficiency_optimal.pdf',
        '20250923_2110_week11_efficiency_optimal.tex'
    ],
    'week12_ethics': [
        '20250923_2110_week12_ethics_optimal.pdf',
        '20250923_2110_week12_ethics_optimal.tex'
    ]
}

def move_to_previous(src_file, week_folder):
    """Move file to previous/ subfolder"""
    previous_dir = week_folder / 'previous'
    previous_dir.mkdir(exist_ok=True)

    dest = previous_dir / src_file.name
    if dest.exists():
        print(f"  Already in previous/: {src_file.name}")
        return

    try:
        shutil.move(str(src_file), str(dest))
        print(f"  Moved to previous/: {src_file.name}")
    except Exception as e:
        print(f"  ERROR moving {src_file.name}: {e}")

def cleanup_week(week_name):
    """Clean up a specific week's presentations folder"""
    presentations_dir = Path(f'NLP_slides/{week_name}/presentations')

    if not presentations_dir.exists():
        print(f"Skipping {week_name}: directory not found")
        return

    print(f"\n=== Cleaning {week_name} ===")

    # Get canonical files to keep
    keep_files = set(CANONICAL_FILES.get(week_name, []))

    # Get all PDF and TEX files
    all_files = list(presentations_dir.glob('*.pdf')) + list(presentations_dir.glob('*.tex'))

    moved_count = 0
    for file_path in all_files:
        if file_path.name in keep_files:
            print(f"  KEEP: {file_path.name}")
        else:
            move_to_previous(file_path, presentations_dir)
            moved_count += 1

    print(f"  Summary: {moved_count} files moved, {len(keep_files)} files kept")
    return moved_count

def main():
    """Main cleanup routine"""
    print("NLP Course Presentations Cleanup")
    print("=" * 60)
    print("This script will:")
    print("- Move older presentation versions to previous/ folders")
    print("- Keep only canonical (newest) versions in main folders")
    print("- NEVER delete files, only move them")
    print("=" * 60)

    # Auto-proceed in automated mode
    print("\nProceeding with cleanup...")

    total_moved = 0
    weeks = [
        'week01_foundations', 'week02_neural_lm', 'week03_rnn',
        'week04_seq2seq', 'week05_transformers', 'week06_pretrained',
        'week07_advanced', 'week08_tokenization', 'week09_decoding',
        'week10_finetuning', 'week11_efficiency', 'week12_ethics'
    ]

    for week in weeks:
        moved = cleanup_week(week)
        if moved:
            total_moved += moved

    print("\n" + "=" * 60)
    print(f"CLEANUP COMPLETE: {total_moved} files moved to previous/ folders")
    print("=" * 60)

if __name__ == '__main__':
    main()