"""
Remove duplicate generation scripts from chart folders.

These are backup/bundled generators that have been copied to multiple folders.
The canonical individual scripts are kept.
"""

from pathlib import Path
import os

# Duplicates to remove (found in multiple folders)
DUPLICATES_TO_REMOVE = [
    'generate_week09_enhanced_charts.py',      # Found in 8+ folders
    'generate_week09_charts_bsc_enhanced.py',  # Found in 9+ folders
    'fix_week09_final_charts.py',              # Found in 6 folders
    'fix_charts_redesign.py',                  # Found in 1 folder
]

def remove_duplicates():
    base_path = Path(__file__).parent / 'charts_individual'

    print("=" * 80)
    print("REMOVING DUPLICATE SCRIPTS")
    print("=" * 80)

    removed_count = 0
    total_size = 0

    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue

        for dup_name in DUPLICATES_TO_REMOVE:
            dup_path = folder / dup_name

            if dup_path.exists():
                file_size = dup_path.stat().st_size
                total_size += file_size

                print(f"\n[DELETE] {folder.name}/{dup_name}")
                print(f"         Size: {file_size:,} bytes")

                dup_path.unlink()
                removed_count += 1
                print(f"         [OK] Removed")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Removed {removed_count} duplicate scripts")
    print(f"[OK] Reclaimed {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    print("=" * 80)

if __name__ == '__main__':
    remove_duplicates()
