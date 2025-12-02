"""
Archive old figure folders and prepare clean figures/ directory.

Old folders to archive:
- figures_with_logo_200px/
- figures_with_logo/
- figures_final/
- Any other figures_* folders

Keep:
- figures/ (will be created if needed for clean charts)
"""

from pathlib import Path
import shutil
from datetime import datetime

def archive_old_figures():
    base_path = Path(__file__).parent.parent  # week09_decoding/

    # Create archive folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_path = base_path / f"figures_archive_{timestamp}"
    archive_path.mkdir(exist_ok=True)

    print("=" * 80)
    print("ARCHIVING OLD FIGURE FOLDERS")
    print("=" * 80)

    # Find all figures_* folders (except plain "figures")
    archived_count = 0
    total_size = 0

    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue

        # Archive all figures_* folders except plain "figures" and existing archives
        if folder.name.startswith('figures') and folder.name != 'figures' and not folder.name.startswith('figures_archive'):
            folder_size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
            total_size += folder_size
            file_count = len(list(folder.rglob('*.pdf')))

            print(f"\n[ARCHIVE] {folder.name}/")
            print(f"          PDFs: {file_count}, Size: {folder_size:,} bytes ({folder_size / 1024 / 1024:.1f} MB)")

            # Move to archive
            dest = archive_path / folder.name
            shutil.move(str(folder), str(dest))
            archived_count += 1
            print(f"          [OK] Moved to {archive_path.name}/")

    # Ensure clean figures/ folder exists
    figures_path = base_path / 'figures'
    if not figures_path.exists():
        figures_path.mkdir()
        print(f"\n[CREATE] figures/ (clean directory for new charts)")
    else:
        pdf_count = len(list(figures_path.glob('*.pdf')))
        print(f"\n[EXISTS] figures/ ({pdf_count} PDFs already present)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Archived {archived_count} figure folder(s)")
    print(f"[OK] Total size archived: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    print(f"[OK] Archive location: {archive_path.name}/")
    print(f"[OK] Clean figures/ directory ready for regeneration")
    print("=" * 80)

if __name__ == '__main__':
    archive_old_figures()
