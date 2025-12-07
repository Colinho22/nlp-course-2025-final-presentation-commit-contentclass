#!/usr/bin/env python3
"""
Master build script for Moodle GitHub Pages site
Run this to regenerate everything from scratch
"""
import subprocess
import sys
from pathlib import Path

def main():
    scripts_dir = Path(__file__).parent

    steps = [
        ('setup_moodle_dirs.py', 'Creating directories...'),
        ('moodle_parser.py', 'Parsing Moodle export...'),
        ('copy_moodle_assets.py', 'Copying assets...'),
        ('generate_moodle_site.py', 'Generating HTML pages...')
    ]

    for script, message in steps:
        print(f"\n{'='*50}")
        print(message)
        print('='*50)
        result = subprocess.run([sys.executable, scripts_dir / script])
        if result.returncode != 0:
            print(f"ERROR: {script} failed!")
            return 1

    print("\n" + "="*50)
    print("BUILD COMPLETE!")
    print("="*50)
    print("\nGenerated site at: docs/moodle/")
    print("Pages: index.html, schedule.html, pdfs.html, notebooks.html, assignments.html")
    return 0

if __name__ == '__main__':
    sys.exit(main())
