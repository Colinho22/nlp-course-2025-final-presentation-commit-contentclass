#!/usr/bin/env python3
"""
Verification script for Moodle GitHub Pages site
Checks that all expected files exist and are valid
"""
import json
from pathlib import Path

def main():
    print("="*60)
    print("MOODLE SITE VERIFICATION")
    print("="*60)
    print()

    base_dir = Path(__file__).parent.parent.parent / 'docs' / 'moodle'

    # Check HTML files
    html_files = [
        'index.html',
        'schedule.html',
        'pdfs.html',
        'notebooks.html',
        'assignments.html'
    ]

    print("Checking HTML pages...")
    missing_html = []
    for html in html_files:
        path = base_dir / html
        if path.exists():
            size = path.stat().st_size
            print(f"  [OK] {html:20s} ({size:>6,} bytes)")
        else:
            print(f"  [MISSING] {html}")
            missing_html.append(html)
    print()

    # Check manifest
    manifest_path = base_dir / 'assets' / 'manifest.json'
    if not manifest_path.exists():
        print("[ERROR] manifest.json not found!")
        return 1

    print("Loading manifest.json...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"  Generated: {manifest['generated_at']}")
    print(f"  PDFs: {len(manifest['pdfs'])}")
    print(f"  Notebooks: {len(manifest['notebooks'])}")
    print()

    # Verify all PDFs exist
    print("Verifying PDF files...")
    missing_pdfs = []
    for pdf in manifest['pdfs']:
        path = base_dir / 'assets' / pdf['output_path']
        if not path.exists():
            print(f"  [MISSING] {pdf['output_filename']}")
            missing_pdfs.append(pdf['output_filename'])

    if not missing_pdfs:
        print(f"  [OK] All {len(manifest['pdfs'])} PDFs present")
    print()

    # Verify all notebooks exist
    print("Verifying notebook files...")
    missing_notebooks = []
    for nb in manifest['notebooks']:
        path = base_dir / 'assets' / nb['output_path']
        if not path.exists():
            print(f"  [MISSING] {nb['output_filename']}")
            missing_notebooks.append(nb['output_filename'])

    if not missing_notebooks:
        print(f"  [OK] All {len(manifest['notebooks'])} notebooks present")
    print()

    # Summary
    print("="*60)
    if missing_html or missing_pdfs or missing_notebooks:
        print("VERIFICATION FAILED")
        print("="*60)
        if missing_html:
            print(f"Missing HTML pages: {len(missing_html)}")
        if missing_pdfs:
            print(f"Missing PDFs: {len(missing_pdfs)}")
        if missing_notebooks:
            print(f"Missing notebooks: {len(missing_notebooks)}")
        return 1
    else:
        print("VERIFICATION PASSED")
        print("="*60)
        print()
        print(f"Site ready at: {base_dir}")
        print(f"Open in browser: file:///{base_dir.as_posix()}/index.html")
        return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
