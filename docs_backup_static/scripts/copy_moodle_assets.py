"""
Copy Moodle assets (PDFs and notebooks) from export to docs folder.

This script:
1. Reads parsed Moodle data from moodle_data.json
2. Copies PDFs to docs/moodle/assets/pdfs/
3. Copies notebooks to docs/moodle/assets/notebooks/
4. Generates a manifest.json with file metadata and statistics
"""

import json
import shutil
import os
from pathlib import Path
from datetime import datetime
import re

# File size limits
MAX_PDF_SIZE_MB = 50
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024


def sanitize_filename(filename, section=None):
    """
    Sanitize filename for safe filesystem use.

    Args:
        filename: Original filename
        section: Optional section name to prefix (for collision avoidance)

    Returns:
        Sanitized filename (lowercase, underscores, safe characters)
    """
    # Remove extension temporarily
    name, ext = os.path.splitext(filename)

    # Replace spaces with underscores
    name = name.replace(' ', '_')

    # Remove special characters (keep only alphanumeric, underscore, hyphen, dot)
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '', name)

    # Lowercase
    name = name.lower()

    # Add section prefix if provided
    if section:
        section_clean = re.sub(r'[^a-zA-Z0-9_\-]', '', section.lower().replace(' ', '_'))
        # Truncate section to 20 chars to avoid overly long names
        section_clean = section_clean[:20]
        name = f"{section_clean}_{name}"

    # Truncate to 100 chars to avoid filesystem issues
    name = name[:100]

    return f"{name}{ext.lower()}"


def copy_file_safe(src, dst):
    """
    Safely copy file with error handling.

    Args:
        src: Source file path (string)
        dst: Destination file path (string)

    Returns:
        Tuple of (success: bool, error_message: str or None, size_bytes: int)
    """
    try:
        # Ensure source exists
        if not os.path.exists(src):
            return False, f"Source file not found: {src}", 0

        # Check file size
        size_bytes = os.path.getsize(src)

        # Create destination directory
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # Copy file
        shutil.copy2(src, dst)

        return True, None, size_bytes

    except Exception as e:
        return False, str(e), 0


def copy_assets(moodle_data, moodle_export_root, output_dir):
    """
    Copy all PDFs and notebooks from Moodle export to docs folder.

    Args:
        moodle_data: Parsed Moodle data dictionary
        moodle_export_root: Root directory of Moodle export
        output_dir: Target directory for assets (docs/moodle/assets/)

    Returns:
        Dictionary with copied file metadata
    """
    copied_pdfs = []
    copied_notebooks = []
    skipped_files = []

    # Create output directories
    pdf_dir = os.path.join(output_dir, 'pdfs')
    notebook_dir = os.path.join(output_dir, 'notebooks')
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(notebook_dir, exist_ok=True)

    # Track used filenames for collision detection
    used_pdf_names = set()
    used_notebook_names = set()

    # Copy PDFs
    print(f"\nCopying {len(moodle_data['all_pdfs'])} PDFs...")
    for pdf_info in moodle_data['all_pdfs']:
        original_filename = pdf_info['filename']
        original_path = os.path.join(moodle_export_root, pdf_info['path'])
        section = pdf_info['section']
        title = pdf_info['title']

        # Check file size first
        if os.path.exists(original_path):
            size_bytes = os.path.getsize(original_path)
            if size_bytes > MAX_PDF_SIZE_BYTES:
                skipped_files.append({
                    'filename': original_filename,
                    'reason': f'File too large ({size_bytes / 1024 / 1024:.1f} MB > {MAX_PDF_SIZE_MB} MB)',
                    'section': section,
                    'title': title
                })
                print(f"  SKIP (too large): {original_filename} ({size_bytes / 1024 / 1024:.1f} MB)")
                continue

        # Sanitize filename
        sanitized_name = sanitize_filename(original_filename)

        # Check for collisions
        if sanitized_name in used_pdf_names:
            # Add section prefix
            sanitized_name = sanitize_filename(original_filename, section)

        used_pdf_names.add(sanitized_name)

        # Destination path
        dst_path = os.path.join(pdf_dir, sanitized_name)

        # Copy file
        success, error, size_bytes = copy_file_safe(original_path, dst_path)

        if success:
            copied_pdfs.append({
                'original_filename': original_filename,
                'output_filename': sanitized_name,
                'original_path': pdf_info['path'],
                'output_path': f"pdfs/{sanitized_name}",
                'size_bytes': size_bytes,
                'title': title,
                'section': section
            })
            print(f"  OK: {original_filename} -> {sanitized_name}")
        else:
            skipped_files.append({
                'filename': original_filename,
                'reason': error,
                'section': section,
                'title': title
            })
            print(f"  SKIP (error): {original_filename} - {error}")

    # Copy notebooks
    print(f"\nCopying {len(moodle_data['all_notebooks'])} notebooks...")
    for notebook_info in moodle_data['all_notebooks']:
        original_filename = notebook_info['filename']
        original_path = os.path.join(moodle_export_root, notebook_info['path'])
        section = notebook_info['section']
        title = notebook_info['title']

        # Sanitize filename
        sanitized_name = sanitize_filename(original_filename)

        # Check for collisions
        if sanitized_name in used_notebook_names:
            # Add section prefix
            sanitized_name = sanitize_filename(original_filename, section)

        used_notebook_names.add(sanitized_name)

        # Destination path
        dst_path = os.path.join(notebook_dir, sanitized_name)

        # Copy file
        success, error, size_bytes = copy_file_safe(original_path, dst_path)

        if success:
            copied_notebooks.append({
                'original_filename': original_filename,
                'output_filename': sanitized_name,
                'original_path': notebook_info['path'],
                'output_path': f"notebooks/{sanitized_name}",
                'size_bytes': size_bytes,
                'title': title,
                'section': section
            })
            print(f"  OK: {original_filename} -> {sanitized_name}")
        else:
            skipped_files.append({
                'filename': original_filename,
                'reason': error,
                'section': section,
                'title': title
            })
            print(f"  SKIP (error): {original_filename} - {error}")

    return {
        'pdfs': copied_pdfs,
        'notebooks': copied_notebooks,
        'skipped': skipped_files
    }


def generate_manifest(copied_files):
    """
    Generate manifest.json with file metadata and statistics.

    Args:
        copied_files: Dictionary with 'pdfs', 'notebooks', 'skipped' keys

    Returns:
        Manifest dictionary
    """
    total_size_bytes = sum(f['size_bytes'] for f in copied_files['pdfs'])
    total_size_bytes += sum(f['size_bytes'] for f in copied_files['notebooks'])

    manifest = {
        'generated_at': datetime.now().isoformat(),
        'pdfs': copied_files['pdfs'],
        'notebooks': copied_files['notebooks'],
        'skipped': copied_files['skipped'],
        'statistics': {
            'total_pdfs': len(copied_files['pdfs']),
            'total_notebooks': len(copied_files['notebooks']),
            'total_skipped': len(copied_files['skipped']),
            'total_size_mb': round(total_size_bytes / 1024 / 1024, 2)
        }
    }

    return manifest


def main():
    """Main execution function."""
    # Paths
    script_dir = Path(__file__).parent
    moodle_data_path = script_dir / 'moodle_data.json'

    # Moodle export root (parent of Course_Text_Analytics_dv-_HS25_.1182614)
    moodle_export_root = Path(r'D:\Joerg\Research\slides\2025_NLP_16\moodle\TA_(dv-)_HS25_1764742483')

    # Output directory
    output_dir = Path(r'D:\Joerg\Research\slides\2025_NLP_16\docs\moodle\assets')

    # Manifest path
    manifest_path = output_dir / 'manifest.json'

    # Load Moodle data
    print(f"Loading Moodle data from: {moodle_data_path}")
    with open(moodle_data_path, 'r', encoding='utf-8') as f:
        moodle_data = json.load(f)

    # Copy assets
    print(f"\nCopying assets to: {output_dir}")
    copied_files = copy_assets(moodle_data, str(moodle_export_root), str(output_dir))

    # Generate manifest
    print(f"\nGenerating manifest...")
    manifest = generate_manifest(copied_files)

    # Save manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest saved to: {manifest_path}")

    # Print summary
    print("\n" + "="*60)
    print("COPY SUMMARY")
    print("="*60)
    print(f"PDFs copied:      {manifest['statistics']['total_pdfs']}")
    print(f"Notebooks copied: {manifest['statistics']['total_notebooks']}")
    print(f"Files skipped:    {manifest['statistics']['total_skipped']}")
    print(f"Total size:       {manifest['statistics']['total_size_mb']} MB")

    # Show skipped files
    if manifest['statistics']['total_skipped'] > 0:
        print(f"\nSKIPPED FILES ({manifest['statistics']['total_skipped']}):")
        for skip in copied_files['skipped']:
            print(f"  - {skip['filename']}")
            print(f"    Reason: {skip['reason']}")
            print(f"    Section: {skip['section']}")

    # Show sample manifest entries
    print(f"\nSAMPLE MANIFEST ENTRIES:")

    if manifest['pdfs']:
        print(f"\nFirst PDF:")
        sample_pdf = manifest['pdfs'][0]
        print(f"  Original:  {sample_pdf['original_filename']}")
        print(f"  Output:    {sample_pdf['output_filename']}")
        print(f"  Path:      {sample_pdf['output_path']}")
        print(f"  Size:      {sample_pdf['size_bytes'] / 1024:.1f} KB")
        print(f"  Title:     {sample_pdf['title']}")
        print(f"  Section:   {sample_pdf['section']}")

    if manifest['notebooks']:
        print(f"\nFirst Notebook:")
        sample_notebook = manifest['notebooks'][0]
        print(f"  Original:  {sample_notebook['original_filename']}")
        print(f"  Output:    {sample_notebook['output_filename']}")
        print(f"  Path:      {sample_notebook['output_path']}")
        print(f"  Size:      {sample_notebook['size_bytes'] / 1024:.1f} KB")
        print(f"  Title:     {sample_notebook['title']}")
        print(f"  Section:   {sample_notebook['section']}")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
