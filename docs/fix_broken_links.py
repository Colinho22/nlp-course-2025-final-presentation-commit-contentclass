"""
Fix broken links in Hugo-generated HTML files.
Adds /Natural-Language-Processing prefix to paths that are missing it.
"""
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent
BASE_PATH = "/Natural-Language-Processing"

# Patterns to fix (missing base path)
PATTERNS = [
    # href=/topics/... -> href=/Natural-Language-Processing/topics/...
    (r'href=/topics/', f'href={BASE_PATH}/topics/'),
    (r'href=/modules/', f'href={BASE_PATH}/modules/'),
    (r'href=/charts/', f'href={BASE_PATH}/charts/'),
    (r'href=/slides/', f'href={BASE_PATH}/slides/'),
    (r'href=/notebooks/', f'href={BASE_PATH}/notebooks/'),
    (r'href=/moodle/', f'href={BASE_PATH}/moodle/'),
    # src=/charts/... -> src=/Natural-Language-Processing/charts/...
    (r'src=/charts/', f'src={BASE_PATH}/charts/'),
    (r'src=/images/', f'src={BASE_PATH}/images/'),
]

def fix_file(filepath: Path) -> int:
    """Fix broken links in a single file. Returns number of fixes."""
    content = filepath.read_text(encoding='utf-8')
    original = content

    fixes = 0
    for pattern, replacement in PATTERNS:
        # Count matches before replacing
        matches = len(re.findall(pattern, content))
        if matches > 0:
            content = re.sub(pattern, replacement, content)
            fixes += matches

    if fixes > 0:
        filepath.write_text(content, encoding='utf-8')
        print(f"  Fixed {fixes} links in {filepath.relative_to(DOCS_DIR)}")

    return fixes

def main():
    """Fix all HTML files in docs directory."""
    total_fixes = 0
    files_fixed = 0

    print("Scanning for broken links...")
    print("-" * 60)

    # Find all HTML files
    for html_file in DOCS_DIR.rglob("*.html"):
        # Skip reports directory
        if "reports" in str(html_file):
            continue

        fixes = fix_file(html_file)
        if fixes > 0:
            total_fixes += fixes
            files_fixed += 1

    print("-" * 60)
    print(f"Total: {total_fixes} links fixed in {files_fixed} files")

if __name__ == "__main__":
    main()
