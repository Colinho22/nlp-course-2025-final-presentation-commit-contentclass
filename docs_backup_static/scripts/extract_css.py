#!/usr/bin/env python3
"""
Extract common CSS from HTML files to external main.css
Reduces ~200KB duplication across 18 HTML files
"""

import re
from pathlib import Path

def get_docs_dir():
    return Path(__file__).parent.parent

def extract_common_css(docs_dir):
    """Extract CSS from index.html as the canonical source"""
    index_file = docs_dir / 'index.html'

    with open(index_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract CSS between <style> tags
    match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
    if not match:
        print("Error: Could not find <style> tag in index.html")
        return None

    css_content = match.group(1)

    # Clean up the CSS
    css_content = css_content.strip()

    return css_content

def create_main_css(docs_dir, css_content):
    """Create main.css file"""
    css_dir = docs_dir / 'assets' / 'css'
    css_dir.mkdir(parents=True, exist_ok=True)

    main_css = css_dir / 'main.css'
    with open(main_css, 'w', encoding='utf-8') as f:
        f.write("/* NLP Course Site - Main Stylesheet */\n")
        f.write("/* Extracted from inline styles for better caching */\n\n")
        f.write(css_content)

    print(f"Created {main_css}")
    return main_css

def update_html_files(docs_dir):
    """Update HTML files to use external CSS"""
    html_files = list(docs_dir.glob('*.html')) + \
                 list((docs_dir / 'topics').glob('*.html')) + \
                 list((docs_dir / 'modules').glob('*.html'))

    updated_count = 0

    for html_file in html_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already using external CSS
        if 'href="' in content and 'main.css' in content:
            print(f"  Skip (already has external CSS): {html_file.name}")
            continue

        original_content = content

        # Determine the correct path prefix
        rel_path = html_file.relative_to(docs_dir)
        if str(rel_path) == 'index.html':
            css_path = 'assets/css/main.css'
        else:
            css_path = '../assets/css/main.css'

        # Replace inline <style>...</style> with link to external CSS
        # Keep the style tag structure but minimize the content
        css_link = f'<link rel="stylesheet" href="{css_path}">'

        # Find and replace the <style> block
        pattern = r'<style>.*?</style>'
        if re.search(pattern, content, re.DOTALL):
            # Keep only page-specific styles if any exist
            # For now, just replace with link
            # Add the CSS link before </head>
            if css_link not in content:
                content = content.replace('</head>', f'  {css_link}\n</head>')

            # Remove the inline style (but keep any page-specific styles)
            # We'll leave a minimal style block for page-specific overrides
            content = re.sub(
                pattern,
                '<style>\n    /* Page-specific styles - common styles in assets/css/main.css */\n  </style>',
                content,
                flags=re.DOTALL
            )

            if content != original_content:
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_count += 1
                print(f"  Updated: {html_file.name}")

    return updated_count

def main():
    print("=" * 60)
    print("CSS Extraction Script")
    print("=" * 60)

    docs_dir = get_docs_dir()

    # Extract CSS from index.html
    print("\n[1/2] Extracting CSS from index.html...")
    css_content = extract_common_css(docs_dir)
    if not css_content:
        return 1

    # Create main.css
    create_main_css(docs_dir, css_content)

    # Update HTML files
    print("\n[2/2] Updating HTML files to use external CSS...")
    updated_count = update_html_files(docs_dir)

    print(f"\n  Updated {updated_count} files")
    print("\n" + "=" * 60)
    print("CSS EXTRACTION COMPLETE")
    print("=" * 60)

    return 0

if __name__ == '__main__':
    exit(main())
