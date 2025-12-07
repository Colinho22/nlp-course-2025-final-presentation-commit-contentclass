#!/usr/bin/env python3
"""
Comprehensive fix script for NLP Course Site
Addresses all issues from 10-agent audit:
1. Add script tags to load JS files
2. Generate search.json
3. Fix color contrast (#959da5 -> #606060)
4. Add skip-to-content link
5. Add keyboard focus indicators
6. Add lazy loading to gallery images
"""

import os
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup
import html

def get_docs_dir():
    """Get docs directory path"""
    return Path(__file__).parent.parent

def find_all_html_files(docs_dir):
    """Find all HTML files in docs directory"""
    html_files = []
    for root, dirs, files in os.walk(docs_dir):
        # Skip scripts directory
        if 'scripts' in root:
            continue
        for file in files:
            if file.endswith('.html'):
                html_files.append(Path(root) / file)
    return html_files

def generate_search_json(docs_dir, html_files):
    """Generate search.json from all HTML pages"""
    print("\n[1/6] Generating search.json...")

    search_data = []

    for html_file in html_files:
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'html.parser')

            # Get title
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else html_file.stem
            title = title.replace(' | NLP Course', '').strip()

            # Get main content text
            main_content = soup.find('main') or soup.find(class_='main-content')
            if main_content:
                # Remove script tags
                for script in main_content.find_all('script'):
                    script.decompose()
                text_content = main_content.get_text(separator=' ', strip=True)
            else:
                text_content = soup.get_text(separator=' ', strip=True)

            # Clean and truncate content
            text_content = re.sub(r'\s+', ' ', text_content)[:500]

            # Determine categories
            categories = []
            rel_path = html_file.relative_to(docs_dir)
            if 'topics' in str(rel_path):
                categories.append('topics')
            elif 'modules' in str(rel_path):
                categories.append('modules')
            else:
                categories.append('main')

            # Build URL (relative to site root)
            url = str(rel_path).replace('\\', '/')
            if url == 'index.html':
                url = './'

            search_data.append({
                'title': title,
                'url': url,
                'content': text_content,
                'categories': categories
            })

        except Exception as e:
            print(f"  Warning: Could not process {html_file}: {e}")

    # Write search.json
    search_json_path = docs_dir / 'search.json'
    with open(search_json_path, 'w', encoding='utf-8') as f:
        json.dump(search_data, f, indent=2, ensure_ascii=False)

    print(f"  Created search.json with {len(search_data)} entries")
    return search_json_path

def fix_color_contrast(content):
    """Fix color contrast issue: #959da5 -> #606060"""
    # This color is used for .topic-count and other secondary text
    content = content.replace('#959da5', '#606060')
    content = content.replace('#64748b', '#505050')  # Also fix this for better contrast
    return content

def add_focus_indicators(content):
    """Add keyboard focus indicator CSS"""
    focus_css = """
    /* Keyboard focus indicators for accessibility */
    a:focus, button:focus, input:focus, select:focus, textarea:focus {
      outline: 2px solid #3333B2;
      outline-offset: 2px;
    }
    a:focus:not(:focus-visible), button:focus:not(:focus-visible) {
      outline: none;
    }
    a:focus-visible, button:focus-visible, input:focus-visible {
      outline: 2px solid #3333B2;
      outline-offset: 2px;
    }
    .skip-link {
      position: absolute;
      top: -40px;
      left: 0;
      background: #3333B2;
      color: white;
      padding: 8px 16px;
      z-index: 1000;
      text-decoration: none;
      font-weight: 600;
    }
    .skip-link:focus {
      top: 0;
    }
"""
    # Insert before </style>
    if '</style>' in content:
        content = content.replace('</style>', focus_css + '\n  </style>')
    return content

def add_skip_link(content):
    """Add skip-to-content link after <body>"""
    skip_link = '<a href="#main-content" class="skip-link">Skip to main content</a>\n  '

    # Add after <body>
    if '<body>' in content:
        content = content.replace('<body>', '<body>\n  ' + skip_link)

    # Add id to main content
    if '<main class="main-content">' in content:
        content = content.replace('<main class="main-content">', '<main class="main-content" id="main-content">')
    elif '<main>' in content:
        content = content.replace('<main>', '<main id="main-content">')

    return content

def add_lazy_loading(content):
    """Add lazy loading to images in gallery"""
    # Add loading="lazy" to gallery images
    content = re.sub(
        r'(<img\s+)(src="assets/images/[^"]+"\s+alt="[^"]+"\s+class="topic-thumb")',
        r'\1loading="lazy" \2',
        content
    )
    content = re.sub(
        r'(<div class="gallery-item"><img\s+)(src=)',
        r'\1loading="lazy" \2',
        content
    )
    return content

def add_script_tags(content, is_index=False):
    """Add script tags to load JS files"""
    # Determine path prefix based on file location
    if is_index:
        prefix = 'assets/js/'
    else:
        prefix = '../assets/js/'

    script_tags = f'''
  <!-- Search and interactivity -->
  <script src="{prefix}search.js"></script>
  <script src="{prefix}gallery-filter.js"></script>
  <script src="{prefix}toc.js"></script>
'''

    # Insert before </body>
    if '</body>' in content:
        content = content.replace('</body>', script_tags + '</body>')

    return content

def fix_single_file(html_file, docs_dir):
    """Apply all fixes to a single HTML file"""
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Determine if this is index.html
    is_index = html_file.name == 'index.html' and html_file.parent == docs_dir

    # Apply fixes
    content = fix_color_contrast(content)
    content = add_focus_indicators(content)
    content = add_skip_link(content)
    content = add_lazy_loading(content)

    # Only add script tags if not already present
    if 'search.js' not in content:
        content = add_script_tags(content, is_index)

    # Write if changed
    if content != original_content:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def fix_embeddings_extended_tabs(docs_dir):
    """Fix the tab functionality in embeddings-extended.html"""
    print("\n[3/6] Fixing embeddings-extended.html tabs...")

    file_path = docs_dir / 'modules' / 'embeddings-extended.html'
    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if tabs JS already exists
    if 'function switchTab' in content:
        print("  Tab JavaScript already present")
        return

    # Add tab switching JavaScript
    tab_js = '''
  <script>
    // Tab switching functionality
    function switchTab(tabId) {
      // Hide all tab contents
      document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
        tab.style.display = 'none';
      });

      // Deactivate all tab buttons
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
      });

      // Show selected tab content
      const selectedTab = document.getElementById(tabId);
      if (selectedTab) {
        selectedTab.classList.add('active');
        selectedTab.style.display = 'block';
      }

      // Activate clicked button
      const clickedBtn = document.querySelector(`[onclick="switchTab('${tabId}')"]`);
      if (clickedBtn) {
        clickedBtn.classList.add('active');
      }
    }

    // Initialize first tab on page load
    document.addEventListener('DOMContentLoaded', function() {
      const firstTab = document.querySelector('.tab-content');
      const firstBtn = document.querySelector('.tab-btn');
      if (firstTab) {
        firstTab.style.display = 'block';
        firstTab.classList.add('active');
      }
      if (firstBtn) {
        firstBtn.classList.add('active');
      }
    });
  </script>
'''

    # Insert before </body>
    if '</body>' in content:
        content = content.replace('</body>', tab_js + '\n</body>')

    # Also add CSS for tab styling if not present
    if '.tab-content.active' not in content:
        tab_css = '''
    /* Tab styling */
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .tab-btn { cursor: pointer; padding: 10px 20px; border: none; background: #f0f0f0; }
    .tab-btn.active { background: #3333B2; color: white; }
    .tab-btn:hover { background: #e0e0e0; }
    .tab-btn.active:hover { background: #2929a0; }
'''
        if '</style>' in content:
            content = content.replace('</style>', tab_css + '\n  </style>')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("  Added tab JavaScript to embeddings-extended.html")

def main():
    """Main function to run all fixes"""
    print("=" * 60)
    print("NLP Course Site - Comprehensive Fix Script")
    print("=" * 60)

    docs_dir = get_docs_dir()
    print(f"Docs directory: {docs_dir}")

    # Find all HTML files
    html_files = find_all_html_files(docs_dir)
    print(f"Found {len(html_files)} HTML files")

    # 1. Generate search.json
    generate_search_json(docs_dir, html_files)

    # 2. Fix all HTML files
    print("\n[2/6] Fixing HTML files (color contrast, focus, skip-link, lazy loading)...")
    fixed_count = 0
    for html_file in html_files:
        if fix_single_file(html_file, docs_dir):
            fixed_count += 1
            print(f"  Fixed: {html_file.relative_to(docs_dir)}")
    print(f"  Modified {fixed_count} files")

    # 3. Fix embeddings-extended tabs
    fix_embeddings_extended_tabs(docs_dir)

    # 4-6 already handled in fix_single_file
    print("\n[4/6] Color contrast fixed (#959da5 -> #606060)")
    print("[5/6] Focus indicators added")
    print("[6/6] Skip-to-content links added")

    print("\n" + "=" * 60)
    print("ALL FIXES APPLIED SUCCESSFULLY")
    print("=" * 60)

    return 0

if __name__ == '__main__':
    exit(main())
