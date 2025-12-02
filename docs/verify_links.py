#!/usr/bin/env python3
"""
Link Verification Script for NLP Course Site
Checks all internal links, images, and external URLs
"""

import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
import urllib.request
import urllib.error
import ssl

# Disable SSL verification for checking external links
ssl._create_default_https_context = ssl._create_unverified_context

def find_html_files(docs_dir):
    """Find all HTML files in the docs directory"""
    html_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.html'):
                html_files.append(Path(root) / file)
    return html_files

def extract_links(html_content, file_path):
    """Extract all links and images from HTML content"""
    links = []
    images = []

    # Extract href links
    href_pattern = r'href=["\']([^"\']+)["\']'
    for match in re.finditer(href_pattern, html_content):
        url = match.group(1)
        if not url.startswith('#') and not url.startswith('javascript:'):
            links.append(url)

    # Extract src images
    src_pattern = r'src=["\']([^"\']+)["\']'
    for match in re.finditer(src_pattern, html_content):
        url = match.group(1)
        images.append(url)

    return links, images

def resolve_path(base_file, relative_url, docs_dir):
    """Resolve a relative URL to an absolute file path"""
    if relative_url.startswith('http://') or relative_url.startswith('https://'):
        return None  # External URL

    if relative_url.startswith('/Natural-Language-Processing/'):
        # Absolute path from site root
        relative_url = relative_url.replace('/Natural-Language-Processing/', '')
        return docs_dir / relative_url

    if relative_url.startswith('/'):
        return docs_dir / relative_url[1:]

    # Relative path
    base_dir = base_file.parent
    return (base_dir / relative_url).resolve()

def check_external_url(url, timeout=5):
    """Check if an external URL is accessible"""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        urllib.request.urlopen(req, timeout=timeout)
        return True, None
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)

def main():
    """Main verification function"""
    docs_dir = Path(__file__).parent
    html_files = find_html_files(docs_dir)

    print("=" * 60)
    print("NLP Course Site Link Verification")
    print("=" * 60)
    print(f"\nFound {len(html_files)} HTML files to check")

    all_errors = []
    all_warnings = []
    checked_external = set()

    for html_file in html_files:
        relative_path = html_file.relative_to(docs_dir)
        print(f"\nChecking: {relative_path}")

        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        links, images = extract_links(content, html_file)
        file_errors = []
        file_warnings = []

        # Check links
        for link in links:
            if link.startswith('http://') or link.startswith('https://'):
                # External link - check if not already checked
                if link not in checked_external:
                    checked_external.add(link)
                    success, error = check_external_url(link)
                    if not success:
                        file_warnings.append(f"  External link may be broken: {link} ({error})")
            else:
                # Internal link
                resolved = resolve_path(html_file, link, docs_dir)
                if resolved:
                    # Remove query strings and anchors
                    clean_path = str(resolved).split('#')[0].split('?')[0]
                    if not Path(clean_path).exists():
                        file_errors.append(f"  Broken link: {link}")

        # Check images
        for img in images:
            if img.startswith('http://') or img.startswith('https://'):
                # External image
                if img not in checked_external:
                    checked_external.add(img)
                    success, error = check_external_url(img)
                    if not success:
                        file_warnings.append(f"  External image may be broken: {img} ({error})")
            else:
                # Internal image
                resolved = resolve_path(html_file, img, docs_dir)
                if resolved and not resolved.exists():
                    file_errors.append(f"  Missing image: {img}")

        # Report file results
        if file_errors:
            for err in file_errors:
                print(f"  [ERROR] {err}")
            all_errors.extend([(relative_path, e) for e in file_errors])
        if file_warnings:
            for warn in file_warnings:
                print(f"  [WARN] {warn}")
            all_warnings.extend([(relative_path, w) for w in file_warnings])

        if not file_errors and not file_warnings:
            print("  [OK] All links valid")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files checked: {len(html_files)}")
    print(f"Errors: {len(all_errors)}")
    print(f"Warnings: {len(all_warnings)}")

    if all_errors:
        print("\n--- ERRORS (must fix) ---")
        for file_path, error in all_errors:
            print(f"{file_path}: {error}")

    if all_warnings:
        print("\n--- WARNINGS (review) ---")
        for file_path, warning in all_warnings:
            print(f"{file_path}: {warning}")

    if not all_errors and not all_warnings:
        print("\nAll links are valid!")

    return len(all_errors)

if __name__ == '__main__':
    sys.exit(main())
