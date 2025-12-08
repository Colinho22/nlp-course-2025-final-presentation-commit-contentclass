#!/usr/bin/env python3
"""
Link Checker for GitHub Pages and Web Sites

Validates all links on a website with deep crawling capability.
Checks: <a href>, <img src>, <script src>, <link href>, <iframe src>, etc.
Exports results to CSV and JSON with timestamps.

Usage:
    python link_checker.py https://example.com
    python link_checker.py https://example.com --depth 5 --timeout 15
"""

import argparse
import csv
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


@dataclass
class LinkResult:
    """Result of checking a single link."""
    url: str
    status_code: Optional[int]
    source_page: str
    link_type: str
    is_internal: bool
    error_message: str = ""


# Link extraction configuration: tag -> attribute
LINK_TAGS = {
    "a": "href",
    "img": "src",
    "script": "src",
    "link": "href",
    "source": "src",
    "video": "src",
    "audio": "src",
    "iframe": "src",
    "embed": "src",
    "object": "data",
}

# Schemes to skip
SKIP_SCHEMES = {"mailto", "tel", "javascript", "data", "ftp"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check all links on a website for broken URLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python link_checker.py https://digital-ai-finance.github.io/Natural-Language-Processing
    python link_checker.py https://example.com --depth 5 --timeout 15 --output ./reports
        """
    )
    parser.add_argument("url", help="Base URL to start crawling from")
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=3,
        help="Maximum crawl depth for internal links (default: 3)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=".",
        help="Output directory for reports (default: current directory)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Number of concurrent workers (default: 10)"
    )
    return parser.parse_args()


def get_base_domain(url: str) -> str:
    """Extract the base domain from a URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def is_internal_url(url: str, base_domain: str) -> bool:
    """Check if URL belongs to the same domain."""
    parsed = urlparse(url)
    if not parsed.netloc:
        return True  # Relative URL
    url_domain = f"{parsed.scheme}://{parsed.netloc}"
    return url_domain == base_domain


def should_skip_url(url: str) -> bool:
    """Check if URL should be skipped (mailto, javascript, etc.)."""
    parsed = urlparse(url)
    return parsed.scheme.lower() in SKIP_SCHEMES


def normalize_url(url: str, base_url: str) -> str:
    """Normalize a URL by making it absolute and removing fragments."""
    # Make absolute
    absolute_url = urljoin(base_url, url)
    # Remove fragment
    defragged, _ = urldefrag(absolute_url)
    return defragged


def extract_links(html: str, page_url: str) -> list[tuple[str, str]]:
    """
    Extract all links from HTML content.

    Returns list of (url, link_type) tuples.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for tag, attr in LINK_TAGS.items():
        for element in soup.find_all(tag):
            url = element.get(attr)
            if url and isinstance(url, str) and url.strip():
                url = url.strip()
                if not should_skip_url(url):
                    normalized = normalize_url(url, page_url)
                    links.append((normalized, tag))

    return links


def check_link(url: str, timeout: int) -> tuple[Optional[int], str]:
    """
    Check if a URL is accessible.

    Returns (status_code, error_message).
    Uses HEAD first, falls back to GET.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinkChecker/1.0"
    }

    try:
        # Try HEAD first (faster)
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)

        # Some servers don't support HEAD, try GET
        if response.status_code == 405:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            response.close()

        return response.status_code, ""

    except requests.exceptions.Timeout:
        return None, "Timeout"
    except requests.exceptions.SSLError as e:
        return None, f"SSL Error: {str(e)[:100]}"
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection Error: {str(e)[:100]}"
    except requests.exceptions.RequestException as e:
        return None, f"Request Error: {str(e)[:100]}"
    except Exception as e:
        return None, f"Unknown Error: {str(e)[:100]}"


def fetch_page(url: str, timeout: int) -> Optional[str]:
    """Fetch HTML content of a page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LinkChecker/1.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        # Only process HTML content
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            return response.text
        return None

    except Exception:
        return None


def crawl_and_check(
    start_url: str,
    max_depth: int,
    timeout: int,
    num_workers: int
) -> list[LinkResult]:
    """
    Crawl website and check all links.

    Uses BFS with depth tracking. Internal links are crawled recursively,
    external links are only checked.
    """
    base_domain = get_base_domain(start_url)

    # Queue: (url, depth)
    queue = deque([(start_url, 0)])
    visited_pages = set()
    checked_links = set()
    results = []

    # Links to check: (url, source_page, link_type, is_internal)
    links_to_check = []

    print(f"\nStarting crawl from: {start_url}")
    print(f"Base domain: {base_domain}")
    print(f"Max depth: {max_depth}")
    print("-" * 60)

    # Phase 1: Crawl and collect all links
    while queue:
        current_url, depth = queue.popleft()

        if current_url in visited_pages:
            continue

        visited_pages.add(current_url)
        print(f"[Depth {depth}] Crawling: {current_url}")

        html = fetch_page(current_url, timeout)
        if not html:
            continue

        page_links = extract_links(html, current_url)

        for link_url, link_type in page_links:
            is_internal = is_internal_url(link_url, base_domain)

            # Add to check list if not already checked
            if link_url not in checked_links:
                checked_links.add(link_url)
                links_to_check.append((link_url, current_url, link_type, is_internal))

            # Add internal HTML pages to crawl queue
            if is_internal and depth < max_depth and link_url not in visited_pages:
                # Only crawl HTML pages (skip assets)
                if link_type == "a":
                    parsed = urlparse(link_url)
                    path = parsed.path.lower()
                    # Skip obvious non-HTML resources
                    skip_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                                      ".css", ".js", ".zip", ".tar", ".gz", ".mp4",
                                      ".mp3", ".wav", ".webp", ".ico", ".woff", ".woff2"}
                    if not any(path.endswith(ext) for ext in skip_extensions):
                        queue.append((link_url, depth + 1))

    print(f"\n{'-' * 60}")
    print(f"Crawl complete. Found {len(links_to_check)} unique links to check.")
    print(f"Checking links with {num_workers} workers...")
    print("-" * 60)

    # Phase 2: Check all collected links in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_link = {
            executor.submit(check_link, url, timeout): (url, source, ltype, internal)
            for url, source, ltype, internal in links_to_check
        }

        completed = 0
        total = len(future_to_link)

        for future in as_completed(future_to_link):
            url, source_page, link_type, is_internal = future_to_link[future]
            status_code, error_msg = future.result()

            result = LinkResult(
                url=url,
                status_code=status_code,
                source_page=source_page,
                link_type=link_type,
                is_internal=is_internal,
                error_message=error_msg
            )
            results.append(result)

            completed += 1
            status_str = str(status_code) if status_code else f"ERROR: {error_msg}"
            symbol = "[OK]" if status_code and status_code < 400 else "[FAIL]"

            if completed % 10 == 0 or completed == total:
                print(f"Progress: {completed}/{total} ({100*completed//total}%)")

    return results


def export_results(
    results: list[LinkResult],
    base_url: str,
    output_dir: str
) -> tuple[str, str]:
    """Export results to CSV and JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Calculate statistics
    total = len(results)
    broken = sum(1 for r in results if r.status_code is None or r.status_code >= 400)
    success = total - broken

    # CSV export
    csv_file = output_path / f"link_report_{timestamp}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "status_code", "source_page", "link_type", "is_internal", "error_message"])
        for result in sorted(results, key=lambda r: (r.status_code or 999, r.url)):
            writer.writerow([
                result.url,
                result.status_code or "",
                result.source_page,
                result.link_type,
                result.is_internal,
                result.error_message
            ])

    # JSON export
    json_file = output_path / f"link_report_{timestamp}.json"
    report = {
        "metadata": {
            "base_url": base_url,
            "timestamp": datetime.now().isoformat(),
            "generator": "link_checker.py"
        },
        "statistics": {
            "total_links": total,
            "successful": success,
            "broken": broken,
            "success_rate": round(100 * success / total, 2) if total > 0 else 0
        },
        "broken_links": [asdict(r) for r in results if r.status_code is None or r.status_code >= 400],
        "all_links": [asdict(r) for r in results]
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(csv_file), str(json_file)


def print_summary(results: list[LinkResult], csv_path: str, json_path: str):
    """Print summary of results to console."""
    total = len(results)
    broken = [r for r in results if r.status_code is None or r.status_code >= 400]

    print("\n" + "=" * 60)
    print("LINK CHECK SUMMARY")
    print("=" * 60)
    print(f"Total links checked: {total}")
    print(f"Successful (2xx/3xx): {total - len(broken)}")
    print(f"Broken/Failed: {len(broken)}")
    print(f"Success rate: {100 * (total - len(broken)) / total:.1f}%" if total > 0 else "N/A")

    if broken:
        print("\n" + "-" * 60)
        print("BROKEN LINKS:")
        print("-" * 60)
        for result in sorted(broken, key=lambda r: r.url):
            status = result.status_code or "ERROR"
            error = f" ({result.error_message})" if result.error_message else ""
            print(f"  [{status}]{error}")
            print(f"    URL: {result.url}")
            print(f"    Found on: {result.source_page}")
            print()

    print("-" * 60)
    print("REPORTS SAVED:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    # Validate URL
    parsed = urlparse(args.url)
    if not parsed.scheme or not parsed.netloc:
        print(f"Error: Invalid URL '{args.url}'")
        print("URL must include scheme (http:// or https://)")
        sys.exit(1)

    # Run crawler and checker
    results = crawl_and_check(
        start_url=args.url,
        max_depth=args.depth,
        timeout=args.timeout,
        num_workers=args.workers
    )

    # Export results
    csv_path, json_path = export_results(results, args.url, args.output)

    # Print summary
    print_summary(results, csv_path, json_path)

    # Exit with error code if broken links found
    broken_count = sum(1 for r in results if r.status_code is None or r.status_code >= 400)
    sys.exit(1 if broken_count > 0 else 0)


if __name__ == "__main__":
    main()
