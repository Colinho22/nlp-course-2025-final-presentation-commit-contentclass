#!/usr/bin/env python3
"""
Verify that all paths referenced in generate_site.py actually exist on GitHub
"""

import urllib.request
import urllib.error
import json
import sys

GITHUB_API = "https://api.github.com/repos/Digital-AI-Finance/Natural-Language-Processing/contents"

# Paths to verify (from generate_site.py)
PATHS_TO_VERIFY = [
    # Week folders
    'NLP_slides/week01_foundations',
    'NLP_slides/week02_neural_lm',
    'NLP_slides/week03_rnn',
    'NLP_slides/week04_seq2seq',
    'NLP_slides/week05_transformers',
    'NLP_slides/week06_pretrained',
    'NLP_slides/week07_advanced',
    'NLP_slides/week08_tokenization',
    'NLP_slides/week09_decoding',
    'NLP_slides/week10_finetuning',
    'NLP_slides/week11_efficiency',
    'NLP_slides/week12_ethics',
    # Module folders
    'embeddings',
    'NLP_slides/summarization_module',
    'NLP_slides/sentiment_analysis_module',
    'NLP_slides/lstm_primer',
]

def check_github_path(path):
    """Check if a path exists in the GitHub repository"""
    url = f"{GITHUB_API}/{path}"
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/vnd.github.v3+json'
        })
        response = urllib.request.urlopen(req, timeout=10)
        return True, None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False, "Not found"
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("GitHub Path Verification")
    print("=" * 60)
    print(f"\nChecking {len(PATHS_TO_VERIFY)} paths...\n")

    found = []
    missing = []

    for path in PATHS_TO_VERIFY:
        exists, error = check_github_path(path)
        if exists:
            print(f"  [OK] {path}")
            found.append(path)
        else:
            print(f"  [MISSING] {path} ({error})")
            missing.append(path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Found: {len(found)}/{len(PATHS_TO_VERIFY)}")
    print(f"  Missing: {len(missing)}")

    if missing:
        print("\n  Missing paths:")
        for path in missing:
            print(f"    - {path}")
        print("\n*** SOME PATHS ARE MISSING ***")
        return 1
    else:
        print("\n*** ALL PATHS EXIST ***")
        return 0

if __name__ == '__main__':
    sys.exit(main())
