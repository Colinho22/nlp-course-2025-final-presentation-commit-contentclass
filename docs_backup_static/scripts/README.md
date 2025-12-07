# Moodle GitHub Pages Site Generator

Complete build system for converting Moodle course export to a static GitHub Pages site.

## Quick Start

**One-command build:**
```bash
python build_moodle_site.py
```

This runs all 4 steps in order:
1. Create directory structure
2. Parse Moodle export
3. Copy assets (PDFs and notebooks)
4. Generate HTML pages

## Build Components

### 1. `setup_moodle_dirs.py`
Creates the required directory structure:
```
docs/moodle/
├── assets/
│   ├── pdfs/
│   └── notebooks/
```

### 2. `moodle_parser.py`
Parses the Moodle course export and extracts:
- 29 course sections
- 28 PDF files
- 5 Jupyter notebooks
- 3 assignments
- Section hierarchy and metadata

Output: `moodle_data.json` (complete course structure)

### 3. `copy_moodle_assets.py`
Copies all assets to the output directory:
- Renames files to URL-friendly names (lowercase, no spaces)
- Creates manifest.json with metadata
- Tracks file sizes and locations

### 4. `generate_moodle_site.py`
Generates 5 HTML pages with embedded CSS:
- `index.html` - Course overview
- `schedule.html` - Weekly schedule
- `pdfs.html` - Categorized PDF downloads
- `notebooks.html` - Jupyter notebook listing
- `assignments.html` - Assessment details

## Site Statistics

**Generated Output:**
- HTML Pages: 5 (69 KB total)
- PDF Files: 28 (16.07 MB)
- Notebooks: 5 (6.28 MB)
- Total Site Size: 18 MB

**Top 5 Largest PDFs:**
1. `decoding_strategies.pdf` (2.57 MB)
2. `2025_nlp.pdf` (2.13 MB)
3. `embeddings_intro.pdf` (1.08 MB)
4. `tsne_tutorial.pdf` (989 KB)
5. `week04_seq2seq.pdf` (951 KB)

## CSS Design

Uses embedded CSS with FHGR branding:
- Primary Color: #2d7dcc (FHGR blue)
- Accent Color: #ff8c00 (FHGR orange)
- Typography: Segoe UI / system fonts
- Responsive sidebar layout (260px left sidebar)

## File Locations

**Input:**
```
moodle/TA_(dv-)_HS25_1764742483/
└── Course_Text_Analytics_dv-_HS25_.1182614/
    ├── index.html (main course index)
    └── [sections with PDFs and notebooks]
```

**Output:**
```
docs/moodle/
├── index.html
├── schedule.html
├── pdfs.html
├── notebooks.html
├── assignments.html
└── assets/
    ├── manifest.json
    ├── pdfs/ (28 files)
    └── notebooks/ (5 files)
```

**Build Scripts:**
```
docs_backup_static/scripts/
├── build_moodle_site.py (MASTER BUILD)
├── setup_moodle_dirs.py
├── moodle_parser.py
├── copy_moodle_assets.py
├── generate_moodle_site.py
├── moodle_data.json (generated)
└── README.md (this file)
```

## Manifest Structure

The `manifest.json` file contains complete metadata for all assets:

```json
{
  "generated_at": "2025-12-07T20:18:02.127746",
  "pdfs": [
    {
      "original_filename": "Foundations_and_Statistical_Language_Modelling.pdf",
      "output_filename": "foundations_and_statistical_language_modelling.pdf",
      "output_path": "pdfs/foundations_and_statistical_language_modelling.pdf",
      "size_bytes": 753152,
      "title": "Slides: Foundations and Statistical Language Modelling",
      "section": "Foundations_and_Statistical_Language_Models"
    }
  ],
  "notebooks": [...]
}
```

## Maintenance

**To regenerate the entire site:**
```bash
python build_moodle_site.py
```

**To update only the HTML (after CSS changes):**
```bash
python generate_moodle_site.py
```

**To verify the build:**
```bash
python -c "
import json
from pathlib import Path
manifest = Path('../../docs/moodle/assets/manifest.json')
data = json.loads(manifest.read_text())
print(f'PDFs: {len(data[\"pdfs\"])}')
print(f'Notebooks: {len(data[\"notebooks\"])}')
print(f'Generated: {data[\"generated_at\"]}')
"
```

## Deployment

The site is ready for GitHub Pages deployment:

1. Commit the `docs/moodle/` directory
2. Enable GitHub Pages from the `docs/` folder
3. Site will be live at: `https://[username].github.io/[repo]/moodle/`

## Implementation Notes

**Character Encoding:**
- All Python scripts use UTF-8 encoding
- German umlauts preserved in section names
- HTML uses `<meta charset="UTF-8">`

**File Naming:**
- Original filenames preserved in manifest
- Output filenames: lowercase, spaces→underscores, URL-safe
- Download links use sanitized names

**Dependencies:**
- Python 3.8+
- Standard library only (no external packages)
- BeautifulSoup4 for HTML parsing

## Future Enhancements

Potential additions:
- Search functionality (client-side JavaScript)
- PDF preview thumbnails
- Progress tracking (completed assignments)
- Mobile-optimized layout
- Dark mode toggle

## Support

For issues or questions, check:
1. `moodle_data.json` - Verify parsing succeeded
2. `manifest.json` - Verify all assets copied
3. Browser console - Check for 404 errors on assets
4. Build output - Look for file copy errors

Last updated: 2025-12-07
