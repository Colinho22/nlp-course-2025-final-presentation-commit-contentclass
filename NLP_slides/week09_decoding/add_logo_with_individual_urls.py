"""
Add Clickable Logo with Individual URLs from Script Metadata
=============================================================
Reads QUANTLET_URL from each Python script and uses it for the clickable link.
"""

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed")
    print("Install with: pip install PyMuPDF")
    exit(1)

from pathlib import Path
import re

# Configuration
LOGO_PATH = Path("quantlet_design.png")
FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("figures_with_quantlet_design")  # Final output with design logo
LOGO_SIZE = 40
PADDING = 10

generators_dir = Path("python/standalone_generators")

# Build mapping of PDF -> URL by reading scripts
pdf_to_url = {}

for script in generators_dir.glob("*.py"):
    # Read script to find QUANTLET_URL
    try:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read(500)  # Read first 500 chars

        # Extract QUANTLET_URL
        match = re.search(r'QUANTLET_URL\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            url = match.group(1)
            # Determine PDF name from script name
            pdf_name = script.stem + '.pdf'
            pdf_to_url[pdf_name] = url
    except:
        pass

print("=" * 70)
print(f"Adding Clickable Logo with Individual URLs")
print("=" * 70)
print(f"Found {len(pdf_to_url)} scripts with URL metadata")
print(f"Logo: {LOGO_PATH}")
print(f"Position: Bottom-right")
print(f"Output: {OUTPUT_DIR}/")
print("=" * 70)

OUTPUT_DIR.mkdir(exist_ok=True)

processed = 0
skipped = 0

for pdf_path in sorted(FIGURES_DIR.glob("*.pdf")):
    # Check if we have a URL for this PDF
    url = pdf_to_url.get(pdf_path.name)

    if not url:
        # Default URL if no metadata
        chart_name = pdf_path.stem.replace('_bsc', '').replace('_graphviz', '')
        url = f"https://quantlet.com/NLPDecoding_{chart_name.replace('_', '').title()}"
        status = "default URL"
    else:
        status = "from script"

    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        page = doc[0]

        # Get dimensions
        page_width = page.rect.width
        page_height = page.rect.height

        # Bottom-right position
        logo_x = page_width - LOGO_SIZE - PADDING
        logo_y = page_height - LOGO_SIZE - PADDING

        logo_rect = fitz.Rect(logo_x, logo_y, logo_x + LOGO_SIZE, logo_y + LOGO_SIZE)

        # Add logo
        page.insert_image(logo_rect, filename=str(LOGO_PATH))

        # Add clickable link
        page.insert_link({
            "kind": fitz.LINK_URI,
            "from": logo_rect,
            "uri": url
        })

        # Save
        output_path = OUTPUT_DIR / pdf_path.name
        doc.save(output_path)
        doc.close()

        print(f"  OK {pdf_path.name} -> {url} ({status})")
        processed += 1

    except Exception as e:
        print(f"  SKIP {pdf_path.name}: {str(e)[:40]}")
        skipped += 1

print("\n" + "=" * 70)
print(f"Processed: {processed} PDFs")
print(f"Skipped: {skipped}")
print("=" * 70)
print(f"\nOutput: {OUTPUT_DIR.absolute()}")
print("\nEach logo links to its individual QuantLet URL!")
