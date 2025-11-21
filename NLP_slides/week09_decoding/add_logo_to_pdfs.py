"""
Add QuantLet Logo to Existing Chart PDFs
=========================================
Uses PyMuPDF (fitz) to overlay logo on bottom-left of each PDF.

Processes all PDFs in figures/ directory.
"""

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed")
    print("Install with: pip install PyMuPDF")
    exit(1)

from pathlib import Path
from PIL import Image

# Configuration
LOGO_PATH = Path("quantlet_logo.png")
FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("figures_with_logo")
LOGO_SIZE = 40  # pixels
PADDING = 10    # pixels from bottom-left

if not LOGO_PATH.exists():
    print(f"ERROR: {LOGO_PATH} not found")
    print("Please ensure quantlet_logo.png is in the current directory")
    exit(1)

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Get all PDFs
pdfs = sorted(FIGURES_DIR.glob("*.pdf"))

print("=" * 70)
print(f"Adding QuantLet Logo to {len(pdfs)} Chart PDFs")
print("=" * 70)
print(f"Logo: {LOGO_PATH}")
print(f"Position: Bottom-left corner")
print(f"Size: {LOGO_SIZE}x{LOGO_SIZE} pixels")
print(f"Output: {OUTPUT_DIR}/")
print("=" * 70)

processed = 0
errors = 0

for pdf_path in pdfs:
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        page = doc[0]  # First page

        # Get page dimensions
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        # Calculate logo position (bottom-left)
        logo_x = PADDING
        logo_y = page_height - LOGO_SIZE - PADDING

        # Create rectangle for logo
        logo_rect = fitz.Rect(logo_x, logo_y, logo_x + LOGO_SIZE, logo_y + LOGO_SIZE)

        # Insert logo image
        page.insert_image(logo_rect, filename=str(LOGO_PATH))

        # Save modified PDF
        output_path = OUTPUT_DIR / pdf_path.name
        doc.save(output_path)
        doc.close()

        print(f"  OK {pdf_path.name}")
        processed += 1

    except Exception as e:
        print(f"  ERROR {pdf_path.name}: {str(e)[:50]}")
        errors += 1

print("\n" + "=" * 70)
print(f"Processed: {processed} PDFs")
print(f"Errors: {errors}")
print(f"Output: {OUTPUT_DIR.absolute()}")
print("=" * 70)

if processed > 0:
    print("\nSUCCESS: Charts now have QuantLet logo in bottom-left corner")
    print(f"Review: {OUTPUT_DIR.absolute()}")
