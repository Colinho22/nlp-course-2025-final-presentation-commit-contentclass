"""
Add Clickable QuantLet Logo to Chart PDFs
==========================================
Uses PyMuPDF (fitz) to:
1. Add logo image at bottom-left
2. Add clickable link annotation over the logo
3. Link points to quantlet.com (or GitHub when uploaded)

The logo becomes clickable in PDF viewers!
"""

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed")
    print("Install with: pip install PyMuPDF")
    exit(1)

from pathlib import Path

# Configuration
LOGO_PATH = Path("quantlet_logo.png")
FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("figures_with_clickable_logo")
LOGO_SIZE = 40      # pixels
PADDING = 10        # pixels from edges
BASE_URL = "https://quantlet.com"  # Will update to GitHub URL later

if not LOGO_PATH.exists():
    print(f"ERROR: {LOGO_PATH} not found")
    exit(1)

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Get all PDFs
pdfs = sorted(FIGURES_DIR.glob("*.pdf"))

print("=" * 70)
print(f"Adding Clickable QuantLet Logo to {len(pdfs)} PDFs")
print("=" * 70)
print(f"Logo: {LOGO_PATH} (official QuantLet from GitHub)")
print(f"Position: Bottom-left corner")
print(f"Size: {LOGO_SIZE}x{LOGO_SIZE} pixels")
print(f"Link: {BASE_URL}")
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

        # Add clickable link annotation over the logo
        # Create link that points to QuantLet
        link_url = f"{BASE_URL}/NLPDecoding_{pdf_path.stem.replace('_bsc', '')}"

        # Add link annotation (invisible rectangle that's clickable)
        link_annot = page.insert_link({
            "kind": fitz.LINK_URI,
            "from": logo_rect,
            "uri": link_url
        })

        # Add tooltip/title to the link (hover text)
        # Note: Not all PDF viewers support this
        try:
            annot = page.annots()[-1] if page.annots() else None
            if annot:
                annot.set_info(title=f"View code at {link_url}")
                annot.update()
        except:
            pass  # Some PDF versions don't support this

        # Save modified PDF
        output_path = OUTPUT_DIR / pdf_path.name
        doc.save(output_path)
        doc.close()

        print(f"  OK {pdf_path.name} -> {link_url}")
        processed += 1

    except Exception as e:
        print(f"  ERROR {pdf_path.name}: {str(e)[:60]}")
        errors += 1
        try:
            doc.close()
        except:
            pass

print("\n" + "=" * 70)
print(f"Processed: {processed} PDFs")
print(f"Errors: {errors}")
print(f"Output: {OUTPUT_DIR.absolute()}")
print("=" * 70)

if processed > 0:
    print("\nSUCCESS: Charts now have CLICKABLE QuantLet logo!")
    print("Click the logo in any PDF viewer to visit QuantLet")
    print(f"\nOpen folder: {OUTPUT_DIR.absolute()}")
    print("\nTo update URLs to GitHub later:")
    print("  1. Update BASE_URL in script")
    print("  2. Re-run this script")
