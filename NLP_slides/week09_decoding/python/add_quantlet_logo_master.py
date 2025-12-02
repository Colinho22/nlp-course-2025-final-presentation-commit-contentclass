"""
Master script to add QuantLet logo to all generated chart PDFs.

Reads clean PDFs from ../figures/ and creates logo versions in ../figures_with_logo/.
"""

from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

def add_quantlet_logo(input_pdf_path, output_pdf_path, logo_height_px=200):
    """
    Add QuantLet logo strip to bottom of PDF.

    Args:
        input_pdf_path: Path to clean PDF
        output_pdf_path: Path for output PDF with logo
        logo_height_px: Height of logo strip in pixels (default: 200px)
    """
    try:
        # Read input PDF
        reader = PdfReader(str(input_pdf_path))
        writer = PdfWriter()

        # Get first page dimensions
        first_page = reader.pages[0]
        width = float(first_page.mediabox.width)
        height = float(first_page.mediabox.height)

        # Create logo overlay (simplified - just text for now)
        # In production, you would add actual QuantLet logo image
        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=(width, height))

        # Add subtle QuantLet branding at bottom
        c.setFillColorRGB(0.9, 0.9, 0.95)  # Light lavender background
        c.rect(0, 0, width, logo_height_px / 4, fill=True, stroke=False)

        c.setFillColorRGB(0.2, 0.2, 0.5)  # Purple text
        c.setFont("Helvetica", 8)
        c.drawString(10, 5, "QuantLet.com")

        c.save()
        packet.seek(0)

        # Merge with original
        logo_pdf = PdfReader(packet)

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            if page_num == 0:  # Only add to first page
                page.merge_page(logo_pdf.pages[0])
            writer.add_page(page)

        # Write output
        with open(output_pdf_path, 'wb') as output_file:
            writer.write(output_file)

        return True

    except Exception as e:
        print(f"Error adding logo to {input_pdf_path.name}: {e}")
        return False


def add_logos_to_all():
    """Add QuantLet logo to all PDFs in ../figures/."""

    base_path = Path(__file__).parent.parent  # week09_decoding/
    input_dir = base_path / 'figures'
    output_dir = base_path / 'figures_with_logo'

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("ADDING QUANTLET LOGOS TO ALL CHARTS")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    # Find all PDFs
    pdf_files = list(input_dir.glob('*.pdf'))

    if not pdf_files:
        print(f"[WARN] No PDF files found in {input_dir}")
        return

    print(f"\nFound {len(pdf_files)} PDF files")

    success_count = 0
    failed = []

    for i, pdf_path in enumerate(sorted(pdf_files), 1):
        output_path = output_dir / pdf_path.name

        print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")

        if add_quantlet_logo(pdf_path, output_path):
            print(f"          [OK] Logo added -> {output_path.name}")
            success_count += 1
        else:
            print(f"          [ERROR] Failed")
            failed.append(pdf_path.name)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Successfully processed: {success_count}/{len(pdf_files)} PDFs")
    if failed:
        print(f"[WARN] Failed: {len(failed)} PDFs")
        for name in failed:
            print(f"       - {name}")
    print(f"\nOutput location: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    add_logos_to_all()
