"""
Add QuantLet Attribution to All Charts
=======================================
Adds small text annotation to bottom-right of each PDF:
"Code: quantlet.com/NLPDecoding_ChartName"

This creates a visual link (not clickable in PDF, but provides URL).

For clickable links, we'd need to:
1. Modify the original Python generation scripts to add the annotation
2. Re-generate all PDFs

This script demonstrates the annotation approach.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import img2pdf
import io

def add_attribution_to_pdf(pdf_path, chart_name):
    """
    Add QuantLet attribution to a PDF chart.

    Note: This approach converts PDF -> Image -> PDF with annotation -> PDF
    For best quality, attribution should be added during chart generation.
    """

    # Convert PDF to image
    images = convert_from_path(pdf_path, dpi=300)

    if not images:
        return False

    # Work with first page
    img = images[0]
    draw = ImageDraw.Draw(img)

    # Create QuantLet attribution text
    quantlet_url = f"quantlet.com/NLPDecoding_{chart_name}"
    attribution_text = f"Code: {quantlet_url}"

    # Font (try to use a nice sans-serif)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Get image size and text size
    width, height = img.size
    bbox = draw.textbbox((0, 0), attribution_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position in bottom-right corner (with padding)
    padding = 30
    x = width - text_width - padding
    y = height - text_height - padding

    # Add semi-transparent background box
    box_padding = 10
    box = [
        x - box_padding,
        y - box_padding,
        x + text_width + box_padding,
        y + text_height + box_padding
    ]
    draw.rectangle(box, fill=(255, 255, 255, 200), outline=(180, 180, 180))

    # Add text in gray
    draw.text((x, y), attribution_text, fill=(100, 100, 100), font=font)

    # Convert back to PDF
    output_path = pdf_path.parent / f"{pdf_path.stem}_with_attribution.pdf"

    # Save image as PDF
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    with open(output_path, 'wb') as f:
        f.write(img2pdf.convert(img_byte_arr))

    return output_path


print("=" * 70)
print("Add QuantLet Attribution to Charts")
print("=" * 70)
print("\nNOTE: This approach has limitations:")
print("  - Converts PDF -> Image -> PDF (quality loss)")
print("  - Links are NOT clickable")
print("  - Large file sizes")
print("\nBETTER APPROACH:")
print("  - Modify the Python chart generation scripts directly")
print("  - Add attribution during chart creation")
print("  - Use matplotlib text annotation with URL metadata")
print("=" * 70)
print("\nExample code to add during chart generation:")
print("""
# Add QuantLet attribution to chart
ax.text(0.98, 0.02, 'Code: quantlet.com/NLPDecoding_ChartName',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=8, color='gray',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='lightgray', alpha=0.8))
""")
print("=" * 70)

# Requirements check
try:
    from PIL import Image
    import pdf2image
    import img2pdf
    print("\nDependencies installed: PIL, pdf2image, img2pdf")
except ImportError as e:
    print(f"\nMISSING DEPENDENCY: {e}")
    print("Install with: pip install Pillow pdf2image img2pdf")
