"""
Convert PDF charts to PNG for HTML display
"""
import os
from pdf2image import convert_from_path
import glob

# Find all sr_*.pdf files
pdf_files = glob.glob('sr_*.pdf')

print(f"Converting {len(pdf_files)} PDF files to PNG...")

for pdf_file in pdf_files:
    # Convert PDF to PNG
    png_name = pdf_file.replace('.pdf', '.png')

    try:
        # Convert with high DPI for good quality
        images = convert_from_path(pdf_file, dpi=150)

        # Save the first page (charts are single page)
        if images:
            images[0].save(png_name, 'PNG')
            print(f"Converted: {pdf_file} -> {png_name}")
    except Exception as e:
        print(f"Error converting {pdf_file}: {e}")

print("\nConversion complete!")
print("Now updating HTML to use PNG files...")

# Update HTML file to use PNG instead of PDF
html_file = 'chart_gallery.html'
with open(html_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace .pdf with .png in img src
content = content.replace('.pdf', '.png')

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Updated {html_file} to use PNG images")